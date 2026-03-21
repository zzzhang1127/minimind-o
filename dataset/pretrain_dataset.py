import io
import os
import wave
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

__package__ = "dataset"

# Make relative imports from repo root work when running from trainer/
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_WHISPER_BASE_DIR: Optional[str] = None
_WHISPER_FEATURE_EXTRACTOR = None
PROMPTS_ZH = [
    "<speech>\n请转录这段语音。",
    "<speech>\n识别以下音频内容：",
    "<speech>\n这段话说了什么？",
    "<speech>\n语音转文字：",
    "<speech>\n请将语音转为文字：",
]


def _get_whisper_base_dir() -> str:
    global _WHISPER_BASE_DIR
    if _WHISPER_BASE_DIR is None:
        # minimind-o/dataset -> minimind-o/model/speech_model/whisper-base
        _WHISPER_BASE_DIR = str(Path(__file__).resolve().parent.parent / "model" / "speech_model" / "whisper-base")
    return _WHISPER_BASE_DIR


def _get_whisper_feature_extractor():
    global _WHISPER_FEATURE_EXTRACTOR
    if _WHISPER_FEATURE_EXTRACTOR is None:
        from transformers import WhisperFeatureExtractor

        _WHISPER_FEATURE_EXTRACTOR = WhisperFeatureExtractor.from_pretrained(_get_whisper_base_dir())
    return _WHISPER_FEATURE_EXTRACTOR


class PretrainDataset(Dataset):
    """
    Pretrain dataset format (created by build_pretrain_parquet.py):
      - speech_bytes: raw wav bytes (pa.binary)
      - transcript_bytes: utf-8 bytes (pa.binary)

    Each __getitem__ constructs a 2-turn chat prompt:
      user:  <speech> + "Please transcribe the speech."
      assistant: transcript text

    Then it returns:
      (input_ids, labels, speech_values, speech_lengths)
    仅语音：无图像、无 CLIP/像素输入，仅 mel 特征与文本 token。
    若传入 preprocess / image_special_token（兼容旧脚本），会被忽略。
    """

    def __init__(
        self,
        parquet_path: str,
        tokenizer,
        max_length: int = 768,
        speech_special_token: str = "#" * 150,
        prompt_text: str = "",
        fallback_speech_T: int = 3000,
        enable_spec_augment: bool = True,
        enable_wave_augment: bool = True,
        # 以下为兼容旧调用（如 train_sft_olm / validate_data_flow），语音预训练不使用图像与 CLIP
        preprocess=None,
        image_special_token=None,
    ):
        super().__init__()
        self.parquet_path = parquet_path
        # NOTE: pyarrow.Table is generally not picklable on Windows spawn,
        # so we keep it as a runtime cache and reload it inside workers when needed.
        self.table = pq.read_table(parquet_path, memory_map=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.speech_token = speech_special_token
        self.prompt_text = prompt_text
        # WhisperFeatureExtractor 默认 mel bins 固定为 80，这里保持一致。
        self.n_mels = 80
        self.fallback_speech_T = fallback_speech_T
        self.enable_spec_augment = enable_spec_augment
        self.enable_wave_augment = enable_wave_augment
        self.extractor = _get_whisper_feature_extractor()

        self.columns = set(self.table.column_names)
        if "speech_bytes" not in self.columns or "transcript_bytes" not in self.columns:
            raise ValueError(
                f"Invalid pretrain parquet schema. Need columns: speech_bytes, transcript_bytes. "
                f"Got: {sorted(self.columns)}"
            )

        # For label generation: locate assistant message span via BOS/EOS token subsequences.
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.table)

    def __getstate__(self):
        """
        Ensure the dataset object can be pickled for DataLoader workers on Windows.
        pyarrow.Table is not safely picklable, so drop it from the state and reload in __getitem__.
        """
        state = self.__dict__.copy()
        state["table"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def create_chat_prompt(self, transcript: str) -> str:
        base_prompt = self.prompt_text if self.prompt_text else random.choice(PROMPTS_ZH)
        conversations = [
            {"content": base_prompt},
            {"content": transcript},
        ]
        messages = []
        for i, turn in enumerate(conversations):
            role = "user" if i % 2 == 0 else "assistant"
            content = turn["content"].replace("<speech>", self.speech_token)
            messages.append({"role": role, "content": content})

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    def generate_labels(self, input_ids: List[int]) -> List[int]:
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        assert any(l != -100 for l in labels), (
            "labels全为-100，assistant边界未找到，请检查tokenizer chat template格式。"
        )
        return labels

    def _decode_wav_bytes_to_audio(self, wav_bytes: bytes) -> Tuple[np.ndarray, int]:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            num_channels = wf.getnchannels()
            n_frames = wf.getnframes()
            pcm = wf.readframes(n_frames)

        if sample_width != 2:
            raise ValueError(f"Only 16-bit PCM wav is supported, got sample_width={sample_width}")

        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if num_channels > 1:
            audio = audio.reshape(-1, num_channels).mean(axis=1)

        # waveform-level augmentation
        if self.enable_wave_augment:
            if random.random() < 0.5:
                audio = audio * np.random.uniform(0.8, 1.2)
            if random.random() < 0.3:
                audio = audio + (np.random.randn(len(audio)).astype(np.float32) * 0.005)
            audio = np.clip(audio, -1.0, 1.0)
        return audio, sample_rate

    @staticmethod
    def _spec_augment(features: torch.Tensor) -> torch.Tensor:
        """
        features: [1, 80, 3000]
        """
        feat = features.clone()
        n_mels = feat.size(1)
        t_steps = feat.size(2)

        # Frequency masking
        for _ in range(2):
            f = random.randint(0, min(15, n_mels))
            if f == 0:
                continue
            f0 = random.randint(0, n_mels - f)
            feat[0, f0:f0 + f, :] = 0

        # Time masking
        for _ in range(2):
            t = random.randint(0, min(80, t_steps))
            if t == 0:
                continue
            t0 = random.randint(0, t_steps - t)
            feat[0, :, t0:t0 + t] = 0
        return feat

    def _speech_bytes_to_tensor(self, speech_bytes: bytes) -> Optional[Tuple[torch.Tensor, torch.LongTensor]]:
        """
        Return:
          speech_tensor: [T, 80] float32
          speech_lengths: [1] long
        """
        try:
            audio, sample_rate = self._decode_wav_bytes_to_audio(speech_bytes)
            if sample_rate != 16000:
                duration = len(audio) / sample_rate
                target_len = int(duration * 16000)
                audio = np.interp(
                    np.linspace(0, duration, target_len, endpoint=False),
                    np.linspace(0, duration, len(audio), endpoint=False),
                    audio,
                )

            features = self.extractor(audio, sampling_rate=16000, return_tensors="pt").input_features  # [1,80,3000]
            if self.enable_spec_augment:
                features = self._spec_augment(features)
            speech_tensor = features[0].transpose(0, 1).contiguous()  # [3000,80]
            speech_lengths = torch.LongTensor([speech_tensor.size(0)])
            return speech_tensor, speech_lengths
        except Exception:
            # Returning None is safer than injecting fake silent samples into training.
            return None

    def __getitem__(self, index: int):
        if self.table is None:
            # Worker-safe lazy init
            self.table = pq.read_table(self.parquet_path, memory_map=True)
        speech_bytes = self.table["speech_bytes"][index].as_py()
        transcript_bytes = self.table["transcript_bytes"][index].as_py()
        if speech_bytes is None or transcript_bytes is None:
            return None

        transcript = transcript_bytes.decode("utf-8", errors="ignore")
        speech_result = self._speech_bytes_to_tensor(speech_bytes)
        if speech_result is None:
            return None
        speech_tensor, speech_lengths = speech_result

        prompt = self.create_chat_prompt(transcript)
        input_ids = self.tokenizer(prompt).input_ids[: self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)

        speech_values = speech_tensor.unsqueeze(0)  # [1, T, 80]
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            speech_values,
            speech_lengths,
        )


def pretrain_collate_fn(batch):
    """
    Filter invalid samples (None) and stack valid items.
    """
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    input_ids, labels, speech_values, speech_lengths = zip(*batch)
    return (
        torch.stack(input_ids, dim=0),
        torch.stack(labels, dim=0),
        torch.stack(speech_values, dim=0),
        torch.stack(speech_lengths, dim=0),
    )

