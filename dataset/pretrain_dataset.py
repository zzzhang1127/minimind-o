import io
import os
import wave
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset

__package__ = "dataset"

# Make relative imports from repo root work when running from trainer/
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model_olm import MiniMindOLM

_WHISPER_BASE_DIR: Optional[str] = None
_WHISPER_FEATURE_EXTRACTOR = None


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
      (input_ids, labels, pixel_values, speech_values, speech_lengths)
    """

    def __init__(
        self,
        parquet_path: str,
        tokenizer,
        preprocess=None,
        max_length: int = 768,
        speech_special_token: str = "#" * 150,
        prompt_text: str = "<speech>\nPlease transcribe the speech.",
        fallback_speech_T: int = 3000,
    ):
        super().__init__()
        self.parquet_path = parquet_path
        # NOTE: pyarrow.Table is generally not picklable on Windows spawn,
        # so we keep it as a runtime cache and reload it inside workers when needed.
        self.table = pq.read_table(parquet_path, memory_map=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        self.speech_token = speech_special_token
        self.prompt_text = prompt_text
        # WhisperFeatureExtractor 默认 mel bins 固定为 80，这里保持一致。
        self.n_mels = 80
        self.fallback_speech_T = fallback_speech_T

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
        conversations = [
            {"content": self.prompt_text},
            {"content": transcript},
        ]
        messages = []
        for i, turn in enumerate(conversations):
            role = "user" if i % 2 == 0 else "assistant"
            # Speech pretrain: never insert image placeholder tokens.
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
        return audio, sample_rate

    def _speech_bytes_to_tensor(self, speech_bytes: bytes) -> Tuple[torch.Tensor, torch.LongTensor]:
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

            extractor = _get_whisper_feature_extractor()
            features = extractor(audio, sampling_rate=16000, return_tensors="pt").input_features  # [1,80,3000]
            speech_tensor = features[0].transpose(0, 1).contiguous()  # [3000,80]
            speech_lengths = torch.LongTensor([speech_tensor.size(0)])
            return speech_tensor, speech_lengths
        except Exception:
            # Fallback: fixed-size zeros so training doesn't crash on a bad wav.
            zeros = torch.zeros((self.fallback_speech_T, 80), dtype=torch.float32)
            lengths = torch.LongTensor([self.fallback_speech_T])
            return zeros, lengths

    def _dummy_image_tensor(self) -> torch.Tensor:
        # Match previous OLMDataset's shape: [1, 1, 3, 224, 224]
        return torch.zeros((1, 1, 3, 224, 224), dtype=torch.float32)

    def __getitem__(self, index: int):
        if self.table is None:
            # Worker-safe lazy init
            self.table = pq.read_table(self.parquet_path, memory_map=True)
        speech_bytes = self.table["speech_bytes"][index].as_py()
        transcript_bytes = self.table["transcript_bytes"][index].as_py()
        if speech_bytes is None or transcript_bytes is None:
            # Extremely defensive fallback
            speech_tensor = torch.zeros((self.fallback_speech_T, self.n_mels), dtype=torch.float32)
            speech_lengths = torch.LongTensor([self.fallback_speech_T])
            transcript = ""
        else:
            transcript = transcript_bytes.decode("utf-8", errors="ignore")
            speech_tensor, speech_lengths = self._speech_bytes_to_tensor(speech_bytes)

        prompt = self.create_chat_prompt(transcript)
        input_ids = self.tokenizer(prompt).input_ids[: self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)

        pixel_values = self._dummy_image_tensor()
        speech_values = speech_tensor.unsqueeze(0)  # [1, T, 80]
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            pixel_values,
            speech_values,
            speech_lengths,
        )

