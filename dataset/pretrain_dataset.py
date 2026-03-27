import bisect
import glob as glob_mod
import io
import os
import wave
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

__package__ = "dataset"

# Make relative imports from repo root work when running from trainer/
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model_olm import num_speech_tokens_from_encoder_length

_WHISPER_BASE_DIR: Optional[str] = None


def resolve_parquet_paths(parquet_path: Union[str, Path]) -> List[Path]:
    """
    支持：
      - 单个 .parquet 文件
      - 目录：递归或非递归收集 *.parquet（此处为单层目录下所有 *.parquet）
      - 含通配符的路径（如 data/part_*.parquet）

    训练大数据集时建议拆成多个 shard 文件，避免单文件过大。
    """
    p = Path(parquet_path)
    if p.is_dir():
        return sorted(p.glob("*.parquet"))
    s = str(parquet_path)
    if any(ch in s for ch in "*?["):
        files = sorted(glob_mod.glob(s))
        if not files:
            raise FileNotFoundError(f"通配路径未匹配到任何文件: {parquet_path}")
        return [Path(x) for x in files]
    if p.is_file():
        return [p]
    raise FileNotFoundError(f"不是有效的 parquet 文件或目录: {parquet_path}")


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
      user:  将 `<speech>` 替换为「#」重复 N 次（N = (mel_len//2) // 10，即 encoder 有效帧 P 每 10 帧对应 1 个 LLM token；P<10 则 N=0，样本丢弃）
      assistant: transcript text

    Then it returns:
      (input_ids, labels, speech_values, speech_lengths)
    仅语音：无图像、无 CLIP/像素输入，仅 mel 特征与文本 token。
    若传入 preprocess / image_special_token（兼容旧脚本），会被忽略。

    **内存**：不一次性 `read_table` 全表。仅扫描元数据建立「全局行号 → (文件, row_group, 行内偏移)」；
    `__getitem__` 时只 `read_row_group` 加载包含该行的那一个行组（仅两列）。
    若单个 row group 仍极大（例如整库一个 group），请在生成 Parquet 时调小 row group 或拆成多个 shard 文件，
    并将 `parquet_path` 设为目录或 `part_*.parquet` 通配。
    """

    def __init__(
        self,
        parquet_path: Union[str, Path],
        tokenizer,
        max_length: int = 768,
        speech_special_token: Optional[str] = None,
        prompt_text: str = "",
        fallback_speech_T: int = 3000,
        enable_spec_augment: bool = True,
        enable_wave_augment: bool = True,
        preprocess=None,
        image_special_token=None,
    ):
        super().__init__()
        self.parquet_path = str(parquet_path)
        self._paths = resolve_parquet_paths(parquet_path)
        if not self._paths:
            raise FileNotFoundError(
                f"未找到任何 .parquet 文件（目录为空或通配无匹配）: {parquet_path}"
            )
        self._pf_cache: Dict[str, pq.ParquetFile] = {}
        # _cum_start[k] 为第 k 个 row_group 对应的全局起始行号；最后一项为总行数
        self._cum_start: List[int] = [0]
        # (parquet 路径, row_group_index, 该组行数)
        self._seg_meta: List[Tuple[str, int, int]] = []

        for path in self._paths:
            pf = pq.ParquetFile(path, memory_map=True)
            for rg in range(pf.num_row_groups):
                n = pf.metadata.row_group(rg).num_rows
                if n <= 0:
                    continue
                self._seg_meta.append((str(path), rg, n))
                self._cum_start.append(self._cum_start[-1] + n)

        if not self._seg_meta:
            raise ValueError(
                f"Parquet 中无有效行（或 row group 均为空）: {parquet_path}"
            )

        # 仅校验 schema 列名（读 Parquet 文件头/metadata），禁止 read_row_group 解压首个行组，避免大行组 OOM
        p0, _, _ = self._seg_meta[0]
        _pf0 = pq.ParquetFile(p0, memory_map=True)
        cols = set(_pf0.schema_arrow.names)
        if "speech_bytes" not in cols or "transcript_bytes" not in cols:
            raise ValueError(
                f"Invalid pretrain parquet schema. Need columns: speech_bytes, transcript_bytes. "
                f"Got: {sorted(cols)}"
            )

        self.tokenizer = tokenizer
        self.max_length = max_length
        _ = speech_special_token
        self.prompt_text = prompt_text
        self.n_mels = 80
        self.fallback_speech_T = fallback_speech_T
        self.enable_spec_augment = enable_spec_augment
        self.enable_wave_augment = enable_wave_augment
        self.extractor = _get_whisper_feature_extractor()

        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids

    def _get_parquet_file(self, path_str: str) -> pq.ParquetFile:
        if path_str not in self._pf_cache:
            self._pf_cache[path_str] = pq.ParquetFile(path_str, memory_map=True)
        return self._pf_cache[path_str]

    def __len__(self):
        return self._cum_start[-1]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_pf_cache"] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._pf_cache = getattr(self, "_pf_cache", {})

    def create_chat_prompt(self, transcript: str, n_speech_tokens: int) -> str:
        base_prompt = self.prompt_text if self.prompt_text else random.choice(PROMPTS_ZH)
        speech_placeholder = "#" * n_speech_tokens
        conversations = [
            {"content": base_prompt},
            {"content": transcript},
        ]
        messages = []
        for i, turn in enumerate(conversations):
            role = "user" if i % 2 == 0 else "assistant"
            content = turn["content"].replace("<speech>", speech_placeholder)
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
        j = bisect.bisect_right(self._cum_start, index) - 1
        path_str, rg, nrows = self._seg_meta[j]
        local_i = index - self._cum_start[j]
        if not (0 <= local_i < nrows):
            raise IndexError(f"index {index} out of range for segment {j} (local {local_i}, nrows {nrows})")
        pf = self._get_parquet_file(path_str)
        tbl = pf.read_row_group(rg, columns=["speech_bytes", "transcript_bytes"])
        speech_bytes = tbl["speech_bytes"][local_i].as_py()
        transcript_bytes = tbl["transcript_bytes"][local_i].as_py()
        del tbl  # 尽快释放本行组解压后的 Table，降低峰值内存
        if speech_bytes is None or transcript_bytes is None:
            return None

        transcript = transcript_bytes.decode("utf-8", errors="ignore")
        speech_result = self._speech_bytes_to_tensor(speech_bytes)
        if speech_result is None:
            return None
        speech_tensor, speech_lengths = speech_result

        mel_len = int(speech_lengths.item())
        encoder_len = mel_len // 2
        n_speech_tokens = num_speech_tokens_from_encoder_length(encoder_len)
        if n_speech_tokens <= 0:
            return None

        prompt = self.create_chat_prompt(transcript, n_speech_tokens)
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

