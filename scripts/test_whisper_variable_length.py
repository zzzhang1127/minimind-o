#!/usr/bin/env python3
"""
验证 Whisper 变长 mel 与 HF WhisperEncoder 的约束。

结论要点（见 main 末尾打印）：
1. WhisperFeatureExtractor(..., padding=False) 可得到 **时间维可变** 的 log-mel（不必 3000 帧）。
2. transformers 的 WhisperEncoder **强制** input_features 最后一维 = expected_seq_length
   （whisper-base 一般为 3000），否则会报错。
3. 将 mel **右 pad 到 3000** 再送入 encoder 时，**last_hidden_state 时间维恒为 1500**
   （与「满 30s」时相同）；短音频的有效信息集中在靠前位置，尾部多为 pad 对应的帧，
   下游应用 speech_lengths / mask 处理，而不是得到「少于 1500 的 encoder 向量」。

用法（在 minimind-o 根目录）:
  python scripts/test_whisper_variable_length.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
WHISPER_DIR = REPO_ROOT / "model" / "speech_model" / "whisper-base"


def load_wav_mono_16k(path: Path) -> np.ndarray:
    import wave

    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        sw = wf.getsampwidth()
        ch = wf.getnchannels()
        n = wf.getnframes()
        pcm = wf.readframes(n)
    if sw != 2:
        raise ValueError(f"need 16-bit PCM: {path}")
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        dur = len(audio) / sr
        tl = int(dur * 16000)
        audio = np.interp(
            np.linspace(0, dur, tl, endpoint=False),
            np.linspace(0, dur, len(audio), endpoint=False),
            audio,
        )
    return audio


def collect_wavs(max_n: int = 500) -> list[Path]:
    d = REPO_ROOT / "dataset"
    if not d.exists():
        return []
    wavs = [p for p in d.rglob("*.wav") if p.is_file()]
    random.shuffle(wavs)
    return wavs[:max_n]


def main() -> int:
    if not WHISPER_DIR.exists():
        print(f"未找到本地 Whisper: {WHISPER_DIR}", file=sys.stderr)
        return 1

    from transformers import WhisperFeatureExtractor, WhisperModel

    feature_extractor = WhisperFeatureExtractor.from_pretrained(str(WHISPER_DIR))
    whisper = WhisperModel.from_pretrained(str(WHISPER_DIR))
    encoder = whisper.encoder.eval()

    expected_mel_len = (
        encoder.config.max_source_positions
        * encoder.conv1.stride[0]
        * encoder.conv2.stride[0]
    )

    wavs = collect_wavs(500)
    samples: list[tuple[str, np.ndarray]] = []
    for p in wavs[:10]:
        try:
            samples.append((str(p.relative_to(REPO_ROOT)), load_wav_mono_16k(p)))
        except Exception as e:
            print(f"skip {p}: {e}", file=sys.stderr)

    if len(samples) < 10:
        rng = np.random.default_rng(0)
        secs = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 25.0]
        for sec in secs:
            if len(samples) >= 10:
                break
            a = rng.standard_normal(int(16000 * sec)).astype(np.float32) * 0.1
            samples.append((f"synthetic_{sec}s_noise", a))

    if len(samples) < 10:
        print("可用样本不足 10 条。", file=sys.stderr)
        return 1
    samples = samples[:10]

    print(
        f"WhisperFeatureExtractor: padding=False → mel 时间维可变（不必 {expected_mel_len}）\n"
        f"WhisperEncoder 要求: input_features.shape[-1] == {expected_mel_len}（见 modeling_whisper.py）\n"
        f"encoder 输出长度: last_hidden_state.shape[1] == {encoder.config.max_source_positions}（固定）\n"
        f"模型路径: {WHISPER_DIR}\n"
        + "-" * 80
    )

    mel_lt_3000 = 0
    for name, audio in samples:
        dur = len(audio) / 16000.0
        inputs = feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=False,
        )
        feats = inputs.input_features  # [1, 80, T_mel]
        t_mel = feats.shape[-1]
        if t_mel < expected_mel_len:
            mel_lt_3000 += 1

        # 必须 pad 到 expected_mel_len 才能过 HF WhisperEncoder
        feats_pad = F.pad(feats, (0, expected_mel_len - t_mel), value=0.0)

        with torch.no_grad():
            enc_out = encoder(feats_pad)
            hidden = enc_out.last_hidden_state
        t_enc = hidden.shape[1]

        print(
            f"{name[:52]:<52}  dur={dur:6.2f}s  mel_T={t_mel:4d}  enc_T={t_enc:4d}  "
            f"({'mel<3000' if t_mel < expected_mel_len else 'mel=3000'})"
        )

    print("-" * 80)
    print(
        f"本批 10 条中，padding=False 时 mel_T < {expected_mel_len} 的条数: {mel_lt_3000}/10\n"
        "\n结论:\n"
        f"  - FeatureExtractor 可以不 pad 到 3000，得到变长 mel（上表 mel_T）。\n"
        f"  - 当前 transformers WhisperEncoder 不接受 mel_T < {expected_mel_len}，需先 pad 再前向。\n"
        f"  - Pad 后 encoder 输出长度恒为 {encoder.config.max_source_positions}，不是「短音频就更少向量」。\n"
        "  - 变长语义应靠 speech_lengths / mask 在投影或融合时忽略 pad 部分，而不是更少个 encoder 位置。"
    )
    return 0


if __name__ == "__main__":
    random.seed(42)
    raise SystemExit(main())
