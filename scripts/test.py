#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从预训练 Parquet 随机抽 5 条语音 → Whisper(mel 仅对有效时长部分计长) → 池化得到变长语音 token 数 N，
与「<speech>\\n语音中提到了什么」拼接后送入全模态 OLM，加载 out/pretrain_olm_768.pth 做推理。

内存：不整表读取。`PretrainDataset` 仅扫 parquet 元数据；`_read_speech_bytes` 只 `read_row_group`
单列取一行（仍受「单行组过大」限制，大数据请多分片/控制 row group 大小）。

用法（在 minimind-o 根目录）:
  python scripts/test.py --parquet dataset/pretrain_s2t.parquet --weight out/pretrain_olm_768.pth
"""
from __future__ import annotations

import argparse
import bisect
import io
import os
import random
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)  # 相对路径 out/、dataset/ 与训练一致

from transformers import AutoTokenizer, WhisperFeatureExtractor

from dataset.pretrain_dataset import PretrainDataset
from eval_olm import (
    _filter_state_dict_for_inference,
    _load_state_dict_from_weight_path,
    build_prompt,
    load_olm_config,
)
from model.model_olm import MiniMindOLM, num_speech_tokens_from_encoder_length


def _read_speech_bytes(dataset: PretrainDataset, index: int) -> bytes:
    """与 PretrainDataset.__getitem__ 相同的行定位，只读 speech_bytes。"""
    j = bisect.bisect_right(dataset._cum_start, index) - 1
    path_str, rg, nrows = dataset._seg_meta[j]
    local_i = index - dataset._cum_start[j]
    if not (0 <= local_i < nrows):
        raise IndexError(f"index {index} out of range")
    pf = dataset._get_parquet_file(path_str)
    tbl = pf.read_row_group(rg, columns=["speech_bytes"])
    b = tbl["speech_bytes"][local_i].as_py()
    del tbl
    return b


def _decode_wav_bytes_to_audio(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        num_channels = wf.getnchannels()
        n_frames = wf.getnframes()
        pcm = wf.readframes(n_frames)
    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM wav, got sample_width={sample_width}")
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if num_channels > 1:
        audio = audio.reshape(-1, num_channels).mean(axis=1)
    if sample_rate != 16000:
        duration = len(audio) / sample_rate
        target_len = int(duration * 16000)
        audio = np.interp(
            np.linspace(0, duration, target_len, endpoint=False),
            np.linspace(0, duration, len(audio), endpoint=False),
            audio,
        )
    return audio, 16000


def wav_bytes_to_speech_batch(
    wav_bytes: bytes,
    extractor: WhisperFeatureExtractor,
    expected_mel: int = 3000,
) -> tuple[torch.Tensor, torch.LongTensor, float, int, int]:
    """
    变长 mel（padding=False）→ 有效 t_mel；再 pad 到 expected_mel 送入 WhisperEncoder。
    speech_lengths 填 **有效 mel 帧数**，与 forward 里 mask / encoder 有效帧一致。
    返回: speech_tensor [T,80], speech_lengths [1], duration_sec, t_mel, encoder_len(P)
    """
    audio, _sr = _decode_wav_bytes_to_audio(wav_bytes)
    duration_sec = len(audio) / 16000.0
    inputs = extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=False,
    )
    feats = inputs.input_features  # [1, 80, t_mel]
    t_mel = int(feats.shape[-1])
    pad = max(0, expected_mel - t_mel)
    feats_pad = F.pad(feats, (0, pad), value=0.0)
    speech_tensor = feats_pad[0].transpose(0, 1).contiguous()  # [expected_mel, 80]
    speech_lengths = torch.LongTensor([min(t_mel, expected_mel)])
    mel_len = int(speech_lengths.item())
    encoder_len = mel_len // 2
    return speech_tensor, speech_lengths, duration_sec, t_mel, encoder_len


def load_model_and_tokenizer(weight_path: Path, device: torch.device):
    args = SimpleNamespace(load_from=str(weight_path), weight="")
    olm_config, resolved_weight = load_olm_config(args, weight_path.expanduser())
    tokenizer = AutoTokenizer.from_pretrained(
        str(REPO_ROOT / "model"),
        local_files_only=True,
    )
    model = MiniMindOLM(
        olm_config,
        vision_model_path=str(REPO_ROOT / "model" / "vision_model" / "clip-vit-base-patch16"),
        load_vision_encoder=False,
        load_speech_encoder=True,
    ).to(device)
    state_dict = _load_state_dict_from_weight_path(resolved_weight)
    model.load_state_dict(_filter_state_dict_for_inference(state_dict), strict=False)
    model.eval()
    return model, tokenizer, olm_config, resolved_weight


def main() -> int:
    parser = argparse.ArgumentParser(description="OLM 语音 batch 推理测试")
    parser.add_argument(
        "--parquet",
        type=str,
        default="dataset/pretrain_s2t.parquet",
        help="预训练 parquet 文件或目录",
    )
    parser.add_argument(
        "--weight",
        type=str,
        default="out/pretrain_olm_768.pth",
        help="权重路径（如 out/pretrain_olm_768.pth）",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--max_length", type=int, default=768, help="tokenizer 截断长度")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        raise FileNotFoundError(f"未找到数据: {parquet_path}")

    weight_path = Path(args.weight)
    if not weight_path.exists():
        raise FileNotFoundError(f"未找到权重: {weight_path}")

    print("加载 tokenizer / 模型 …")
    model, tokenizer, olm_config, resolved_weight = load_model_and_tokenizer(weight_path, device)
    print(f"  权重: {resolved_weight}")
    print(f"  hidden_size={olm_config.hidden_size}, vocab_size={olm_config.vocab_size}")

    # 数据集仅用于按行读 parquet（不跑 __getitem__ 的随机增强）
    dataset = PretrainDataset(
        parquet_path,
        tokenizer,
        max_length=args.max_length,
        enable_spec_augment=False,
        enable_wave_augment=False,
        prompt_text="",
    )
    n_total = len(dataset)
    if n_total < 5:
        raise RuntimeError(f"样本不足 5 条（当前 {n_total}）")

    whisper_dir = REPO_ROOT / "model" / "speech_model" / "whisper-base"
    extractor = WhisperFeatureExtractor.from_pretrained(str(whisper_dir))

    indices = random.sample(range(n_total), 5)
    print(f"\n随机索引 {indices}（共 {n_total} 条）\n")

    # 收集 5 条（跳过无效样本）
    rows: list[dict] = []
    for idx in indices:
        try:
            wav_b = _read_speech_bytes(dataset, idx)
        except Exception as e:
            print(f"[WARN] skip index {idx}: {e}")
            continue
        speech_tensor, speech_lengths, dur, t_mel, enc_len = wav_bytes_to_speech_batch(
            wav_b, extractor
        )
        n_speech = num_speech_tokens_from_encoder_length(enc_len, olm_config.speech_frames_per_token)
        if n_speech <= 0:
            print(f"[WARN] index {idx}: P={enc_len} 过小，语音 token 数 N=0，跳过")
            continue
        rows.append(
            {
                "index": idx,
                "speech_tensor": speech_tensor,
                "speech_lengths": speech_lengths,
                "duration_sec": dur,
                "t_mel": t_mel,
                "encoder_len": enc_len,
                "n_speech": n_speech,
            }
        )
        if len(rows) >= 5:
            break

    if len(rows) < 5:
        raise RuntimeError(f"有效样本不足 5 条（当前 {len(rows)}），请换 --seed 或检查数据。")

    print("-" * 80)
    print(
        "每条样本：有效时长（秒）→ mel 帧 t_mel → encoder 有效帧 P=t_mel//2 → "
        f"池化后语音 token 数 N=P//{olm_config.speech_frames_per_token}（例如约 2s → P≈100 → N≈10）"
    )
    print("-" * 80)
    for r in rows:
        print(
            f"  idx={r['index']:6d}  dur={r['duration_sec']:.2f}s  "
            f"t_mel={r['t_mel']:4d}  P={r['encoder_len']:4d}  N={r['n_speech']:3d}"
        )
    print("-" * 80)

    # 组 batch：speech 已 pad 到 3000，长度一致
    speech_values = torch.stack([r["speech_tensor"] for r in rows], dim=0).unsqueeze(1)  # [B,1,3000,80]
    speech_lengths_b = torch.stack([r["speech_lengths"] for r in rows], dim=0)  # [B,1]

    user_text = "<speech>\n语音中提到了什么"
    prompts = []
    for r in rows:
        p = build_prompt(
            model,
            user_text,
            with_image=False,
            with_speech=True,
            n_speech_tokens=r["n_speech"],
        )
        prompts.append(p)

    messages_list = [[{"role": "user", "content": p}] for p in prompts]
    texts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_list
    ]
    batch = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length,
    )
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    speech_values = speech_values.to(device)
    speech_lengths_b = speech_lengths_b.to(device)

    # 打印每条样本中「#」占位符个数（应与 N 一致）
    speech_id = tokenizer("#", add_special_tokens=False).input_ids[0]
    for i, r in enumerate(rows):
        cnt = (input_ids[i] == speech_id).sum().item()
        print(f"  batch[{i}]  input_ids 中 '#' (id={speech_id}) 的个数: {cnt}  (期望 N={r['n_speech']})")

    # 一次 forward：取每条序列最后一个非 pad 位置的 next-token logits
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            speech_values=speech_values,
            speech_lengths=speech_lengths_b,
        )
        logits = out.logits
        B, L, _V = logits.shape
        lengths = attention_mask.sum(dim=1).long()
        last_pos = (lengths - 1).clamp(min=0)
        next_logits = logits[torch.arange(B, device=device), last_pos]
        next_ids = next_logits.argmax(dim=-1)

    print("\n[Batch 前向] 最后一个上下文位置的 argmax 预测 token id（下一 token 的贪心预测）:")
    for i in range(B):
        tid = int(next_ids[i].item())
        tok_txt = tokenizer.decode([tid], skip_special_tokens=False)
        print(f"  batch[{i}]  pred_id={tid:5d}  decode={tok_txt!r}")

    # 逐条 generate（多模态 batch generate 在不同 HF 版本上行为不一，这里用单条最稳）
    print("\n[逐条 generate] 用户问题: 语音中提到了什么")
    for i, r in enumerate(rows):
        single = tokenizer(
            texts[i],
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )
        sv = speech_values[i : i + 1]
        sl = speech_lengths_b[i : i + 1]
        with torch.no_grad():
            gen_ids = model.generate(
                inputs=single["input_ids"].to(device),
                attention_mask=single["attention_mask"].to(device),
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                speech_values=sv,
                speech_lengths=sl,
            )
        in_len = single["input_ids"].shape[1]
        new_tokens = gen_ids[0, in_len:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(f"\n  --- idx={r['index']}  N={r['n_speech']}  dur={r['duration_sec']:.2f}s ---")
        print(f"  回复: {decoded}")

    print("\n[OK] 完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
