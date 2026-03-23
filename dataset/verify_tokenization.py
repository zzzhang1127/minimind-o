"""
验证语音占位符 tokenization：占位符为「#」重复 N 次，N 随音频有效长度变化
（N = encoder_len//10，见 model_olm.num_speech_tokens_from_encoder_length）。

在 minimind-o 仓库根目录执行：
  python dataset/verify_tokenization.py
"""
import sys
from pathlib import Path

__package__ = "dataset"
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_REPO_ROOT))

from transformers import AutoTokenizer

from model.model_olm import num_speech_tokens_from_encoder_length

# 与 eval_olm._resolve_tokenizer_source 一致：tokenizer.json 在 minimind-o/model/
TOKENIZER_DIR = _REPO_ROOT / "model"


def verify_placeholder_ids():
    tokenizer_path = str(TOKENIZER_DIR)
    if not TOKENIZER_DIR.is_dir():
        print(f"[ERROR] 未找到 model 目录: {TOKENIZER_DIR}")
        return False
    if not (TOKENIZER_DIR / "tokenizer.json").is_file() and not (TOKENIZER_DIR / "tokenizer_config.json").is_file():
        print(f"[ERROR] model 目录下缺少 tokenizer.json / tokenizer_config.json: {TOKENIZER_DIR}")
        return False

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    except Exception as e:
        print(f"[ERROR] 无法加载 tokenizer: {e}")
        return False

    print("[Speech] 占位符验证（可变长 #，N = encoder_len // 10）：")
    for enc_len, label in [(9, "不足10帧"), (100, "约2s"), (500, "约10s"), (1500, "满30s")]:
        n = num_speech_tokens_from_encoder_length(enc_len)
        if n == 0:
            print(f"   encoder_len={enc_len} ({label}) -> N=0（无占位符）")
            continue
        speech_placeholder = "#" * n
        speech_tokens = tokenizer(speech_placeholder, add_special_tokens=False).input_ids
        ok = len(speech_tokens) == n and all(t == 5 for t in speech_tokens)
        print(f"   encoder_len={enc_len} ({label}) -> N={n} tokens, tokenizer_len={len(speech_tokens)}, ok={ok}")

    # 测试 image 占位符
    image_placeholder = '@' * 196
    image_tokens = tokenizer(image_placeholder, add_special_tokens=False).input_ids

    print(f"\n[Image] 占位符验证:")
    print(f"   输入: '@'*196")
    print(f"   Token 数量: {len(image_tokens)}")
    print(f"   Token IDs 前 20 个: {image_tokens[:20]}")
    print(f"   是否全为 34: {all(t == 34 for t in image_tokens)}")
    print(f"   是否恰好 196 个: {len(image_tokens) == 196}")

    if len(image_tokens) != 196:
        print(f"\n[WARN] 期望 196 个 tokens，实际得到 {len(image_tokens)} 个")

    success = len(image_tokens) == 196 and all(t == 34 for t in image_tokens)
    return success


if __name__ == "__main__":
    success = verify_placeholder_ids()
    if success:
        print("\n[OK] 图像占位符 tokenization 正确（语音为可变长，请按 N 构造 '#'）。")
        sys.exit(0)
    else:
        print("\n[FAIL] 占位符 tokenization 存在问题，需要修复。")
        sys.exit(1)
