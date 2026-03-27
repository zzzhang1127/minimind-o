import argparse
import json
import os
import random
import re
import time
import warnings
from pathlib import Path
from typing import Optional
import sys

import torch
from PIL import Image
from transformers import AutoTokenizer

from model.model_olm import MiniMindOLM, OLMConfig, num_speech_tokens_from_encoder_length
from trainer.trainer_utils import setup_seed, get_model_params

warnings.filterwarnings('ignore')

# Make console output encoding stable on Windows (PowerShell sometimes uses a non-UTF8 code page)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}

# 各 mode 在未指定 --prompt 时使用的默认用户提示（不含占位符，build_prompt 会按需追加 <image>/<speech>）
DEFAULT_PROMPTS = {
    "text": "你好，请用一句话介绍你自己。",
    "vision": "图片里有什么？",
    "speech": "请转录这段语音。",
    "both": "请结合图片与语音内容，用中文简要回答。",
}
WEIGHT_EXTS_PRIORITY = [
    '.pth',
    '.bin',
    '.safetensors',
]
WEIGHT_EXTS = set(WEIGHT_EXTS_PRIORITY)


def _load_json(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _resolve_tokenizer_source(repo_dir: Path) -> str:
    # tokenizer in minimind-o/model
    local_model_dir = repo_dir / "model"
    if local_model_dir.exists():
        return str(local_model_dir)
    return str(repo_dir)


def _pick_first_weight_file(candidates: list[Path]) -> Path:
    if not candidates:
        raise FileNotFoundError("未找到任何权重文件")
    # Prefer .pth > .bin > .safetensors (when no --weight prefix is provided)
    def key(p: Path):
        ext = p.suffix.lower()
        prio = WEIGHT_EXTS_PRIORITY.index(ext) if ext in WEIGHT_EXTS_PRIORITY else 999
        return (prio, p.name)
    return sorted(candidates, key=key)[0]


def _resolve_weight_path(args, load_from: Path, config: Optional[OLMConfig]) -> Path:
    if load_from.is_file():
        # user passed explicit file
        return load_from

    # Directory case: allow .bin/.pth/.safetensors
    candidates = [p for p in load_from.iterdir() if p.is_file() and p.suffix.lower() in WEIGHT_EXTS]
    if not candidates:
        raise FileNotFoundError(f"在目录中未找到任何权重文件(.pth/.bin/.safetensors): {load_from}")

    weight_prefix = str(args.weight).strip() if getattr(args, "weight", None) else ""

    # If --weight provided, try to match {prefix}_{hidden_size} when config is available.
    if weight_prefix:
        matched = []
        hs = getattr(config, "hidden_size", None) if config is not None else None
        for p in candidates:
            name = p.name
            if not (name.startswith(weight_prefix + "_") or name.startswith(weight_prefix)):
                continue
            if hs is not None and f"_{hs}" in name:
                matched.append(p)
            else:
                matched.append(p)

        if matched:
            # Prefer .pth when multiple match.
            return _pick_first_weight_file(matched)

    # No --weight: return the first weight (prefer .pth)
    return _pick_first_weight_file(candidates)


def _filter_state_dict_for_inference(state_dict: dict) -> dict:
    out = {}
    for k, v in state_dict.items():
        lk = k.lower()
        if 'mask' in lk:
            continue
        if lk.startswith('vision_encoder.') or lk.startswith('speech_encoder.'):
            continue
        if not torch.is_tensor(v):
            continue
        out[k] = v
    return out


def _maybe_extract_state_dict(obj) -> dict:
    # torch.load could return: state_dict | {"state_dict": ...} | {"model": ...} | others
    if isinstance(obj, dict):
        if 'state_dict' in obj and isinstance(obj['state_dict'], dict):
            return obj['state_dict']
        if 'model' in obj and isinstance(obj['model'], dict):
            return obj['model']
        # fallback: if it looks like a state_dict (contains tensor values)
        if any(torch.is_tensor(v) for v in obj.values()):
            return obj
    raise ValueError("无法从权重文件中提取 state_dict")


def _load_state_dict_from_weight_path(weight_path: Path) -> dict:
    ext = weight_path.suffix.lower()
    if ext == '.safetensors':
        try:
            from safetensors.torch import load_file
        except Exception as e:
            raise RuntimeError(
                "加载 .safetensors 权重需要 safetensors 包，但当前环境缺失。"
            ) from e
        return load_file(str(weight_path), device="cpu")

    # .pth / .bin
    raw = torch.load(str(weight_path), map_location='cpu')
    return _maybe_extract_state_dict(raw)


def _infer_hidden_size_layers_use_moe_from_state_dict(state_dict: dict) -> dict:
    hidden_size = None
    vocab_size = None
    layer_indices = set()
    use_moe = False

    # Find hidden_size/vocab_size
    for k, v in state_dict.items():
        if not torch.is_tensor(v):
            continue
        if k.endswith('embed_tokens.weight') or k.endswith('lm_head.weight'):
            if v.ndim == 2:
                vocab_size = int(v.shape[0])
                hidden_size = int(v.shape[1])
                break

    # num_hidden_layers and moe
    layer_re = re.compile(r'(?:^|\.)(?:model\.)?layers\.(\d+)\.')
    for k, v in state_dict.items():
        lk = k.lower()
        if ('mlp.experts.' in lk) or ('mlp.shared_experts.' in lk):
            use_moe = True
        m = layer_re.search(k)
        if m:
            idx = m.group(1)
            if idx is not None:
                layer_indices.add(int(idx))

    num_hidden_layers = (max(layer_indices) + 1) if layer_indices else None

    # Infer intermediate_size (FFN hidden) to avoid shape mismatch when config.json is missing
    intermediate_size = None
    if hidden_size is not None:
        for k, v in state_dict.items():
            if not torch.is_tensor(v):
                continue
            lk = k.lower()
            if lk.endswith('gate_proj.weight') and v.ndim == 2 and int(v.shape[1]) == hidden_size:
                intermediate_size = int(v.shape[0])
                break

    # Infer attention head counts from k_proj weight
    num_attention_heads = 8
    num_key_value_heads = 2
    if hidden_size is not None:
        kproj = None
        for k, v in state_dict.items():
            if not torch.is_tensor(v):
                continue
            if k.lower().endswith('k_proj.weight') and v.ndim == 2 and int(v.shape[1]) == hidden_size:
                kproj = v
                break
        if kproj is not None:
            out_k = int(kproj.shape[0])
            candidates = []
            # brute force: num_attention_heads divides hidden_size
            for h in range(1, hidden_size + 1):
                if hidden_size % h != 0:
                    continue
                # out_k = num_key_value_heads * (hidden_size/num_attention_heads)
                nk = out_k * h / hidden_size
                if abs(nk - round(nk)) < 1e-6:
                    nk_int = int(round(nk))
                    if nk_int >= 1 and h % nk_int == 0:
                        candidates.append((h, nk_int))
            pref = [8, 12, 16, 24, 32, 48, 64]
            candidates.sort(key=lambda x: (pref.index(x[0]) if x[0] in pref else 9999, x[0]))
            if candidates:
                num_attention_heads, num_key_value_heads = candidates[0]

    # MoE-specific: n_routed_experts and n_shared_experts (shape-affecting)
    n_routed_experts = None
    n_shared_experts = None
    if use_moe and hidden_size is not None:
        for k, v in state_dict.items():
            if not torch.is_tensor(v):
                continue
            if k.lower().endswith('mlp.gate.weight') and v.ndim == 2 and int(v.shape[1]) == hidden_size:
                n_routed_experts = int(v.shape[0])
                break
        shared_idxs = set()
        shared_re = re.compile(r'mlp\.shared_experts\.(\d+)\.')
        for k in state_dict.keys():
            m = shared_re.search(k)
            if m:
                shared_idxs.add(int(m.group(1)))
        n_shared_experts = (max(shared_idxs) + 1) if shared_idxs else 0

    return {
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "num_hidden_layers": num_hidden_layers,
        "use_moe": use_moe,
        "intermediate_size": intermediate_size,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "n_routed_experts": n_routed_experts,
        "n_shared_experts": n_shared_experts,
    }


def load_olm_config(args, load_from: Path) -> tuple[OLMConfig, Path]:
    load_from_dir = load_from if load_from.is_dir() else load_from.parent
    config_json = load_from_dir / "config.json"

    if config_json.exists():
        cfg = _load_json(config_json)
        # Only pick fields we care about (avoid unexpected keys in config.json)
        cfg_hidden_size = cfg.get("hidden_size", None)
        cfg_num_layers = cfg.get("num_hidden_layers", None)
        cfg_use_moe = cfg.get("use_moe", cfg.get("use_moe", False))

        olm_config = OLMConfig(
            hidden_size=int(cfg_hidden_size) if cfg_hidden_size is not None else 768,
            num_hidden_layers=int(cfg_num_layers) if cfg_num_layers is not None else 8,
            use_moe=bool(cfg_use_moe),
            vocab_size=int(cfg.get("vocab_size", 6400)),
            intermediate_size=cfg.get("intermediate_size", None),
            num_attention_heads=int(cfg.get("num_attention_heads", 8)),
            num_key_value_heads=int(cfg.get("num_key_value_heads", 2)),
            n_routed_experts=int(cfg.get("n_routed_experts", 4)),
            n_shared_experts=int(cfg.get("n_shared_experts", 1)),
        )
        # keep image/speech special tokens as requested by minimind-o defaults
        # If config.json provides image tokens, you can choose to override them later.

        weight_path = _resolve_weight_path(args, load_from, olm_config)
        return olm_config, weight_path

    # No config.json: pick a weight file first, then infer model parameters from it.
    weight_path = _resolve_weight_path(args, load_from, None)
    state_dict = _load_state_dict_from_weight_path(weight_path)
    inferred = _infer_hidden_size_layers_use_moe_from_state_dict(state_dict)

    if inferred["hidden_size"] is None or inferred["num_hidden_layers"] is None:
        raise RuntimeError(f"从权重文件推断失败: {weight_path}")

    olm_config = OLMConfig(
        hidden_size=int(inferred["hidden_size"]),
        num_hidden_layers=int(inferred["num_hidden_layers"]),
        use_moe=bool(inferred["use_moe"]),
        vocab_size=int(inferred["vocab_size"]) if inferred["vocab_size"] is not None else 6400,
        intermediate_size=int(inferred["intermediate_size"]) if inferred["intermediate_size"] is not None else None,
        num_attention_heads=int(inferred["num_attention_heads"]),
        num_key_value_heads=int(inferred["num_key_value_heads"]),
        n_routed_experts=int(inferred["n_routed_experts"]) if inferred["n_routed_experts"] is not None else 4,
        n_shared_experts=int(inferred["n_shared_experts"]) if inferred["n_shared_experts"] is not None else 1,
    )
    return olm_config, weight_path


def init_model(args):
    repo_dir = Path(__file__).resolve().parent
    tokenizer_source = _resolve_tokenizer_source(repo_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    load_from = Path(args.load_from).expanduser()
    # If user passes a relative path that doesn't exist (common when running from minimind-o/),
    # try resolving it relative to github root (repo_dir.parent).
    if not load_from.exists() and not load_from.is_absolute():
        alt = (repo_dir.parent / load_from).resolve()
        if alt.exists():
            load_from = alt
    olm_config, weight_path = load_olm_config(args, load_from)

    load_vision_encoder = args.mode in {"vision", "both"}
    load_speech_encoder = args.mode in {"speech", "both"}

    model = MiniMindOLM(
        olm_config,
        vision_model_path=str(repo_dir / "model" / "vision_model" / "clip-vit-base-patch16"),
        load_vision_encoder=load_vision_encoder,
        load_speech_encoder=load_speech_encoder,
    )

    state_dict = _load_state_dict_from_weight_path(weight_path)
    filtered = _filter_state_dict_for_inference(state_dict)
    if args.mode in {"vision", "both"} and not any("vision_proj" in k.lower() for k in filtered.keys()):
        print("⚠️  当前权重不包含 `vision_proj`，视觉对齐能力可能缺失（输出可能与图片无关）。")
    if args.mode in {"speech", "both"} and not any("speech_proj" in k.lower() for k in filtered.keys()):
        print("⚠️  当前权重不包含 `speech_proj`，语音对齐能力可能缺失（输出可能与语音无关）。")

    model.load_state_dict(filtered, strict=False)
    get_model_params(model, model.params)

    preprocess = model.processor if load_vision_encoder else None
    return model.eval().to(args.device), tokenizer, preprocess


def build_prompt(model, text_prompt, with_image=False, with_speech=False, n_speech_tokens: int = 0):
    prompt = text_prompt
    if with_image and '<image>' not in prompt:
        prompt = prompt + "\n<image>"
    if with_speech and '<speech>' not in prompt:
        prompt = prompt + "\n<speech>"

    prompt = prompt.replace('<image>', model.params.image_special_token)
    if with_speech:
        if n_speech_tokens > 0:
            prompt = prompt.replace('<speech>', '#' * n_speech_tokens)
        else:
            prompt = prompt.replace('<speech>', '')
    return prompt


def _pick_random_image(repo_dir: Path) -> str:
    eval_images_dir = repo_dir / "dataset" / "eval_images"
    if not eval_images_dir.exists():
        raise FileNotFoundError(f"未找到评测图片目录: {eval_images_dir}")

    image_files = [p for p in eval_images_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    if not image_files:
        raise FileNotFoundError(f"目录内未找到图片文件: {eval_images_dir}")

    chosen = random.choice(image_files)
    print(f"[Vision] 随机选择图片: {chosen.name}")
    return str(chosen)


def _pick_random_speech(repo_dir: Path) -> str:
    eval_speech_dir = repo_dir / "dataset" / "eval_speeches"
    if not eval_speech_dir.exists():
        raise FileNotFoundError(f"未找到评测语音目录: {eval_speech_dir}")

    audio_files = [
        p for p in eval_speech_dir.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    ]
    if not audio_files:
        raise FileNotFoundError(f"目录内未找到音频文件({', '.join(sorted(AUDIO_EXTS))}): {eval_speech_dir}")

    chosen = random.choice(audio_files)
    print(f"[Speech] 随机选择语音: {chosen.name}")
    return str(chosen)


def _validate_mode_inputs(args, repo_dir: Path):
    has_img = bool(args.image_path)
    has_audio = bool(args.audio_path)

    # hard validations
    if args.mode == "text":
        if has_img:
            raise ValueError("mode=text 不允许传入 --image_path")
        if has_audio:
            raise ValueError("mode=text 不允许传入 --audio_path")
        return

    if args.mode == "vision":
        if has_audio:
            raise ValueError("mode=vision 不允许传入 --audio_path")
        if args.image_path and not os.path.exists(args.image_path):
            raise FileNotFoundError(f"--image_path 不存在: {args.image_path}")
        return

    if args.mode == "speech":
        if has_img:
            raise ValueError("mode=speech 不允许传入 --image_path")
        if args.audio_path and not os.path.exists(args.audio_path):
            raise FileNotFoundError(f"--audio_path 不存在: {args.audio_path}")
        return

    if args.mode == "both":
        if args.image_path and not os.path.exists(args.image_path):
            raise FileNotFoundError(f"--image_path 不存在: {args.image_path}")
        if args.audio_path and not os.path.exists(args.audio_path):
            raise FileNotFoundError(f"--audio_path 不存在: {args.audio_path}")
        return

    raise ValueError(f"未知 mode: {args.mode}")


def main():
    repo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="MiniMind-O Eval")
    parser.add_argument('--mode', choices=["text", "vision", "speech", "both"], default="speech", type=str)
    parser.add_argument(
        '--load_from',
        default=str(repo_dir / "out"),
        type=str,
        help="模型配置/权重加载目录或权重文件路径（默认: minimind-o/out）"
    )
    parser.add_argument(
        '--weight',
        default='',
        type=str,
        help="模型前缀；不传则从目录里优先取第一个权重文件(.pth/.bin/.safetensors)"
    )

    parser.add_argument('--max_new_tokens', default=512, type=int, help="max new tokens")
    parser.add_argument('--temperature', default=0.1, type=float, help="temperature")
    parser.add_argument('--top_p', default=0.65, type=float, help="top p")
    parser.add_argument(
        '--prompt',
        default=None,
        type=str,
        help="用户提示；不传则按 mode 使用内置默认（vision: 看图问答；speech: 转录；both: 图文音综合）",
    )
    parser.add_argument('--image_path', default='', type=str, help="optional image path")
    parser.add_argument('--audio_path', default='', type=str, help="optional audio path")
    parser.add_argument('--show_speed', default=1, type=int, help="show speed")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="device")
    args = parser.parse_args()

    if args.prompt is None:
        args.prompt = DEFAULT_PROMPTS[args.mode]

    setup_seed(2026)

    # 未指定路径时从评测目录随机抽取（与 vision 行为一致）
    if args.mode == "vision" and not args.image_path:
        args.image_path = _pick_random_image(repo_dir)
    if args.mode == "speech":
        if args.audio_path:
            if not os.path.exists(args.audio_path):
                raise FileNotFoundError(f"--audio_path 不存在: {args.audio_path}")
        else:
            args.audio_path = _pick_random_speech(repo_dir)
    if args.mode == "both":
        if args.image_path:
            if not os.path.exists(args.image_path):
                raise FileNotFoundError(f"--image_path 不存在: {args.image_path}")
        else:
            args.image_path = _pick_random_image(repo_dir)
        if args.audio_path:
            if not os.path.exists(args.audio_path):
                raise FileNotFoundError(f"--audio_path 不存在: {args.audio_path}")
        else:
            args.audio_path = _pick_random_speech(repo_dir)

    _validate_mode_inputs(args, repo_dir)

    model, tokenizer, preprocess = init_model(args)

    pixel_values = None
    if args.mode in {"vision", "both"}:
        if not args.image_path or not os.path.exists(args.image_path):
            raise FileNotFoundError(f"mode={args.mode} 需要存在的 --image_path")
        image = Image.open(args.image_path).convert('RGB')
        if preprocess is None:
            raise RuntimeError("vision encoder 没有加载成功，processor 为 None")
        pixel_values = MiniMindOLM.image2tensor(image, preprocess).to(args.device).unsqueeze(0)

    speech_values = None
    speech_lengths = None
    n_speech_tokens = 0
    enc_len = 0
    if args.mode in {"speech", "both"}:
        if not args.audio_path or not os.path.exists(args.audio_path):
            raise FileNotFoundError(f"mode={args.mode} 需要存在的 --audio_path")
        speech_tensor = MiniMindOLM.speech2tensor(args.audio_path)
        enc_len = speech_tensor.size(0) // 2
        n_speech_tokens = num_speech_tokens_from_encoder_length(enc_len)
        if n_speech_tokens > 0:
            speech_values = speech_tensor.unsqueeze(0).unsqueeze(0).to(args.device)
            speech_lengths = torch.LongTensor([[speech_tensor.size(0)]]).to(args.device)

    prompt = build_prompt(
        model,
        args.prompt,
        with_image=pixel_values is not None,
        with_speech=(args.mode in {"speech", "both"}),
        n_speech_tokens=n_speech_tokens,
    )
    if args.mode in {"speech", "both"}:
        print(
            f"[Speech] mel 时间维 T={speech_tensor.size(0)}, encoder 有效帧 P={enc_len} "
            f"→ 语音 token 数 N={n_speech_tokens}"
            + ("（P<10 无语音 token，仅文本）" if n_speech_tokens == 0 else "")
        )
    messages = [{"role": "user", "content": prompt}]
    inputs_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(inputs_text, return_tensors="pt", truncation=True).to(args.device)

    print(f"Prompt: {args.prompt}")
    print("Answer: ", end="")

    st = time.time()
    generated_ids = model.generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        top_p=args.top_p,
        temperature=args.temperature,
        pixel_values=pixel_values,
        speech_values=speech_values,
        speech_lengths=speech_lengths,
    )

    input_len = inputs["input_ids"].shape[1]
    gen_only_ids = generated_ids[0][input_len:]
    decoded = tokenizer.decode(gen_only_ids, skip_special_tokens=True)
    if args.show_speed:
        gen_tokens = len(generated_ids[0]) - input_len
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n')
    print(decoded)


if __name__ == "__main__":
    main()
