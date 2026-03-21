import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
from pathlib import Path
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_olm import OLMConfig
from dataset.pretrain_dataset import PretrainDataset, pretrain_collate_fn
from trainer.trainer_utils import (
    get_lr,
    Logger,
    is_main_process,
    init_distributed_mode,
    setup_seed,
    init_olm_model,
    olm_checkpoint,
    SkipBatchSampler,
)

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    for step, batch in enumerate(loader, start=start_step + 1):
        if batch is None:
            continue
        input_ids, labels, speech_values, speech_lengths = batch
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        speech_values = speech_values.to(args.device)
        speech_lengths = speech_lengths.to(args.device)

        cur_step = epoch * iters + step
        total_steps = args.epochs * iters
        if getattr(args, "_optimizer_grouped", False):
            optimizer.param_groups[0]["lr"] = get_lr(
                cur_step, total_steps, args.lr_speech_proj
            )
            optimizer.param_groups[1]["lr"] = get_lr(
                cur_step, total_steps, args.lr_llm_last
            )
        else:
            lr = get_lr(cur_step, total_steps, args.learning_rate)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        with autocast_ctx:
            # 仅训练语音编码器输出 + speech_proj；不传入任何视觉输入。
            res = model(
                input_ids,
                labels=labels,
                speech_values=speech_values,
                speech_lengths=speech_lengths,
            )
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            if getattr(args, "_optimizer_grouped", False):
                current_lr = optimizer.param_groups[0]["lr"]
                current_lr_llm = optimizer.param_groups[1]["lr"]
            else:
                current_lr = optimizer.param_groups[-1]["lr"]
                current_lr_llm = None
            # Predicted total epoch time (minutes)
            eta_total_min = (spend_time / (step + 1)) * iters / 60.0
            if current_lr_llm is not None:
                lr_msg = (
                    f'lr_speech_proj: {current_lr:.2e}, lr_llm_last: {current_lr_llm:.2e}'
                )
                wb_lr = {"lr_speech_proj": current_lr, "lr_llm_last": current_lr_llm}
            else:
                lr_msg = f'lr: {current_lr:.8f}'
                wb_lr = {"learning_rate": current_lr}
            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                f'loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, '
                f'aux_loss: {current_aux_loss:.4f}, {lr_msg}, epoch_total_time: {eta_total_min:.1f}min'
            )
            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "logits_loss": current_logits_loss,
                    "aux_loss": current_aux_loss,
                    **wb_lr,
                    "epoch_total_time": eta_total_min,
                })

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if olm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{olm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            clean_state_dict = {
                key: value for key, value in state_dict.items()
                if not key.startswith('vision_encoder.') and not key.startswith('speech_encoder.')
            }
            clean_state_dict = {k: v.half().cpu() for k, v in clean_state_dict.items()}
            torch.save(clean_state_dict, ckp)
            olm_checkpoint(
                olm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir='../checkpoints',
                scaler=scaler,
            )
            model.train()
            del state_dict, clean_state_dict

        del input_ids, labels, speech_values, speech_lengths, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-O Pretrain")
    parser.add_argument("--save_dir", type=str, default="../out", help="model save dir")
    parser.add_argument('--save_weight', default='pretrain_olm', type=str, help="save weight prefix")
    parser.add_argument('--weight', default='llm_768', type=str, help="initial weight prefix/path (default: out/llm_768.pth)")
    parser.add_argument("--epochs", type=int, default=4, help="epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument(
        "--lr_speech_proj",
        type=float,
        default=1e-4,
        help="speech_proj 学习率（随机初始化，通常较大）",
    )
    parser.add_argument(
        "--lr_llm_last",
        type=float,
        default=1e-5,
        help="最后一层 Transformer 学习率（预训练权重，通常较小）",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="未冻结整模时使用的单一学习率（--freeze_llm 0 时生效）",
    )
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="train device")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="mixed precision")
    parser.add_argument("--num_workers", type=int, default=4, help="num workers")
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=1,
        help="DataLoader 每个 worker 预取的 batch 数（仅 num_workers>0 时生效）。"
        "batch 较大时请保持为 1，否则预取队列会成倍占用内存并可能触发换页导致磁盘满、GPU 空转。",
    )
    parser.add_argument("--accumulation_steps", type=int, default=1, help="grad accumulation")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="grad clip")
    parser.add_argument("--log_interval", type=int, default=50, help="log interval")
    parser.add_argument("--save_interval", type=int, default=500, help="save interval")
    parser.add_argument('--max_seq_len', default=768, type=int, help="max seq len")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_s2t.parquet", help="train data path")
    parser.add_argument('--mode', default='speech', type=str, choices=['speech'], help="training mode (pretrain supports speech only)")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="resume training")
    parser.add_argument('--freeze_llm', default=1, type=int, choices=[0, 1], help="freeze llm")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="use torch.compile")
    parser.add_argument("--use_wandb", action="store_true", help="use wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-O-Pretrain", help="wandb project")
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    import re
    from typing import Optional

    def _maybe_extract_state_dict(obj) -> dict:
        if isinstance(obj, dict):
            if 'state_dict' in obj and isinstance(obj['state_dict'], dict):
                return obj['state_dict']
            if 'model' in obj and isinstance(obj['model'], dict):
                return obj['model']
            if any(torch.is_tensor(v) for v in obj.values()):
                return obj
        raise ValueError("无法从权重中提取 state_dict")

    def _load_weight_file(weight_path: Path) -> dict:
        ext = weight_path.suffix.lower()
        if ext == ".safetensors":
            try:
                from safetensors.torch import load_file
            except Exception as e:
                raise RuntimeError("加载 safetensors 权重需要安装 safetensors") from e
            return load_file(str(weight_path), device="cpu")
        raw = torch.load(str(weight_path), map_location='cpu')
        return _maybe_extract_state_dict(raw)

    def _resolve_initial_weight(save_dir: str, weight_arg: str) -> Path:
        sd = Path(save_dir)
        p = Path(weight_arg)
        # explicit file
        if p.suffix.lower() in {".pth", ".bin", ".safetensors"}:
            if p.exists():
                return p
        # prefix inside save_dir
        for ext in [".pth", ".bin", ".safetensors"]:
            cand = sd / f"{weight_arg}{ext}"
            if cand.exists():
                return cand
        # fallback: any file that starts with the prefix
        cands = [x for x in sd.iterdir() if x.is_file() and x.name.startswith(weight_arg) and x.suffix.lower() in {".pth", ".bin", ".safetensors"}]
        if cands:
            cands.sort(key=lambda x: (x.suffix.lower() not in {".pth", ".bin", ".safetensors"}, x.name))
            return sorted(cands)[0]
        raise FileNotFoundError(f"在 {save_dir} 中未找到初始权重: {weight_arg}(.pth/.bin/.safetensors)")

    def _infer_config_from_state_dict(state_dict: dict) -> dict:
        hidden_size: Optional[int] = None
        vocab_size: Optional[int] = None
        layer_indices = set()
        use_moe = False

        for k, v in state_dict.items():
            if not torch.is_tensor(v):
                continue
            lk = k.lower()
            if (k.endswith('embed_tokens.weight') or k.endswith('lm_head.weight')) and v.ndim == 2:
                vocab_size = int(v.shape[0])
                hidden_size = int(v.shape[1])
                break

        layer_re = re.compile(r'(?:^|\.)(?:model\.)?layers\.(\d+)\.')
        for k, v in state_dict.items():
            lk = k.lower()
            if ('mlp.experts.' in lk) or ('mlp.shared_experts.' in lk):
                use_moe = True
            m = layer_re.search(k)
            if m:
                layer_indices.add(int(m.group(1)))

        num_hidden_layers = (max(layer_indices) + 1) if layer_indices else None

        # infer n_routed_experts & n_shared_experts
        n_routed_experts = None
        n_shared_experts = None
        if hidden_size is not None:
            for k, v in state_dict.items():
                if not torch.is_tensor(v):
                    continue
                lk = k.lower()
                if use_moe and lk.endswith('mlp.gate.weight') and v.ndim == 2 and int(v.shape[1]) == hidden_size:
                    n_routed_experts = int(v.shape[0])
                    break
            shared_idxs = set()
            shared_re = re.compile(r'mlp\.shared_experts\.(\d+)\.')
            for k in state_dict.keys():
                m = shared_re.search(k)
                if m:
                    shared_idxs.add(int(m.group(1)))
            n_shared_experts = (max(shared_idxs) + 1) if shared_idxs else 0

        # infer num_attention_heads & num_key_value_heads from k_proj
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
                for h in range(1, hidden_size + 1):
                    if hidden_size % h != 0:
                        continue
                    nk = out_k * h / hidden_size
                    if abs(nk - round(nk)) < 1e-6:
                        nk_int = int(round(nk))
                        if nk_int >= 1 and h % nk_int == 0:
                            candidates.append((h, nk_int))
                pref = [8, 12, 16, 24, 32, 48, 64]
                candidates.sort(key=lambda x: (pref.index(x[0]) if x[0] in pref else 9999, x[0]))
                if candidates:
                    num_attention_heads, num_key_value_heads = candidates[0]

        return {
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "num_hidden_layers": num_hidden_layers,
            "use_moe": use_moe,
            "n_routed_experts": n_routed_experts,
            "n_shared_experts": n_shared_experts,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
        }

    os.makedirs(args.save_dir, exist_ok=True)
    initial_weight_path = _resolve_initial_weight(args.save_dir, args.weight)
    state_dict = _load_weight_file(initial_weight_path)
    inferred = _infer_config_from_state_dict(state_dict)

    if inferred["hidden_size"] is None or inferred["num_hidden_layers"] is None:
        raise RuntimeError(f"无法从权重推断 hidden_size/num_hidden_layers: {initial_weight_path}")

    olm_config = OLMConfig(
        hidden_size=int(inferred["hidden_size"]),
        num_hidden_layers=int(inferred["num_hidden_layers"]),
        max_seq_len=args.max_seq_len,
        use_moe=bool(inferred["use_moe"]),
        num_attention_heads=int(inferred["num_attention_heads"]),
        num_key_value_heads=int(inferred["num_key_value_heads"]),
        n_routed_experts=int(inferred["n_routed_experts"]) if inferred["n_routed_experts"] is not None else 4,
        n_shared_experts=int(inferred["n_shared_experts"]) if inferred["n_shared_experts"] is not None else 1,
        vocab_size=int(inferred["vocab_size"]) if inferred["vocab_size"] is not None else 6400,
    )
    ckp_data = olm_checkpoint(olm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume == 1 else None

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        if bool(args.freeze_llm):
            run_name = (
                f"MiniMind-O-Pretrain-E{args.epochs}-B{args.batch_size}"
                f"-lr_sp{args.lr_speech_proj}-lr_last{args.lr_llm_last}"
            )
        else:
            run_name = (
                f"MiniMind-O-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}"
                f"-LearningRate-{args.learning_rate}"
            )
        wandb.init(project=args.wandb_project, name=run_name, id=wandb_id, resume=resume)

    model, tokenizer, _ = init_olm_model(
        olm_config,
        from_weight=args.weight,
        device=args.device,
        freeze_llm=bool(args.freeze_llm),
        mode=args.mode,
    )
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')

    train_ds = PretrainDataset(
        args.data_path,
        tokenizer,
        speech_special_token=olm_config.speech_special_token,
        max_length=olm_config.max_seq_len,
    )
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    wd = 0.01
    if bool(args.freeze_llm):
        last_idx = olm_config.num_hidden_layers - 1
        speech_proj_params = [p for p in model.speech_proj.parameters() if p.requires_grad]
        last_layer_params = [
            p for n, p in model.model.named_parameters()
            if f"layers.{last_idx}." in n and p.requires_grad
        ]
        if not speech_proj_params or not last_layer_params:
            raise RuntimeError(
                f"分组优化器需要可训练的 speech_proj 与最后一层；"
                f"got speech_proj={len(speech_proj_params)}, last_layer={len(last_layer_params)}"
            )
        optimizer = optim.AdamW(
            [
                {"params": speech_proj_params, "lr": args.lr_speech_proj},
                {"params": last_layer_params, "lr": args.lr_llm_last},
            ],
            weight_decay=wd,
        )
        args._optimizer_grouped = True
        if is_main_process():
            Logger(
                f"Optimizer param groups: speech_proj lr={args.lr_speech_proj}, "
                f"layers.{last_idx} lr={args.lr_llm_last}, weight_decay={wd}"
            )
    else:
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.learning_rate,
            weight_decay=wd,
        )
        args._optimizer_grouped = False
        if is_main_process():
            Logger(f"Optimizer (single group): lr={args.learning_rate}, weight_decay={wd}")

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        dl_kwargs = dict(
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=pretrain_collate_fn,
        )
        if args.num_workers > 0:
            dl_kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))
            dl_kwargs["persistent_workers"] = True
        loader = DataLoader(train_ds, **dl_kwargs)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: skip first {start_step} steps, start from {start_step + 1}')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

    if dist.is_initialized():
        dist.destroy_process_group()
