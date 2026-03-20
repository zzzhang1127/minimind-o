import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_olm import OLMConfig
from dataset.lm_dataset import OLMDataset
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
    for step, (input_ids, labels, pixel_values, speech_values, speech_lengths) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        pixel_values = pixel_values.to(args.device)
        speech_values = speech_values.to(args.device)
        speech_lengths = speech_lengths.to(args.device)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(
                input_ids,
                labels=labels,
                pixel_values=pixel_values,
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
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                f'loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, '
                f'aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min'
            )
            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "logits_loss": current_logits_loss,
                    "aux_loss": current_aux_loss,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min,
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

        del input_ids, labels, pixel_values, speech_values, speech_lengths, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-O SFT")
    parser.add_argument("--save_dir", type=str, default="../out", help="model save dir")
    parser.add_argument('--save_weight', default='sft_olm', type=str, help="save weight prefix")
    parser.add_argument("--epochs", type=int, default=2, help="epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="init learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="train device")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="mixed precision")
    parser.add_argument("--num_workers", type=int, default=4, help="num workers")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="grad accumulation")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="grad clip")
    parser.add_argument("--log_interval", type=int, default=50, help="log interval")
    parser.add_argument("--save_interval", type=int, default=500, help="save interval")
    parser.add_argument('--hidden_size', default=768, type=int, help="hidden size")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="num hidden layers")
    parser.add_argument('--max_seq_len', default=1536, type=int, help="max seq len")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="use moe")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_olm.parquet", help="train data path")
    parser.add_argument('--from_weight', default='pretrain_olm', type=str, help="load from which weight")
    parser.add_argument('--mode', default='both', type=str, choices=['speech', 'vision', 'both'], help="training mode")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="resume training")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="use torch.compile")
    parser.add_argument("--use_wandb", action="store_true", help="use wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-O-SFT", help="wandb project")
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    olm_config = OLMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len,
        use_moe=bool(args.use_moe),
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
        run_name = f"MiniMind-O-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=run_name, id=wandb_id, resume=resume)

    model, tokenizer, preprocess = init_olm_model(
        olm_config,
        from_weight=args.from_weight,
        device=args.device,
        freeze_llm=False,
        mode=args.mode,
    )
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')

    train_ds = OLMDataset(
        args.data_path,
        tokenizer,
        preprocess=preprocess,
        image_special_token=olm_config.image_special_token,
        speech_special_token=olm_config.speech_special_token,
        max_length=olm_config.max_seq_len,
    )
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

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
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: skip first {start_step} steps, start from {start_step + 1}')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

    if dist.is_initialized():
        dist.destroy_process_group()
