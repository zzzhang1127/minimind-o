"""
Training utility functions.
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from model.model_olm import MiniMindOLM


def get_model_params(model, config, ignore_patterns=None):
    if ignore_patterns is None:
        ignore_patterns = ['vision_encoder', 'speech_encoder']

    def should_count(n):
        return not any(p in n for p in ignore_patterns)

    total = sum(p.numel() for n, p in model.named_parameters() if should_count(n)) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n and should_count(n)) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n and should_count(n)) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total:
        Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else:
        Logger(f'Model Params: {total:.2f}M')


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_olm_model(
        olm_config,
        from_weight='llm_768',
        tokenizer_path='../model',
        vision_model_path='../model/vision_model/clip-vit-base-patch16',
        save_dir='../out',
        device='cuda',
        freeze_llm=False,
        train_modality='speech',
):
    train_modality = str(train_modality).lower()
    if train_modality not in {'speech', 'vision', 'both'}:
        raise ValueError(f'Invalid train_modality: {train_modality}. Choose from speech, vision, both.')

    load_vision_encoder = train_modality in {'vision', 'both'}
    load_speech_encoder = train_modality in {'speech', 'both'}

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindOLM(
        olm_config,
        vision_model_path=vision_model_path,
        load_vision_encoder=load_vision_encoder,
        load_speech_encoder=load_speech_encoder,
    )

    if from_weight != 'none':
        moe_suffix = '_moe' if olm_config.use_moe else ''
        # Try standard naming first: {from_weight}_{hidden_size}.pth
        weight_path = f'{save_dir}/{from_weight}_{olm_config.hidden_size}{moe_suffix}.pth'
        # If not found, try direct naming: {from_weight}.pth
        if not os.path.exists(weight_path):
            weight_path = f'{save_dir}/{from_weight}.pth'
        
        if os.path.exists(weight_path):
            weights = torch.load(weight_path, map_location=device)
            # Remove vision_encoder and speech_encoder from state_dict (will be reinitialized)
            cleaned_weights = {k: v for k, v in weights.items() 
                               if not k.startswith('vision_encoder.') and not k.startswith('speech_encoder.')}
            model.load_state_dict(cleaned_weights, strict=False)
            Logger(f'Loaded weights from: {weight_path}')
        else:
            Logger(f'Weight not found, skip loading. Searched paths: {weight_path}')

    if freeze_llm:
        for name, param in model.named_parameters():
            if 'vision_proj' not in name and 'speech_proj' not in name:
                param.requires_grad = False

    # Keep the inactive modality projector untouched during single-modality training.
    if train_modality == 'speech':
        for param in model.vision_proj.parameters():
            param.requires_grad = False
    elif train_modality == 'vision':
        for param in model.speech_proj.parameters():
            param.requires_grad = False

    last_layer_idx = olm_config.num_hidden_layers - 1
    for name, param in model.model.named_parameters():
        if f'layers.{last_layer_idx}.' in name:
            param.requires_grad = True

    get_model_params(model, olm_config)
    Logger(f'Train Modality: {train_modality} (vision_encoder={load_vision_encoder}, speech_encoder={load_speech_encoder})')
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    preprocess = model.processor
    return model.to(device), tokenizer, preprocess


def olm_checkpoint(
        olm_config,
        weight='pretrain_olm',
        model=None,
        optimizer=None,
        epoch=0,
        step=0,
        wandb=None,
        save_dir='../checkpoint',
        **kwargs,
):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if olm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{olm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{olm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        clean_state_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith('vision_encoder.') and not k.startswith('speech_encoder.')
        }
        ckp_tmp = ckp_path + '.tmp'
        torch.save({k: v.half().cpu() for k, v in clean_state_dict.items()}, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)

        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id,
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, clean_state_dict, resume_data
        torch.cuda.empty_cache()
    else:
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'World size changed ({saved_ws}->{current_ws}), step converted to {ckp_data["step"]}')
            return ckp_data
        return None


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)
