import os
from typing import Any, Dict, Optional
import torch
from src.ARC_ViT import ARCViT
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler
from utils.lr_scheduler import get_cosine_schedule_with_warmup

def count_parameters(model):
    ret = 0
    for (name, p) in model.named_parameters():
        if p.requires_grad and 'task_token' not in name:
            ret += p.numel()
    return ret

def get_model_arch(args, train_dataset):
    return ARCViT(num_tasks=train_dataset.num_tasks, image_size=args.image_size, num_colors=args.num_colors, embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads, mlp_dim=args.mlp_dim, dropout=args.dropout, num_task_tokens=args.num_task_tokens, patch_size=args.patch_size)

def wrap_distributed_model(args, model, device, local_rank):
    ddp_kwargs = {'device_ids': [local_rank] if device.type == 'cuda' else None, 'output_device': local_rank if device.type == 'cuda' else None}
    return DDP(model, **ddp_kwargs)

def _compile_supported(args) -> bool:
    return not args.no_compile and hasattr(torch, 'compile')

def build_optimizer(args, model, distributed=False, rank=0):
    betas = (args.adam_beta1, args.adam_beta2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=betas)
    if not distributed or rank == 0:
        print(f'Optimizer profile: varc (lr={args.learning_rate:g}, wd={args.weight_decay:g}, betas={betas})')
    return optimizer

def build_scheduler(args, optimizer):
    if args.lr_scheduler == 'cosine':
        return get_cosine_schedule_with_warmup(optimizer, min(args.epochs // 5, 10), args.epochs, min_lr_ratio=args.lr_min_ratio)
    return None

def _load_model_state_dict_compat(args, model, state_dict):
    state_dict = {key.replace('_orig_mod.', '', 1): value for (key, value) in state_dict.items()}
    if args.resume_skip_task_token and 'task_token_embed.weight' in state_dict:
        state_dict = {k: v for (k, v) in state_dict.items() if k != 'task_token_embed.weight'}
    if args.resume_skip_task_token or False or False or False:
        (missing, unexpected) = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f'Skipped loading parameters: {sorted(missing)}')
        if unexpected:
            print(f'Unexpected parameters ignored from checkpoint: {sorted(unexpected)}')
        return
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        checkpoint_weight = state_dict.get('task_token_embed.weight')
        current_weight = model.task_token_embed.weight
        if checkpoint_weight is not None and checkpoint_weight.shape != current_weight.shape and (not args.resume_skip_task_token):
            raise RuntimeError('Mismatch in task_token_embed.weight shape. Re-run with --resume-skip-task-token to reuse other weights.') from exc
        raise

def load_models(args, train_dataset, device, distributed, rank, local_rank):
    resume_checkpoint = getattr(args, 'resume_checkpoint', None)
    resume_reset_epoch = bool(getattr(args, 'resume_reset_epoch', False))
    start_epoch = 1
    checkpoint: Optional[Dict[str, Any]] = None
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f'Resuming from checkpoint: {resume_checkpoint}')
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model = get_model_arch(args, train_dataset)
        _load_model_state_dict_compat(args, model, checkpoint.get('model_state', {}))
        model.to(device)
        if _compile_supported(args):
            if not distributed or rank == 0:
                print('Applying torch.compile for optimization...')
            model = torch.compile(model, mode=args.compile_mode)
        if distributed:
            model = wrap_distributed_model(args, model, device, local_rank)
        model_for_eval = model.module if distributed else model
        optimizer = build_optimizer(args, model, distributed=distributed, rank=rank)
        if 'epoch' in checkpoint and (not resume_reset_epoch):
            start_epoch = checkpoint['epoch'] + 1
        elif 'epoch' in checkpoint and resume_reset_epoch and (not distributed or rank == 0):
            print('Ignoring checkpoint epoch due to --resume-reset-epoch; restarting from epoch 1.')
    else:
        model = get_model_arch(args, train_dataset)
        print(f'Parameter count: {count_parameters(model) / 1000000:.2f}M (excluding task tokens)')
        model.to(device)
        if _compile_supported(args):
            if not distributed or rank == 0:
                print('Applying torch.compile for optimization...')
            model = torch.compile(model, mode=args.compile_mode)
        if distributed:
            model = wrap_distributed_model(args, model, device, local_rank)
        model_for_eval = model.module if distributed else model
        optimizer = build_optimizer(args, model, distributed=distributed, rank=rank)
    scaler = GradScaler(enabled=device.type == 'cuda' and (not args.no_amp))
    if not distributed or rank == 0:
        if scaler.is_enabled():
            print('Using automatic mixed precision (AMP) training')
        else:
            print('AMP disabled (CPU or --no-amp flag)')
    scheduler = build_scheduler(args, optimizer)
    resume_reset_optimizer = bool(getattr(args, 'resume_reset_optimizer', False))
    if checkpoint is not None and (not resume_reset_optimizer):
        if 'optimizer_state' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            except Exception as exc:
                raise RuntimeError('Failed to restore optimizer state from checkpoint. Use --resume-reset-optimizer to resume weights/epoch with a fresh optimizer.') from exc
        if scheduler is not None and 'scheduler_state' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state'])
            except Exception as exc:
                raise RuntimeError('Failed to restore scheduler state from checkpoint. Use --resume-reset-optimizer to resume weights/epoch with a fresh scheduler.') from exc
        if scaler.is_enabled() and 'scaler_state' in checkpoint:
            try:
                scaler.load_state_dict(checkpoint['scaler_state'])
            except Exception as exc:
                raise RuntimeError('Failed to restore AMP scaler state from checkpoint. Use --resume-reset-optimizer to resume weights/epoch with a fresh scaler.') from exc
        if not distributed or rank == 0:
            print('Restored optimizer/scheduler/scaler states from checkpoint.')
    elif checkpoint is not None and resume_reset_optimizer and (not distributed or rank == 0):
        print('Ignoring optimizer/scheduler/scaler states due to --resume-reset-optimizer.')
    return (model, model_for_eval, optimizer, scaler, scheduler, start_epoch)

def load_optimizer(args, model, device, distributed, rank):
    optimizer = build_optimizer(args, model, distributed=distributed, rank=rank)
    scaler = GradScaler(enabled=device.type == 'cuda' and (not args.no_amp))
    if not distributed or rank == 0:
        if scaler.is_enabled():
            print('Using automatic mixed precision (AMP) training')
        else:
            print('AMP disabled (CPU or --no-amp flag)')
    scheduler = build_scheduler(args, optimizer)
    return (optimizer, scaler, scheduler)

def load_model_only(args, train_dataset, device, distributed, rank, local_rank):
    resume_checkpoint = getattr(args, 'resume_checkpoint', None)
    if resume_checkpoint:
        if not os.path.exists(resume_checkpoint):
            raise FileNotFoundError(f'Resume checkpoint not found: {resume_checkpoint}')
        print(f'Resuming from checkpoint: {resume_checkpoint}')
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model = get_model_arch(args, train_dataset)
        _load_model_state_dict_compat(args, model, checkpoint.get('model_state', {}))
        model.to(device)
        if _compile_supported(args):
            if not distributed or rank == 0:
                print('Applying torch.compile for optimization...')
            model = torch.compile(model, mode=args.compile_mode)
        if distributed:
            model = wrap_distributed_model(args, model, device, local_rank)
    else:
        if not distributed or rank == 0:
            print('No checkpoint provided; running test-time training from random initialization.')
        model = get_model_arch(args, train_dataset)
        if not distributed or rank == 0:
            print(f'Parameter count: {count_parameters(model) / 1000000:.2f}M (excluding task tokens)')
        model.to(device)
        if _compile_supported(args):
            if not distributed or rank == 0:
                print('Applying torch.compile for optimization...')
            model = torch.compile(model, mode=args.compile_mode)
        if distributed:
            model = wrap_distributed_model(args, model, device, local_rank)
    return model
