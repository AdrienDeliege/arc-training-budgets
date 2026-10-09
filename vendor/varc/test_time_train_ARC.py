import argparse
from copy import deepcopy
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import autocast
from torch.utils.data import DataLoader
from utils.args import parse_args
from utils.distribution import init_distributed_mode
from utils.load_model import load_model_only, load_optimizer
from src.ARC_loader import ARCDataset, build_dataloaders, collate_fn, IGNORE_INDEX
from utils.eval_utils_ttt import generate_predictions, get_eval_rot_transform_resolver
try:
    import wandb
except ImportError:
    wandb = None

def _format_eta(seconds: float) -> str:
    total_seconds = int(max(seconds, 0))
    (hours, remainder) = divmod(total_seconds, 3600)
    (minutes, secs) = divmod(remainder, 60)
    return f'{hours:02d}h{minutes:02d}m{secs:02d}s'

def _write_timing_summary(save_name: str, timing: Dict[str, Any]) -> None:
    output_dir = Path('outputs') / save_name
    output_dir.mkdir(parents=True, exist_ok=True)
    timing_path = output_dir / 'timing.json'
    with timing_path.open('w') as fh:
        json.dump(timing, fh, indent=2, sort_keys=True)
    print(f'Timing summary written to {timing_path}')

def _unwrap_model_for_state_dict(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module behind DDP and torch.compile wrappers."""
    unwrapped = model
    seen_ids = set()
    while id(unwrapped) not in seen_ids:
        seen_ids.add(id(unwrapped))
        if hasattr(unwrapped, 'module'):
            unwrapped = unwrapped.module
            continue
        if hasattr(unwrapped, '_orig_mod'):
            unwrapped = unwrapped._orig_mod
            continue
        break
    return unwrapped

def _resolve_ttt_model_save_path(args: argparse.Namespace, save_name: str, attempt_idx: int) -> Path:
    if args.ttt_model_save_path:
        path = Path(args.ttt_model_save_path)
        if args.ttt_num_each > 1:
            return path.with_name(f'{path.stem}_attempt_{attempt_idx}{path.suffix}')
        return path
    return Path('outputs') / save_name / 'final_ttt_model.pt'

def _save_ttt_model_checkpoint(*, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scaler, scheduler, train_loader, args: argparse.Namespace, save_name: str, attempt_idx: int) -> None:
    save_path = _resolve_ttt_model_save_path(args, save_name, attempt_idx)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    base_model = _unwrap_model_for_state_dict(model)
    train_dataset = getattr(train_loader, 'dataset', None)
    payload: Dict[str, Any] = {'checkpoint_type': 'ttt_final', 'model_state': base_model.state_dict(), 'config': dict(vars(args)), 'attempt': attempt_idx, 'configured_epochs': args.epochs, 'epoch_loop_count': args.epochs + 1}
    task_lookup = getattr(train_dataset, 'task_lookup', None)
    if task_lookup is not None:
        payload['task_lookup'] = dict(task_lookup)
    if optimizer is not None:
        payload['optimizer_state'] = optimizer.state_dict()
    if scheduler is not None:
        payload['scheduler_state'] = scheduler.state_dict()
    if scaler is not None and scaler.is_enabled():
        payload['scaler_state'] = scaler.state_dict()
    torch.save(payload, save_path)
    print(f'Saved final TTT model checkpoint to {save_path}')

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _pixel_ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    num_colors = logits.size(1)
    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, num_colors)
    return F.cross_entropy(logits_flat, targets.view(-1), ignore_index=IGNORE_INDEX)

def _model_kwargs_from_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    return {}

def ttt_once(model, device, distributed, rank, train_loader, train_sampler, eval_loader, cur_attempt_idx, wandb_run=None):
    autocast_device_type = device.type if device.type in {'cuda', 'cpu', 'mps'} else 'cuda'
    is_main_process = not distributed or rank == 0
    global_start = time.time()
    training_start = time.time()
    wandb_step_base = cur_attempt_idx * (args.epochs + 2)
    train_batch_visits_total = 0
    train_examples_seen_total = 0
    optimizer_steps_total = 0
    previous_total_steps = 0
    (optimizer, scaler, scheduler) = load_optimizer(model=model, args=args, device=device, distributed=distributed, rank=rank)
    try:
        for epoch in range(0, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            running_loss = 0.0
            sample_count = 0
            total_batches = len(train_loader)
            epoch_start = time.time()
            train_exact = 0
            train_examples = 0
            for (step, batch) in enumerate(train_loader, 1):
                inputs = batch['inputs'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['targets'].to(device)
                task_ids = batch['task_ids'].to(device)
                model_kwargs = _model_kwargs_from_batch(batch, device)
                batch_size = inputs.size(0)
                train_batch_visits_total += 1
                train_examples_seen_total += batch_size
                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type=autocast_device_type, enabled=scaler.is_enabled()):
                    logits = model(inputs, task_ids, attention_mask=attention_mask, **model_kwargs)
                    loss = _pixel_ce_loss(logits, targets)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer_steps_total += 1
                running_loss += loss.item() * batch_size
                sample_count += batch_size
                predictions = logits.detach().argmax(dim=1)
                for idx in range(batch_size):
                    target = targets[idx]
                    prediction = predictions[idx]
                    valid = target != IGNORE_INDEX
                    if valid.any():
                        is_exact = bool(torch.equal(prediction[valid], target[valid]))
                    else:
                        is_exact = False
                    train_exact += int(is_exact)
                    train_examples += 1
                if total_batches > 0 and is_main_process and (step % 10 == 0):
                    elapsed = time.time() - epoch_start
                    avg_step_time = elapsed / step
                    steps_completed = previous_total_steps + step
                    total_steps = len(train_loader) * args.epochs
                    remaining_steps = total_steps - steps_completed
                    elapsed_global = time.time() - global_start
                    avg_time_per_step_global = elapsed_global / max(steps_completed, 1)
                    eta = remaining_steps * avg_time_per_step_global
                    bar_length = 30
                    progress_ratio = steps_completed / total_steps if total_steps else 0
                    filled = int(bar_length * progress_ratio)
                    bar = '#' * filled + '-' * (bar_length - filled)
                    progress = 100.0 * progress_ratio
                    sys.stdout.write(f'\rEpoch {epoch} [{bar}] {progress:5.1f}% ETA {_format_eta(eta)}')
                    sys.stdout.flush()
            if total_batches > 0 and is_main_process:
                sys.stdout.write('\n')
            previous_total_steps += total_batches
            epoch_duration = time.time() - epoch_start if total_batches > 0 else 0.0
            train_totals = torch.tensor([running_loss, sample_count, train_exact, train_examples], dtype=torch.float64, device=device)
            if distributed and dist.is_initialized():
                dist.all_reduce(train_totals, op=dist.ReduceOp.SUM)
            (running_loss_total, sample_count_total, train_exact_total, train_examples_total) = train_totals.tolist()
            avg_train_loss = running_loss_total / max(sample_count_total, 1)
            train_acc = train_exact_total / max(train_examples_total, 1)
            total_elapsed = time.time() - global_start
            total_steps = len(train_loader) * args.epochs
            steps_completed = min(previous_total_steps, total_steps)
            remaining_steps = total_steps - steps_completed
            avg_time_per_step_global = total_elapsed / max(steps_completed, 1)
            total_eta = remaining_steps * avg_time_per_step_global
            log_parts = [f'epoch={epoch}', f'train_loss={avg_train_loss:.4f}', f'train_acc={train_acc:.4f}', f'epoch_time={epoch_duration:.1f}s', f'eta_total={_format_eta(total_eta)}']
            current_lr = optimizer.param_groups[0]['lr'] if optimizer.param_groups else args.learning_rate
            log_parts.append(f'lr={current_lr:.6f}')
            if is_main_process:
                print(' | '.join(log_parts))
            if wandb_run is not None and is_main_process:
                metrics = {'epoch': epoch, 'attempt': cur_attempt_idx, 'train/loss': avg_train_loss, 'train/exact_accuracy': train_acc, 'train/epoch_time': epoch_duration, 'train/lr': current_lr, 'train/batch_visits': train_batch_visits_total, 'train/examples_seen': train_examples_seen_total, 'train/optimizer_steps': optimizer_steps_total}
                wandb_run.log(metrics, step=wandb_step_base + epoch)
            if scheduler is not None:
                scheduler.step()
    finally:
        if distributed and dist.is_initialized():
            dist.barrier()
    if distributed and dist.is_initialized():
        dist.destroy_process_group()
    training_seconds = time.time() - training_start
    if is_main_process:
        print(f'TTT training timing | seconds={training_seconds:.1f} | duration={_format_eta(training_seconds)} | batch_visits={train_batch_visits_total} | examples_seen={train_examples_seen_total} | optimizer_steps={optimizer_steps_total}')
    save_name = args.eval_save_name + '_attempt_' + str(cur_attempt_idx)
    if args.save_ttt_model and is_main_process:
        _save_ttt_model_checkpoint(model=model, optimizer=optimizer, scaler=scaler, scheduler=scheduler, train_loader=train_loader, args=args, save_name=save_name, attempt_idx=cur_attempt_idx)
    output_img_size = getattr(args, 'output_image_size', None) or args.image_size
    inference_start = time.time()
    eval_metrics = generate_predictions(model, eval_loader, device, img_size=output_img_size, attempt_nums=args.num_attempts, task_transform_resolver=get_eval_rot_transform_resolver(), fix_scale_factor=args.fix_scale_factor, disable_translation=args.disable_translation, if_fix_scale=args.disable_resolution_augmentation, save_name=save_name, eval_split=args.eval_split, task_type=args.data_root.split('/')[-1], score_subset='test')
    inference_seconds = time.time() - inference_start
    total_seconds = time.time() - global_start
    if is_main_process:
        eval_examples = len(getattr(eval_loader, 'dataset', [])) if eval_loader is not None else 0
        effective_num_attempts = 1 if args.disable_translation else args.num_attempts
        train_examples_per_second = train_examples_seen_total / training_seconds if training_seconds > 0 else 0.0
        inference_forward_items = eval_examples * effective_num_attempts
        inference_items_per_second = inference_forward_items / inference_seconds if inference_seconds > 0 else 0.0
        print(f'TTT inference timing | seconds={inference_seconds:.1f} | duration={_format_eta(inference_seconds)} | eval_examples={eval_examples} | attempts={effective_num_attempts} | forward_items={inference_forward_items}')
        print(f'TTT total timing | seconds={total_seconds:.1f} | duration={_format_eta(total_seconds)}')
        if wandb_run is not None:
            timing_metrics = {'attempt': cur_attempt_idx, 'timing/training_seconds': training_seconds, 'timing/inference_seconds': inference_seconds, 'timing/total_seconds': total_seconds, 'timing/train_examples_per_second': train_examples_per_second, 'timing/inference_items_per_second': inference_items_per_second, 'timing/inference_forward_items': inference_forward_items}
            if eval_metrics is not None:
                timing_metrics.update({'eval/oracle': eval_metrics['oracle'], 'eval/pass_at_1': eval_metrics['pass_at_1'], 'eval/pass_at_2': eval_metrics['pass_at_2']})
            wandb_run.log(timing_metrics, step=wandb_step_base + args.epochs + 1)
        timing_summary = {'architecture': 'vit', 'data_root': args.data_root, 'train_split': args.train_split, 'eval_split': args.eval_split, 'save_name': save_name, 'configured_epochs': args.epochs, 'epoch_loop_count': args.epochs + 1, 'batch_size': args.batch_size, 'train_examples_per_epoch': len(train_loader.dataset), 'eval_examples': eval_examples, 'num_attempts': args.num_attempts, 'effective_num_attempts': effective_num_attempts, 'ttt_num_each': args.ttt_num_each, 'image_size': args.image_size, 'patch_size': args.patch_size, 'output_image_size': output_img_size, 'output_patch_size': getattr(args, 'output_patch_size', None) or args.patch_size, 'supervision_steps': 1, 'eval_supervision_steps': 1, 'training_seconds': training_seconds, 'training_duration': _format_eta(training_seconds), 'train_batch_visits': train_batch_visits_total, 'train_examples_seen': train_examples_seen_total, 'optimizer_steps': optimizer_steps_total, 'train_examples_per_second': train_examples_per_second, 'inference_seconds': inference_seconds, 'inference_duration': _format_eta(inference_seconds), 'inference_forward_items': inference_forward_items, 'inference_items_per_second': inference_items_per_second, 'eval_oracle': eval_metrics['oracle'] if eval_metrics is not None else None, 'eval_pass_at_1': eval_metrics['pass_at_1'] if eval_metrics is not None else None, 'eval_pass_at_2': eval_metrics['pass_at_2'] if eval_metrics is not None else None, 'total_seconds': total_seconds, 'total_duration': _format_eta(total_seconds)}
        _write_timing_summary(save_name, timing_summary)

def train(args: argparse.Namespace) -> None:
    (distributed, rank, world_size, local_rank, device) = init_distributed_mode(args)
    set_seed(args.seed + (rank if distributed else 0))
    (train_dataset, train_loader, eval_dataset, eval_loader, train_sampler, eval_sampler) = build_dataloaders(args, distributed=distributed, rank=rank, world_size=world_size)
    total_train_examples = len(train_dataset)
    if not distributed or rank == 0:
        print(f'Total training examples: {total_train_examples}')
    model_original = load_model_only(args=args, train_dataset=train_dataset, device=device, distributed=distributed, rank=rank, local_rank=local_rank)
    is_main_process = not distributed or rank == 0
    wandb_run = None
    if args.use_wandb and is_main_process:
        if wandb is None:
            raise RuntimeError('Weights & Biases is not installed. Install wandb or disable --use-wandb.')
        wandb_kwargs: Dict[str, Any] = {'project': args.wandb_project, 'config': dict(vars(args))}
        if args.wandb_run_name:
            wandb_kwargs['name'] = args.wandb_run_name
        wandb_run = wandb.init(**wandb_kwargs)
    try:
        for attempt_idx in range(args.ttt_num_each):
            model = deepcopy(model_original)
            print(f'Starting test-time training attempt {attempt_idx + 1}/{args.ttt_num_each}...')
            ttt_once(model=model, device=device, distributed=distributed, rank=rank, train_loader=train_loader, train_sampler=train_sampler, eval_loader=eval_loader, cur_attempt_idx=attempt_idx, wandb_run=wandb_run)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
if __name__ == '__main__':
    args = parse_args()
    train(args)
