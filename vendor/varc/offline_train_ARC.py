import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.amp import autocast
from utils.args import parse_args
from utils.distribution import init_distributed_mode
from utils.load_model import load_models
from utils.wandb_vis import grid_to_pil
import wandb
from src.ARC_loader import build_dataloaders, IGNORE_INDEX, PAD_INDEX

def _format_eta(seconds: float) -> str:
    total_seconds = int(max(seconds, 0))
    (hours, remainder) = divmod(total_seconds, 3600)
    (minutes, secs) = divmod(remainder, 60)
    return f'{hours:02d}h{minutes:02d}m{secs:02d}s'

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

def _require_finite_across_ranks(value: torch.Tensor, *, label: str, epoch: int, step: int, distributed: bool) -> None:
    finite = torch.isfinite(value.detach()).all().to(dtype=torch.int32)
    if distributed and dist.is_initialized():
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not bool(finite.item()):
        raise FloatingPointError(f'Non-finite {label} detected at epoch={epoch}, step={step}.')

def _save_offline_checkpoint(*, args: argparse.Namespace, model_for_eval: torch.nn.Module, optimizer: torch.optim.Optimizer, scaler, scheduler, epoch: int, tag: str) -> None:
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    elif args.save_path:
        checkpoint_dir = Path(args.save_path).parent
    else:
        checkpoint_dir = Path('outputs') / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f'{tag}_epoch_{epoch:04d}.pt'
    payload: Dict[str, Any] = {'model_state': model_for_eval.state_dict(), 'config': vars(args), 'epoch': epoch}
    if optimizer is not None:
        payload['optimizer_state'] = optimizer.state_dict()
    if scheduler is not None:
        payload['scheduler_state'] = scheduler.state_dict()
    if scaler is not None and scaler.is_enabled():
        payload['scaler_state'] = scaler.state_dict()
    torch.save(payload, checkpoint_path)
    print(f'Saved periodic checkpoint to {checkpoint_path}')

@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, *, distributed: bool=False, resolution_factor: int=1) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_pixels = 0
    total_exact = 0
    total_examples = 0
    visualizations = {}
    dataset = getattr(loader, 'dataset', None)
    dataset.disable_translation()
    dataset.disable_resolution_augmentation(fix_scale_factor=resolution_factor)
    for batch in loader:
        inputs = batch['inputs'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['targets'].to(device)
        task_ids = batch['task_ids'].to(device)
        offsets = batch['offset'].to(device)
        scale_factors = batch['scale_factors'].to(device)
        raw_outputs = batch['raw_outputs']
        model_kwargs = _model_kwargs_from_batch(batch, device)
        logits = model(inputs, task_ids, attention_mask=attention_mask, **model_kwargs)
        num_colors = logits.size(1)
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, num_colors)
        loss = F.cross_entropy(logits_flat, targets.view(-1), ignore_index=IGNORE_INDEX, reduction='sum')
        total_loss += loss.item()
        total_pixels += (targets != IGNORE_INDEX).sum().item()
        predictions = logits.argmax(dim=1)
        batch_size = predictions.size(0)
        for idx in range(batch_size):
            target = targets[idx]
            prediction = predictions[idx]
            valid = target != IGNORE_INDEX
            if valid.any():
                is_exact = bool(torch.equal(prediction[valid], target[valid]))
            else:
                is_exact = False
            total_exact += int(is_exact)
            total_examples += 1
            if inputs.dim() == 4:
                input_grid = inputs[idx, 0]
                mask = attention_mask[idx, 0]
            else:
                input_grid = inputs[idx]
                mask = attention_mask[idx]
            visualizations[task_ids[idx].item()] = grid_to_pil(mask, input_grid, target, prediction, IGNORE_INDEX=IGNORE_INDEX)
    if distributed and dist.is_initialized():
        totals = torch.tensor([total_loss, total_pixels, total_exact, total_examples], dtype=torch.float64, device=device)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        (total_loss, total_pixels, total_exact, total_examples) = totals.tolist()
    avg_loss = total_loss / max(total_pixels, 1)
    accuracy = total_exact / max(total_examples, 1)
    if not args.disable_translation:
        dataset.enable_translation()
    if not args.disable_resolution_augmentation:
        dataset.enable_resolution_augmentation()
    return (avg_loss, accuracy, visualizations)

def train(args: argparse.Namespace) -> None:
    (distributed, rank, world_size, local_rank, device) = init_distributed_mode(args)
    set_seed(args.seed + (rank if distributed else 0))
    (train_dataset, train_loader, eval_dataset, eval_loader, train_sampler, eval_sampler) = build_dataloaders(args, distributed=distributed, rank=rank, world_size=world_size)
    if args.disable_translation:
        train_dataset.disable_translation()
        if eval_dataset is not None:
            eval_dataset.disable_translation()
    else:
        train_dataset.enable_translation()
        if eval_dataset is not None:
            eval_dataset.enable_translation()
    if args.disable_resolution_augmentation:
        train_dataset.disable_resolution_augmentation(fix_scale_factor=args.fix_scale_factor)
        if eval_dataset is not None:
            eval_dataset.disable_resolution_augmentation(fix_scale_factor=args.fix_scale_factor)
    else:
        train_dataset.enable_resolution_augmentation()
        if eval_dataset is not None:
            eval_dataset.enable_resolution_augmentation()
    total_train_examples = len(train_dataset)
    if not distributed or rank == 0:
        print(f'Total training examples: {total_train_examples}')
    (model, model_for_eval, optimizer, scaler, scheduler, start_epoch) = load_models(args=args, train_dataset=train_dataset, device=device, distributed=distributed, rank=rank, local_rank=local_rank)
    autocast_device_type = device.type if device.type in {'cuda', 'cpu', 'mps'} else 'cuda'
    wandb_run = None
    is_main_process = not distributed or rank == 0
    if args.use_wandb and is_main_process:
        if wandb is None:
            raise RuntimeError('Weights & Biases is not installed. Install wandb or disable --use-wandb.')
        wandb_kwargs: Dict[str, Any] = {'project': args.wandb_project, 'config': dict(vars(args))}
        if args.wandb_run_name:
            wandb_kwargs['name'] = args.wandb_run_name
        wandb_run = wandb.init(**wandb_kwargs)
    best_eval_acc = float('-inf')
    global_start = time.time()
    previous_total_steps = 0
    canvas_views_seen_total = 0
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            running_loss = 0.0
            sample_count = 0
            canvas_view_count = 0
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
                canvas_view_count += batch_size
                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type=autocast_device_type, enabled=scaler.is_enabled()):
                    logits = model(inputs, task_ids, attention_mask=attention_mask, **model_kwargs)
                    loss = _pixel_ce_loss(logits, targets)
                _require_finite_across_ranks(loss, label='loss', epoch=epoch, step=step, distributed=distributed)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if not scaler.is_enabled():
                    _require_finite_across_ranks(grad_norm, label='gradient norm', epoch=epoch, step=step, distributed=distributed)
                scaler.step(optimizer)
                scaler.update()
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
            train_totals = torch.tensor([running_loss, sample_count, train_exact, train_examples, canvas_view_count], dtype=torch.float64, device=device)
            if distributed and dist.is_initialized():
                dist.all_reduce(train_totals, op=dist.ReduceOp.SUM)
            (running_loss_total, sample_count_total, train_exact_total, train_examples_total, canvas_view_count_total) = train_totals.tolist()
            canvas_views_seen_total += int(canvas_view_count_total)
            avg_train_loss = running_loss_total / max(sample_count_total, 1)
            train_acc = train_exact_total / max(train_examples_total, 1)
            if distributed and dist.is_initialized():
                pass
            if distributed and dist.is_initialized():
                pass
            total_elapsed = time.time() - global_start
            total_steps = len(train_loader) * args.epochs
            steps_completed = min(previous_total_steps, total_steps)
            remaining_steps = total_steps - steps_completed
            avg_time_per_step_global = total_elapsed / max(steps_completed, 1)
            total_eta = remaining_steps * avg_time_per_step_global
            log_parts = [f'epoch={epoch}', f'train_loss={avg_train_loss:.4f}', f'train_acc={train_acc:.4f}', f'canvas_views={int(canvas_view_count_total)}', f'epoch_time={epoch_duration:.1f}s', f'eta_total={_format_eta(total_eta)}']
            current_lr = optimizer.param_groups[0]['lr'] if optimizer.param_groups else args.learning_rate
            log_parts.append(f'lr={current_lr:.6f}')
            eval_loss = None
            eval_acc = None
            visualizations = {}
            eval_every = max(1, int(getattr(args, 'eval_every', 1)))
            run_eval = eval_loader is not None and epoch % eval_every == 0
            if run_eval:
                (eval_loss, eval_acc, visualizations) = evaluate(model, eval_loader, device, distributed=distributed, resolution_factor=args.fix_scale_factor if args.disable_resolution_augmentation else 2)
                if is_main_process:
                    log_parts.append(f'eval_loss={eval_loss:.4f}')
                    log_parts.append(f'eval_acc={eval_acc:.4f}')
                    if eval_acc > best_eval_acc:
                        best_eval_acc = eval_acc
                        if args.best_save_path:
                            best_path = Path(args.best_save_path)
                            best_path.parent.mkdir(parents=True, exist_ok=True)
                            model_to_save = model_for_eval
                            best_payload: Dict[str, Any] = {'model_state': model_to_save.state_dict(), 'config': vars(args), 'best_eval_accuracy': best_eval_acc, 'epoch': epoch}
                            if optimizer is not None:
                                best_payload['optimizer_state'] = optimizer.state_dict()
                            if scheduler is not None:
                                best_payload['scheduler_state'] = scheduler.state_dict()
                            if scaler.is_enabled():
                                best_payload['scaler_state'] = scaler.state_dict()
                            torch.save(best_payload, best_path)
            if is_main_process:
                print(' | '.join(log_parts))
            if wandb_run is not None and is_main_process:
                metrics = {'epoch': epoch, 'steps': previous_total_steps, 'train/loss': avg_train_loss, 'train/accuracy': train_acc, 'train/epoch_time': epoch_duration, 'train/lr': current_lr, 'train/canvas_views': int(canvas_view_count_total), 'train/canvas_views_seen': canvas_views_seen_total}
                if eval_loss is not None and eval_acc is not None:
                    metrics['eval/loss'] = eval_loss
                    metrics['eval/accuracy'] = eval_acc
                    if (epoch + 1) % args.vis_every == 0:
                        reverse_task_lookup = {v: k for (k, v) in eval_dataset.task_lookup.items()}
                        metrics['visualizations/eval'] = [wandb.Image(v, mode='RGBA', caption=f'task {reverse_task_lookup[k]}') for (k, v) in visualizations.items()]
                if best_eval_acc > float('-inf'):
                    metrics['eval/best_accuracy'] = best_eval_acc
                wandb.log(metrics, step=epoch)
            if scheduler is not None:
                scheduler.step()
            if is_main_process and args.checkpoint_every > 0 and (epoch > 0) and (epoch % args.checkpoint_every == 0):
                tag = Path(args.save_path).stem if args.save_path else 'offline_train'
                _save_offline_checkpoint(args=args, model_for_eval=model_for_eval, optimizer=optimizer, scaler=scaler, scheduler=scheduler, epoch=epoch, tag=tag)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    if args.save_path and is_main_process:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        final_payload = {'model_state': model_for_eval.state_dict(), 'config': vars(args)}
        if scaler.is_enabled():
            final_payload['scaler_state'] = scaler.state_dict()
        torch.save(final_payload, save_path)
    if distributed and dist.is_initialized():
        dist.destroy_process_group()
if __name__ == '__main__':
    args = parse_args()
    train(args)
