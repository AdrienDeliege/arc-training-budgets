#!/usr/bin/env python3
"""Portable entry point for the paper's budget configurations (standard library only)."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "arc-agi-1"
BUDGETS = json.loads((ROOT / "configs" / "budgets.json").read_text())


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", choices=["varc", "trm"])
    p.add_argument("stage", choices=["pretrain", "ttt"])
    p.add_argument("--config", required=True, help="Pretrain: v51_40m, v51_130m, v1001_40m, v1001_130m. TTT: v51_low, v51_high, v1001_low, v1001_high.")
    p.add_argument("--task-id", help="One ARC-1 evaluation task. Omit to run all 400 sequentially.")
    p.add_argument("--task-start", type=int, default=0)
    p.add_argument("--task-count", type=int, default=400)
    p.add_argument("--checkpoint", type=Path, help="Pretrained checkpoint for TTT; omit for independent random initialization.")
    p.add_argument("--output", type=Path, help="Fresh run directory; defaults to outputs/<model>/<stage>_<config>_<scratch|pretrained>.")
    p.add_argument("--gpus", type=int, default=8, help="Processes for pretraining. TTT runs one task on one visible GPU.")
    p.add_argument("--dry-run", action="store_true", help="Print commands without creating data, logs or training outputs.")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--epochs", type=int, help="Diagnostic override; changes the paper budget.")
    p.add_argument("--seed", type=int, help="Override native default: VARC 42; TRM 0.")
    a = p.parse_args(argv)
    group = "pretraining" if a.stage == "pretrain" else "ttt"
    if a.config not in BUDGETS[group]:
        p.error(f"Choose --config from {', '.join(BUDGETS[group])}")
    if a.gpus < 1 or a.task_count < 1 or a.task_start < 0 or a.task_start >= 400:
        p.error("--gpus/--task-count must be positive; --task-start must be in 0..399")
    if a.epochs is not None and a.epochs < 1:
        p.error("--epochs must be positive")
    if a.stage == "pretrain" and (a.checkpoint or a.task_id):
        p.error("--checkpoint and --task-id apply to TTT")
    if a.checkpoint and not a.checkpoint.is_file():
        p.error(f"Checkpoint not found: {a.checkpoint}")
    a.output = (a.output or ROOT / "outputs" / a.model / f"{a.stage}_{a.config}_{'pretrained' if a.checkpoint else 'scratch'}").resolve()
    a.checkpoint = a.checkpoint.resolve() if a.checkpoint else None
    return a


def task_ids(a):
    ids = sorted(p.stem for p in (DATA / "data" / "evaluation").glob("*.json"))
    if a.task_id:
        if a.task_id not in ids:
            raise ValueError(f"Unknown ARC-1 evaluation task: {a.task_id}")
        return [a.task_id]
    return ids[a.task_start:a.task_start + a.task_count]


def build_command(a, task=None):
    c = BUDGETS["pretraining" if a.stage == "pretrain" else "ttt"][a.config]
    v = c["variants"]
    epochs = a.epochs or c[f"{a.model}_epochs"]
    if a.model == "trm":
        script = "trm_arc_pretrain.py" if a.stage == "pretrain" else "trm_arc_ttt_task.py"
        cmd = [sys.executable, str(ROOT / "harness" / script), "--trm-root", str(ROOT / "vendor" / "trm"), "--work-root", str(a.output), "--partition-tag", "run", "--num-aug", str(v - 1), "--epochs", str(epochs), "--submission-k", str(v)]
        if a.stage == "pretrain":
            interval = epochs if a.epochs else c["trm_eval_interval"]
            cmd += ["--eval-interval", str(interval), "--nproc-per-node", str(a.gpus), "--global-batch-size", "768", "--lr-warmup-steps", "2000", "--best-metric", "pass@1"]
        else:
            cmd += ["--task-id", task, "--global-batch-size", "64", "--lr-warmup-steps", str(c["trm_warmup"])]
            if a.checkpoint:
                cmd += ["--load-checkpoint", str(a.checkpoint)]
        if a.no_compile:
            cmd += ["--disable-compile"]
        if a.seed is not None:
            cmd += ["--seed", str(a.seed)]
        return cmd, ROOT
    cwd = ROOT / "vendor" / "varc"
    if a.stage == "pretrain":
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", str(a.gpus), "offline_train_ARC.py"]
    else:
        cmd = [sys.executable, "test_time_train_ARC.py"]
    cmd += ["--epochs", str(epochs), "--depth", "10", "--batch-size", "32" if a.stage == "pretrain" else "8", "--image-size", "64", "--patch-size", "2", "--learning-rate", "3e-4", "--weight-decay", "0", "--embed-dim", "512", "--num-heads", "8", "--num-colors", "12", "--data-root", str(DATA), "--architecture", "vit", "--lr-scheduler", "cosine"]
    if a.stage == "pretrain":
        cmd += ["--train-split", "training", "--eval-split", "training", "--eval-subset", "test", "--on-the-fly-augmentations", "--augmentation-schedule", "fixed", "--augmentation-task-id-mode", "per_variant", "--virtual-variants-per-epoch", str(v), "--eval-augmentations", str(v), "--geometry-set", "varc_basic", "--augmentation-seed", "12345", "--num-workers", "0", "--save-path", str(a.output / "checkpoint_final.pt"), "--best-save-path", str(a.output / "checkpoint_best.pt"), "--vis-every", "50", "--distributed"]
    else:
        # Historical scratch cells used materialized augmentation files;
        # pretrained cells used fixed on-the-fly variants.
        if a.checkpoint:
            split = f"evaluation/{task}"
            cmd += ["--resume-checkpoint", str(a.checkpoint), "--resume-skip-task-token", "--ttt-on-the-fly-augmentations", "--ttt-augmentation-schedule", "fixed", "--ttt-augmentation-task-id-mode", "per_variant", "--ttt-virtual-variants-per-epoch", str(v), "--ttt-eval-augmentations", str(v), "--ttt-geometry-set", "varc_basic"]
        else:
            split = f"eval_color_permute_ttt_{9 if v == 51 else 199}/{task}"
        cmd += ["--train-split", split, "--eval-split", split, "--eval-save-name", str(a.output / "predictions"), "--num-attempts", "10", "--ttt-num-each", "1"]
    if a.no_compile:
        cmd += ["--no-compile"]
    if a.seed is not None:
        cmd += ["--seed", str(a.seed)]
    return cmd, cwd


def main():
    a = parse_args()
    selected = task_ids(a) if a.stage == "ttt" else [None]
    plan = [build_command(a, task) for task in selected]
    for cmd, cwd in plan:
        print(f"cd {shlex.quote(str(cwd))}\n{shlex.join(cmd)}", flush=True)
    if a.dry_run:
        return
    if a.output.exists() and any(a.output.iterdir()):
        raise FileExistsError(f"Use a fresh --output directory: {a.output}")
    a.output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"
    env["TOKENIZERS_PARALLELISM"] = "false"
    manifest = {"model": a.model, "stage": a.stage, "config": a.config, "checkpoint": str(a.checkpoint) if a.checkpoint else None, "budget_override_epochs": a.epochs, "seed_override": a.seed, "commands": [{"argv": cmd, "cwd": str(cwd)} for cmd, cwd in plan]}
    (a.output / "launch.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for task, (cmd, cwd) in zip(selected, plan):
        if a.model == "varc" and a.stage == "ttt" and not a.checkpoint:
            subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_varc_task.py"), "--variants", str(BUDGETS["ttt"][a.config]["variants"]), "--task-id", task], check=True, env=env)
        subprocess.run(cmd, cwd=cwd, env=env, check=True)


if __name__ == "__main__":
    main()
