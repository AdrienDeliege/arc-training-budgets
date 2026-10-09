#!/usr/bin/env python3
"""Run released-TRM scratch TTT on one ARC task.

This script intentionally uses the upstream TRM data builder, model, training
loop, and ARC evaluator. It only orchestrates the one-task scratch-TTT protocol:

1. Select one ARC evaluation task.
2. Build a tiny TRM dataset containing that task only, with TRM augmentations.
3. Train TRM from random initialization on that one task's demo examples.
4. Evaluate on that one task's test examples.
5. Score the saved TRM submission.

The script runs one ARC task on one GPU and supports independent task shards.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_TRM_ROOT = PROJECT_ROOT / "vendor" / "trm"


def parse_int_list(raw: str) -> List[int]:
    values = [int(item) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run upstream TRM from scratch on one ARC task."
    )
    parser.add_argument("--trm-root", type=Path, default=DEFAULT_TRM_ROOT)
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path(os.environ.get("TRM_ARC_TTT_WORK_ROOT", "outputs/trm_arc_ttt")),
        help="Directory for generated one-task datasets, checkpoints, logs, and metrics.",
    )
    parser.add_argument(
        "--split",
        choices=("evaluation", "evaluation2"),
        default="evaluation",
        help="ARC split to use from TRM/kaggle/combined.",
    )
    parser.add_argument("--task-id", default=None, help="ARC task id, e.g. 00576224.")
    parser.add_argument(
        "--task-index",
        type=int,
        default=None,
        help="Zero-based index into sorted ARC task ids. Useful for task sharding.",
    )
    parser.add_argument("--num-aug", type=int, default=1000)
    parser.add_argument(
        "--epochs",
        type=int,
        default=100100,
        help=(
            "TRM epochs. For one task/group, 100100 roughly matches 100 sweeps over "
            "1001 augmented variants in sample-presentation budget."
        ),
    )
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--puzzle-emb-lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--puzzle-emb-weight-decay", type=float, default=0.1)
    parser.add_argument("--lr-warmup-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--submission-k", type=int, default=1001)
    parser.add_argument("--score-ks", default="1,2", help="Comma-separated pass@k values.")
    parser.add_argument("--partition-tag", default="arc1_eval")
    parser.add_argument("--project-name", default="TRM-ARC-TTT")
    parser.add_argument("--run-prefix", default="trm_arc_ttt")
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument(
        "--load-checkpoint",
        type=Path,
        default=None,
        help="Optional upstream TRM checkpoint to initialize from before task TTT.",
    )
    parser.add_argument(
        "--disable-compile",
        action="store_true",
        help="Set DISABLE_COMPILE=1 for upstream TRM. Useful for first smoke tests.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Delete and rebuild this task's generated dataset/checkpoint directory.",
    )
    parser.add_argument(
        "--skip-train-if-metrics-exist",
        action="store_true",
        help="Exit successfully if metrics.json already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and paths without executing dataset build or training.",
    )
    parser.add_argument(
        "--extra-hydra-arg",
        action="append",
        default=[],
        help="Additional raw Hydra override forwarded to upstream TRM pretrain.py.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r") as fh:
        return json.load(fh)


def write_json(path: Path, payload: Any, *, indent: Optional[int] = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=indent)


def command_to_str(command: Sequence[str]) -> str:
    return shlex.join(command)


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Dict[str, str],
    log_path: Path,
    dry_run: bool,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = command_to_str(command)
    print(rendered)
    with log_path.open("a") as log:
        log.write(f"\n$ {rendered}\n")
        log.flush()
        if dry_run:
            return
        process = subprocess.run(
            list(command),
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"Command failed with exit code {process.returncode}. See {log_path}"
            )


def challenge_solution_paths(trm_root: Path, split: str) -> tuple[Path, Path]:
    prefix = trm_root / "kaggle" / "combined" / f"arc-agi_{split}"
    return (
        prefix.with_name(prefix.name + "_challenges.json"),
        prefix.with_name(prefix.name + "_solutions.json"),
    )


def resolve_task_id(args: argparse.Namespace, challenges: Dict[str, Any]) -> str:
    if args.task_id is not None and args.task_index is not None:
        raise ValueError("Use only one of --task-id or --task-index.")
    task_ids = sorted(challenges)
    if args.task_id is not None:
        if args.task_id not in challenges:
            raise KeyError(f"Task id {args.task_id!r} not found in split {args.split}.")
        return args.task_id
    if args.task_index is None:
        raise ValueError("One of --task-id or --task-index is required.")
    if args.task_index < 0 or args.task_index >= len(task_ids):
        raise IndexError(
            f"task-index={args.task_index} outside 0..{len(task_ids) - 1}."
        )
    return task_ids[args.task_index]


def one_task_prefix(task_dir: Path, split: str) -> Path:
    return task_dir / "source" / "arc-agi"


def write_one_task_arc_files(
    *,
    prefix: Path,
    split: str,
    task_id: str,
    task: Dict[str, Any],
    solution: List[Any],
) -> None:
    challenges_path = Path(f"{prefix}_{split}_challenges.json")
    solutions_path = Path(f"{prefix}_{split}_solutions.json")
    write_json(challenges_path, {task_id: task})
    write_json(solutions_path, {task_id: solution})


def attach_test_solutions(task: Dict[str, Any], solution: List[Any]) -> Dict[str, Any]:
    task_with_solutions = json.loads(json.dumps(task))
    test_examples = task_with_solutions.get("test", [])
    if len(test_examples) != len(solution):
        raise ValueError(
            f"Challenge/solution count mismatch: {len(test_examples)} vs {len(solution)}"
        )
    for example, output in zip(test_examples, solution):
        example["output"] = output
    return task_with_solutions


def latest_submission_path(checkpoint_path: Path) -> Path:
    evaluator_dirs = sorted(
        checkpoint_path.glob("evaluator_ARC_step_*"),
        key=lambda path: int(re.search(r"step_(\d+)", path.name).group(1))  # type: ignore[union-attr]
        if re.search(r"step_(\d+)", path.name)
        else -1,
    )
    for evaluator_dir in reversed(evaluator_dirs):
        submission_path = evaluator_dir / "submission.json"
        if submission_path.exists():
            return submission_path
    raise FileNotFoundError(f"No TRM submission.json found under {checkpoint_path}")


def score_submission(
    *,
    task_id: str,
    task: Dict[str, Any],
    submission_path: Path,
    ks: Sequence[int],
) -> Dict[str, Any]:
    submission = load_json(submission_path)
    task_submission = submission.get(task_id)
    if task_submission is None:
        raise KeyError(f"Submission does not contain task {task_id}.")

    test_examples = task.get("test", [])
    if len(task_submission) != len(test_examples):
        raise ValueError(
            f"Submission/test count mismatch: {len(task_submission)} vs {len(test_examples)}"
        )

    correct_by_k = {str(k): 0 for k in ks}
    oracle_correct = 0
    examples = []
    for index, (prediction_entry, truth_pair) in enumerate(
        zip(task_submission, test_examples)
    ):
        ordered_attempts = [
            value
            for key, value in sorted(
                prediction_entry.items(),
                key=lambda item: int(item[0].split("_")[-1]),
            )
        ]
        truth = truth_pair["output"]
        per_example = {"index": index, "correct_by_k": {}, "oracle": False}
        for k in ks:
            ok = any(pred == truth for pred in ordered_attempts[:k])
            correct_by_k[str(k)] += int(ok)
            per_example["correct_by_k"][str(k)] = ok
        oracle = any(pred == truth for pred in ordered_attempts)
        oracle_correct += int(oracle)
        per_example["oracle"] = oracle
        examples.append(per_example)

    total = max(len(test_examples), 1)
    return {
        "task_id": task_id,
        "num_test_examples": len(test_examples),
        "num_saved_attempts": len(task_submission[0]) if task_submission else 0,
        "pass_at_k": {str(k): correct_by_k[str(k)] / total for k in ks},
        "oracle": oracle_correct / total,
        "correct_by_k": correct_by_k,
        "oracle_correct": oracle_correct,
        "submission_path": str(submission_path),
        "examples": examples,
    }


def main() -> None:
    args = parse_args()
    trm_root = args.trm_root.resolve()
    if not (trm_root / "pretrain.py").exists():
        raise FileNotFoundError(f"Could not find TRM pretrain.py under {trm_root}")

    challenges_path, solutions_path = challenge_solution_paths(trm_root, args.split)
    challenges = load_json(challenges_path)
    solutions = load_json(solutions_path)
    task_id = resolve_task_id(args, challenges)
    task = challenges[task_id]
    solution = solutions[task_id]
    task_with_solutions = attach_test_solutions(task, solution)

    task_dir = (args.work_root / args.partition_tag / task_id).resolve()
    dataset_dir = task_dir / "dataset_aug"
    checkpoint_path = task_dir / "checkpoints"
    metrics_path = task_dir / "metrics.json"
    manifest_path = task_dir / "manifest.json"
    build_log_path = task_dir / "logs" / "build_dataset.log"
    train_log_path = task_dir / "logs" / "train.log"

    if args.skip_train_if_metrics_exist and metrics_path.exists():
        print(f"metrics already exist: {metrics_path}")
        return
    if args.force_rebuild and task_dir.exists():
        shutil.rmtree(task_dir)

    prefix = one_task_prefix(task_dir, args.split)
    write_one_task_arc_files(
        prefix=prefix,
        split=args.split,
        task_id=task_id,
        task=task,
        solution=solution,
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(trm_root) + os.pathsep + env.get("PYTHONPATH", "")
    env["WANDB_MODE"] = args.wandb_mode
    if args.disable_compile:
        env["DISABLE_COMPILE"] = "1"

    eval_interval = args.eval_interval if args.eval_interval is not None else args.epochs
    if args.epochs % eval_interval != 0:
        raise ValueError("--eval-interval must divide --epochs.")

    build_command = [
        sys.executable,
        "-m",
        "dataset.build_arc_dataset",
        "--input-file-prefix",
        str(prefix),
        "--output-dir",
        str(dataset_dir),
        "--subsets",
        args.split,
        "--test-set-name",
        args.split,
        "--num-aug",
        str(args.num_aug),
        "--seed",
        str(args.seed),
    ]
    run_command(
        build_command,
        cwd=trm_root,
        env=env,
        log_path=build_log_path,
        dry_run=args.dry_run,
    )

    run_name = f"{args.run_prefix}_{args.split}_{task_id}"
    evaluator_override = f"evaluators=[{{name:arc@ARC,submission_K:{args.submission_k}}}]"
    train_command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        "1",
        "pretrain.py",
        "arch=trm",
        f"data_paths=[{dataset_dir}]",
        evaluator_override,
        f"epochs={args.epochs}",
        f"eval_interval={eval_interval}",
        "checkpoint_every_eval=True",
        f"global_batch_size={args.global_batch_size}",
        f"lr={args.lr}",
        f"puzzle_emb_lr={args.puzzle_emb_lr}",
        f"weight_decay={args.weight_decay}",
        f"puzzle_emb_weight_decay={args.puzzle_emb_weight_decay}",
        f"lr_warmup_steps={args.lr_warmup_steps}",
        f"seed={args.seed}",
        f"+project_name={args.project_name}",
        f"+run_name={run_name}",
        f"+checkpoint_path={checkpoint_path}",
    ]
    if args.load_checkpoint is not None:
        train_command.append(f"+load_checkpoint={args.load_checkpoint.resolve()}")
    train_command.extend(args.extra_hydra_arg)
    started_at = time.time()
    run_command(
        train_command,
        cwd=trm_root,
        env=env,
        log_path=train_log_path,
        dry_run=args.dry_run,
    )

    manifest = {
        "task_id": task_id,
        "task_index": args.task_index,
        "split": args.split,
        "num_aug": args.num_aug,
        "epochs": args.epochs,
        "eval_interval": eval_interval,
        "global_batch_size": args.global_batch_size,
        "lr": args.lr,
        "puzzle_emb_lr": args.puzzle_emb_lr,
        "weight_decay": args.weight_decay,
        "puzzle_emb_weight_decay": args.puzzle_emb_weight_decay,
        "lr_warmup_steps": args.lr_warmup_steps,
        "seed": args.seed,
        "dataset_dir": str(dataset_dir),
        "checkpoint_path": str(checkpoint_path),
        "build_log": str(build_log_path),
        "train_log": str(train_log_path),
        "train_command": train_command,
        "load_checkpoint": str(args.load_checkpoint.resolve()) if args.load_checkpoint else None,
        "elapsed_seconds": time.time() - started_at,
    }
    write_json(manifest_path, manifest)

    if args.dry_run:
        print(f"dry_run_manifest={manifest_path}")
        return

    submission_path = latest_submission_path(checkpoint_path)
    metrics = score_submission(
        task_id=task_id,
        task=task_with_solutions,
        submission_path=submission_path,
        ks=parse_int_list(args.score_ks),
    )
    metrics.update(
        {
            "manifest_path": str(manifest_path),
            "elapsed_seconds": manifest["elapsed_seconds"],
        }
    )
    write_json(metrics_path, metrics)
    print(json.dumps({k: metrics[k] for k in ("task_id", "pass_at_k", "oracle")}, indent=2))
    print(f"metrics={metrics_path}")


if __name__ == "__main__":
    main()
