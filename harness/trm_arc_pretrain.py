#!/usr/bin/env python3
"""Inductive ARC-1 pretraining with released TRM code.

This script builds a TRM dataset from ARC-AGI-1 training tasks only:

- ARC training-task `train` examples become TRM train examples.
- ARC training-task `test` examples become TRM validation/eval examples.

It then launches upstream TRM pretraining and scores each saved validation
submission so we can keep both the final checkpoint and the best validation
checkpoint for later per-task TTT on unseen ARC eval tasks.
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
from typing import Any, Dict, Iterable, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_TRM_ROOT = PROJECT_ROOT / "vendor" / "trm"


def parse_int_list(raw: str) -> List[int]:
    values = [int(item) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain upstream TRM on ARC-1 train tasks.")
    parser.add_argument("--trm-root", type=Path, default=DEFAULT_TRM_ROOT)
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path(os.environ.get("TRM_ARC_PRETRAIN_WORK_ROOT", "outputs/trm_arc_pretrain")),
    )
    parser.add_argument("--split", choices=("training", "training2"), default="training")
    parser.add_argument("--partition-tag", default="arc1_train_inductive_pretrain")
    parser.add_argument("--num-aug", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=100000)
    parser.add_argument("--eval-interval", type=int, default=10000)
    parser.add_argument("--global-batch-size", type=int, default=768)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--puzzle-emb-lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--puzzle-emb-weight-decay", type=float, default=0.1)
    parser.add_argument("--lr-warmup-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--submission-k", type=int, default=1001)
    parser.add_argument("--score-ks", default="1,2")
    parser.add_argument(
        "--best-metric",
        choices=("pass@1", "pass@2", "oracle"),
        default="pass@1",
        help="Validation metric used to copy best_checkpoint.pt.",
    )
    parser.add_argument("--project-name", default="TRM-ARC-Inductive-Pretrain")
    parser.add_argument("--run-name", default="trm_arc1_train_inductive_pretrain")
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument("--load-checkpoint", type=Path, default=None)
    parser.add_argument("--disable-compile", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--skip-train-if-summary-exists", action="store_true")
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Skip build/train and score existing evaluator submissions/checkpoints.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra-hydra-arg", action="append", default=[])
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


def arc_prefix(trm_root: Path) -> Path:
    return trm_root / "kaggle" / "combined" / "arc-agi"


def challenge_solution_paths(trm_root: Path, split: str) -> tuple[Path, Path]:
    prefix = trm_root / "kaggle" / "combined" / f"arc-agi_{split}"
    return (
        prefix.with_name(prefix.name + "_challenges.json"),
        prefix.with_name(prefix.name + "_solutions.json"),
    )


def step_from_path(path: Path) -> int:
    match = re.search(r"step_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def evaluator_submission_paths(checkpoint_path: Path) -> List[Path]:
    return sorted(
        checkpoint_path.glob("evaluator_ARC_step_*/submission.json"),
        key=lambda path: step_from_path(path.parent),
    )


def checkpoint_for_step(checkpoint_path: Path, step: int) -> Path:
    path = checkpoint_path / f"step_{step}"
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint for validation step {step}: {path}")
    return path


def score_submission(
    *,
    submission_path: Path,
    challenges: Dict[str, Any],
    solutions: Dict[str, Any],
    ks: Sequence[int],
) -> Dict[str, Any]:
    submission = load_json(submission_path)
    correct_by_k = {str(k): 0 for k in ks}
    oracle_correct = 0
    total_examples = 0
    solved_tasks = {f"fully_pass@{k}": 0 for k in ks}
    solved_tasks.update({f"any_pass@{k}": 0 for k in ks})
    solved_tasks.update({"fully_oracle": 0, "any_oracle": 0})
    per_task: Dict[str, Any] = {}

    for task_id, task in sorted(challenges.items()):
        task_submission = submission.get(task_id)
        if task_submission is None:
            raise KeyError(f"Submission {submission_path} does not contain task {task_id}.")
        truths = solutions[task_id]
        if len(task_submission) != len(truths):
            raise ValueError(
                f"Submission/test count mismatch for {task_id}: "
                f"{len(task_submission)} vs {len(truths)}"
            )

        task_correct_by_k = {str(k): 0 for k in ks}
        task_oracle_correct = 0
        for prediction_entry, truth in zip(task_submission, truths):
            ordered_attempts = [
                value
                for key, value in sorted(
                    prediction_entry.items(),
                    key=lambda item: int(item[0].split("_")[-1]),
                )
            ]
            for k in ks:
                ok = any(pred == truth for pred in ordered_attempts[:k])
                task_correct_by_k[str(k)] += int(ok)
                correct_by_k[str(k)] += int(ok)
            oracle = any(pred == truth for pred in ordered_attempts)
            task_oracle_correct += int(oracle)
            oracle_correct += int(oracle)
            total_examples += 1

        num_task_examples = max(len(truths), 1)
        task_pass_at_k = {
            str(k): task_correct_by_k[str(k)] / num_task_examples
            for k in ks
        }
        task_oracle = task_oracle_correct / num_task_examples
        for k in ks:
            solved_tasks[f"fully_pass@{k}"] += int(task_pass_at_k[str(k)] >= 1.0)
            solved_tasks[f"any_pass@{k}"] += int(task_pass_at_k[str(k)] > 0.0)
        solved_tasks["fully_oracle"] += int(task_oracle >= 1.0)
        solved_tasks["any_oracle"] += int(task_oracle > 0.0)
        per_task[task_id] = {
            "num_test_examples": len(truths),
            "pass_at_k": task_pass_at_k,
            "oracle": task_oracle,
        }

    total = max(total_examples, 1)
    return {
        "submission_path": str(submission_path),
        "num_tasks": len(challenges),
        "num_test_examples": total_examples,
        "pass_at_k": {str(k): correct_by_k[str(k)] / total for k in ks},
        "oracle": oracle_correct / total,
        "task_counts": solved_tasks,
        "tasks": per_task,
    }


def metric_value(score: Dict[str, Any], metric: str) -> float:
    if metric.startswith("pass@"):
        return float(score["pass_at_k"][metric.split("@", 1)[1]])
    return float(score[metric])


def select_best_score(scores: List[Dict[str, Any]], best_metric: str) -> Dict[str, Any]:
    return max(
        scores,
        key=lambda score: (
            metric_value(score, best_metric),
            metric_value(score, "pass@2") if "2" in score["pass_at_k"] else 0.0,
            metric_value(score, "oracle"),
            score["step"],
        ),
    )


def latest_checkpoint(checkpoint_path: Path) -> Optional[Path]:
    checkpoints = sorted(
        [path for path in checkpoint_path.glob("step_*") if path.is_file()],
        key=step_from_path,
    )
    return checkpoints[-1] if checkpoints else None


def main() -> None:
    args = parse_args()
    trm_root = args.trm_root.resolve()
    if not (trm_root / "pretrain.py").exists():
        raise FileNotFoundError(f"Could not find TRM pretrain.py under {trm_root}")
    if args.epochs % args.eval_interval != 0:
        raise ValueError("--eval-interval must divide --epochs.")
    if args.global_batch_size % args.nproc_per_node != 0:
        raise ValueError("--global-batch-size must be divisible by --nproc-per-node.")
    if args.score_only and args.force_rebuild:
        raise ValueError("--score-only cannot be combined with --force-rebuild.")

    run_root = (args.work_root / args.partition_tag).resolve()
    dataset_dir = run_root / "dataset_aug"
    checkpoint_path = run_root / "checkpoints"
    build_log_path = run_root / "logs" / "build_dataset.log"
    train_log_path = run_root / "logs" / "train.log"
    manifest_path = run_root / "pretrain_manifest.json"
    validation_scores_path = run_root / "validation_scores.json"
    best_summary_path = run_root / "best_checkpoint.json"

    if args.skip_train_if_summary_exists and best_summary_path.exists():
        print(f"summary already exists: {best_summary_path}")
        return
    if args.force_rebuild and run_root.exists():
        shutil.rmtree(run_root)

    if not args.score_only:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(trm_root) + os.pathsep + env.get("PYTHONPATH", "")
        env["WANDB_MODE"] = args.wandb_mode
        if args.disable_compile:
            env["DISABLE_COMPILE"] = "1"

        build_command = [
            sys.executable,
            "-m",
            "dataset.build_arc_dataset",
            "--input-file-prefix",
            str(arc_prefix(trm_root)),
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

        evaluator_override = f"evaluators=[{{name:arc@ARC,submission_K:{args.submission_k}}}]"
        train_command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node",
            str(args.nproc_per_node),
            "pretrain.py",
            "arch=trm",
            f"data_paths=[{dataset_dir}]",
            evaluator_override,
            f"epochs={args.epochs}",
            f"eval_interval={args.eval_interval}",
            "checkpoint_every_eval=True",
            f"global_batch_size={args.global_batch_size}",
            f"lr={args.lr}",
            f"puzzle_emb_lr={args.puzzle_emb_lr}",
            f"weight_decay={args.weight_decay}",
            f"puzzle_emb_weight_decay={args.puzzle_emb_weight_decay}",
            f"lr_warmup_steps={args.lr_warmup_steps}",
            f"seed={args.seed}",
            f"+project_name={args.project_name}",
            f"+run_name={args.run_name}",
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
            "split": args.split,
            "num_aug": args.num_aug,
            "epochs": args.epochs,
            "eval_interval": args.eval_interval,
            "global_batch_size": args.global_batch_size,
            "nproc_per_node": args.nproc_per_node,
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

    challenges_path, solutions_path = challenge_solution_paths(trm_root, args.split)
    challenges = load_json(challenges_path)
    solutions = load_json(solutions_path)
    score_ks = parse_int_list(args.score_ks)

    scores: List[Dict[str, Any]] = []
    for submission_path in evaluator_submission_paths(checkpoint_path):
        step = step_from_path(submission_path.parent)
        score = score_submission(
            submission_path=submission_path,
            challenges=challenges,
            solutions=solutions,
            ks=score_ks,
        )
        score["step"] = step
        score["checkpoint_path"] = str(checkpoint_for_step(checkpoint_path, step))
        scores.append(score)

    if not scores:
        raise FileNotFoundError(f"No evaluator submissions found under {checkpoint_path}")

    best = select_best_score(scores, args.best_metric)
    best_checkpoint = Path(best["checkpoint_path"])
    best_copy_path = checkpoint_path / "best_checkpoint.pt"
    shutil.copy2(best_checkpoint, best_copy_path)

    final_checkpoint = latest_checkpoint(checkpoint_path)
    final_copy_path = None
    if final_checkpoint is not None:
        final_copy_path = checkpoint_path / "final_checkpoint.pt"
        shutil.copy2(final_checkpoint, final_copy_path)

    best_summary = {
        "best_metric": args.best_metric,
        "best_step": best["step"],
        "best_metric_value": metric_value(best, args.best_metric),
        "best_checkpoint_path": str(best_checkpoint),
        "best_checkpoint_copy_path": str(best_copy_path),
        "final_checkpoint_path": str(final_checkpoint) if final_checkpoint else None,
        "final_checkpoint_copy_path": str(final_copy_path) if final_copy_path else None,
        "validation_scores_path": str(validation_scores_path),
        "manifest_path": str(manifest_path),
        "best_score": {
            "pass_at_k": best["pass_at_k"],
            "oracle": best["oracle"],
            "task_counts": best["task_counts"],
            "num_test_examples": best["num_test_examples"],
        },
    }
    write_json(validation_scores_path, {"scores": scores})
    write_json(best_summary_path, best_summary)
    print(json.dumps(best_summary, indent=2))


if __name__ == "__main__":
    main()
