#!/usr/bin/env python3
"""Collect per-task TRM scratch-TTT metrics."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect TRM ARC TTT metrics.")
    parser.add_argument("--root", type=Path, required=True, help="Work root or partition directory.")
    parser.add_argument("--partition-tag", default=None, help="Optional subdir under --root.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--expected-tasks", type=int, default=400)
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print full JSON, including every task. By default, print a compact report.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r") as fh:
        return json.load(fh)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def median(values: Iterable[float]) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return 0.5 * (values[middle - 1] + values[middle])


def percentile(values: Iterable[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return values[index]


def weighted_mean(task_metrics: List[Dict[str, Any]], key: str, subkey: Optional[str] = None) -> float:
    numerator = 0.0
    denominator = 0.0
    for metric in task_metrics:
        weight = float(metric.get("num_test_examples", 0))
        if subkey is None:
            value = float(metric.get(key, 0.0))
        else:
            value = float(metric.get(key, {}).get(subkey, 0.0))
        numerator += value * weight
        denominator += weight
    return numerator / denominator if denominator else 0.0


def print_compact_report(summary: Dict[str, Any]) -> None:
    completed = summary["completed_tasks"]
    expected = summary["expected_tasks"]
    print(f"root: {summary['root']}")
    print(f"completed_tasks: {completed}/{expected} ({completed / expected:.1%})")
    print(f"total_test_examples: {summary['total_test_examples']}")
    print()
    print("task-mean metrics:")
    for key, value in summary["task_mean_pass_at_k"].items():
        print(f"  pass@{key}: {value:.4f}")
    print(f"  oracle: {summary['task_mean_oracle']:.4f}")
    print()
    print("example-weighted metrics:")
    for key, value in summary["example_weighted_pass_at_k"].items():
        print(f"  pass@{key}: {value:.4f}")
    print(f"  oracle: {summary['example_weighted_oracle']:.4f}")
    print()
    print("task counts:")
    for key, value in summary["task_counts"].items():
        print(f"  {key}: {value}")
    print()
    elapsed = summary["elapsed_seconds"]
    print("elapsed per task:")
    print(f"  mean: {elapsed['mean']:.1f}s")
    print(f"  median: {elapsed['median']:.1f}s")
    print(f"  p90: {elapsed['p90']:.1f}s")
    print(f"  max: {elapsed['max']:.1f}s")
    print(f"  total_gpu_hours: {elapsed['total_gpu_hours']:.2f}")


def main() -> None:
    args = parse_args()
    root = args.root / args.partition_tag if args.partition_tag else args.root
    metric_paths = sorted(root.glob("*/metrics.json"))
    task_metrics: List[Dict[str, Any]] = []
    for path in metric_paths:
        try:
            task_metrics.append(load_json(path))
        except Exception as exc:
            print(f"skip unreadable {path}: {exc}")

    pass_keys = sorted(
        {
            key
            for metric in task_metrics
            for key in metric.get("pass_at_k", {}).keys()
        },
        key=lambda item: int(item),
    )
    elapsed_values = [float(metric.get("elapsed_seconds", 0.0)) for metric in task_metrics]
    total_test_examples = sum(int(metric.get("num_test_examples", 0)) for metric in task_metrics)
    task_counts = {
        "fully_pass@1": sum(
            int(metric.get("pass_at_k", {}).get("1", 0.0) >= 1.0)
            for metric in task_metrics
        ),
        "any_pass@1": sum(
            int(metric.get("pass_at_k", {}).get("1", 0.0) > 0.0)
            for metric in task_metrics
        ),
        "fully_pass@2": sum(
            int(metric.get("pass_at_k", {}).get("2", 0.0) >= 1.0)
            for metric in task_metrics
        ),
        "any_pass@2": sum(
            int(metric.get("pass_at_k", {}).get("2", 0.0) > 0.0)
            for metric in task_metrics
        ),
        "fully_oracle": sum(
            int(metric.get("oracle", 0.0) >= 1.0)
            for metric in task_metrics
        ),
        "any_oracle": sum(
            int(metric.get("oracle", 0.0) > 0.0)
            for metric in task_metrics
        ),
    }

    summary = {
        "root": str(root),
        "expected_tasks": args.expected_tasks,
        "completed_tasks": len(task_metrics),
        "total_test_examples": total_test_examples,
        "task_mean_pass_at_k": {
            key: mean(metric.get("pass_at_k", {}).get(key, 0.0) for metric in task_metrics)
            for key in pass_keys
        },
        "task_mean_oracle": mean(metric.get("oracle", 0.0) for metric in task_metrics),
        "example_weighted_pass_at_k": {
            key: weighted_mean(task_metrics, "pass_at_k", key)
            for key in pass_keys
        },
        "example_weighted_oracle": weighted_mean(task_metrics, "oracle"),
        "task_counts": task_counts,
        "elapsed_seconds": {
            "mean": mean(elapsed_values),
            "median": median(elapsed_values),
            "p90": percentile(elapsed_values, 0.90),
            "max": max(elapsed_values) if elapsed_values else 0.0,
            "total": sum(elapsed_values),
            "total_gpu_hours": sum(elapsed_values) / 3600.0,
        },
        "tasks": {
            metric["task_id"]: {
                "pass_at_k": metric.get("pass_at_k", {}),
                "oracle": metric.get("oracle", 0.0),
                "num_test_examples": metric.get("num_test_examples", 0),
                "elapsed_seconds": metric.get("elapsed_seconds", 0.0),
            }
            for metric in task_metrics
        },
    }

    if args.print_json:
        print(json.dumps(summary, indent=2))
    else:
        print_compact_report(summary)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as fh:
            json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
