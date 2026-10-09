#!/usr/bin/env python3
"""Score VARC candidate lists with task-mean pass@1/pass@2/oracle."""
import argparse
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def score_task(task, predictions):
    scores = {"pass_at_1": 0., "pass_at_2": 0., "oracle": 0.}
    if set(predictions) != {str(i) for i in range(len(task["test"]))}:
        raise ValueError("Prediction keys do not match the task's test examples")
    for i, example in enumerate(task["test"]):
        candidates = predictions[str(i)]
        # Stable ties preserve first occurrence, as in upstream get_majority_vote.
        counts = Counter(json.dumps(grid) for grid in candidates)
        ranked = sorted(counts, key=counts.get, reverse=True)
        truth = json.dumps(example["output"])
        n = len(task["test"])
        scores["pass_at_1"] += (truth in ranked[:1]) / n
        scores["pass_at_2"] += (truth in ranked[:2]) / n
        scores["oracle"] += (truth in ranked) / n
    return scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("predictions", type=Path, help="Directory containing <task>_predictions.json")
    p.add_argument("--out", type=Path)
    p.add_argument("--allow-partial", action="store_true", help="Score completed tasks only, reporting coverage")
    a = p.parse_args()
    tasks = sorted((ROOT / "data" / "arc-agi-1" / "data" / "evaluation").glob("*.json"))
    records = {}
    for task_path in tasks:
        pred_path = a.predictions / f"{task_path.stem}_predictions.json"
        if pred_path.exists():
            records[task_path.stem] = score_task(json.loads(task_path.read_text()), json.loads(pred_path.read_text()))
    if not records:
        p.error("No predictions found")
    if len(records) != len(tasks) and not a.allow_partial:
        p.error(f"Incomplete evaluation: {len(records)}/{len(tasks)} tasks; use --allow-partial for diagnostic reporting")
    summary = {"completed_tasks": len(records), "expected_tasks": len(tasks), "aggregation": "mean over tasks, after mean over test examples within each task", "task_mean": {k: sum(v[k] for v in records.values()) / len(records) for k in ["pass_at_1", "pass_at_2", "oracle"]}, "tasks": records}
    text = json.dumps(summary, indent=2)
    print(text)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
