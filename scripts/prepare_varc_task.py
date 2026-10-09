#!/usr/bin/env python3
"""Materialize one scratch-TTT task, preserving original global file-index seeds."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "varc"))
from utils.data_augmentation import augment_raw_data_split_per_task

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants", type=int, choices=[51, 1001], required=True)
    p.add_argument("--task-id", required=True)
    a = p.parse_args()
    data = ROOT / "data" / "arc-agi-1"
    if not (data / "data" / "evaluation" / f"{a.task_id}.json").is_file():
        p.error("Unknown task ID")
    permutations = 9 if a.variants == 51 else 199
    augment_raw_data_split_per_task(dataset_root=data, split="evaluation", output_subdir=f"eval_color_permute_ttt_{permutations}", num_permuate=permutations, task_ids={a.task_id})
