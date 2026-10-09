#!/usr/bin/env python3
"""Check frozen source hashes, source syntax, ARC data consistency and result coverage."""
import ast
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only", action="store_true", help="Check frozen source/data and protocol consistency without enforcing the packaged archive's full-file hashes; suitable for CI after documentation edits.")
    args = parser.parse_args()
    release_path = ROOT / "release_manifest.json"
    if release_path.exists() and not args.source_only:
        for entry in json.loads(release_path.read_text())["files"]:
            assert hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest() == entry["sha256"], f"Release file mismatch: {entry['path']}"
    manifest = json.loads((ROOT / "source_manifest.json").read_text())
    for entry in manifest["files"]:
        p = ROOT / entry["path"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == entry["sha256"], f"Hash mismatch: {p}"
    python_files = sorted(ROOT.rglob("*.py"))
    for p in python_files:
        ast.parse(p.read_text(), filename=str(p))
    demo_count = 0
    for split in ["training", "evaluation"]:
        tasks = sorted((ROOT / "data" / "arc-agi-1" / "data" / split).glob("*.json"))
        assert len(tasks) == 400
        prefix = ROOT / "vendor" / "trm" / "kaggle" / "combined" / f"arc-agi_{split}"
        challenges = json.loads(prefix.with_name(prefix.name + "_challenges.json").read_text())
        solutions = json.loads(prefix.with_name(prefix.name + "_solutions.json").read_text())
        assert set(challenges) == {p.stem for p in tasks} == set(solutions)
        for p in tasks:
            task = json.loads(p.read_text())
            assert challenges[p.stem] == {"train": task["train"], "test": [{"input": t["input"]} for t in task["test"]]}
            assert solutions[p.stem] == [t["output"] for t in task["test"]]
            if split == "training":
                demo_count += len(task["train"])
    assert demo_count == 1302
    with (ROOT / "results" / "reported_metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 40
    keys = {(r["model"], r["pretraining"], r["ttt_variants"], r["ttt_budget"]) for r in rows}
    assert len(keys) == 40
    for r in rows:
        assert 0 <= float(r["pass_at_1"]) <= float(r["pass_at_2"]) <= float(r["oracle"]) <= 1
        assert int(r["evaluation_tasks"]) == 400
    print(f"PASS: {len(manifest['files'])} source/data hashes, {len(python_files)} Python syntax checks, 800 matching ARC tasks, 1302 training demos, 40 matrix records.")


if __name__ == "__main__":
    main()
