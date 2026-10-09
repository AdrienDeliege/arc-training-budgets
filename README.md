# Training Budgets Matter for ARC

*Release packaging and documentation prepared with assistance from Codex (GPT-6.1 Sol). Model implementations retain attribution to their original authors. Tested and validated locally. Reach out in case of questions.*

Code accompanying **Pretraining and Test-Time Training Budgets Matter for ARC: A Controlled Study of VARC and TRM**, by Adrien Deliège and Marc Van Droogenbroeck (Accepted for BNAIC/BeNeLearn 2026).

The paper compares VARC and TRM across pretraining and independent per-task test-time training (TTT) budgets, using ARC-AGI-1.

## What is included

- The paper's VARC and TRM models used, with the training, augmentation and evaluation code needed for the study.
- Paper configurations, all 800 public ARC-AGI-1 tasks, reported aggregate results, scoring and figure scripts.

The model code is **bundled**, so downloading the original repositories is unnecessary. These are frozen study snapshots with paper-specific adaptations, including VARC's augmentation/task-ID handling and our experiment wrappers. The package excludes unrelated experimental models and dataset builders. See [third-party attribution](THIRD_PARTY_NOTICES.md) and [adaptations and validation](docs/VALIDATION.md).

## Quick start

Use Python 3.10 or 3.11. Full experiments require Linux and NVIDIA CUDA GPUs. From the repository root, check the package and preview a command without installing ML dependencies:

```sh
python scripts/verify_package.py
python -m unittest discover -s tests -v
python run.py varc ttt --config v51_low --task-id 00576224 --dry-run
```

Install each model in a **separate virtual environment** using `requirements-varc.txt` or `requirements-trm.txt`. For example:

```sh
python3.10 -m venv .venv-varc
source .venv-varc/bin/activate
python -m pip install -r requirements-varc.txt
python scripts/smoke_models.py varc
python run.py varc ttt --config v51_low --task-id 00576224 --output outputs/varc-one
```

The launcher keeps WandB offline and requires a fresh output directory. Omitting `--task-id` runs all 400 evaluation tasks sequentially; each task starts independently.

Pretraining configurations: `v51_40m`, `v51_130m`, `v1001_40m`, `v1001_130m`. TTT configurations: `v51_low`, `v51_high`, `v1001_low`, `v1001_high`. Exact budgets are in [configs/budgets.json](configs/budgets.json).

See [running experiments](docs/RUNNING.md) for TRM setup, pretraining, checkpoint reuse, scoring and figure generation, and [the protocol](docs/PROTOCOL.md) for evaluation details.

## Results and reproducibility

[Reported results](results/reported_metrics.csv) are archived paper aggregates, not newly recomputed scores. Historical checkpoints and raw predictions are not included. Source/data hashes are recorded in [source_manifest.json](source_manifest.json).

Package checks and reduced model diagnostics passed. Full GPU matrices have not been rerun during packaging; see [validation limits](docs/VALIDATION.md) and the [privacy audit](docs/PRIVACY_AUDIT.md).

## Citation and licenses

```bibtex
@inproceedings{deliege2026trainingbudgets,
  title = {Pretraining and Test-Time Training Budgets Matter for {ARC}: A Controlled Study of {VARC} and {TRM}},
  author = {Deliege, Adrien and Van Droogenbroeck, Marc},
  year = {2026},
  booktitle = {BNAIC/BeNeLearn}
}
```

New release tooling uses MIT; upstream VARC/TRM retain their MIT notices, and ARC-AGI-1 uses Apache-2.0. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
