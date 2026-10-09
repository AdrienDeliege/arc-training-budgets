# Running the paper experiments

## Start here

Requires Python 3.10 or 3.11. Full experiments require Linux and NVIDIA CUDA GPUs; historical pretraining used eight A100 GPUs per model and TTT used one GPU per task. The complete matrices took approximately 4,904 A100 GPU-hours. Use dry runs and small diagnostic checks before a full run.

From the repository root, verify the package without installing ML dependencies:

```sh
python scripts/verify_package.py
python -m unittest discover -s tests -v
python run.py varc ttt --config v51_low --task-id 00576224 --dry-run
python run.py trm pretrain --config v51_40m --dry-run
```

Create separate environments for the two models; their archived dependencies use different WandB versions. The requirement files pin the main direct dependencies, rather than claiming to be complete historical environment lockfiles.

```sh
python3.10 -m venv .venv-varc
source .venv-varc/bin/activate
python -m pip install -r requirements-varc.txt
python scripts/smoke_models.py varc
```

```sh
python3.10 -m venv .venv-trm
source .venv-trm/bin/activate
python -m pip install -r requirements-trm.txt
python scripts/smoke_models.py trm
```

The requirement files use PyTorch 2.7.0, as in the archived dependency lists. Use an appropriate CUDA wheel index for your machine if necessary. CPU model smoke checks use reduced dimensions and do not reproduce experimental scores. TRM's complete training loop requires CUDA. WandB runs offline; no account login is needed for these commands.

## Reproduce a configuration

Scratch TTT on one evaluation task:

```sh
python run.py varc ttt --config v51_low --task-id 00576224 --output outputs/varc-scratch-one
python run.py trm ttt --config v51_low --task-id 00576224 --output outputs/trm-scratch-one
```

Pretrain on the 400 ARC-AGI-1 training tasks, then adapt the selected checkpoint independently to each evaluation task:

```sh
python run.py varc pretrain --config v51_40m --gpus 8 --output outputs/varc-v51-40m
python run.py varc ttt --config v51_low --checkpoint outputs/varc-v51-40m/checkpoint_best.pt --output outputs/varc-v51-40m-ttt-v51-low
```

```sh
python run.py trm pretrain --config v51_40m --gpus 8 --output outputs/trm-v51-40m
python run.py trm ttt --config v51_low --checkpoint outputs/trm-v51-40m/run/checkpoints/best_checkpoint.pt --output outputs/trm-v51-40m-ttt-v51-low
```

Omitting `--task-id` runs the 400 evaluation tasks sequentially. To distribute tasks across GPUs, invoke separate processes with `CUDA_VISIBLE_DEVICES`, disjoint `--task-start`/`--task-count` ranges, and **different output directories**. Combine the per-task files before scoring. Each task starts from its own random initialization or reloads the same pretraining checkpoint; model updates are never shared between evaluation tasks.

Pretraining configurations: `v51_40m`, `v51_130m`, `v1001_40m`, `v1001_130m`. TTT configurations: `v51_low`, `v51_high`, `v1001_low`, `v1001_high`. Counts include the identity variant. Exact model-specific epoch conversions are in [budget configurations](../configs/budgets.json).

Use `--epochs 1` only for diagnostics: it changes the paper's budget. `--no-compile` disables compilation. The launcher records the exact command and overrides in `launch.json` and requires a fresh output directory to prevent accidental mixing of runs. Scratch VARC TTT materializes the selected task's augmentation files; these are generated locally and excluded from Git.

## Score predictions and regenerate figures

```sh
python scripts/score_varc.py outputs/varc-v51-40m-ttt-v51-low/predictions_attempt_0 --out outputs/varc-summary.json
python harness/collect_trm_arc_ttt_results.py --root outputs/trm-v51-40m-ttt-v51-low/run --out outputs/trm-summary.json
python -m pip install -r requirements-analysis.txt
python scripts/make_figures.py
```

VARC scoring requires all 400 tasks unless `--allow-partial` is specified. TRM's collector reports completed/expected coverage; only 400/400-task summaries constitute full matrix cells. Metrics average test-example correctness within each task, then average over tasks. VARC ranks candidates by majority vote and TRM uses its native ranked submissions. See [protocol](PROTOCOL.md).

[reported results](../results/reported_metrics.csv) contains reported aggregates transcribed from the archived experiment notes. Figure generation uses these same reported pass@2 values. These are **not newly recomputed scores**. The package includes no trained checkpoints or raw historical matrix predictions; train checkpoints locally using the commands above. Upstream VARC checkpoints pretrained with RE-ARC are unsuitable substitutes for the paper's controlled ARC-only rows.

