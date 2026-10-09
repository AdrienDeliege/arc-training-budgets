# Experimental protocol

Pretraining uses demonstration pairs from ARC-AGI-1 **training** tasks only (1,302 pairs). The held-out pairs belonging to those same training tasks serve as validation for checkpoint selection. The 400 evaluation tasks are absent from pretraining.

During TTT, each evaluation task's demonstrations train a fresh per-task model. Held-out evaluation outputs are used by the evaluator and scorer, not as training targets. VARC skips pretrained task tokens; TRM's archived loader resizes mismatched puzzle embeddings using the mean pretrained embedding. The same checkpoint is loaded independently for every task.

The total sample-presentation budget is `B = D × V × P`, with `D` the demonstration count, `V` the number of task variants including identity, and `P` passes over those variants. Native TRM epochs sample variants rather than sweeping all variants, so their counts differ from VARC's epochs.

| Pretraining | VARC epochs | TRM epochs | Approximate presentations |
|---|---:|---:|---:|
| 51 variants / 40M | 603 | 30,906 | 40M |
| 51 variants / 130M | 1,958 | 100,000 | 130M |
| 1001 variants / 40M | 31 | 30,906 | 40M |
| 1001 variants / 130M | 100 | 100,000 | 130M |

VARC rounds pretraining epochs up to reach the target presentation count. These settings are approximately budget matched; they do not force identical exact presentation counts.

| TTT | VARC epochs | TRM epochs | Presentations per demonstration |
|---|---:|---:|---:|
| 51 / low | 100 | 5,100 | 5,100 |
| 51 / high | 1,962 | 100,100 | VARC 100,062; TRM 100,100 |
| 1001 / low | 4 | 5,100 | VARC 4,004; TRM 5,100 |
| 1001 / high | 100 | 100,100 | 100,100 |

The original VARC 1001/low setting uses four epochs, so its realized presentation count is lower than 5,100. This release preserves that implementation instead of replacing it with an invented fractional epoch. Budget labels follow the reported comparison.

The table gives configured epoch counts and their nominal presentation conversion. The archived VARC TTT loop starts a fresh run at epoch 0 and includes the configured final epoch (`range(start_epoch, epochs + 1)`), so a fresh run executes `epochs + 1` passes. For example, a diagnostic `--epochs 1` executes two passes. This release preserves the historical loop. Use the emitted `timing.json` (`train_examples_seen`, batch visits, and optimizer steps) to distinguish measured visits from nominal budget labels; do not assume the configured integer is an exact measured presentation count.

VARC uses a 64×64 canvas, 2×2 patches, depth 10, width 512, eight attention heads, AdamW-style optimization with learning rate 3e-4 and zero weight decay. Pretraining batch size is 32 per process; TTT batch size is 8. Native default seed is 42. Augmentation uses the original six-element `varc_basic` geometry set and task-consistent color maps. Pretraining uses fixed on-the-fly variants and augmentation seed 12345. Scratch TTT uses the historical materialized augmentation generator (seed 0); pretrained TTT uses fixed on-the-fly variants. Ten inference views per variant produce 10V candidates before inversion and voting.

TRM retains the frozen upstream model/trainer/config (`vendor/trm/config/arch/trm.yaml`): width 512, two low-level layers, H_cycles=3, L_cycles=6, halt_max_steps=16. The harness uses learning rate 1e-4, puzzle-embedding learning rate 1e-2, weight decays 0.1, pretraining global batch size 768, TTT global batch size 64, and seed 0. Pretraining warmup is 2,000 steps; TTT warmup is 50 (low) or 200 (high). TRM uses 50 or 1000 extra augmentations and native ARC evaluation with submission_K=51 or 1001.

Metrics are task means, with each task first averaging over all its test examples. Oracle measures whether the candidate set contains a correct output; it is not a deployable selection rule. Candidate generation differs between model families, so oracle is most directly comparable within a model family.

The two matrices contain 40 TTT runs of 400 tasks each, plus eight pretraining runs. RE-ARC runs, ConceptARC transfer, PromptViT, visual TRM, and other research forks are outside this release's experiment scope.
