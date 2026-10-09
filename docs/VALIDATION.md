# Release validation — 9 October 2026

This is a prepared code release candidate. It has not been publicly published and the historical full GPU matrices have not been rerun during packaging.

## Completed checks

- All 800 bundled ARC-AGI-1 tasks agree exactly in content between the VARC task-per-file representation and the TRM challenge/solution representation. The training demonstrations total 1,302 pairs.
- Source/data SHA-256 verification and syntax checks pass for every bundled Python source.
- Seven standard-library test cases pass. An additional PyTorch checkpoint test passes all eight combinations of compiled/eager source and destination keys with unchanged/resized embeddings. They exercise all pretraining/TTT configurations through the native argument parsers or native TRM dry-run command builder, budget conversion, scratch/pretrained VARC routing, vote ties and task weighting, and side-effect-free release dry runs.
- Both VARC and TRM pass reduced CPU forward/backward checks with PyTorch 2.7.0. VARC uses torchvision 0.22.0. Tests used Python 3.10, with those two wheels installed into a temporary directory; other smoke-test dependencies came from the existing local environment. This is not a clean installation test of every pinned requirement.
- Reduced VARC pretraining, scratch TTT and pretrained TTT diagnostics completed. The scratch diagnostic completed dataset loading, two native passes for the diagnostic `--epochs 1`, inference, prediction saving and scoring on ARC task `00576224`. It used depth 1, width 32, four heads, patch size 8, one prediction view, no AMP and no compilation. The release scorer agrees with the native evaluator. These reduced settings do not validate the paper's performance figures.
- Selecting a single scratch task preserves the original global file-index augmentation seed. For a non-first task, all 51 generated files matched the untouched archived generator byte-for-byte.
- The paper's four figure-generation routines execute successfully and produce eight PDF/PNG outputs from bundled inputs.

## Release adaptations

1. VARC's augmentation generator accepts an optional task selection **after** enumeration, to avoid generating every task's augmentation files for a single-task run.
2. VARC's prediction writers use `Path("outputs") / save_name` so absolute destinations work consistently with its existing timing writer.
3. The TRM harness defaults to bundled `vendor/trm` rather than a sibling research checkout.
4. The camera-ready figure script resolves the bundled ARC task and output directory relative to this repository.
5. TRM checkpoint loading supports both compiled and eager checkpoint keys, including runs with `--no-compile`. It retains the study's mean-embedding initialization when the number of task identifiers changes. Checkpoints load onto the model's device.

Original and adapted file hashes are recorded in `source_manifest.json`; private repository identifiers and commit metadata are omitted. The package also narrows VARC's shared model/trainer/loader/parser and augmentation helpers to the plain ViT used in the paper, and removes unused TRM models/configs and non-ARC dataset builders. Reduced comparisons against the pre-trim archive show identical seeded VARC weights, masked forward outputs and parameter gradients in both training and evaluation modes, plus identical materialized and fixed on-the-fly dataset samples. The study's SDPA attention, pixel cross-entropy, optimizer settings, augmentation sampling, vote ordering and native epoch loops are preserved.

## Remaining validation limits

- Full pretraining and 400-task TTT require Linux/CUDA and were not executed here. The prepared direct dependency pins are not a recovered complete container lockfile.
- The reported CSV comes from archived experiment notes, not rescoring of recovered historical predictions. Historical matrix checkpoints and raw per-task predictions are absent from this package.
- VARC and harness archive commits postdate the original runs. The local upstream TRM snapshot has no Git metadata; its file hashes are pinned, but exact historical per-run commits remain unavailable.
- Nominal epoch-to-budget conversions differ from measured presentation counts in some native settings, including VARC's inclusive TTT epoch loop. These implementation details are preserved and documented in `PROTOCOL.md`.

Useful next validation on Linux/CUDA: install each requirement file in a clean environment; run one native pretraining diagnostic and one scratch/pretrained TTT diagnostic per model, preserving configuration, logs and hashes. Complete-score reproduction is a separate compute task.

## Final release check

The final package was checked again from a fresh ZIP extraction:

- All 120 aggregate metric values match the camera-ready `main.tex` tables; all documentation links resolve.
- Source/data integrity, Python syntax, paper-only code scope and all eight tests pass in the PyTorch environment. The checkpoint test is skipped when ML dependencies are absent.
- Both model forward/backward smoke tests and all four figure routines pass; eight figure files are generated.
- Reduced VARC pretraining, scratch TTT and pretrained TTT complete using ten inference views. Each TTT diagnostic records 204 training examples seen and 510 inference items; the independent release scorer agrees with native scoring.
- Compiled/eager TRM checkpoint loading and mean-embedding resizing pass, including checks with actual reduced TRM model checkpoints.
- The exact ZIP passes sensitive-identifier scans, normalized metadata checks, archive integrity and SHA-256 verification. Its README matches the current source folder.

These are local release checks. Full CUDA training, a fresh installation of every dependency pin and reproduction of the historical GPU matrices remain unverified.
