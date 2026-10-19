# Reproducibility and execution

## 1. Verify the published result on CPU

From the repository root:

```bash
pip install -r docs/setup/requirements-metrics.txt
python run.py verify-results
```

This recomputes 5-group-B mIoU and group IoUs from the published full-Test confusion matrix. It does not execute GPU inference.

## 2. Model environment

The observed environment used Python 3.11, PyTorch 2.5.1+cu124, Windows, and an RTX 3070 Ti (8 GB). FlashAttention was disabled. See [observed package versions](setup/environment_observed.json) and [core requirements](setup/requirements.txt).

The requirements describe the observed versions, rather than a clean-machine installation that has been independently verified. Windows spconv-cu120 2.3.6 and the torch-scatter wheel require compatible Python/CUDA builds. Virtual environments and installation wheels are not included. Utonia source attribution is retained in [third-party notices](THIRD_PARTY_NOTICES.md).

## 3. Check the official dataset

```bash
python run.py check-data --data-root D:/datasets/rohbau3d
```

Download the original arrays separately. `data/splits.json` preserves the official 354/54/96 scene assignments. This command checks required file presence, not complete dataset checksums.

## 4. Evaluate the published head with an existing cache

```bash
python run.py evaluate --data-root D:/datasets/rohbau3d --cache-root D:/rohbau3d_utonia_work/cache_spill/94739f87a44ad936 --split test --output runs/test30
```

The backbone is frozen, so its verified feature cache is reused while the epoch 30 head executes new GPU inference. All 96 scenes and 2,294,339,698 original points are required for the full-Test result. `--scene-limit 1` performs a partial smoke check. Do not report partial scores as final performance.

Paths are command-line arguments. No personal dataset or cache path is embedded in the release CLI.

## 5. Build a fresh backbone feature cache

```bash
python run.py build-cache --data-root D:/datasets/rohbau3d --cache-root D:/derived/rohbau3d/fresh_cache --backbone weights/utonia_tiny.pth --split test
```

Obtain `utonia_tiny.pth` from the official distribution and check the SHA256 in `configs/recipe.json`. Keep original data and derived caches separate. Run cache construction for Train and Validation as well if retraining is required. The historical full feature cache is approximately 112 GB, so reserve additional working space.

A fresh build uses a separate cache namespace. Random serialization and GPU numerical behavior mean rebuilt features are not guaranteed to match the historical cache bit for bit. Rebuilding every scene and retraining all 30 epochs were not repeated as part of release verification.

## 6. Retrain only when explicitly requested

Inspect the recipe without starting training:

```bash
python run.py train --cache-root D:/derived/rohbau3d/fresh_cache --dry-run
```

Start an independent retraining run:

```bash
python run.py train --data-root D:/datasets/rohbau3d --cache-root D:/derived/rohbau3d/fresh_cache --epochs 30 --output runs/retrain --start-training
```

This trains the head on frozen features with 18-class CrossEntropy/AdamW, saves each epoch, and evaluates the full Validation split every 5 epochs. The release adapter does not reconstruct the historical cache-generation/RNG/resume chronology. Original local experiment records are preserved separately. Release verification did not restart training. Do not use Test for retraining decisions.
