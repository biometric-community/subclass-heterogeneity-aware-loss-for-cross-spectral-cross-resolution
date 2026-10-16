# SHEAL: Subclass Heterogeneity Aware Loss

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-green?logo=creativecommons&logoColor=white)](https://creativecommons.org/licenses/by/4.0/)

Lab twin of Singh et al., *Subclass Heterogeneity Aware Loss for Cross-Spectral Cross-Resolution Face Recognition* (IEEE TBIOM 2020).

**Repository:** https://github.com/biometric-community/subclass-heterogeneity-aware-loss-for-cross-spectral-cross-resolution

## Setup

```bash
bash scripts/setup_env.sh
```

## Data

Uses CelebA (HR/LR subclasses) + LFW. Paper SCface / FaceSurv / CASIA NIR-VIS → see `DEVIATIONS.md`.

```bash
bash scripts/check_dataset_size.sh ../../datasets/celeba ../../datasets/lfw
```

## Train / Eval / Report

```bash
export CUDA_VISIBLE_DEVICES=0
bash scripts/smoke.sh
bash scripts/train_full.sh
```

Results: [`REPORT.md`](REPORT.md).
