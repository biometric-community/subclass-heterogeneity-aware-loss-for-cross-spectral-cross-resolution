# REPORT — SHEAL

Lab twin of Singh et al., *Subclass Heterogeneity Aware Loss* (IEEE TBIOM 2020). CelebA HR/LR subclasses substitute cross-spectral/cross-resolution protocols (see DEVIATIONS).

## Train

- CelebA train: n=20000, classes=5765
- Wall_s=212.50160932540894
- Protocol: full

## Metrics (ours)

| Protocol | Value |
|----------|------:|
| CelebA LR→HR Rank-1 | 57.84% |
| LFW accuracy | 71.87% |

Gallery IDs=300, probes=593, LFW pairs=6000.

## Figures

![Metrics](outputs/figures/full/metrics.svg)

## Regenerate

```bash
bash scripts/report.sh
```
