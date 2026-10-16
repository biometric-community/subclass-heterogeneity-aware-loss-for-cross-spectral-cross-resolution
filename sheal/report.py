"""REPORT.md + SVG figures from real metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sheal.plot_style import BAR_COLOR, FIGSIZE, MULTI_SERIES_COLORS, apply_rcparams, apply_style


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    log_dir = Path(cfg["paths"]["log_dir"])
    fig_dir = Path(cfg["paths"]["figure_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    eval_path = log_dir / "eval_results.json"
    train_path = log_dir / "train_summary.json"
    if not eval_path.is_file():
        Path("REPORT.md").write_text("# REPORT\n\n**BLOCKED:** run eval first.\n")
        return
    metrics = json.loads(eval_path.read_text())
    train = json.loads(train_path.read_text()) if train_path.is_file() else {}
    cr = metrics["cross_resolution"]
    lfw = metrics["lfw"]

    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(["LR→HR Rank-1", "LFW Acc"], [cr["rank1_lr_to_hr"] * 100, lfw["accuracy"] * 100], color=[BAR_COLOR, MULTI_SERIES_COLORS[1]], edgecolor="#002080")
    ax.set_ylabel("%")
    ax.set_title("SHEAL lab twin metrics (ours)")
    apply_style(ax, grid_axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / "metrics.svg", format="svg")
    plt.close(fig)

    hist = train.get("history") or []
    if hist:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot([h["epoch"] for h in hist], [h["loss"] for h in hist], color=MULTI_SERIES_COLORS[0], linewidth=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("SHEAL loss")
        ax.set_title("Training loss (ours)")
        apply_style(ax)
        fig.tight_layout()
        fig.savefig(fig_dir / "train_loss.svg", format="svg")
        plt.close(fig)

    report = f"""# REPORT — SHEAL

Lab twin of Singh et al., *Subclass Heterogeneity Aware Loss* (IEEE TBIOM 2020). CelebA HR/LR subclasses substitute cross-spectral/cross-resolution protocols (see DEVIATIONS).

## Train

- CelebA train: n={train.get('n_train')}, classes={train.get('num_classes')}
- Wall_s={train.get('seconds')}
- Protocol: {train.get('protocol')}

## Metrics (ours)

| Protocol | Value |
|----------|------:|
| CelebA LR→HR Rank-1 | {cr['rank1_lr_to_hr']*100:.2f}% |
| LFW accuracy | {lfw['accuracy']*100:.2f}% |

Gallery IDs={cr['n_gallery']}, probes={cr['n_probes']}, LFW pairs={lfw['n_pairs']}.

## Figures

![Metrics]({(fig_dir / 'metrics.svg').as_posix()})

## Regenerate

```bash
bash scripts/report.sh
```
"""
    Path("REPORT.md").write_text(report)
    print(f"Wrote REPORT.md + {fig_dir}")


if __name__ == "__main__":
    main()
