"""Eval: CelebA cross-resolution Rank-1 + LFW verification accuracy."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sheal.data.celeba import resolve_celeba_root
from sheal.data.celeba_lrhr import CelebALRHR
from sheal.data.lfw import LFWPairsDataset, load_lfw_pairs, resolve_lfw_root
from sheal.models.embedder import FaceNet


def best_acc(scores: np.ndarray, labels: np.ndarray) -> float:
    thresholds = np.linspace(scores.min(), scores.max(), 200)
    best = 0.0
    for t in thresholds:
        pred = (scores >= t).astype(np.int32)
        best = max(best, float((pred == labels).mean()))
    return best


@torch.no_grad()
def eval_cross_res(model, cfg, device) -> dict:
    root = resolve_celeba_root(cfg)
    max_n = cfg["data"].get("max_images")
    ds = CelebALRHR(
        root,
        image_size=int(cfg["model"].get("image_size", 112)),
        lr_scale=int(cfg["model"].get("lr_scale", 4)),
        max_n=int(max_n) if max_n is not None else None,
        seed=int(cfg["train"].get("seed", 42)),
        split="test",
    )
    max_ids = cfg["eval"].get("max_ids")
    if max_ids is not None:
        keep = set(sorted(ds.by_class.keys())[: int(max_ids)])
        idxs = [i for i, (_, y) in enumerate(ds.samples) if y in keep]
        from torch.utils.data import Subset

        ds = Subset(ds, idxs)
    loader = DataLoader(ds, batch_size=int(cfg["eval"].get("batch_size", 64)), shuffle=False, num_workers=2)
    gallery = {}
    probes = defaultdict(list)
    for lr, hr, y in tqdm(loader, desc="index"):
        y_np = y.numpy()
        hr = hr.to(device)
        feats = model.embed(hr).cpu().numpy()
        for i, yi in enumerate(y_np):
            yi = int(yi)
            if yi not in gallery:
                gallery[yi] = feats[i]
            probes[yi].append(lr[i])
    g_labels = np.array(sorted(gallery.keys()))
    g_mat = np.stack([gallery[k] for k in g_labels], axis=0)
    correct = total = 0
    pp = int(cfg["eval"].get("probes_per_id", 2))
    for y, lrs in probes.items():
        for lr in lrs[:pp]:
            e = model.embed(lr.unsqueeze(0).to(device)).cpu().numpy()[0]
            pred = g_labels[int(np.argmax(g_mat @ e))]
            correct += int(pred == y)
            total += 1
    return {"n_gallery": len(g_labels), "n_probes": total, "rank1_lr_to_hr": correct / max(total, 1)}


@torch.no_grad()
def eval_lfw(model, cfg, device) -> dict:
    root = resolve_lfw_root(cfg)
    max_p = cfg["eval"].get("max_lfw_pairs")
    pairs = load_lfw_pairs(root, max_pairs=int(max_p) if max_p is not None else None)
    ds = LFWPairsDataset(pairs, image_size=int(cfg["model"].get("image_size", 112)))
    loader = DataLoader(ds, batch_size=int(cfg["eval"].get("batch_size", 64)), shuffle=False, num_workers=2)
    scores, labels = [], []
    for a, b, y in tqdm(loader, desc="lfw"):
        ea = model.embed(a.to(device))
        eb = model.embed(b.to(device))
        scores.append((ea * eb).sum(dim=1).cpu().numpy())
        labels.append(y.numpy())
    s = np.concatenate(scores)
    y = np.concatenate(labels)
    acc = best_acc(s, y)
    return {"accuracy": float(acc), "n_pairs": int(len(y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(Path(cfg["paths"]["checkpoint_dir"]) / "sheal.pt", map_location=device, weights_only=False)
    model = FaceNet(ckpt["num_classes"], emb_dim=ckpt.get("emb_dim", 128), pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    results = {
        "cross_resolution": eval_cross_res(model, cfg, device),
        "lfw": eval_lfw(model, cfg, device),
    }
    print(json.dumps(results, indent=2))
    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "eval_results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
