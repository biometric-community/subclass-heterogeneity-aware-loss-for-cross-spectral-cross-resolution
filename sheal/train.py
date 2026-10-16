"""Train SHEAL embedder on CelebA HR+LR subclasses."""

from __future__ import annotations

import argparse
import json
import sys
import time
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
from sheal.losses import sheal_loss
from sheal.models.embedder import FaceNet


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device("cuda:0" if torch.cuda.is_available() and str(cfg.get("device", "cuda")).startswith("cuda") else "cpu")

    root = resolve_celeba_root(cfg)
    max_n = cfg["data"].get("max_images")
    ds = CelebALRHR(
        root,
        image_size=int(cfg["model"].get("image_size", 112)),
        lr_scale=int(cfg["model"].get("lr_scale", 4)),
        max_n=int(max_n) if max_n is not None else None,
        seed=int(cfg["train"].get("seed", 42)),
        split="train",
    )
    loader = DataLoader(
        ds,
        batch_size=int(cfg["train"].get("batch_size", 64)),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    print(f"[data] n={len(ds)} classes={ds.num_classes} device={device}")

    model = FaceNet(ds.num_classes, emb_dim=int(cfg["model"].get("emb_dim", 128)), pretrained=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["train"].get("lr", 1e-3)))
    epochs = int(cfg["train"].get("epochs", 5))
    hist = []
    t0 = time.time()
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for lr, hr, y in tqdm(loader, desc=f"ep{ep}/{epochs}", leave=False):
            lr, hr, y = lr.to(device), hr.to(device), y.to(device)
            emb_hr = model.embed(hr)
            emb_lr = model.embed(lr)
            loss = sheal_loss(
                emb_hr,
                emb_lr,
                y,
                model.weight,
                s=float(cfg["train"].get("cosface_s", 30.0)),
                m=float(cfg["train"].get("cosface_m", 0.35)),
                alpha_cross=float(cfg["train"].get("alpha_cross", 1.0)),
                beta_neg=float(cfg["train"].get("beta_neg", 0.5)),
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        mean = float(np.mean(losses))
        hist.append({"epoch": ep, "loss": mean})
        print(f"[train] epoch={ep} loss={mean:.4f}", flush=True)

    torch.save({"model": model.state_dict(), "num_classes": ds.num_classes, "emb_dim": model.fc.out_features, "cfg": cfg}, ckpt_dir / "sheal.pt")
    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "protocol": cfg["train"].get("protocol", "full"),
        "n_train": len(ds),
        "num_classes": ds.num_classes,
        "history": hist,
        "seconds": time.time() - t0,
        "device": str(device),
    }
    (log_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"n_train": summary["n_train"], "seconds": summary["seconds"]}, indent=2))


if __name__ == "__main__":
    main()
