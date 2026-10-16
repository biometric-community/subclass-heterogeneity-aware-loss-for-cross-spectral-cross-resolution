"""SHEAL-style subclass heterogeneity-aware objective (lab twin of paper Sec. III).

Subclasses = resolution modes {HR, LR}. We combine:
  1) CosFace classification on both subclasses
  2) Same-identity cross-subclass pull (HR↔LR embedding alignment)
  3) Subclass-aware hard-negative push across identities
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sheal.models.embedder import cosface_logits


def sheal_loss(
    emb_hr: torch.Tensor,
    emb_lr: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    s: float = 30.0,
    m: float = 0.35,
    alpha_cross: float = 1.0,
    beta_neg: float = 0.5,
) -> torch.Tensor:
    logits_hr = cosface_logits(emb_hr, weight, y, s=s, m=m)
    logits_lr = cosface_logits(emb_lr, weight, y, s=s, m=m)
    cls = F.cross_entropy(logits_hr, y) + F.cross_entropy(logits_lr, y)

    # same-id cross-subclass attraction
    cross = (1.0 - (emb_hr * emb_lr).sum(dim=1)).mean()

    # push LR away from other identities' HR (batch hard negatives)
    sim = emb_lr @ emb_hr.t()  # [B,B]
    mask = y.unsqueeze(1).eq(y.unsqueeze(0))
    sim = sim.masked_fill(mask, -1.0)
    hard = sim.max(dim=1).values.clamp(min=-1.0, max=1.0)
    neg = F.relu(hard - 0.2).mean()

    return cls + alpha_cross * cross + beta_neg * neg
