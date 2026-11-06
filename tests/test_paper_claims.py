"""The paper's premise, reproduced with this repo's loss on synthetic HR/LR features.

Abstract: a loss aware of subclass heterogeneity should produce embeddings that match
across subclasses. Measured on identities unseen in training.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sheal.losses import sheal_loss

DIM, EMB, TRAIN_IDS, TEST_IDS = 64, 32, 40, 40


def train_and_measure(alpha_cross: float, beta_neg: float, seed: int = 0) -> tuple[float, float]:
    """Return (LR to HR Rank-1, mean same-identity HR/LR cosine) on unseen identities."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    shift = 2.0 * rng.normal(size=DIM)  # the fixed LR domain offset

    def sample(centers: np.ndarray, lr: bool) -> torch.Tensor:
        x = centers + rng.normal(size=centers.shape)
        if lr:
            x = 0.6 * x + shift + rng.normal(size=centers.shape)
        return torch.tensor(x, dtype=torch.float32)

    train_centers, test_centers = rng.normal(size=(TRAIN_IDS, DIM)), rng.normal(size=(TEST_IDS, DIM))
    embedder = torch.nn.Linear(DIM, EMB)
    weight = torch.nn.Parameter(torch.randn(TRAIN_IDS, EMB) * 0.1)
    opt = torch.optim.Adam(list(embedder.parameters()) + [weight], lr=1e-2)
    y = torch.arange(TRAIN_IDS)
    for _ in range(300):
        hr = F.normalize(embedder(sample(train_centers, lr=False)), dim=1)
        lr = F.normalize(embedder(sample(train_centers, lr=True)), dim=1)
        loss = sheal_loss(hr, lr, y, weight, alpha_cross=alpha_cross, beta_neg=beta_neg)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        gallery = F.normalize(embedder(sample(test_centers, lr=False)), dim=1)
        probes = F.normalize(embedder(sample(test_centers, lr=True)), dim=1)
    rank1 = float(((probes @ gallery.t()).argmax(dim=1) == torch.arange(TEST_IDS)).float().mean())
    return rank1, float((probes * gallery).sum(dim=1).mean())


def test_cross_subclass_term_is_the_mean_hr_lr_cosine_distance():
    torch.manual_seed(0)
    hr, lr = F.normalize(torch.randn(8, EMB), dim=1), F.normalize(torch.randn(8, EMB), dim=1)
    weight, y = torch.randn(4, EMB), torch.arange(8) % 4
    with_term = sheal_loss(hr, lr, y, weight, alpha_cross=1.0, beta_neg=0.0)
    without_term = sheal_loss(hr, lr, y, weight, alpha_cross=0.0, beta_neg=0.0)
    expected = (1.0 - (hr * lr).sum(dim=1)).mean()
    assert (with_term - without_term).item() == pytest.approx(expected.item(), abs=1e-5)


def test_full_loss_matches_well_above_chance():
    assert train_and_measure(1.0, 0.5)[0] >= 5.0 / TEST_IDS
