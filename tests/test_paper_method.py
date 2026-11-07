"""Loss terms and the evaluation protocol, checked on small controlled inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from sheal.data.celeba_lrhr import CelebALRHR, make_lr
from sheal.data.lfw import load_lfw_pairs
from sheal.eval import eval_cross_res
from sheal.losses import sheal_loss
from sheal.models.embedder import cosface_logits


def test_cosface_subtracts_the_margin_from_the_target_class_only():
    emb = F.normalize(torch.randn(4, 8), dim=1)
    weight = torch.randn(5, 8)
    y = torch.tensor([0, 1, 2, 3])
    cosine = emb @ F.normalize(weight, dim=1).t()
    expected = 30.0 * cosine
    expected[torch.arange(4), y] -= 30.0 * 0.35
    assert torch.allclose(cosface_logits(emb, weight, y, s=30.0, m=0.35), expected, atol=1e-5)


def test_without_subclass_terms_the_loss_is_cosface_on_both_subclasses():
    torch.manual_seed(0)
    hr, lr = F.normalize(torch.randn(6, 8), dim=1), F.normalize(torch.randn(6, 8), dim=1)
    weight, y = torch.randn(4, 8), torch.tensor([0, 1, 2, 3, 0, 1])
    expected = F.cross_entropy(cosface_logits(hr, weight, y), y) + F.cross_entropy(
        cosface_logits(lr, weight, y), y
    )
    assert sheal_loss(hr, lr, y, weight, alpha_cross=0.0, beta_neg=0.0).item() == pytest.approx(expected.item())


def test_cross_subclass_term_vanishes_when_subclasses_align():
    torch.manual_seed(1)
    emb = F.normalize(torch.randn(6, 8), dim=1)
    weight, y = torch.randn(4, 8), torch.tensor([0, 1, 2, 3, 0, 1])
    base = sheal_loss(emb, emb, y, weight, alpha_cross=0.0, beta_neg=0.0)
    assert sheal_loss(emb, emb, y, weight, alpha_cross=5.0, beta_neg=0.0).item() == pytest.approx(base.item())


def test_hard_negative_push_ignores_same_identity_pairs():
    torch.manual_seed(2)
    emb = F.normalize(torch.randn(6, 8), dim=1)
    weight, y = torch.randn(4, 8), torch.zeros(6, dtype=torch.long)
    base = sheal_loss(emb, emb, y, weight, alpha_cross=0.0, beta_neg=0.0)
    assert sheal_loss(emb, emb, y, weight, alpha_cross=0.0, beta_neg=5.0).item() == pytest.approx(base.item())


def test_batch_size_matches_the_paper(default_cfg, targets):
    assert default_cfg["train"]["batch_size"] == targets["batch_size"]


def test_lr_probe_keeps_size_and_loses_detail():
    hr = torch.rand(3, 112, 112) * 2 - 1
    lr = make_lr(hr, 4)
    assert lr.shape == hr.shape
    assert lr.std() < hr.std()


@pytest.fixture(scope="module")
def celeba(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("celeba")
    images = root / "img_align_celeba"
    images.mkdir()
    rng = np.random.RandomState(0)
    lines = []
    for subject in range(1, 21):
        for k in range(3):
            name = f"{subject:03d}{k}.png"
            Image.fromarray(rng.randint(0, 255, (32, 32, 3), dtype=np.uint8)).save(images / name)
            lines.append(f"{name} {subject}")
    (root / "identity_CelebA.txt").write_text("\n".join(lines) + "\n")
    return root


def test_train_and_test_identities_are_disjoint(celeba):
    def subjects(split: str) -> set:
        return {fn[:3] for fn, _ in CelebALRHR(celeba, image_size=32, split=split).samples}

    assert not subjects("train") & subjects("test")


class RecordingEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        self.calls.append(x.clone())
        return F.normalize(x.flatten(1)[:, :16], dim=1)


def test_probes_are_not_degraded_copies_of_gallery_images(celeba):
    """A probe made from the gallery image itself is matched to its own source."""
    cfg = {
        "data": {"celeba_root": str(celeba), "max_images": None},
        "model": {"image_size": 32, "lr_scale": 4},
        "train": {"seed": 42},
        "eval": {"max_ids": None, "batch_size": 64, "probes_per_id": 2},
    }
    model = RecordingEmbedder()
    eval_cross_res(model, cfg, torch.device("cpu"))
    probes = [call[0] for call in model.calls if call.shape[0] == 1]
    ds = CelebALRHR(celeba, image_size=32, lr_scale=4, seed=42, split="test")
    gallery_lr = [ds[indices[0]][0] for indices in ds.by_class.values()]
    leaked = sum(any(torch.equal(p, g) for g in gallery_lr) for p in probes)
    assert leaked == 0


def test_capped_lfw_pairs_keep_both_classes(tmp_path):
    lines = ["1\t5"]
    lines += [f"A {i + 1} {i + 2}" for i in range(5)]
    lines += [f"A {i + 1} B {i + 1}" for i in range(5)]
    (tmp_path / "pairs.txt").write_text("\n".join(lines) + "\n")
    pairs = load_lfw_pairs(tmp_path, max_pairs=4)
    assert sum(p.same for p in pairs) == 2
    assert len(pairs) == 4
