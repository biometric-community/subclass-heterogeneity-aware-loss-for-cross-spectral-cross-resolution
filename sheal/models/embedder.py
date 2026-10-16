"""ResNet18 face embedder + CosFace-style classifier head."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FaceNet(nn.Module):
    def __init__(self, n_classes: int, emb_dim: int = 128, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(net.children())[:-1])
        self.fc = nn.Linear(512, emb_dim)
        self.bn = nn.BatchNorm1d(emb_dim)
        self.weight = nn.Parameter(torch.FloatTensor(n_classes, emb_dim))
        nn.init.xavier_uniform_(self.weight)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1.0) * 0.5
        h = torch.flatten(self.features(x), 1)
        h = self.bn(self.fc(h))
        return F.normalize(h, p=2, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


def cosface_logits(emb: torch.Tensor, weight: torch.Tensor, y: torch.Tensor, s: float = 30.0, m: float = 0.35) -> torch.Tensor:
    """CosFace (additive cosine margin) logits."""
    w = F.normalize(weight, p=2, dim=1)
    cosine = F.linear(emb, w)
    one_hot = torch.zeros_like(cosine)
    one_hot.scatter_(1, y.view(-1, 1), 1.0)
    logits = s * (cosine - one_hot * m)
    return logits
