"""LFW pairs protocol."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def resolve_lfw_root(cfg: dict) -> Path:
    root = Path(cfg["data"]["lfw_root"])
    if not root.is_absolute():
        root = (Path(__file__).resolve().parents[2] / root).resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    return root


def _lfw_image_dir(root: Path) -> Path:
    for cand in (root / "lfw", root / "lfw_funneled", root / "images", root):
        if cand.is_dir() and any(cand.iterdir()):
            sub = next(cand.iterdir(), None)
            if sub is not None and sub.is_dir():
                return cand
    return root / "lfw"


def _lfw_path(img_dir: Path, name: str, idx: int) -> Path:
    return img_dir / name / f"{name}_{idx:04d}.jpg"


@dataclass
class PairItem:
    path_a: Path
    path_b: Path
    same: bool
    fold: int = 0


def load_lfw_pairs(root: Path, max_pairs: Optional[int] = None) -> List[PairItem]:
    pairs_path = root / "pairs.txt"
    if not pairs_path.exists():
        raise FileNotFoundError(f"Missing pairs.txt under {root}")
    img_dir = _lfw_image_dir(root)
    lines = pairs_path.read_text().strip().splitlines()
    header = lines[0].split()
    n_folds, n_per = int(header[0]), int(header[1])
    pairs: List[PairItem] = []
    i = 1
    for fold in range(n_folds):
        for _ in range(n_per):
            parts = lines[i].split()
            i += 1
            name, a, b = parts[0], int(parts[1]), int(parts[2])
            pairs.append(PairItem(_lfw_path(img_dir, name, a), _lfw_path(img_dir, name, b), True, fold))
        for _ in range(n_per):
            parts = lines[i].split()
            i += 1
            n1, a, n2, b = parts[0], int(parts[1]), parts[2], int(parts[3])
            pairs.append(PairItem(_lfw_path(img_dir, n1, a), _lfw_path(img_dir, n2, b), False, fold))
    if max_pairs is not None:
        same = [p for p in pairs if p.same]
        diff = [p for p in pairs if not p.same]
        n_each = max(1, int(max_pairs) // 2)
        pairs = same[:n_each] + diff[:n_each]
    return pairs


class LFWPairsDataset(Dataset):
    def __init__(self, pairs: List[PairItem], image_size: int = 112):
        self.pairs = pairs
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        a = self.tf(Image.open(p.path_a).convert("RGB"))
        b = self.tf(Image.open(p.path_b).convert("RGB"))
        return a, b, int(p.same)
