"""REPORT.md and README.md, the committed outputs, against the paper's record."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def metric(report: str, name: str) -> float:
    match = re.search(rf"\| {re.escape(name)} \| ([\d.]+)% \|", report)
    assert match, f"{name} row missing from REPORT.md"
    return float(match.group(1))


@pytest.mark.parametrize("path", ["README.md", "REPORT.md"])
def test_paper_is_credited_to_its_first_author(targets, path):
    text = (ROOT / path).read_text(encoding="utf-8")
    assert f"{targets['first_author']} et al." in text


def test_cross_resolution_rank1_beats_chance(report):
    gallery = int(re.search(r"Gallery IDs=(\d+)", report).group(1))
    assert metric(report, "CelebA LR→HR Rank-1") > 100.0 / gallery


def test_cross_resolution_rank1_reaches_table_ii(report, targets):
    """28x28 probes sit between the 24x24 and 32x32 columns, so 24x24 is the floor."""
    assert metric(report, "CelebA LR→HR Rank-1") >= targets["scface_cross_resolution"]["probe_24"]
