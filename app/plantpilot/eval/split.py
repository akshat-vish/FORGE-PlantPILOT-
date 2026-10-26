"""Reproducible dataset split with leakage safeguards.

Fixed seed, stratified by Machine failure to preserve class balance.
No duplicate records across splits (UDI is unique identifier).
Labels kept out of agent-facing interface; only used after predictions.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set

DATASET_PATH = (Path(__file__).resolve().parent.parent.parent / "ai4i2020.csv") if (Path(__file__).resolve().parent.parent.parent / "ai4i2020.csv").exists() else Path(__file__).resolve().parent.parent.parent.parent / "ai4i2020.csv"
# Fallback: if not at parent-2, use project root
SEED = 42


def load_dataset(path: Path = DATASET_PATH) -> List[Dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def split_dataset(
    rows: List[Dict[str, Any]],
    train_fraction: float = 0.7,
    dev_fraction: float = 0.15,
    seed: int = SEED,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Stratified split by Machine failure (0/1). Fixed seed."""
    # Separate by label
    pos = [r for r in rows if int(r.get("Machine failure", 0)) == 1]
    neg = [r for r in rows if int(r.get("Machine failure", 0)) == 0]

    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)

    # Stratified proportions
    def split_group(group, tf, df):
        n = len(group)
        train_n = int(n * tf)
        dev_n = int(n * df)
        return group[:train_n], group[train_n:train_n+dev_n], group[train_n+dev_n:]

    pos_t, pos_d, pos_e = split_group(pos, train_fraction, dev_fraction)
    neg_t, neg_d, neg_e = split_group(neg, train_fraction, dev_fraction)

    train = pos_t + neg_t
    dev = pos_d + neg_d
    eval_ = pos_e + neg_e  # "eval" = held-out

    # Check no overlap (UDI unique)
    train_udis = {r["UDI"] for r in train}
    dev_udis = {r["UDI"] for r in dev}
    eval_udis = {r["UDI"] for r in eval_}
    assert len(train_udis & dev_udis) == 0, "Leakage: train/dev overlap"
    assert len(train_udis & eval_udis) == 0, "Leakage: train/eval overlap"
    assert len(dev_udis & eval_udis) == 0, "Leakage: dev/eval overlap"

    # Re-shuffle within splits for randomness (preserves stratification)
    rng.shuffle(train); rng.shuffle(dev); rng.shuffle(eval_)
    return train, dev, eval_


def get_label_distribution(rows: List[Dict], label_col: str = "Machine failure") -> Dict[str, int]:
    dist = {}
    for r in rows:
        v = int(r.get(label_col, 0))
        dist[v] = dist.get(v, 0) + 1
    return dist
