"""
Train/validation/test splitting for SkillGenerator rollout data.

This module is the SINGLE shared place that decides how collected .npz
rollout records get divided into train/val/test groups, AND it persists
that decision to a manifest file on disk.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TypeVar

T = TypeVar("T")

DEFAULT_MANIFEST_PATH = "data/generator_split_manifest.json"


@dataclass(frozen=True)
class DatasetSplit:
    """Container holding the three non-overlapping groups produced by a split."""

    train: list
    val: list
    test: list

    def __post_init__(self) -> None:
        train_ids = {id(r) for r in self.train}
        val_ids = {id(r) for r in self.val}
        test_ids = {id(r) for r in self.test}
        if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
            raise ValueError("Dataset split produced overlapping records.")


def _validate_fracs(train_frac: float, val_frac: float, test_frac: float) -> None:
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train_frac + val_frac + test_frac must sum to 1.0, got {total}")


def split_dataset(
    records: Sequence[T],
    train_frac: float = 0.75,
    val_frac: float = 0.125,
    test_frac: float = 0.125,
    seed: int = 42,
) -> DatasetSplit:
    
   # Split `records` into train/val/test groups by index (in-memory only,
   # no file identity, no manifest). 

    _validate_fracs(train_frac, val_frac, test_frac)
    if len(records) == 0:
        raise ValueError("Cannot split an empty dataset.")

    indices = list(range(len(records)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_total = len(indices)
    n_train = int(round(n_total * train_frac))
    n_val = int(round(n_total * val_frac))
    n_test = n_total - n_train - n_val

    if n_train == 0 or n_val == 0 or n_test == 0:
        raise ValueError(
            f"Dataset too small to split into non-empty train/val/test "
            f"groups (got {n_total} records -> train={n_train}, "
            f"val={n_val}, test={n_test}). Collect more rollout data."
        )

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return DatasetSplit(
        train=[records[i] for i in train_idx],
        val=[records[i] for i in val_idx],
        test=[records[i] for i in test_idx],
    )


def compute_split_assignment(
    file_paths: Sequence[str],
    train_frac: float = 0.75,
    val_frac: float = 0.125,
    test_frac: float = 0.125,
    seed: int = 42,
) -> dict[str, str]:
   
    _validate_fracs(train_frac, val_frac, test_frac)
    if len(file_paths) == 0:
        raise ValueError("Cannot split an empty file list.")

    basenames = [os.path.basename(p) for p in file_paths]
    if len(set(basenames)) != len(basenames):
        raise ValueError("Duplicate file basenames found; cannot build a unique manifest.")

    indices = list(range(len(basenames)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_total = len(indices)
    n_train = int(round(n_total * train_frac))
    n_val = int(round(n_total * val_frac))
    n_test = n_total - n_train - n_val

    if n_train == 0 or n_val == 0 or n_test == 0:
        raise ValueError(
            f"Dataset too small to split into non-empty train/val/test "
            f"groups (got {n_total} files -> train={n_train}, "
            f"val={n_val}, test={n_test}). Collect more rollout data."
        )

    assignment: dict[str, str] = {}
    for i in indices[:n_train]:
        assignment[basenames[i]] = "train"
    for i in indices[n_train:n_train + n_val]:
        assignment[basenames[i]] = "val"
    for i in indices[n_train + n_val:]:
        assignment[basenames[i]] = "test"
    return assignment


def save_split_manifest(
    assignment: dict[str, str],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    path: str | Path = DEFAULT_MANIFEST_PATH,
) -> None:
    """Persist a split assignment (and the settings used to produce it) to JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "assignment": assignment,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_split_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> dict:
    """Load a previously saved split manifest. Raises FileNotFoundError if missing."""
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No split manifest found at {manifest_path}. "
            f"Run generator.train_generator first to create one."
        )
    with open(manifest_path) as f:
        return json.load(f)


def apply_split_manifest(
    file_paths: Sequence[str],
    records: Sequence[T],
    manifest: dict,
) -> DatasetSplit:
   
    if len(file_paths) != len(records):
        raise ValueError(
            f"file_paths ({len(file_paths)}) and records ({len(records)}) "
            f"must be the same length and in the same order."
        )
    assignment = manifest["assignment"]
    buckets: dict[str, list] = {"train": [], "val": [], "test": []}
    missing = []
    for fp, rec in zip(file_paths, records):
        basename = os.path.basename(fp)
        label = assignment.get(basename)
        if label is None:
            missing.append(basename)
            continue
        buckets[label].append(rec)

    if missing:
        raise ValueError(
            f"{len(missing)} file(s) in this data directory are not present "
            f"in the split manifest (e.g. {missing[:3]}). This usually means "
            f"the data was regenerated/added to after the manifest was "
            f"created. Re-run train_generator.py to produce a fresh manifest "
            f"covering the current data, or point --data-dir at the exact "
            f"data used for training."
        )

    return DatasetSplit(train=buckets["train"], val=buckets["val"], test=buckets["test"])