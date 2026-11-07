"""Strict raw_mixed-only loader and context-grouped persistent splitting."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np

from generator.policy_registry import POLICY_NAMES, policy_index

SPLIT_NAMES = {"train", "val", "test"}


def load_records(data_dir: str = "data/raw_mixed") -> list[dict]:
    paths = sorted(Path(data_dir).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"No .npz records in {data_dir}; collect fresh raw_mixed data first"
        )

    records = []
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            needed = {"obs", "payoff", "motives", "skill_id", "context_seed"}
            if not needed.issubset(z.files):
                raise ValueError(
                    f"{path}: missing {sorted(needed - set(z.files))}; "
                    "recollect fresh raw_mixed data"
                )

            obs = np.asarray(z["obs"], dtype=np.float32).reshape(-1)
            motives = np.asarray(z["motives"], dtype=np.float32).reshape(-1)
            if obs.shape != (8,) or motives.shape != (2,):
                raise ValueError(f"{path}: expected obs=(8,), motives=(2,)")

            name = str(z["skill_id"].item())
            seed = int(z["context_seed"])
            payoff = float(z["payoff"])
            if (
                not np.isfinite(obs).all()
                or not np.isfinite(motives).all()
                or not np.isfinite(payoff)
            ):
                raise ValueError(f"{path}: nonfinite outcome")

            records.append(
                dict(
                    file=path.name,
                    context_id=f"seed:{seed}",
                    seed=seed,
                    policy=name,
                    policy_id=policy_index(name),
                    obs=obs,
                    payoff=payoff,
                    motives=motives,
                    terminated=bool(z["terminated"]) if "terminated" in z.files else None,
                )
            )

    seen = set()
    for record in records:
        key = (record["context_id"], record["policy"])
        if key in seen:
            raise ValueError(
                f"Duplicate context/policy {key}; use a fresh empty collection directory"
            )
        seen.add(key)

    groups = group_contexts(records, complete=False)
    for context, rows in groups.items():
        if not all(
            np.allclose(rows[0]["obs"], row["obs"], rtol=1e-6, atol=1e-6)
            for row in rows
        ):
            raise ValueError(f"Inconsistent starting observations for {context}")
    return records


def group_contexts(records: list[dict], complete: bool = True) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(record["context_id"], []).append(record)

    if complete:
        expected = set(POLICY_NAMES)
        for context, rows in groups.items():
            got = {row["policy"] for row in rows}
            if got != expected or len(rows) != len(expected):
                raise ValueError(
                    f"{context}: incomplete candidate set: missing {sorted(expected - got)}"
                )
    return groups


def _json_value(value):
    """Convert NumPy values into deterministic JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def dataset_signature(records: list[dict]) -> str:
    """Hash record identity and available content so changed records are detected."""
    canonical_records = []
    for record in sorted(
        records,
        key=lambda row: (str(row.get("context_id")), str(row.get("policy"))),
    ):
        canonical_records.append(
            {
                key: _json_value(record.get(key))
                for key in (
                    "file", "context_id", "seed", "policy", "policy_id",
                    "obs", "payoff", "motives", "terminated",
                )
            }
        )

    encoded = json.dumps(
        canonical_records,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def split_signature(manifest: dict) -> str:
    """Hash the dataset identity and exact context-to-split assignments."""
    identity = {
        "dataset_signature": manifest["dataset_signature"],
        "assignment": manifest["assignment"],
        "fractions": manifest["fractions"],
        "seed": manifest["seed"],
        "policy_names": manifest["policy_names"],
    }
    encoded = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_manifest(records: list[dict], manifest: dict) -> dict:
    """Validate dataset contents, assignments, and the stored split identity."""
    required = {
        "seed", "fractions", "dataset_signature", "assignment",
        "policy_names", "split_signature",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"Split manifest is missing required fields: {sorted(missing)}")

    actual_dataset_signature = dataset_signature(records)
    if manifest["dataset_signature"] != actual_dataset_signature:
        raise ValueError(
            "Split manifest dataset signature does not match raw_mixed records. "
            "Do not evaluate this dataset with the saved split."
        )

    contexts = set(group_contexts(records, complete=True))
    assignment = manifest["assignment"]
    if not isinstance(assignment, dict) or set(assignment) != contexts:
        raise ValueError(
            "Split manifest context assignments do not match raw_mixed dataset"
        )
    if any(split not in SPLIT_NAMES for split in assignment.values()):
        raise ValueError("Split manifest contains an unknown split name")
    if manifest["policy_names"] != list(POLICY_NAMES):
        raise ValueError("Split manifest policy registry mismatch")

    fractions = manifest["fractions"]
    if (
        not isinstance(fractions, list)
        or len(fractions) != 3
        or any(not isinstance(value, (int, float)) for value in fractions)
        or not np.isclose(sum(fractions), 1.0)
    ):
        raise ValueError("Split manifest contains invalid split fractions")

    if set(assignment.values()) != SPLIT_NAMES:
        raise ValueError("Split manifest must contain nonempty train, val, and test splits")

    if manifest["split_signature"] != split_signature(manifest):
        raise ValueError("Split manifest identity is invalid or has been modified")
    return manifest


def load_manifest(records: list[dict], path: str = "data/mixed_split_manifest.json") -> dict:
    """Load and validate an existing training manifest. Never create one."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Required training split manifest not found: {manifest_path}. "
            "Run multi-policy training first; evaluation will not create a split."
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read split manifest {manifest_path}: {exc}") from exc
    return validate_manifest(records, manifest)


def load_or_create_manifest(
    records: list[dict],
    path: str = "data/mixed_split_manifest.json",
    seed: int = 42,
    fractions=(0.75, 0.125, 0.125),
) -> dict:
    """Training helper: reuse a valid manifest or create the initial split."""
    groups = group_contexts(records, complete=True)
    keys = sorted(groups)
    manifest_path = Path(path)

    if manifest_path.exists():
        return load_manifest(records, str(manifest_path))

    if len(keys) < 8:
        raise ValueError("At least 8 complete contexts required for train/val/test")
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1.0):
        raise ValueError("Split fractions must contain train/val/test and sum to 1")

    random.Random(seed).shuffle(keys)
    count = len(keys)
    n_train = round(count * fractions[0])
    n_val = round(count * fractions[1])
    if min(n_train, n_val, count - n_train - n_val) < 1:
        raise ValueError("Empty train/validation/test split")

    assignment = {
        key: (
            "train" if index < n_train
            else "val" if index < n_train + n_val
            else "test"
        )
        for index, key in enumerate(keys)
    }
    manifest = {
        "seed": seed,
        "fractions": list(fractions),
        "dataset_signature": dataset_signature(records),
        "assignment": assignment,
        "policy_names": list(POLICY_NAMES),
    }
    manifest["split_signature"] = split_signature(manifest)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def partition(records: list[dict], manifest: dict) -> dict[str, list[dict]]:
    """Partition records using an already validated manifest."""
    buckets = {name: [] for name in SPLIT_NAMES}
    assignment = manifest.get("assignment", {})
    for record in records:
        context = record["context_id"]
        if context not in assignment:
            raise ValueError(f"Context {context} is absent from the split manifest")
        split = assignment[context]
        if split not in buckets:
            raise ValueError(f"Unknown split assignment {split!r}")
        buckets[split].append(record)
    return buckets