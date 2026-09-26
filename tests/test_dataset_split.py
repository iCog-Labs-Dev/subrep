"""Tests for generator/dataset_split.py -- the train/val/test partitioning logic."""

import pytest

from generator.dataset_split import (
    compute_split_assignment,
    save_split_manifest,
    load_split_manifest,
    apply_split_manifest,
)


def test_compute_split_assignment_covers_every_file_exactly_once():
    files = [f"episode_{i}.npz" for i in range(100)]
    assignment = compute_split_assignment(files, train_frac=0.75, val_frac=0.125, test_frac=0.125, seed=1)

    assert set(assignment.keys()) == set(files)
    assert set(assignment.values()) <= {"train", "val", "test"}


def test_compute_split_assignment_is_deterministic_given_same_seed():
    files = [f"episode_{i}.npz" for i in range(50)]
    assignment_a = compute_split_assignment(files, seed=7)
    assignment_b = compute_split_assignment(files, seed=7)
    assert assignment_a == assignment_b


def test_compute_split_assignment_rejects_bad_fractions():
    files = [f"episode_{i}.npz" for i in range(10)]
    with pytest.raises(ValueError):
        compute_split_assignment(files, train_frac=0.5, val_frac=0.3, test_frac=0.3, seed=1)


def test_compute_split_assignment_rejects_too_small_dataset():
    files = [f"episode_{i}.npz" for i in range(3)]
    with pytest.raises(ValueError):
        compute_split_assignment(files, train_frac=0.75, val_frac=0.125, test_frac=0.125, seed=1)


def test_compute_split_assignment_rejects_duplicate_basenames():
    files = ["dir_a/episode_0.npz", "dir_b/episode_0.npz"] + [f"episode_{i}.npz" for i in range(1, 10)]
    with pytest.raises(ValueError):
        compute_split_assignment(files, seed=1)


def test_compute_split_assignment_covers_every_file_exactly_once_larger():
    files = [f"episode_{i}.npz" for i in range(40)]
    assignment = compute_split_assignment(files, seed=3)

    assert set(assignment.keys()) == set(files)
    assert set(assignment.values()) <= {"train", "val", "test"}


def test_manifest_round_trip_matches_original_assignment(tmp_path):
    files = [f"episode_{i}.npz" for i in range(40)]
    assignment = compute_split_assignment(files, seed=5)

    manifest_path = tmp_path / "manifest.json"
    save_split_manifest(assignment, 0.75, 0.125, 0.125, seed=5, path=manifest_path)

    loaded = load_split_manifest(manifest_path)
    assert loaded["assignment"] == assignment
    assert loaded["seed"] == 5


def test_apply_split_manifest_matches_records_to_saved_labels(tmp_path):
    files = [f"/some/dir/episode_{i}.npz" for i in range(20)]
    records = [f"record_{i}" for i in range(20)]

    assignment = compute_split_assignment(files, seed=9)
    manifest_path = tmp_path / "manifest.json"
    save_split_manifest(assignment, 0.75, 0.125, 0.125, seed=9, path=manifest_path)
    manifest = load_split_manifest(manifest_path)

    split = apply_split_manifest(files, records, manifest)

    for basename_path, record in zip(files, records):
        import os
        label = assignment[os.path.basename(basename_path)]
        bucket = {"train": split.train, "val": split.val, "test": split.test}[label]
        assert record in bucket


def test_apply_split_manifest_raises_on_unknown_file(tmp_path):
    files = [f"episode_{i}.npz" for i in range(20)]
    assignment = compute_split_assignment(files, seed=2)
    manifest_path = tmp_path / "manifest.json"
    save_split_manifest(assignment, 0.75, 0.125, 0.125, seed=2, path=manifest_path)
    manifest = load_split_manifest(manifest_path)

    new_files = files + ["episode_never_seen.npz"]
    new_records = list(range(21))
    with pytest.raises(ValueError):
        apply_split_manifest(new_files, new_records, manifest)


def test_load_split_manifest_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_split_manifest(tmp_path / "does_not_exist.json")