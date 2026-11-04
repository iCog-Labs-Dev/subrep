from __future__ import annotations

from pathlib import Path
import fcntl

import pytest

from env.minecraft.reset_manager import ResetManager, make_manifest, verify_snapshot


class FakeStdin:
    def write(self, value: str) -> None:
        return

    def flush(self) -> None:
        return


class FakeProcess:
    stdin = FakeStdin()

    def poll(self):
        return None


def test_non_allowlisted_command_is_rejected(tmp_path: Path) -> None:
    manager = ResetManager(tmp_path / "server", tmp_path / "snapshot", tmp_path / "manifest.json")
    manager._process = FakeProcess()
    with pytest.raises(ValueError, match="not allowlisted"):
        manager.send_command("say forbidden")


def test_manifest_verifies_and_detects_tampering(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    data_file = snapshot / "level.dat"
    data_file.write_bytes(b"stable snapshot")
    manifest = tmp_path / "manifest.json"
    make_manifest(snapshot, manifest)
    assert verify_snapshot(snapshot, manifest)[0] == 1
    data_file.write_bytes(b"tampered snapshot")
    with pytest.raises(ValueError, match="do not match"):
        verify_snapshot(snapshot, manifest)


def test_manifest_rejects_symlinked_snapshot_file(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (snapshot / "level.dat").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        make_manifest(snapshot, tmp_path / "manifest.json")


def test_manifest_rejects_traversal_entry(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "level.dat").write_bytes(b"stable")
    manifest = tmp_path / "manifest.json"
    make_manifest(snapshot, manifest)
    data = manifest.read_text(encoding="utf-8").replace('"level.dat"', '"../outside"')
    manifest.write_text(data, encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe path"):
        verify_snapshot(snapshot, manifest)


def test_snapshot_directory_symlink_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "level.dat").write_bytes(b"stable")
    link = tmp_path / "snapshot"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        make_manifest(link, tmp_path / "manifest.json")


def test_reset_lock_rejects_concurrent_reset(tmp_path: Path) -> None:
    server = tmp_path / "server"
    server.mkdir()
    lock = server / ".subrep-reset.lock"
    held = lock.open("a+")
    fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        manager = ResetManager(server, tmp_path / "snapshot", tmp_path / "manifest.json")
        with pytest.raises(RuntimeError, match="already in progress"):
            manager.reset()
    finally:
        fcntl.flock(held.fileno(), fcntl.LOCK_UN)
        held.close()
