from __future__ import annotations

from pathlib import Path

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
