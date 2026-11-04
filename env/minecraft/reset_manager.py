from __future__ import annotations

import hashlib
import json
import os
import select
import fcntl
from pathlib import Path
import shutil
import subprocess
import sys
import time


ALLOWED_CONSOLE_COMMANDS = frozenset(
    {
        "stop",
        "tick freeze",
        "time set 12000",
        "time query daytime",
        "save-all",
        "execute as @e run data get entity @s",
        "data get block 55 64 0",
        "data get block 32 64 -8",
        "data get block 32 64 -7",
        "data get block 32 64 -6",
        "data get block 32 64 -5",
        "data get block 32 64 5",
        "data get block 32 64 6",
        "data get block 32 64 7",
        "data get block 32 64 8",
        "data get block 48 64 -8",
        "data get block 48 64 -7",
        "data get block 48 64 -6",
        "data get block 48 64 -5",
        "data get block 48 64 5",
        "data get block 48 64 6",
        "data get block 48 64 7",
        "data get block 48 64 8",
        "data get block 48 64 -1",
        "data get block 48 64 0",
        "data get block 48 64 1",
        "data get block 34 64 -10",
        "data get block 35 64 -10",
        "data get block 36 64 -10",
        "data get block 44 64 -10",
        "data get block 45 64 -10",
        "data get block 46 64 -10",
        "data get block 34 64 10",
        "data get block 35 64 10",
        "data get block 36 64 10",
        "data get block 44 64 10",
        "data get block 45 64 10",
        "data get block 46 64 10",
        "data get block 32 64 0",
        "data get block 40 64 0",
    }
)
BLOCK_PROBE_POSITIONS = (
    ((32, 64, -8), "air"), ((32, 64, -7), "air"),
    ((32, 64, -6), "air"), ((32, 64, -5), "air"),
    ((32, 64, 5), "air"), ((32, 64, 6), "air"),
    ((32, 64, 7), "air"), ((32, 64, 8), "air"),
    ((48, 64, -8), "air"), ((48, 64, -7), "air"),
    ((48, 64, -6), "air"), ((48, 64, -5), "air"),
    ((48, 64, 5), "air"), ((48, 64, 6), "air"),
    ((48, 64, 7), "air"), ((48, 64, 8), "air"),
    ((48, 64, -1), "air"), ((48, 64, 0), "torch"),
    ((48, 64, 1), "air"), ((34, 64, -10), "air"),
    ((35, 64, -10), "air"), ((36, 64, -10), "torch"),
    ((44, 64, -10), "torch"), ((45, 64, -10), "air"),
    ((46, 64, -10), "air"), ((34, 64, 10), "air"),
    ((35, 64, 10), "air"), ((36, 64, 10), "torch"),
    ((44, 64, 10), "torch"), ((45, 64, 10), "air"),
    ((46, 64, 10), "air"), ((32, 64, 0), "torch"),
    ((40, 64, 0), "torch"),
)
BLOCK_PROBE_COMMANDS = frozenset(
    "execute if block {} {} {} minecraft:{} run say block_id=minecraft:{}".format(
        x, y, z, block_id, block_id
    )
    for (x, y, z), block_id in BLOCK_PROBE_POSITIONS
)
ALLOWED_CONSOLE_COMMANDS = ALLOWED_CONSOLE_COMMANDS | BLOCK_PROBE_COMMANDS

AREAS_PATH = Path(__file__).resolve().parents[2] / "minecraft" / "world" / "areas.json"
with AREAS_PATH.open(encoding="utf-8") as _areas_file:
    _areas = json.load(_areas_file)
_dynamic_probe_positions = tuple(
    dict.fromkeys(
        tuple(position)
        for group in (
            _areas["perimeter"]["gap_positions"],
            _areas["torch_positions"],
            _areas["light_sample_positions"],
            [_areas["settlement"]["delivery_chest"]],
        )
        for position in group
    )
)
_torch_positions = {tuple(position) for position in _areas["torch_positions"]}
_closed_gate_positions = {
    tuple(position) for position in _areas["perimeter"]["closed_gate_positions"]
}
_chest_position = tuple(_areas["settlement"]["delivery_chest"])
BLOCK_PROBE_POSITIONS = tuple(
    (
        position,
        "chest" if position == _chest_position else
        "torch" if position in _torch_positions else
        "oak_fence_gate" if position in _closed_gate_positions else "air",
    )
    for position in _dynamic_probe_positions
)
BLOCK_PROBE_COMMANDS = frozenset(
    "execute if block {} {} {} minecraft:{}".format(
        *position,
        dict(BLOCK_PROBE_POSITIONS)[position],
    )
    for position in _dynamic_probe_positions
)
ALLOWED_CONSOLE_COMMANDS = frozenset(
    {
        "stop",
        "tick freeze",
        "time set 12000",
        "time query daytime",
        "save-all",
        "execute as @e run data get entity @s",
        "data get block {} {} {}".format(*_chest_position),
    }
) | BLOCK_PROBE_COMMANDS | BLOCK_PROBE_COMMANDS

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SERVER_DIR = DEFAULT_ROOT / "minecraft" / "server"
DEFAULT_SNAPSHOT_DIR = DEFAULT_SERVER_DIR / "snapshots" / "dusk_v1"
DEFAULT_MANIFEST_PATH = DEFAULT_ROOT / "minecraft" / "world" / "dusk_v1_snapshot_manifest.json"
DEFAULT_BOT_DIR = DEFAULT_ROOT / "minecraft" / "bot"
DEFAULT_BOT_ENTRY = DEFAULT_BOT_DIR / "index.js"
TIME_READY_TIMEOUT = 120



def snapshot_hashes(snapshot_dir):
    """Return file hashes and their canonical top-level hash."""
    snapshot_dir = Path(snapshot_dir)
    if snapshot_dir.is_symlink():
        raise ValueError("snapshot directory must not be a symlink")
    snapshot_dir = snapshot_dir.resolve()
    if not snapshot_dir.is_dir():
        raise ValueError("snapshot directory does not exist: {}".format(snapshot_dir))

    files = {}
    for path in sorted(snapshot_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError("snapshot contains a symlink: {}".format(path))
        if not path.is_file() or path.name == "session.lock":
            continue
        relative_path = path.relative_to(snapshot_dir).as_posix()
        digest = hashlib.sha256()
        with path.open("rb") as snapshot_file:
            for chunk in iter(lambda: snapshot_file.read(1024 * 1024), b""):
                digest.update(chunk)
        files[relative_path] = digest.hexdigest()

    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return files, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def make_manifest(snapshot_dir, manifest_path):
    snapshot_dir = Path(snapshot_dir)
    manifest_path = Path(manifest_path).resolve()
    files, top_hash = snapshot_hashes(snapshot_dir)
    manifest = {
        "snapshot_name": "dusk_v1",
        "source_files": [
            "minecraft/world/dusk_v1_build_commands.txt",
            "minecraft/world/areas.json",
        ],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": files,
        "sha256": top_hash,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")


def verify_snapshot(snapshot_dir, manifest_path):
    snapshot_dir = Path(snapshot_dir).resolve()
    manifest_path = Path(manifest_path).resolve()
    try:
        with manifest_path.open(encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
    except (OSError, ValueError) as error:
        raise ValueError("snapshot manifest is unreadable: {}".format(error))

    if not isinstance(manifest, dict):
        raise ValueError("snapshot manifest must be a JSON object")
    expected_files = manifest.get("files")
    expected_hash = manifest.get("sha256")
    if not isinstance(expected_files, dict) or not isinstance(expected_hash, str):
        raise ValueError("snapshot manifest is missing files or sha256")

    if any(
        not isinstance(path, str)
        or Path(path).is_absolute()
        or ".." in Path(path).parts
        for path in expected_files
    ):
        raise ValueError("snapshot manifest contains an unsafe path")
    actual_files, actual_hash = snapshot_hashes(snapshot_dir)
    if actual_files != expected_files:
        missing = sorted(set(expected_files) - set(actual_files))
        extra = sorted(set(actual_files) - set(expected_files))
        mismatched = sorted(
            path
            for path in set(expected_files) & set(actual_files)
            if expected_files[path] != actual_files[path]
        )
        details = []
        if missing:
            details.append("missing: {}".format(", ".join(missing)))
        if extra:
            details.append("extra: {}".format(", ".join(extra)))
        if mismatched:
            details.append("mismatched: {}".format(", ".join(mismatched)))
        raise ValueError("snapshot files do not match ({})".format("; ".join(details)))
    if actual_hash != expected_hash:
        raise ValueError(
            "snapshot top-level SHA-256 mismatch: expected {}, got {}".format(
                expected_hash, actual_hash
            )
        )
    return len(actual_files), actual_hash


class ResetManager:
    def __init__(
        self,
        server_dir=DEFAULT_SERVER_DIR,
        snapshot_dir=DEFAULT_SNAPSHOT_DIR,
        manifest_path=DEFAULT_MANIFEST_PATH,
        bot_dir=DEFAULT_BOT_DIR,
        bot_entry=DEFAULT_BOT_ENTRY,
    ):
        self.server_dir = Path(server_dir).resolve()
        self.snapshot_dir = Path(snapshot_dir).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.bot_dir = Path(bot_dir).resolve()
        self.bot_entry = Path(bot_entry).resolve()
        self.live_world_dir = self.server_dir / "world"
        try:
            self.live_world_dir.relative_to(self.server_dir)
        except ValueError:
            raise ValueError("live world directory must be inside server directory")
        if self.snapshot_dir == self.live_world_dir:
            raise ValueError("snapshot directory cannot equal live world directory")
        if self.live_world_dir.is_symlink():
            raise ValueError("live world directory must not be a symlink")
        self._process = None
        self._bot_process = None
        self._lock_file = None

    def send_command(self, cmd):
        if cmd not in ALLOWED_CONSOLE_COMMANDS:
            raise ValueError("command is not allowlisted: {!r}".format(cmd))
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("server process is not running")
        self._process.stdin.write(cmd + "\n")
        self._process.stdin.flush()

    def _read_until(self, predicate, timeout):
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("server process is not running")
        deadline = time.monotonic() + timeout
        stdout = self._process.stdout
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            ready, _, _ = select.select([stdout], [], [], remaining)
            if not ready:
                break
            line = stdout.readline()
            if not line:
                if self._process.poll() is not None:
                    raise RuntimeError("server exited before expected output")
                break
            if predicate(line):
                return line
        raise TimeoutError("timed out waiting for server output")

    def _start_bot(self):
        if self._bot_process is not None and self._bot_process.poll() is None:
            return
        token = os.environ.get("SUBREP_BOT_TOKEN")
        if not token:
            raise RuntimeError("SUBREP_BOT_TOKEN is required to start the bot")
        self._bot_process = subprocess.Popen(
            ["node", str(self.bot_entry)],
            cwd=str(self.bot_dir),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=dict(os.environ),
        )
        deadline = time.monotonic() + 60
        stdout = self._bot_process.stdout
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            ready, _, _ = select.select([stdout], [], [], remaining)
            if not ready:
                break
            line = stdout.readline()
            if "Bot spawned at world spawn" in line:
                return
            if self._bot_process.poll() is not None:
                raise RuntimeError("bot exited before spawning")
        self._stop_bot()
        raise TimeoutError("timed out waiting for bot spawn")

    def _stop_bot(self):
        if self._bot_process is None:
            return
        if self._bot_process.poll() is None:
            self._bot_process.terminate()
            try:
                self._bot_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._bot_process.kill()
                self._bot_process.wait()
        self._bot_process = None

    def start(self):
        if self._process is not None and self._process.poll() is None:
            return
        self._process = subprocess.Popen(
            ["./start.sh"],
            cwd=str(self.server_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._read_until(lambda line: "Done (" in line, 180)

    def stop(self):
        self._stop_bot()
        if self._process is None or self._process.poll() is not None:
            return
        self.send_command("stop")
        try:
            self._process.wait(timeout=90)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        finally:
            self._process = None

    def reset(self):
        started_at = time.monotonic()
        lock_path = self.server_dir / ".subrep-reset.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file = lock_path.open("a+")
        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self._lock_file.close()
            self._lock_file = None
            raise RuntimeError("another reset is already in progress") from error
        if self._process is not None and self._process.poll() is None:
            self.stop()
        try:
            verify_snapshot(self.snapshot_dir, self.manifest_path)
            if self.live_world_dir.is_symlink():
                raise ValueError("live world directory must not be a symlink")
            if self.live_world_dir.exists():
                shutil.rmtree(str(self.live_world_dir))
            shutil.copytree(
                str(self.snapshot_dir),
                str(self.live_world_dir),
                ignore=shutil.ignore_patterns("session.lock"),
                symlinks=False,
            )
            self.start()
            self.send_command("tick freeze")
            self.send_command("time set 12000")
            self.send_command("time query daytime")
            self._read_until(
                lambda line: "The time is 12000" in line,
                TIME_READY_TIMEOUT,
            )
            self._start_bot()
            duration = time.monotonic() - started_at
            print("Reset completed in {:.3f} seconds".format(duration))
            return duration
        finally:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()
            self._lock_file = None


def _option_value(arguments, name, default):
    if name not in arguments:
        return default
    index = arguments.index(name)
    if index + 1 >= len(arguments):
        raise ValueError("missing value for {}".format(name))
    return arguments[index + 1]


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        raise ValueError("expected make-manifest, verify, or reset")
    command = arguments.pop(0)
    if command in ("make-manifest", "verify"):
        if len(arguments) != 2:
            raise ValueError("{} requires snapshot_dir and manifest_path".format(command))
        snapshot_dir, manifest_path = arguments
        if command == "make-manifest":
            make_manifest(snapshot_dir, manifest_path)
            return 0
        count, top_hash = verify_snapshot(snapshot_dir, manifest_path)
        print("OK: {} files, top SHA-256 {}".format(count, top_hash))
        return 0
    if command == "reset":
        manager = ResetManager(
            server_dir=_option_value(arguments, "--server-dir", DEFAULT_SERVER_DIR),
            snapshot_dir=_option_value(arguments, "--snapshot-dir", DEFAULT_SNAPSHOT_DIR),
            manifest_path=_option_value(arguments, "--manifest-path", DEFAULT_MANIFEST_PATH),
        )
        manager.reset()
        return 0
    raise ValueError("unknown command: {}".format(command))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, TimeoutError, ValueError, OSError) as error:
        print("Error: {}".format(error), file=sys.stderr)
        sys.exit(1)
