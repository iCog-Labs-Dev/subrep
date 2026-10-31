from __future__ import annotations

import json
from pathlib import Path
import statistics
import time

from env.minecraft.digest import digest_for, observe
from env.minecraft.reset_manager import ResetManager


ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = ROOT / "minecraft" / "logs" / "reset_proof_dusk_v1.jsonl"


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    manager = ResetManager()
    digests = []
    timings = []
    failure = None
    try:
        with LOG_PATH.open("a", encoding="utf-8") as log_file:
            for iteration in range(1, 101):
                timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                try:
                    reset_seconds = manager.reset()
                    digest = digest_for(observe(manager))
                    record = {
                        "iteration": iteration,
                        "digest": digest,
                        "reset_seconds": reset_seconds,
                        "timestamp_utc": timestamp,
                    }
                    digests.append((iteration, digest))
                    timings.append(reset_seconds)
                except Exception as error:
                    record = {
                        "iteration": iteration,
                        "error": str(error),
                        "timestamp_utc": timestamp,
                    }
                    log_file.write(json.dumps(record, sort_keys=True) + "\n")
                    log_file.flush()
                    failure = error
                    raise
                log_file.write(json.dumps(record, sort_keys=True) + "\n")
                log_file.flush()
    finally:
        manager.stop()
        _print_summary(digests, timings)
    if failure is not None:
        raise failure


def _print_summary(digests, timings):
    print("Successful digests: {}".format(len(digests)))
    if digests:
        print("Digest: {}".format(digests[0][1]))
        identical = sum(1 for _, digest in digests if digest == digests[0][1])
        print("Identical digests: {}/{}".format(identical, len(digests)))
    if timings:
        print("Reset seconds: mean={:.3f}, min={:.3f}, max={:.3f}, total={:.3f}".format(
            statistics.mean(timings), min(timings), max(timings), sum(timings)
        ))
    if digests and any(digest != digests[0][1] for _, digest in digests):
        print("Digest mismatches:")
        for iteration, digest in digests:
            if digest != digests[0][1]:
                print("{}: {}".format(iteration, digest))


if __name__ == "__main__":
    main()
