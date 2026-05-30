from __future__ import annotations

import subprocess
import sys


def test_diagnose_mdn_training_cli_succeeds():
    result = subprocess.run(
        [sys.executable, "-m", "generator.diagnose_mdn_training", "--synthetic", "--epochs", "5", "--seed", "42"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "MDN Offline Training Diagnostic" in result.stdout
    assert "[PASS] offline record uses recorded weights" in result.stdout
