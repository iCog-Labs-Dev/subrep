from __future__ import annotations

from pathlib import Path


BOT_DIR = Path(__file__).resolve().parents[1] / "minecraft" / "bot"


def test_bot_source_has_no_chat_or_whisper_calls() -> None:
    source = "\n".join(path.read_text(encoding="utf-8") for path in BOT_DIR.glob("*.js"))
    assert "bot.chat" not in source
    assert "bot.whisper" not in source


def test_bot_binds_to_localhost_only() -> None:
    source = "\n".join(path.read_text(encoding="utf-8") for path in BOT_DIR.glob("*.js"))
    assert "0.0.0.0" not in source
