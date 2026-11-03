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


def test_bot_protocol_has_limits_replay_and_health_surfaces() -> None:
    source = (BOT_DIR / "index.js").read_text(encoding="utf-8")
    assert "MAX_BODY_BYTES" in source
    assert "statusCode: 413" in source
    assert "statusCode: 429" in source or "sendJson(response, 429" in source
    assert "seenMessageIds" in source
    assert '"/health"' in source
    assert '"/status"' in source
    assert 'auth: "offline"' in source
