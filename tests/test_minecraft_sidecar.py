from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any, Dict
from urllib import error, request

import pytest

from env.minecraft.sidecar_client import SidecarClient


TOKEN = "test-token"
GATED_ACTIONS = [
    "walk_route",
    "sprint_route",
    "deliver",
    "craft_torches",
    "place_torch",
    "build_gate",
    "build_golem",
    "buy_book",
    "cheap_trade",
    "cure_villager",
    "set_trade_hotkeys",
    "shoot",
    "eat",
]


class FakeBotHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.headers.get("X-SubRep-Token") != TOKEN:
            self._send(401, {"status": "error", "reason": "unauthorized"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/observe":
            payload = {
                "status": "ok",
                "position": [0, 64, 0],
                "health": 20,
                "food": 20,
                "inventory": [],
            }
        elif self.path == "/act":
            action = body.get("action")
            if action in GATED_ACTIONS:
                payload = {"status": "not_implemented", "action": action}
            elif action == "fly":
                payload = {"status": "error", "reason": "unknown_action"}
            else:
                payload = {"status": "ok", "position": [0, 64, 0]}
        elif self.path == "/stop":
            payload = {"status": "ok"}
        else:
            self._send(404, {"status": "error", "reason": "not_found"})
            return
        self._send(200, payload)

    def _send(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


@pytest.fixture
def fake_client() -> SidecarClient:
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeBotHandler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
        yield SidecarClient(
            "http://127.0.0.1:{}".format(server.server_port), TOKEN
        )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_observe_parses_fake_state(fake_client: SidecarClient) -> None:
    assert fake_client.observe()["position"] == [0, 64, 0]


def test_act_returns_ok(fake_client: SidecarClient) -> None:
    assert fake_client.act("wait")["status"] == "ok"


def test_act_passes_not_implemented(fake_client: SidecarClient) -> None:
    result = fake_client.act("walk_route")
    assert result == {"status": "not_implemented", "action": "walk_route"}


def test_act_passes_unknown_action(fake_client: SidecarClient) -> None:
    assert fake_client.act("fly") == {
        "status": "error",
        "reason": "unknown_action",
    }


def test_bad_token_raises(fake_client: SidecarClient) -> None:
    bad = SidecarClient(fake_client.base_url, "wrong-token")
    with pytest.raises(RuntimeError, match="authentication"):
        bad.observe()


def test_missing_token_raises(fake_client: SidecarClient) -> None:
    missing = SidecarClient(fake_client.base_url)
    with pytest.raises(RuntimeError, match="authentication"):
        missing.observe()


def test_stop_returns_ok(fake_client: SidecarClient) -> None:
    assert fake_client.emergency_stop() == {"status": "ok"}


_LIVE_CLIENT = None


def _live_client() -> SidecarClient:
    global _LIVE_CLIENT
    if _LIVE_CLIENT is None:
        _LIVE_CLIENT = SidecarClient(
            os.getenv("SUBREP_SIDECAR_URL", "http://127.0.0.1:25566"),
            os.environ["SUBREP_SIDECAR_TOKEN"],
        )
    return _LIVE_CLIENT


@pytest.mark.real_server
@pytest.mark.skipif(
    os.getenv("SUBREP_REAL_SERVER") != "1"
    or not os.getenv("SUBREP_SIDECAR_TOKEN"),
    reason="requires a live bot",
)
def test_live_observe() -> None:
    result = _live_client().observe()
    assert result["status"] == "ok"
    assert len(result["position"]) == 3


@pytest.mark.real_server
@pytest.mark.skipif(
    os.getenv("SUBREP_REAL_SERVER") != "1"
    or not os.getenv("SUBREP_SIDECAR_TOKEN"),
    reason="requires a live bot",
)
def test_live_gated_actions_do_not_move() -> None:
    client = _live_client()
    before = client.observe()["position"]
    for action in GATED_ACTIONS:
        assert client.act(action) == {
            "status": "not_implemented",
            "action": action,
        }
    assert client.observe()["position"] == before


@pytest.mark.real_server
@pytest.mark.skipif(
    os.getenv("SUBREP_REAL_SERVER") != "1"
    or not os.getenv("SUBREP_SIDECAR_TOKEN"),
    reason="requires a live bot",
)
def test_live_move_hall() -> None:
    result = _live_client().act("move_hall")
    assert result["status"] == "ok"
    assert result["position"] == [40, 64, 0]


@pytest.mark.real_server
@pytest.mark.skipif(
    os.getenv("SUBREP_REAL_SERVER") != "1"
    or not os.getenv("SUBREP_SIDECAR_TOKEN"),
    reason="requires a live bot",
)
def test_live_bad_token() -> None:
    client = SidecarClient(
        os.getenv("SUBREP_SIDECAR_URL", "http://127.0.0.1:25566"),
        "wrong-token",
    )
    with pytest.raises(RuntimeError, match="authentication"):
        client.observe()


@pytest.mark.real_server
@pytest.mark.skipif(
    os.getenv("SUBREP_REAL_SERVER") != "1"
    or not os.getenv("SUBREP_SIDECAR_TOKEN"),
    reason="requires a live bot",
)
def test_live_unknown_action() -> None:
    assert _live_client().act("fly") == {
        "status": "error",
        "reason": "unknown_action",
    }


@pytest.mark.real_server
@pytest.mark.skipif(
    os.getenv("SUBREP_REAL_SERVER") != "1"
    or not os.getenv("SUBREP_SIDECAR_TOKEN"),
    reason="requires a live bot",
)
def test_live_undefined_path_is_not_ok() -> None:
    http_request = request.Request(
        os.getenv("SUBREP_SIDECAR_URL", "http://127.0.0.1:25566") + "/chat",
        data=b"{}",
        method="POST",
        headers={"X-SubRep-Token": os.environ["SUBREP_SIDECAR_TOKEN"]},
    )
    with pytest.raises(error.HTTPError) as raised:
        request.urlopen(http_request, timeout=5)
    assert raised.value.code != 200
