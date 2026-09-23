"""Private WebSocket server implementing Omega's existing channel protocol."""

from __future__ import annotations

import json
import hmac
import queue
import threading
import time
from collections import deque
from typing import Any


class OmegaGatewayError(RuntimeError):
    """Base error for live Omega transport failures."""


class OmegaGatewayTimeout(OmegaGatewayError):
    """Raised when Omega does not connect or answer before the deadline."""


class OmegaWebSocketGateway:
    """Host the server endpoint consumed by Omega's ``WSChannel`` client."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        token: str,
        path: str = "/agent",
        max_message_bytes: int = 64 * 1024,
    ) -> None:
        if not token or not token.strip():
            raise ValueError("token must be a non-empty dedicated bearer token")
        if not path.startswith("/"):
            raise ValueError("path must start with '/'")
        self.host = host
        self.port = int(port)
        self.token = token
        self.path = path
        self.max_message_bytes = int(max_message_bytes)

        self._server = None
        self._server_thread: threading.Thread | None = None
        self._connection = None
        self._connection_lock = threading.Lock()
        self._ready = threading.Event()
        self._request_lock = threading.Lock()
        self._responses: queue.Queue[str] = queue.Queue(maxsize=256)
        self._seq = 0
        self._pending_frame: dict[str, Any] | None = None
        self._seen_client_sequences: deque[str] = deque(maxlen=512)

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}{self.path}"

    def start(self) -> "OmegaWebSocketGateway":
        if self._server is not None:
            raise OmegaGatewayError("gateway is already running")
        try:
            from websockets.sync.server import serve
        except ImportError as exc:
            raise OmegaGatewayError(
                "websockets is required; install the SubRep requirements"
            ) from exc

        self._server = serve(
            self._handle_connection,
            self.host,
            self.port,
            max_size=self.max_message_bytes,
            ping_interval=20,
            ping_timeout=20,
        )
        if self.port == 0:
            self.port = int(self._server.socket.getsockname()[1])
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="subrep-omega-websocket",
        )
        self._server_thread.start()
        return self

    def stop(self) -> None:
        server = self._server
        self._server = None
        self._ready.clear()
        with self._connection_lock:
            connection = self._connection
            self._connection = None
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
        if server is not None:
            server.shutdown()
        if self._server_thread is not None:
            self._server_thread.join(timeout=5)
        self._server_thread = None

    def __enter__(self) -> "OmegaWebSocketGateway":
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()

    def complete(self, prompt: str, request_id: str, timeout_seconds: float) -> str:
        """Send one request and wait for one correlated structured response."""

        deadline = time.monotonic() + timeout_seconds
        with self._request_lock:
            if not self._ready.wait(timeout=max(0.0, deadline - time.monotonic())):
                raise OmegaGatewayTimeout("Omega did not connect and send a resume frame")
            self._drain_response_queue()
            self._seq += 1
            frame = {"type": "user_message", "seq": self._seq, "text": prompt}
            self._pending_frame = frame
            try:
                self._send(frame)
                unrelated: list[str] = []
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        if unrelated:
                            return unrelated[-1]
                        raise OmegaGatewayTimeout(
                            f"Omega did not answer request {request_id!r} before timeout"
                        )
                    try:
                        message = self._responses.get(timeout=remaining)
                    except queue.Empty as exc:
                        if unrelated:
                            return unrelated[-1]
                        raise OmegaGatewayTimeout(
                            f"Omega did not answer request {request_id!r} before timeout"
                        ) from exc
                    if _looks_like_structured_response(message, request_id):
                        return message
                    unrelated.append(message)
            finally:
                self._pending_frame = None

    def _handle_connection(self, websocket) -> None:
        request = websocket.request
        if request.path != self.path:
            websocket.close(code=1008, reason="unexpected path")
            return
        authorization = request.headers.get("Authorization", "")
        if not hmac.compare_digest(authorization, f"Bearer {self.token}"):
            websocket.close(code=1008, reason="unauthorized")
            return

        with self._connection_lock:
            if self._connection is not None:
                websocket.close(code=1013, reason="an Omega client is already connected")
                return
            self._connection = websocket

        try:
            for raw_message in websocket:
                self._handle_frame(websocket, raw_message)
        finally:
            with self._connection_lock:
                if self._connection is websocket:
                    self._connection = None
                    self._ready.clear()

    def _handle_frame(self, websocket, raw_message: str | bytes) -> None:
        if isinstance(raw_message, bytes):
            raw_message = raw_message.decode("utf-8")
        try:
            frame = json.loads(raw_message)
        except json.JSONDecodeError:
            self._send_to(websocket, {"type": "error", "code": "INVALID_JSON", "message": "JSON required"})
            return
        if not isinstance(frame, dict):
            self._send_to(websocket, {"type": "error", "code": "INVALID_FRAME", "message": "object required"})
            return

        frame_type = frame.get("type")
        if frame_type == "resume":
            last_seen = frame.get("last_seen_seq")
            if last_seen is not None and not isinstance(last_seen, int):
                self._send_to(websocket, {"type": "error", "code": "INVALID_RESUME", "message": "last_seen_seq must be int or null"})
                return
            self._ready.set()
            pending = self._pending_frame
            if pending is not None and (last_seen is None or pending["seq"] > last_seen):
                self._send_to(websocket, pending)
            return

        if frame_type == "agent_message":
            client_seq = frame.get("client_seq")
            text = frame.get("text")
            if not isinstance(client_seq, str) or not client_seq or not isinstance(text, str):
                self._send_to(websocket, {"type": "error", "code": "INVALID_AGENT_MESSAGE", "message": "client_seq and text are required"})
                return
            if client_seq not in self._seen_client_sequences:
                self._seen_client_sequences.append(client_seq)
                try:
                    self._responses.put_nowait(text)
                except queue.Full:
                    websocket.close(code=1013, reason="response queue is full")
                    return
            self._send_to(
                websocket,
                {
                    "type": "ack",
                    "seq": self._pending_frame["seq"] if self._pending_frame else None,
                    "client_seq": client_seq,
                },
            )
            return

        self._send_to(
            websocket,
            {"type": "error", "code": "UNSUPPORTED_FRAME", "message": f"unsupported type: {frame_type!r}"},
        )

    def _send(self, payload: dict[str, Any]) -> None:
        with self._connection_lock:
            connection = self._connection
        if connection is None:
            raise OmegaGatewayError("Omega disconnected before the request was sent")
        self._send_to(connection, payload)

    @staticmethod
    def _send_to(websocket, payload: dict[str, Any]) -> None:
        websocket.send(json.dumps(payload, allow_nan=False))

    def _drain_response_queue(self) -> None:
        while True:
            try:
                self._responses.get_nowait()
            except queue.Empty:
                return


def _looks_like_structured_response(message: str, request_id: str) -> bool:
    """Ignore progress chatter while still surfacing malformed response attempts."""

    text = message.strip()
    if "selected_skill_id" in text:
        return True
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("request_id") == request_id
