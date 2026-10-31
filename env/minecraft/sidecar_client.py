from __future__ import annotations

import json
import time
from typing import Dict, Any
from urllib import error, request


class SidecarClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:25566",
        token: str = "",
        timeout: float = 35,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._last_request_at = 0.0

    def observe(self) -> Dict[str, Any]:
        return self._post("/observe", {}, 5)

    def act(self, action: str) -> Dict[str, Any]:
        return self._post("/act", {"action": action}, self.timeout)

    def emergency_stop(self) -> Dict[str, Any]:
        return self._post("/stop", {}, 5)

    def _post(
        self, path: str, payload: Dict[str, Any], timeout: float
    ) -> Dict[str, Any]:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < 0.11:
            time.sleep(0.11 - elapsed)
        self._last_request_at = time.monotonic()
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            self.base_url + path,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-SubRep-Token": self.token,
            },
        )
        try:
            with request.urlopen(http_request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            if exc.code == 401:
                raise RuntimeError("sidecar authentication failed") from exc
            if exc.code == 413:
                raise RuntimeError("sidecar rejected an oversized request") from exc
            if exc.code == 429:
                raise RuntimeError("sidecar rate limit exceeded") from exc
            if exc.code == 400:
                try:
                    result = json.loads(exc.read().decode("utf-8"))
                except (ValueError, UnicodeDecodeError) as parse_error:
                    raise RuntimeError("sidecar returned invalid JSON") from parse_error
                if isinstance(result, dict):
                    return result
            raise RuntimeError(
                "sidecar returned HTTP {}".format(exc.code)
            ) from exc
        except error.URLError as exc:
            raise RuntimeError("sidecar connection failed: {}".format(exc.reason)) from exc
        except TimeoutError as exc:
            raise RuntimeError("sidecar request timed out") from exc
        if not isinstance(result, dict):
            raise RuntimeError("sidecar returned invalid JSON")
        return result
