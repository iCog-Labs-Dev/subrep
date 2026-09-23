from __future__ import annotations

import json
import threading
from uuid import uuid4

from integration.omegaclaw.adapter import OmegaRecommendationAdapter
from integration.omegaclaw.audit import JsonlAuditStore
from integration.omegaclaw.contracts import RESPONSE_SCHEMA_VERSION
from integration.omegaclaw.scenarios import predefined_scenarios
from integration.omegaclaw.service import SubRepOmegaRecommendationService
from integration.omegaclaw.websocket_gateway import OmegaWebSocketGateway


def test_complete_subrep_websocket_validated_response_flow(tmp_path):
    token = "test-only-token"
    gateway = OmegaWebSocketGateway(host="127.0.0.1", port=0, token=token)
    with gateway:
        client = threading.Thread(
            target=_simulated_omega_client,
            args=(gateway.url, token),
            daemon=True,
        )
        client.start()
        scenario = predefined_scenarios()[0]
        service = SubRepOmegaRecommendationService(
            OmegaRecommendationAdapter(
                gateway,
                JsonlAuditStore(tmp_path / "websocket-audit.jsonl"),
                timeout_seconds=5,
            )
        )

        outcome = service.recommend_from_library(
            task_context=scenario.task_context,
            objective_weights=scenario.objective_weights,
            risk_budget=scenario.risk_budget,
            skill_library=scenario.skill_library,
            evidence_label="SYNTHETIC",
            request_id="websocket-flow",
        )
        client.join(timeout=5)

    assert outcome.status == "accepted"
    assert outcome.selected_skill_id == "skill_safety"
    assert not client.is_alive()


def test_connection_timeout_is_recorded_as_backend_error(tmp_path):
    gateway = OmegaWebSocketGateway(host="127.0.0.1", port=0, token="test-only-token")
    with gateway:
        scenario = predefined_scenarios()[0]
        service = SubRepOmegaRecommendationService(
            OmegaRecommendationAdapter(
                gateway,
                JsonlAuditStore(tmp_path / "timeout-audit.jsonl"),
                timeout_seconds=0.05,
            )
        )
        outcome = service.recommend_from_library(
            task_context=scenario.task_context,
            objective_weights=scenario.objective_weights,
            risk_budget=scenario.risk_budget,
            skill_library=scenario.skill_library,
            evidence_label="SYNTHETIC",
        )

    assert outcome.status == "backend_error"
    assert "did not connect" in outcome.error


def _simulated_omega_client(url: str, token: str) -> None:
    from websockets.sync.client import connect

    headers = {"Authorization": f"Bearer {token}"}
    try:
        connection = connect(url, additional_headers=headers)
    except TypeError:
        connection = connect(url, extra_headers=headers)

    with connection as websocket:
        websocket.send(json.dumps({"type": "resume", "last_seen_seq": None}))
        request_frame = json.loads(websocket.recv(timeout=5))
        assert request_frame["type"] == "user_message"
        prompt = request_frame["text"]
        request_json = prompt.split("SUBREP_REQUEST_JSON_BEGIN\n", 1)[1].split(
            "\nSUBREP_REQUEST_JSON_END", 1
        )[0]
        request = json.loads(request_json)
        selected = max(request["admitted_skills"], key=lambda item: item["score"])
        websocket.send(
            json.dumps(
                {
                    "type": "agent_message",
                    "client_seq": uuid4().hex,
                    "text": f"Reviewing request {request['request_id']}.",
                }
            )
        )
        progress_acknowledgement = json.loads(websocket.recv(timeout=5))
        assert progress_acknowledgement["type"] == "ack"
        response = {
            "schema_version": RESPONSE_SCHEMA_VERSION,
            "request_id": request["request_id"],
            "selected_skill_id": selected["skill_id"],
            "abstain": False,
            "explanation": (
                f"{selected['skill_id']} has the highest supplied score "
                f"{selected['score']:.3f}."
            ),
            "cited_skill_ids": [selected["skill_id"]],
        }
        websocket.send(
            json.dumps(
                {
                    "type": "agent_message",
                    "client_seq": uuid4().hex,
                    "text": json.dumps(response),
                }
            )
        )
        acknowledgement = json.loads(websocket.recv(timeout=5))
        assert acknowledgement["type"] == "ack"
