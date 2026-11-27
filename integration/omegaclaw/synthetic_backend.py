"""Deterministic test backend for synthetic integration scenarios."""

from __future__ import annotations

import json

from .contracts import (
    INTEGRATION_MODE,
    RESPONSE_SCHEMA_VERSION,
    SELECTION_BASIS,
)


class SyntheticOmegaBackend:
    """Mimic a grounded Omega reply without calling an LLM or live agent.

    It claims what explains the decision: the objective weights, the deltas of the
    selected skill and the runner-up, and the stage of every exclusion.
    """

    evidence_label = "SYNTHETIC"

    def complete(self, prompt: str, request_id: str, timeout_seconds: float) -> str:
        del timeout_seconds
        request = _request_from_prompt(prompt)
        if request["request_id"] != request_id:
            raise ValueError("prompt request_id does not match backend request_id")

        decision = request["subrep_decision"]
        claims = [
            _claim("exclusion_stage", f"exclusions.{index}.stage", item["stage"])
            for index, item in enumerate(request["exclusions"])
        ]
        if not decision["abstain"]:
            evidence = request["explanation_evidence"]
            claims = [
                *(
                    _claim(
                        "objective_weight",
                        f"explanation_evidence.objective_weights.{name}",
                        weight,
                    )
                    for name, weight in evidence["objective_weights"].items()
                ),
                *(
                    _claim(
                        "objective_delta",
                        f"explanation_evidence.skills.{skill_id}.objective_deltas.{name}",
                        delta,
                    )
                    for skill_id in (decision["selected_skill_id"], decision["runner_up_skill_id"])
                    if skill_id is not None
                    for name, delta in evidence["skills"][skill_id]["objective_deltas"].items()
                ),
                *claims,
            ]
        return json.dumps(
            {
                "schema_version": RESPONSE_SCHEMA_VERSION,
                "request_id": request_id,
                "selected_skill_id": decision["selected_skill_id"],
                "abstain": decision["abstain"],
                "mode": INTEGRATION_MODE,
                "selection_basis": SELECTION_BASIS,
                "claims": claims,
                "advisory_concerns": [],
            },
            sort_keys=True,
        )


def _claim(claim_type: str, evidence_ref: str, value) -> dict:
    return {"type": claim_type, "evidence_ref": evidence_ref, "value": value}


def _request_from_prompt(prompt: str) -> dict:
    start_marker = "SUBREP_REQUEST_JSON_BEGIN\n"
    end_marker = "\nSUBREP_REQUEST_JSON_END"
    if start_marker not in prompt or end_marker not in prompt:
        raise ValueError("prompt does not contain the SubRep request markers")
    request_json = prompt.split(start_marker, 1)[1].split(end_marker, 1)[0]
    payload = json.loads(request_json)
    if not isinstance(payload, dict):
        raise ValueError("embedded request must be a JSON object")
    return payload
