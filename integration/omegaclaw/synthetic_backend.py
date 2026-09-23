"""Deterministic test backend for synthetic integration scenarios."""

from __future__ import annotations

import json

from .contracts import RESPONSE_SCHEMA_VERSION


class SyntheticOmegaBackend:
    """Mimic a grounded Omega reply without calling an LLM or live agent."""

    evidence_label = "SYNTHETIC"

    def complete(self, prompt: str, request_id: str, timeout_seconds: float) -> str:
        del timeout_seconds
        request = _request_from_prompt(prompt)
        if request["request_id"] != request_id:
            raise ValueError("prompt request_id does not match backend request_id")

        admitted = request["admitted_skills"]
        if not admitted:
            excluded_count = len(request["exclusions"])
            return json.dumps(
                {
                    "schema_version": RESPONSE_SCHEMA_VERSION,
                    "request_id": request_id,
                    "selected_skill_id": None,
                    "abstain": True,
                    "explanation": (
                        "No admitted skills were supplied; abstaining. "
                        f"The request listed {excluded_count} excluded option(s)."
                    ),
                    "cited_skill_ids": [
                        item["skill_id"] for item in request["exclusions"]
                    ],
                },
                sort_keys=True,
            )

        selected = min(
            admitted,
            key=lambda item: (-float(item["score"]), item["skill_id"]),
        )
        return json.dumps(
            {
                "schema_version": RESPONSE_SCHEMA_VERSION,
                "request_id": request_id,
                "selected_skill_id": selected["skill_id"],
                "abstain": False,
                "explanation": (
                    f"{selected['skill_id']} has the highest supplied score "
                    f"{float(selected['score']):.3f} under the supplied objective weights."
                ),
                "cited_skill_ids": [selected["skill_id"]],
            },
            sort_keys=True,
        )


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
