"""Deterministic test backend for synthetic integration scenarios."""

from __future__ import annotations

import json

from .contracts import (
    INTEGRATION_MODE,
    RESPONSE_SCHEMA_VERSION,
    SELECTION_BASIS,
)


class SyntheticOmegaBackend:
    """Mimic a grounded Omega reply without calling an LLM or live agent."""

    evidence_label = "SYNTHETIC"

    def complete(self, prompt: str, request_id: str, timeout_seconds: float) -> str:
        del timeout_seconds
        request = _request_from_prompt(prompt)
        if request["request_id"] != request_id:
            raise ValueError("prompt request_id does not match backend request_id")

        decision = request["subrep_decision"]
        exclusions = _describe_exclusions(request["exclusions"])
        if decision["abstain"]:
            return json.dumps(
                {
                    "schema_version": RESPONSE_SCHEMA_VERSION,
                    "request_id": request_id,
                    "selected_skill_id": None,
                    "abstain": True,
                    "mode": INTEGRATION_MODE,
                    "selection_basis": SELECTION_BASIS,
                    "explanation": (
                        "No recommendation was made because SubRep found no admitted skills "
                        f"in this synthetic scenario.{exclusions}"
                    ),
                    "evidence_refs": ["subrep_decision.reason_code"],
                    "advisory_concerns": [],
                },
                sort_keys=True,
            )

        selected_id = decision["selected_skill_id"]
        selected_score = float(decision["selected_score"])
        runner_id = decision["runner_up_skill_id"]
        runner_score = decision["runner_up_score"]
        refs = ["subrep_decision.selected_score"]
        if runner_id is not None:
            refs.append("subrep_decision.runner_up_score")
            explanation = (
                f"Using synthetic evidence, {selected_id} was selected by SubRep with score "
                f"{selected_score:.3f}, ahead of {runner_id} at {float(runner_score):.3f}."
            )
        else:
            explanation = (
                f"Using synthetic evidence, {selected_id} was selected by SubRep as the "
                f"only admitted skill with score {selected_score:.3f}."
            )
        explanation += exclusions
        return json.dumps(
            {
                "schema_version": RESPONSE_SCHEMA_VERSION,
                "request_id": request_id,
                "selected_skill_id": selected_id,
                "abstain": False,
                "mode": INTEGRATION_MODE,
                "selection_basis": SELECTION_BASIS,
                "explanation": explanation,
                "evidence_refs": refs,
                "advisory_concerns": [],
            },
            sort_keys=True,
        )


def _describe_exclusions(exclusions: list[dict]) -> str:
    """Restate each exclusion with the first sentence of its SubRep stage description."""

    return "".join(
        f" {item['skill_id']} was excluded: {item['stage_description'].split('. ')[0].rstrip('.')}."
        for item in exclusions
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
