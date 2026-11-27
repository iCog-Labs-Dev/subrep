"""Explanation-only bridge between SubRep and Omega's WebSocket channel."""

from .adapter import OmegaRecommendationAdapter
from .audit import JsonlAuditStore
from .contracts import (
    AdvisoryConcern,
    Claim,
    ObjectiveWeight,
    RecommendationOutcome,
    RecommendationRequest,
    RiskBudget,
    SkillEvidence,
    SkillExclusion,
    SubRepDecision,
)
from .service import SubRepOmegaRecommendationService
from .websocket_gateway import OmegaWebSocketGateway

__all__ = [
    "JsonlAuditStore",
    "AdvisoryConcern",
    "Claim",
    "ObjectiveWeight",
    "OmegaRecommendationAdapter",
    "RecommendationOutcome",
    "RecommendationRequest",
    "RiskBudget",
    "SkillEvidence",
    "SkillExclusion",
    "SubRepOmegaRecommendationService",
    "SubRepDecision",
    "OmegaWebSocketGateway",
]
