"""Recommendation-only bridge between SubRep and Omega's WebSocket channel."""

from .adapter import OmegaRecommendationAdapter
from .audit import JsonlAuditStore
from .contracts import (
    ObjectiveWeight,
    RecommendationOutcome,
    RecommendationRequest,
    RiskBudget,
    SkillEvidence,
    SkillExclusion,
)
from .service import SubRepOmegaRecommendationService
from .websocket_gateway import OmegaWebSocketGateway

__all__ = [
    "JsonlAuditStore",
    "ObjectiveWeight",
    "OmegaRecommendationAdapter",
    "RecommendationOutcome",
    "RecommendationRequest",
    "RiskBudget",
    "SkillEvidence",
    "SkillExclusion",
    "SubRepOmegaRecommendationService",
    "OmegaWebSocketGateway",
]
