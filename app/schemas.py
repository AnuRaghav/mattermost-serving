from typing import Literal, Optional

from pydantic import BaseModel, Field


ChannelType = Literal["public", "private", "dm"]
RiskBucket = Literal["low", "medium", "high", "critical", "unknown"]
ActionRecommendation = Literal[
    "no_action",
    "low_priority_queue",
    "high_priority_queue",
    "highest_priority_queue",
    "allow_and_log",
]
InferenceStatus = Literal["success", "fallback"]


class PredictRequest(BaseModel):
    message_id: str = Field(..., min_length=1)
    text: str
    channel_type: ChannelType
    prior_violation_count: int = Field(..., ge=0)


class PredictResponse(BaseModel):
    message_id: str
    model_version: str
    toxicity_probability: Optional[float] = None
    risk_bucket: RiskBucket
    action_recommendation: ActionRecommendation
    inference_status: InferenceStatus
