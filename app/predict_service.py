"""Shared prediction path for FastAPI and Ray Serve."""

import logging
from typing import Optional

from app.model_loader import ToxicityModel
from app.policy import probability_to_risk_and_action
from app.schemas import PredictRequest, PredictResponse
from app.settings import settings

logger = logging.getLogger(__name__)


def run_predict(model: Optional[ToxicityModel], body: PredictRequest) -> PredictResponse:
    version = settings.model_version
    if model is None or not model.is_loaded:
        logger.warning("Predict fallback: model not loaded (message_id=%s)", body.message_id)
        return PredictResponse(
            message_id=body.message_id,
            model_version=version,
            toxicity_probability=None,
            risk_bucket="unknown",
            action_recommendation="allow_and_log",
            inference_status="fallback",
        )

    try:
        prob = model.predict_proba(body.text)
    except Exception as e:
        logger.exception("Predict fallback: inference error (message_id=%s): %s", body.message_id, e)
        return PredictResponse(
            message_id=body.message_id,
            model_version=version,
            toxicity_probability=None,
            risk_bucket="unknown",
            action_recommendation="allow_and_log",
            inference_status="fallback",
        )

    risk, action = probability_to_risk_and_action(prob)
    return PredictResponse(
        message_id=body.message_id,
        model_version=version,
        toxicity_probability=prob,
        risk_bucket=risk,
        action_recommendation=action,
        inference_status="success",
    )
