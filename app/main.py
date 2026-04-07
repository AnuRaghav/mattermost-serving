import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from app.model_loader import ToxicityModel
from app.policy import probability_to_risk_and_action
from app.schemas import PredictRequest, PredictResponse
from app.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_model: Optional[ToxicityModel] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    _model = ToxicityModel()
    try:
        _model.load(settings.model_path)
    except Exception as e:
        logger.error("Could not load model from %s: %s", settings.model_path, e)
        _model = ToxicityModel()
    yield
    _model = None


app = FastAPI(
    title="Mattermost moderation assist (serving)",
    description="HITL toxicity scoring API — placeholder sklearn artifact.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    loaded = _model is not None and _model.is_loaded
    return {"status": "healthy", "model_loaded": loaded}


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest) -> PredictResponse:
    version = settings.model_version
    if _model is None or not _model.is_loaded:
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
        prob = _model.predict_proba(body.text)
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
