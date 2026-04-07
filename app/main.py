import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from app.model_loader import ToxicityModel
from app.predict_service import run_predict
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
    return run_predict(_model, body)
