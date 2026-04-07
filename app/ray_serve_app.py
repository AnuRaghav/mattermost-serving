"""Ray Serve HTTP deployment: same /health and /predict contract as FastAPI (no FastAPI ingress)."""

from __future__ import annotations

import logging
import os

import ray
from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.model_loader import ToxicityModel
from app.predict_service import run_predict
from app.schemas import PredictRequest
from app.settings import settings

logger = logging.getLogger(__name__)


def _num_replicas() -> int:
    raw = os.environ.get("RAY_SERVE_NUM_REPLICAS", "1")
    try:
        n = int(raw)
    except ValueError:
        return 1
    return max(1, n)


@serve.deployment
class ModerationIngress:
    """Single deployment handles routing (Starlette Request) for benchmark-compatible paths."""

    def __init__(self) -> None:
        self._model = ToxicityModel()
        try:
            self._model.load(settings.model_path)
        except Exception as e:
            logger.error("Could not load model from %s: %s", settings.model_path, e)
            self._model = ToxicityModel()

    async def __call__(self, request: Request) -> Response:
        path = request.url.path
        if path != "/health" and path.endswith("/") and len(path) > 1:
            path = path.rstrip("/")

        if path == "/health" and request.method == "GET":
            return JSONResponse(
                {"status": "healthy", "model_loaded": self._model.is_loaded}
            )

        if path == "/predict" and request.method == "POST":
            try:
                payload = await request.json()
                body = PredictRequest.model_validate(payload)
            except Exception as e:
                return JSONResponse({"detail": str(e)}, status_code=422)
            out = run_predict(self._model, body)
            return JSONResponse(out.model_dump(mode="json"))

        return JSONResponse({"detail": "Not Found"}, status_code=404)


def main() -> None:
    host = os.environ.get("RAY_SERVE_HOST", "0.0.0.0")
    port = int(os.environ.get("RAY_SERVE_PORT", "8000"))
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    serve.start(http_options={"host": host, "port": port})
    deployment = ModerationIngress.options(num_replicas=_num_replicas())
    serve.run(deployment.bind(), blocking=True, route_prefix="/")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
