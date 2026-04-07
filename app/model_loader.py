import logging
from pathlib import Path
from typing import Any, Optional

import joblib

logger = logging.getLogger(__name__)


class ToxicityModel:
    """Loads a sklearn-style pipeline and exposes P(toxic) for the positive class (index 1)."""

    def __init__(self) -> None:
        self._pipeline: Optional[Any] = None

    def load(self, path: Path) -> None:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Model artifact not found: {path}")
        self._pipeline = joblib.load(path)
        if not hasattr(self._pipeline, "predict_proba"):
            raise TypeError("Loaded object must implement predict_proba")
        logger.info("Loaded model from %s", path)

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def predict_proba(self, text: str) -> float:
        if self._pipeline is None:
            raise RuntimeError("Model is not loaded")
        proba = self._pipeline.predict_proba([text])
        # Binary classifier: column 1 = positive (toxic) class
        return float(proba[0][1])
