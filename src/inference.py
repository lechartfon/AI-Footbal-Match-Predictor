"""
Inference service for the web demo.

Loads a single production model plus preprocessing artifacts once at
startup and exposes a blocking ``predict`` method that the async layer
calls through ``asyncio.to_thread`` so Keras' blocking ``model.predict``
never blocks the event loop.

There is no training, no subprocess and no shell in this module.
"""
import asyncio
import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACTS_DIR = BASE_DIR / "artifacts"

FEATURE_ORDER = [
    "HomeTarget", "HomeCorners", "HomeFouls", "HomeSaves",
    "AwayTarget", "AwayCorners", "AwayFouls", "AwaySaves",
]
CLASSES = ["H", "D", "A"]


class ValidationError(Exception):
    """Raised when an input value is invalid."""


class InferenceService:
    """Holds the production model and all preprocessing artifacts."""

    def __init__(self, artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR):
        self.artifacts_dir = Path(artifacts_dir)
        self.model: Any = None
        self.scaler: Dict[str, Any] = {}
        self.defaults: Dict[str, Any] = {}
        self.config: Dict[str, Any] = {}
        self.manifest: Dict[str, Any] = {}
        self.loaded = False

    def load(self) -> None:
        """Load model and artifacts once at startup."""
        a = self.artifacts_dir
        self.model = load_model(a / "production_model.keras", compile=False)
        self.scaler = json.loads((a / "scaler.json").read_text(encoding="utf-8"))
        self.defaults = json.loads((a / "defaults.json").read_text(encoding="utf-8"))
        self.config = json.loads((a / "config.json").read_text(encoding="utf-8"))
        self.manifest = json.loads((a / "manifest.json").read_text(encoding="utf-8"))

        expected = self.manifest.get("input_features")
        if expected != FEATURE_ORDER:
            raise RuntimeError(f"Feature schema mismatch: {expected}")
        if self.model.input_shape[-1] != len(FEATURE_ORDER):
            raise RuntimeError("Model input shape does not match feature schema")
        self.loaded = True

    def _check_number(self, value: Any, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValidationError(f"{name} must be a number")
        if not math.isfinite(value) or value < 0:
            raise ValidationError(f"{name} must be a non-negative finite number")
        lo, hi = self.config["limits"][name]
        if not (lo <= value <= hi):
            raise ValidationError(f"{name} must be between {lo} and {hi}")

    def validate_stat(self, name: str, value: Any) -> None:
        """Public validation used by the terminal session and the API."""
        self._check_number(value, name)

    def _build_features(self, home: Dict[str, float], away: Dict[str, float]) -> np.ndarray:
        for key in ("shots_on_target", "corners", "fouls"):
            self._check_number(home[key], key)
            self._check_number(away[key], key)
        # Order: HomeTarget, HomeCorners, HomeFouls, HomeSaves,
        #        AwayTarget, AwayCorners, AwayFouls, AwaySaves
        # Saves = opponent's shots on target.
        return np.array([
            home["shots_on_target"], home["corners"], home["fouls"], away["shots_on_target"],
            away["shots_on_target"], away["corners"], away["fouls"], home["shots_on_target"],
        ], dtype=np.float64)

    def predict(self, home: Dict[str, float], away: Dict[str, float]) -> Dict[str, Any]:
        """Blocking prediction. Call via predict_async from async code."""
        if not self.loaded:
            raise RuntimeError("Inference service not loaded")
        features = self._build_features(home, away)
        scaled = (features - np.asarray(self.scaler["mean"])) / np.asarray(self.scaler["scale"])
        probs = self.model.predict(scaled.reshape(1, -1), verbose=0)[0].astype(float)
        idx = int(np.argmax(probs))
        return {
            "prediction": CLASSES[idx],
            "probabilities": {
                "home_win": round(float(probs[0]), 4),
                "draw": round(float(probs[1]), 4),
                "away_win": round(float(probs[2]), 4),
            },
            "model_version": self.manifest["model_version"],
        }

    async def predict_async(self, home: Dict[str, float], away: Dict[str, float]) -> Dict[str, Any]:
        """Non-blocking wrapper: runs the Keras call in the default executor."""
        return await asyncio.to_thread(self.predict, home, away)