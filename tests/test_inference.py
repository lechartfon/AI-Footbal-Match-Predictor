"""Tests for the inference service (model + scaler + validation)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import InferenceService, ValidationError  # noqa: E402

SERVICE = None


def get_service():
    global SERVICE
    if SERVICE is None:
        SERVICE = InferenceService()
        SERVICE.load()
    return SERVICE


def test_predict_returns_valid_distribution():
    svc = get_service()
    result = svc.predict(
        {"shots_on_target": 6, "corners": 7, "fouls": 15},
        {"shots_on_target": 5, "corners": 4, "fouls": 12},
    )
    assert result["prediction"] in ("H", "D", "A")
    probs = result["probabilities"]
    assert all(0.0 <= v <= 1.0 for v in probs.values())
    assert abs(sum(probs.values()) - 1.0) < 0.02
    assert "model_version" in result


def test_defaults_available():
    svc = get_service()
    assert "home" in svc.defaults
    assert "away" in svc.defaults
    assert set(svc.defaults["home"]) == {"shots_on_target", "corners", "fouls"}


def test_out_of_range_rejected():
    svc = get_service()
    with pytest.raises(ValidationError):
        svc.predict(
            {"shots_on_target": 999, "corners": 1, "fouls": 1},
            {"shots_on_target": 1, "corners": 1, "fouls": 1},
        )


def test_negative_rejected():
    svc = get_service()
    with pytest.raises(ValidationError):
        svc.predict(
            {"shots_on_target": -3, "corners": 1, "fouls": 1},
            {"shots_on_target": 1, "corners": 1, "fouls": 1},
        )
