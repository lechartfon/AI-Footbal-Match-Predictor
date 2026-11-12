"""API tests: static UI, health, REST predict, docs disabled, WebSocket."""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.web_app import app, service  # noqa: E402


def type_line(ws, text):
    for ch in text:
        ws.send_text(ch)
    ws.send_text("\r")


def recv(ws):
    # starlette >=1.6 returns (text, flag) tuples in some cases; older
    # versions return plain strings. Normalize so the test is portable.
    message = ws.receive_text()
    return message[0] if isinstance(message, tuple) else message


@pytest.fixture(scope="module")
def client():
    # NOTE: starlette's TestClient does not run the FastAPI lifespan here,
    # so we load the inference service explicitly. Under real uvicorn the
    # lifespan (service.load()) runs automatically at startup.
    service.load()
    with TestClient(app) as c:
        yield c


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "AI Match Predictor" in r.text


def test_health(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_ready(client):
    assert client.get("/readyz").json() == {"status": "ready"}


def test_docs_disabled(client):
    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_meta(client):
    r = client.get("/api/v1/meta")
    assert r.status_code == 200
    data = r.json()
    assert data["model_version"]
    assert data["features"] == [
        "HomeTarget", "HomeCorners", "HomeFouls", "HomeSaves",
        "AwayTarget", "AwayCorners", "AwayFouls", "AwaySaves",
    ]


def test_predict_valid(client):
    payload = {
        "home_team": "Team A",
        "away_team": "Team B",
        "home": {"shots_on_target": 6, "corners": 7, "fouls": 15},
        "away": {"shots_on_target": 5, "corners": 4, "fouls": 12},
    }
    r = client.post("/api/v1/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["prediction"] in ("H", "D", "A")
    assert set(data["probabilities"]) == {"home_win", "draw", "away_win"}


def test_predict_invalid_range(client):
    payload = {
        "home_team": "A",
        "away_team": "B",
        "home": {"shots_on_target": 999, "corners": 1, "fouls": 1},
        "away": {"shots_on_target": 1, "corners": 1, "fouls": 1},
    }
    r = client.post("/api/v1/predict", json=payload)
    assert r.status_code == 400
    assert "error" in r.json()


def test_websocket_full_session(client):
    with client.websocket_connect("/ws/predict") as ws:
        banner = recv(ws)
        assert "HOME team name: " in banner

        type_line(ws, "Team A")
        assert recv(ws).endswith("AWAY team name: ")

        type_line(ws, "Team B")
        assert "Statistics for HOME team" in recv(ws)

        result = None
        for _ in range(6):
            type_line(ws, "")
            result = recv(ws)

        # The 6th stat (AWAY_FOULS) returns the full prediction + continue prompt.
        assert "PREDICTION: Team A vs Team B" in result
        assert "Make another prediction? (y/n): " in result

        type_line(ws, "n")
        goodbye = recv(ws)
        assert "Thanks for using the predictor!" in goodbye
