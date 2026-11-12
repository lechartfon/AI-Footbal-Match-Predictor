"""
FastAPI application: serves the terminal web UI and exposes the predictor
over a WebSocket (one session per connection) plus a small REST API.

Design notes:
  - Single uvicorn worker (start command in deploy/). Multiple WebSockets
    are served concurrently by asyncio; the only blocking call is Keras'
    model.predict, which is delegated to the event loop executor via
    InferenceService.predict_async (asyncio.to_thread).
  - No training, no subprocess, no shell, no admin pages.
  - Auto-generated docs (/docs, /redoc, /openapi.json) are disabled.
  - Basic per-IP rate limiting for the public demo.
"""
import asyncio
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .inference import InferenceService, ValidationError
from .terminal_session import TerminalSession

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"

service = InferenceService()

_sessions_per_ip: defaultdict[str, int] = defaultdict(int)
_message_log_per_ip: defaultdict[str, deque] = defaultdict(deque)
_sessions: dict[str, TerminalSession] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    service.load()
    yield


app = FastAPI(
    title="AI Match Predictor",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    return {"status": "ready" if service.loaded else "not_ready"}


@app.get("/api/v1/meta")
async def meta():
    return {
        "model_version": service.manifest["model_version"],
        "features": service.manifest["input_features"],
        "classes": service.manifest["classes"],
        "limits": service.config["limits"],
        "note": "Educational demo. Post-match statistics, not a pre-match prediction.",
    }


class TeamStatsIn(BaseModel):
    shots_on_target: float
    corners: float
    fouls: float


class PredictRequest(BaseModel):
    home_team: str = Field(min_length=1, max_length=50)
    away_team: str = Field(min_length=1, max_length=50)
    home: TeamStatsIn
    away: TeamStatsIn


@app.post("/api/v1/predict")
async def predict_api(req: PredictRequest):
    home = {"shots_on_target": req.home.shots_on_target, "corners": req.home.corners, "fouls": req.home.fouls}
    away = {"shots_on_target": req.away.shots_on_target, "corners": req.away.corners, "fouls": req.away.fouls}
    try:
        result = await service.predict_async(home, away)
    except ValidationError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    result["home_team"] = req.home_team
    result["away_team"] = req.away_team
    return result


def _check_rate(ip: str) -> bool:
    """Simple sliding window: max N messages per minute per IP."""
    now = time.monotonic()
    log = _message_log_per_ip[ip]
    limit = service.config["max_messages_per_minute_per_ip"]
    while log and log[0] < now - 60:
        log.popleft()
    if len(log) >= limit:
        return False
    log.append(now)
    return True


@app.websocket("/ws/predict")
async def predict_ws(ws: WebSocket):
    ip = ws.client.host
    await ws.accept()

    if _sessions_per_ip[ip] >= service.config["max_sessions_per_ip"]:
        await ws.send_text("\r\nToo many active sessions from this IP.\r\n")
        await ws.close(code=1008)
        return
    _sessions_per_ip[ip] += 1

    session_id = uuid.uuid4().hex[:12]
    session = TerminalSession(service, session_id)
    _sessions[session_id] = session
    idle = service.config["idle_timeout_seconds"]

    try:
        await ws.send_text(session.initial_text())
        while True:
            try:
                char = await asyncio.wait_for(ws.receive_text(), timeout=idle)
            except asyncio.TimeoutError:
                await ws.send_text("\r\nSession idle. Closing...\r\n")
                break
            if not _check_rate(ip):
                await ws.send_text("\r\nRate limit exceeded. Closing.\r\n")
                break
            text = await session.feed(char)
            if text:
                await ws.send_text(text)
            if session.closed:
                break
    except WebSocketDisconnect:
        pass
    finally:
        _sessions_per_ip[ip] = max(0, _sessions_per_ip[ip] - 1)
        _sessions.pop(session_id, None)
