"""Tests for the terminal session state machine."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import InferenceService  # noqa: E402
from src.terminal_session import TerminalSession  # noqa: E402

SERVICE = None


def get_service():
    global SERVICE
    if SERVICE is None:
        SERVICE = InferenceService()
        SERVICE.load()
    return SERVICE


def run(coro):
    return asyncio.run(coro)


async def type_line(session, text):
    for ch in text:
        await session.feed(ch)
    return await session.feed("\r")


def test_full_flow_with_defaults_and_exit():
    svc = get_service()
    session = TerminalSession(svc, "t1")

    initial = session.initial_text()
    assert "FOOTBALL RESULT PREDICTOR" in initial
    assert initial.endswith("HOME team name: ")

    text = run(type_line(session, "Team A"))
    assert text.endswith("AWAY team name: ")
    assert not session.closed

    text = run(type_line(session, "Team B"))
    assert "Statistics for HOME team" in text
    assert "Shots on target (avg:" in text
    assert not session.closed

    # Use dataset defaults for the 6 stats (just press Enter).
    for _ in range(6):
        text = run(type_line(session, ""))
        assert not session.closed

    # After the 6th stat we get the prediction + continue prompt.
    assert "PREDICTION: Team A vs Team B" in text
    assert "Make another prediction? (y/n): " in text
    assert "Prediction:" in text

    # Exit.
    text = run(type_line(session, "n"))
    assert session.closed
    assert "Thanks for using the predictor!" in text


def test_invalid_value_reprompts():
    svc = get_service()
    session = TerminalSession(svc, "t2")
    session.state = "HOME_SHOTS"
    session.home_team = "A"
    session.away_team = "B"

    text = run(type_line(session, "not-a-number"))
    assert "Invalid value" in text
    assert "Shots on target" in text
    assert not session.closed


def test_empty_name_rejected():
    svc = get_service()
    session = TerminalSession(svc, "t3")
    text = run(type_line(session, ""))
    assert "Team name cannot be empty" in text
    assert "HOME team name: " in text
    assert not session.closed
