"""
WebSocket terminal session.

A small state machine that drives the console-style conversation between
the browser and the predictor. There is NO shell, NO PTY and NO subprocess:
the browser sends keystrokes, the session answers with text.

The session buffers keystrokes per prompt, validates every answer, uses the
dataset averages when the user presses Enter, and finally runs the blocking
Keras prediction through the service's ``predict_async`` (to_thread) so the
single uvicorn worker never blocks the event loop for other clients.
"""
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from .inference import InferenceService, ValidationError

# (state, side, stat key, label)
STAT_STEPS = [
    ("HOME_SHOTS", "home", "shots_on_target", "Shots on target"),
    ("HOME_CORNERS", "home", "corners", "Corners"),
    ("HOME_FOULS", "home", "fouls", "Fouls"),
    ("AWAY_SHOTS", "away", "shots_on_target", "Shots on target"),
    ("AWAY_CORNERS", "away", "corners", "Corners"),
    ("AWAY_FOULS", "away", "fouls", "Fouls"),
]
STAT_BY_STATE = {state: (side, key, label) for state, side, key, label in STAT_STEPS}

BANNER = (
    "\r\n============================================================\r\n"
    "FOOTBALL RESULT PREDICTOR\r\n"
    "============================================================\r\n\r\n"
)

GREEN = "\x1b[92m"
YELLOW = "\x1b[93m"
BLUE = "\x1b[94m"
BOLD = "\x1b[1m"
END = "\x1b[0m"


class TerminalSession:
    """One interactive session = one WebSocket connection."""

    def __init__(self, service: InferenceService, session_id: str):
        self.service = service
        self.session_id = session_id
        self.state = "HOME_TEAM"
        self.buffer = ""
        self.home_team: Optional[str] = None
        self.away_team: Optional[str] = None
        self.home_stats: Dict[str, float] = {}
        self.away_stats: Dict[str, float] = {}
        self.closed = False
        self.last_activity = time.monotonic()
        self.created_at = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------ #
    # Prompt rendering
    # ------------------------------------------------------------------ #
    def _stat_prompt(self, state: str) -> str:
        side, key, label = STAT_BY_STATE[state]
        default = self.service.defaults[side][key]
        prefix = ""
        if key == "shots_on_target":
            team = "HOME" if side == "home" else "AWAY"
            prefix = (
                f"Statistics for {team} team:\r\n"
                f"(Press Enter to use dataset averages)\r\n"
            )
        return f"{prefix}  {label} (avg: {default:.1f}): "

    def current_prompt(self) -> str:
        if self.state == "HOME_TEAM":
            return "HOME team name: "
        if self.state == "AWAY_TEAM":
            return "AWAY team name: "
        if self.state in STAT_BY_STATE:
            return self._stat_prompt(self.state)
        if self.state == "CONTINUE":
            return "Make another prediction? (y/n): "
        return ""

    def initial_text(self) -> str:
        return BANNER + self.current_prompt()

    # ------------------------------------------------------------------ #
    # Keystroke feeding
    # ------------------------------------------------------------------ #
    async def feed(self, char: str) -> Optional[str]:
        """Feed one keystroke. Returns text to send to the client, or None."""
        if self.closed:
            return None
        self.last_activity = time.monotonic()
        if char in ("\r", "\n"):
            line = self.buffer.strip()
            self.buffer = ""
            text, closed = await self.handle_line(line)
            self.closed = closed
            return text
        if char == "\x7f":  # backspace
            self.buffer = self.buffer[:-1]
            return None
        if len(self.buffer) < 200:
            self.buffer += char
        return None

    # ------------------------------------------------------------------ #
    # Line handling
    # ------------------------------------------------------------------ #
    def _error(self, message: str) -> str:
        return f"\x1b[91m{message}\x1b[0m\r\n{self.current_prompt()}"

    async def handle_line(self, line: str) -> Tuple[str, bool]:
        """Process a submitted line. Returns (text_to_send, close_session)."""
        if self.state == "HOME_TEAM":
            return self._set_team("home", line, "AWAY_TEAM")
        if self.state == "AWAY_TEAM":
            return self._set_team("away", line, "HOME_SHOTS")

        if self.state in STAT_BY_STATE:
            side, key, _ = STAT_BY_STATE[self.state]
            try:
                value = float(line) if line else float(self.service.defaults[side][key])
                self.service.validate_stat(key, value)
            except ValidationError:
                return self._error("Invalid value. Please enter a valid number."), False
            except (ValueError, TypeError):
                return self._error("Invalid value. Please enter a number."), False
            (self.home_stats if side == "home" else self.away_stats)[key] = value
            self.state = self._next_stat_state()
            if self.state == "PREDICTING":
                return await self._run_prediction()
            return self.current_prompt(), False

        if self.state == "CONTINUE":
            answer = line.lower()
            if answer in ("y", "yes"):
                self.home_team = None
                self.away_team = None
                self.home_stats = {}
                self.away_stats = {}
                self.state = "HOME_TEAM"
                return self.current_prompt(), False
            if answer in ("n", "no", ""):
                self.closed = True
                return "\r\nThanks for using the predictor!\r\n", True
            return self._error('Please answer "y" or "n".'), False

        return "", False

    def _set_team(self, which: str, line: str, next_state: str) -> Tuple[str, bool]:
        if not line:
            return self._error("Team name cannot be empty."), False
        if len(line) > self.service.config["max_name_length"]:
            return self._error(f"Team name too long (max {self.service.config['max_name_length']})."), False
        if which == "home":
            self.home_team = line
        else:
            self.away_team = line
        self.state = next_state
        return self.current_prompt(), False

    def _next_stat_state(self) -> str:
        states = [s for s, *_ in STAT_STEPS]
        idx = states.index(self.state)
        return states[idx + 1] if idx + 1 < len(states) else "PREDICTING"

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    async def _run_prediction(self) -> Tuple[str, bool]:
        result = await self.service.predict_async(self.home_stats, self.away_stats)
        self.state = "CONTINUE"
        text = self._format_result(result) + self.current_prompt()
        return text, False

    def _format_result(self, result: Dict[str, Any]) -> str:
        home, away = self.home_team or "Home", self.away_team or "Away"
        p = result["probabilities"]
        home_w, draw, away_w = p["home_win"] * 100, p["draw"] * 100, p["away_win"] * 100
        pred = result["prediction"]

        def bar(value: float) -> str:
            return "#" * int(value / 2)

        def entry(label: str, value: float, selected: bool, color: str) -> str:
            if not selected:
                return f"{label.ljust(24)}: {value:5.2f}% {bar(value)}"
            return f"{color}{BOLD}{label.ljust(24)}: {value:5.2f}% {bar(value)}{END}"

        lines = [
            "\r\n" + "=" * 60,
            f"{BOLD}PREDICTION: {home} vs {away}{END}",
            "=" * 60,
            entry(f"{home} Win", home_w, pred == "H", GREEN),
            entry("Draw", draw, pred == "D", YELLOW),
            entry(f"{away} Win", away_w, pred == "A", BLUE),
            "=" * 60,
        ]
        if pred == "H":
            lines.append(f"\r\n{GREEN}{BOLD}Prediction: {home} WINS{END}\r\n")
        elif pred == "D":
            lines.append(f"\r\n{YELLOW}{BOLD}Prediction: DRAW{END}\r\n")
        else:
            lines.append(f"\r\n{BLUE}{BOLD}Prediction: {away} WINS{END}\r\n")
        return "\r\n".join(lines) + "\r\n"