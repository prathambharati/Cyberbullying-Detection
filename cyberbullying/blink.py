"""Typing a PIN in Morse code by blinking.

Three keys light up in turn: dot, dash and delete. To press the lit key, close
your eyes and keep them shut for half a second. Normal blinks are shorter than
that, so they don't count. Five presses of dot or dash make one digit.

This module is plain logic driven by timestamps, which keeps it easy to test.
The webcam loop that feeds it lives in webcam.py.
"""

import statistics
from dataclasses import dataclass, field

from .morse import SYMBOLS_PER_DIGIT, decode_digit

KEYS = (".", "-", "del")
KEY_NAMES = {".": "dot", "-": "dash", "del": "delete"}


@dataclass
class EyeState:
    """Turns a 0 to 1 "how closed are the eyes" score into open or closed.

    The gap between the two thresholds stops a score that hovers around one
    value from flickering between states.
    """

    close_above: float = 0.5
    open_below: float = 0.3
    closed: bool = False

    @classmethod
    def calibrated(cls, open_scores: list[float]) -> "EyeState":
        """Thresholds that fit one person's face.

        Some people's open eyes already score 0.3 or more, so the cut-off is
        set relative to their resting score instead of a fixed number. The
        median ignores the odd blink during calibration.
        """
        resting = statistics.median(open_scores)
        close_above = min(0.85, max(0.45, resting + 0.3))
        return cls(close_above=close_above, open_below=close_above - 0.2)

    def update(self, score: float) -> bool:
        if self.closed and score < self.open_below:
            self.closed = False
        elif not self.closed and score > self.close_above:
            self.closed = True
        return self.closed


@dataclass
class BlinkKeypad:
    pin_length: int = 4
    dwell: float = 1.2  # seconds each key stays lit
    hold: float = 0.5  # seconds the eyes must stay shut to press

    digits: str = field(default="", init=False)
    code: str = field(default="", init=False)  # dots and dashes of the digit in progress
    lit: int = field(default=0, init=False)
    _lit_since: float | None = field(default=None, init=False, repr=False)
    _closed_since: float | None = field(default=None, init=False, repr=False)
    _pressed: bool = field(default=False, init=False, repr=False)

    @property
    def done(self) -> bool:
        return len(self.digits) >= self.pin_length

    @property
    def pin(self) -> str | None:
        return self.digits if self.done else None

    @property
    def lit_key(self) -> str:
        return KEYS[self.lit]

    def hold_progress(self, now: float) -> float:
        """How far a press is from registering, 0 to 1. Useful for drawing a bar."""
        if self._closed_since is None or self._pressed:
            return 0.0
        return min(1.0, (now - self._closed_since) / self.hold)

    def update(self, eyes_closed: bool | None, now: float) -> str | None:
        """Feed one video frame.

        `eyes_closed` is None when no face is in view. Returns an event name
        when something happens: "dot", "dash", "delete", "digit", "invalid"
        (five symbols that aren't a digit) or "done".
        """
        if self.done:
            return None
        if self._lit_since is None:
            self._lit_since = now

        if eyes_closed is None:
            # Lost the face. Drop any half-finished press and hold the current key.
            self._closed_since = None
            self._pressed = False
            self._lit_since = now
            return None

        if eyes_closed:
            if self._closed_since is None:
                self._closed_since = now
            if not self._pressed and now - self._closed_since >= self.hold:
                self._pressed = True
                return self._press(now)
            return None  # the lit key waits while the eyes are shut

        if self._closed_since is not None:
            # Eyes just opened. After a press, the next key gets its full time.
            # After an ordinary blink, pick up the countdown where it paused.
            if self._pressed:
                self._lit_since = now
            else:
                self._lit_since += now - self._closed_since
            self._closed_since = None
            self._pressed = False
            return None

        if now - self._lit_since >= self.dwell:
            self.lit = (self.lit + 1) % len(KEYS)
            self._lit_since = now
        return None

    def _press(self, now: float) -> str:
        key = KEYS[self.lit]
        # Start again from "dot" after every press, so the rhythm is predictable.
        self.lit = 0
        self._lit_since = now

        if key == "del":
            if self.code:
                self.code = self.code[:-1]
            elif self.digits:
                self.digits = self.digits[:-1]
            return "delete"

        self.code += key
        if len(self.code) < SYMBOLS_PER_DIGIT:
            return KEY_NAMES[key]

        digit = decode_digit(self.code)
        self.code = ""
        if digit is None:
            return "invalid"
        self.digits += digit
        return "done" if self.done else "digit"
