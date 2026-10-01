import pytest

from cyberbullying.blink import BlinkKeypad, EyeState
from cyberbullying.morse import CODES, DIGITS, decode_digit, encode

FPS = 30


class Eyes:
    """Plays a keypad forward in time, one video frame at a time."""

    def __init__(self, keypad):
        self.keypad = keypad
        self.now = 0.0
        self.events = []

    def hold(self, closed, seconds):
        for _ in range(round(seconds * FPS)):
            self.now += 1 / FPS
            event = self.keypad.update(closed, self.now)
            if event:
                self.events.append(event)
        return self

    def press(self, key):
        """Wait for `key` to light up, then shut the eyes long enough to press it."""
        waits = {".": 0.0, "-": 1.25, "del": 2.45}  # keys restart at "dot" after every press
        return self.hold(False, 0.1 + waits[key]).hold(True, 0.6).hold(False, 0.1)

    def type_code(self, code):
        for symbol in code:
            self.press(symbol)
        return self


def test_morse_digits():
    assert decode_digit("..---") == "2"
    assert decode_digit("-----") == "0"
    assert decode_digit(".-.-.") is None
    assert encode("42") == "....- ..---"
    assert len(DIGITS) == 10 and all(len(code) == 5 for code in DIGITS)


def test_eye_state_has_a_dead_zone():
    eyes = EyeState(close_above=0.5, open_below=0.3)
    assert [eyes.update(s) for s in (0.1, 0.45, 0.6, 0.4, 0.35, 0.2)] == [False, False, True, True, True, False]


def test_calibration_follows_the_resting_score():
    assert EyeState.calibrated([0.05, 0.1, 0.9, 0.05]).close_above == pytest.approx(0.45)
    relaxed = EyeState.calibrated([0.35, 0.4, 0.38])
    assert relaxed.close_above == pytest.approx(0.68)
    assert relaxed.open_below == pytest.approx(0.48)
    assert EyeState.calibrated([0.8, 0.8]).close_above == pytest.approx(0.85)


def test_normal_blinks_do_not_press_anything():
    eyes = Eyes(BlinkKeypad())
    eyes.hold(False, 0.5).hold(True, 0.25).hold(False, 0.5)
    assert eyes.events == []
    assert eyes.keypad.code == ""


def test_holding_the_eyes_shut_presses_the_lit_key():
    eyes = Eyes(BlinkKeypad())
    eyes.press(".").press("-")
    assert eyes.events == ["dot", "dash"]
    assert eyes.keypad.code == ".-"


def test_keeping_the_eyes_shut_only_presses_once():
    eyes = Eyes(BlinkKeypad())
    eyes.hold(True, 3.0)
    assert eyes.events == ["dot"]


def test_keys_do_not_move_while_the_eyes_are_shut():
    keypad = BlinkKeypad()
    eyes = Eyes(keypad).hold(False, 1.0)  # "dot" still lit, 0.2 s left on it
    eyes.hold(True, 0.3)  # a slow blink that isn't a press
    assert keypad.lit_key == "."
    eyes.hold(False, 0.15)
    assert keypad.lit_key == "."  # the countdown resumed where it paused
    eyes.hold(False, 0.2)
    assert keypad.lit_key == "-"


def test_a_full_pin():
    keypad = BlinkKeypad(pin_length=2)
    eyes = Eyes(keypad).type_code(CODES["4"]).type_code(CODES["2"])
    assert keypad.done and keypad.pin == "42"
    assert eyes.events[4] == "digit" and eyes.events[-1] == "done"


def test_delete_removes_the_last_symbol_then_the_last_digit():
    keypad = BlinkKeypad(pin_length=4)
    eyes = Eyes(keypad).type_code(CODES["7"]).press(".")
    assert (keypad.digits, keypad.code) == ("7", ".")
    eyes.press("del")
    assert (keypad.digits, keypad.code) == ("7", "")
    eyes.press("del")
    assert (keypad.digits, keypad.code) == ("", "")
    assert eyes.events[-2:] == ["delete", "delete"]


def test_five_symbols_that_are_not_a_digit_are_thrown_away():
    keypad = BlinkKeypad()
    eyes = Eyes(keypad).type_code(".-.-.")
    assert eyes.events[-1] == "invalid"
    assert keypad.code == "" and keypad.digits == ""


def test_losing_the_face_cancels_a_half_finished_press():
    keypad = BlinkKeypad()
    eyes = Eyes(keypad).hold(True, 0.3).hold(None, 0.2).hold(True, 0.3).hold(False, 0.2)
    assert eyes.events == []


def test_no_more_input_after_the_pin_is_done():
    keypad = BlinkKeypad(pin_length=1)
    eyes = Eyes(keypad).type_code(CODES["1"])
    assert keypad.pin == "1"
    eyes.press(".")
    assert keypad.pin == "1" and eyes.events[-1] == "done"


def test_hold_progress():
    keypad = BlinkKeypad()
    keypad.update(False, 0.0)
    keypad.update(True, 1.0)
    assert keypad.hold_progress(1.25) == pytest.approx(0.5)
    assert keypad.hold_progress(5.0) == 1.0
