"""The webcam steps, driven by fake frames and a fake clock so no camera or window is needed."""

import numpy as np
import pytest

from cyberbullying.face import MATCH_THRESHOLD, similarity
from cyberbullying.morse import CODES
from cyberbullying.webcam import Cancelled, FaceCheck, capture_face, enter_pin, verify_face
from tests.conftest import ColorMatcher, image, solid

RED, BLUE, BLACK = solid((0, 0, 255)), solid((255, 0, 0)), solid((0, 0, 0))


class Clock:
    def __init__(self, step):
        self.now, self.step = 0.0, step

    def __call__(self):
        self.now += self.step
        return self.now


def no_window(frame, lines, keypad=None, now=None):
    pass


def test_face_check_needs_three_of_the_last_five_frames():
    enrolled = np.array([1.0, 0.0], np.float32)
    match, other = enrolled, np.array([0.0, 1.0], np.float32)
    check = FaceCheck(enrolled)
    assert [check.add(e) for e in (match, None, match, other)] == [False, False, False, False]
    assert check.add(match) is True
    # Old frames drop out of the window.
    check = FaceCheck(enrolled)
    for e in (match, match, other, other, other, other):
        check.add(e)
    assert check.add(match) is False


def test_capture_face_averages_several_frames():
    embedding = capture_face(iter([RED] * 10), ColorMatcher(), no_window, count=5, gap=0.4, clock=Clock(0.5))
    assert similarity(embedding, ColorMatcher().embed(RED)) == pytest.approx(1.0)


def test_capture_face_waits_for_a_face_and_can_run_out_of_frames():
    with pytest.raises(Cancelled):
        capture_face(iter([BLACK] * 20), ColorMatcher(), no_window, clock=Clock(0.5))


def test_verify_face_accepts_the_right_face():
    enrolled = ColorMatcher().embed(RED)
    assert verify_face(iter([RED] * 10), ColorMatcher(), enrolled, no_window, clock=Clock(0.1))


def test_verify_face_gives_up_after_the_timeout():
    enrolled = ColorMatcher().embed(RED)
    frames = iter([BLUE] * 1000)
    assert not verify_face(frames, ColorMatcher(), enrolled, no_window, timeout=5, clock=Clock(0.1))
    assert len(list(frames)) > 900  # stopped at the timeout, not at the end of the frames


def test_verify_face_with_real_photos(face_matcher):
    enrolled = face_matcher.embed(image("sally_ride_1979.jpg"))
    assert verify_face(iter([image("sally_ride_1984.jpg")] * 5), face_matcher, enrolled, no_window, clock=Clock(0.1))
    assert not verify_face(iter([image("chris_hadfield_2011.jpg")] * 50), face_matcher, enrolled, no_window,
                           timeout=2, clock=Clock(0.1))
    assert similarity(enrolled, face_matcher.embed(image("sally_ride_1984.jpg"))) >= MATCH_THRESHOLD


class ScriptedEyes:
    """Plays back eye-closure scores, one per frame, like the MediaPipe tracker would."""

    FPS = 30

    def __init__(self):
        self.scores = [0.05] * 20  # calibration: eyes open

    def hold(self, score, seconds):
        self.scores += [score] * round(seconds * self.FPS)

    def press(self, key):
        waits = {".": 0.0, "-": 1.25, "del": 2.45}
        self.hold(0.05, 0.1 + waits[key])
        self.hold(0.9, 0.6)
        self.hold(0.05, 0.1)

    def closure(self, frame):
        return self.scores.pop(0) if self.scores else 0.05


def test_enter_pin_by_blinking():
    eyes = ScriptedEyes()
    for symbol in CODES["3"] + CODES["9"]:
        eyes.press(symbol)
    frames = iter([RED] * len(eyes.scores))
    assert enter_pin(frames, eyes, no_window, pin_length=2, clock=Clock(1 / 30)) == "39"


def test_enter_pin_can_be_cancelled_by_running_out_of_frames():
    with pytest.raises(Cancelled):
        enter_pin(iter([RED] * 10), ScriptedEyes(), no_window, clock=Clock(1 / 30))


def test_the_real_eye_tracker_sees_open_eyes():
    from cyberbullying import downloads
    from cyberbullying.webcam import EyeTracker

    if not downloads.is_available("eye-tracker"):
        pytest.skip("eye model not downloaded (python -m cyberbullying.downloads)")
    tracker = EyeTracker(downloads.ASSETS["eye-tracker"].path)
    try:
        score = tracker.closure(image("sally_ride_1984.jpg"))
        assert score is not None and 0.0 <= score < 0.3  # eyes open in the portrait
        assert tracker.closure(solid((128, 128, 128))) is None
    finally:
        tracker.close()
