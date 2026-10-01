"""Webcam login tools that run in a desktop window.

    python -m cyberbullying.webcam enroll alice   # save alice's face
    python -m cyberbullying.webcam login alice    # face check, then blink the PIN
    python -m cyberbullying.webcam pin            # just try blink typing

They use the web app's database, so someone who registered in the browser can
log in here with their face and PIN, hands free. Press q or Esc to give up.
"""

import argparse
import sys
import time
from collections import deque

import cv2
import numpy as np

from . import db, downloads
from .blink import KEY_NAMES, KEYS, BlinkKeypad, EyeState
from .face import MATCH_THRESHOLD, FaceMatcher, NoFaceError, average, similarity
from .morse import encode
from .paths import DATABASE

WINDOW = "Cyberbullying Detection"
QUIT_KEYS = {ord("q"), ord("Q"), 27}
WHITE, GREY, DARK, GREEN = (255, 255, 255), (150, 150, 150), (35, 35, 35), (80, 200, 120)


class Cancelled(Exception):
    pass


class FaceCheck:
    """Passes once `needed` of the last `window` frames match the enrolled face.

    One lucky frame isn't enough, and one blurry frame doesn't ruin it.
    """

    def __init__(self, enrolled: np.ndarray, needed: int = 3, window: int = 5, threshold: float = MATCH_THRESHOLD):
        self.enrolled = enrolled
        self.needed = needed
        self.threshold = threshold
        self.recent = deque(maxlen=window)

    def add(self, embedding: np.ndarray | None) -> bool:
        matched = embedding is not None and similarity(embedding, self.enrolled) >= self.threshold
        self.recent.append(matched)
        return sum(self.recent) >= self.needed


class EyeTracker:
    """How closed the eyes are, from MediaPipe's blink blendshape scores."""

    def __init__(self, model_path):
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ms = -1

    def closure(self, frame: np.ndarray) -> float | None:
        """0 means wide open, 1 means shut. None when there's no face."""
        import mediapipe as mp

        # Video mode needs timestamps that always go up.
        ms = max(int(time.monotonic() * 1000), self._last_ms + 1)
        self._last_ms = ms
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)
        if not result.face_blendshapes:
            return None
        scores = {c.category_name: c.score for c in result.face_blendshapes[0]}
        return (scores["eyeBlinkLeft"] + scores["eyeBlinkRight"]) / 2

    def close(self) -> None:
        self._landmarker.close()


# The three steps. Each takes a stream of frames and a `show` callback, so
# they can run against a real camera or against test images.

def capture_face(frames, matcher, show, count: int = 5, gap: float = 0.4, clock=time.monotonic) -> np.ndarray:
    embeddings, last = [], -gap
    for frame in frames:
        now = clock()
        status = f"Look at the camera. Captured {len(embeddings)} of {count}."
        if now - last >= gap:
            try:
                embeddings.append(matcher.embed(frame))
                last = now
            except NoFaceError:
                status = "No face in view. Move into the frame."
        show(frame, [status])
        if len(embeddings) == count:
            return average(embeddings)
    raise Cancelled


def verify_face(frames, matcher, enrolled, show, timeout: float = 20, clock=time.monotonic) -> bool:
    check = FaceCheck(enrolled)
    started = clock()
    for frame in frames:
        try:
            embedding = matcher.embed(frame)
        except NoFaceError:
            embedding = None
        if check.add(embedding):
            return True
        left = timeout - (clock() - started)
        if left <= 0:
            return False
        show(frame, [f"Checking your face... {left:.0f}s"])
    return False


def enter_pin(frames, tracker, show, pin_length: int = 4, clock=time.monotonic, calibration_frames: int = 20) -> str:
    keypad = BlinkKeypad(pin_length=pin_length)
    eyes, resting = None, []
    message = "Shut your eyes for half a second to press the lit key."
    for frame in frames:
        now = clock()
        score = tracker.closure(frame)
        if eyes is None:
            # First learn what this person's open eyes look like.
            if score is not None:
                resting.append(score)
            if len(resting) >= calibration_frames:
                eyes = EyeState.calibrated(resting)
            show(frame, ["Look at the screen with your eyes open for a moment."])
            continue

        event = keypad.update(None if score is None else eyes.update(score), now)
        if event == "invalid":
            message = "That wasn't a digit, so those five symbols were cleared."
        elif event in ("digit", "done"):
            message = f"Got digit {len(keypad.digits)} of {pin_length}."
        elif event:
            message = f"Pressed {event}."
        show(frame, [message if score is not None else "Can't see your face."], keypad=keypad, now=now)
        if keypad.done:
            return keypad.pin
    raise Cancelled


# Camera and window plumbing

def camera(index: int):
    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        raise SystemExit(f"Couldn't open camera {index}. Is another app using it?")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise SystemExit("The camera stopped sending frames.")
            yield frame
    finally:
        capture.release()


def show_window(frame, lines, keypad=None, now=None) -> None:
    view = cv2.flip(frame, 1)  # a mirror image feels natural; the models see the original
    for i, line in enumerate(lines):
        _text(view, line, (16, 36 + 30 * i), 0.7)
    if keypad is not None:
        _draw_keypad(view, keypad, now)
    cv2.imshow(WINDOW, view)
    if cv2.waitKey(1) & 0xFF in QUIT_KEYS:
        raise Cancelled
    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:  # closed with the X button
        raise Cancelled


def _text(image, text, origin, scale, color=WHITE):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, DARK, 4, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _draw_keypad(image, keypad, now):
    height, width = image.shape[:2]
    box = width // len(KEYS)
    for i, key in enumerate(KEYS):
        lit = i == keypad.lit
        top_left, bottom_right = (i * box + 10, height - 100), ((i + 1) * box - 10, height - 20)
        cv2.rectangle(image, top_left, bottom_right, WHITE if lit else DARK, -1)
        label = KEY_NAMES[key]
        (w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(image, label, (i * box + (box - w) // 2, height - 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, DARK if lit else GREY, 2, cv2.LINE_AA)
    progress = keypad.hold_progress(now)
    if progress:
        cv2.rectangle(image, (10, height - 112), (10 + int((width - 20) * progress), height - 106), GREEN, -1)
    entered = "*" * len(keypad.digits) + "_" * (keypad.pin_length - len(keypad.digits))
    _text(image, f"PIN {entered}   this digit: {keypad.code or 'nothing yet'}", (16, height - 124), 0.7)


def _face_matcher() -> FaceMatcher:
    for name in ("face-detector", "face-recognizer"):
        downloads.fetch(name)
    return FaceMatcher.from_downloads()


def _eye_tracker() -> EyeTracker:
    return EyeTracker(downloads.fetch("eye-tracker"))


# Commands

def cmd_enroll(args) -> int:
    conn = db.connect(args.db)
    user = db.find_user(conn, args.username)
    if user is None:
        print(f"No user called {args.username}. Register in the web app first.")
        return 1
    matcher = _face_matcher()
    embedding = capture_face(camera(args.camera), matcher, show_window)
    db.set_face(conn, user["id"], embedding)
    print(f"Saved {user['username']}'s face.")
    return 0


def cmd_login(args) -> int:
    conn = db.connect(args.db)
    user = db.find_user(conn, args.username)
    enrolled = db.face_of(user) if user else None
    if user is None or enrolled is None or not user["pin_hash"]:
        print("That account needs a saved face and a PIN. Set both on the Account page of the web app.")
        return 1

    matcher, tracker = _face_matcher(), _eye_tracker()
    frames = camera(args.camera)
    try:
        if not verify_face(frames, matcher, enrolled, show_window):
            print("Face didn't match.")
            return 1
        pin = enter_pin(frames, tracker, show_window, pin_length=4)
    finally:
        frames.close()
        tracker.close()

    if not db.check_pin(user, pin):
        print("Wrong PIN.")
        return 1
    print(f"Welcome back, {user['username']}. Face and PIN both matched.")
    return 0


def cmd_pin(args) -> int:
    tracker = _eye_tracker()
    try:
        pin = enter_pin(camera(args.camera), tracker, show_window, pin_length=args.length)
    finally:
        tracker.close()
    print(f"You blinked {pin} ({encode(pin)}).")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Webcam login tools.")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--db", default=str(DATABASE), help="database file shared with the web app")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("enroll", help="save a user's face").add_argument("username")
    commands.add_parser("login", help="log in with face and a blinked PIN").add_argument("username")
    pin = commands.add_parser("pin", help="practise typing digits by blinking")
    pin.add_argument("--length", type=int, default=4)
    args = parser.parse_args(argv)

    handler = {"enroll": cmd_enroll, "login": cmd_login, "pin": cmd_pin}[args.command]
    try:
        return handler(args)
    except Cancelled:
        print("Cancelled.")
        return 1
    except downloads.DownloadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
