from pathlib import Path

import cv2
import numpy as np
import pytest

from cyberbullying.classifier import Prediction
from cyberbullying.data import KINDS
from cyberbullying.face import NoFaceError
from cyberbullying.paths import MODELS_DIR
from cyberbullying.text import PAD, UNKNOWN, Vocabulary

DATA = Path(__file__).parent / "data"


def image(name: str) -> np.ndarray:
    return cv2.imread(str(DATA / name))


def solid(bgr, size=(120, 160)) -> np.ndarray:
    return np.full((*size, 3), bgr, dtype=np.uint8)


def jpeg(img: np.ndarray) -> bytes:
    ok, data = cv2.imencode(".jpg", img)
    assert ok
    return data.tobytes()


class KeywordClassifier:
    """Stands in for the real model in route tests: anything with "idiot" in it is bullying."""

    threshold = 0.5
    vocab = Vocabulary([PAD, UNKNOWN, "idiot", "hello"])

    def predict(self, texts):
        return [self.predict_one(text) for text in texts]

    def predict_one(self, text):
        bad = "idiot" in text.lower()
        categories = {kind: 0.2 for kind in KINDS}
        return Prediction(score=0.97 if bad else 0.03, is_bullying=bad, category="other" if bad else None,
                          categories=categories)


class ColorMatcher:
    """Stands in for the face models: an image's strongest colour channel is its "face".

    Red pictures match red pictures and not blue ones. Nearly black pictures have no face.
    """

    def embed(self, img):
        mean = img.reshape(-1, 3).mean(axis=0)
        if mean.max() < 20:
            raise NoFaceError("too dark")
        vector = (mean == mean.max()).astype(np.float32)
        return vector / np.linalg.norm(vector)


class FakeNews:
    def __init__(self, headlines=(), error=None):
        self.headlines, self.error = list(headlines), error

    def get(self, url):
        return self.headlines, self.error


@pytest.fixture(scope="session")
def classifier():
    if not (MODELS_DIR / "model_card.json").exists():
        pytest.skip("no trained model in models/")
    from cyberbullying.classifier import Classifier

    return Classifier.load()


@pytest.fixture(scope="session")
def face_matcher():
    from cyberbullying.face import FaceMatcher

    matcher = FaceMatcher.from_downloads()
    if matcher is None:
        pytest.skip("face models not downloaded (python -m cyberbullying.downloads)")
    return matcher


def make_app(tmp_path, **config):
    from cyberbullying.web import create_app

    settings = {
        "TESTING": True,
        "SECRET_KEY": "test-secret",
        "DATABASE": str(tmp_path / "test.db"),
        "CLASSIFIER": KeywordClassifier(),
        "FACE_MATCHER": ColorMatcher(),
        "NEWS_CACHE": FakeNews(),
    }
    settings.update(config)
    return create_app(settings)


@pytest.fixture
def app(tmp_path):
    return make_app(tmp_path)


@pytest.fixture
def client(app):
    return app.test_client()


def csrf(client) -> str:
    """The session's CSRF token, created if the session doesn't have one yet."""
    with client.session_transaction() as session:
        return session.setdefault("csrf_token", "test-csrf-token")


def post(client, url, data=None, **kwargs):
    data = dict(data or {})
    data.setdefault("csrf_token", csrf(client))
    return client.post(url, data=data, **kwargs)


def register(client, username="maya", password="correct horse"):
    return post(client, "/register", {"username": username, "password": password, "confirm": password})
