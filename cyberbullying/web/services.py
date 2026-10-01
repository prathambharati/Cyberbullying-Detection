"""Shared things the views need: the database, the classifier, the face matcher
and the news cache.

The models load on first use, so starting the app (or running a CLI command)
stays quick. Tests can hand in their own objects through the app config:
CLASSIFIER, FACE_MATCHER and NEWS_CACHE. Setting FACE_MATCHER to False turns
face login off.
"""

import threading

from flask import current_app, g

from .. import db
from ..classifier import Classifier
from ..face import FaceMatcher
from ..news import NewsCache

_lock = threading.Lock()


def get_db():
    if "db" not in g:
        g.db = db.connect(current_app.config["DATABASE"])
    return g.db


def classifier() -> Classifier:
    return _shared("CLASSIFIER", lambda: Classifier.load(threshold=current_app.config.get("THRESHOLD")))


def face_matcher() -> FaceMatcher | None:
    """None when the face models haven't been downloaded."""
    return _shared("FACE_MATCHER", FaceMatcher.from_downloads)


def news_cache() -> NewsCache:
    return _shared("NEWS_CACHE", NewsCache)


def _shared(name: str, build):
    configured = current_app.config.get(name)
    if configured is False:
        return None
    if configured is not None:
        return configured
    store = current_app.extensions.setdefault("cyberbullying", {})
    with _lock:
        if name not in store:
            store[name] = build()
    return store[name]
