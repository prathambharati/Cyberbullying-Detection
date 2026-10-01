"""The web app: a small feed where every post and comment is checked before it goes up.

    flask --app cyberbullying.web run

Settings can come from environment variables with a CB_ prefix, for example
CB_SECRET_KEY, CB_DATABASE or CB_THRESHOLD.
"""

import secrets
from datetime import datetime, timezone
from pathlib import Path

import click
from flask import Flask, g, render_template

from .. import db
from ..news import DEFAULT_FEED
from ..paths import DATABASE, INSTANCE_DIR
from . import auth, security, views
from .services import classifier, get_db


def create_app(config: dict | None = None) -> Flask:
    app = Flask(__name__, instance_path=str(INSTANCE_DIR))
    app.config.update(
        DATABASE=str(DATABASE),
        MAX_TEXT_LENGTH=1000,
        MAX_CONTENT_LENGTH=6 * 1024 * 1024,  # a few webcam snapshots
        NEWS_FEED_URL=DEFAULT_FEED,
        SESSION_COOKIE_SAMESITE="Lax",
        THRESHOLD=None,  # None means use the threshold tuned during training
    )
    app.config.from_prefixed_env("CB")
    if config:
        app.config.update(config)
    if not app.config.get("SECRET_KEY"):
        app.config["SECRET_KEY"] = _secret_key(Path(app.instance_path) / "secret_key")

    @app.teardown_appcontext
    def close_db(_error):
        conn = g.pop("db", None)
        if conn is not None:
            conn.close()

    security.init_app(app)
    app.register_blueprint(auth.bp)
    app.register_blueprint(views.bp)
    app.add_template_filter(_ago, "ago")

    for status in (400, 404, 405, 413, 500):
        app.register_error_handler(status, _error_page)

    @app.cli.command("seed")
    def seed():
        """Add two demo users and a few posts so the feed isn't empty."""
        if _seed_demo(get_db()):
            click.echo("Added demo users maya and sam (password: demo-password) and a few posts.")
        else:
            click.echo("The feed already has posts, so nothing was added.")

    return app


def _secret_key(path: Path) -> str:
    """Keep one random key per install so logins survive restarts."""
    if path.exists():
        return path.read_text().strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_hex(32)
    path.write_text(key)
    return key


def _error_page(error):
    code = getattr(error, "code", 500)
    messages = {
        400: getattr(error, "description", "That request didn't make sense."),
        404: "There's nothing here.",
        405: "That page doesn't take this kind of request.",
        413: "That upload is too big.",
        500: "Something broke on our side.",
    }
    return render_template("error.html", code=code, message=messages.get(code, "Something went wrong.")), code


def _ago(timestamp: str) -> str:
    seconds = (datetime.now(timezone.utc) - datetime.fromisoformat(timestamp)).total_seconds()
    for size, unit in ((86400, "day"), (3600, "hour"), (60, "minute")):
        if seconds >= size:
            n = int(seconds // size)
            return f"{n} {unit}{'s' if n != 1 else ''} ago"
    return "just now"


DEMO_POSTS = [
    ("maya", "Finally finished my first 10k run this morning. Legs are jelly but I'm so happy!"),
    ("maya", "Does anyone have a good recipe for dal makhani? Mine always turns out too thin."),
    ("sam", "Our robotics team made it to the state finals. So proud of everyone who stayed late all month."),
    ("sam", "Reminder that the library is open until midnight during exam week. Good luck, everyone."),
]


def _seed_demo(conn) -> bool:
    if db.recent_posts(conn, limit=1):
        return False
    ids = {}
    for name in ("maya", "sam"):
        user = db.find_user(conn, name)
        ids[name] = user["id"] if user else db.create_user(conn, name, "demo-password")
    predictions = classifier().predict([body for _, body in DEMO_POSTS])
    for (name, body), prediction in zip(DEMO_POSTS, predictions):
        db.add_post(conn, ids[name], body, prediction.score)
    return True
