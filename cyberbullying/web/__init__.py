"""The web app: a small social feed where every post and comment is checked before it goes up.

    flask --app cyberbullying.web run

Settings can come from environment variables with a CB_ prefix, for example
CB_SECRET_KEY, CB_DATABASE, CB_THRESHOLD or CB_APP_NAME.
"""

import secrets
from datetime import datetime, timedelta, timezone
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
        APP_NAME="Kindfeed",
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
    app.add_template_filter(_month_year, "month_year")
    app.add_template_filter(_hue, "hue")
    app.add_template_filter(_initial, "initial")
    # Only pages with a composer ask for this, so the model isn't loaded just to show a login form.
    app.jinja_env.globals["blocking_threshold"] = lambda: classifier().threshold

    for status in (400, 403, 404, 405, 413, 500):
        app.register_error_handler(status, _error_page)

    @app.cli.command("seed")
    def seed():
        """Add demo people, posts, comments and likes so the feed feels alive."""
        added = _seed_demo(get_db())
        if added:
            click.echo(f"Added {added['users']} demo users (password: demo-password), {added['posts']} posts, "
                       f"{added['comments']} comments and {added['likes']} likes.")
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
        403: "That isn't yours to change.",
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


def _month_year(timestamp: str) -> str:
    return datetime.fromisoformat(timestamp).strftime("%B %Y")


def _hue(name: str) -> int:
    """A stable colour for someone's avatar, picked from their username."""
    return sum(ord(c) * (i + 1) for i, c in enumerate(name.lower())) * 47 % 360


def _initial(name: str) -> str:
    return next((c.upper() for c in name if c.isalnum()), "?")


DEMO_USERS = ("maya", "sam", "arjun", "lena", "kofi")
# (who, hours ago, what they posted), oldest first
DEMO_POSTS = [
    ("lena", 70, "Tried pottery for the first time today. My bowl looks more like an ashtray but I love it anyway."),
    ("arjun", 52, "Anyone else watching the cricket tonight? That last over was unbelievable."),
    ("maya", 30, "Finally finished my first 10k run this morning. Legs are jelly but I'm so happy!"),
    ("kofi", 26, "Our study group finally cracked the problem set that has been haunting us all week. Pizza is on me."),
    ("sam", 20, "Our robotics team made it to the state finals. So proud of everyone who stayed late all month."),
    ("lena", 9, "Sunset from the rooftop tonight. Some days the city is really kind to you."),
    ("maya", 5, "Does anyone have a good recipe for dal makhani? Mine always turns out too thin."),
    ("arjun", 2, "Reminder: be kind in the comments. Everyone here is somebody's favourite person."),
    ("sam", 0.5, "Reminder that the library is open until midnight during exam week. Good luck, everyone."),
]
# (which post, who, minutes after it went up, what they said)
DEMO_COMMENTS = [
    (0, "kofi", 120, "It's a bowl with personality. Keep going!"),
    (2, "kofi", 40, "Congrats! What's next, a half marathon?"),
    (2, "lena", 90, "So proud of you! Stretch well tonight."),
    (4, "maya", 25, "Good luck at the finals! Bring the trophy home."),
    (6, "sam", 30, "Let it simmer longer and add a spoonful of butter at the end. Game changer."),
    (6, "arjun", 55, "My mum adds a little cream right before serving."),
]


def _seed_demo(conn) -> dict | None:
    if db.recent_posts(conn, limit=1):
        return None
    now = datetime.now(timezone.utc)

    def stamp(hours: float) -> str:
        return (now - timedelta(hours=hours)).isoformat(timespec="seconds")

    ids = {}
    for name in DEMO_USERS:
        user = db.find_user(conn, name)
        ids[name] = user["id"] if user else db.create_user(conn, name, "demo-password", created_at=stamp(24 * 40))

    # The demo goes through the same check as everyone else. Anything flagged is left out.
    post_ids, counts = {}, {"users": len(ids), "posts": 0, "comments": 0, "likes": 0}
    for index, ((name, hours, body), prediction) in enumerate(zip(DEMO_POSTS, classifier().predict(p[2] for p in DEMO_POSTS))):
        if not prediction.is_bullying:
            post_ids[index] = db.add_post(conn, ids[name], body, prediction.score, created_at=stamp(hours))
            counts["posts"] += 1
    for (index, name, minutes, body), prediction in zip(DEMO_COMMENTS, classifier().predict(c[3] for c in DEMO_COMMENTS)):
        if index in post_ids and not prediction.is_bullying:
            hours = DEMO_POSTS[index][1] - minutes / 60
            db.add_comment(conn, post_ids[index], ids[name], body, prediction.score, created_at=stamp(hours))
            counts["comments"] += 1
    for index, post_id in post_ids.items():
        author = DEMO_POSTS[index][0]
        # A different handful of people like each post.
        fans = [name for k, name in enumerate(DEMO_USERS) if name != author and (index + k) % 3 != 0]
        for name in fans:
            db.toggle_like(conn, ids[name], post_id)
            counts["likes"] += 1
    return counts
