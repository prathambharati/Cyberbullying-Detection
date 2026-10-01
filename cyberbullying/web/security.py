"""Sessions, CSRF protection and the login_required decorator."""

import hmac
import secrets
from functools import wraps

from flask import abort, flash, g, redirect, request, session, url_for

from .. import db
from .services import get_db

# The JSON API has no session or cookies to protect.
CSRF_EXEMPT = {"views.api_score"}


def csrf_token() -> str:
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_urlsafe(32)
    return session["csrf_token"]


def init_app(app) -> None:
    app.jinja_env.globals["csrf_token"] = csrf_token

    @app.before_request
    def load_user():
        g.user = None
        if request.endpoint == "static":
            return
        user_id = session.get("user_id")
        if user_id is not None:
            g.user = db.get_user(get_db(), user_id)
            if g.user is None:  # the account was deleted
                session.clear()

    @app.before_request
    def check_csrf():
        if request.method != "POST" or request.endpoint in CSRF_EXEMPT:
            return
        sent = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token", "")
        expected = session.get("csrf_token", "")
        if not expected or not hmac.compare_digest(sent, expected):
            abort(400, description="Your session expired. Reload the page and try again.")


def log_in(user) -> None:
    session.clear()  # a fresh session on login stops session fixation
    session["user_id"] = user["id"]


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if g.user is None:
            flash("Log in first.", "error")
            return redirect(url_for("auth.login", next=request.path))
        return view(*args, **kwargs)

    return wrapped


def safe_next(target: str | None, fallback: str) -> str:
    """Only follow ?next= links that stay on this site."""
    if target and target.startswith("/") and not target.startswith("//") and "\\" not in target:
        return target
    return fallback
