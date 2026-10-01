"""Registering, logging in (password, or face plus PIN) and account settings."""

import re
import statistics

from flask import Blueprint, flash, g, jsonify, redirect, render_template, request, session, url_for

from .. import db
from ..face import MATCH_THRESHOLD, NoFaceError, average, decode_image, similarity
from .security import log_in, login_required, safe_next
from .services import face_matcher, get_db

bp = Blueprint("auth", __name__)

USERNAME = re.compile(r"[A-Za-z0-9_]{3,20}")
PIN = re.compile(r"\d{4}")
MAX_FRAMES = 5


@bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    if not USERNAME.fullmatch(username):
        error = "Usernames are 3 to 20 letters, numbers or underscores."
    elif len(password) < 8:
        error = "Pick a password with at least 8 characters."
    elif password != request.form.get("confirm", ""):
        error = "The two passwords don't match."
    else:
        try:
            user_id = db.create_user(get_db(), username, password)
        except db.UsernameTaken:
            error = "That username is taken."
        else:
            log_in(db.get_user(get_db(), user_id))
            flash(f"Welcome, {username}! Say hi to the feed.")
            return redirect(url_for("views.feed"))
    return render_template("register.html", error=error, username=username), 422


@bp.route("/login", methods=["GET", "POST"])
def login():
    target = safe_next(request.values.get("next"), url_for("views.feed"))
    if request.method == "GET":
        return render_template("login.html", next=target)

    username = request.form.get("username", "").strip()
    user = db.find_user(get_db(), username)
    if user is None or not db.check_password(user, request.form.get("password", "")):
        return render_template("login.html", error="Wrong username or password.", username=username, next=target), 401
    log_in(user)
    flash(f"Welcome back, {user['username']}.")
    return redirect(target)


@bp.post("/logout")
def logout():
    session.clear()
    flash("You're logged out.")
    return redirect(url_for("views.feed"))


@bp.route("/login/face", methods=["GET", "POST"])
def face_login():
    matcher = face_matcher()
    if request.method == "GET":
        return render_template("face_login.html", available=matcher is not None)
    if matcher is None:
        return _reply(False, "Face login is switched off on this server.", 503)

    frames = _read_frames()
    if not frames:
        return _reply(False, "No camera pictures arrived. Allow camera access and try again.", 400)
    embeddings = _embed(matcher, frames)
    if len(embeddings) * 2 < len(frames):
        return _reply(False, "Couldn't see your face clearly. Face the camera in good light.", 422)

    user = db.find_user(get_db(), request.form.get("username", "").strip())
    enrolled = db.face_of(user) if user else None
    matched = enrolled is not None and statistics.median(similarity(e, enrolled) for e in embeddings) >= MATCH_THRESHOLD
    # One vague message for every failure, so it doesn't reveal which part was wrong.
    if not matched or not db.check_pin(user, request.form.get("pin", "")):
        return _reply(False, "Your face or PIN didn't match.", 401)

    log_in(user)
    flash(f"Welcome back, {user['username']}. Face and PIN both matched.")
    return _reply(True, "Logged in.", redirect=url_for("views.feed"))


@bp.get("/account")
@login_required
def account():
    return render_template(
        "account.html",
        has_face=g.user["face_embedding"] is not None,
        has_pin=bool(g.user["pin_hash"]),
        face_available=face_matcher() is not None,
    )


@bp.post("/account/pin")
@login_required
def set_pin():
    pin = request.form.get("pin", "")
    if not db.check_password(g.user, request.form.get("password", "")):
        flash("That password isn't right.", "error")
    elif not PIN.fullmatch(pin):
        flash("The PIN has to be exactly 4 digits.", "error")
    elif pin != request.form.get("confirm", ""):
        flash("The two PINs don't match.", "error")
    else:
        db.set_pin(get_db(), g.user["id"], pin)
        flash("PIN saved.")
    return redirect(url_for("auth.account"))


@bp.post("/account/face")
@login_required
def save_face():
    matcher = face_matcher()
    if matcher is None:
        return _reply(False, "Face login is switched off on this server.", 503)
    if not db.check_password(g.user, request.form.get("password", "")):
        return _reply(False, "That password isn't right.", 401)

    embeddings = _embed(matcher, _read_frames())
    if len(embeddings) < 2:
        return _reply(False, "Couldn't see your face in enough of the pictures. Try again in better light.", 422)
    db.set_face(get_db(), g.user["id"], average(embeddings))
    if g.user["pin_hash"]:
        flash("Face saved. You can log in with your face and PIN now.")
    else:
        flash("Face saved. Set a PIN too, and face login is ready.")
    return _reply(True, "Saved.", redirect=url_for("auth.account"))


@bp.post("/account/face/delete")
@login_required
def delete_face():
    db.set_face(get_db(), g.user["id"], None)
    flash("Your face data is deleted.")
    return redirect(url_for("auth.account"))


def _read_frames() -> list:
    images = []
    for upload in request.files.getlist("frames")[:MAX_FRAMES]:
        image = decode_image(upload.read())
        if image is not None:
            images.append(image)
    return images


def _embed(matcher, frames) -> list:
    embeddings = []
    for frame in frames:
        try:
            embeddings.append(matcher.embed(frame))
        except NoFaceError:
            pass
    return embeddings


def _reply(ok: bool, message: str, status: int = 200, redirect: str | None = None):
    body = {"ok": ok, "message": message}
    if redirect:
        body["redirect"] = redirect
    return jsonify(body), status
