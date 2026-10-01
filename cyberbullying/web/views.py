"""The feed, posts, comments, likes and profiles, the model playground, the JSON API and the news page."""

from dataclasses import dataclass
from urllib.parse import urlsplit

from flask import Blueprint, abort, current_app, flash, g, jsonify, redirect, render_template, request, url_for

from .. import db
from ..classifier import Prediction
from .security import login_required, safe_next
from .services import classifier, get_db, news_cache

bp = Blueprint("views", __name__)

MAX_API_TEXTS = 100

# One click fills the playground with one of these.
EXAMPLES = [
    "Happy birthday! Hope you have an amazing day.",
    "shut up you dumb idiot",
    "Go back to your country, nobody wants your kind here",
    "As a gay man, this show made me feel seen",
    "Nobody at school wants to sit with you, fat freak",
    "Can someone explain how photosynthesis works?",
]


@dataclass
class Review:
    """What the model made of a post, comment or playground text."""

    text: str
    error: str | None = None  # it couldn't be checked: empty or too long
    prediction: Prediction | None = None
    why: list | None = None  # {"word", "weight", "share"} for each word, filled in when it's blocked

    @property
    def blocked(self) -> bool:
        return self.prediction is not None and self.prediction.is_bullying

    @property
    def ok(self) -> bool:
        return self.prediction is not None and not self.prediction.is_bullying


def review(text: str, explain_always: bool = False) -> Review:
    text = text.strip()
    limit = current_app.config["MAX_TEXT_LENGTH"]
    if not text:
        return Review(text, error="Write something first.")
    if len(text) > limit:
        return Review(text, error=f"Keep it under {limit} characters.")
    result = Review(text, prediction=classifier().predict_one(text))
    if result.blocked or explain_always:
        result.why = explain(text)
    return result


def explain(text: str) -> list[dict]:
    weights = classifier().explain(text)
    top = max((weight for _, weight in weights), default=0.0)
    return [{"word": word, "weight": weight, "share": weight / top if top else 0.0} for word, weight in weights]


def _viewer() -> int | None:
    return g.user["id"] if g.user else None


def _feed(**extra):
    conn = get_db()
    return render_template(
        "feed.html",
        posts=db.recent_posts(conn, viewer_id=_viewer()),
        stories=db.recent_posters(conn),
        **extra,
    )


@bp.get("/")
def feed():
    return _feed()


@bp.post("/posts")
@login_required
def create_post():
    result = review(request.form.get("body", ""))
    if not result.ok:
        return _feed(review=result, draft=request.form.get("body", "")), 422
    db.add_post(get_db(), g.user["id"], result.text, result.prediction.score)
    flash("Posted. The model gave it a clean bill of health.")
    return redirect(url_for("views.feed"))


@bp.get("/posts/<int:post_id>")
def show_post(post_id: int):
    post = db.get_post(get_db(), post_id, viewer_id=_viewer()) or abort(404)
    return render_template("post.html", post=post, comments=db.comments_for(get_db(), post_id))


@bp.post("/posts/<int:post_id>/comments")
@login_required
def add_comment(post_id: int):
    post = db.get_post(get_db(), post_id, viewer_id=_viewer()) or abort(404)
    result = review(request.form.get("body", ""))
    if not result.ok:
        comments = db.comments_for(get_db(), post_id)
        return render_template("post.html", post=post, comments=comments, review=result,
                               draft=request.form.get("body", "")), 422
    db.add_comment(get_db(), post_id, g.user["id"], result.text, result.prediction.score)
    flash("Comment added.")
    return redirect(url_for("views.show_post", post_id=post_id))


@bp.post("/posts/<int:post_id>/like")
@login_required
def like(post_id: int):
    db.get_post(get_db(), post_id) or abort(404)
    liked, likes = db.toggle_like(get_db(), g.user["id"], post_id)
    if request.headers.get("X-Requested-With") == "fetch":
        return jsonify(liked=liked, likes=likes)
    # Without JavaScript, go back to the page the like came from, but only ever within this site.
    return redirect(safe_next(urlsplit(request.referrer or "").path, url_for("views.feed")))


@bp.post("/posts/<int:post_id>/delete")
@login_required
def delete_post(post_id: int):
    post = db.get_post(get_db(), post_id) or abort(404)
    if post["user_id"] != g.user["id"]:
        abort(403)
    db.delete_post(get_db(), post_id)
    flash("Post deleted.")
    return redirect(url_for("views.feed"))


@bp.get("/u/<username>")
def profile(username: str):
    conn = get_db()
    person = db.find_user(conn, username) or abort(404)
    return render_template(
        "profile.html",
        person=person,
        stats=db.user_stats(conn, person["id"]),
        posts=db.posts_by(conn, person["id"], viewer_id=_viewer()),
    )


@bp.route("/check", methods=["GET", "POST"])
def check():
    """Type anything and see what the model thinks, and why."""
    result, tokens = None, []
    if request.method == "POST":
        result = review(request.form.get("text", ""), explain_always=True)
        if result.prediction:
            vocab = classifier().vocab
            tokens = [(item["word"], item["word"] in vocab, item["share"]) for item in result.why or []]
    return render_template(
        "check.html",
        text=request.form.get("text", ""),
        review=result,
        tokens=tokens,
        examples=EXAMPLES,
        threshold=classifier().threshold,
    ), 422 if result and result.error else 200


@bp.post("/api/score")
def api_score():
    """Score text as JSON.

    Send {"text": "..."} for one result, or {"texts": [...]} for up to 100.
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or ("text" in data) == ("texts" in data):
        return jsonify(error='Send JSON with either "text" or "texts".'), 400
    texts = [data["text"]] if "text" in data else data["texts"]
    limit = current_app.config["MAX_TEXT_LENGTH"]
    if not isinstance(texts, list) or not 0 < len(texts) <= MAX_API_TEXTS:
        return jsonify(error=f'"texts" has to be a list of 1 to {MAX_API_TEXTS} strings.'), 400
    if not all(isinstance(t, str) and t.strip() and len(t) <= limit for t in texts):
        return jsonify(error=f"Every text has to be a non-empty string of at most {limit} characters."), 400

    threshold = classifier().threshold
    results = [
        {"score": p.score, "is_bullying": p.is_bullying, "category": p.category, "categories": p.categories}
        for p in classifier().predict(texts)
    ]
    if "text" in data:
        return jsonify(**results[0], threshold=threshold)
    return jsonify(results=results, threshold=threshold)


@bp.get("/news")
def news():
    headlines, error = news_cache().get(current_app.config["NEWS_FEED_URL"])
    return render_template("news.html", headlines=headlines, error=error)
