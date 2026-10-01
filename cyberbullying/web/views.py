"""The feed, posts and comments, the model playground, the JSON API and the news page."""

from flask import Blueprint, abort, current_app, flash, g, jsonify, redirect, render_template, request, url_for

from .. import db
from ..text import tokenize
from .security import login_required
from .services import classifier, get_db, news_cache

bp = Blueprint("views", __name__)

MAX_API_TEXTS = 100


def moderate(text: str):
    """Check a post before it's saved. Returns (error message or None, prediction)."""
    text = text.strip()
    limit = current_app.config["MAX_TEXT_LENGTH"]
    if not text:
        return "Write something first.", None
    if len(text) > limit:
        return f"Keep it under {limit} characters.", None
    prediction = classifier().predict_one(text)
    if prediction.is_bullying:
        kind = "cyberbullying" if prediction.category == "other" else f"{prediction.category}-based cyberbullying"
        return (
            f"This wasn't posted because it reads like {kind} "
            f"(score {prediction.score:.2f}). Could you say it another way?"
        ), prediction
    return None, prediction


@bp.get("/")
def feed():
    return render_template("feed.html", posts=db.recent_posts(get_db()))


@bp.post("/posts")
@login_required
def create_post():
    body = request.form.get("body", "")
    error, prediction = moderate(body)
    if error:
        return render_template("feed.html", posts=db.recent_posts(get_db()), draft=body, error=error), 422
    db.add_post(get_db(), g.user["id"], body.strip(), prediction.score)
    flash("Posted.")
    return redirect(url_for("views.feed"))


@bp.get("/posts/<int:post_id>")
def show_post(post_id: int):
    post = db.get_post(get_db(), post_id) or abort(404)
    return render_template("post.html", post=post, comments=db.comments_for(get_db(), post_id))


@bp.post("/posts/<int:post_id>/comments")
@login_required
def add_comment(post_id: int):
    post = db.get_post(get_db(), post_id) or abort(404)
    body = request.form.get("body", "")
    error, prediction = moderate(body)
    if error:
        comments = db.comments_for(get_db(), post_id)
        return render_template("post.html", post=post, comments=comments, draft=body, error=error), 422
    db.add_comment(get_db(), post_id, g.user["id"], body.strip(), prediction.score)
    flash("Comment added.")
    return redirect(url_for("views.show_post", post_id=post_id))


@bp.route("/check", methods=["GET", "POST"])
def check():
    """Type anything and see what the model thinks, label by label."""
    text, result, tokens, error = request.form.get("text", ""), None, [], None
    if request.method == "POST":
        limit = current_app.config["MAX_TEXT_LENGTH"]
        if not text.strip():
            error = "Type something to check."
        elif len(text) > limit:
            error = f"Keep it under {limit} characters."
        else:
            result = classifier().predict_one(text)
            vocab = classifier().vocab
            tokens = [(token, token in vocab) for token in tokenize(text)]
    return render_template("check.html", text=text, result=result, tokens=tokens, error=error,
                           threshold=classifier().threshold), 422 if error else 200


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

    results = [
        {"score": p.score, "is_bullying": p.is_bullying, "category": p.category, "categories": p.categories}
        for p in classifier().predict(texts)
    ]
    if "text" in data:
        return jsonify(results[0])
    return jsonify(results=results, threshold=classifier().threshold)


@bp.get("/news")
def news():
    headlines, error = news_cache().get(current_app.config["NEWS_FEED_URL"])
    return render_template("news.html", headlines=headlines, error=error)
