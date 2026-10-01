from datetime import datetime, timedelta, timezone
from io import BytesIO

from cyberbullying import db
from cyberbullying.news import Headline
from cyberbullying.web import DEMO_COMMENTS, DEMO_POSTS, DEMO_USERS, _ago, _hue, _initial
from tests.conftest import FakeNews, csrf, jpeg, make_app, post, register, solid

RED, BLUE, BLACK = solid((0, 0, 255)), solid((255, 0, 0)), solid((0, 0, 0))
FETCH = {"X-Requested-With": "fetch"}


def page(response) -> str:
    return response.get_data(as_text=True)


def logged_in_user(client):
    with client.session_transaction() as session:
        return session.get("user_id")


def rows(app, sql):
    conn = db.connect(app.config["DATABASE"])
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def send_frames(client, url, picture, count=3, **fields):
    data = {"csrf_token": csrf(client), **fields}
    data["frames"] = [(BytesIO(jpeg(picture)), f"frame{i}.jpg") for i in range(count)]
    return client.post(url, data=data, content_type="multipart/form-data")


def like(client, post_id, **kwargs):
    return client.post(f"/posts/{post_id}/like", headers={"X-CSRF-Token": csrf(client), **FETCH}, **kwargs)


# Accounts

def test_visitors_get_a_welcome(client):
    response = client.get("/")
    assert response.status_code == 200
    html = page(response)
    assert "Share kindly." in html and "Nothing here yet" in html
    assert 'href="/register"' in html


def test_signing_up_logs_you_in(client):
    response = register(client)
    assert response.status_code == 302 and response.headers["Location"] == "/"
    assert logged_in_user(client)
    html = page(client.get("/"))
    assert "What&#39;s on your mind, maya?" in html
    assert "Welcome, maya!" in html  # shown as a toast


def test_sign_up_checks_its_fields(client):
    def attempt(username, password, confirm=None):
        return post(client, "/register", {"username": username, "password": password,
                                          "confirm": password if confirm is None else confirm})

    assert "Usernames are 3 to 20" in page(attempt("a!", "long enough"))
    assert "at least 8 characters" in page(attempt("maya", "short"))
    assert "don&#39;t match" in page(attempt("maya", "long enough", "different"))
    assert attempt("maya", "long enough").status_code == 302
    post(client, "/logout")
    response = attempt("MAYA", "long enough")
    assert response.status_code == 422 and "taken" in page(response)


def test_logging_in_and_out(client):
    register(client)
    post(client, "/logout")
    assert logged_in_user(client) is None

    wrong = post(client, "/login", {"username": "maya", "password": "nope"})
    assert wrong.status_code == 401 and "Wrong username or password" in page(wrong)
    right = post(client, "/login", {"username": "Maya", "password": "correct horse"})
    assert right.status_code == 302 and logged_in_user(client)


def test_login_only_redirects_within_the_site(client):
    register(client)
    post(client, "/logout")
    credentials = {"username": "maya", "password": "correct horse"}
    assert post(client, "/login", {**credentials, "next": "//evil.example"}).headers["Location"] == "/"
    post(client, "/logout")
    assert post(client, "/login", {**credentials, "next": "/check"}).headers["Location"] == "/check"


def test_forms_need_the_csrf_token(client):
    response = client.post("/register", data={"username": "maya", "password": "correct horse", "confirm": "correct horse"})
    assert response.status_code == 400
    assert "session expired" in page(response)


# Posting and moderation

def test_posting_needs_an_account(client):
    response = post(client, "/posts", {"body": "hello"})
    assert response.status_code == 302 and "/login" in response.headers["Location"]


def test_a_kind_post_is_published(client):
    register(client)
    assert post(client, "/posts", {"body": "Hello everyone, happy Friday!"}).status_code == 302
    html = page(client.get("/"))
    assert "Hello everyone, happy Friday!" in html
    assert "Checked · 0.03" in html


def test_a_bullying_post_is_sent_back_with_the_reason(client, app):
    register(client)
    response = post(client, "/posts", {"body": "you absolute idiot"})
    html = page(response)
    assert response.status_code == 422
    assert "Hold on, this reads like cyberbullying." in html
    assert "you absolute idiot</textarea>" in html  # the draft is kept for editing
    # The word that set it off is highlighted at full strength, the others not at all.
    assert 'style="--w: 1.00" title="Weight 0.90">idiot</mark>' in html
    assert 'style="--w: 0.00" title="Weight 0.00">absolute</mark>' in html
    assert rows(app, "SELECT * FROM posts") == []


def test_empty_and_overlong_posts(client):
    register(client)
    assert "Write something first" in page(post(client, "/posts", {"body": "   "}))
    assert "Keep it under 1000" in page(post(client, "/posts", {"body": "a" * 1001}))


def test_the_composer_is_wired_for_live_scoring(client):
    register(client)
    html = page(client.get("/"))
    assert 'data-composer' in html and 'data-score-url="/api/score"' in html and 'data-threshold="0.5"' in html


def test_comments_are_moderated_too(client, app):
    register(client)
    post(client, "/posts", {"body": "Who wants to study together?"})
    assert post(client, "/posts/1/comments", {"body": "Count me in"}).status_code == 302
    blocked = post(client, "/posts/1/comments", {"body": "not you, idiot"})
    assert blocked.status_code == 422 and "Hold on" in page(blocked)

    thread = page(client.get("/posts/1"))
    assert "Count me in" in thread and 'class="bubble"' in thread
    assert len(rows(app, "SELECT * FROM comments")) == 1
    assert 'aria-label="Comments">' in page(client.get("/")) and "<span>1</span>" in page(client.get("/"))


def test_missing_posts_are_404(client):
    register(client)
    assert client.get("/posts/99").status_code == 404
    assert post(client, "/posts/99/comments", {"body": "hi"}).status_code == 404
    assert "nothing here" in page(client.get("/no-such-page"))


def test_post_bodies_are_escaped(client):
    register(client)
    post(client, "/posts", {"body": "<script>alert(1)</script>"})
    assert "<script>alert(1)" not in page(client.get("/"))


# Likes

def test_liking_and_unliking(client, app):
    register(client)
    post(client, "/posts", {"body": "Sunny day at the beach"})
    assert like(client, 1).get_json() == {"liked": True, "likes": 1}
    assert 'aria-pressed="true"' in page(client.get("/"))
    assert like(client, 1).get_json() == {"liked": False, "likes": 0}
    assert rows(app, "SELECT * FROM likes") == []


def test_likes_count_everyone(client, tmp_path):
    register(client)
    post(client, "/posts", {"body": "Sunny day at the beach"})
    other = make_app(tmp_path).test_client()  # same database, a second person
    register(other, "sam", "another password")
    like(client, 1)
    assert like(other, 1).get_json() == {"liked": True, "likes": 2}


def test_liking_needs_login_a_real_post_and_the_csrf_token(client):
    assert client.post("/posts/1/like", headers={"X-CSRF-Token": csrf(client), **FETCH}).status_code == 302
    register(client)
    assert like(client, 42).status_code == 404
    post(client, "/posts", {"body": "hello"})
    assert client.post("/posts/1/like", headers=FETCH).status_code == 400


def test_liking_without_javascript_goes_back_to_the_same_page(client):
    register(client)
    post(client, "/posts", {"body": "hello"})
    back = post(client, "/posts/1/like", headers={"Referer": "http://localhost/posts/1"})
    assert back.status_code == 302 and back.headers["Location"] == "/posts/1"
    elsewhere = post(client, "/posts/1/like", headers={"Referer": "https://evil.example//steal"})
    assert elsewhere.headers["Location"] == "/"


# Deleting and profiles

def test_only_the_author_can_delete_a_post(client, tmp_path, app):
    register(client)
    post(client, "/posts", {"body": "My first post"})
    other = make_app(tmp_path).test_client()
    register(other, "sam", "another password")
    assert post(other, "/posts/1/delete").status_code == 403
    assert "isn&#39;t yours" in page(post(other, "/posts/1/delete"))

    like(client, 1)
    post(client, "/posts/1/comments", {"body": "Adding a note"})
    response = post(client, "/posts/1/delete")
    assert response.status_code == 302
    assert rows(app, "SELECT * FROM posts") == []
    assert rows(app, "SELECT * FROM likes") == [] and rows(app, "SELECT * FROM comments") == []
    assert post(client, "/posts/1/delete").status_code == 404


def test_profiles_show_posts_and_stats(client):
    register(client)
    post(client, "/posts", {"body": "Morning run done"})
    post(client, "/posts", {"body": "Coffee time"})
    like(client, 1)
    html = page(client.get("/u/maya"))
    assert "Morning run done" in html and "Coffee time" in html
    assert "<strong>2</strong><span>posts</span>" in html
    assert "<strong>1</strong><span>like</span>" in html
    assert "Settings" in html  # it's your own profile
    assert client.get("/u/nobody").status_code == 404


def test_the_feed_shows_recent_posters_as_stories(client, tmp_path):
    other = make_app(tmp_path).test_client()
    register(other, "sam", "another password")
    post(other, "/posts", {"body": "Hi from Sam"})
    register(client)
    html = page(client.get("/"))
    assert 'class="stories"' in html and 'href="/u/sam"' in html
    assert "<span>You</span>" in html


# The model playground and API

def test_check_page(client):
    html = page(client.get("/check"))
    assert 'data-example="shut up you dumb idiot"' in html
    kind = page(post(client, "/check", {"text": "hello there"}))
    assert "This would go live" in kind and 'class="word" style="--w: 0.00">hello</mark>' in kind
    mean = page(post(client, "/check", {"text": "what an idiot"}))
    assert "This would be sent back" in mean and 'class="word" style="--w: 1.00">idiot</mark>' in mean
    assert 'class="word unknown"' in mean  # "what" isn't in the stand-in's tiny vocabulary
    assert post(client, "/check", {"text": " "}).status_code == 422


def test_api_scores_one_or_many_texts(client):
    one = client.post("/api/score", json={"text": "hello"}).get_json()
    assert one["is_bullying"] is False and one["score"] == 0.03 and one["threshold"] == 0.5
    assert set(one["categories"]) == {"age", "ethnicity", "gender", "religion", "other"}
    many = client.post("/api/score", json={"texts": ["hello", "idiot"]}).get_json()
    assert [r["is_bullying"] for r in many["results"]] == [False, True]
    assert many["results"][1]["category"] == "other" and many["threshold"] == 0.5


def test_api_rejects_bad_input(client):
    for body in (None, [], {}, {"text": "a", "texts": ["b"]}, {"texts": "hello"}, {"texts": []},
                 {"text": ""}, {"text": 5}, {"texts": ["ok"] * 101}, {"text": "a" * 1001}):
        response = client.post("/api/score", json=body) if body is not None else client.post("/api/score")
        assert response.status_code == 400, body
        assert "error" in response.get_json()


# News

def test_news_page_lists_headlines_with_pictures(tmp_path):
    with_picture = Headline("Monsoon arrives early", "https://example.com/rain", "Rain everywhere.",
                            datetime.now(timezone.utc) - timedelta(hours=2), "https://example.com/rain.jpg")
    without = Headline("Library hours extended", "https://example.com/library", "", None)
    client = make_app(tmp_path, NEWS_CACHE=FakeNews([with_picture, without])).test_client()
    html = page(client.get("/news"))
    assert 'href="https://example.com/rain"' in html and "Monsoon arrives early" in html and "2 hours ago" in html
    assert 'src="https://example.com/rain.jpg"' in html and html.count('class="thumb"') == 1


def test_news_page_explains_when_the_feed_is_down(tmp_path):
    client = make_app(tmp_path, NEWS_CACHE=FakeNews(error="The news feed can't be reached right now.")).test_client()
    assert "can&#39;t be reached" in page(client.get("/news"))


# PIN and face login

def test_account_page_needs_login(client):
    assert "/login" in client.get("/account").headers["Location"]


def test_setting_a_pin(client, app):
    register(client)

    def set_pin(pin, confirm=None, password="correct horse"):
        post(client, "/account/pin", {"pin": pin, "confirm": pin if confirm is None else confirm, "password": password})
        return page(client.get("/account"))

    assert "password isn&#39;t right" in set_pin("1234", password="wrong")
    assert "exactly 4 digits" in set_pin("12a4")
    assert "don&#39;t match" in set_pin("1234", "4321")
    assert "PIN saved" in set_pin("1234")
    user = rows(app, "SELECT * FROM users")[0]
    assert db.check_pin(user, "1234") and not db.check_pin(user, "0000")


def enrol(client, picture=RED, pin="2468"):
    register(client)
    post(client, "/account/pin", {"pin": pin, "confirm": pin, "password": "correct horse"})
    return send_frames(client, "/account/face", picture, password="correct horse")


def test_saving_a_face(client, app):
    response = enrol(client)
    assert response.status_code == 200 and response.get_json()["ok"]
    assert "Face saved" in page(client.get("/account"))
    assert rows(app, "SELECT face_embedding FROM users")[0][0] is not None


def test_saving_a_face_needs_the_password_and_a_visible_face(client):
    register(client)
    assert send_frames(client, "/account/face", RED, password="wrong").status_code == 401
    dark = send_frames(client, "/account/face", BLACK, password="correct horse")
    assert dark.status_code == 422 and "enough of the pictures" in dark.get_json()["message"]


def test_face_and_pin_login(client):
    enrol(client)
    post(client, "/logout")
    response = send_frames(client, "/login/face", RED, username="maya", pin="2468")
    assert response.status_code == 200 and response.get_json() == {"ok": True, "message": "Logged in.", "redirect": "/"}
    assert logged_in_user(client)


def test_face_login_failures(client):
    enrol(client)
    post(client, "/logout")
    wrong_face = send_frames(client, "/login/face", BLUE, username="maya", pin="2468")
    wrong_pin = send_frames(client, "/login/face", RED, username="maya", pin="0000")
    stranger = send_frames(client, "/login/face", RED, username="nobody", pin="2468")
    for response in (wrong_face, wrong_pin, stranger):
        assert response.status_code == 401
        assert response.get_json()["message"] == "Your face or PIN didn't match."
    no_face = send_frames(client, "/login/face", BLACK, username="maya", pin="2468")
    assert no_face.status_code == 422 and "see your face" in no_face.get_json()["message"]
    no_frames = send_frames(client, "/login/face", RED, count=0, username="maya", pin="2468")
    assert no_frames.status_code == 400
    assert logged_in_user(client) is None


def test_deleting_face_data(client, app):
    enrol(client)
    post(client, "/account/face/delete")
    assert rows(app, "SELECT face_embedding FROM users")[0][0] is None
    post(client, "/logout")
    assert send_frames(client, "/login/face", RED, username="maya", pin="2468").status_code == 401


def test_face_login_can_be_switched_off(tmp_path):
    client = make_app(tmp_path, FACE_MATCHER=False).test_client()
    assert "Face login is off" in page(client.get("/login/face"))
    assert send_frames(client, "/login/face", RED, username="maya", pin="2468").status_code == 503
    register(client)
    assert "Face login is off" in page(client.get("/account"))


# Odds and ends

def test_seed_command_fills_the_feed_once(app):
    runner = app.test_cli_runner()
    output = runner.invoke(args=["seed"]).output
    assert f"Added {len(DEMO_USERS)} demo users" in output
    assert "nothing was added" in runner.invoke(args=["seed"]).output
    assert len(rows(app, "SELECT * FROM posts")) == len(DEMO_POSTS)
    assert len(rows(app, "SELECT * FROM comments")) == len(DEMO_COMMENTS)
    assert len(rows(app, "SELECT * FROM likes")) > len(DEMO_POSTS)
    newest = rows(app, "SELECT body FROM posts ORDER BY created_at DESC LIMIT 1")[0][0]
    assert newest == DEMO_POSTS[-1][2]


def test_every_page_has_the_app_shell(client):
    html = page(client.get("/check"))
    assert 'class="sidebar"' in html and 'class="tabbar"' in html and "data-theme-toggle" in html
    assert 'name="csrf-token"' in html and "Kindfeed" in html


def test_the_app_name_can_be_changed(tmp_path):
    client = make_app(tmp_path, APP_NAME="Hearth").test_client()
    assert "Hearth" in page(client.get("/")) and "Kindfeed" not in page(client.get("/"))


def test_method_not_allowed_page(client):
    response = client.get("/logout")
    assert response.status_code == 405 and "kind of request" in page(response)


def test_avatar_helpers():
    assert _hue("maya") == _hue("MAYA") and 0 <= _hue("maya") < 360 and _hue("maya") != _hue("sam")
    assert _initial("maya") == "M" and _initial("_x") == "X" and _initial("___") == "?"


def test_ago():
    now = datetime.now(timezone.utc)
    assert _ago(now.isoformat()) == "just now"
    assert _ago((now - timedelta(minutes=1, seconds=5)).isoformat()) == "1 minute ago"
    assert _ago((now - timedelta(hours=5)).isoformat()) == "5 hours ago"
    assert _ago((now - timedelta(days=3)).isoformat()) == "3 days ago"


def test_with_the_real_model(tmp_path, classifier):
    client = make_app(tmp_path, CLASSIFIER=classifier).test_client()
    register(client)
    kind = post(client, "/posts", {"body": "Thanks for helping me move this weekend, you're the best"})
    mean = post(client, "/posts", {"body": "shut up you dumb idiot, nobody likes you"})
    assert kind.status_code == 302
    assert mean.status_code == 422 and "Hold on" in page(mean) and "<mark" in page(mean)


def test_the_real_demo_posts_all_pass_moderation(tmp_path, classifier):
    app = make_app(tmp_path, CLASSIFIER=classifier)
    app.test_cli_runner().invoke(args=["seed"])
    assert len(rows(app, "SELECT * FROM posts")) == len(DEMO_POSTS)
    assert len(rows(app, "SELECT * FROM comments")) == len(DEMO_COMMENTS)
