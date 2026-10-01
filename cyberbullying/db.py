"""SQLite storage for users, posts, comments and likes.

Blocked posts are never saved. For each user the database keeps a password
hash, an optional PIN hash and an optional face embedding: 128 numbers, not a
photo. The web app and the webcam tools share this file.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from werkzeug.security import check_password_hash, generate_password_hash

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    pin_hash TEXT,
    face_embedding BLOB,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    body TEXT NOT NULL,
    score REAL NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS comments (
    id INTEGER PRIMARY KEY,
    post_id INTEGER NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    body TEXT NOT NULL,
    score REAL NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS likes (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    post_id INTEGER NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    PRIMARY KEY (user_id, post_id)
);
CREATE INDEX IF NOT EXISTS comments_by_post ON comments (post_id);
CREATE INDEX IF NOT EXISTS likes_by_post ON likes (post_id);
CREATE INDEX IF NOT EXISTS posts_by_user ON posts (user_id);
"""

# Every post query returns the same columns, including how many comments and
# likes it has and whether the person looking at it has liked it.
_POSTS = """
    SELECT posts.*, users.username,
           (SELECT COUNT(*) FROM comments WHERE comments.post_id = posts.id) AS comment_count,
           (SELECT COUNT(*) FROM likes WHERE likes.post_id = posts.id) AS like_count,
           EXISTS (SELECT 1 FROM likes WHERE likes.post_id = posts.id AND likes.user_id = :viewer) AS liked
    FROM posts JOIN users ON users.id = posts.user_id
"""
_NEWEST_FIRST = " ORDER BY posts.created_at DESC, posts.id DESC LIMIT :limit"


class UsernameTaken(ValueError):
    pass


def connect(path) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Users

def create_user(conn, username: str, password: str, created_at: str | None = None) -> int:
    try:
        cursor = conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, generate_password_hash(password), created_at or _now()),
        )
    except sqlite3.IntegrityError as exc:
        raise UsernameTaken(username) from exc
    conn.commit()
    return cursor.lastrowid


def find_user(conn, username: str):
    return conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()


def get_user(conn, user_id: int):
    return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def check_password(user, password: str) -> bool:
    return check_password_hash(user["password_hash"], password)


def set_pin(conn, user_id: int, pin: str) -> None:
    conn.execute("UPDATE users SET pin_hash = ? WHERE id = ?", (generate_password_hash(pin), user_id))
    conn.commit()


def check_pin(user, pin: str) -> bool:
    return bool(user["pin_hash"]) and check_password_hash(user["pin_hash"], pin)


def set_face(conn, user_id: int, embedding: np.ndarray | None) -> None:
    blob = None if embedding is None else np.asarray(embedding, dtype=np.float32).tobytes()
    conn.execute("UPDATE users SET face_embedding = ? WHERE id = ?", (blob, user_id))
    conn.commit()


def face_of(user) -> np.ndarray | None:
    blob = user["face_embedding"]
    return None if blob is None else np.frombuffer(blob, dtype=np.float32)


def user_stats(conn, user_id: int) -> dict:
    def count(sql):
        return conn.execute(sql, (user_id,)).fetchone()[0]

    return {
        "posts": count("SELECT COUNT(*) FROM posts WHERE user_id = ?"),
        "likes": count("SELECT COUNT(*) FROM likes JOIN posts ON posts.id = likes.post_id WHERE posts.user_id = ?"),
        "comments": count("SELECT COUNT(*) FROM comments WHERE user_id = ?"),
    }


def recent_posters(conn, limit: int = 12):
    """People who posted most recently, newest first. The feed shows them as stories."""
    return conn.execute(
        """
        SELECT users.username, MAX(posts.created_at) AS last_post
        FROM posts JOIN users ON users.id = posts.user_id
        GROUP BY users.id
        ORDER BY last_post DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


# Posts, comments and likes

def add_post(conn, user_id: int, body: str, score: float, created_at: str | None = None) -> int:
    cursor = conn.execute(
        "INSERT INTO posts (user_id, body, score, created_at) VALUES (?, ?, ?, ?)",
        (user_id, body, score, created_at or _now()),
    )
    conn.commit()
    return cursor.lastrowid


def recent_posts(conn, limit: int = 50, viewer_id: int | None = None):
    return conn.execute(_POSTS + _NEWEST_FIRST, {"viewer": viewer_id or 0, "limit": limit}).fetchall()


def posts_by(conn, user_id: int, limit: int = 50, viewer_id: int | None = None):
    return conn.execute(
        _POSTS + " WHERE posts.user_id = :author" + _NEWEST_FIRST,
        {"viewer": viewer_id or 0, "author": user_id, "limit": limit},
    ).fetchall()


def get_post(conn, post_id: int, viewer_id: int | None = None):
    return conn.execute(_POSTS + " WHERE posts.id = :post", {"viewer": viewer_id or 0, "post": post_id}).fetchone()


def delete_post(conn, post_id: int) -> None:
    conn.execute("DELETE FROM posts WHERE id = ?", (post_id,))  # comments and likes go with it
    conn.commit()


def toggle_like(conn, user_id: int, post_id: int) -> tuple[bool, int]:
    """Like the post, or unlike it if it was liked already. Returns (liked now, total likes)."""
    removed = conn.execute("DELETE FROM likes WHERE user_id = ? AND post_id = ?", (user_id, post_id)).rowcount
    if not removed:
        conn.execute("INSERT INTO likes (user_id, post_id, created_at) VALUES (?, ?, ?)", (user_id, post_id, _now()))
    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM likes WHERE post_id = ?", (post_id,)).fetchone()[0]
    return not removed, total


def add_comment(conn, post_id: int, user_id: int, body: str, score: float, created_at: str | None = None) -> int:
    cursor = conn.execute(
        "INSERT INTO comments (post_id, user_id, body, score, created_at) VALUES (?, ?, ?, ?, ?)",
        (post_id, user_id, body, score, created_at or _now()),
    )
    conn.commit()
    return cursor.lastrowid


def comments_for(conn, post_id: int):
    return conn.execute(
        """
        SELECT comments.*, users.username
        FROM comments JOIN users ON users.id = comments.user_id
        WHERE comments.post_id = ?
        ORDER BY comments.created_at, comments.id
        """,
        (post_id,),
    ).fetchall()
