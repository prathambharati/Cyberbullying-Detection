from datetime import datetime, timezone

import pytest
import requests

from cyberbullying import news
from tests.conftest import DATA

FEED = (DATA / "feed.xml").read_bytes()


def test_parse_feed_strips_html_and_skips_unsafe_links():
    headlines = news.parse_feed(FEED)
    assert [h.title for h in headlines] == [
        "Monsoon reaches Kerala three days early",
        "Library hours extended for exam week",
        "Local team wins the state robotics final",
        "A story whose picture link is not a web address",
    ]
    # Pictures come from the enclosure when there is one, shrunk to a thumbnail.
    assert headlines[2].image == "https://static.toiimg.com/thumb/msid-12345,width-320,resizemode-4/12345.jpg"
    assert headlines[3].image is None
    first = headlines[0]
    assert first.summary == "Rain reached the coast three days ahead of schedule."
    assert first.link == "https://example.com/monsoon"
    assert first.image == "https://example.com/rain.jpg"
    assert first.published == datetime(2026, 9, 28, 4, 0, tzinfo=timezone.utc)
    assert headlines[1].summary == "" and headlines[1].published is None and headlines[1].image is None


def test_thumbnails_only_rewrite_toi_photos():
    assert news.thumbnail("https://static.toiimg.com/photo/msid-99,imgsize-5.cms") == \
        "https://static.toiimg.com/thumb/msid-99,width-320,resizemode-4/99.jpg"
    assert news.thumbnail("https://example.com/cat.jpg") == "https://example.com/cat.jpg"


def test_only_web_images_are_kept():
    assert news._read('<img src="javascript:alert(1)"><img src="https://example.com/b.jpg">') == ("", "https://example.com/b.jpg")
    assert news._read('<img src="data:image/png;base64,AAAA"> text') == ("text", None)


def test_parse_feed_limit():
    assert len(news.parse_feed(FEED, limit=2)) == 2


def test_strip_html():
    assert news.strip_html("<p>Hello <b>there</b>&nbsp;friend</p>") == "Hello there friend"
    assert news.strip_html("") == ""


class FakeResponse:
    def __init__(self, content=b"", status=200):
        self.content, self.status_code = content, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


def test_fetch_headlines_uses_a_timeout(monkeypatch):
    calls = {}

    def fake_get(url, timeout, headers):
        calls.update(url=url, timeout=timeout)
        return FakeResponse(FEED)

    monkeypatch.setattr(news.requests, "get", fake_get)
    assert len(news.fetch_headlines("https://example.com/rss")) == 4
    assert calls == {"url": "https://example.com/rss", "timeout": 10}


def test_fetch_headlines_raises_on_http_errors(monkeypatch):
    monkeypatch.setattr(news.requests, "get", lambda *a, **k: FakeResponse(status=503))
    with pytest.raises(requests.HTTPError):
        news.fetch_headlines()


def test_cache_reuses_results_until_they_expire(monkeypatch):
    fetches = []
    clock = [100.0]
    monkeypatch.setattr(news.time, "monotonic", lambda: clock[0])
    cache = news.NewsCache(ttl=60, fetch=lambda url: fetches.append(url) or ["headline"])

    assert cache.get("feed") == (["headline"], None)
    clock[0] += 30
    assert cache.get("feed") == (["headline"], None)
    assert len(fetches) == 1
    clock[0] += 40
    cache.get("feed")
    assert len(fetches) == 2


def test_cache_falls_back_to_old_news_then_to_a_message(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(news.time, "monotonic", lambda: clock[0])
    results = [["old headline"]]

    def fetch(url):
        if results:
            return results.pop()
        raise requests.ConnectionError("offline")

    cache = news.NewsCache(ttl=10, fetch=fetch)
    assert cache.get("feed") == (["old headline"], None)
    clock[0] += 60
    assert cache.get("feed") == (["old headline"], None)

    empty = news.NewsCache(ttl=10, fetch=fetch)
    headlines, error = empty.get("feed")
    assert headlines == [] and "can't be reached" in error
