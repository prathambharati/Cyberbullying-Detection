"""Headlines for the News page, from the Times of India top stories feed."""

import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser

import feedparser
import requests

# The "Top Headlines" feed the first version used stopped updating in 2022. This one is live.
DEFAULT_FEED = "https://timesofindia.indiatimes.com/rssfeedstopstories.cms"
USER_AGENT = "Mozilla/5.0 (compatible; CyberbullyingDetection/2.0)"
_TOI_PHOTO = re.compile(r"https://static\.toiimg\.com/photo/msid-(\d+)")


@dataclass(frozen=True)
class Headline:
    title: str
    link: str
    summary: str
    published: datetime | None
    image: str | None = None


class _Reader(HTMLParser):
    """Collects the text of a snippet of HTML, plus its first web image."""

    def __init__(self):
        super().__init__()
        self.parts = []
        self.image = None

    def handle_data(self, data):
        self.parts.append(data)

    def handle_starttag(self, tag, attrs):
        if tag == "img" and self.image is None:
            src = dict(attrs).get("src") or ""
            if src.startswith(("https://", "http://")):
                self.image = src


def _read(markup: str) -> tuple[str, str | None]:
    parser = _Reader()
    parser.feed(markup or "")
    parser.close()
    return " ".join(" ".join(parser.parts).split()), parser.image


def strip_html(markup: str) -> str:
    """Feed summaries are HTML with images in them. Keep just the words."""
    return _read(markup)[0]


def _enclosure_image(entry) -> str | None:
    for link in entry.get("links", []):
        href = link.get("href", "")
        if link.get("rel") == "enclosure" and str(link.get("type", "")).startswith("image/") \
                and href.startswith(("https://", "http://")):
            return href
    return None


def thumbnail(url: str) -> str:
    """TOI links full-size photos, often well over a megabyte each. Ask its image
    server for a 320 pixel wide copy instead, about a fifteenth of the size."""
    match = _TOI_PHOTO.match(url)
    if not match:
        return url
    return f"https://static.toiimg.com/thumb/msid-{match[1]},width-320,resizemode-4/{match[1]}.jpg"


def parse_feed(content: bytes, limit: int = 20) -> list[Headline]:
    feed = feedparser.parse(content)
    headlines = []
    for entry in feed.entries:
        title = strip_html(entry.get("title", ""))
        link = entry.get("link", "")
        # Only real web links, so a bad feed can't sneak a javascript: link onto the page.
        if not title or not link.startswith(("https://", "http://")):
            continue
        published = None
        if entry.get("published_parsed"):
            published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        summary, inline_image = _read(entry.get("summary", ""))
        image = _enclosure_image(entry) or inline_image
        headlines.append(Headline(title, link, summary, published, thumbnail(image) if image else None))
        if len(headlines) == limit:
            break
    return headlines


def fetch_headlines(url: str = DEFAULT_FEED, limit: int = 20, timeout: float = 10) -> list[Headline]:
    response = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    return parse_feed(response.content, limit)


class NewsCache:
    """Remembers the last good result for a while, so page views don't hammer the feed."""

    def __init__(self, ttl: float = 900, fetch=fetch_headlines):
        self.ttl = ttl
        self.fetch = fetch
        self._lock = threading.Lock()
        self._results = {}  # url -> (fetched at, headlines)

    def get(self, url: str = DEFAULT_FEED) -> tuple[list[Headline], str | None]:
        """Returns (headlines, error message). The message is None when all is well."""
        with self._lock:
            cached = self._results.get(url)
        if cached and time.monotonic() - cached[0] < self.ttl:
            return cached[1], None
        try:
            headlines = self.fetch(url)
        except requests.RequestException:
            if cached:
                return cached[1], None  # old news beats no news
            return [], "The news feed can't be reached right now. Try again in a little while."
        with self._lock:
            self._results[url] = (time.monotonic(), headlines)
        return headlines, None
