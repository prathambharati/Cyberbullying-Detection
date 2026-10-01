"""Turning raw posts into tokens and token ids.

The tokenizer copies the conventions GloVe's vocabulary was built with, so
"don't" becomes "do" + "n't" and "it's" becomes "it" + "'s". Those pieces
all have pretrained vectors, while the joined forms mostly don't.
"""

import html
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

PAD = "<pad>"
UNKNOWN = "<unk>"

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
# Letters or !?. repeated three or more times. Digits are left alone so 1000 stays 1000.
_REPEATED = re.compile(r"([^\W\d_]|[!?.])\1{2,}")
# A word (letters and digits, maybe with inner apostrophes, or asterisks as in
# a censored "f*ck"), or any single character that isn't a word character or space.
_TOKEN = re.compile(r"[^\W_]+(?:['*]+[^\W_]+)*|[^\w\s]")
_CLITICS = ("n't", "'s", "'re", "'ve", "'ll", "'d", "'m")


def normalize(text: str) -> str:
    """Lowercase and strip the parts of a post that carry no meaning."""
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = text.lower()
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = text.replace("#", " ")
    # "soooo" and "sooo" should look the same to the model.
    return _REPEATED.sub(r"\1\1", text)


def _split_clitic(word: str) -> list[str]:
    for clitic in _CLITICS:
        if word.endswith(clitic) and len(word) > len(clitic):
            return [word[: -len(clitic)], clitic]
    return [word]


def tokenize(text: str) -> list[str]:
    tokens = []
    for piece in _TOKEN.findall(normalize(text)):
        if piece[0].isalnum():
            tokens.extend(_split_clitic(piece))
        elif piece in "!?" or unicodedata.category(piece) == "So":
            # Keep ! and ? and emoji. Other punctuation is mostly noise here.
            tokens.append(piece)
    if tokens[:1] == ["rt"]:
        tokens = tokens[1:]
    return tokens


def pad(ids: list[int], length: int) -> list[int]:
    return (ids + [0] * length)[:length]


def windows(ids: list[int], size: int) -> list[list[int]]:
    """Split a long sequence into overlapping windows of `size` ids.

    The model reads a fixed number of tokens. Rather than cutting long posts
    off, the classifier scores every window and keeps the worst one, so an
    insult at the end of a long comment still gets caught.
    """
    if len(ids) <= size:
        return [pad(ids, size)]
    stride = max(1, size // 2)
    starts = list(range(0, len(ids) - size + 1, stride))
    if starts[-1] != len(ids) - size:
        starts.append(len(ids) - size)
    return [ids[start : start + size] for start in starts]


class Vocabulary:
    """Maps tokens to ids. Id 0 is padding and id 1 is any unknown token."""

    def __init__(self, tokens: list[str]):
        if tokens[:2] != [PAD, UNKNOWN]:
            raise ValueError("A vocabulary has to start with the padding and unknown tokens.")
        self.tokens = tokens
        self.index = {token: i for i, token in enumerate(tokens)}

    @classmethod
    def build(cls, token_lists, max_size: int = 20_000, min_count: int = 2) -> "Vocabulary":
        counts = Counter(token for tokens in token_lists for token in tokens)
        frequent = [token for token, n in counts.most_common() if n >= min_count]
        return cls([PAD, UNKNOWN] + frequent[: max_size - 2])

    def __len__(self) -> int:
        return len(self.tokens)

    def __contains__(self, token: str) -> bool:
        return token in self.index

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.index.get(token, 1) for token in tokens]

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.tokens, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))
