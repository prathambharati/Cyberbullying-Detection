"""Loading, cleaning and splitting the training data.

The model learns two things:

* Harm, "should this be blocked?", from text labelled for abuse itself: the
  Jigsaw toxic comments (Wikipedia talk pages), OLID (offensive tweets) and the
  Measuring Hate Speech corpus (comments about identities, hateful and not).
* Kind, "who is it aimed at?", from the cyberbullying tweets dataset and the
  targets annotated in Measuring Hate Speech.

The tweets dataset isn't used for harm. Its classes were collected by keyword,
so a lot of its "bullying" tweets are people talking about bullying, and a
model trained on it learns topics instead of abuse. It's still a good source
for the kind of bullying once something has been flagged.

Measuring Hate Speech fills two gaps Jigsaw leaves: identity attacks with no
swear words in them ("go back to your country"), and thousands of supportive
comments that mention identities, which teach the model that "as a gay man"
isn't an insult.

Each dataset is deduplicated before it's split. Validation drops anything that
is in training, and test drops anything in training or validation for either
head, since both heads share one encoder.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .text import tokenize

KINDS = ("age", "ethnicity", "gender", "religion", "other")
SPLITS = ("train", "val", "test")

TWEET_KINDS = {
    "age": "age",
    "ethnicity": "ethnicity",
    "gender": "gender",
    "religion": "religion",
    "other_cyberbullying": "other",
    "not_cyberbullying": None,
}
JIGSAW_LABELS = ("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")
# Which Measuring Hate Speech targets count as which kind.
HATE_SPEECH_TARGETS = {
    "age": ("target_age",),
    "ethnicity": ("target_race", "target_origin"),
    "gender": ("target_gender", "target_sexuality"),
    "religion": ("target_religion",),
    "other": ("target_disability",),
}
# The corpus authors' cut-offs: above 0.5 is roughly hate speech, below -1 is
# supportive or counter speech, and the band in between is ambiguous.
HATEFUL_ABOVE, SUPPORTIVE_BELOW = 0.5, -1.0


@dataclass
class Examples:
    tokens: list[list[str]] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.labels)

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "Examples":
        return cls(list(df["tokens"]), [int(label) for label in df["label"]], list(df["source"]))


@dataclass
class TaskData:
    train: Examples
    val: Examples
    test: Examples
    stats: dict


def _tokenized(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tokens"] = df["text"].astype(str).map(tokenize)
    df["key"] = df["tokens"].map(" ".join)
    return df[df["key"] != ""].drop_duplicates("key")


def _sample(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    return df.sample(n=min(n, len(df)), random_state=seed) if n else df


def _three_way(df: pd.DataFrame, stratify: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """80/10/10, stratified so every label keeps its share."""
    train, rest = train_test_split(df, test_size=0.2, stratify=df[stratify], random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest[stratify], random_state=seed)
    return train, val, test


def _jigsaw(paths: dict, seed: int, limit: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_train = pd.read_csv(paths["jigsaw-train"])
    raw_test = pd.read_csv(paths["jigsaw-test"]).merge(pd.read_csv(paths["jigsaw-test-labels"]), on="id")
    raw_test = raw_test[(raw_test[list(JIGSAW_LABELS)] >= 0).all(axis=1)]  # -1 means the competition didn't score it

    def frame(df):
        harmful = (df[list(JIGSAW_LABELS)].max(axis=1) > 0).astype(int)
        return _tokenized(pd.DataFrame({"text": df["comment_text"], "label": harmful, "source": "jigsaw"}))

    return frame(_sample(raw_train, limit, seed)), frame(_sample(raw_test, limit and limit // 4, seed))


def _olid(paths: dict, split: str, seed: int, limit: int | None) -> pd.DataFrame:
    texts = Path(paths[f"olid-{split}-text"]).read_text(encoding="utf-8").splitlines()
    labels = [int(x) for x in Path(paths[f"olid-{split}-labels"]).read_text().split()]
    # OLID swaps links for a literal "URL" placeholder. Drop it like a real link.
    df = pd.DataFrame({"text": [t.replace("URL", " ") for t in texts], "label": labels, "source": "olid"})
    return _tokenized(_sample(df, limit and limit // 10, seed))


def _hate_speech(path: Path, seed: int, limit: int | None) -> tuple[pd.DataFrame, dict]:
    """One row per comment, with a harm label and, for hateful comments, a kind."""
    target_columns = [column for columns in HATE_SPEECH_TARGETS.values() for column in columns]
    raw = pd.read_parquet(path, columns=["comment_id", "text", "hate_speech_score", *target_columns])
    # Each row is one annotator's view. The score is per comment; targets are
    # averaged so 0.5 means half the annotators picked that group.
    comments = raw.groupby("comment_id").agg(
        text=("text", "first"), score=("hate_speech_score", "first"), **{c: (c, "mean") for c in target_columns}
    )
    comments = _sample(comments, limit, seed)
    clear = (comments["score"] > HATEFUL_ABOVE) | (comments["score"] < SUPPORTIVE_BELOW)
    stats = {"comments": len(comments), "dropped_ambiguous": int((~clear).sum())}
    comments = comments[clear]

    shares = pd.DataFrame({kind: comments[list(cols)].max(axis=1) for kind, cols in HATE_SPEECH_TARGETS.items()})
    top_share = shares.max(axis=1)
    kind = shares.idxmax(axis=1).map(KINDS.index).where(top_share >= 0.5, -1)
    harmful = (comments["score"] > HATEFUL_ABOVE).astype(int)
    df = pd.DataFrame({
        "text": comments["text"],
        "harm": harmful,
        "kind": kind.where(harmful == 1, -1).astype(int),  # only abuse gets a kind
        "source": "hate_speech",
    })
    df = _tokenized(df)
    stats["hateful"], stats["supportive"] = int(df["harm"].sum()), int((df["harm"] == 0).sum())
    return df, stats


def _tweets(path: Path, seed: int, limit: int | None) -> tuple[pd.DataFrame, dict]:
    raw = _sample(pd.read_csv(path), limit, seed)
    df = raw.rename(columns={"tweet_text": "text", "cyberbullying_type": "label"})[["text", "label"]].dropna()
    df = df[df["label"].isin(TWEET_KINDS)].copy()
    df["tokens"] = df["text"].astype(str).map(tokenize)
    df["key"] = df["tokens"].map(" ".join)
    df = df[df["key"] != ""]

    # The same tweet sometimes appears under two labels. It can't be both, so drop it.
    labels_per_key = df.groupby("key")["label"].nunique()
    conflicting = df["key"].isin(labels_per_key[labels_per_key > 1].index)
    stats = {"rows": len(raw), "dropped_conflicting": int(conflicting.sum())}
    df = df[~conflicting].drop_duplicates("key")

    df = df[df["label"].map(TWEET_KINDS).notna()].copy()
    df["label"] = df["label"].map(lambda label: KINDS.index(TWEET_KINDS[label]))
    df["source"] = "tweets"
    return df, stats


def _drop_seen(df: pd.DataFrame, seen: set) -> tuple[pd.DataFrame, int]:
    overlap = df["key"].isin(seen)
    return df[~overlap], int(overlap.sum())


def load_data(paths: dict, seed: int = 42, limit: int | None = None) -> tuple[TaskData, TaskData]:
    """Returns (harm, kind). `paths` maps asset names from downloads.py to files."""
    harm = {split: [] for split in SPLITS}
    kind = {split: [] for split in SPLITS}

    jigsaw_train, jigsaw_test = _jigsaw(paths, seed, limit)
    train, val = train_test_split(jigsaw_train, test_size=0.1, stratify=jigsaw_train["label"], random_state=seed)
    harm["train"].append(train)
    harm["val"].append(val)
    harm["test"].append(jigsaw_test)

    for split in SPLITS:
        harm[split].append(_olid(paths, split, seed, limit))

    hate, hate_stats = _hate_speech(paths["hate-speech"], seed, limit)
    for split, part in zip(SPLITS, _three_way(hate, "harm", seed)):
        harm[split].append(part.assign(label=part["harm"]))
        kind[split].append(part[part["kind"] >= 0].assign(label=part["kind"]))

    tweets, tweet_stats = _tweets(paths["tweets"], seed, limit)
    for split, part in zip(SPLITS, _three_way(tweets, "label", seed)):
        kind[split].append(part)

    harm = {split: pd.concat(parts).drop_duplicates("key") for split, parts in harm.items()}
    kind = {split: pd.concat(parts).drop_duplicates("key") for split, parts in kind.items()}

    seen = set(harm["train"]["key"]) | set(kind["train"]["key"])
    dropped = {}
    for name, task in (("harm", harm), ("kind", kind)):
        task["val"], dropped[f"{name}_val"] = _drop_seen(task["val"], seen)
    seen |= set(harm["val"]["key"]) | set(kind["val"]["key"])
    for name, task in (("harm", harm), ("kind", kind)):
        task["test"], dropped[f"{name}_test"] = _drop_seen(task["test"], seen)

    def summary(task):
        return {
            "splits": {split: len(task[split]) for split in SPLITS},
            "test_by_source": {src: int(n) for src, n in task["test"]["source"].value_counts().sort_index().items()},
        }

    harm_stats = summary(harm)
    harm_stats["harmful_share"] = {split: round(float(harm[split]["label"].mean()), 4) for split in SPLITS}
    kind_stats = summary(kind)
    shared = {"dropped_as_seen": dropped, "hate_speech": hate_stats, "tweets": tweet_stats}
    return (
        TaskData(*(Examples.from_frame(harm[split]) for split in SPLITS), {**harm_stats, **shared}),
        TaskData(*(Examples.from_frame(kind[split]) for split in SPLITS), {**kind_stats, **shared}),
    )
