"""Runs the whole training script on a tiny made-up dataset, so the pipeline is
tested end to end without the real data or a long wait."""

import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from cyberbullying import downloads, train
from cyberbullying.classifier import Classifier
from cyberbullying.data import KINDS

NICE = ["have a lovely day", "thanks for the help", "great game last night", "see you at the library",
        "the weather is nice today"]
MEAN = ["you are an idiot", "shut up loser", "nobody likes you idiot", "you stupid fool", "go away you loser"]
TWEETS = {"age": "school kids", "ethnicity": "race colour", "gender": "women girls", "religion": "church faith",
          "other_cyberbullying": "random troll", "not_cyberbullying": "sunny weekend"}
JIGSAW_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def write_tiny_dataset(folder: Path) -> None:
    path = {name: folder / downloads.ASSETS[name].path.name for name in downloads.GROUPS["training"]}
    rng = random.Random(0)

    def comment(i, toxic):
        return f"{rng.choice(MEAN if toxic else NICE)} number {i}"

    with open(path["jigsaw-train"], "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "comment_text", *JIGSAW_LABELS])
        for i in range(200):
            toxic = i % 4 == 0
            writer.writerow([f"a{i}", comment(i, toxic), int(toxic), 0, 0, 0, int(toxic), 0])

    with open(path["jigsaw-test"], "w", newline="", encoding="utf-8") as texts, \
            open(path["jigsaw-test-labels"], "w", newline="", encoding="utf-8") as labels:
        text_writer, label_writer = csv.writer(texts), csv.writer(labels)
        text_writer.writerow(["id", "comment_text"])
        label_writer.writerow(["id", *JIGSAW_LABELS])
        for i in range(1000, 1060):
            toxic = i % 4 == 0
            text_writer.writerow([f"b{i}", comment(i, toxic)])
            # The competition marks rows it didn't score with -1. Those have to be skipped.
            values = [-1] * 6 if i >= 1050 else [int(toxic), 0, 0, 0, int(toxic), 0]
            label_writer.writerow([f"b{i}", *values])

    for split, start in (("train", 2000), ("val", 3000), ("test", 4000)):
        rows = [(f"@USER {comment(i, i % 3 == 0)} URL", int(i % 3 == 0)) for i in range(start, start + 20)]
        path[f"olid-{split}-text"].write_text("\n".join(text for text, _ in rows) + "\n", encoding="utf-8")
        path[f"olid-{split}-labels"].write_text("\n".join(str(label) for _, label in rows) + "\n")

    with open(path["tweets"], "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tweet_text", "cyberbullying_type"])
        for label, words in TWEETS.items():
            for i in range(30):
                writer.writerow([f"{words} tweet {i}", label])
        writer.writerow(["the same tweet with two labels", "age"])
        writer.writerow(["the same tweet with two labels", "gender"])

    # Measuring Hate Speech: two annotators per comment. 24 hateful, 24
    # supportive and 12 ambiguous comments, which the loader should drop.
    targets = ["target_race", "target_origin", "target_religion", "target_gender",
               "target_sexuality", "target_age", "target_disability"]
    rows = []
    for i in range(60):
        score = 1.5 if i < 24 else (-2.0 if i < 48 else 0.0)
        aimed_at = targets[i % len(targets)]
        text = f"{'you people are vile' if score > 0 else 'proud of my community'} comment {i}"
        for annotator in (1, 2):
            rows.append({"comment_id": i, "annotator_id": annotator, "text": text, "hate_speech_score": score,
                         **{t: t == aimed_at for t in targets}})
    pd.DataFrame(rows).to_parquet(path["hate-speech"])


def test_training_end_to_end(tmp_path):
    data, output = tmp_path / "data", tmp_path / "model"
    data.mkdir()
    write_tiny_dataset(data)

    exit_code = train.main([
        "--data-dir", str(data), "--glove", "none", "--output", str(output), "--epochs", "1",
        "--max-len", "16", "--vocab-size", "500", "--embedding-dim", "16", "--lstm-units", "8",
    ])
    assert exit_code == 0

    card = json.loads((output / "model_card.json").read_text(encoding="utf-8"))
    assert card["kinds"] == list(KINDS)
    assert 0 < card["threshold"] < 1
    cleaning = card["data"]["cleaning"]
    assert cleaning["tweets"]["dropped_conflicting"] == 2
    assert cleaning["hate_speech"] == {"comments": 60, "dropped_ambiguous": 12, "hateful": 24, "supportive": 24}
    assert card["test"]["harm"]["jigsaw"]["examples"] == 50  # the ten -1 rows are gone
    assert card["test"]["harm"]["olid"]["examples"] == 20
    assert card["test"]["harm"]["hate_speech"]["examples"] == 5
    assert card["test"]["kind"]["by_source"]["tweets"]["examples"] == 15
    assert card["baseline_test"]["harm"]["all"]["roc_auc"] is not None
    assert card["spot_checks"]["total"] == len(train.SPOT_CHECKS)

    classifier = Classifier.load(output)
    predictions = classifier.predict(["you are an idiot", "have a lovely day"])
    assert len(predictions) == 2
    assert all(0.0 <= p.score <= 1.0 for p in predictions)
    assert all(np.isclose(sum(p.categories.values()), 1.0, atol=1e-3) for p in predictions)


def test_glove_loader_reads_both_file_layouts(tmp_path):
    from cyberbullying.text import PAD, UNKNOWN, Vocabulary

    vocab = Vocabulary([PAD, UNKNOWN, "cat", "dog", "zebra"])
    plain = tmp_path / "glove.txt"
    plain.write_text("cat 1 2 3\ndog 4 5 6\nbird 7 8 9\n", encoding="utf-8")
    matrix, found = train.load_glove(plain, vocab, dim=3, seed=0)
    assert found == 2
    assert matrix[0].tolist() == [0, 0, 0]  # padding
    assert matrix[2].tolist() == [1, 2, 3] and matrix[3].tolist() == [4, 5, 6]
    assert np.any(matrix[4] != 0)  # "zebra" isn't in GloVe, so it starts random

    with_header = tmp_path / "gensim.txt"
    with_header.write_text("3 3\ncat 1 2 3\n", encoding="utf-8")
    assert train.load_glove(with_header, vocab, dim=3, seed=0)[1] == 1


def test_threshold_picks_the_best_f1():
    scores = np.array([0.1, 0.2, 0.35, 0.4, 0.8, 0.9])
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert train.pick_threshold(scores, labels) == 0.4
