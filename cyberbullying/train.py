"""Train the classifier and save it to models/.

    python -m cyberbullying.train

The first run downloads about 300 MB into data/: the Jigsaw toxic comments,
the OLID offensive tweets, the Measuring Hate Speech corpus, the cyberbullying
tweets and GloVe vectors.

The script trains the two-headed BiLSTM, tunes the blocking threshold on
validation data, scores both heads on held-out test data, trains TF-IDF +
logistic regression baselines on the same splits, and runs a spot check on
hand-written messages. Test scores alone can look great while a model fails
on ordinary sentences, so the spot check is there to catch that.
"""

import argparse
import gzip
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__, downloads
from .data import KINDS, Examples, load_data
from .model import build_model
from .paths import MODELS_DIR
from .text import Vocabulary, pad

# (message, should it be blocked?) Written by hand, never used for training or
# tuning. Half of the harmless ones mention an identity on purpose, to catch a
# model that flags words like "gay" or "Muslim" instead of actual abuse.
SPOT_CHECKS = [
    ("You are such a stupid loser, nobody likes you!", True),
    ("Go back to your country, nobody wants your kind here", True),
    ("Girls like you should stay in the kitchen and shut up", True),
    ("You're too old to be on here, go die already grandpa", True),
    ("Kill yourself, the world would be better without you", True),
    ("Nobody at school wants to sit with you, fat freak", True),
    ("All muslims are terrorists and should be banned", True),
    ("shut up you dumb idiot", True),
    ("Immigrants are ruining this country, send them all back", True),
    ("Women are too stupid to vote", True),
    ("Black people are all criminals", True),
    ("Jews control the media and the banks", True),
    ("Gay people are disgusting and should be locked up", True),
    ("Old people like you should just die already", True),
    ("I'm a proud Muslim woman and I love my community", False),
    ("My grandparents moved here from Nigeria in the seventies", False),
    ("As a gay man, this show made me feel seen", False),
    ("Happy Diwali to all my Hindu friends!", False),
    ("Black history month starts on Monday, here's a reading list", False),
    ("My Jewish neighbours invited us over for Shabbat dinner", False),
    ("Women's football is getting so good to watch", False),
    ("My 80 year old grandpa just ran his first 5k", False),
    ("Finally finished my first 10k run this morning. Legs are jelly but I'm so happy!", False),
    ("Does anyone have a good recipe for dal makhani? Mine always turns out too thin.", False),
    ("Our robotics team made it to the state finals. So proud of everyone who stayed late all month.", False),
    ("Reminder that the library is open until midnight during exam week. Good luck, everyone.", False),
    ("Happy birthday! Hope you have an amazing day.", False),
    ("Great game last night, that final goal was unreal", False),
    ("I disagree with you on this, but I see your point.", False),
    ("Can someone explain how photosynthesis works?", False),
    ("The new update broke my phone's battery life, so annoying", False),
    ("Thanks for helping me move this weekend, you're the best", False),
    ("School starts next week and I haven't bought anything yet", False),
    ("My grandma just learned how to video call us, she is adorable", False),
]


CLEANING_STATS = ("dropped_as_seen", "hate_speech", "tweets")


def load_glove(path: Path, vocab: Vocabulary, dim: int, seed: int) -> tuple[np.ndarray, int]:
    """Build an embedding matrix for `vocab` out of a GloVe file.

    Words GloVe has never seen (slang, typos, emoji) start as small random
    vectors and get learned during training.
    """
    matrix = np.zeros((len(vocab), dim), dtype="float32")
    found = np.zeros(len(vocab), dtype=bool)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            word, _, rest = line.rstrip().partition(" ")
            i = vocab.index.get(word)
            if i is None or i < 2:
                continue
            values = rest.split(" ")
            if len(values) == dim:  # skips the "400000 100" header in the gensim copy
                matrix[i] = np.asarray(values, dtype="float32")
                found[i] = True

    missing = ~found
    missing[0] = False  # padding stays all zeros
    scale = float(matrix[found].std()) if found.any() else 0.1
    rng = np.random.default_rng(seed)
    matrix[missing] = rng.normal(0.0, scale, size=(int(missing.sum()), dim))
    return matrix, int(found.sum())


def encode(token_lists, vocab: Vocabulary, max_len: int) -> np.ndarray:
    return np.array([pad(vocab.encode(tokens), max_len) for tokens in token_lists], dtype="int32")


def stack(harm: Examples, kind: Examples, vocab: Vocabulary, max_len: int):
    """One batch of inputs for both heads. Each example only teaches the head
    it has a label for; the other head gets a sample weight of zero."""
    n_harm, n_kind = len(harm), len(kind)
    x = encode(harm.tokens + kind.tokens, vocab, max_len)
    y = {
        "harm": np.array(harm.labels + [0] * n_kind, dtype="float32").reshape(-1, 1),
        "kind": np.array([0] * n_harm + kind.labels, dtype="int32"),
    }
    weights = {
        "harm": np.array([1.0] * n_harm + [0.0] * n_kind, dtype="float32"),
        "kind": np.array([0.0] * n_harm + [1.0] * n_kind, dtype="float32"),
    }
    return x, y, weights


def pick_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """The cut-off with the best F1 for "harmful" on validation data."""
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    best = int(np.argmax(f1[:-1]))  # the curve's last point has no threshold
    return round(float(thresholds[best]), 2)


def _r(value) -> float:
    return round(float(value), 4)


def harm_metrics(scores: np.ndarray, examples: Examples, threshold: float) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = np.array(examples.labels)
    sources = np.array(examples.sources)

    def summary(mask):
        y, s = labels[mask], scores[mask]
        flagged = s >= threshold
        tp, fp = int(np.sum(flagged & (y == 1))), int(np.sum(flagged & (y == 0)))
        fn, tn = int(np.sum(~flagged & (y == 1))), int(np.sum(~flagged & (y == 0)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        both = 0 < y.sum() < len(y)
        return {
            "examples": int(mask.sum()),
            "roc_auc": _r(roc_auc_score(y, s)) if both else None,
            "average_precision": _r(average_precision_score(y, s)) if both else None,
            "precision": _r(precision),
            "recall": _r(recall),
            "f1": _r(2 * precision * recall / (precision + recall)) if tp else 0.0,
            "false_positive_rate": _r(fp / (fp + tn)) if fp + tn else 0.0,
        }

    results = {"threshold": threshold, "all": summary(np.ones(len(labels), dtype=bool))}
    for source in sorted(set(sources)):
        results[source] = summary(sources == source)
    return results


def kind_metrics(probs: np.ndarray, examples: Examples) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

    labels = np.array(examples.labels)
    sources = np.array(examples.sources)
    predicted = probs.argmax(axis=1)
    ids = list(range(len(KINDS)))
    precision, recall, f1, support = precision_recall_fscore_support(labels, predicted, labels=ids, zero_division=0)

    def summary(mask):
        return {
            "examples": int(mask.sum()),
            "accuracy": _r(accuracy_score(labels[mask], predicted[mask])),
            "macro_f1": _r(f1_score(labels[mask], predicted[mask], labels=ids, average="macro", zero_division=0)),
        }

    results = summary(np.ones(len(labels), dtype=bool))
    results["per_kind"] = {
        kind: {"precision": _r(p), "recall": _r(rc), "f1": _r(f), "support": int(n)}
        for kind, p, rc, f, n in zip(KINDS, precision, recall, f1, support)
    }
    results["by_source"] = {source: summary(sources == source) for source in sorted(set(sources))}
    results["confusion_matrix"] = confusion_matrix(labels, predicted, labels=ids).tolist()
    return results


def run_baselines(harm, kind) -> dict:
    """TF-IDF over words and word pairs, then logistic regression, for each head."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score

    def vectorize(task):
        vectorizer = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False,
                                     ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
        joined = [[" ".join(t) for t in split.tokens] for split in (task.train, task.val, task.test)]
        return vectorizer.fit_transform(joined[0]), vectorizer.transform(joined[1]), vectorizer.transform(joined[2])

    x_train, x_val, x_test = vectorize(harm)
    best = None
    for c in (1.0, 4.0, 16.0):
        model = LogisticRegression(C=c, max_iter=3000).fit(x_train, harm.train.labels)
        auc = roc_auc_score(harm.val.labels, model.predict_proba(x_val)[:, 1])
        if best is None or auc > best[0]:
            best = (auc, model)
    harm_model = best[1]
    threshold = pick_threshold(harm_model.predict_proba(x_val)[:, 1], np.array(harm.val.labels))
    harm_results = harm_metrics(harm_model.predict_proba(x_test)[:, 1], harm.test, threshold)

    x_train, x_val, x_test = vectorize(kind)
    best = None
    for c in (0.5, 1.0, 2.0, 4.0, 8.0):
        model = LogisticRegression(C=c, max_iter=3000).fit(x_train, kind.train.labels)
        score = f1_score(kind.val.labels, model.predict(x_val), average="macro")
        if best is None or score > best[0]:
            best = (score, model)
    kind_results = kind_metrics(best[1].predict_proba(x_test), kind.test)
    return {"harm": harm_results, "kind": kind_results}


def spot_check(classifier) -> dict:
    predictions = classifier.predict([text for text, _ in SPOT_CHECKS])
    results = [
        {"text": text, "should_block": expected, "score": p.score, "blocked": p.is_bullying, "category": p.category}
        for (text, expected), p in zip(SPOT_CHECKS, predictions)
    ]
    passed = sum(r["blocked"] == r["should_block"] for r in results)
    return {"passed": passed, "total": len(results), "results": results}


def print_report(card: dict) -> None:
    def fmt(value):
        return "n/a" if value is None else f"{value:.3f}"

    test, base = card["test"], card["baseline_test"]
    print(f"\nHarm head, threshold {test['harm']['threshold']:.2f}")
    print(f"  {'':24}{'BiLSTM':>10}{'TF-IDF + LR':>14}")
    for source in [s for s in test["harm"] if s != "threshold"]:
        for key in ("roc_auc", "precision", "recall", "f1", "false_positive_rate"):
            line = f"  {source + ' ' + key:24}{fmt(test['harm'][source][key]):>10}"
            if base:
                line += f"{fmt(base['harm'][source][key]):>14}"
            print(line)
    print("\nKind head (abusive texts with a known target)")
    groups = [("all", test["kind"], base and base["kind"])]
    groups += [(source, m, base and base["kind"]["by_source"][source]) for source, m in test["kind"]["by_source"].items()]
    for name, metrics, baseline in groups:
        for key in ("accuracy", "macro_f1"):
            line = f"  {name + ' ' + key:24}{fmt(metrics[key]):>10}"
            if baseline:
                line += f"{fmt(baseline[key]):>14}"
            print(line)
    spots = card["spot_checks"]
    print(f"\nSpot check: {spots['passed']} of {spots['total']} hand-written messages handled right")
    for r in spots["results"]:
        mark = "ok  " if r["blocked"] == r["should_block"] else "MISS"
        print(f"  {mark} {r['score']:.2f} {'block' if r['should_block'] else 'allow'}  {r['text'][:64]}")


def data_paths(data_dir: Path | None) -> dict:
    names = [name for name in downloads.GROUPS["training"] if name != "glove"]
    if data_dir is None:
        return {name: downloads.fetch(name) for name in names}
    return {name: Path(data_dir) / downloads.ASSETS[name].path.name for name in names}


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Train the cyberbullying classifier.")
    parser.add_argument("--data-dir", type=Path, help="use local copies of the data files instead of downloading")
    parser.add_argument("--glove", default="", help="path to a GloVe file, or 'none' to learn embeddings from scratch")
    parser.add_argument("--output", type=Path, default=MODELS_DIR)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=30_000)
    parser.add_argument("--embedding-dim", type=int, default=100)
    parser.add_argument("--lstm-units", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="use a small sample of each dataset (for quick checks)")
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args(argv)
    if not args.glove and args.embedding_dim != 100:
        parser.error("the default GloVe file has 100 dimensions, so --embedding-dim must be 100")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)

    import keras
    import tensorflow as tf

    from .classifier import Classifier

    keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()

    paths = data_paths(args.data_dir)
    print("Loading data")
    harm, kind = load_data(paths, seed=args.seed, limit=args.limit)
    print(f"  harm: {harm.stats['splits']}, harmful share {harm.stats['harmful_share']}")
    print(f"  kind: {kind.stats['splits']}")
    print(f"  dropped {harm.stats['tweets']['dropped_conflicting']:,} tweets with conflicting labels and "
          f"{harm.stats['hate_speech']['dropped_ambiguous']:,} ambiguous hate speech comments")

    vocab = Vocabulary.build(harm.train.tokens + kind.train.tokens, max_size=args.vocab_size)
    all_train = harm.train.tokens + kind.train.tokens
    fits = float(np.mean([len(tokens) <= args.max_len for tokens in all_train]))
    print(f"  vocabulary {len(vocab):,} words, {fits:.1%} of training texts fit in {args.max_len} tokens")

    model = build_model(len(vocab), args.max_len, len(KINDS), args.embedding_dim, args.lstm_units)
    glove_words = None
    if args.glove.lower() != "none":
        glove_path = Path(args.glove) if args.glove else downloads.fetch("glove")
        matrix, glove_words = load_glove(glove_path, vocab, args.embedding_dim, args.seed)
        model.get_layer("embedding").set_weights([matrix])
        print(f"  GloVe has vectors for {glove_words:,} of those words")

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={"harm": "binary_crossentropy", "kind": "sparse_categorical_crossentropy"},
        # Harm examples outnumber kind examples about 5 to 1, so give kind a little more say.
        loss_weights={"harm": 1.0, "kind": 2.0},
    )
    x_train, y_train, w_train = stack(harm.train, kind.train, vocab, args.max_len)
    x_val, y_val, w_val = stack(harm.val, kind.val, vocab, args.max_len)
    started = time.time()
    history = model.fit(
        x_train, y_train, sample_weight=w_train,
        validation_data=(x_val, y_val, w_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
        verbose=2,
    )
    minutes = (time.time() - started) / 60

    def predict(token_lists):
        return model.predict(encode(token_lists, vocab, args.max_len), batch_size=1024, verbose=0)

    threshold = pick_threshold(predict(harm.val.tokens)["harm"].ravel(), np.array(harm.val.labels))
    classifier = Classifier(model, vocab, KINDS, args.max_len, threshold)
    val_losses = history.history["val_loss"]
    card = {
        "name": "cyberbullying-bilstm",
        "version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kinds": list(KINDS),
        "threshold": threshold,
        "architecture": {
            "vocab_size": len(vocab),
            "max_len": args.max_len,
            "embedding_dim": args.embedding_dim,
            "lstm_units": args.lstm_units,
        },
        "training": {
            "epochs_run": len(val_losses),
            "best_epoch": int(np.argmin(val_losses)) + 1,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "minutes": round(minutes, 1),
            "glove_words": glove_words,
            "texts_within_max_len": round(fits, 4),
        },
        "data": {
            "harm": {k: v for k, v in harm.stats.items() if k not in CLEANING_STATS},
            "kind": {k: v for k, v in kind.stats.items() if k not in CLEANING_STATS},
            "cleaning": {k: harm.stats[k] for k in CLEANING_STATS},
            "limit": args.limit,
        },
        "test": {
            "harm": harm_metrics(predict(harm.test.tokens)["harm"].ravel(), harm.test, threshold),
            "kind": kind_metrics(predict(kind.test.tokens)["kind"], kind.test),
        },
        "baseline_test": None if args.skip_baselines else run_baselines(harm, kind),
        "spot_checks": spot_check(classifier),
        "versions": {"python": sys.version.split()[0], "tensorflow": tf.__version__, "keras": keras.__version__},
    }
    print_report(card)

    args.output.mkdir(parents=True, exist_ok=True)
    # Saving the trained model directly would also store the optimizer's state,
    # which triples the file size and isn't needed to make predictions.
    slim = build_model(len(vocab), args.max_len, len(KINDS), args.embedding_dim, args.lstm_units)
    slim.set_weights(model.get_weights())
    slim.save_weights(args.output / "classifier.weights.h5")
    vocab.save(args.output / "vocab.json")
    (args.output / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(f"\nSaved to {args.output} (threshold {threshold:.2f}, {minutes:.1f} minutes of training)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
