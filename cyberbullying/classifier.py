"""Loads the trained model and scores text."""

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .paths import MODELS_DIR
from .text import Vocabulary, tokenize, windows

# Only name a target (age, gender...) when the kind head is at least this sure,
# otherwise call it "other". Generic insults have no target, and without this
# the head would still pick one. On the validation set this cut-off names the
# target for 89% of targeted abuse, and 96% of those names are right.
CATEGORY_CONFIDENCE = 0.8


@dataclass(frozen=True)
class Prediction:
    score: float  # chance the text is abusive: insults, threats, hate and so on
    is_bullying: bool  # score is at or above the blocking threshold
    category: str | None  # what the abuse is about, only set when it's flagged
    categories: dict[str, float]  # the kind head's full guess, for display


class Classifier:
    def __init__(self, model, vocab: Vocabulary, kinds, max_len: int, threshold: float):
        self.model = model
        self.vocab = vocab
        self.kinds = list(kinds)
        self.max_len = max_len
        self.threshold = threshold
        self._lock = threading.Lock()

    @classmethod
    def load(cls, model_dir: Path = MODELS_DIR, threshold: float | None = None) -> "Classifier":
        from .model import build_model

        model_dir = Path(model_dir)
        card_path = model_dir / "model_card.json"
        if not card_path.exists():
            raise FileNotFoundError(f"No trained model in {model_dir}. Train one with: python -m cyberbullying.train")
        card = json.loads(card_path.read_text(encoding="utf-8"))
        arch = card["architecture"]
        vocab = Vocabulary.load(model_dir / "vocab.json")
        model = build_model(len(vocab), arch["max_len"], len(card["kinds"]), arch["embedding_dim"], arch["lstm_units"])
        model.load_weights(model_dir / "classifier.weights.h5")
        return cls(model, vocab, card["kinds"], arch["max_len"], card["threshold"] if threshold is None else threshold)

    def predict(self, texts) -> list[Prediction]:
        texts = list(texts)
        rows, owners = [], []
        for i, text in enumerate(texts):
            ids = self.vocab.encode(tokenize(text))
            # No words at all (empty, or only links and @mentions) means nothing to judge.
            # The model would otherwise score pure padding, which means nothing.
            if not ids:
                continue
            for window in windows(ids, self.max_len):
                rows.append(window)
                owners.append(i)

        worst = {}
        if rows:
            harm, kind = self._run(np.array(rows, dtype="int32"))
            # A long post is split into windows. Judge it by its worst one.
            for row, owner in enumerate(owners):
                if owner not in worst or harm[row] > harm[worst[owner]]:
                    worst[owner] = row
        return [self._prediction(harm[worst[i]], kind[worst[i]]) if i in worst else self._nothing()
                for i in range(len(texts))]

    def predict_one(self, text: str) -> Prediction:
        return self.predict([text])[0]

    def _run(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        harm, kind = [], []
        with self._lock:
            for start in range(0, len(x), 256):
                out = self.model.predict_on_batch(x[start : start + 256])
                harm.append(np.asarray(out["harm"]).ravel())
                kind.append(np.asarray(out["kind"]))
        return np.concatenate(harm), np.concatenate(kind)

    def _nothing(self) -> Prediction:
        even = round(1 / len(self.kinds), 4)
        return Prediction(score=0.0, is_bullying=False, category=None, categories={name: even for name in self.kinds})

    def _prediction(self, harm: float, kind: np.ndarray) -> Prediction:
        score = float(harm)
        is_bullying = score >= self.threshold
        category = None
        if is_bullying:
            best = int(np.argmax(kind))
            category = self.kinds[best] if kind[best] >= CATEGORY_CONFIDENCE else "other"
        return Prediction(
            score=round(score, 4),
            is_bullying=is_bullying,
            category=category,
            categories={name: round(float(p), 4) for name, p in zip(self.kinds, kind)},
        )
