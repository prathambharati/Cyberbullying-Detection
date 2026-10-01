"""Checks on the trained model in models/, which skip if it hasn't been trained,
plus a few rules of the Classifier itself, tested with a stand-in network."""

import json

import numpy as np
import pytest

from cyberbullying.classifier import Classifier
from cyberbullying.data import KINDS
from cyberbullying.paths import MODELS_DIR
from cyberbullying.text import PAD, UNKNOWN, Vocabulary
from cyberbullying.train import SPOT_CHECKS


class FixedNetwork:
    """Returns the same scores for every input, like a model that has made up its mind."""

    def __init__(self, harm, kind):
        self.harm, self.kind = harm, np.array(kind, dtype="float32")

    def predict_on_batch(self, x):
        return {"harm": np.full((len(x), 1), self.harm, dtype="float32"), "kind": np.tile(self.kind, (len(x), 1))}


def fixed(harm, kind):
    return Classifier(FixedNetwork(harm, kind), Vocabulary([PAD, UNKNOWN, "hello"]), KINDS, max_len=8, threshold=0.5)


def test_a_target_is_only_named_when_the_model_is_sure():
    assert fixed(0.9, [0.02, 0.9, 0.04, 0.02, 0.02]).predict_one("hello").category == "ethnicity"
    assert fixed(0.9, [0.02, 0.7, 0.24, 0.02, 0.02]).predict_one("hello").category == "other"
    assert fixed(0.1, [0.02, 0.9, 0.04, 0.02, 0.02]).predict_one("hello").category is None


def test_the_threshold_decides_what_is_blocked():
    assert fixed(0.5, [0.2] * 5).predict_one("hello").is_bullying
    assert not fixed(0.49, [0.2] * 5).predict_one("hello").is_bullying

CLEARLY_ABUSIVE = [
    "shut up you dumb idiot",
    "You are such a stupid loser, nobody likes you!",
    "Kill yourself, the world would be better without you",
]
CLEARLY_FINE = [
    "Happy birthday! Hope you have an amazing day.",
    "Thanks for helping me move this weekend, you're the best",
    "Can someone explain how photosynthesis works?",
]


def test_model_card_matches_the_saved_model(classifier):
    card = json.loads((MODELS_DIR / "model_card.json").read_text(encoding="utf-8"))
    assert card["kinds"] == list(KINDS) == classifier.kinds
    assert card["architecture"]["vocab_size"] == len(classifier.vocab)
    assert 0 < classifier.threshold < 1


def test_clear_cases(classifier):
    assert all(p.is_bullying for p in classifier.predict(CLEARLY_ABUSIVE))
    assert not any(p.is_bullying for p in classifier.predict(CLEARLY_FINE))


def test_spot_checks_mostly_pass(classifier):
    predictions = classifier.predict([text for text, _ in SPOT_CHECKS])
    passed = sum(p.is_bullying == expected for p, (_, expected) in zip(predictions, SPOT_CHECKS))
    assert passed / len(SPOT_CHECKS) >= 0.85, f"only {passed} of {len(SPOT_CHECKS)} spot checks passed"


def test_identity_attacks_get_the_right_category(classifier):
    religion = classifier.predict_one("All muslims are terrorists and should be banned")
    gender = classifier.predict_one("Girls like you should stay in the kitchen and shut up")
    assert religion.category == "religion"
    assert gender.category == "gender"


def test_scores_are_probabilities_and_categories_add_up(classifier):
    for p in classifier.predict(CLEARLY_ABUSIVE + CLEARLY_FINE):
        assert 0.0 <= p.score <= 1.0
        assert set(p.categories) == set(KINDS)
        assert sum(p.categories.values()) == pytest.approx(1.0, abs=1e-3)
        assert (p.category is not None) == p.is_bullying


def test_batching_does_not_change_scores(classifier):
    texts = CLEARLY_ABUSIVE + CLEARLY_FINE
    together = [p.score for p in classifier.predict(texts)]
    alone = [classifier.predict_one(text).score for text in texts]
    assert np.allclose(together, alone, atol=1e-4)


def test_an_insult_at_the_end_of_a_long_post_is_still_caught(classifier):
    filler = "I had a pretty normal day at work and then went for a long walk in the park. " * 30
    ending = classifier.predict_one(filler + "Anyway, you are a worthless idiot and everyone hates you.")
    assert len(filler.split()) > 3 * classifier.max_len
    assert ending.is_bullying
    assert not classifier.predict_one(filler).is_bullying


def test_text_without_words_scores_zero(classifier):
    assert classifier.predict([]) == []
    for text in ("", "   ", "https://example.com @someone", "..."):
        prediction = classifier.predict_one(text)
        assert prediction.score == 0.0 and not prediction.is_bullying and prediction.category is None
    # Mixed batches keep their order.
    scores = [p.score for p in classifier.predict(["", "shut up you dumb idiot", "@someone"])]
    assert scores[0] == 0.0 and scores[1] > 0.9 and scores[2] == 0.0


def test_emoji_only_text_is_not_flagged(classifier):
    assert not classifier.predict_one("\U0001F602\U0001F602").is_bullying
