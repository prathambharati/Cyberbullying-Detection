import pytest

from cyberbullying.text import PAD, UNKNOWN, Vocabulary, normalize, pad, tokenize, windows


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Don't do that", ["do", "n't", "do", "that"]),
        ("I can't, you won't", ["i", "ca", "n't", "you", "wo", "n't"]),
        ("it's fine, you're great", ["it", "'s", "fine", "you", "'re", "great"]),
        ("Curly ’ apostrophes: it’s ok", ["curly", "apostrophes", "it", "'s", "ok"]),
    ],
)
def test_contractions_split_like_glove(text, expected):
    assert tokenize(text) == expected


def test_links_mentions_and_hashtags_are_cleaned():
    text = "RT @someone: check this https://t.co/abc123 #MondayMotivation www.example.com"
    assert tokenize(text) == ["check", "this", "mondaymotivation"]


def test_html_entities_and_long_repeats():
    assert tokenize("you &amp; me sooooo tired!!!!") == ["you", "me", "soo", "tired", "!", "!"]


def test_censored_words_stay_whole():
    assert tokenize("shut the f*ck up, you b**ch") == ["shut", "the", "f*ck", "up", "you", "b**ch"]
    assert tokenize("*hugs* for you") == ["hugs", "for", "you"]


def test_numbers_are_not_shortened():
    assert tokenize("1000 followers in 2020") == ["1000", "followers", "in", "2020"]


def test_keeps_emoji_and_question_marks_but_drops_other_punctuation():
    assert tokenize("really?! wow... \U0001F602 (ok); fine.") == ["really", "?", "!", "wow", "\U0001F602", "ok", "fine"]


def test_only_a_leading_rt_is_dropped():
    assert tokenize("rt this is art") == ["this", "is", "art"]
    assert tokenize("please rt") == ["please", "rt"]


def test_empty_and_symbol_only_text():
    assert tokenize("") == []
    assert tokenize("@user https://x.y ...") == []


def test_normalize_lowercases_and_unescapes():
    assert normalize("HELLO &lt;3") == "hello <3"


def test_pad_truncates_and_fills():
    assert pad([5, 6], 4) == [5, 6, 0, 0]
    assert pad([1, 2, 3, 4, 5], 3) == [1, 2, 3]


def test_short_text_is_one_padded_window():
    assert windows([4, 5, 6], 5) == [[4, 5, 6, 0, 0]]


def test_long_text_windows_overlap_and_reach_the_end():
    ids = list(range(1, 12))  # 11 ids, windows of 4
    result = windows(ids, 4)
    assert all(len(window) == 4 for window in result)
    assert result[0] == [1, 2, 3, 4]
    assert result[-1] == [8, 9, 10, 11]
    covered = {i for window in result for i in window}
    assert covered == set(ids)


def test_vocabulary_build_respects_min_count_and_size():
    token_lists = [["a", "b", "a"], ["a", "c", "b"], ["d"]]
    vocab = Vocabulary.build(token_lists, max_size=4, min_count=2)
    assert vocab.tokens == [PAD, UNKNOWN, "a", "b"]
    assert vocab.encode(["a", "b", "c", "zzz"]) == [2, 3, 1, 1]
    assert "a" in vocab and "c" not in vocab


def test_vocabulary_round_trips_through_json(tmp_path):
    vocab = Vocabulary([PAD, UNKNOWN, "café", "\U0001F602"])
    vocab.save(tmp_path / "vocab.json")
    assert Vocabulary.load(tmp_path / "vocab.json").tokens == vocab.tokens


def test_vocabulary_must_start_with_special_tokens():
    with pytest.raises(ValueError):
        Vocabulary(["a", "b"])
