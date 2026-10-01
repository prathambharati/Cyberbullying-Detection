"""Score text from the command line.

    python -m cyberbullying "nobody likes you, just leave"
    python -m cyberbullying < comments.txt      (one comment per line)
"""

import sys

USAGE = 'usage: python -m cyberbullying "some text" ["more text" ...]'


def main(argv=None) -> int:
    texts = sys.argv[1:] if argv is None else list(argv)
    if not texts and not sys.stdin.isatty():
        texts = [line for line in sys.stdin.read().splitlines() if line.strip()]
    if not texts:
        print(USAGE, file=sys.stderr)
        return 2

    # Windows consoles can't print every emoji. Print a ? instead of crashing.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    from .classifier import Classifier

    classifier = Classifier.load()
    for text, prediction in zip(texts, classifier.predict(texts)):
        verdict = f"cyberbullying ({prediction.category})" if prediction.is_bullying else "fine"
        print(f"{prediction.score:.2f}  {verdict:26}  {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
