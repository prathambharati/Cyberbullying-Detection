"""Morse code for the ten digits, which is all a PIN needs.

Every digit is exactly five symbols long, so the keypad knows a digit is done
after the fifth dot or dash. No timing gaps between letters are needed.
"""

DIGITS = {
    "-----": "0",
    ".----": "1",
    "..---": "2",
    "...--": "3",
    "....-": "4",
    ".....": "5",
    "-....": "6",
    "--...": "7",
    "---..": "8",
    "----.": "9",
}
CODES = {digit: code for code, digit in DIGITS.items()}
SYMBOLS_PER_DIGIT = 5


def decode_digit(code: str) -> str | None:
    """'..---' -> '2'. Returns None for anything that isn't a digit."""
    return DIGITS.get(code)


def encode(number: str) -> str:
    """'42' -> '....- ..---', handy for showing people what to blink."""
    return " ".join(CODES[digit] for digit in number)
