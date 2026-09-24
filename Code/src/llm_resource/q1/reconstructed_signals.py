"""Text-derived RedPajama-like signals audited against the supplied A1 scalars.

The source definitions are in togethercomputer/RedPajama-Data (Apache-2.0).
The supplied release stores document scalars, with some semantics that differ
from the original per-line/word definitions. Every field must pass A1 audit
before entering Q transfer.
"""
from __future__ import annotations

from collections import Counter
import math
import re
import string
import unicodedata


RPS_FIELDS = (
    "rps_doc_word_count", "rps_doc_num_sentences", "rps_doc_unigram_entropy",
    "rps_doc_frac_unique_words", "rps_doc_frac_no_alph_words",
    "rps_doc_frac_chars_top_2gram", "rps_doc_frac_chars_top_3gram",
    "rps_lines_uppercase_letter_fraction",
    "rps_lines_ending_with_terminal_punctution_mark",
    "rps_lines_numerical_chars_fraction", "rps_doc_mean_word_length",
)
PERCENT_FIELDS = frozenset(x for x in RPS_FIELDS if "frac" in x or "ending_with" in x)
_DELETE_PUNCT = str.maketrans("", "", string.punctuation)
_SENTENCE = re.compile(r"\b[^.!?]+[.!?]*", re.UNICODE)
_LINES = re.compile(r"([^\n]*\n|[^\n]+$)")
_ALPHA = re.compile(r"[a-zA-Z]")


def normalize(text: str) -> str:
    text = text.translate(_DELETE_PUNCT).lower().strip()
    return unicodedata.normalize("NFD", re.sub(r"\s+", " ", text))


def _top_ngram_fraction(words: list[str], n: int) -> float:
    grams = Counter(zip(*(words[i:] for i in range(n))))
    if not grams:
        return 0.0
    gram, count = grams.most_common(1)[0]
    total_chars = sum(map(len, words))
    return round(sum(map(len, gram)) * count / total_chars, 8) if count > 1 and total_chars else 0.0


def rps_signals(text: str) -> dict[str, float]:
    """Return all 11 raw-scalar fields in the A1 release's percentage units."""
    norm_words = normalize(text).split()
    n = len(norm_words)
    counts = Counter(norm_words)
    entropy = -sum((count / n) * math.log(count / n) for count in counts.values()) if n else math.nan
    raw_lines = [match.group() for match in _LINES.finditer(text)]
    norm_lines = [normalize(line) for line in raw_lines]
    uppercase = [sum(ch.isupper() for ch in line) / len(line) if line else 0.0 for line in raw_lines]
    numerical = [sum(ch.isdigit() for ch in line) / len(line) if line else 0.0 for line in norm_lines]
    # Candidate aggregation for an ambiguous released line field. It is not
    # assumed equivalent to the official scalar; the A1 audit can reject it.
    punctuation_lines = text.split("\n")
    punctuation = [float(line.rstrip().rstrip("}])\"' ”").endswith((".", "!", "?", "”")))
                   for line in punctuation_lines]
    # The A1 release aggregates RedPajama line-level signals to percentages.
    line_mean = lambda values: 100.0 * sum(values) / len(values) if values else math.nan
    return {
        "rps_doc_word_count": float(n),
        "rps_doc_num_sentences": float(len(_SENTENCE.findall(text))),
        "rps_doc_unigram_entropy": round(entropy, 8) if n else math.nan,
        "rps_doc_frac_unique_words": 100.0 * len(counts) / n if n else math.nan,
        # The released A1 scalar is a nonalphabetic CHARACTER percentage,
        # despite its RedPajama word-oriented field name. Verify on A1.
        "rps_doc_frac_no_alph_words": 100.0 * sum(not bool(_ALPHA.search(ch)) for ch in text) / len(text) if text else math.nan,
        "rps_doc_frac_chars_top_2gram": 100.0 * _top_ngram_fraction(norm_words, 2),
        "rps_doc_frac_chars_top_3gram": 100.0 * _top_ngram_fraction(norm_words, 3),
        "rps_lines_uppercase_letter_fraction": line_mean(uppercase),
        "rps_lines_ending_with_terminal_punctution_mark": line_mean(punctuation),
        "rps_lines_numerical_chars_fraction": line_mean([value for line, value in zip(norm_lines, numerical) if line]),
        "rps_doc_mean_word_length": sum(map(len, norm_words)) / n if n else math.nan,
    }
