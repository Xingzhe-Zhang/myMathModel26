"""Field semantics. Percent fields in the supplied release use 0..100."""
import numpy as np

FIELDS = [
    "fineweb_edu", "fluency_en", "modernbert_cleanliness", "modernbert_readability",
    "modernbert_reasoning", "modernbert_professionalism", "qurater", "ad_en",
    "dsir_books", "dsir_wiki", "dsir_math", "rps_doc_word_count",
    "rps_doc_num_sentences", "rps_doc_unigram_entropy", "rps_doc_frac_unique_words",
    "rps_doc_frac_no_alph_words", "rps_doc_frac_chars_top_2gram", "rps_doc_frac_chars_top_3gram",
    "rps_lines_uppercase_letter_fraction", "rps_lines_ending_with_terminal_punctution_mark",
    "rps_lines_numerical_chars_fraction", "rps_doc_mean_word_length",
]
MODERN = [f for f in FIELDS if f.startswith("modernbert_")]
PERCENT = [f for f in FIELDS if "frac" in f or f == "rps_lines_ending_with_terminal_punctution_mark"]
LIST_LENGTHS = {"fineweb_edu": 1, "fluency_en": 2, "ad_en": 2, "qurater": 4, **{f: 6 for f in MODERN}}
RAW_COLUMNS = [c for f in FIELDS for c in ([f"qurater_{i}" for i in range(4)] if f == "qurater" else [f])]


def softmax(x):
    x = np.asarray(x, dtype=float)
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def decode_signals(record):
    """Return 25 scalars, four alternative ordinal ratings, and issue codes."""
    values, argmax, issues = [], [], []
    for f in FIELDS:
        v = record.get(f)
        try:
            if f in LIST_LENGTHS:
                if not isinstance(v, list) or len(v) != LIST_LENGTHS[f]:
                    raise ValueError("shape")
                a = np.asarray(v, dtype=float)
                if not np.all(np.isfinite(a)):
                    raise ValueError("nonfinite")
                if f == "qurater":
                    values.extend(a.tolist())
                    continue
                if f == "fineweb_edu":
                    v = a[0]
                elif f in ("fluency_en", "ad_en"):
                    v = softmax(a)[1]
                else:
                    v = softmax(a) @ (np.arange(6) / 5)
                    argmax.append(float(a.argmax() / 5))
            else:
                if isinstance(v, (list, dict, bool)):
                    raise ValueError("type")
                v = float(v)
                if not np.isfinite(v):
                    raise ValueError("nonfinite")
                if f in PERCENT:
                    if not 0 <= v <= 100:
                        raise ValueError("percent_range")
                    v /= 100
                elif f.startswith("rps_") and v < 0:
                    raise ValueError("negative_statistic")
            values.append(float(v))
        except (TypeError, ValueError, OverflowError) as exc:
            values.extend([np.nan] * (4 if f == "qurater" else 1))
            if f in MODERN:
                argmax.append(np.nan)
            issues.append(f"{f}:{exc}")
    return values, argmax, issues
