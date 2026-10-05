# backend/answer_grounding.py
"""
Answer grounding — the code that decides whether a sentence written by
the LLM is allowed to be shown next to (or instead of) a verified result.

WHY THIS IS ITS OWN MODULE
These functions used to live inside app.py. They are pure Python (no
Streamlit), so they belong in the backend where they can be unit-tested
directly — app.py is a Streamlit script and can't be imported in a test.

THE DESIGN RULE EVERYTHING HERE SERVES
    A number, a category name, or a claim about the data may only be
    shown to the user if it came out of code that actually ran on the
    user's DataFrame. Anything else is either regenerated from that
    verified output or suppressed — never displayed as if it were fact.

WHAT THIS CAN AND CAN'T GUARANTEE (honest limits)
    * It reliably catches: made-up or wrong numbers, numbers attached to
      the wrong label, category/product names that aren't in the verified
      output, hedging that contradicts a real result.
    * It cannot prove a *qualitative* sentence is true ("sales are
      improving"). That's why the prompt tells the model to state only
      what the verified output shows, and why the verified result box —
      not the sentence — is always the primary answer.
"""

import re
import logging

import pandas as pd

logger = logging.getLogger("axiomrow.grounding")


# Shown in place of the model's sentence when it fails validation.
# Never blank (a question that asks for an explanation shouldn't silently
# show nothing) and never a guess (so it can't itself need fact-checking).
UNTRUSTWORTHY_FRAMING_FALLBACK = (
    "The verified result above is accurate — I wasn't able to generate "
    "a reliable written summary for this one."
)

# Shown when the model answered a data question in plain prose (no code),
# twice in a row. We refuse to display a guess about the user's data.
UNVERIFIED_ANSWER_MESSAGE = (
    "I couldn't verify an answer to that by running code on your data, so "
    "I'm not going to guess. Try naming the column or the calculation you "
    "want — for example: \"how many unique values are in the product "
    "column?\""
)

# Shown when code ran but Python and SQL gave different answers.
RESULTS_DISAGREE_MESSAGE = (
    "The Python and SQL results above don't agree, so I'm not giving a "
    "written summary. Treat both with caution and try rephrasing the "
    "question more specifically."
)

# Shown when neither the Python nor the SQL could be executed.
NO_VERIFIED_RESULT_MESSAGE = (
    "I wasn't able to run an analysis for this question, so I don't have "
    "a verified answer. Try rephrasing it, or name the column you want."
)


# ============================================================
# NUMBER EXTRACTION
# ============================================================

def extract_numbers(text: str) -> list:
    """Numeric tokens in text as floats, normalising thousands separators
    ("350,110,212" -> 350110212.0). A '-' only counts as a minus sign when
    it isn't glued to a preceding digit, so dates like 2024-03-01 don't
    produce a phantom -3 and -1."""
    if not text:
        return []
    numbers = []
    for tok in re.findall(r'(?<!\d)-?\d[\d,]*(?:\.\d+)?', text):
        try:
            numbers.append(float(tok.replace(",", "")))
        except ValueError:
            continue
    return numbers


_LABEL_STOPWORDS = {
    "has", "have", "had", "are", "was", "were", "with", "from", "than", "about", "over",
    "under", "around", "across", "between", "only", "just", "all", "and", "for", "the",
    "contains", "contain", "spans", "ranges", "approximately", "roughly", "nearly",
}


def extract_labeled_numbers(text: str) -> dict:
    """(label, value) pairs using the single word right before each
    number as the label: "Laptop 40.0" -> {"laptop": 40.0}. Catches
    values that are individually real but attached to the wrong label
    (distribution answers reuse the same round numbers across labels).
    Heuristic by design — layered on top of the plain number check."""
    pairs = {}
    if not text:
        return pairs
    for m in re.finditer(r'\b([A-Za-z][A-Za-z]{2,20})\s+(-?\d[\d,]*(?:\.\d+)?)\s*%?', text):
        label = m.group(1).strip().lower()
        if label in _LABEL_STOPWORDS:   # "has 5", "from 2": grammar, not a data label
            continue
        try:
            value = float(m.group(2).replace(",", ""))
        except ValueError:
            continue
        pairs.setdefault(label, value)
    return pairs


def derived_counts(verified_text: str) -> list:
    """Counts that are *visible* in the verified output even though they
    aren't printed as digits — e.g. ['Headphones', 'Laptop', 'Phone',
    'Tablet'] has 4 items, and a 5-line table has 5 (or 4, without its
    header) rows. Lets a correct sentence like "there are 4 products"
    pass when the code printed the list but not the count.

    Deliberately narrow: only list length and line counts, nothing that
    would let an arbitrary wrong number through."""
    counts = []
    if not verified_text:
        return counts

    lines = [ln for ln in verified_text.splitlines() if ln.strip()]
    if lines:
        counts.extend([float(len(lines)), float(max(len(lines) - 1, 0))])

    for inner in re.findall(r'\[([^\[\]]*)\]', verified_text):
        items = [i for i in inner.split(",") if i.strip()]
        if items:
            counts.append(float(len(items)))
    return counts


# ============================================================
# ENTITY / CATEGORY-VALUE GROUNDING
# ============================================================

_MAX_DISTINCT_PER_COLUMN = 5000   # skip ID-like columns; values there aren't "categories"
_MIN_VALUE_LEN = 3                # ignore 1-2 char values ("A", "NY") — too many false hits


# Caches the last DataFrame's known values, keyed by identity + shape.
# Scanning every text column and calling .unique() on each is real work
# on a large upload (the project's 1.4M-row test file), and this function
# runs on every single question. Without caching that scan repeated every
# time, which is what made the app "slow to load" after grounding was
# added — a dataset doesn't change between questions, so there's no need
# to redo it. Size-1 cache is enough: only one dataset is loaded at a time.
_known_values_cache = {}


def build_known_values(df) -> set:
    """Lower-cased distinct values of every text/categorical column.
    Used to spot a category name appearing in prose that the verified
    output never mentioned. Built with iloc so duplicate column names
    can't break it. Cached per DataFrame — see note above."""
    if df is None:
        return set()

    key = (id(df), df.shape, tuple(str(c) for c in df.columns))
    if _known_values_cache.get("key") == key:
        return _known_values_cache["values"]

    values = set()
    for i in range(df.shape[1]):
        try:
            s = df.iloc[:, i]
            is_texty = (
                pd.api.types.is_object_dtype(s)
                or pd.api.types.is_string_dtype(s)
                or isinstance(s.dtype, pd.CategoricalDtype)
            )
            if not is_texty:
                continue
            uniq = s.dropna().unique()
            if len(uniq) > _MAX_DISTINCT_PER_COLUMN:
                continue
            for v in uniq:
                sv = str(v).strip().lower()
                if len(sv) >= _MIN_VALUE_LEN and not sv.replace(".", "").isdigit():
                    values.add(sv)
        except Exception:
            logger.exception("build_known_values skipped a column")
            continue

    _known_values_cache["key"] = key
    _known_values_cache["values"] = values
    return values


# Capitalised words that are ordinary English / product vocabulary and
# don't need to appear in the verified output.
_COMMON_CAPITALIZED = {
    "the", "this", "that", "these", "those", "there", "here", "then", "they",
    "and", "but", "for", "with", "from", "into", "over", "under", "about",
    "overall", "also", "based", "note", "result", "results", "answer",
    "total", "average", "count", "number", "value", "values", "data",
    "dataset", "column", "columns", "row", "rows", "table", "sales",
    "python", "sql", "dataframe", "axiomrow", "verified", "unique",
    "distinct", "each", "every", "some", "most", "least", "both", "only",
    "yes", "not", "you", "your", "its", "our", "all", "any", "one", "two",
    "when", "where", "which", "while", "since", "because", "however",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday",
}


def _norm_word(w: str) -> str:
    """Cheap singular/plural + case normalisation ('Laptops' == 'laptop')."""
    w = w.lower()
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("s") and len(w) > 3:
        return w[:-1]
    return w


def _mid_sentence_capitalized_words(text: str) -> list:
    """Capitalised words that are NOT the first word of a sentence —
    those are the ones that look like proper names / category values
    (Laptop, Monitor, North). Sentence-initial words are skipped because
    every sentence starts with a capital letter."""
    found = []
    for m in re.finditer(r"(?<![A-Za-z0-9'’])([A-Z][a-z]{2,})(?![A-Za-z0-9])", text):
        before = text[:m.start()].rstrip(" \t*_#>|\"'(-•")
        if not before or before[-1] in ".!?\n":
            continue
        found.append(m.group(1))
    return found


def _word_in(term: str, text_lower: str) -> bool:
    if term not in text_lower:          # fast path before the regex
        return False
    return re.search(rf'(?<!\w){re.escape(term)}(?!\w)', text_lower) is not None


def find_ungrounded_entities(content: str, verified_text: str, allowed_text: str = "") -> list:
    """Capitalised, name-like words in `content` that appear neither in
    the verified output nor in `allowed_text` (the user's question and
    the real column names). This is what catches an invented value like
    "Monitor" in "Laptop, Phone, Tablet, and Monitor" when the verified
    output listed Headphones, Laptop, Phone and Tablet."""
    if not content:
        return []
    pool = (verified_text or "") + " " + (allowed_text or "")
    pool_words = {_norm_word(w) for w in re.findall(r"[A-Za-z]{3,}", pool)}
    bad = []
    for word in _mid_sentence_capitalized_words(content):
        n = _norm_word(word)
        if n in pool_words or word.lower() in _COMMON_CAPITALIZED or n in _COMMON_CAPITALIZED:
            continue
        if word not in bad:
            bad.append(word)
    return bad


# ============================================================
# THE MAIN CHECK
# ============================================================

_HEDGE_PHRASES = (
    "don't have access", "do not have access",
    "no access to the", "unable to compute", "unable to access",
    "can't compute the exact", "cannot compute the exact",
    "i'm sorry, but i don't", "i am sorry, but i do not",
    "i don't have the actual data", "without access to the actual",
)

# Causal/statistical language a groupby or aggregation never actually
# establishes. A sentence can pass the number- and name-checks above
# (e.g. "caused by a marketing campaign in the North region" — North is a
# real region, no invented number) while still overclaiming. This is a
# separate check because it's about the STRENGTH of a claim, not its facts.
_CAUSAL_OVERREACH_PHRASES = (
    "caused by", "was caused", "were caused", "caused the", "caused a",
    "causes the", "causes a",
    "led to", "resulted in", "resulting in", "due to the", "is due to",
    "because of the", "is because", "the reason for", "the reason behind",
    "significant correlation", "statistically significant", "is an outlier",
    "are outliers", "is an anomaly", "an anomaly", "are anomalies",
    "is anomalous", "this proves", "guaranteed to",
    "will definitely", "predicts that", "forecasts that",
)
# "this confirms" was removed: it flagged harmless sentences like "this
# confirms the dataset size" (found via testing, not assumed) — too
# generic a phrase to ban on its own, unlike "this proves" which reads
# as a stronger, rarer claim of certainty.


def find_causal_overreach(text: str) -> list:
    """Phrases that assert causation, statistical significance, or
    prediction — none of which a plain aggregation or groupby can
    establish on its own. Not a data-accuracy check (the facts named
    may be entirely real); it's a claim-strength check."""
    if not text:
        return []
    lowered = text.lower()
    return [p for p in _CAUSAL_OVERREACH_PHRASES if p in lowered]


_UNSUPPORTED_PREDICTION_PATTERNS = (
    # Allow adverbs such as "definitely", "probably", or "likely" between
    # the future auxiliary and the predicted outcome.
    r"\bwill(?:\s+\w+){0,2}\s+(?:increase|decrease|rise|fall|grow|drop|reach|remain|be|become|decline|improve|worsen)\b",
    r"\b(?:is|are|was|were)\s+(?:expected|projected|forecast)\s+to\b",
    r"\b(?:forecast|forecasts|forecasted|project|projects|projected|predict|predicts|predicted)\b",
    r"\b(?:likely|unlikely)\s+to\s+(?:increase|decrease|rise|fall|grow|drop|reach|remain|decline|improve|worsen)\b",
    r"\b(?:should|would)\s+(?:increase|decrease|rise|fall|grow|drop|reach|remain|decline|improve|worsen)\b",
)


def find_unsupported_prediction_claims(text: str) -> list:
    """Return future/prediction language requiring verified forecasting."""
    if not text:
        return []
    lowered = str(text).lower()
    return [pattern for pattern in _UNSUPPORTED_PREDICTION_PATTERNS
            if re.search(pattern, lowered)]


def find_grounding_problems(content: str, verified_text: str,
                            allowed_text: str = "", known_values=None) -> list:
    """Returns a list of human-readable reasons `content` can't be
    trusted next to `verified_text` (empty list == passes).

      allowed_text  — text the user supplied that the model may legitimately
                      repeat back: their question + the real column names.
      known_values  — output of build_known_values(df); enables the
                      "category name absent from verified output" check.
                      Pass None to skip it (cheap history-replay path).

    Checks: hedging; numbers not in the verified output; numbers attached
    to the wrong label; name-like words absent from the verified output;
    real category values the verified output never mentioned; causal or
    statistical-significance language stronger than an aggregation supports.
    """
    problems = []
    if not content:
        return problems

    lowered = content.lower()

    overreach = find_causal_overreach(content)
    if overreach:
        problems.append("claims causation/significance not established by the code: " + ", ".join(overreach))

    prediction = find_unsupported_prediction_claims(content)
    if prediction:
        problems.append("contains unsupported future/prediction language")

    if any(p in lowered for p in _HEDGE_PHRASES):
        problems.append("hedges or claims no data access despite a real result")

    # Numbers. Pool = verified numbers + counts visible in the output
    # (list lengths / row counts) + numbers the user themselves typed.
    pool = (extract_numbers(verified_text)
            + derived_counts(verified_text)
            + extract_numbers(allowed_text))
    for n in extract_numbers(content):
        # Absolute tolerance, not relative: a cent of rounding is fine,
        # but 1% of a 350M total would let a genuinely wrong figure pass.
        if not any(abs(v - n) <= 0.01 for v in pool):
            problems.append(f"number {n:g} is not in the verified output")
            break

    # Same-label comparison (catches swapped values that each "exist").
    labeled_content = extract_labeled_numbers(content)
    labeled_verified = extract_labeled_numbers(verified_text)
    for label, c_val in labeled_content.items():
        v_val = labeled_verified.get(label)
        if v_val is not None and abs(v_val - c_val) > 0.01:
            problems.append(f"value for '{label}' differs from the verified output")
            break

    # Invented names.
    bad_entities = find_ungrounded_entities(content, verified_text, allowed_text)
    if bad_entities:
        problems.append("names not present in the verified output: " + ", ".join(bad_entities))

    # Real category values that the verified output never mentioned.
    if known_values:
        ver_l = (verified_text or "").lower()
        allow_l = (allowed_text or "").lower()
        stray = [v for v in known_values
                 if _word_in(v, lowered) and not _word_in(v, ver_l) and not _word_in(v, allow_l)]
        if stray:
            problems.append("data values not in the verified output: " + ", ".join(sorted(stray)[:5]))

    return problems


def content_is_untrustworthy(content: str, verified_text: str,
                             allowed_text: str = "", known_values=None) -> bool:
    problems = find_grounding_problems(content, verified_text, allowed_text, known_values)
    if problems:
        logger.info("Framing suppressed: %s", "; ".join(problems))
    return bool(problems)


# ============================================================
# TEXT-ONLY ANSWERS
# ============================================================

def text_makes_data_claims(text: str, df=None, question: str = "", known_values=None) -> bool:
    """True if a plain-prose answer (no code behind it) says anything
    about the *contents* of the data: any digit, any real category value,
    or any name-like word not found in the question or column names.

    A reply with no such content ("Could you tell me which column you
    mean?", "Hello! Ask me about your data.") is safe to show as-is.
    Everything else must come from executed code."""
    if not text:
        return False
    if re.search(r'\d', text):
        return True

    if known_values is None:
        known_values = build_known_values(df)
    lowered = text.lower()
    if any(_word_in(v, lowered) for v in known_values):
        return True

    cols = " ".join(str(c) for c in df.columns) if df is not None else ""
    if find_ungrounded_entities(text, "", f"{question} {cols}"):
        return True
    return False


# ============================================================
# PYTHON vs SQL CROSS-CHECK
# ============================================================

def results_disagree(code_output, sql_output) -> bool:
    """The model writes Python AND SQL for the same question, which gives
    us two independent computations to compare. If the SQL returned a
    single number and none of the numbers Python printed match it, one
    of the two is wrong. Only the scalar case is compared (tables differ
    in layout too much to compare safely); a percent-vs-fraction
    difference (0.4 vs 40) is treated as agreement."""
    if not code_output or not sql_output:
        return False
    sql = str(sql_output).strip()
    if "\n" in sql:
        return False
    try:
        sql_val = float(sql.replace(",", ""))
    except ValueError:
        return False

    code_nums = extract_numbers(str(code_output))
    if not code_nums:
        return False

    for n in code_nums:
        tol = max(0.01, 1e-4 * abs(n), 0.5 if float(n).is_integer() else 0.0)
        for cand in (sql_val, sql_val * 100, sql_val / 100):
            if abs(cand - n) <= tol:
                return False
    return True


# ============================================================
# FRAMING TEXT TRIMMING
# ============================================================

def trim_framing_text(text: str, max_chars: int = 420) -> str:
    """Cuts a model response where it stops being a short explanation and
    starts restating data itself — a second paragraph, a markdown table,
    or a heading. Does NOT cut after the first sentence (that used to
    discard the useful part of multi-sentence explanations)."""
    if not text:
        return ""
    text = text.strip()
    for stop_marker in ("\n\n", "\n|", "\n#", "\n-", "\n*"):
        idx = text.find(stop_marker)
        if idx != -1:
            text = text[:idx]
    return text[:max_chars].strip()
