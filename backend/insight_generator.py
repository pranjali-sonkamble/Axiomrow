# backend/insight_generator.py

import logging
import re

import numpy as np
import pandas as pd

from backend.llm_agent import call_llm
from backend.data_loader import get_llm_context, get_data_profile
from backend.answer_grounding import content_is_untrustworthy, build_known_values
from prompts.system_prompts import INSIGHT_GENERATOR_PROMPT

logger = logging.getLogger("axiomrow.insights")

# ── CLAIM CHECKS ────────────────────────────────────────────────────
# The number/name grounding in answer_grounding.py cannot see these: a
# derived ratio ("double the average") adds no new digits, and an
# editorial word ("Outlier", "Extremes") makes a statistical claim with
# no numbers at all. Both were reaching the UI.
#   * ratio words: the model does arithmetic itself, unverified, and for
#     any evenly spread column max/mean is ~2, so "double" is trivially
#     "true" and meaningless. Always rejected.
#   * severity words: allowed only for a column the fact sheet reports
#     as actually having values beyond the 1.5xIQR fences.
_RATIO_RE = re.compile(
    r"\b(?:double[sd]?|doubling|twice|triple[sd]?|thrice|"
    r"\d+(?:\.\d+)?\s*(?:x|times)\b|times\s+(?:the|as|higher|more|larger|greater))",
    re.IGNORECASE)
_SEVERITY_RE = re.compile(r"\b(?:outliers?|anomal\w*|spikes?|surges?|extremes?|extremely)\b", re.IGNORECASE)
_MAX_FACT_NUMERIC = 10
_MAX_FACT_CATEGORICAL = 6


def _is_id_like(series, name) -> bool:
    n = str(name).lower().replace(" ", "_")
    if n in ("id", "index", "uuid") or n.endswith("_id") or n.startswith("id_"):
        return True
    return bool(pd.api.types.is_integer_dtype(series) and len(series) > 20
                and series.nunique() == series.notna().sum())


def _fact_sheet(df):
    """Verified, per-column facts computed from the FULL DataFrame, given to
    the model so it can tell evenly spread data from skewed data and knows
    whether outliers genuinely exist. Returns (text, outlier_columns)."""
    lines, outlier_cols = [], []

    numeric = [c for c in df.select_dtypes(include="number").columns
               if not pd.api.types.is_bool_dtype(df[c]) and not _is_id_like(df[c], c)]
    for c in numeric[:_MAX_FACT_NUMERIC]:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) < 4:
            continue
        q1, med, q3 = (float(x) for x in s.quantile([.25, .5, .75]))
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out = int(((s < lo) | (s > hi)).sum()) if iqr > 0 else 0
        skew = float(s.skew()) if len(s) >= 3 else 0.0
        shape = ("roughly symmetric" if abs(skew) < 0.5
                 else "right-skewed" if skew > 0 else "left-skewed")
        parts = [f"min {s.min():.2f}", f"q1 {q1:.2f}", f"median {med:.2f}", f"q3 {q3:.2f}",
                 f"max {s.max():.2f}", f"mean {s.mean():.2f}", f"std {s.std():.2f}",
                 f"distribution {shape} (skew {skew:.2f})"]
        if out:
            outlier_cols.append(str(c))
            parts.append(f"{out:,} values ({out / len(s) * 100:.2f}%) fall outside the 1.5xIQR fences")
        else:
            parts.append("every value lies inside the 1.5xIQR fences")
        neg = int((s < 0).sum())
        if neg:
            parts.append(f"{neg:,} values are negative")
        lines.append(f"- {c}: " + "; ".join(parts))

    cats = [c for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_datetime64_any_dtype(df[c])
            and 1 < df[c].nunique() <= 30]
    for c in cats[:_MAX_FACT_CATEGORICAL]:
        vc = df[c].dropna().astype(str).value_counts()
        if len(vc):
            lines.append(f"- {c}: {len(vc)} distinct values; most frequent '{vc.index[0]}' "
                         f"is {vc.iloc[0] / vc.sum() * 100:.1f}% of rows")

    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            d = df[c].dropna()
            if len(d):
                lines.append(f"- {c}: {d.min():%Y-%m-%d} to {d.max():%Y-%m-%d}")

    return "\n".join(lines), outlier_cols


def _unsupported_claims(title, body, outlier_cols) -> list:
    """Reasons an insight's wording over-claims. Checks title AND body."""
    text = f"{title} {body}"
    problems = []
    m = _RATIO_RE.search(text)
    if m:
        problems.append(f"model-computed ratio '{m.group(0)}'")
    m = _SEVERITY_RE.search(text)
    if m and not any(col.lower() in text.lower() for col in outlier_cols):
        problems.append(f"'{m.group(0)}' not supported by the computed outlier check")
    return problems


def generate_insights(df, session_id: str = None) -> list:
    """
    Automatically generates insights when a CSV is uploaded.
    Returns a list of insight dicts: [{"title": "...", "body": "..."}]

    Every insight is checked against the dataset profile before being
    returned — the SAME grounding logic used for chat answers
    (backend/answer_grounding.py). This used to be missing entirely:
    the parsed LLM text was shown as-is, so a fabricated number, a
    category that doesn't exist, or an unearned "caused by" claim
    would have gone straight to the UI as an "AI insight" with no
    check at all. An ungrounded insight is dropped, not shown — the
    UI already renders "No insights could be generated" for an empty
    list, so there is nothing extra to build for that case.

    session_id is optional and enables per-session API rate limiting
    (see call_llm / rate_limiter.py) — pass the caller's real session
    id in production; leave as None for direct/test calls.
    """
    # Build the profile once and reuse it for the LLM context. Profiling can
    # scan every column, so doing it twice on every upload is unnecessary work.
    profile = get_data_profile(df)
    context = get_llm_context(df, profile=profile)

    facts, outlier_cols = _fact_sheet(df)

    prompt = f"""Dataset overview:
- {profile['row_count']} rows, {profile['column_count']} columns
- Missing values: {profile['missing_values_total']}

Full column details:
{context}

FACT SHEET (computed from every row; the ONLY source for claims about spread,
outliers, skew, negatives and category shares):
{facts}

Generate 4 concise, specific insights about this dataset.
"""

    messages = [
        {"role": "system", "content": INSIGHT_GENERATOR_PROMPT},
        {"role": "user", "content": prompt}
    ]

    raw = call_llm(messages, temperature=0.4, session_id=session_id)

    if raw.startswith("Error:"):
        return [{"title": "Insights unavailable", "body": raw[len("Error: "):]}]

    insights = parse_insights(raw)

    # Ground each insight against the profile: it is a real computation
    # over the actual DataFrame (get_data_profile), so anything the
    # insight claims should be traceable to it. allowed_text carries the
    # column names so referencing a real column by name is never flagged.
    # The fact sheet is computed from the real frame, so numbers in it are
    # legitimate evidence too.
    context = context + "\n" + facts
    allowed_text = context + " " + " ".join(str(c) for c in df.columns)
    known_values = build_known_values(df)

    grounded = []
    for item in insights:
        # Ground the BODY only, not the title. Insight titles are
        # editorial labels ("Revenue Spread", "Missing Data") and are
        # usually Title Case — every word capitalized — which trips the
        # entity check built for normal prose (only proper nouns
        # capitalized after the first word). The title makes no factual
        # claim on its own; the body is where a fabricated number,
        # category, or causal claim would actually appear.
        if content_is_untrustworthy(item["body"], context, allowed_text, known_values):
            logger.info("Dropped ungrounded insight: %r", item["title"][:80])
            continue
        # Title included here: the title is where "Outlier" / "Extremes"
        # slipped through when only the body was checked.
        claims = _unsupported_claims(item["title"], item["body"], outlier_cols)
        if claims:
            logger.info("Dropped over-claiming insight %r: %s", item["title"][:80], "; ".join(claims))
            continue
        grounded.append(item)

    # Too few survived: top up with the dataframe-verified findings that the
    # exported report already uses, so this tab and the report agree and
    # the user is not left with one or two cards.
    if len(grounded) < 3:
        try:
            from backend.report_generator import _deterministic_insights
            have = {i["title"] for i in grounded}
            for title, body in _deterministic_insights(df):
                if title not in have and len(grounded) < 4:
                    grounded.append({"title": title, "body": body})
        except Exception:
            logger.exception("Deterministic insight top-up failed (non-fatal)")

    return grounded


def parse_insights(raw_text: str) -> list:
    """Parses the model's reply into [{"title", "body"}, ...].

    Handles TWO shapes, because the prompt was changed to ask for a
    structured block (title line, then Finding/Evidence/Interpretation
    lines) but the model doesn't always follow formatting exactly:

      [Insight Title]                    <- title line
      Finding: ...                       <- these three lines group
      Evidence: ...                         into ONE insight under
      Interpretation: ...                   the title above them

    The old line-by-line parser split each of "Finding:", "Evidence:",
    "Interpretation:" into a SEPARATE fake insight the moment the prompt
    changed to this block format — three throwaway cards per real
    insight, each missing its actual title. This version groups a title
    line together with the Finding/Evidence/Interpretation lines that
    follow it into a single insight.
    """
    insights = []
    lines = [ln.strip() for ln in raw_text.strip().split('\n')]

    _PARTS = ("finding:", "evidence:", "interpretation:")

    def is_part_line(line):
        return line.lower().startswith(_PARTS)

    def clean(line):
        # Strip markdown bold/heading markers the model sometimes adds
        # despite being told not to.
        return line.replace('*', '').lstrip('#').strip()

    current_title = None
    current_parts = []

    def flush():
        if current_title and current_parts:
            body = " ".join(current_parts).strip()
            if len(body) > 15:
                insights.append({"title": clean(current_title).strip("[] "), "body": clean(body)})

    for line in lines:
        if not line:
            continue
        if is_part_line(line):
            # "Finding: X" -> keep "X" (with its label, for readability:
            # "Finding: X. Evidence: Y." reads fine as a card body).
            current_parts.append(clean(line))
            continue

        # A new non-part line starts a new insight; flush the previous one.
        flush()
        current_title = line
        current_parts = []

    flush()

    # Fallback: the model ignored the block format entirely and wrote
    # old-style single "Title: body" or "Title\nbody" lines with no
    # Finding/Evidence/Interpretation labels at all.
    if not insights:
        for line in lines:
            if not line:
                continue
            c = clean(line)
            if ':' in c and len(c.split(':')[0]) < 50:
                title, body = c.split(':', 1)
                title, body = title.strip().lstrip('-0123456789. '), body.strip()
                if title and len(body) > 20:
                    insights.append({"title": title, "body": body})
            elif len(c) > 30:
                insights.append({"title": "Observation", "body": c})

    # Deduplicate and limit.
    seen = set()
    unique = []
    for item in insights:
        if item['body'] not in seen:
            seen.add(item['body'])
            unique.append(item)

    return unique[:5]


def _looks_like_date_column(series, name) -> bool:
    """True for a real datetime64 dtype, OR for an object column whose
    name hints at dates/times AND whose actual values mostly parse as
    dates. Needing BOTH avoids two opposite failure modes found while
    testing this: a name-only check misses columns like "created_at"
    (no "date"/"time" substring), while a slightly broader name-only
    check would then wrongly catch unrelated columns that happen to
    share a substring (e.g. "flat_rate", "update_reason"). Requiring
    the content to actually parse as dates rules those out."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    name_l = str(name).lower()
    if not any(hint in name_l for hint in ("date", "time", "_at", "_on", "day", "timestamp")):
        return False
    sample = series.dropna().astype(str).head(20)
    if len(sample) == 0:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() >= 0.8


def get_quick_stats(df) -> dict:
    """
    Returns fast, no-LLM statistics for instant display.

    WHY have this separate from insights?
    LLM calls take 2-5 seconds. We show these quick stats
    immediately while insights are loading. Good UX pattern:
    show something fast, then enrich with AI results.
    """
    stats = {
        "numeric_columns": [],
        "categorical_columns": [],
        "date_columns": [],
        "high_missing": []
    }

    for col in df.columns:
        series = df[col]
        n = len(series)
        missing_pct = (series.isnull().sum() / n) * 100 if n else 0

        if missing_pct > 20:
            stats["high_missing"].append({
                "column": col,
                "missing_pct": round(missing_pct, 1)
            })

        # Date/time check FIRST. A date/datetime-like column loaded from
        # CSV is dtype "object" (pandas does not auto-parse dates), so
        # putting the "object" branch first — as this used to — meant
        # every such column was caught there and the date-name check
        # below it never ran on the common case. is_numeric_dtype/
        # is_datetime64_any_dtype (not a literal 'int64'/'object' string
        # check) also keeps this correct on pandas 3.0+, where string
        # columns report a native "str" dtype instead of "object".
        is_date_like = _looks_like_date_column(series, col)

        if pd.api.types.is_numeric_dtype(series):
            valid = series.dropna()
            if len(valid) == 0:
                continue
            stats["numeric_columns"].append({
                "name": col,
                "mean": round(float(valid.mean()), 2),
                "max": valid.max(),
                "min": valid.min()
            })

        elif is_date_like:
            stats["date_columns"].append(col)

        else:
            valid = series.dropna()
            top_value = "N/A"
            if len(valid) > 0:
                modes = valid.mode()
                # mode() on a Series that's non-empty but has no repeats
                # still returns something, but guard anyway: an all-null
                # column used to raise IndexError here on modes[0].
                if len(modes) > 0:
                    top_value = modes.iloc[0]
            stats["categorical_columns"].append({
                "name": col,
                "unique_count": int(series.nunique()),
                "top_value": top_value
            })

    return stats
