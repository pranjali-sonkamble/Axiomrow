# backend/report_generator.py
"""
Production-grade Axiomrow report generator.

Public API (kept compatible with the Streamlit app):
    generate_markdown_report(df, profile, insights, quick_stats, chats)
    generate_pdf_report(df, profile, insights, quick_stats, chats)

Design goals:
- deterministic, dataframe-verified statistics and insights
- appropriate EDA for numeric, categorical and date/time fields
- meaningful aggregation for category/time charts
- truthful chart interpretations (aggregation and granularity always match)
- professional PDF layout with wrapped tables, page control and chart captions
- saved chat Q&A at the very end, with chart reconstruction from chart_config
- no dependence on Plotly/Kaleido for report generation
"""

import base64
import io
import math
import os
import re
import logging
from datetime import datetime

import numpy as np
import pandas as pd


# Small in-process cache used only during report construction.  The dataframe
# object is kept alive by Streamlit, so its id is stable for the current run.
_REPORT_CACHE = {}
logger = logging.getLogger(__name__)


def _cache_for(df):
    key = id(df)
    cache = _REPORT_CACHE.get(key)
    if cache is None or cache.get("shape") != df.shape:
        cache = {"shape": df.shape}
        _REPORT_CACHE[key] = cache
    return cache


def _cached_column_groups(df):
    cache = _cache_for(df)
    if "column_groups" not in cache:
        cache["column_groups"] = _column_groups(df)
    return cache["column_groups"]


def _cached_whitespace_cols(df):
    cache = _cache_for(df)
    if "whitespace_cols" not in cache:
        found = []
        for col in df.select_dtypes(include=["object", "string"]).columns:
            sr = df[col].dropna()
            if len(sr) == 0:
                continue
            # Test the distinct values only: same answer, far fewer string ops.
            u = pd.Series(sr.astype(str).unique())
            if (u != u.str.strip()).any():
                found.append(str(col))
        cache["whitespace_cols"] = found
    return cache["whitespace_cols"]


def _cached_missing_total(df):
    cache = _cache_for(df)
    if "missing_total" not in cache:
        cache["missing_total"] = int(df.isna().sum().sum())
    return cache["missing_total"]


def _cached_duplicate_total(df):
    cache = _cache_for(df)
    if "duplicate_total" not in cache:
        cache["duplicate_total"] = int(df.duplicated().sum())
    return cache["duplicate_total"]


# ---------------------------------------------------------------------------
# TEXT / NUMBER HELPERS
# ---------------------------------------------------------------------------

def _safe_text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return "N/A"
    except Exception:
        pass
    if isinstance(value, float):
        if math.isnan(value):
            return "N/A"
        if value.is_integer():
            return f"{int(value):,}"
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    return str(value)


def _clean(text: str) -> str:
    """Convert text to a safe subset for FPDF built-in Helvetica fonts.
    Used only as a fallback when the bundled Poppins TTF isn't available
    — see _clean_unicode() for the normal path."""
    if text is None:
        return ""
    text = str(text)
    replacements = {
        "•": "-", "→": "->", "←": "<-", "—": "-", "–": "-",
        "“": '"', "”": '"', "‘": "'", "’": "'", "₹": "Rs.",
        "€": "EUR ", "£": "GBP ", "…": "...", "×": "x", "≤": "<=", "≥": ">=",
        "−": "-",
    }
    for a, b in replacements.items():
        text = text.replace(a, b)
    return text.encode("latin-1", "replace").decode("latin-1")


def _clean_unicode(text) -> str:
    """Like _clean(), but for the bundled Poppins TTF, which covers far
    more than FPDF core fonts' latin-1 subset (accented Latin, common
    punctuation, currency symbols) — but it's still a Latin-script font,
    not a universal one, so it has no glyphs for e.g. CJK or Cyrillic.
    Left alone, FPDF silently drops characters it has no glyph for,
    which doesn't fail loudly — it just makes real data vanish from the
    report with no indication anything was there. Checking against the
    font's actual glyph coverage (cached once) and substituting a visible
    "?" for anything unsupported keeps that failure honest instead of
    silent, matching how _clean() already handles this for the
    Helvetica-fallback path."""
    if text is None:
        return ""
    text = str(text)
    covered = _poppins_glyph_coverage()
    if covered is None:
        # Coverage check unavailable (e.g. fontTools import failed) —
        # strip only control characters, same as before.
        return "".join(ch for ch in text if ch in ("\n", "\t") or ord(ch) >= 32)
    return "".join(
        ch if (ch in ("\n", "\t") or (ord(ch) >= 32 and ord(ch) in covered)) else "?"
        for ch in text
    )


_POPPINS_CMAP_CACHE = None


def _poppins_glyph_coverage():
    """Returns the set of Unicode code points the bundled Poppins font
    can actually render, computed once and cached for the process
    lifetime (checking this per-character per-call would be needlessly
    slow across a full report). Returns None if it can't be determined,
    so callers can fail open rather than crash report generation over a
    font-introspection problem."""
    global _POPPINS_CMAP_CACHE
    if _POPPINS_CMAP_CACHE is not None:
        return _POPPINS_CMAP_CACHE
    try:
        from fontTools.ttLib import TTFont
        font_path = os.path.join(os.path.dirname(__file__), "assets", "fonts", "Poppins-Regular.ttf")
        _POPPINS_CMAP_CACHE = set(TTFont(font_path).getBestCmap().keys())
    except Exception:
        _POPPINS_CMAP_CACHE = None
    return _POPPINS_CMAP_CACHE


def _fmt_number(value, decimals=2) -> str:
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
        value = float(value)
        if value.is_integer():
            return f"{int(value):,}"
        return f"{value:,.{decimals}f}"
    except Exception:
        return _safe_text(value)


def _fmt_pct(value, decimals=1) -> str:
    try:
        return f"{float(value):.{decimals}f}%"
    except Exception:
        return "N/A"


def _dataset_name(df, profile=None):
    # WHY not getattr(df, attr): pandas DataFrames expose columns as
    # attributes (df.name returns the "name" COLUMN if one exists), so
    # this used to crash with "truth value of a Series is ambiguous" on
    # any dataset with a column literally called name/filename/dataset_name
    # — a very common real-world column name. profile.get(key) is safe
    # (plain dict lookup, no collision risk) and is how app.py now passes
    # the real uploaded filename through (see st.session_state.loaded_filename).
    if isinstance(profile, dict):
        for key in ("dataset_name", "name", "file_name", "filename", "source"):
            if profile.get(key):
                return str(profile[key]).split("/")[-1]
    return "Uploaded dataset"


def _display_col(col):
    return str(col)


# ---------------------------------------------------------------------------
# TYPE DETECTION / PROFILING
# ---------------------------------------------------------------------------

def _column_groups(df):
    numeric = list(df.select_dtypes(include=[np.number]).columns)
    date_cols = list(df.select_dtypes(include=["datetime", "datetimetz"]).columns)

    # Detect date-like object/string columns without stealing numeric columns.
    for col in df.columns:
        if col in numeric or col in date_cols:
            continue
        name_hint = str(col).lower()
        # Cheap pre-check on a strided sample: skip columns that clearly are not dates
        # (avoids parsing 1M+ strings per text column).
        stride = max(1, len(df) // 500)
        probe = df[col].iloc[::stride].dropna().head(500)
        if len(probe):
            try:
                pr = pd.to_datetime(probe, errors="coerce", format="mixed")
            except (TypeError, ValueError):
                pr = pd.to_datetime(probe, errors="coerce")
            if float(pr.notna().mean()) < 0.5:
                continue
        try:
            parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
        except (TypeError, ValueError):
            parsed = pd.to_datetime(df[col], errors="coerce")
        ratio = float(parsed.notna().mean()) if len(parsed) else 0.0
        name_signal = any(k in name_hint for k in ("date", "time", "timestamp", "datetime"))
        if ratio >= 0.80 or (name_signal and ratio >= 0.60):
            date_cols.append(col)

    categorical = [c for c in df.columns if c not in numeric and c not in date_cols]
    return numeric, categorical, date_cols


def _normalized_datetime(df, col):
    return pd.to_datetime(df[col], errors="coerce")


def _profile_columns(df, profile=None):
    supplied = {}
    if isinstance(profile, dict):
        for item in profile.get("columns", []) or []:
            if isinstance(item, dict) and "name" in item:
                supplied[item["name"]] = item

    numeric, categorical, date_cols = _cached_column_groups(df)
    result = []
    for col in df.columns:
        s = df[col]
        item = dict(supplied.get(col, {}))
        item["name"] = col
        item["dtype"] = str(s.dtype)
        item["unique_count"] = int(s.nunique(dropna=True))
        item["missing_count"] = int(s.isna().sum())
        item["missing_pct"] = item["missing_count"] / max(len(df), 1) * 100

        if col in date_cols:
            item["detected_type"] = "Date/Time"
            item["description"] = "Date/time field used for temporal analysis."
        elif col in numeric:
            item["detected_type"] = "Numeric"
            item["description"] = "Numerical measure suitable for statistical analysis."
        else:
            item["detected_type"] = "Categorical"
            item["description"] = "Categorical field used for grouping and comparison."

        if col in numeric:
            clean = pd.to_numeric(s, errors="coerce").dropna()
            if len(clean):
                item["stats"] = {
                    "min": clean.min(), "max": clean.max(), "mean": clean.mean(),
                    "median": clean.median(), "std": clean.std(),
                    "q1": clean.quantile(.25), "q3": clean.quantile(.75),
                }
        else:
            vc = s.dropna().astype(str).value_counts().head(5)
            item["top_values"] = [(str(k), int(v)) for k, v in vc.items()]
        result.append(item)
    return result


# ---------------------------------------------------------------------------
# DATA QUALITY
# ---------------------------------------------------------------------------

def _outlier_profile(df):
    numeric, _, _ = _cached_column_groups(df)
    rows = []
    for col in numeric:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 4:
            rows.append({"column": col, "n": len(s), "q1": np.nan, "q3": np.nan, "iqr": np.nan, "count": None, "status": "Not assessed (<4 values)"})
            continue
        q1, q3 = s.quantile([.25, .75])
        iqr = q3 - q1
        if iqr == 0:
            count = 0
        else:
            count = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        rows.append({
            "column": col, "n": len(s), "q1": q1, "q3": q3, "iqr": iqr,
            "count": count, "status": "Potential outliers detected" if count else "No IQR outliers detected",
        })
    return rows


def _is_numeric_like_name(name: str) -> bool:
    """Match the same business-semantic numeric-name heuristic used by the
    deterministic Data Quality gate.  Negative values are only surfaced when
    a field name suggests a normally non-negative measure; they are not
    treated as confirmed invalid data.
    """
    s = re.sub(r"[^a-z0-9]+", "_", str(name).lower())
    hints = (
        "amount", "price", "revenue", "sales", "income", "salary",
        "quantity", "qty", "age", "score", "rate", "percent", "percentage",
        "discount", "cost", "profit", "balance",
    )
    return any(h in s.split("_") for h in hints) or any(
        h in s for h in ("revenue", "salary", "amount", "quantity", "price")
    )


def _negative_measure_findings(df):
    """Return non-blocking negative-value warnings consistent with data_quality.py."""
    findings = []
    numeric, _, _ = _cached_column_groups(df)
    for col in numeric:
        if not _is_numeric_like_name(col):
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if not len(s):
            continue
        negative_count = int((s < 0).sum())
        if negative_count:
            pct = negative_count / len(s) * 100
            findings.append({
                "column": str(col),
                "count": negative_count,
                "pct": pct,
            })
    return findings


def _quality_findings(df):
    findings = []
    duplicates = _cached_duplicate_total(df)
    missing_total = _cached_missing_total(df)

    findings.append(
        "No duplicate rows were detected." if not duplicates else
        f"{duplicates:,} duplicate row(s) detected ({duplicates / max(len(df), 1) * 100:.2f}% of records)."
    )
    if not missing_total:
        findings.append("No missing values were detected.")
    else:
        affected = [str(c) for c in df.columns if df[c].isna().any()]
        findings.append(f"{missing_total:,} missing value(s) detected across {len(affected)} column(s): {', '.join(affected[:8])}{'...' if len(affected) > 8 else ''}.")

    object_cols = list(df.select_dtypes(include=["object", "string"]).columns)
    whitespace_cols = _cached_whitespace_cols(df)
    findings.append(
        "No leading/trailing whitespace issue was detected in text fields." if not whitespace_cols else
        "Leading/trailing whitespace detected in: " + ", ".join(whitespace_cols[:8]) + "."
    )

    constant_cols = [str(c) for c in df.columns if df[c].nunique(dropna=False) <= 1]
    findings.append(
        "No constant columns were detected." if not constant_cols else
        "Constant column(s) detected: " + ", ".join(constant_cols[:8]) + "."
    )

    # Keep the report's Data Quality section aligned with the application gate:
    # suspicious negative values are warnings, not confirmed errors and not
    # readiness blockers.  This is intentionally placed before outlier
    # findings so the PDF/Markdown report surfaces the same issue the UI shows.
    for item in _negative_measure_findings(df):
        findings.append(
            f"Potentially invalid negative values: {item['count']:,} negative "
            f"value(s) ({item['pct']:.1f}%) detected in `{item['column']}`. "
            "Negative values may be legitimate (for example refunds or losses); "
            "validate the business rule before correcting source values."
        )

    outliers = _outlier_profile(df)
    detected = [f"{r['column']} ({r['count']})" for r in outliers if r["count"]]
    if detected:
        findings.append("Potential IQR outliers detected in: " + ", ".join(detected[:8]) + ".")
    else:
        findings.append("No IQR-based outliers were detected in numeric columns that were eligible for assessment.")
    return findings


def _quality_actions(df):
    numeric, categorical, date_cols = _cached_column_groups(df)
    actions = []
    dup = _cached_duplicate_total(df)
    miss = _cached_missing_total(df)
    actions.append("Duplicate check completed; no duplicate-row removal was required." if not dup else "Duplicate rows were identified and should be reviewed before aggregation; this report does not silently delete source records.")
    actions.append("Missing-value check completed; no missing-value treatment was required." if not miss else "Missing values were profiled; treatment should be chosen according to the analytical objective rather than silently imputing values.")
    if date_cols:
        actions.append(f"Date/time field(s) detected and parsed for temporal analysis: {', '.join(map(str, date_cols[:5]))}.")
    else:
        actions.append("No sufficiently reliable date/time column was detected.")
    if categorical:
        actions.append("Categorical fields were profiled for cardinality, frequency and group-level comparison.")
    if numeric:
        actions.append("Numeric fields were profiled using descriptive statistics, distributions and relationship analysis where applicable.")

    negative_findings = _negative_measure_findings(df)
    if negative_findings:
        for item in negative_findings:
            actions.append(
                f"Review {item['count']:,} negative value(s) ({item['pct']:.1f}%) "
                f"in {item['column']}; confirm the business definition before making corrections."
            )
    return actions


# ---------------------------------------------------------------------------
# STANDARD REPORT METRICS / METHODOLOGY
# ---------------------------------------------------------------------------

def _quality_score(df):
    """Transparent 0-100 data-quality score for reporting only.

    The score is a screening indicator, not an analysis-readiness gate.
    Penalties are intentionally bounded so a single issue cannot dominate.
    """
    n = max(len(df), 1)
    penalties = 0.0

    missing_pct = _cached_missing_total(df) / (n * max(len(df.columns), 1)) * 100
    duplicate_pct = _cached_duplicate_total(df) / n * 100
    constant_pct = (
        sum(df[c].nunique(dropna=False) <= 1 for c in df.columns)
        / max(len(df.columns), 1) * 100
    )

    object_cols = list(df.select_dtypes(include=["object", "string"]).columns)
    whitespace_pct = (
        len(_cached_whitespace_cols(df)) / max(len(object_cols), 1) * 100
        if object_cols else 0.0
    )

    outlier_total = sum(
        int(r["count"] or 0) for r in _outlier_profile(df)
        if r["count"] is not None
    )
    numeric_cells = sum(df[c].notna().sum() for c in _cached_column_groups(df)[0])
    outlier_pct = outlier_total / max(int(numeric_cells), 1) * 100

    # Bounded, transparent penalties.
    penalties += min(missing_pct * 0.60, 30)
    penalties += min(duplicate_pct * 0.50, 20)
    penalties += min(constant_pct * 0.40, 10)
    penalties += min(whitespace_pct * 0.20, 5)
    penalties += min(outlier_pct * 0.15, 10)

    score = max(0.0, min(100.0, 100.0 - penalties))
    if score >= 90:
        band = "Excellent"
    elif score >= 75:
        band = "Good"
    elif score >= 60:
        band = "Fair"
    else:
        band = "Needs Attention"

    return {
        "score": score,
        "band": band,
        "missing_pct": missing_pct,
        "duplicate_pct": duplicate_pct,
        "constant_pct": constant_pct,
        "whitespace_pct": whitespace_pct,
        "outlier_pct": outlier_pct,
    }


def _methodology_items():
    return [
        "Schema and column types are detected directly from the uploaded dataframe.",
        "Missing values and duplicate rows are counted from the source dataframe without silently modifying records.",
        "Numeric summaries use count, mean, median, standard deviation, variance, quartiles, minimum, maximum, skewness and coefficient of variation where calculable.",
        "Categorical analysis uses unique-value counts, frequency and share of non-missing observations.",
        "Date/time analysis uses detected date fields and adapts aggregation to the observed time span.",
        "Outliers are screened using the 1.5 x IQR rule; an outlier is not automatically treated as an error.",
        "Relationships are assessed with Pearson correlation for usable numeric pairs; correlation is not interpreted as causation.",
        "Charts are generated from the same dataframe-backed calculations used by the narrative so visual interpretations remain consistent with the reported statistics.",
    ]


def _date_profile_rows(df, date_cols):
    rows = []
    for col in date_cols:
        parsed = _normalized_datetime(df, col)
        valid = parsed.dropna()
        if valid.empty:
            rows.append([col, 0, "N/A", "N/A", 0])
            continue
        rows.append([
            col,
            int(valid.notna().sum()),
            valid.min().strftime("%d %b %Y"),
            valid.max().strftime("%d %b %Y"),
            int(valid.dt.normalize().nunique()),
        ])
    return rows


# ---------------------------------------------------------------------------
# DESCRIPTIVE STATISTICS
# ---------------------------------------------------------------------------

def _numeric_statistics(df, numeric):
    rows = []
    for col in numeric:
        s = pd.to_numeric(df[col], errors="coerce")
        valid = s.dropna()
        skew = valid.skew() if len(valid) >= 3 else np.nan
        variance = valid.var() if len(valid) >= 2 else np.nan
        cv = (valid.std() / valid.mean() * 100) if len(valid) >= 2 and valid.mean() != 0 else np.nan
        rows.append({
            "column": col, "count": int(s.count()), "mean": valid.mean() if len(valid) else np.nan,
            "median": valid.median() if len(valid) else np.nan, "std": valid.std() if len(valid) > 1 else np.nan,
            "variance": variance, "q1": valid.quantile(.25) if len(valid) else np.nan,
            "q3": valid.quantile(.75) if len(valid) else np.nan, "min": valid.min() if len(valid) else np.nan,
            "max": valid.max() if len(valid) else np.nan, "skew": skew, "cv": cv,
        })
    return rows


def _categorical_summary(df, categorical):
    rows = []
    for col in categorical:
        s = df[col].dropna().astype(str)
        vc = s.value_counts()
        if len(vc):
            max_freq = int(vc.iloc[0])
            modes = [str(k) for k, v in vc.items() if int(v) == max_freq]
            top = ", ".join(modes[:5]) + ("..." if len(modes) > 5 else "")
            share = max_freq / len(s) * 100
        else:
            modes, top, max_freq, share = [], "N/A", 0, 0
        rows.append({"column": col, "unique": int(s.nunique()), "top": top, "frequency": max_freq, "share": share, "modes": modes})
    return rows


def _is_identifier_column(df, col):
    """order_id-style columns are labels, not measures: never correlate,
    sum, or chart them as if they were quantities."""
    cache = _cache_for(df).setdefault("ident", {})
    if col in cache:
        return cache[col]
    name = re.sub(r"[^a-z0-9]+", "_", str(col).lower()).strip("_")
    ident = name in ("id", "index", "uuid", "row_number") or name.endswith("_id") or name.startswith("id_")
    if not ident:
        sr = df[col]
        ident = bool(pd.api.types.is_integer_dtype(sr) and len(sr) > 20 and sr.nunique() == sr.notna().sum())
    cache[col] = ident
    return ident


def _measure_columns(df, numeric):
    return [c for c in numeric if not _is_identifier_column(df, c)]


def _best_correlation(df, numeric):
    numeric = _measure_columns(df, numeric)
    if len(numeric) < 2:
        return None
    corr = df[numeric].apply(pd.to_numeric, errors="coerce").corr()
    best = None
    for i, c1 in enumerate(corr.columns):
        for c2 in corr.columns[i + 1:]:
            val = corr.loc[c1, c2]
            if pd.notna(val):
                candidate = (abs(float(val)), float(val), c1, c2)
                if best is None or candidate[0] > best[0]:
                    best = candidate
    return best


def _date_summary(df, date_col):
    parsed = _normalized_datetime(df, date_col).dropna()
    if not len(parsed):
        return None
    return {
        "min": parsed.min(), "max": parsed.max(),
        "days": int((parsed.max().normalize() - parsed.min().normalize()).days) + 1,
        "unique_days": int(parsed.dt.normalize().nunique()),
    }


def _primary_measure(df, numeric):
    measures = _measure_columns(df, numeric) or list(numeric)
    if not measures:
        return None
    # Priority order of hints (revenue beats price), not column order.
    hints = ("revenue", "sales", "amount", "income", "turnover", "profit", "cost", "value", "price", "score")
    for h in hints:
        for col in measures:
            if h in str(col).lower():
                return col
    return measures[0]


def _secondary_measure(df, numeric, primary):
    return next((c for c in _measure_columns(df, numeric) if c != primary), None)


def _group_metrics(df, category, measure):
    if not measure or category not in df.columns:
        return pd.DataFrame()
    work = df[[category, measure]].copy()
    work[measure] = pd.to_numeric(work[measure], errors="coerce")
    work = work.dropna(subset=[category, measure])
    if work.empty:
        return pd.DataFrame()
    return work.groupby(category, dropna=False)[measure].agg(total="sum", average="mean", count="count").sort_values("total", ascending=False)


def _time_aggregation(df, date_col, measure=None):
    dates = pd.to_datetime(df[date_col], errors="coerce")
    work = df.copy()
    work["__report_date"] = dates
    work = work.dropna(subset=["__report_date"])
    if work.empty:
        return pd.Series(dtype=float), "observed date"
    unique_dates = int(work["__report_date"].dt.normalize().nunique())
    span_days = int((work["__report_date"].max().normalize() - work["__report_date"].min().normalize()).days) + 1
    if span_days <= 31:
        idx = work["__report_date"].dt.normalize()
        label = "observed date"
    elif span_days <= 180:
        idx = work["__report_date"].dt.to_period("W").dt.start_time
        label = "week"
    else:
        idx = work["__report_date"].dt.to_period("M").dt.to_timestamp()
        label = "month"
    if measure:
        work["__measure"] = pd.to_numeric(work[measure], errors="coerce")
        series = work.groupby(idx)["__measure"].sum(min_count=1).dropna()
    else:
        series = work.groupby(idx).size()
    return series.sort_index(), label


# ---------------------------------------------------------------------------
# VERIFIED INSIGHTS / RECOMMENDATIONS
# ---------------------------------------------------------------------------

def _row_context(df, idx, numeric_primary=None):
    """Return human-readable context for a max/min observation when available."""
    parts = []
    numeric, categorical, date_cols = _cached_column_groups(df)
    if date_cols:
        d = pd.to_datetime(df.loc[idx, date_cols[0]], errors="coerce")
        if pd.notna(d):
            parts.append(d.strftime("%d %b %Y"))
    for col in categorical[:2]:
        value = df.loc[idx, col]
        if pd.notna(value):
            parts.append(f"{col}={value}")
    if parts:
        return " on " + ", ".join(parts)
    return ""


def _deterministic_insights(df):
    numeric, categorical, date_cols = _cached_column_groups(df)
    insights = []
    primary = _primary_measure(df, numeric)
    secondary = _secondary_measure(df, numeric, primary)

    if primary:
        s = pd.to_numeric(df[primary], errors="coerce").dropna()
        if len(s):
            max_value = s.max()
            min_value = s.min()

            # Report ties explicitly instead of showing only the first row
            # returned by idxmax()/idxmin(). Keep examples deterministic and
            # capped at two records so the report stays concise.
            max_mask = s.eq(max_value)
            min_mask = s.eq(min_value)
            max_count = int(max_mask.sum())
            min_count = int(min_mask.sum())

            max_examples = list(s.index[max_mask][:2])
            min_examples = list(s.index[min_mask][:2])

            def _example_contexts(indices):
                return [_row_context(df, idx, primary).lstrip(" on ")
                        for idx in indices]

            max_contexts = _example_contexts(max_examples)
            min_contexts = _example_contexts(min_examples)

            max_text = f"The maximum is {_fmt_number(max_value)} ({max_count:,} record(s))"
            if max_contexts:
                max_text += ", with examples: " + "; ".join(max_contexts) + "."

            min_text = f"The minimum is {_fmt_number(min_value)} ({min_count:,} record(s))"
            if min_contexts:
                min_text += ", with examples: " + "; ".join(min_contexts) + "."

            insights.append((
                "Highest and lowest",
                f"{primary} ranges from {_fmt_number(min_value)} to {_fmt_number(max_value)}. "
                f"{max_text} {min_text}"
            ))
            if len(s) >= 4:
                # Shape and outliers, not max/min: a max/min ratio is
                # meaningless (4,980x for an evenly spread column) and says
                # nothing about how the values are actually distributed.
                q1, q3 = s.quantile([.25, .75])
                iqr = float(q3 - q1)
                n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0
                skew = float(s.skew())
                shape = ("roughly symmetric" if abs(skew) < 0.5
                         else "right-skewed (a tail of high values)" if skew > 0
                         else "left-skewed (a tail of low values)")
                out_txt = (f"{n_out:,} value(s) ({n_out / len(s) * 100:.2f}%) fall outside the 1.5 x IQR fences."
                           if n_out else "No values fall outside the 1.5 x IQR fences, so there are no statistical outliers.")
                insights.append(("Value distribution", f"{primary} is {shape} (skewness {skew:.2f}): mean {_fmt_number(s.mean())}, median {_fmt_number(s.median())}. {out_txt}"))

    if secondary and primary:
        pair = df[[primary, secondary]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pair) >= 3:
            r = pair[primary].corr(pair[secondary])
            if pd.notna(r):
                direction = "positive" if r > 0 else "negative" if r < 0 else "near-zero"
                caveat = " This is an exploratory result because the sample contains fewer than 30 observations." if len(pair) < 30 else ""
                if abs(r) < 0.1:
                    # r this close to zero is the absence of a relationship,
                    # not a "weak positive" one.
                    insights.append(("Numeric relationship", f"{primary} and {secondary} show no meaningful linear association (Pearson r = {r:.3f}).{caveat}"))
                else:
                    strength = "very strong" if abs(r) >= .8 else "strong" if abs(r) >= .7 else "moderate" if abs(r) >= .4 else "weak"
                    insights.append(("Numeric relationship", f"{primary} and {secondary} show a {strength} {direction} linear association (Pearson r = {r:.3f}). Correlation does not establish causation.{caveat}"))

    for col in categorical[:4]:
        vc = df[col].dropna().astype(str).value_counts()
        if len(vc):
            max_freq = int(vc.iloc[0])
            modes = [str(k) for k, v in vc.items() if int(v) == max_freq]
            share = max_freq / len(df[col].dropna()) * 100
            if len(modes) == 1:
                body = f"{modes[0]} is the most frequent {col} value with {max_freq:,} record(s), representing {share:.1f}% of non-missing records."
            else:
                body = f"{', '.join(modes[:5])} are tied as the most frequent {col} values, each with {max_freq:,} record(s), representing {share:.1f}% of non-missing records."
            insights.append((f"Most frequent {col}", body))

        if primary:
            g = _group_metrics(df, col, primary)
            if len(g) >= 2:
                top = g.iloc[0]
                bottom = g.iloc[-1]
                share_top = top["total"] / g["total"].sum() * 100 if g["total"].sum() else np.nan
                insights.append((f"{col} performance", f"{g.index[0]} has the highest total {primary} at {_fmt_number(top['total'])} across {int(top['count']):,} record(s), contributing {_fmt_pct(share_top)} of grouped {primary}. The lowest group is {g.index[-1]} at {_fmt_number(bottom['total'])}."))

    if date_cols:
        ds = _date_summary(df, date_cols[0])
        if ds:
            insights.append(("Observed time period", f"The dataset spans {ds['min'].strftime('%d %b %Y')} to {ds['max'].strftime('%d %b %Y')} ({ds['days']:,} calendar day(s)) across {ds['unique_days']:,} unique date(s)."))

    return insights


def _key_insights(df, supplied_insights=None):
    # Deterministic results are intentionally preferred over LLM text.
    return _deterministic_insights(df)[:10]


def _group_performance_rows(df, categorical, primary):
    rows = []
    if not primary:
        return rows
    for col in categorical[:2]:
        g = _group_metrics(df, col, primary)
        if len(g) < 2:
            continue
        total_all = g["total"].sum()
        for idx, r in g.head(10).iterrows():
            rows.append({
                "dimension": col, "group": idx, "records": int(r["count"]),
                "total": r["total"], "average": r["average"],
                "share": (r["total"] / total_all * 100) if total_all else np.nan,
            })
    return rows


def _recommendations(df):
    numeric, categorical, date_cols = _cached_column_groups(df)
    recs = []
    primary = _primary_measure(df, numeric)

    missing = _cached_missing_total(df)
    dup = _cached_duplicate_total(df)
    if missing:
        recs.append("Review missing values in affected fields before making decisions from the impacted measures.")
    if dup:
        recs.append("Review duplicate rows before using totals, averages or record counts because duplicates can inflate grouped results.")

    if categorical and primary:
        g = _group_metrics(df, categorical[0], primary)
        if len(g) >= 2:
            top, bottom = g.iloc[0], g.iloc[-1]
            recs.append(f"Compare {categorical[0]} segments using both total and average {primary}; {g.index[0]} leads total {primary}, while {g.index[-1]} is lowest in the observed data.")

    if len(categorical) >= 2 and primary:
        g = _group_metrics(df, categorical[1], primary)
        if len(g) >= 2:
            recs.append(f"Monitor {categorical[1]} performance using total {primary} and transaction count rather than relying on frequency alone.")

    if date_cols and primary:
        series, granularity = _time_aggregation(df, date_cols[0], primary)
        if len(series) >= 2:
            recs.append(f"Track {primary} at {granularity} granularity to investigate peaks, declines and changes in contribution over time.")

    if len(numeric) >= 2:
        best = _best_correlation(df, numeric)
        if best:
            rec = f"Treat the {best[1]:.3f} correlation between {best[2]} and {best[3]} as exploratory and investigate the underlying drivers before making causal decisions."
            recs.append(rec)

    if not recs:
        recs.append("Use the descriptive statistics and EDA as a baseline for further analysis; expand the analysis only where the dataset supports it.")
    return recs[:6]


def _executive_summary(df, profile=None, insights=None, quick_stats=None):
    numeric, categorical, date_cols = _cached_column_groups(df)
    missing = _cached_missing_total(df)
    duplicates = _cached_duplicate_total(df)
    primary = _primary_measure(df, numeric)
    parts = [f"The dataset contains {len(df):,} records and {len(df.columns):,} columns, including {len(numeric)} numeric, {len(categorical)} categorical, and {len(date_cols)} date/time field(s)."]
    parts.append("No missing values were detected." if not missing else f"{missing:,} missing value(s) were detected across the dataset.")
    parts.append("No duplicate rows were detected." if not duplicates else f"{duplicates:,} duplicate row(s) were detected.")
    if primary:
        s = pd.to_numeric(df[primary], errors="coerce").dropna()
        if len(s):
            parts.append(f"The primary measure, {primary}, has a mean of {_fmt_number(s.mean())}, a median of {_fmt_number(s.median())}, and a range of {_fmt_number(s.min())} to {_fmt_number(s.max())}.")
    if categorical and primary:
        g = _group_metrics(df, categorical[0], primary)
        if len(g) >= 2:
            parts.append(f"By {categorical[0]}, {g.index[0]} has the highest total {primary} at {_fmt_number(g.iloc[0]['total'])}.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# CHART GENERATION
# ---------------------------------------------------------------------------

def _partial_final_period(df, date_col, granularity, series):
    """Return the last data date (as text) if the final aggregated period is only partly covered, else ''."""
    if granularity not in ("month", "week") or len(series) < 2:
        return ""
    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    last = pd.Timestamp(series.index[-1])
    if granularity == "month":
        start, end = last.to_period("M").start_time, last.to_period("M").end_time.normalize()
    else:
        start, end = last, last + pd.Timedelta(days=6)
    dmax = dates.max().normalize()
    if dates[dates >= start].dt.normalize().nunique() < 3:
        return ""
    coverage = ((dmax - start).days + 1) / float((end - start).days + 1)
    return dmax.strftime("%d %b %Y") if coverage < 0.9 else ""


def _chart_interpretation(title, df, context=None):
    numeric, categorical, date_cols = _cached_column_groups(df)
    primary = _primary_measure(df, numeric)
    if title.startswith("Distribution of"):
        col = title.replace("Distribution of ", "", 1)
        s = pd.to_numeric(df[col], errors="coerce").dropna() if col in df.columns else pd.Series(dtype=float)
        if len(s):
            median = s.median()
            skew = s.skew() if len(s) >= 3 else np.nan
            text = f"Values span {_fmt_number(s.min())} to {_fmt_number(s.max())}; mean {_fmt_number(s.mean())} and median {_fmt_number(median)}."
            if pd.notna(skew):
                text += f" Skewness is {skew:.2f}."
            return text
    if title.startswith("Category Frequency") and context:
        vc = df[context].dropna().astype(str).value_counts()
        if len(vc):
            maxf = int(vc.iloc[0]); modes = [str(k) for k, v in vc.items() if int(v) == maxf]
            if len(modes) > 1:
                return f"{', '.join(modes)} are tied for the highest record count at {maxf:,} each."
            return f"{modes[0]} has the highest record count at {maxf:,}, representing {maxf / len(df[context].dropna()) * 100:.1f}% of non-missing records."
    if title.startswith("Total ") and " by " in title and context:
        measure = title.replace("Total ", "", 1).split(" by ", 1)[0]
        g = _group_metrics(df, context, measure)
        if len(g):
            return f"{g.index[0]} leads total {measure} at {_fmt_number(g.iloc[0]['total'])}; {g.index[-1]} is lowest at {_fmt_number(g.iloc[-1]['total'])}."
    if title.startswith("Time Trend") and date_cols:
        series, granularity = _time_aggregation(df, date_cols[0], primary)
        if len(series):
            note = ""
            partial = _partial_final_period(df, date_cols[0], granularity, series)
            if partial and len(series) > 2:
                note = (f" The final period ({series.index[-1].strftime('%b %Y') if granularity == 'month' else series.index[-1].strftime('%d %b %Y')}) "
                        f"only has data up to {partial}, so its total is incomplete and is not comparable with full periods; it is excluded from the highest/lowest comparison.")
                series = series.iloc[:-1]
            hi, lo = series.idxmax(), series.idxmin()
            fmt = "%d %b %Y" if granularity == "observed date" else "%d %b %Y" if granularity == "week" else "%b %Y"
            label = f"{hi.strftime(fmt)}"; low_label = f"{lo.strftime(fmt)}"
            metric = primary if primary else "record count"
            return f"Using {granularity} aggregation, the highest total {metric} is {_fmt_number(series.max())} at {label}; the lowest is {_fmt_number(series.min())} at {low_label}.{note}"
    if title.startswith("Relationship") and " vs " in title:
        x, y = title.replace("Relationship - ", "", 1).split(" vs ", 1)
        if x not in df.columns or y not in df.columns:
            return "The scatter plot shows the row-level relationship between the two variables."
        pair = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pair) >= 2:
            r = pair[x].corr(pair[y])
            return f"The scatter plot shows the row-level relationship between {x} and {y}. Pearson r = {r:.3f}; with {len(pair)} observations, this is exploratory and does not establish causation."
    if title == "Correlation Matrix":
        best = _best_correlation(df, numeric)
        if best:
            return f"The strongest absolute Pearson correlation is {best[1]:.3f} between {best[2]} and {best[3]}. Correlation measures linear association, not causation."
    return "The visualization summarizes the corresponding dataset pattern; interpret it together with the supporting statistics."


def _make_charts(df):
    """Return list of (section, title, png_bytes, interpretation).

    Drawn with backend.chart_renderer (Pillow only) - no matplotlib, so the
    report's charts cannot disappear because a plotting library is missing.
    Each chart is isolated: one failure never removes the others.
    """
    from backend import chart_renderer as cr

    charts = []
    numeric, categorical, date_cols = _cached_column_groups(df)
    primary = _primary_measure(df, numeric)

    def add(section, title, context, draw):
        try:
            data = draw()
        except Exception:
            logger.exception("[REPORT-CHART] could not draw %r", title)
            return
        if data:
            charts.append((section, title, data, _chart_interpretation(title, df, context)))

    measures = _measure_columns(df, numeric)
    if primary in measures:
        measures = [primary] + [c for c in measures if c != primary]

    # 1. Numeric distributions (full data: counts, mean and median must match the tables).
    for col in measures[:4]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 2:
            continue
        add("6.1 Univariate Analysis", f"Distribution of {col}", None,
            lambda s=s, col=col: cr.render_hist(f"Distribution of {col}", s.to_numpy(dtype=float), str(col)))

    # 2. Categorical frequency.
    for col in categorical[:3]:
        vc = df[col].dropna().astype(str).value_counts().head(10)
        if len(vc) < 2:
            continue
        add("6.2 Categorical Analysis", f"Category Frequency - {col}", col,
            lambda vc=vc, col=col: cr.render_bar_h(f"Category Frequency - {col}", vc.index, vc.values,
                                                   xlabel="Record count", ylabel=str(col)))

    # 3. Category performance by primary measure.
    if primary:
        for col in categorical[:2]:
            g = _group_metrics(df, col, primary)
            if len(g) < 2:
                continue
            show = g.head(10)
            add("6.2 Categorical Analysis", f"Total {primary} by {col}", col,
                lambda show=show, col=col: cr.render_bar_h(f"Total {primary} by {col}", show.index,
                                                           show["total"].values,
                                                           xlabel=f"Total {primary}", ylabel=str(col)))

    # 4. Time trend.
    if date_cols:
        dcol = date_cols[0]
        series, granularity = _time_aggregation(df, dcol, primary)
        if len(series) >= 2:
            metric_label = f"Total {primary}" if primary else "Record count"
            title = f"Time Trend - {metric_label} ({granularity})"
            fmt = "%b %Y" if granularity == "month" else "%d %b"
            labels = [pd.Timestamp(i).strftime(fmt) for i in series.index]
            add("6.3 Time-Based Analysis", title, None,
                lambda: cr.render_line(title, labels, {metric_label: series.values},
                                       xlabel=str(dcol), ylabel=metric_label))

    # 5. Relationship + correlation matrix.
    if len(measures) >= 2:
        xcol, ycol = measures[:2]
        pair = df[[xcol, ycol]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pair) >= 2:
            plot_pair = pair.sample(10000, random_state=42) if len(pair) > 10000 else pair
            title = f"Relationship - {xcol} vs {ycol}"
            add("6.4 Relationship Analysis", title, None,
                lambda: cr.render_scatter(title, plot_pair[xcol].values, plot_pair[ycol].values,
                                          xlabel=str(xcol), ylabel=str(ycol)))
        corr = df[measures[:10]].apply(pd.to_numeric, errors="coerce").corr()
        if not corr.empty:
            add("6.4 Relationship Analysis", "Correlation Matrix", None,
                lambda: cr.render_heatmap("Correlation Matrix", corr.columns, corr.values))

    return charts


# ---------------------------------------------------------------------------
# CHAT SUPPORT
# ---------------------------------------------------------------------------

def _chat_messages(chats):
    rows = []
    if not chats:
        return rows
    # Support both {chat_id: {messages: [...]}} and a direct message list.
    if isinstance(chats, list):
        source = {"Chat": {"messages": chats}}
    elif isinstance(chats, dict):
        source = chats
    else:
        return rows
    for _, chat in source.items():
        if not isinstance(chat, dict):
            continue
        for msg in chat.get("messages", []) or []:
            if not isinstance(msg, dict):
                continue
            rows.append({
                "chat_title": chat.get("title", "Chat"),
                "question": msg.get("question", ""),
                "answer": msg.get("answer", msg.get("content", "")),
                "type": msg.get("type", msg.get("answer_type", "text")),
                "code": msg.get("code"), "code_result": msg.get("code_result"),
                "sql": msg.get("sql"), "sql_result": msg.get("sql_result"),
                "chart_config": msg.get("chart_config"), "chart_bytes": msg.get("chart_bytes"),
                "chart_title": msg.get("chart_title", "Chart"),
            })
    return rows


def _infer_chat_chart_config(question, df):
    """Infer a semantically correct chart config for saved chat messages.

    Prefer the same deterministic chart specification used by the live chat
    query engine. This keeps report reconstruction aligned with the actual
    user request for top-N, date-filtered trends, and multi-dimensional charts.
    """
    # First reuse the authoritative deterministic chart parser.  It already
    # understands semantic constructs such as:
    #   - top N products by revenue
    #   - monthly revenue for 2024
    #   - revenue by region and category
    try:
        from backend.query_engine import _chart_spec
        spec = _chart_spec(str(question or ""), df)
        if isinstance(spec, dict):
            chart = spec.get("chart")
            if isinstance(chart, dict):
                cfg = dict(chart)
                ctype = str(cfg.get("type", "bar")).lower()
                if ctype == "hist":
                    ctype = "histogram"
                cfg["type"] = ctype
                if cfg.get("x"):
                    cfg["x"] = str(cfg["x"])
                if cfg.get("y"):
                    cfg["y"] = str(cfg["y"])
                return cfg
    except Exception:
        # Fall through to the self-contained inference below.
        pass

    # Self-contained high-confidence patterns. These deliberately do not
    # depend on query_engine being imported from the same package instance,
    # which matters when reports are generated from a copied/deployed app.
    q = str(question or "").strip().lower()
    metric = "revenue" if "revenue" in q else None
    if metric and re.search(r"\btop\s+\d+\s+products?\b", q) and "bar" in q:
        m = re.search(r"\btop\s+(\d+)\s+products?\b", q)
        if m and "product" in df.columns and metric in df.columns:
            return {"type":"bar", "x":"product", "y":"revenue", "aggregation":"sum",
                    "top_n":int(m.group(1)), "title":f"Top {int(m.group(1))} product by revenue"}
    if metric and re.search(r"\bmonthly\s+revenue\b", q) and "line" in q:
        date_col = next((c for c in df.columns if "date" in str(c).lower() or "time" in str(c).lower()), None)
        if date_col and metric in df.columns:
            years = re.findall(r"\b(20\d{2})\b", q)
            return {"type":"line", "x":str(date_col), "y":"revenue", "aggregation":"sum",
                    "date_granularity":"month", "date_filter_year":int(years[0]) if len(years)==1 else None,
                    "title":"revenue by month"}
    if metric and re.search(r"\bby\s+region\s+and\s+category\b", q) and "grouped" in q and "bar" in q:
        if "region" in df.columns and "category" in df.columns and metric in df.columns:
            return {"type":"bar", "x":"region", "y":"revenue", "group_by":"category",
                    "aggregation":"sum", "title":"Sum revenue by region and category"}

    numeric, categorical, date_cols = _cached_column_groups(df)
    chart_words = ("chart", "plot", "histogram", "bar", "line", "scatter", "pie", "visual", "distribution")
    if not any(w in q for w in chart_words):
        return None

    def find_col(pool, tokens=None):
        tokens = tokens or []
        for col in pool:
            name = str(col).lower()
            if name in q:
                return col
        for col in pool:
            if any(t in str(col).lower() for t in tokens):
                return col
        return None

    if "histogram" in q or "distribution" in q:
        col = find_col(numeric) or (numeric[0] if numeric else None)
        return {"type": "histogram", "x": col, "title": f"Distribution of {col}"} if col else None

    ctype = "scatter" if "scatter" in q else "line" if any(k in q for k in ("line", "trend")) else "pie" if "pie" in q else "bar"
    aggregation = "mean" if any(k in q for k in ("average", "avg", "mean")) else "count" if "count" in q else "sum"
    if ctype == "scatter":
        if len(numeric) < 2:
            return None
        x = find_col(numeric) or numeric[0]
        y = next((c for c in numeric if c != x), None)
        return {"type": "scatter", "x": x, "y": y, "title": f"Relationship - {x} vs {y}"}
    x = find_col(date_cols) or find_col(categorical) or (categorical[0] if categorical else None)
    y = find_col(numeric) or (numeric[0] if numeric else None)
    if not x or not y:
        return None
    return {"type": ctype, "x": x, "y": y, "aggregation": aggregation, "title": f"{aggregation.title()} {y} by {x}"}


def _chart_label(v):
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    return str(v)


def _render_chat_chart_from_config(df, msg):
    """Render a chat chart from a chart_config dict using chart_renderer
    (Pillow only). Returns (png_bytes, title) or (None, None)."""
    cfg = msg.get("chart_config") if isinstance(msg, dict) else None
    if not isinstance(cfg, dict) or not cfg:
        return None, None
    from backend import chart_renderer as cr

    ctype = str(cfg.get("type", "bar")).lower()
    if ctype == "hist":
        ctype = "histogram"
    x, y = cfg.get("x"), cfg.get("y")
    title = str(cfg.get("title") or "Chart")
    group_by = cfg.get("group_by")
    top_n = cfg.get("top_n")
    grain = cfg.get("date_granularity")
    year = cfg.get("date_filter_year")
    agg = str(cfg.get("aggregation", "sum")).lower()
    agg_func = {"mean": "mean", "average": "mean", "avg": "mean", "count": "count",
                "min": "min", "max": "max"}.get(agg, "sum")
    y_label = f"{agg_func.title()} {y}" if agg_func != "sum" else f"Total {y}"

    try:
        if ctype == "histogram":
            col = x or y
            if col not in df.columns:
                return None, None
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) > 200000:
                s = s.sample(200000, random_state=42)
            return cr.render_hist(title, s.values, str(col)), title

        if ctype == "scatter":
            if x not in df.columns or y not in df.columns:
                return None, None
            pair = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
            return cr.render_scatter(title, pair[x].values, pair[y].values, str(x), str(y)), title

        if ctype == "box":
            col = y or x
            if col not in df.columns:
                return None, None
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            return cr.render_box(title, {str(col): s.values}, str(col)), title

        if ctype not in ("bar", "line", "pie"):
            return None, None
        if not x or not y or x not in df.columns or y not in df.columns:
            return None, None

        cols = [x, y]
        if group_by and group_by in df.columns and group_by not in cols:
            cols.append(group_by)
        work = df[cols].copy()
        if year is not None:
            dates = pd.to_datetime(work[x], errors="coerce")
            work = work.loc[dates.dt.year == int(year)].copy()
        work[y] = pd.to_numeric(work[y], errors="coerce")
        work = work.dropna(subset=[x, y])
        if work.empty:
            return None, None

        chronological = bool(grain) or pd.api.types.is_datetime64_any_dtype(work[x])
        if grain:
            dates = pd.to_datetime(work[x], errors="coerce")
            work["__x__"] = (dates.dt.year.astype("Int64").astype(str) if grain == "year"
                             else dates.dt.to_period("M").astype(str))
            xp = "__x__"
        else:
            xp = x

        if group_by and group_by in work.columns:
            g = work.groupby([xp, group_by], dropna=False)[y].agg(agg_func).reset_index(name=y)
            pivot = g.pivot(index=xp, columns=group_by, values=y)
            if not chronological and len(pivot) > 12:
                pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).head(12).index]
            pivot = pivot.sort_index() if chronological else pivot
            if pivot.shape[1] > 8:
                pivot = pivot[pivot.sum().sort_values(ascending=False).head(8).index]
            cats = [_chart_label(i) for i in pivot.index]
            series = {str(c): pivot[c].values for c in pivot.columns}
            if ctype == "line":
                return cr.render_line(title, cats, series, str(x), y_label), title
            series = {k: np.nan_to_num(v) for k, v in series.items()}
            return cr.render_grouped_bar(title, cats, series, str(x), y_label), title

        s = work.groupby(xp, dropna=False)[y].agg(agg_func)
        if s.empty:
            return None, None
        if ctype == "pie":
            s = s.sort_values(ascending=False)
            return cr.render_pie(title, [_chart_label(i) for i in s.index], s.values), title
        if chronological or ctype == "line":
            s = s.sort_index()
            labels = [_chart_label(i) for i in s.index]
            if ctype == "line":
                return cr.render_line(title, labels, {y_label: s.values}, str(x), y_label), title
            return cr.render_grouped_bar(title, labels, {y_label: s.values}, str(x), y_label), title
        s = s.sort_values(ascending=False)
        limit = int(top_n) if top_n else 15
        if len(s) > limit:
            s = s.head(limit)
            if not top_n:
                title = f"{title} (top {limit})"
        return cr.render_bar_h(title, [_chart_label(i) for i in s.index], s.values,
                               xlabel=y_label, ylabel=str(x)), title
    except Exception:
        logger.exception("[REPORT-CHART] chat chart render failed: %r cfg=%r", title, cfg)
        return None, None


def render_chart_config_png(df, chart_config):
    """Public: PNG bytes for a chart_config using the Pillow renderer (no
    kaleido / browser needed). Returns None if it cannot be drawn."""
    png, _title = _render_chat_chart_from_config(df, {"chart_config": chart_config})
    return png


def render_forecast_png(forecast_result):
    """PNG (history + forecast + band) for a forecast_dataframe() result."""
    from backend import chart_renderer as cr
    fc = (forecast_result or {}).get("forecast") or {}
    values = fc.get("values") or []
    if not values:
        return None
    hist = fc.get("historical") or []
    metric = fc.get("metric", "value")
    def lab(p):
        return pd.Timestamp(p).strftime("%b %Y")
    try:
        return cr.render_forecast(
            f"{metric} - forecast", [lab(h["period"]) for h in hist], [h["value"] for h in hist],
            [lab(v["period"]) for v in values], [v["value"] for v in values],
            [x["value"] for x in fc.get("lower", [])] or None,
            [x["value"] for x in fc.get("upper", [])] or None, ylabel=str(metric))
    except Exception:
        logger.exception("forecast chart render failed")
        return None


def _looks_like_chart_question(question):
    q = str(question or "").strip().lower()
    if not q:
        return False
    return bool(re.search(r"\b(chart|plot|graph|visuali[sz]e|bar chart|line chart|pie chart|scatter plot|histogram|trend)\b", q))


def reconstruct_chat_chart(df, question):
    """Public deterministic chart reconstruction used at chat-save time.

    This intentionally works from the user's original question and the live
    dataframe, so a missing/invalid LLM chart_config cannot prevent the report
    from receiving the chart image.
    """
    msg = {"question": str(question or ""), "type": "chart"}
    chart, title = _resolve_chat_chart(df, msg)
    if chart:
        return chart, title
    return None, None


def _resolve_chat_chart(df, msg):
    """Resolve the chart image for a saved chat message.

    Order: (1) bytes already saved on the message, (2) the saved
    chart_config, (3) a config inferred from the question text. The
    report never depends on Plotly/Kaleido/matplotlib: everything is drawn
    by backend.chart_renderer (Pillow only).
    """
    if not isinstance(msg, dict):
        return None, None
    question = str(msg.get("question", "") or "").strip()
    logger.debug("[CHART-REPORT-FIX] resolving chart (%d chars)", len(question))

    if msg.get("chart_bytes"):
        return msg.get("chart_bytes"), msg.get("chart_title", "Chart")

    cfg = msg.get("chart_config")
    if isinstance(cfg, dict) and cfg:
        chart, title = _render_chat_chart_from_config(df, {"chart_config": cfg})
        if chart:
            return chart, title

    if question:
        try:
            inferred = _infer_chat_chart_config(question, df)
        except Exception:
            logger.exception("[CHART-REPORT-FIX] inference failed (%d chars)", len(question))
            inferred = None
        logger.debug("[CHART-REPORT-FIX] inferred config: %r", inferred)
        if isinstance(inferred, dict):
            chart, title = _render_chat_chart_from_config(df, {"chart_config": inferred})
            if chart:
                return chart, title

    logger.warning("[CHART-REPORT-FIX] no chart could be built (%d chars)", len(question))
    return None, None


# ---------------------------------------------------------------------------
# SHARED REPORT ANALYSIS
# ---------------------------------------------------------------------------

def prepare_report_analysis(df, profile=None, insights=None, quick_stats=None, chats=None,
                            include_preview=True, include_eda=True, include_charts=True,
                            include_chat=True, preview_rows=5):
    """Compute report facts once so Markdown and PDF reuse identical results.

    This is deliberately dataframe-backed: no report section independently
    recalculates the same expensive statistics. Charts are generated at most
    once and only when requested.
    """
    numeric, categorical, date_cols = _cached_column_groups(df)
    columns = _profile_columns(df, profile)
    stats = _numeric_statistics(df, numeric)
    cat_stats = _categorical_summary(df, categorical)
    quality = _quality_findings(df)
    actions = _quality_actions(df)
    outliers = _outlier_profile(df)
    quality_score = _quality_score(df)
    methodology = _methodology_items()
    date_profile = _date_profile_rows(df, date_cols)
    key_items = _key_insights(df, insights)
    recommendations = _recommendations(df)
    chats_flat = _chat_messages(chats) if include_chat else []
    primary = _primary_measure(df, numeric)

    # These values are reused by both report formats.
    summary = _executive_summary(df, profile, insights, quick_stats)
    best_corr = _best_correlation(df, numeric) if len(numeric) >= 2 else None
    date_summary = _date_summary(df, date_cols[0]) if date_cols else None
    time_series = None
    time_granularity = None
    group_rows = []
    categorical_top = {}

    if include_eda:
        if date_cols:
            time_series, time_granularity = _time_aggregation(df, date_cols[0], primary)
        for col in categorical:
            categorical_top[col] = df[col].dropna().astype(str).value_counts().head(10)
        group_rows = _group_performance_rows(df, categorical, primary)

    chart_data = _make_charts(df) if include_charts else []
    preview = df.head(max(0, int(preview_rows))) if include_preview and preview_rows > 0 else pd.DataFrame()

    return {
        "numeric": numeric, "categorical": categorical, "date_cols": date_cols,
        "columns": columns, "stats": stats, "cat_stats": cat_stats,
        "quality": quality, "actions": actions, "outliers": outliers,
        "quality_score": quality_score, "methodology": methodology,
        "date_profile": date_profile,
        "key_items": key_items, "recommendations": recommendations,
        "chats_flat": chats_flat, "primary": primary, "summary": summary,
        "best_corr": best_corr, "date_summary": date_summary,
        "time_series": time_series, "time_granularity": time_granularity,
        "group_rows": group_rows, "categorical_top": categorical_top,
        "chart_data": chart_data, "preview": preview,
        "include_preview": bool(include_preview), "include_eda": bool(include_eda),
        "include_charts": bool(include_charts), "include_chat": bool(include_chat),
        "missing_total": _cached_missing_total(df), "duplicate_total": _cached_duplicate_total(df),
    }


def _log_chat_summary(chats_flat):
    """Diagnostic: shows exactly which saved chat messages reach the report."""
    logger.info("[CHART-REPORT-FIX] report received %d chat message(s)", len(chats_flat))
    for m in chats_flat:
        logger.info(
            "[CHART-REPORT-FIX]   q=%r type=%r has_config=%s has_bytes=%s looks_like_chart=%s",
            "<redacted>", m.get("type"), bool(m.get("chart_config")),
            bool(m.get("chart_bytes")), _looks_like_chart_question(m.get("question", "")),
        )


def _get_analysis(df, profile, insights, quick_stats, chats, report_data=None, **options):
    if isinstance(report_data, dict) and report_data.get("stats") is not None:
        # Chat history is mutable Streamlit session state.  Never trust a
        # potentially older report-analysis snapshot for chat content.
        # Re-read it at report-build time so charts asked after an earlier
        # analysis pass cannot disappear from the final report.
        out = dict(report_data)
        out["chats_flat"] = _chat_messages(chats) if options.get("include_chat", True) else []
        _log_chat_summary(out["chats_flat"])
        return out
    return prepare_report_analysis(df, profile, insights, quick_stats, chats, **options)


# ---------------------------------------------------------------------------
# MARKDOWN REPORT
# ---------------------------------------------------------------------------

def generate_markdown_report(df, profile: dict, insights: list, quick_stats: dict, chats: dict, report_data=None, **options) -> str:
    generated = datetime.now().strftime("%B %d, %Y at %H:%M")
    a = _get_analysis(df, profile, insights, quick_stats, chats, report_data, **options)
    numeric, categorical, date_cols = a["numeric"], a["categorical"], a["date_cols"]
    columns, stats, cat_stats = a["columns"], a["stats"], a["cat_stats"]
    quality, actions, outliers = a["quality"], a["actions"], a["outliers"]
    chart_data, key_items, recommendations = a["chart_data"], a["key_items"], a["recommendations"]
    chats_flat, name = a["chats_flat"], _dataset_name(df, profile)
    summary = a["summary"]
    best_corr, date_summary = a["best_corr"], a["date_summary"]
    time_series, time_granularity = a["time_series"], a["time_granularity"]

    lines = [
        "# DATA ANALYSIS REPORT", "", f"**Dataset:** {name}", f"**Generated:** {generated}", "",
        "## Executive Summary", "", summary, "",
        "## 1. Analysis Methodology", "", *[f"- {m}" for m in a["methodology"]], "",
        "## 2. Dataset Overview", "", "| Metric | Value |", "|---|---:|",
        f"| Records | {len(df):,} |", f"| Features | {len(df.columns):,} |",
        f"| Numeric columns | {len(numeric):,} |", f"| Categorical columns | {len(categorical):,} |",
        f"| Date/time columns | {len(date_cols):,} |", f"| Missing values | {_cached_missing_total(df):,} |",
        f"| Duplicate rows | {_cached_duplicate_total(df):,} |", "", "### Data Preview", "",
    ]
    preview = a["preview"]
    if a["include_preview"] and not preview.empty:
        lines.append("| " + " | ".join(map(str, preview.columns)) + " |")
        lines.append("| " + " | ".join(["---"] * len(preview.columns)) + " |")
        for _, row in preview.iterrows():
            lines.append("| " + " | ".join(_safe_text(v).replace("|", "\\|") for v in row.tolist()) + " |")

    lines += ["", "## 3. Column / Feature Description", "", "| Column | Detected Type | Original dtype | Description | Unique | Missing | Missing % |", "|---|---|---|---|---:|---:|---:|"]
    for c in columns:
        lines.append(f"| {_safe_text(c['name'])} | {_safe_text(c['detected_type'])} | {_safe_text(c['dtype'])} | {_safe_text(c['description'])} | {c['unique_count']:,} | {c['missing_count']:,} | {_fmt_pct(c['missing_pct'],2)} |")

    lines += ["", "## 4. Data Quality & Cleaning", "", "### Quality Score", ""]
    qs = a["quality_score"]
    lines += [f"**Overall Data Quality Score:** **{qs['score']:.1f}/100 ({qs['band']})**",
              "", "The score is a transparent screening indicator and should be interpreted alongside the detailed findings.", "",
              "### Quality Findings", ""]
    lines += [f"- {x}" for x in quality]
    lines += ["", "### Cleaning / Preparation Actions", ""] + [f"- {x}" for x in actions]
    lines += ["", "### Outlier Assessment", "", "| Column | Valid N | Q1 | Q3 | IQR | Potential Outliers | Status |", "|---|---:|---:|---:|---:|---:|---|"]
    for r in outliers:
        lines.append(f"| {_safe_text(r['column'])} | {r['n']:,} | {_fmt_number(r['q1'])} | {_fmt_number(r['q3'])} | {_fmt_number(r['iqr'])} | {r['count'] if r['count'] is not None else 'N/A'} | {_safe_text(r['status'])} |")

    lines += ["", "## 5. Descriptive Statistics", "", "### Numeric Statistics", "", "| Column | Count | Mean | Median | Std Dev | Variance | Q1 | Q3 | Min | Max | Skew | CV % |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in stats:
        lines.append("| " + " | ".join([_safe_text(r["column"]), f"{r['count']:,}", _fmt_number(r['mean']), _fmt_number(r['median']), _fmt_number(r['std']), _fmt_number(r['variance']), _fmt_number(r['q1']), _fmt_number(r['q3']), _fmt_number(r['min']), _fmt_number(r['max']), _fmt_number(r['skew']), _fmt_number(r['cv'])]) + " |")
    if not stats: lines.append("No numeric columns were detected.")

    lines += ["", "### Categorical Statistics", ""]
    if cat_stats:
        lines += ["| Column | Unique | Most Frequent | Frequency | Share |", "|---|---:|---|---:|---:|"]
        for r in cat_stats: lines.append(f"| {_safe_text(r['column'])} | {r['unique']:,} | {_safe_text(r['top'])} | {r['frequency']:,} | {_fmt_pct(r['share'])} |")
    else: lines.append("No categorical columns were detected.")

    if a["include_eda"]:
        lines += ["", "## 6. Exploratory Data Analysis", "", "### 6.1 Univariate Analysis", ""]
        for r in stats:
            lines.append(f"- **{_safe_text(r['column'])}:** mean {_fmt_number(r['mean'])}, median {_fmt_number(r['median'])}, range {_fmt_number(r['min'])} to {_fmt_number(r['max'])}, skewness {_fmt_number(r['skew'])}.")
        if not stats: lines.append("No numeric variables are available for univariate analysis.")

        lines += ["", "### 6.2 Categorical Analysis", ""]
        for col, vc in a["categorical_top"].items():
            if len(vc): lines.append(f"- **{_safe_text(col)}:** " + ", ".join(f"{_safe_text(k)} ({int(v):,})" for k,v in vc.items()))
        if not categorical: lines.append("No categorical variables are available for categorical analysis.")

        lines += ["", "### 6.3 Time-Based Analysis", ""]
        if date_cols:
            lines.append(f"- **Date/time field:** {_safe_text(date_cols[0])}")
            if a["date_summary"]:
                ds=a["date_summary"]; lines += [f"- **Observed period:** {ds['min'].strftime('%Y-%m-%d')} to {ds['max'].strftime('%Y-%m-%d')}", f"- **Unique dates:** {ds['unique_days']:,}", f"- **Chart aggregation:** {a['time_granularity']}"]
        else: lines.append("No sufficiently reliable date/time column was detected.")
        if a["date_profile"]:
            lines += ["", "| Date/Time Column | Valid Values | Earliest | Latest | Unique Dates |",
                      "|---|---:|---|---|---:|"]
            for r in a["date_profile"]:
                lines.append("| " + " | ".join(_safe_text(v) for v in r) + " |")

        lines += ["", "### 6.4 Relationship Analysis", ""]
        if a["best_corr"]:
            best=a["best_corr"]; lines += [f"- **Strongest absolute Pearson correlation:** {_safe_text(best[2])} vs {_safe_text(best[3])} = **{best[1]:.3f}**.", "- Correlation measures linear association and does not establish causation."]
        else: lines.append("At least two usable numeric columns are required for relationship analysis.")

        if a["include_charts"]:
            lines += ["", "### Visual Analysis", ""]
            for _, title, data, interpretation in chart_data:
                b64 = base64.b64encode(data).decode("ascii")
                lines += [f"#### {title}", "", f"![{title}](data:image/png;base64,{b64})", "", f"**Interpretation:** {interpretation}", ""]

    lines += ["## 7. Key Insights", ""]
    for title, body in key_items: lines += [f"### {title}", body, ""]
    lines += ["## 8. Conclusion", "", summary, "", "## 9. Limitations & Scope", "", "- Findings are descriptive and exploratory; they do not imply causation.", "- Small datasets can produce unstable distributions and correlations.", "- Outlier detection uses the IQR rule and is a screening method, not a definitive data-quality judgment.", "", "## 10. Recommendations", ""]
    lines += [f"{i}. {r}" for i, r in enumerate(recommendations, 1)]
    lines += ["", "## 11. Chat Analysis", ""] if a["include_chat"] else []
    if a["include_chat"] and chats_flat:
        for i, msg in enumerate(chats_flat, 1):
            lines += [f"### Question {i}", f"**Question:** {_safe_text(msg['question'])}", "", f"**Answer:** {_safe_text(msg['answer'])}", ""]
            if msg.get("code_result") is not None: lines += ["**Result:**", "", "```text", _safe_text(msg["code_result"]), "```", ""]
            if msg.get("sql_result") is not None: lines += ["**SQL Result:**", "", "```text", _safe_text(msg["sql_result"]), "```", ""]
            chart = None; title = None
            # A saved chart can have lost its type/config during older chat
            # state migrations.  The original question remains authoritative.
            if (str(msg.get("type", "")).lower() == "chart" or msg.get("chart_config")
                    or msg.get("chart_bytes") or _looks_like_chart_question(msg.get("question", ""))):
                chart, title = _resolve_chat_chart(df, msg)
            if chart:
                b64 = base64.b64encode(chart).decode("ascii"); lines += [f"**{_safe_text(title or 'Chart')}**", "", f"![{_safe_text(title or 'Chart')}](data:image/png;base64,{b64})", ""]
    elif a["include_chat"]: lines.append("No saved chat questions are available.")
    lines += ["", "---", "", "*Report generated by Axiomrow - AI-powered data analysis*"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PDF REPORT
# ---------------------------------------------------------------------------

def generate_pdf_report(df, profile: dict, insights: list, quick_stats: dict, chats: dict, report_data=None, **options) -> bytes:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos, Corner, MethodReturnValue

    PRIMARY = (37, 99, 235)
    PRIMARY_DARK = (29, 78, 216)
    DARK = (15, 23, 42)
    MUTED = (71, 85, 105)
    LIGHT = (241, 245, 249)
    LIGHT_BLUE = (239, 246, 255)
    BORDER = (226, 232, 240)
    ZEBRA = (248, 250, 252)
    WHITE = (255, 255, 255)

    # WHY bundle these rather than rely on system fonts: this runs
    # wherever the app is deployed (e.g. Streamlit Cloud), which has no
    # guarantee of the same fonts this was developed with. Shipping the
    # TTFs in backend/assets/fonts/ makes the report look identical
    # everywhere. If the files are ever missing, fall back to the core
    # Helvetica font rather than crashing report generation entirely —
    # same "degrade, don't crash" principle used throughout this file.
    _FONT_DIR = os.path.join(os.path.dirname(__file__), "assets", "fonts")
    _FONT_FILES = {
        "Poppins": "Poppins-Regular.ttf",
        "Poppins-Bold": "Poppins-Bold.ttf",
        "Poppins-Italic": "Poppins-Italic.ttf",
        "Poppins-Medium": "Poppins-Medium.ttf",
        "Poppins-Light": "Poppins-Light.ttf",
    }
    _fonts_available = all(
        os.path.isfile(os.path.join(_FONT_DIR, fname)) for fname in _FONT_FILES.values()
    )
    if _fonts_available:
        FONT, FONT_MEDIUM, FONT_LIGHT = "Poppins", "Poppins-Medium", "Poppins-Light"
    else:
        FONT = FONT_MEDIUM = FONT_LIGHT = "Helvetica"
    # Picked once, used by every drawing helper below — avoids threading
    # a "which font mode" flag through every single call site.
    _clean_fn = _clean_unicode if _fonts_available else _clean

    a = _get_analysis(df, profile, insights, quick_stats, chats, report_data, **options)
    numeric, categorical, date_cols = a["numeric"], a["categorical"], a["date_cols"]
    columns, stats, cat_stats = a["columns"], a["stats"], a["cat_stats"]
    quality, actions, outliers = a["quality"], a["actions"], a["outliers"]
    chart_data, key_items, recommendations = a["chart_data"], a["key_items"], a["recommendations"]
    chats_flat, dataset_name = a["chats_flat"], _dataset_name(df, profile)
    summary = a["summary"]

    class ReportPDF(FPDF):
        def footer(self):
            self.set_y(-12)
            self.set_draw_color(*BORDER)
            self.line(self.l_margin, self.y, self.w - self.r_margin, self.y)
            self.set_y(-9)
            self.set_font(FONT, "", 7.5)
            self.set_text_color(*MUTED)
            self.cell(0, 5, f"Axiomrow | Data Analysis Report | Page {self.page_no()}", align="C")

    pdf = ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=17)

    if _fonts_available:
        try:
            pdf.add_font("Poppins", "", os.path.join(_FONT_DIR, "Poppins-Regular.ttf"))
            pdf.add_font("Poppins", "B", os.path.join(_FONT_DIR, "Poppins-Bold.ttf"))
            pdf.add_font("Poppins", "I", os.path.join(_FONT_DIR, "Poppins-Italic.ttf"))
            pdf.add_font("Poppins-Medium", "", os.path.join(_FONT_DIR, "Poppins-Medium.ttf"))
            pdf.add_font("Poppins-Light", "", os.path.join(_FONT_DIR, "Poppins-Light.ttf"))
        except Exception:
            FONT = FONT_MEDIUM = FONT_LIGHT = "Helvetica"

    def ensure_space(height=20):
        if pdf.get_y() + height > pdf.h - 19:
            pdf.add_page()

    def text(value="", size=9.2, style="", color=DARK, height=5.2, font=None):
        pdf.set_font(font or FONT, style, size); pdf.set_text_color(*color)
        pdf.multi_cell(0, height, _clean_fn(value), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def heading(value, level=1):
        if level == 1:
            ensure_space(19); pdf.ln(2)
            pdf.set_font(FONT, "B", 15.5); pdf.set_text_color(*DARK)
            pdf.multi_cell(0, 8.2, _clean_fn(value), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            # A short, bold accent rule under the heading text reads as more
            # deliberately designed than a full-width hairline repeated
            # eleven times down the report.
            pdf.set_draw_color(*PRIMARY); pdf.set_line_width(0.9)
            pdf.line(pdf.l_margin, pdf.y + 1.2, pdf.l_margin + 20, pdf.y + 1.2)
            pdf.set_line_width(0.2); pdf.ln(5)
        elif level == 2:
            ensure_space(11); pdf.set_font(FONT_MEDIUM, "", 12); pdf.set_text_color(*DARK)
            pdf.multi_cell(0, 6.8, _clean_fn(value), new_x=XPos.LMARGIN, new_y=YPos.NEXT); pdf.ln(1)
        else:
            ensure_space(8); pdf.set_font(FONT, "B", 10.3); pdf.set_text_color(*PRIMARY)
            pdf.multi_cell(0, 5.8, _clean_fn(value), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def bullet(value, size=8.9):
        pdf.set_x(pdf.l_margin + 2); text("\u2022  " + _clean_fn(value), size=size, height=4.9)

    def wrapped_table(headers, rows, widths=None, font_size=6.8, padding=1.5, header_fill=LIGHT):
        if not rows: return
        available = pdf.w - pdf.l_margin - pdf.r_margin
        if widths is None: widths = [available / len(headers)] * len(headers)
        scale = available / sum(widths); widths = [w * scale for w in widths]
        table_w = sum(widths)

        def row_height(values, fs):
            # WHY not a hand-rolled word-wrap simulation: an earlier
            # version estimated wrapped line count itself (splitting on
            # spaces, comparing get_string_width to column width), but
            # that only approximates FPDF's actual wrapping — it doesn't
            # handle single unbreakable long tokens (e.g. "datetime64[ns]")
            # or Poppins' real glyph metrics the same way FPDF's own
            # renderer does, so estimated and actual height quietly
            # drifted apart and text got clipped by the next row. Asking
            # FPDF itself, via a dry run, uses the exact same wrapping
            # code path draw_row() renders with, so it can't drift.
            pdf.set_font(FONT, "B" if values is headers else "", fs)
            line_h = fs * .52 + .6
            max_h = 0.0
            for i, val in enumerate(values):
                s = _clean_fn(val)
                if not s: continue
                h = pdf.multi_cell(widths[i] - 2 * padding, line_h, s,
                                    dry_run=True, output=MethodReturnValue.HEIGHT)
                max_h = max(max_h, h)
            return max(5.6, max_h + 2 * padding + 1.5)

        def draw_row(values, header=False, row_index=0):
            fs = font_size
            h = row_height(values, fs)
            if pdf.get_y() + h > pdf.h - 19:
                pdf.add_page(); draw_row(headers, header=True)
            x0, y0 = pdf.l_margin, pdf.get_y()
            # One continuous fill spanning the full row width (rounded top
            # corners on the header only) reads as a real designed table
            # rather than a grid of individually-bordered spreadsheet
            # cells — the previous per-cell border+fill was the single
            # biggest thing making this report look like a raw data dump.
            if header:
                pdf.set_fill_color(*PRIMARY)
                pdf.rect(x0, y0, table_w, h, "F",
                         round_corners=(Corner.TOP_LEFT, Corner.TOP_RIGHT), corner_radius=1.6)
                row_text_color = WHITE
            else:
                pdf.set_fill_color(*(ZEBRA if row_index % 2 == 1 else WHITE))
                pdf.rect(x0, y0, table_w, h, "F")
                row_text_color = DARK
            pdf.set_font(FONT_MEDIUM if header else FONT, "", fs); pdf.set_text_color(*row_text_color)
            for i, val in enumerate(values):
                x = x0 + sum(widths[:i])
                pdf.set_xy(x + padding, y0 + padding)
                pdf.multi_cell(widths[i] - 2 * padding, fs * .52 + .6, _clean_fn(str(val)), border=0, fill=False, new_x=XPos.RIGHT, new_y=YPos.TOP)
            if not header:
                pdf.set_draw_color(*BORDER); pdf.set_line_width(0.2)
                pdf.line(x0, y0 + h, x0 + table_w, y0 + h)
            pdf.set_y(y0 + h)

        # WHY check here, separate from draw_row()'s own per-row overflow
        # guard below: that guard only fires once we're already mid-table,
        # so it correctly repeats the header when a LATER row overflows.
        # But if the header itself gets drawn when only the header (not
        # even one data row) fits in the remaining page space, it prints
        # orphaned alone at the bottom of the page — and then draw_row()'s
        # own guard repeats it again at the top of the next page before
        # the real data, producing a visible duplicated header. Checking
        # header + first row together, before drawing anything, ensures a
        # header is never printed without at least one row beneath it.
        header_h = row_height(headers, font_size)
        first_row_h = row_height(rows[0], font_size)
        if pdf.get_y() + header_h + first_row_h > pdf.h - 19:
            pdf.add_page()

        draw_row(headers, True)
        for idx, r in enumerate(rows): draw_row(r, False, row_index=idx)
        pdf.ln(2)

    def chart_image(data, max_w=165):
        if not data: return
        try:
            ensure_space(65); stream = io.BytesIO(data)
            pdf.image(stream, w=min(max_w, pdf.w - pdf.l_margin - pdf.r_margin)); pdf.ln(2)
        except Exception:
            text("[Chart could not be embedded in the PDF.]", size=8, color=MUTED)

    def section_label(label):
        pdf.set_fill_color(*LIGHT_BLUE); pdf.set_draw_color(*BORDER)
        pdf.rect(pdf.l_margin, pdf.get_y(), pdf.w - pdf.l_margin - pdf.r_margin, 7.5, "DF",
                 round_corners=True, corner_radius=1.8)
        pdf.set_xy(pdf.l_margin + 3.5, pdf.get_y() + 1.4); pdf.set_font(FONT_MEDIUM, "", 8.5); pdf.set_text_color(*PRIMARY_DARK)
        pdf.cell(0, 4.5, _clean_fn(label)); pdf.ln(8)

    # COVER
    pdf.add_page()
    hero_h = 58
    pdf.set_fill_color(*PRIMARY); pdf.rect(0, 0, pdf.w, hero_h, "F")
    pdf.set_xy(pdf.l_margin, 19)
    pdf.set_font(FONT, "B", 24); pdf.set_text_color(*WHITE)
    pdf.multi_cell(0, 11.5, _clean_fn("DATA ANALYSIS REPORT"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_x(pdf.l_margin); pdf.set_font(FONT_MEDIUM, "", 11.5); pdf.set_text_color(219, 234, 254)
    pdf.cell(0, 7, _clean_fn("Axiomrow \u2014 Automated Data Analysis"))
    pdf.set_y(hero_h + 9)
    text(_clean_fn(dataset_name), size=12, style="B", color=DARK, height=7)
    text(datetime.now().strftime("Generated on %B %d, %Y at %H:%M"), size=9.2, color=MUTED, height=6, font=FONT_LIGHT)
    pdf.ln(5)

    # KPI cards: one row of four, rounded corners, a colored accent bar
    # on the left edge of each — replaces the previous flat 2x2 grid of
    # plain gray boxes with something that reads as an actual designed
    # dashboard summary rather than a placeholder.
    card_y = pdf.get_y(); gap = 5
    card_w = (pdf.w - pdf.l_margin - pdf.r_margin - gap * 3) / 4
    card_h = 23
    cards = [
        ("RECORDS", f"{len(df):,}"), ("FEATURES", f"{len(df.columns):,}"),
        ("NUMERIC", f"{len(numeric):,}"), ("CATEGORICAL", f"{len(categorical):,}"),
    ]
    for i, (label, val) in enumerate(cards):
        x = pdf.l_margin + i * (card_w + gap)
        pdf.set_fill_color(*LIGHT); pdf.set_draw_color(*BORDER)
        pdf.rect(x, card_y, card_w, card_h, "DF", round_corners=True, corner_radius=2.0)
        pdf.set_fill_color(*PRIMARY)
        pdf.rect(x, card_y, 1.8, card_h, "F",
                 round_corners=(Corner.TOP_LEFT, Corner.BOTTOM_LEFT), corner_radius=1.8)
        pdf.set_xy(x + 5, card_y + 4); pdf.set_font(FONT_MEDIUM, "", 6.8); pdf.set_text_color(*MUTED)
        pdf.cell(card_w - 8, 4, label)
        pdf.set_xy(x + 5, card_y + 11); pdf.set_font(FONT, "B", 15); pdf.set_text_color(*PRIMARY)
        pdf.cell(card_w - 8, 8, val)
    pdf.set_y(card_y + card_h + 7)

    text("A structured, dataset-driven report covering data quality, descriptive statistics, exploratory analysis, verified insights, recommendations and saved chat analysis.", size=9.2, color=MUTED, height=5.2)
    pdf.ln(5)

    # Report Contents — turns what used to be a large empty lower half
    # of the cover page into something actually useful: a real preview
    # of what the report contains, respecting which optional sections
    # (EDA, chat analysis) are actually included for this run.
    toc_sections = ["Executive Summary", "1. Analysis Methodology", "2. Dataset Overview",
                     "3. Column / Feature Description", "4. Data Quality & Cleaning",
                     "5. Descriptive Statistics"]
    if a["include_eda"]:
        toc_sections.append("6. Exploratory Data Analysis")
    toc_sections += ["7. Key Insights", "8. Conclusion", "9. Limitations & Scope", "10. Recommendations"]
    if a["include_chat"]:
        toc_sections.append("11. Chat Analysis")

    section_label("Report Contents")
    col_w = (pdf.w - pdf.l_margin - pdf.r_margin) / 2
    start_y = pdf.get_y()
    for i, sec in enumerate(toc_sections):
        col = i % 2; row = i // 2
        x = pdf.l_margin + col * col_w; y = start_y + row * 7.2
        pdf.set_xy(x, y); pdf.set_font(FONT, "", 9); pdf.set_text_color(*DARK)
        pdf.cell(col_w - 4, 6, _clean_fn(sec))
    rows_used = -(-len(toc_sections) // 2)  # ceil division
    pdf.set_y(start_y + rows_used * 7.2)

    # EXECUTIVE SUMMARY
    pdf.add_page(); heading("EXECUTIVE SUMMARY", 1)
    text(summary, size=9.5, height=5.5)
    if key_items:
        section_label("At-a-glance findings")
        for title, body in key_items[:4]:
            heading(title, 3); text(body, size=8.8, height=4.9)

    # 1 METHODOLOGY
    heading("1. ANALYSIS METHODOLOGY", 1)
    for item in a["methodology"]:
        bullet(item, size=8.6)

    # 2 OVERVIEW
    heading("2. DATASET OVERVIEW", 1)
    overview_rows = [["Records", f"{len(df):,}"], ["Features", f"{len(df.columns):,}"], ["Numeric columns", f"{len(numeric):,}"], ["Categorical columns", f"{len(categorical):,}"], ["Date/time columns", f"{len(date_cols):,}"], ["Missing values", f"{a['missing_total']:,}"], ["Duplicate rows", f"{a['duplicate_total']:,}"]]
    wrapped_table(["Metric", "Value"], overview_rows, widths=[75, 95], font_size=8.1)

    if a["include_preview"]:
        heading("Data Preview", 2)
        preview = a["preview"]
        preview_rows = [[_safe_text(v) for v in row.tolist()] for _, row in preview.iterrows()]
        if not preview.empty:
            wrapped_table([str(c)[:24] for c in preview.columns], preview_rows, font_size=6.0, padding=1.2)

    # 2 COLUMN DESCRIPTION
    heading("3. COLUMN / FEATURE DESCRIPTION", 1)
    rows = []
    for c in columns:
        rows.append([_safe_text(c["name"]), _safe_text(c["detected_type"]), _safe_text(c["dtype"]), _safe_text(c["description"]), f"{c['unique_count']:,}", f"{c['missing_count']:,}", _fmt_pct(c["missing_pct"], 2)])
    wrapped_table(["Column", "Detected", "dtype", "Description", "Unique", "Missing", "Missing %"], rows, widths=[30, 23, 21, 60, 20, 21, 23], font_size=5.7, padding=1.1)

    # 3 QUALITY
    heading("4. DATA QUALITY & CLEANING", 1)
    heading("Quality Findings", 2)
    for x in quality: bullet(x)
    heading("Cleaning / Preparation Actions", 2)
    for x in actions: bullet(x)
    heading("Outlier Assessment", 2)
    out_rows = []
    for r in outliers:
        out_rows.append([_safe_text(r["column"]), f"{r['n']:,}", _fmt_number(r["q1"]), _fmt_number(r["q3"]), _fmt_number(r["iqr"]), "N/A" if r["count"] is None else f"{r['count']:,}", _safe_text(r["status"])])
    if out_rows:
        wrapped_table(["Column", "N", "Q1", "Q3", "IQR", "Outliers", "Status"], out_rows, widths=[35, 17, 22, 22, 22, 24, 58], font_size=6.0, padding=1.1)

    # 4 DESCRIPTIVE STATS
    heading("5. DESCRIPTIVE STATISTICS", 1)
    heading("Numeric Statistics", 2)
    stat_rows = []
    for r in stats:
        stat_rows.append([_safe_text(r["column"]), f"{r['count']:,}", _fmt_number(r["mean"]), _fmt_number(r["median"]), _fmt_number(r["std"]), _fmt_number(r["variance"]), _fmt_number(r["q1"]), _fmt_number(r["q3"]), _fmt_number(r["min"]), _fmt_number(r["max"]), _fmt_number(r["skew"]), _fmt_number(r["cv"])])
    if stat_rows:
        wrapped_table(["Column", "N", "Mean", "Median", "Std", "Variance", "Q1", "Q3", "Min", "Max", "Skew", "CV %"],
                      stat_rows, widths=[25, 12, 18, 18, 18, 22, 18, 18, 18, 18, 18, 18], font_size=5.3, padding=0.8)
    else: text("No numeric columns were detected.", size=9, color=MUTED)

    heading("Categorical Statistics", 2)
    cat_rows = [[_safe_text(r["column"]), f"{r['unique']:,}", _safe_text(r["top"]), f"{r['frequency']:,}", _fmt_pct(r["share"])] for r in cat_stats]
    if cat_rows: wrapped_table(["Column", "Unique", "Most Frequent", "Frequency", "Share"], cat_rows, widths=[50, 24, 68, 30, 28], font_size=6.8)
    else: text("No categorical columns were detected.", size=9, color=MUTED)

    primary_for_groups = a["primary"]
    group_rows = a["group_rows"]
    if group_rows:
        heading("Group Performance Summary", 2)
        table_rows = [[_safe_text(r["dimension"]), _safe_text(r["group"]), f"{r['records']:,}", _fmt_number(r["total"]), _fmt_number(r["average"]), _fmt_pct(r["share"])] for r in group_rows]
        wrapped_table(["Dimension", "Group", "Records", f"Total {primary_for_groups}", f"Average {primary_for_groups}", "Share"], table_rows, widths=[35, 40, 22, 38, 42, 25], font_size=5.9, padding=1.0)

    # 5 EDA narrative
    if a["include_eda"]:
        heading("6. EXPLORATORY DATA ANALYSIS", 1)
        heading("6.1 Univariate Analysis", 2)
        for r in stats:
            text(f"{r['column']}: mean {_fmt_number(r['mean'])}; median {_fmt_number(r['median'])}; range {_fmt_number(r['min'])} to {_fmt_number(r['max'])}; skewness {_fmt_number(r['skew'])}.", size=8.8, height=4.8)
        if not stats: text("No numeric variables are available for univariate analysis.", size=8.8, color=MUTED)

        heading("6.2 Categorical Analysis", 2)
        for col, vc in a["categorical_top"].items():
            if len(vc): bullet(f"{col}: " + ", ".join(f"{k} ({int(v):,})" for k,v in vc.items()), size=8.6)
        if not categorical: text("No categorical variables are available for categorical analysis.", size=8.8, color=MUTED)

        heading("6.3 Time-Based Analysis", 2)
        if date_cols:
            text(f"Date/time field: {date_cols[0]}", size=8.8)
            if a["date_summary"]:
                ds=a["date_summary"]; bullet(f"Observed period: {ds['min'].strftime('%d %b %Y')} to {ds['max'].strftime('%d %b %Y')}."); bullet(f"Unique dates: {ds['unique_days']:,}."); bullet(f"Time-trend aggregation: {a['time_granularity']}.")
        else: text("No sufficiently reliable date/time column was detected.", size=8.8, color=MUTED)
        if a["date_profile"]:
            date_rows = [[_safe_text(v) for v in r] for r in a["date_profile"]]
            wrapped_table(["Date/Time Column", "Valid", "Earliest", "Latest", "Unique Dates"],
                          date_rows, widths=[48, 22, 35, 35, 30], font_size=6.6)

        heading("6.4 Relationship Analysis", 2)
        if a["best_corr"]:
            best=a["best_corr"]; bullet(f"Strongest absolute Pearson correlation: {best[2]} vs {best[3]} = {best[1]:.3f}."); bullet("Correlation measures linear association and does not establish causation.")
        else: text("At least two usable numeric columns are required for relationship analysis.", size=8.8, color=MUTED)

        if a["include_charts"]:
            for section, title, data, interpretation in chart_data:
                ensure_space(100); section_label(f"{section} | {title}"); chart_image(data, max_w=165); heading("Interpretation", 3); text(interpretation, size=8.5, color=DARK, height=4.8)

    # 6 INSIGHTS
    heading("7. KEY INSIGHTS", 1)
    for title, body in key_items:
        heading(title, 3); text(body, size=9.0, height=5.0)

    # 7 CONCLUSION
    heading("8. CONCLUSION", 1)
    text(summary, size=9.0, height=5.0)
    # WHY conditional: this used to state charts were included
    # unconditionally, even when include_charts was False (or produced
    # zero charts) — a report can't claim to contain visual evidence it
    # doesn't actually contain.
    closing_note = "This report separates descriptive facts from interpreted findings and uses dataframe-verified calculations for the reported insights."
    if a["include_charts"] and chart_data:
        closing_note += " Visualizations are included as supporting evidence."
    text(closing_note, size=9.0, color=MUTED, height=5.0)

    # 8 LIMITATIONS & SCOPE
    heading("9. LIMITATIONS & SCOPE", 1)
    bullet("Findings are descriptive and exploratory; they do not establish causation.")
    bullet("Small datasets can produce unstable distributions and correlations.")
    bullet("IQR outlier detection is a screening method and should be reviewed with domain context.")

    # 9 RECOMMENDATIONS
    heading("10. RECOMMENDATIONS", 1)
    for i, rec in enumerate(recommendations, 1): bullet(f"{i}. {rec}")

    # 10 CHAT ANALYSIS - deliberately last.
    if a["include_chat"]:
        heading("11. CHAT ANALYSIS", 1)
    if a["include_chat"] and chats_flat:
        for i, msg in enumerate(chats_flat, 1):
            ensure_space(30)
            heading(f"Question {i}", 2)
            text(f"Question: {_safe_text(msg['question'])}", size=9.1, style="B", height=5.0)
            answer = _safe_text(msg["answer"]).strip()
            if answer: text(f"Answer: {answer}", size=9.0, height=5.0)

            if msg.get("code_result") is not None:
                heading("Result", 3); text(_safe_text(msg["code_result"]), size=7.8, color=MUTED, height=4.3, font="Courier")
            if msg.get("sql_result") is not None:
                heading("SQL Result", 3); text(_safe_text(msg["sql_result"]), size=7.8, color=MUTED, height=4.3, font="Courier")

            chart = None; chart_title = None
            chart_requested = (
                str(msg.get("type", "")).lower() == "chart"
                or bool(msg.get("chart_config"))
                or bool(msg.get("chart_bytes"))
                or _looks_like_chart_question(msg.get("question", ""))
            )
            if chart_requested:
                chart, chart_title = _resolve_chat_chart(df, msg)
            if chart:
                ensure_space(90)
                heading(_safe_text(chart_title or "Chart"), 3)
                chart_image(chart, max_w=155)
            elif chart_requested:
                # Keep the failure explicit rather than silently omitting a
                # chart, but this should only be reached after all deterministic
                # reconstruction paths have failed.
                logger.warning("Chat chart could not be reconstructed")
                text("The chat requested a chart, but the chart could not be reconstructed from the saved question/configuration.", size=8.2, color=MUTED)
            pdf.ln(3)
    elif a["include_chat"]:
        text("No saved chat questions are available.", size=9, color=MUTED)

    return bytes(pdf.output())
