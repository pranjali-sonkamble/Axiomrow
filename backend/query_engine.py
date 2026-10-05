"""Deterministic dataframe query handling for exact questions.

The live DataFrame is the source of truth.  The parser deliberately returns
None for questions it cannot understand; callers can then use their normal
LLM/fallback path instead of incorrectly reporting zero matching rows.
"""

import re
from typing import Optional, Sequence, Any

import pandas as pd


MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

# Natural-language aliases.  These are semantic aliases, not dataset answers.
COLUMN_ALIASES = {
    "product": ("product", "products", "item", "items", "product_name", "productname"),
    "customer": ("customer", "customers", "customer_id", "customerid",
                 "customer_name", "customername", "client", "client_id", "clientid"),
    "date": ("date", "dates", "datetime", "timestamp", "transaction_date"),
    "region": ("region", "regions", "area", "territory"),
    "revenue": ("revenue", "sales", "sale", "amount", "total_revenue", "total_sales"),
    "quantity": ("quantity", "quantities", "qty", "units"),
    "transaction": ("transaction", "transactions", "record", "records", "row", "rows"),
}

_CUSTOMER_ALIASES = {
    a for a in COLUMN_ALIASES["customer"]
    if a not in {"customer", "customers"}
}

def _text(content: str) -> dict:
    return {"answer_type": "text", "content": content}


def _word_in_text(word: str, text: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(str(word)) + r"(?!\w)", str(text)) is not None


def _normalise(s: Any) -> str:
    return re.sub(r"[^a-z0-9_]+", " ", str(s).strip().lower()).strip()


def _column_tokens(col: Any) -> set[str]:
    raw = str(col).strip().lower()
    normal = _normalise(raw)
    tokens = {raw, normal, normal.replace(" ", "_")}
    # Singular/plural forms and common semantic aliases.
    if normal.endswith("s"):
        tokens.add(normal[:-1])
    else:
        tokens.add(normal + "s")
    # "sales rep" / "sales reps" must resolve to the column sales_rep.
    spaced = normal.replace("_", " ")
    tokens.add(spaced)
    tokens.add(spaced[:-1] if spaced.endswith("s") else spaced + "s")
    for canonical, aliases in COLUMN_ALIASES.items():
        if normal in {_normalise(a) for a in aliases}:
            tokens.update(_normalise(a) for a in aliases)
            tokens.add(canonical)
    return {t for t in tokens if t}


def _find_column(question: str, columns, preferred: Optional[str] = None):
    """Find a real dataframe column mentioned by name or common semantic alias."""
    q = _normalise(question)
    candidates = list(columns)

    if preferred:
        preferred_norm = _normalise(preferred)
        preferred_candidates = [
            c for c in candidates if preferred_norm in _column_tokens(c)
        ]
        if preferred_candidates:
            return preferred_candidates[0]

    # Exact/alias match first, longest names first.
    for col in sorted(candidates, key=lambda x: len(str(x)), reverse=True):
        if any(_word_in_text(token, q) for token in _column_tokens(col)):
            return col
    return None


def _find_column_by_tokens(tokens: Sequence[str], columns):
    for token in tokens:
        found = _find_column(token, columns)
        if found is not None:
            return found
    return None


def _find_customer_column(df: pd.DataFrame):
    """Return an actual customer identifier, never a row-count substitute."""
    for c in df.columns:
        n = _normalise(c)
        if n in _CUSTOMER_ALIASES:
            return c
    # Conservative fuzzy handling for equivalent identifiers.
    for c in df.columns:
        n = _normalise(c).replace(" ", "_")
        if ("customer" in n or "client" in n) and (
            "id" in n or "name" in n or n in {"customer", "client"}
        ):
            return c
    return None


def _find_numeric_column_mentioned(df: pd.DataFrame, ql: str):
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if _find_column(ql, [c]) is not None:
            return c
        # Semantic metric aliases (e.g. "sales" for revenue).
        n = _normalise(c)
        for canonical, aliases in COLUMN_ALIASES.items():
            if canonical in {"revenue", "quantity"} and n in {_normalise(a) for a in aliases}:
                if any(_word_in_text(a, ql) for a in aliases):
                    return c
    return None


def _find_date_column(df: pd.DataFrame):
    for c in df.columns:
        n = _normalise(c)
        if n in {"date", "datetime", "timestamp", "transaction_date"} or n.endswith("_date") or n.endswith("_datetime") or " date " in f" {n} ":
            return c
    # Date dtype is a safe fallback when the name is non-standard.
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return None


def _find_named_value_column(question: str, df: pd.DataFrame):
    """Find a categorical column/value pair from 'for Laptop' style wording."""
    q = question.strip().strip(" .?!")
    # Explicit column=value syntax.
    m = re.search(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:is|equals|equal\s+to|=)\s*"
        r"['\"]?([^,';.!?]+?)['\"]?(?=\s+(?:and|or)\s+|$)",
        q, re.I,
    )
    if m:
        col = _find_column(m.group(1), df.columns)
        if col is not None:
            return col, m.group(2).strip().strip("'\"")

    # "for Laptop", "for North", "of Laptop" — only resolve if exactly
    # one categorical column contains that value. This avoids treating
    # arbitrary explanatory text as a filter.
    for prep in ("for", "of"):
        matches = list(re.finditer(
            rf"\b{prep}\s+['\"]?([^,';.!?]+?)['\"]?(?=\s+(?:as|by|per|where|on|in)\b|$)",
            q, re.I,
        ))
        for m in matches:
            value = m.group(1).strip().strip("'\"")
            if not value or len(value.split()) > 5:
                continue
            hits = []
            for c in df.columns:
                if pd.api.types.is_numeric_dtype(df[c]) or c == _find_date_column(df):
                    continue
                vals = df[c].astype(str).str.strip().str.casefold()
                if vals.eq(value.casefold()).any():
                    hits.append(c)
            if len(hits) == 1:
                return hits[0], value
    return None


def _parse_numeric_filter(question: str, df: pd.DataFrame):
    """Parse numeric comparisons and return (matched_dataframe, understood).

    Supported operators: >, <, >=, <=, =, ==, !=, greater/more/above/over,
    less/below/under, at least/at most, and between X and Y.
    """
    q = str(question).lower()
    numeric_columns = [
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_columns:
        return None, False

    # Match actual column names plus semantic metric aliases where an alias
    # maps unambiguously to a numeric column.
    col_alternatives = []
    for c in numeric_columns:
        col_alternatives.append((str(c).lower(), c))
        n = _normalise(c)
        for canonical, aliases in COLUMN_ALIASES.items():
            if canonical in {"revenue", "quantity"} and n in {_normalise(a) for a in aliases}:
                for a in aliases:
                    col_alternatives.append((_normalise(a), c))
    col_alternatives = sorted(set(col_alternatives), key=lambda x: len(x[0]), reverse=True)
    col_pattern = "|".join(re.escape(a) for a, _ in col_alternatives)
    alias_to_col = {a: c for a, c in col_alternatives}

    number = r"-?\d+(?:\.\d+)?"
    between_pattern = re.compile(
        rf"(?P<col>{col_pattern})\s+(?:is\s+)?between\s+"
        rf"(?P<low>{number})\s+and\s+(?P<high>{number})\b", re.I,
    )
    comparison_pattern = re.compile(
        rf"(?P<col>{col_pattern})\s*(?:is\s+)?"
        rf"(?P<op>greater\s+than\s+or\s+equal\s+to|"
        rf"less\s+than\s+or\s+equal\s+to|"
        rf"greater\s+than|more\s+than|above|over|"
        rf"less\s+than|fewer\s+than|below|under|"
        rf"at\s+least|at\s+most|equal\s+to|>=|<=|!=|==|=|>|<)"
        rf"\s*(?P<num>{number})\b", re.I,
    )

    matches = []
    masks = []

    for m in between_pattern.finditer(q):
        col = alias_to_col.get(m.group("col").lower())
        if col is None:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        low, high = float(m.group("low")), float(m.group("high"))
        masks.append(((series >= low) & (series <= high)).fillna(False))
        matches.append(m)

    for m in comparison_pattern.finditer(q):
        col = alias_to_col.get(m.group("col").lower())
        if col is None:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() == 0:
            return None, False
        op = m.group("op").lower().strip()
        value = float(m.group("num"))
        if op in (">", "greater than", "more than", "above", "over"):
            mask = series > value
        elif op in (">=", "greater than or equal to", "at least"):
            mask = series >= value
        elif op in ("<", "less than", "fewer than", "below", "under"):
            mask = series < value
        elif op in ("<=", "less than or equal to", "at most"):
            mask = series <= value
        elif op in ("=", "==", "equal to"):
            mask = series == value
        elif op == "!=":
            mask = series != value
        else:
            return None, False
        masks.append(mask.fillna(False))
        matches.append(m)

    if not matches:
        return None, False

    ordered = sorted(zip(matches, masks), key=lambda x: x[0].start())
    ordered_matches = [x[0] for x in ordered]
    ordered_masks = [x[1] for x in ordered]

    connectors = re.findall(
        r"\b(and|or)\b",
        q[ordered_matches[0].end():ordered_matches[-1].start()],
        re.I,
    )
    # If no connector occurs, all comparisons are ANDed. This is the
    # conservative interpretation for multiple filters.
    if "or" in [c.lower() for c in connectors]:
        combined = ordered_masks[0]
        for mask in ordered_masks[1:]:
            combined = combined | mask
    else:
        combined = ordered_masks[0]
        for mask in ordered_masks[1:]:
            combined = combined & mask

    return df.loc[combined], True


def _format_rows(result: pd.DataFrame, max_rows: int = 30) -> str:
    if result.empty:
        return "No matching rows found."
    if len(result) > max_rows:
        preview = result.head(max_rows).to_string(index=False)
        return f"{preview}\n... and {len(result) - max_rows:,} more rows ({len(result):,} total)"
    return result.to_string(index=False)


def _format_unique_values(values) -> str:
    cleaned = []
    for value in values:
        if pd.isna(value):
            continue
        value = str(value).strip()
        if value and value not in cleaned:
            cleaned.append(value)
    return ", ".join(cleaned)


def _strip_categorical_phrases(question: str, df: pd.DataFrame) -> str:
    """Remove multi-word categorical column names ('sales rep', 'customer_segment')
    from a question so their words ('sales') are not mistaken for a metric alias."""
    q = str(question).lower()
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        for tok in sorted(_column_tokens(c), key=len, reverse=True):
            if " " in tok or "_" in tok:
                q = re.sub(r"(?<!\w)" + re.escape(tok) + r"(?!\w)", " ", q)
    return q


def _metric_column(question: str, df: pd.DataFrame):
    """Resolve a numeric metric, including natural plural forms such as 'profit margins'."""
    question = _strip_categorical_phrases(question, df)
    col = _find_column(question, df.columns)
    if col is not None and pd.api.types.is_numeric_dtype(df[col]):
        return col
    found = _find_numeric_column_mentioned(df, question.lower())
    if found is not None:
        return found
    qn = _normalise(question)
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cn = _normalise(c)
        cn_space = cn.replace("_", " ")
        singular = cn_space[:-1] if cn_space.endswith("s") else cn_space
        if cn in qn or cn_space in qn or singular in qn:
            return c
    return None


def _aggregation(question: str) -> Optional[str]:
    q = question.lower()
    if re.search(r"\b(average|avg|mean)\b", q):
        return "mean"
    if re.search(r"\b(total|sum)\b", q):
        return "sum"
    # "revenue by product" is a grouped sum by default.
    if re.search(r"\b(revenue|sales|amount)\b", q):
        return "sum"
    return None


def _group_column(question: str, df: pd.DataFrame):
    q = question.lower()

    # Explicit grouping prepositions.
    m = re.search(r"\b(?:by|per)\s+(?:the\s+)?([A-Za-z_][A-Za-z0-9_]*)(?:\s+([A-Za-z_][A-Za-z0-9_]*))?", q)
    if m:
        candidates = [m.group(1)]
        if m.group(2):
            candidates.insert(0, f"{m.group(1)} {m.group(2)}")
        for cand in candidates:
            col = _find_column(cand, df.columns)
            if col is not None:
                return col

    # "for each date/product/region", "across each date".
    m = re.search(r"\b(?:for|across)\s+(?:each|every)\s+([A-Za-z_][A-Za-z0-9_]*)", q)
    if m:
        col = _find_column(m.group(1), df.columns)
        if col is not None:
            return col

    # Entity phrases without a preposition, used by "which product generated..."
    for phrase, preferred in (
        ("products", "product"), ("product", "product"),
        ("dates", "date"), ("date", "date"),
        ("regions", "region"), ("region", "region"),
        ("customers", "customer"), ("customer", "customer"),
    ):
        if _word_in_text(phrase, q):
            if preferred == "customer":
                return _find_customer_column(df)
            col = _find_column(preferred, df.columns)
            if col is not None:
                return col
    return None


def _year_filter(df: pd.DataFrame, question: str):
    """Filter by an explicitly mentioned four-digit year."""
    date_col = _find_date_column(df)
    if date_col is None:
        return df, False
    years = sorted({int(y) for y in re.findall(r"\b(20\d{2})\b", str(question))})
    if len(years) != 1:
        return df, False
    dates = pd.to_datetime(df[date_col], errors="coerce")
    return df.loc[dates.dt.year == years[0]], True


def _extract_years(question: str):
    return sorted({int(y) for y in re.findall(r"\b(20\d{2})\b", str(question))})


def _extract_month_year_pairs(question: str):
    q = str(question).lower()
    # First capture month tokens in textual order.
    month_hits = []
    month_pattern = r"\b(" + "|".join(MONTH_NAMES) + r")\b"
    for m in re.finditer(month_pattern, q):
        month_hits.append((m.start(), MONTH_NAMES[m.group(1)], m.group(1).title()))
    years = _extract_years(q)
    pairs = []
    # Common natural-language form: "January and February 2024".
    if len(month_hits) >= 2 and len(years) == 1:
        year = years[0]
        for _, month_num, label in month_hits:
            pairs.append((month_num, year, label))
        return pairs
    # Explicit forms: "January 2024", "February 2024".
    for month, month_num in MONTH_NAMES.items():
        for m in re.finditer(rf"\b{month}\s+(20\d{{2}})\b", q):
            pairs.append((month_num, int(m.group(1)), month.title()))
    # Deduplicate while preserving textual month order.
    seen = set(); ordered = []
    for pair in sorted(pairs, key=lambda x: q.find(x[2].lower())):
        key = (pair[0], pair[1])
        if key not in seen:
            seen.add(key); ordered.append(pair)
    return ordered


def _comparison_request(question: str, df: pd.DataFrame):
    """Return a verified comparison result for explicit two-period comparisons."""
    ql = str(question).lower()
    if not re.search(r"\b(compare|comparison|versus|vs\.?|higher|lower)\b", ql):
        # Also support the concise "January and February revenue" form.
        if not (re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", ql)
                and re.search(r"\band\b", ql)
                and _metric_column(question, df) is not None):
            return None

    metric_col = _metric_column(question, df)
    if metric_col is None:
        return None

    pairs = _extract_month_year_pairs(question)
    if len(pairs) < 2:
        month_hits = re.findall(r"\b(" + "|".join(MONTH_NAMES) + r")\b", ql)
        date_col = _find_date_column(df)
        if len(month_hits) >= 2 and date_col is not None and not _extract_years(question):
            dates = pd.to_datetime(df[date_col], errors="coerce")
            unique_years = sorted(dates.dt.year.dropna().unique().tolist())
            if len(unique_years) == 1:
                year = int(unique_years[0])
                pairs = [(MONTH_NAMES[month], year, month.title()) for month in month_hits[:2]]
    if len(pairs) >= 2:
        # Require a year when the dataset spans multiple years; otherwise the
        # same month name would be ambiguous.
        selected = pairs[:2]
        vals = []
        date_col = _find_date_column(df)
        if date_col is None:
            return None
        dates = pd.to_datetime(df[date_col], errors="coerce")
        for month_num, year, label in selected:
            mask = (dates.dt.year == year) & (dates.dt.month == month_num)
            value = float(df.loc[mask, metric_col].sum())
            vals.append((label, year, value))
        a_label, a_year, a = vals[0]; b_label, b_year, b = vals[1]
        diff = b - a
        pct = (diff / a * 100.0) if a != 0 else None
        pct_text = f"{pct:.2f}%" if pct is not None else "N/A"
        return _text(
            f"Comparison of {a_label} {a_year} {metric_col}: {a:,.2f} "
            f"{b_label} {b_year}: {b:,.2f} "
            f"Absolute difference: {diff:,.2f} Percentage difference: {pct_text}"
        )

    years = _extract_years(question)
    if len(years) >= 2:
        date_col = _find_date_column(df)
        if date_col is None:
            return None
        dates = pd.to_datetime(df[date_col], errors="coerce")
        vals = []
        for year in years[:2]:
            value = float(df.loc[dates.dt.year == year, metric_col].sum())
            vals.append((year, value))
        a_year, a = vals[0]; b_year, b = vals[1]
        diff = b - a
        pct = (diff / a * 100.0) if a != 0 else None
        pct_text = f"{pct:.2f}%" if pct is not None else "N/A"
        return _text(
            f"Comparison of {metric_col} {a_year}: {a:,.2f} {b_year}: {b:,.2f} "
            f"Absolute difference: {diff:,.2f} Percentage difference: {pct_text}"
        )
    return None


def _multi_group_columns(question: str, df: pd.DataFrame):
    """Resolve two explicit grouping dimensions from 'by A and B'."""
    q = str(question).lower()
    m = re.search(r"\bby\s+(.+?)(?:\s+(?:as|using|with)\b|$|[.!?])", q)
    if not m:
        return []
    phrase = m.group(1).strip()
    parts = re.split(r"\s*(?:,|\band\b|&|\bvs\.?\b)\s*", phrase)
    cols = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Strip chart words that may follow the grouping phrase.
        part = re.sub(r"\b(?:a|an)?\s*(?:grouped\s+)?(?:bar|line|pie)\s+(?:chart|plot)\b.*$", "", part).strip()
        col = _find_column(part, df.columns)
        if col is not None and col not in cols:
            cols.append(col)
    return cols[:2]


def _monthly_group(df: pd.DataFrame, metric_col: str):
    date_col = _find_date_column(df)
    if date_col is None:
        return None
    dates = pd.to_datetime(df[date_col], errors="coerce")
    temp = df.assign(__month__=dates.dt.to_period("M"))
    return temp.groupby("__month__", dropna=False)[metric_col].sum().sort_index()


def _duplicate_check(question: str, df: pd.DataFrame):
    ql = str(question).lower()
    if not re.search(r"\bduplicate\b", ql):
        return None
    if not re.search(r"\b(row|rows|record|records|dataset)\b", ql):
        return None
    count = int(df.duplicated().sum())
    return _text(f"Duplicate rows: {count:,}")


def _negative_value_check(question: str, df: pd.DataFrame):
    ql = str(question).lower()
    if not re.search(r"\bnegative\b|\bless\s+than\s+zero\b|\bbelow\s+zero\b", ql):
        return None
    metric_col = _metric_column(question, df)
    if metric_col is not None:
        numeric = pd.to_numeric(df[metric_col], errors="coerce")
        count = int((numeric < 0).sum())
        label = str(metric_col)
        if count:
            return _text(f"Negative {label} values present: {count:,}")
        return _text(f"No negative {label} values were found.")
    # Dataset-wide check when no metric was named explicitly.
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    counts = {str(c): int((pd.to_numeric(df[c], errors="coerce") < 0).sum()) for c in numeric_cols}
    total = sum(counts.values())
    if total == 0:
        return _text("Negative values present: False")
    labels = ", ".join(f"{c}: {n:,}" for c, n in counts.items() if n)
    return _text(f"Negative values present: True ({labels})")


def _date_filter(df: pd.DataFrame, question: str):
    """Return a filtered frame for an exact date, month/year, or year."""
    date_col = _find_date_column(df)
    if date_col is None:
        return df, False

    q = str(question).lower()
    dates = pd.to_datetime(df[date_col], errors="coerce")

    iso = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", q)
    natural = re.search(
        r"\b(" + "|".join(MONTH_NAMES) + r")\s+(\d{1,2}),?\s+(20\d{2})\b", q
    )
    month_year = re.search(
        r"\b(" + "|".join(MONTH_NAMES) + r")\s+(20\d{2})\b", q
    )
    if iso:
        target = pd.to_datetime(iso.group(1), errors="coerce")
        return df.loc[dates.dt.normalize() == target.normalize()], True
    if natural:
        month, day, year = natural.group(1), int(natural.group(2)), int(natural.group(3))
        target = pd.Timestamp(year=year, month=MONTH_NAMES[month], day=day)
        return df.loc[dates.dt.normalize() == target.normalize()], True
    if month_year:
        year, month = int(month_year.group(2)), MONTH_NAMES[month_year.group(1)]
        return df.loc[(dates.dt.year == year) & (dates.dt.month == month)], True

    # A bare year such as "for 2024" is an explicit date filter too.
    years = _extract_years(q)
    if len(years) == 1:
        return df.loc[dates.dt.year == years[0]], True
    return df, False


def _has_aggregation_operation(question: str) -> bool:
    q = question.lower()
    return bool(
        re.search(r"\b(total|sum|average|avg|mean|revenue|sales|amount)\b", q)
        and (
            re.search(r"\b(by|per|for\s+each|each|on|for)\b", q)
            or re.search(r"\b20\d{2}-\d{2}-\d{2}\b", q)
            or re.search(r"\b(" + "|".join(MONTH_NAMES) + r")\b", q)
        )
    )


def _rank_entity(question: str, df: pd.DataFrame):
    q = question.lower()
    if re.search(r"\btransaction(s)?\b|\b(row|rows|record|records)\b", q):
        return "transaction", None
    if _word_in_text("customer", q) or _word_in_text("customers", q):
        return "customer", _find_customer_column(df)
    if _word_in_text("product", q) or _word_in_text("products", q):
        return "product", _find_column("product", df.columns)
    if _word_in_text("date", q) or _word_in_text("dates", q):
        return "date", _find_date_column(df)
    if _word_in_text("region", q) or _word_in_text("regions", q):
        return "region", _find_column("region", df.columns)
    # Any other categorical column named in the question (sales_rep,
    # customer_segment, ...). Only when exactly one is named, never a guess.
    date_col = _find_date_column(df)
    hits = [c for c in df.columns
            if c != date_col and not pd.api.types.is_numeric_dtype(df[c])
            and _find_column(q, [c]) is not None]
    if len(hits) == 1:
        return "category", hits[0]
    return None, None


def _json_scalar(value):
    """Convert pandas/numpy scalars to JSON-safe values for persisted chart state."""
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass
    return value


def _find_categorical_value(question: str, df: pd.DataFrame):
    """Resolve a literal categorical value mentioned in a purchase/filter phrase."""
    q = str(question)
    # First handle explicit "column is value" / "for value".
    explicit = _find_named_value_column(q, df)
    if explicit:
        return explicit

    # Then scan categorical columns for a value literally present in the data.
    # This is used for phrases such as "customers who purchased Laptop".
    ignored = {_find_date_column(df)}
    candidates = []
    ql = q.casefold()
    for c in df.columns:
        if c in ignored or pd.api.types.is_numeric_dtype(df[c]):
            continue
        values = df[c].dropna().astype(str).str.strip()
        for value in values.drop_duplicates():
            if value and _word_in_text(value.casefold(), ql):
                candidates.append((c, value))
    # Only accept an unambiguous data-backed value.
    unique = {(c, v.casefold()): (c, v) for c, v in candidates}
    if len(unique) == 1:
        return next(iter(unique.values()))
    return None

def _detect_membership_check(question: str, df: pd.DataFrame):
    """Answer 'Is there any X in the Y column?' style yes/no questions."""
    m = re.search(
        r"\bis\s+there\s+(?:any|an?)\s+(.+?)\s+in\s+(?:the\s+)?"
        r"([A-Za-z_][A-Za-z0-9_ ]*?)\s+column\b",
        question, re.I,
    )
    if not m:
        return None

    value = m.group(1).strip().strip("'\"")
    col = _find_column(m.group(2), df.columns)
    if col is None or not value:
        return None

    values = df[col].astype(str).str.strip().str.casefold()
    found = values.eq(value.casefold()).any()
    return _text("Yes." if found else "No.")

def _chart_spec(question: str, df: pd.DataFrame):
    """Build a verified chart specification, including filters/top-N metadata."""
    q = str(question).lower()
    if not re.search(r"\b(bar\s*chart|bar\s*plot|scatter|histogram|line\s*(?:chart|plot)?|pie\s*chart|pie|plot|graph|visuali[sz]e|draw|distribution)\b", q):
        return None

    if re.search(r"\b(distribution|histogram)\b", q):
        chart_type = "histogram"
    elif re.search(r"\bscatter\b", q):
        chart_type = "scatter"
    elif re.search(r"\b(?:line|trend)\b", q):
        chart_type = "line"
    elif re.search(r"\bpie\b", q):
        chart_type = "pie"
    else:
        chart_type = "bar"

    metric_col = _metric_column(question, df)
    if chart_type == "histogram":
        metric_col = metric_col or next((c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])), None)
        if metric_col is None:
            return None
        return {"answer_type":"chart","chart":{"type":"histogram","x":str(metric_col),"y":None,
                "title":f"Distribution of {metric_col}"}}

    if chart_type == "scatter":
        vs = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s+(?:vs\.?|versus)\s+([A-Za-z_][A-Za-z0-9_]*)", q)
        if vs:
            x = _find_column(vs.group(1), df.columns); y = _find_column(vs.group(2), df.columns)
        else:
            nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            x, y = (nums[:2] if len(nums) >= 2 else (None, None))
        if x is None or y is None: return None
        return {"answer_type":"chart","chart":{"type":"scatter","x":str(x),"y":str(y),"title":f"{y} vs {x}"}}

    # Explicit multi-dimensional grouping: "by region and category".
    group_cols = _multi_group_columns(question, df)
    if len(group_cols) >= 2 and metric_col is not None:
        filtered = df
        filtered, _ = _date_filter(filtered, question)
        agg = _aggregation(question) or "sum"
        grouped = getattr(filtered.groupby(group_cols, dropna=False)[metric_col], agg)().reset_index()
        return {"answer_type":"chart","chart":{
            "type":chart_type,"x":str(group_cols[0]),"y":str(metric_col),"group_by":str(group_cols[1]),
            "aggregation":agg,"title":f"{agg.title()} {metric_col} by {group_cols[0]} and {group_cols[1]}"
        }}

    # Top-N chart: entity must come from the noun (products), not the metric after "by".
    rank_match = re.search(r"\btop\s+(\d+)\b", q)
    if rank_match and metric_col is not None:
        n = int(rank_match.group(1))
        entity, entity_col = _rank_entity(question, df)
        if entity_col is not None:
            filtered, _ = _date_filter(df, question)
            ranked = filtered.groupby(entity_col, dropna=False)[metric_col].sum().sort_values(ascending=False).head(n)
            return {"answer_type":"chart","chart":{
                "type":chart_type,"x":str(entity_col),"y":str(metric_col),"aggregation":"sum","top_n":n,
                "title":f"Top {n} {entity_col} by {metric_col}"
            }}

    # Monthly/yearly trend charts. "monthly ... for 2024" must filter first.
    date_col = _find_date_column(df)
    if date_col is not None and metric_col is not None and (
        re.search(r"\bmonth|monthly\b", q) or re.search(r"\byear|yearly\b", q)
    ):
        filtered, _ = _date_filter(df, question)
        granularity = "month" if re.search(r"\bmonth|monthly\b", q) else "year"
        return {"answer_type":"chart","chart":{
            "type":chart_type,"x":str(date_col),"y":str(metric_col),"aggregation":_aggregation(question) or "sum",
            "date_granularity":granularity,"date_filter_year":(_extract_years(question)[0] if len(_extract_years(question))==1 else None),
            "title":f"{metric_col} by {granularity}"
        }}

    group_col = _group_column(q, df)
    if group_col is not None and metric_col is not None and group_col != metric_col:
        agg = _aggregation(q) or "sum"
        return {"answer_type":"chart","chart":{"type":chart_type,"x":str(group_col),"y":str(metric_col),"aggregation":agg,
                "title":f"{agg.title()} {metric_col} by {group_col}"}}
    return None



def _contextual_followup(question: str, df: pd.DataFrame, chat_history=None):
    """Resolve the small set of high-value conversational follow-ups deterministically.

    The previous answer is reference context, never a numeric source: all values are
    recomputed from the live dataframe. This prevents phrases such as "which region is
    highest?" from being parsed as a literal category value ("highest"), and lets
    pronouns such as "its" resolve to the entity established immediately before it.
    """
    if not chat_history or not isinstance(chat_history, list):
        return None
    q = str(question).strip()
    ql = q.casefold()
    if not q:
        return None

    # Normalize message history to user/assistant records.
    hist = [m for m in chat_history if isinstance(m, dict)]
    if not hist:
        return None

    def last_user_before(idx=None):
        seq = hist if idx is None else hist[:idx]
        for m in reversed(seq):
            if m.get("question"):
                return str(m.get("question"))
        return ""

    # "Which region/product/category is highest/lowest?" should refer to the
    # grouping requested in the immediately preceding analytical turn.
    rank_m = re.search(
        r"\bwhich\s+(region|regions|category|categories|product|products|sales\s*rep|sales\s*reps)\b"
        r".*\b(highest|lowest|best|worst)\b", ql
    )
    if rank_m:
        noun = rank_m.group(1)
        direction = "lowest" if rank_m.group(2) in {"lowest", "worst"} else "highest"
        # Prefer the most recent user turn that explicitly established the
        # grouping and metric, e.g. "Show revenue by region."
        context_q = ""
        for m in reversed(hist):
            uq = str(m.get("question", ""))
            if uq and re.search(r"\bby\s+(?:the\s+)?" + re.escape(noun.rstrip("s")), uq, re.I):
                context_q = uq
                break
        if not context_q:
            # A prior answer may itself establish the grouping.
            context_q = last_user_before()
        entity_col = _find_column(noun, df.columns)
        metric_col = _metric_column(context_q, df) if context_q else None
        if entity_col is None:
            return None
        if metric_col is None:
            metric_col = _metric_column(q, df)
        if metric_col is None:
            return None
        ranked = df.groupby(entity_col, dropna=False)[metric_col].sum().sort_values(
            ascending=(direction == "lowest")
        )
        if ranked.empty:
            return None
        entity = ranked.index[0]
        value = float(ranked.iloc[0])
        verb = "lowest" if direction == "lowest" else "highest"
        return _text(f"{entity} has the {verb} {metric_col}, at {value:,.2f}.")

    # "Show its/their monthly revenue" resolves the entity from the most recent
    # verified ranking answer, then recomputes the monthly series from df.
    if re.search(r"\b(its|their)\b", ql) and re.search(r"\bmonthly\b|\bby\s+month\b", ql):
        target = None
        target_col = None
        metric_col = _metric_column(q, df)
        # First parse a deterministic ranking answer already stored in history.
        for m in reversed(hist):
            ans = str(m.get("answer", ""))
            mm = re.search(r"^(.+?)\s+has\s+the\s+(?:highest|lowest)\s+([A-Za-z_][A-Za-z0-9_ ]*)\s*,", ans, re.I | re.M)
            if mm:
                candidate = mm.group(1).strip()
                # Identify which categorical column contains that exact value.
                for c in df.columns:
                    if pd.api.types.is_numeric_dtype(df[c]) or c == _find_date_column(df):
                        continue
                    vals = df[c].astype(str).str.strip().str.casefold()
                    if vals.eq(candidate.casefold()).any():
                        target, target_col = candidate, c
                        break
                if target:
                    if metric_col is None:
                        metric_col = mm.group(2).strip()
                    break
        # If the ranking answer was not stored in the expected shape, infer the
        # entity from the previous "which X is highest" turn and recompute it.
        if target is None:
            for m in reversed(hist):
                uq = str(m.get("question", ""))
                mm = re.search(r"\bwhich\s+(region|category|product|sales\s*rep)\b.*\b(highest|lowest)\b", uq, re.I)
                if mm:
                    noun = mm.group(1)
                    target_col = _find_column(noun, df.columns)
                    prev = last_user_before(hist.index(m))
                    metric_col = metric_col or _metric_column(prev, df)
                    if target_col is not None and metric_col is not None:
                        ranked = df.groupby(target_col, dropna=False)[metric_col].sum().sort_values(ascending=mm.group(2).lower() == "lowest")
                        if not ranked.empty:
                            target = str(ranked.index[0])
                    break
        if target is None or target_col is None or metric_col is None:
            return None
        date_col = _find_date_column(df)
        if date_col is None:
            return None
        mask = df[target_col].astype(str).str.strip().str.casefold().eq(target.casefold())
        work = df.loc[mask].copy()
        dates = pd.to_datetime(work[date_col], errors="coerce")
        work = work.assign(__month__=dates.dt.to_period("M"))
        series = work.groupby("__month__", dropna=True)[metric_col].sum().sort_index()
        if series.empty:
            return None
        lines = [f"{idx}: {value:,.2f}" for idx, value in series.items()]
        return _text(f"Monthly {metric_col} for {target}\n" + "\n".join(lines))
    return None

def query_dataframe(question: str, df: pd.DataFrame, chat_history=None):
    """Answer exact dataframe questions directly.

    Return:
      * ``{"answer_type": "text", ...}`` for deterministic text answers;
      * ``{"answer_type": "chart", "chart": {...}}`` for deterministic charts;
      * ``None`` only when the question is not deterministically understood.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return None

    q = str(question).strip()
    ql = q.lower()
    if not q:
        return None

    # =========================================================
    # 0.1 DATA-QUALITY CHECKS — before generic aggregation
    # =========================================================
    duplicate = _duplicate_check(q, df)
    if duplicate is not None:
        return duplicate

    negative = _negative_value_check(q, df)
    if negative is not None:
        return negative

    # =========================================================
    # 0.1 NON-DETERMINISTIC INTENT GUARD
    # =========================================================
    # Forecasting and causal/root-cause reasoning belong to the dedicated
    # forecasting/LLM paths. They must never fall through to the generic
    # aggregation handler, which could otherwise return the grand total.
    if re.search(
        r"^\s*(?:why|how\s+come)\b|\b(?:what\s+(?:caused|causes|explains)|"
        r"reason|driver|drivers|factor|factors)\b",
        ql,
    ):
        return None

    if re.search(
        r"\b(?:next|upcoming|coming)\s+(?:\d+\s+)?"
        r"(?:day|days|week|weeks|month|months|quarter|quarters|year|years)\b|"
        r"\b(?:future|going\s+forward|going\s+ahead)\b|"
        r"\b(?:predict|forecast|forecasting|project|projected|projection|"
        r"estimate|estimated|expected|expect)\b|"
        r"\b(?:what|how\s+much|how\s+many)\b.*\bwill\b",
        ql,
    ):
        return None

    # Explicit calendar years are only future intent when the requested year
    # is after the latest year represented in the live dataframe. Historical
    # year questions must remain available to the deterministic engine.
    future_years = [int(y) for y in re.findall(r"\b20\d{2}\b", q)]
    if future_years:
        date_col = _find_date_column(df)
        if date_col is not None:
            dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
            if not dates.empty and max(future_years) > dates.max().year:
                return None

    # =========================================================
    # 0. CONTEXTUAL FOLLOW-UPS — before literal categorical parsing
    # =========================================================
    contextual = _contextual_followup(q, df, chat_history=chat_history)
    if contextual is not None:
        return contextual

    # =========================================================
    # 0. CHART INTENT — highest priority
    # =========================================================
    # A chart request must never be swallowed by a text/grouping handler.
    chart = _chart_spec(q, df)
    if chart is not None:
        return chart

    # =========================================================
    # 0.2 EXPLICIT COMPARISONS — before generic aggregation
    # =========================================================
    comparison = _comparison_request(q, df)
    if comparison is not None:
        return comparison

    # =========================================================
    # 0.3 MONTH/YEAR TREND — filter first, then aggregate
    # =========================================================
    if re.search(r"\b(monthly|by\s+month|month\s+trend)\b", ql):
        metric_col = _metric_column(q, df)
        date_col = _find_date_column(df)
        if metric_col is not None and date_col is not None:
            filtered, has_date = _date_filter(df, q)
            if has_date or _extract_years(q):
                series = _monthly_group(filtered, metric_col)
                if series is not None:
                    lines = [f"{idx}: {value:,.2f}" for idx, value in series.items()]
                    return _text("Month trend for " + metric_col + "\n" + "\n".join(lines))

    # =========================================================
    # 0.4 MULTI-DIMENSION GROUPING — before single-column grouping
    # =========================================================
    group_cols = _multi_group_columns(q, df)
    if len(group_cols) >= 2:
        metric_col = _metric_column(q, df)
        agg = _aggregation(q) or "sum"
        if metric_col is not None:
            filtered, _ = _date_filter(df, q)
            grouped = getattr(filtered.groupby(group_cols, dropna=False)[metric_col], agg)().sort_values(ascending=False)
            fmt = "{:,.2f}" if agg == "mean" else "{:,.0f}"
            lines = [f"{i}. " + " | ".join(str(v) for v in idx) + f": {fmt.format(value)}" for i, (idx, value) in enumerate(grouped.items(), 1)]
            return _text("\n".join(lines))

        # =========================================================
    # 0.5 MEMBERSHIP CHECK — "is there any X in the Y column?"
    # =========================================================
    membership = _detect_membership_check(q, df)
    if membership is not None:
        return membership

    # =========================================================
    # 1. UNIQUE / DISTINCT VALUES
    # =========================================================
    if re.search(r"\b(unique|distinct)\b", ql):
        col = _find_column(ql, df.columns)
        if col is not None:
            unique_values = df[col].dropna().drop_duplicates()
            if re.search(r"\b(how\s+many|number\s+of|count\s+of|count)\b", ql):
                return _text(f"There are {len(unique_values):,} unique {col} values.")
            formatted = _format_unique_values(unique_values.tolist())
            return _text(formatted if formatted else "No non-empty values found.")

    # =========================================================
    # 2. CUSTOMER COUNT / CUSTOMER LOOKUP — never substitute rows
    # =========================================================
    asks_customer = bool(re.search(r"\b(customer|customers|client|clients)\b", ql))
    customer_col = _find_customer_column(df)
    if asks_customer and re.search(r"\b(how\s+many|number\s+of|count)\b", ql):
        if customer_col is None:
            return _text(
                "I can't determine the number of customers because this dataset "
                "does not contain a customer identifier."
            )
        # Filter first if a product/category/date/numeric constraint is present.
        working = df
        cat = _find_categorical_value(q, df)
        if cat:
            col, value = cat
            working = working.loc[
                working[col].astype(str).str.strip().str.casefold() == value.casefold()
            ]
        numeric_filtered, understood_numeric = _parse_numeric_filter(q, df)
        if understood_numeric:
            working = working.loc[working.index.intersection(numeric_filtered.index)]
        date_filtered, understood_date = _date_filter(working, q)
        if understood_date:
            working = date_filtered
        return _text(f"{working[customer_col].dropna().nunique():,}")

    if asks_customer and re.search(r"\b(show|list|which)\b", ql) and not re.search(
        r"\b(highest|lowest|top|bottom|rank|ranking|most|least)\b", ql
    ):
        if customer_col is None:
            return _text(
                "I can't show the customers because this dataset does not contain "
                "a customer identifier."
            )
        working = df
        cat = _find_categorical_value(q, df)
        if cat:
            col, value = cat
            working = working.loc[
                working[col].astype(str).str.strip().str.casefold() == value.casefold()
            ]
        numeric_filtered, understood_numeric = _parse_numeric_filter(q, df)
        if understood_numeric:
            working = working.loc[working.index.intersection(numeric_filtered.index)]
        return _text(_format_unique_values(working[customer_col].dropna().drop_duplicates().tolist()))

    # =========================================================
    # 3. CATEGORICAL + NUMERIC FILTER
    # =========================================================
    combined_categorical_match = re.search(
        r"\b(?:where\s+)?(?:the\s+)?"
        r"([A-Za-z_][A-Za-z0-9_]*)\s+"
        r"(?:is|equals|equal\s+to|=)\s+"
        r"([^,.;!?]+?)\s+(?:and|or)\s+",
        q, re.I,
    )
    if combined_categorical_match:
        cat_col = _find_column(combined_categorical_match.group(1), df.columns)
        cat_value = combined_categorical_match.group(2).strip()
        if cat_col is not None and not pd.api.types.is_numeric_dtype(df[cat_col]):
            categorical_mask = df[cat_col].astype(str).str.strip().str.casefold().eq(cat_value.casefold())
            numeric_filtered, understood_numeric = _parse_numeric_filter(q, df)
            if understood_numeric:
                combined = df.loc[categorical_mask & df.index.isin(numeric_filtered.index)]
                return _text(_format_rows(combined))

    # =========================================================
    # 4. CATEGORICAL ROW FILTER
    # =========================================================
    categorical_match = re.search(
        r"\b(?:where\s+)?(?:the\s+)?"
        r"([A-Za-z_][A-Za-z0-9_]*)\s+"
        r"(?:is|equals|=|equal\s+to)\s+"
        r"(.+?)(?:[.!?;:]?\s*)$",
        q, re.I,
    )
    if categorical_match:
        col = _find_column(categorical_match.group(1), df.columns)
        if col is not None and not pd.api.types.is_numeric_dtype(df[col]):
            value_text = categorical_match.group(2).strip()
            value_text = re.sub(r"\s+give\s+(?:me\s+)?(?:the\s+)?complete\s+rows?\s*$", "", value_text, flags=re.I).strip()
            values = [
                v.strip().strip("'\".,!?;:")
                for v in re.split(r"\s+\bor\b\s+", value_text, flags=re.I)
                if v.strip()
            ]
            mask = pd.Series(False, index=df.index)
            column_values = df[col].astype(str).str.strip().str.casefold()
            for value in values:
                mask |= column_values.eq(value.casefold())
            filtered = df.loc[mask]
            return _text(
                _format_rows(filtered)
                if not filtered.empty
                else f"No transactions found where {col} is " + " or ".join(values) + "."
            )

    # =========================================================
    # 5. NUMERIC FILTERS
    # =========================================================
    numeric_signals = (
        re.search(r"(?:>=|<=|!=|==|=|>|<)\s*-?\d", ql)
        or re.search(
            r"\b(greater\s+than|more\s+than|above|over|less\s+than|below|under|"
            r"at\s+least|at\s+most|between)\b", ql
        )
    )
    if numeric_signals:
        filtered, understood = _parse_numeric_filter(ql, df)
        if understood:
            return _text(_format_rows(filtered))
        # Parser failure is NOT a zero-row result.
        return None

    # =========================================================
    # 6. DATE + AGGREGATION (before exact-date lookup)
    # =========================================================
    if _has_aggregation_operation(q):
        agg = _aggregation(q)
        metric_col = _metric_column(q, df)
        date_col = _find_date_column(df)
        if agg and metric_col is not None and date_col is not None:
            filtered, has_date = _date_filter(df, q)
            if has_date:
                value = getattr(filtered[metric_col], agg)()
                return _text(f"{value:,.2f}" if agg == "mean" else f"{value:,.0f}")

    # =========================================================
    # 7. HIGHEST / LOWEST ENTITY-AWARE QUERIES
    # =========================================================
    if re.search(r"\b(highest|maximum|max|largest|most|best)\b", ql):
        metric_col = _metric_column(q, df)
        entity, entity_col = _rank_entity(q, df)
        if metric_col is not None:
            if entity == "transaction":
                row = df.loc[df[metric_col].idxmax()]
                return _text(row.to_string())
            if entity == "customer":
                if entity_col is None:
                    return _text(
                        "I can't determine the highest-revenue customer because this "
                        "dataset does not contain a customer identifier."
                    )
                ranked = df.groupby(entity_col, dropna=False)[metric_col].sum().sort_values(ascending=False)
                return _text(f"{ranked.index[0]} has the highest {metric_col}, at {ranked.iloc[0]:,.2f}.")
            if entity_col is not None:
                if entity == "date":
                    dates = pd.to_datetime(df[entity_col], errors="coerce")
                    temp = df.assign(__date__=dates.dt.normalize())
                    ranked = temp.groupby("__date__", dropna=False)[metric_col].sum().sort_values(ascending=False)
                else:
                    ranked = df.groupby(entity_col, dropna=False)[metric_col].sum().sort_values(ascending=False)
                return _text(f"{ranked.index[0]} has the highest {metric_col}, at {ranked.iloc[0]:,.2f}.")

    if re.search(r"\b(lowest|minimum|min|smallest|least)\b", ql):
        metric_col = _metric_column(q, df)
        entity, entity_col = _rank_entity(q, df)
        if metric_col is not None:
            if entity == "transaction":
                row = df.loc[df[metric_col].idxmin()]
                return _text(row.to_string())
            if entity == "customer" and entity_col is None:
                return _text(
                    "I can't determine the lowest-revenue customer because this dataset "
                    "does not contain a customer identifier."
                )
            if entity_col is not None:
                if entity == "date":
                    dates = pd.to_datetime(df[entity_col], errors="coerce")
                    temp = df.assign(__date__=dates.dt.normalize())
                    ranked = temp.groupby("__date__", dropna=False)[metric_col].sum().sort_values()
                else:
                    ranked = df.groupby(entity_col, dropna=False)[metric_col].sum().sort_values()
                return _text(f"{ranked.index[0]} has the lowest {metric_col}, at {ranked.iloc[0]:,.2f}.")

    # =========================================================
    # 8. TOP/BOTTOM N — entity-aware grouped ranking
    # =========================================================
    rank_match = re.search(r"\b(?:top|best|bottom|worst)\s+(\d+)\b", ql)
    if rank_match or re.search(r"\b(products?|customers?|dates?)\s+with\s+(?:the\s+)?(?:highest|most)\b", ql):
        n = int(rank_match.group(1)) if rank_match else 5
        direction = "top"
        if rank_match and re.search(r"\b(bottom|worst)\b", ql):
            direction = "bottom"
        metric_col = _metric_column(q, df)
        entity, entity_col = _rank_entity(q, df)

        if metric_col is not None and entity is not None:
            if entity == "transaction":
                result = df.sort_values(metric_col, ascending=(direction == "bottom")).head(n)
                return _text(_format_rows(result))
            if entity == "customer" and entity_col is None:
                return _text(
                    "I can't rank customers because this dataset does not contain "
                    "a customer identifier."
                )
            if entity == "date":
                dates = pd.to_datetime(df[entity_col], errors="coerce").dt.normalize()
                temp = df.assign(__date__=dates)
                ranked = temp.groupby("__date__", dropna=False)[metric_col].sum()
            else:
                ranked = df.groupby(entity_col, dropna=False)[metric_col].sum()
            ranked = ranked.sort_values(ascending=(direction == "bottom")).head(n)
            lines = []
            if len(ranked) < n:
                lines.append(
                    f"This dataset only has {len(ranked)} {entity_col} value"
                    f"{'s' if len(ranked) != 1 else ''} — showing all {len(ranked)}:"
                )
            lines += [f"{i}. {idx}: {value:,.2f}" for i, (idx, value) in enumerate(ranked.items(), 1)]
            return _text("\n".join(lines))

    # "Rank products by revenue" has no top-N keyword.
    if re.search(r"\b(rank|ranking|ranked)\b", ql):
        entity, entity_col = _rank_entity(q, df)
        metric_col = _metric_column(q, df)
        if entity == "customer" and entity_col is None:
            return _text(
                "I can't rank customers because this dataset does not contain "
                "a customer identifier."
            )
        if entity_col is not None and metric_col is not None:
            ranked = df.groupby(entity_col, dropna=False)[metric_col].sum().sort_values(ascending=False)
            lines = [f"{i}. {idx}: {value:,.2f}" for i, (idx, value) in enumerate(ranked.items(), 1)]
            return _text("\n".join(lines))

    # =========================================================
    # 9. GROUPED AGGREGATION — by / for each / per / each
    # =========================================================
    group_col = _group_column(q, df)
    agg = _aggregation(q)
    metric_col = _metric_column(q, df)
    if group_col is not None and metric_col is not None and agg is not None and group_col != metric_col:
        # If this is a date group, group by normalized dates for consistent
        # date-only semantics.
        if group_col == _find_date_column(df):
            dates = pd.to_datetime(df[group_col], errors="coerce").dt.normalize()
            grouped = getattr(df.assign(__date__=dates).groupby("__date__", dropna=False)[metric_col], agg)()
        else:
            grouped = getattr(df.groupby(group_col, dropna=False)[metric_col], agg)()
        grouped = grouped.sort_index()
        fmt = "{:,.2f}" if agg == "mean" else "{:,.0f}"
        lines = [f"{index}: {fmt.format(value)}" for index, value in grouped.items()]
        return _text("\n".join(lines))

    # =========================================================
    # 10. FILTERED AGGREGATION — metric + aggregation + categorical/date filter
    # =========================================================
    # A question that asks "which / who / highest / how many / top ..." but was
    # not understood above must NOT be answered with an overall total - that
    # would be a confident, wrong answer. Fall through (and let the LLM path
    # compute it from the data) instead.
    _asks_structure = re.search(
        r"\b(which|who|whom|highest|lowest|most|least|top|bottom|best|worst|rank|ranking|"
        r"how\s+many|number\s+of|count|trend|growth|change|increase|decrease|decline|over\s+time)\b", ql)
    if agg is not None and metric_col is not None and not _asks_structure:
        working = df

        cat = _find_categorical_value(q, df)
        if cat:
            col, value = cat
            working = working.loc[
                working[col].astype(str).str.strip().str.casefold() == value.casefold()
            ]

        numeric_filtered, understood_numeric = _parse_numeric_filter(q, df)
        if understood_numeric:
            working = working.loc[working.index.intersection(numeric_filtered.index)]

        date_filtered, understood_date = _date_filter(working, q)
        if understood_date:
            working = date_filtered

        # Only apply this deterministic aggregate when a meaningful filter
        # is actually present, or when it is a plain overall aggregate.
        has_filter_language = bool(cat or understood_numeric or understood_date)
        if has_filter_language or not re.search(r"\b(by|per|each)\b", ql):
            value = getattr(working[metric_col], agg)()
            return _text(f"{value:,.2f}" if agg == "mean" else f"{value:,.0f}")

    # =========================================================
    # 11. EXACT DATE ROW LOOKUP — low priority, after aggregation
    # =========================================================
    exact_filtered, has_exact_date = _date_filter(df, q)
    if has_exact_date:
        if len(exact_filtered):
            return _text(_format_rows(exact_filtered))
        return _text("No matching rows found for the requested date.")

    # =========================================================
    # 12. UNIQUE / CATEGORICAL COUNTS without explicit "unique"
    # =========================================================
    if re.search(r"\bhow\s+many\b|\bnumber\s+of\b|\bcount\b", ql):
        # Explicit rows/records/transactions remain row counts.
        if re.search(r"\b(rows?|records?|transactions?)\b", ql):
            return _text(f"{len(df):,}")

        # For a named categorical entity, COUNT DISTINCT is the semantic
        # default. Customer has already been handled above.
        for canonical in ("product", "region", "date"):
            if _word_in_text(canonical, ql) or _word_in_text(canonical + "s", ql):
                if canonical == "date":
                    col = _find_date_column(df)
                    if col is not None:
                        values = pd.to_datetime(df[col], errors="coerce").dt.normalize().dropna()
                        return _text(f"{values.nunique():,}")
                else:
                    col = _find_column(canonical, df.columns)
                    if col is not None:
                        return _text(f"{df[col].dropna().nunique():,}")

        # Any other single categorical column ("how many sales reps are there?").
        # Not for "... per/by/in each X" questions, which ask for counts per group.
        if not re.search(r"\b(per|each|by|in|for|with|where|from|than|having)\b", ql):
            _dc = _find_date_column(df)
            hits = [c for c in df.columns
                    if c != _dc and not pd.api.types.is_numeric_dtype(df[c])
                    and _find_column(ql, [c]) is not None]
            if len(hits) == 1:
                return _text(f"{df[hits[0]].dropna().nunique():,}")

    # "What are the unique dates?" without the word count.
    if re.search(r"\b(unique|distinct)\b", ql):
        col = _find_column(ql, df.columns)
        if col is not None:
            return _text(_format_unique_values(df[col].dropna().drop_duplicates().tolist()))

    # No deterministic match — let the LLM handle it.
    return None
