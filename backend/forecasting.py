"""Deterministic, backtested time-series forecasting for Axiomrow."""
import re
from typing import Optional

import numpy as np
import pandas as pd

_FUTURE_PATTERNS = (
    r"\bnext\s+(?:\d+\s+)?(?:day|days|week|weeks|month|months|quarter|quarters|year|years)\b",
    r"\b(?:upcoming|coming)\s+(?:\d+\s+)?(?:day|days|week|weeks|month|months|quarter|quarters|year|years)\b",
    r"\b(?:future|going\s+forward|going\s+ahead)\b",
    r"\bin\s+\d+\s+(?:day|days|week|weeks|month|months|quarter|quarters|year|years)\b",
    r"\b(?:for|during)\s+(?:20\d{2})\b",
    r"\b(?:expected|expect|estimate|estimated|project|projected|projection|predict|predicted|prediction)\b",
    r"\b(?:what|how\s+much|how\s+many)\b.*\b(?:will|can\s+we\s+expect|should\s+we\s+expect)\b",
)
_PAST_PATTERNS = (
    r"\b(?:last|previous|prior)\s+(?:day|week|month|quarter|year)\b",
    r"\b(?:was|were)\b.*\b(?:last|previous|prior)\b",
)


def is_forecast_request(question: str, df: pd.DataFrame) -> bool:
    if not question or df is None or df.empty:
        return False
    q = str(question).strip().lower()
    if any(re.search(p, q) for p in _PAST_PATTERNS):
        return False

    # A calendar year mentioned in the question is only a forecast request
    # when it is after the latest year in the dataset. The current year can
    # still be a forecast target when the dataset ends part-way through it.
    year_match = re.search(r"\b(?:for|during|in)\s+(20\d{2})\b", q)
    if year_match:
        target_year = int(year_match.group(1))
        date_col = _find_date_column(df)
        if date_col is None:
            return True
        dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if dates.empty:
            return True
        max_date = dates.max()
        forecast_words = re.search(
            r"\b(?:predict|forecast|forecasting|estimate|estimated|project|projected|"
            r"projection|expected|expect|will)\b",
            q,
        )
        if target_year < max_date.year:
            # Explicit forecasting language is still a forecast request even
            # when the requested year is historical; forecast_dataframe() will
            # refuse it with a clear "already covered" message.
            return bool(forecast_words)
        if target_year == max_date.year:
            # The current (possibly part-observed) year is a forecast request only
            # with explicit forecast language ("estimate/predict/will..."). A plain
            # "revenue in 2025" is a historical question about the data we DO have.
            return bool(forecast_words)
        return True

    return any(re.search(p, q) for p in _FUTURE_PATTERNS)


def _find_date_column(df):
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    hints = ("date", "datetime", "timestamp", "time", "month", "year")
    for c in df.columns:
        if any(h in str(c).lower() for h in hints):
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() >= 0.8:
                return c
    return None


def _find_metric_column(question, df):
    """Resolve the requested business metric without silently changing meaning."""
    q = str(question).lower()
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        return None

    # 1. Exact column-name mention always wins.
    mentioned = [
        c for c in numeric
        if re.search(r"(?<!\w)" + re.escape(str(c).lower()) + r"(?!\w)", q)
    ]
    if mentioned:
        return mentioned[0]

    # 2. Business synonyms. These are semantic mappings, not substring
    #    matching, so "sales" can resolve to "revenue" while "profit" cannot
    #    silently resolve to "profit_margin".
    semantic_aliases = {
        "revenue": ("revenue", "sales", "sale", "turnover"),
        "profit": ("profit",),
        "quantity": ("quantity", "qty", "units"),
        "amount": ("amount",),
    }

    for canonical, words in semantic_aliases.items():
        if not any(re.search(r"(?<!\w)" + re.escape(w) + r"(?!\w)", q) for w in words):
            continue

        # Prefer a true semantic column.
        candidates = []
        for c in numeric:
            name = str(c).lower().replace("_", " ")
            if canonical == "revenue":
                if any(token in name for token in ("revenue", "sales", "sale", "turnover")):
                    candidates.append(c)
            elif canonical == "profit":
                # Explicitly exclude margin/rate/percentage columns.
                if "profit" in name and not any(x in name for x in ("margin", "rate", "percent", "%")):
                    candidates.append(c)
            elif canonical == "quantity":
                if any(token in name for token in ("quantity", "qty", "unit")):
                    candidates.append(c)
            elif canonical == "amount" and "amount" in name:
                candidates.append(c)

        if candidates:
            return candidates[0]

        # A request for "profit" must NOT become "profit_margin".
        if canonical == "profit":
            return None

        # "sales" is a normal business synonym for revenue. If the dataset
        # has exactly one revenue-like numeric field, resolve it explicitly.
        if canonical == "revenue":
            revenue_like = [
                c for c in numeric
                if "revenue" in str(c).lower() or "sales" in str(c).lower()
            ]
            if len(revenue_like) == 1:
                return revenue_like[0]

            # Common transaction datasets may name the metric simply
            # "unit_price", but we must not guess that this is revenue.
            return None

    # 3. If the question contains a numeric column name we handled it above.
    #    Only infer automatically when there is exactly one numeric column.
    return numeric[0] if len(numeric) == 1 else None


def _frequency(question):
    q = str(question).lower()
    if re.search(r"\b(?:day|daily|days)\b", q): return "D"
    if re.search(r"\b(?:week|weekly|weeks)\b", q): return "W"
    if re.search(r"\b(?:quarter|quarterly|quarters)\b", q): return "QE"
    if re.search(r"\b(?:year|yearly|annual|years)\b", q): return "Y"
    return "MS"


def _parse_horizon(question, freq, max_date):
    """Return (horizon, frequency) using business-friendly future periods."""
    q = str(question).lower()

    unit = {
        "day": "D", "days": "D",
        "week": "W", "weeks": "W",
        "month": "MS", "months": "MS",
        "quarter": "QE", "quarters": "QE",
        "year": "Y", "years": "Y",
    }

    # Explicit next/upcoming periods.
    for pattern in (
        r"\bnext\s+(?:(\d+)\s+)?(day|days|week|weeks|month|months|quarter|quarters|year|years)\b",
        r"\b(?:upcoming|coming)\s+(?:(\d+)\s+)?(day|days|week|weeks|month|months|quarter|quarters|year|years)\b",
        r"\bin\s+(\d+)\s+(day|days|week|weeks|month|months|quarter|quarters|year|years)\b",
    ):
        m = re.search(pattern, q)
        if m:
            n = int(m.group(1) or 1)
            requested = unit[m.group(2)]

            # "next year" means the next 12 monthly periods for a business
            # revenue/sales forecast. This uses the available monthly history
            # instead of requiring multiple annual observations.
            if requested == "Y" and n == 1:
                return 12, "MS"

            return n, requested

    # A specific future calendar year means the complete target year. For
    # business metrics, keep monthly frequency so 2027 produces Jan-Dec 2027.
    m = re.search(r"\b(?:for|during|in)\s+(20\d{2})\b", q)
    if m:
        target = int(m.group(1))
        if target <= max_date.year:
            return 0, freq

        if freq == "Y":
            return target - max_date.year, "Y"

        if freq == "QE":
            return (target - max_date.year) * 4, "QE"

        if freq == "MS":
            months = (target - max_date.year) * 12 - max_date.month + 12
            return max(1, months), "MS"

        if freq == "W":
            target_end = pd.Timestamp(target, 12, 31)
            return max(1, int(np.ceil((target_end - max_date).days / 7))), "W"

        target_end = pd.Timestamp(target, 12, 31)
        return max(1, int((target_end - max_date).days)), "D"

    return 1, freq


def _future_index(last_date, freq, horizon):
    if freq == "D":
        return pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    if freq == "W":
        return pd.date_range(last_date + pd.Timedelta(weeks=1), periods=horizon, freq="W")
    if freq == "QE":
        start = last_date.to_period("Q").end_time + pd.Timedelta(nanoseconds=1)
        return pd.date_range(start, periods=horizon, freq="QE")
    if freq == "Y":
        return pd.date_range(pd.Timestamp(last_date.year + 1, 12, 31), periods=horizon, freq="YE")
    start = last_date.to_period("M").to_timestamp() + pd.offsets.MonthBegin(1)
    return pd.date_range(start, periods=horizon, freq="MS")


def _seasonal_naive(y, horizon, season):
    if not season or len(y) < season * 2:
        return None
    return np.resize(y[-season:], horizon).astype(float)


def _linear(y, horizon):
    x = np.arange(len(y), dtype=float)
    coef = np.polyfit(x, y, 1)
    return np.polyval(coef, np.arange(len(y), len(y) + horizon))


def _evaluate(y, season):
    holdout = max(1, min(max(3, len(y)//5), len(y)//2))
    train, test = y[:-holdout], y[-holdout:]
    if len(train) < 3:
        return None
    candidates = {"naive": np.repeat(train[-1], holdout), "linear_trend": _linear(train, holdout)}
    seasonal = _seasonal_naive(train, holdout, season) if season else None
    if seasonal is not None:
        candidates["seasonal_naive"] = seasonal
    scores = {}
    for name, pred in candidates.items():
        err = test - pred
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        denom = np.where(np.abs(test) > 1e-9, np.abs(test), np.nan)
        mape = float(np.nanmean(np.abs(err) / denom) * 100) if np.isfinite(denom).any() else None
        scores[name] = {"mae": mae, "rmse": rmse, "mape": mape}
    best = min(scores, key=lambda k: scores[k]["mae"])
    return best, scores[best]


def _fit(y, method, horizon, season):
    if method == "naive": return np.repeat(y[-1], horizon).astype(float)
    if method == "seasonal_naive": return _seasonal_naive(y, horizon, season)
    return _linear(y, horizon)


def _text_result(content):
    return {"answer_type": "text", "verified": False, "content": content}



def _latest_period_partial(work: pd.DataFrame, freq: str):
    """Return (is_partial, label) for the latest calendar period.

    A period is considered partial only when the raw dataset has multiple
    observations inside that period and its latest observed date is before
    the calendar period end. One-row-per-period data is treated as complete.
    """
    if work.empty or freq not in {"MS", "QE", "Y"}:
        return False, None

    dates = pd.to_datetime(work["date"], errors="coerce").dropna()
    if dates.empty:
        return False, None

    latest_date = dates.max()
    period = latest_date.to_period({"MS": "M", "QE": "Q", "Y": "Y"}[freq])
    in_period = dates[dates.dt.to_period({"MS": "M", "QE": "Q", "Y": "Y"}[freq]) == period]

    # One observation for a period (e.g. one row per month) is a complete
    # business-period observation, not evidence of a partial calendar period.
    if in_period.nunique() <= 1:
        return False, None

    period_end = period.end_time.normalize()
    if latest_date.normalize() >= period_end:
        return False, None

    label = str(period)
    if freq == "MS":
        label = latest_date.strftime("%B %Y")
    elif freq == "QE":
        label = f"Q{period.quarter} {period.year}"
    else:
        label = str(period.year)
    return True, label


def _trim_partial_period(work: pd.DataFrame, freq: str):
    """Remove the incomplete latest period from raw history before resampling."""
    partial, label = _latest_period_partial(work, freq)
    if not partial:
        return work, None

    dates = pd.to_datetime(work["date"], errors="coerce")
    period_code = {"MS": "M", "QE": "Q", "Y": "Y"}[freq]
    latest_period = dates.max().to_period(period_code)
    mask = dates.dt.to_period(period_code) != latest_period
    return work.loc[mask].copy(), label


def forecast_dataframe(question: str, df: pd.DataFrame) -> Optional[dict]:
    if not is_forecast_request(question, df):
        return None
    date_col = _find_date_column(df)
    metric_col = _find_metric_column(question, df)
    if date_col is None:
        return _text_result("I can estimate future values only when the dataset contains a usable date or time column.")
    if metric_col is None:
        q_lower = str(question).lower()
        numeric_names = [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if re.search(r"\bprofit\b", q_lower) and not any(
            "profit" in name.lower() and not any(x in name.lower() for x in ("margin", "rate", "percent", "%"))
            for name in numeric_names
        ):
            margin_cols = [name for name in numeric_names if "profit_margin" in name.lower() or "profit margin" in name.lower()]
            if margin_cols:
                return _text_result(
                    f"The dataset does not contain a `profit` column. It contains "
                    f"`{margin_cols[0]}`, which is profit margin and is a different metric. "
                    f"I won't substitute profit margin for profit."
                )
        return _text_result("I can estimate a future metric, but I need you to specify which numeric metric to project.")

    work = pd.DataFrame({"date": pd.to_datetime(df[date_col], errors="coerce"), "value": pd.to_numeric(df[metric_col], errors="coerce")}).dropna()
    work = work.groupby("date", as_index=False)["value"].sum().sort_values("date")
    if len(work) < 8:
        return _text_result(f"There are only {len(work):,} usable time points for {metric_col}. I need more historical observations before producing a validated estimate.")

    freq = _frequency(question)
    q_lower = str(question).lower()
    target_year_match = re.search(r"\b(?:for|during|in)\s+(20\d{2})\b", q_lower)
    target_year = int(target_year_match.group(1)) if target_year_match else None

    # For "next year" and a specific future calendar year, use monthly history
    # so the model can learn monthly business patterns.
    if re.search(r"\bnext\s+(?:year|years)\b", q_lower) or target_year is not None:
        freq = "MS"

    # Exclude an incomplete latest calendar period from model history. This
    # matters for daily/transaction-level data, where the latest month or
    # quarter can contain only part of the period. Do not exclude one-row-per-
    # period datasets merely because the row is dated at period start.
    model_work, partial_period_excluded = _trim_partial_period(work, freq)

    series = model_work.set_index("date")["value"].resample(freq).sum().dropna()
    max_data_date = work["date"].max()

    # For a target equal to the current partial year, forecast only the
    # remaining calendar months after the latest observed month. The model
    # history ends at the last complete month, but the forecast starts after
    # the actual data end (e.g. data through June 18 -> forecast July-Dec).
    if target_year is not None and target_year == max_data_date.year and partial_period_excluded:
        horizon = 12 - max_data_date.month
        requested_freq = "MS"
    else:
        horizon, requested_freq = _parse_horizon(question, freq, series.index.max())

    # For a target equal to the current partial year, forecast only the
    # remaining calendar months. For a fully covered historical year, refuse.
    if target_year is not None:
        if target_year < max_data_date.year:
            return _text_result("The requested year is already covered by the dataset, so I won't treat it as a future estimate.")
        if target_year == max_data_date.year and partial_period_excluded is None:
            # A year-end dataset has no remaining periods to estimate.
            if max_data_date.normalize() >= pd.Timestamp(target_year, 12, 31):
                return _text_result("The requested year is already covered by the dataset, so I won't treat it as a future estimate.")

    if horizon <= 0:
        return _text_result("The requested year is already covered by the dataset, so I won't treat it as a future estimate.")
    freq = requested_freq

    # If the requested frequency changes, re-run partial-period exclusion at
    # that frequency so quarterly forecasts do not train on an incomplete Q.
    model_work, partial_period_excluded_2 = _trim_partial_period(work, freq)
    partial_period_excluded = partial_period_excluded_2 or partial_period_excluded
    series = model_work.set_index("date")["value"].resample(freq).sum().dropna()
    if len(series) < 8:
        return _text_result(f"There is not enough historical {metric_col} data at the requested frequency to produce a validated estimate.")

    season = {"D":7, "W":52, "MS":12, "QE":4, "Y":None}.get(freq)
    validation = _evaluate(series.to_numpy(dtype=float), season)
    if validation is None:
        return _text_result("There is not enough history to backtest a future estimate reliably.")
    method, metrics = validation
    # Forecast periods begin after the actual last observed date when a
    # partial period was excluded; otherwise the model's latest historical
    # period is the correct anchor.
    forecast_anchor = work["date"].max() if partial_period_excluded else series.index.max()
    future_idx = _future_index(forecast_anchor, freq, horizon)
    # When the incomplete latest period was left out of the model history, the
    # model's first predicted step is that incomplete period, but the labels in
    # future_idx start AFTER it. Predict the gap too and drop it so every value
    # lines up with its label (otherwise "July" gets June's seasonal value).
    gap = max(0, len(pd.date_range(series.index.max(), future_idx[0], freq=freq)) - 2)
    pred = _fit(series.to_numpy(dtype=float), method, horizon + gap, season)
    pred = pred[gap:] if pred is not None else None
    if pred is None or len(pred) != horizon or not np.isfinite(pred).all():
        return _text_result("I couldn't produce a stable estimate from the available history.")

    band = 1.96 * max(metrics["mae"], float(np.std(np.diff(series.to_numpy(dtype=float)))) if len(series) > 2 else 0.0)
    all_rows = [
        {
            "period": pd.Timestamp(d).strftime("%Y-%m-%d"),
            "value": float(v),
            "lower": float(v - band),
            "upper": float(v + band),
        }
        for d, v in zip(future_idx, pred)
    ]

    # A request such as "estimate revenue for 2027" should display only the
    # requested calendar year, even though the model may need to generate the
    # intervening periods internally.
    if target_year is not None:
        rows = [
            r for r in all_rows
            if int(r["period"][:4]) == target_year
        ]
        if not rows:
            return _text_result(f"I couldn't produce future estimates for {target_year}.")
    else:
        rows = all_rows
    history = [{"period": pd.Timestamp(d).strftime("%Y-%m-%d"), "value": float(v)} for d,v in series.tail(min(24,len(series))).items()]
    period_name = {"MS":"month","QE":"quarter","Y":"year","W":"week","D":"day"}.get(freq,"period")
    if target_year is not None:
        lines = [f"Estimated {metric_col} for {target_year}:"]
    else:
        lines = [f"Estimated {metric_col} for the next {horizon} {period_name}{'' if horizon==1 else 's'}:"]
    if partial_period_excluded:
        lines.append(
            f"Note: {partial_period_excluded} was incomplete in the data and was excluded from the model history."
        )
    lines += [f"{r['period']}: {r['value']:,.2f}" for r in rows]
    lines += [f"Method: {method.replace('_',' ')}", f"Backtest MAE: {metrics['mae']:,.2f}"]
    if metrics.get("mape") is not None: lines.append(f"Backtest MAPE: {metrics['mape']:.2f}%")
    lines.append("These are model-based estimates from historical data, not guaranteed future values.")

    # IMPORTANT: keep the original `future` payload for compatibility, but
    # also expose the chart renderer's expected `values/lower/upper` schema.
    # Previously the UI received `future` only, so it incorrectly displayed
    # "Forecast data is incomplete" even though a valid forecast existed.
    lower = [{"period": r["period"], "value": r["lower"]} for r in rows]
    upper = [{"period": r["period"], "value": r["upper"]} for r in rows]
    values = [{"period": r["period"], "value": r["value"]} for r in rows]
    return {
        "answer_type": "forecast",
        "verified": True,
        "content": "\n".join(lines),
        "forecast": {
            "metric": str(metric_col),
            "date_column": str(date_col),
            "frequency": freq,
            "horizon": len(rows) if target_year is not None else horizon,
            "method": method,
            "model": method,
            "validation": metrics,
            "historical": history,
            "future": rows,
            "values": values,
            "lower": lower,
            "upper": upper,
            "band": float(band),
            "partial_period_excluded": partial_period_excluded,
        },
    }
