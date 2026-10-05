"""Regression tests for Axiomrow deterministic forecasting (backend/forecasting.py).

These replace the older tests that called query_dataframe() and expected a
forecast back: forecasting now lives in forecasting.forecast_dataframe(), and
query_dataframe() must never answer a future-looking question with a total."""
import numpy as np
import pandas as pd

from backend.forecasting import forecast_dataframe, is_forecast_request
from backend.query_engine import query_dataframe


def _monthly(n=36, start="2023-01-01"):
    dates = pd.date_range(start, periods=n, freq="MS")
    season = np.array([0, 50, 100, 40, -20, 80, 120, 60, -10, 30, 90, 140], dtype=float)
    revenue = 1000 + np.arange(n) * 25 + np.tile(season, n // 12 + 1)[:n]
    return pd.DataFrame({"date": dates, "revenue": revenue})


def _daily(days=900):
    d = pd.date_range("2023-01-01", periods=days, freq="D")
    return pd.DataFrame({"date": d, "revenue": 100.0 + (d.month * 10)})


def test_next_months_forecast():
    r = forecast_dataframe("predict next 3 months revenue", _monthly())
    assert r and r["verified"] is True and r["answer_type"] == "forecast"
    fc = r["forecast"]
    assert fc["horizon"] == 3 and len(fc["values"]) == len(fc["lower"]) == len(fc["upper"]) == 3
    assert fc["validation"]["mae"] is not None
    assert "not guaranteed" in r["content"]


def test_target_year_returns_only_target_year():
    r = forecast_dataframe("forecast revenue for 2027", _monthly())
    assert r and r["verified"] is True
    assert r["forecast"]["horizon"] == 12
    assert all(str(x["period"]).startswith("2027-") for x in r["forecast"]["values"])


def test_missing_date_is_refused():
    df = pd.DataFrame({"revenue": [1, 2, 3, 4, 5], "product": list("abcde")})
    r = forecast_dataframe("predict next month revenue", df)
    assert r and r["answer_type"] == "text" and r["verified"] is False
    assert "date" in r["content"].lower()


def test_non_forecast_query_still_works():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=12, freq="MS"), "revenue": np.arange(12) + 100})
    assert forecast_dataframe("what is the total revenue", df) is None
    assert query_dataframe("what is the total revenue", df) is not None


def test_query_engine_never_answers_a_future_question_with_a_total():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=12, freq="MS"), "revenue": np.arange(12) + 100})
    total = f"{int(df['revenue'].sum()):,}"
    r = query_dataframe("predict next month revenue", df)
    assert r is None or total not in str(r.get("content", ""))


def test_profit_is_not_silently_replaced_by_profit_margin():
    df = _monthly().assign(profit_margin=0.2)
    r = forecast_dataframe("What will profit be next month?", df)
    assert r["answer_type"] == "text" and "won't substitute" in r["content"]


def test_past_year_questions_are_not_forecast_requests():
    df = _daily()                                    # data covers all of 2023 and 2024 and part of 2025
    assert not is_forecast_request("What was total revenue for 2024?", df)
    assert not is_forecast_request("Show monthly revenue for 2024 as a line chart", df)
    assert is_forecast_request("Estimate revenue for 2027", df)
    assert is_forecast_request("revenue for 2030", df)


def test_incomplete_last_month_is_left_out_of_the_model():
    df = _daily(900)                                 # ends 2025-06-18: June 2025 is partial
    r = forecast_dataframe("What will revenue be next month?", df)
    fc = r["forecast"]
    assert fc["partial_period_excluded"] == "June 2025"
    assert fc["historical"][-1]["period"] == "2025-05-01"
    assert fc["values"][0]["period"] == "2025-07-01"          # still the month after the data ends
    assert "incomplete in the data" in r["content"]


def test_incomplete_quarter_is_not_repeated_as_the_next_quarter():
    df = _daily(900)
    r = forecast_dataframe("What will sales be next quarter?", df.rename(columns={"revenue": "revenue"}))
    full_q = df.groupby(df["date"].dt.to_period("Q"))["revenue"].sum().iloc[:-1]
    assert abs(r["forecast"]["values"][0]["value"] / full_q.tail(4).mean() - 1) < 0.15


def test_current_partial_year_can_still_be_estimated():
    df = _daily(900)
    r = forecast_dataframe("Estimate revenue for 2025", df)
    assert r["answer_type"] == "forecast"
    assert [x["period"][:7] for x in r["forecast"]["values"]] == [f"2025-{m:02d}" for m in range(7, 13)]


def test_fully_observed_year_is_refused_politely():
    r = forecast_dataframe("Estimate revenue for 2024", _daily(900))
    assert r["answer_type"] == "text" and "already covered" in r["content"]


def test_one_row_per_month_data_is_never_treated_as_partial():
    r = forecast_dataframe("What will revenue be next month?", _monthly())
    assert r["forecast"]["partial_period_excluded"] is None


# ---- regressions found reviewing the "trim partial period before resampling" version ----
def test_seasonal_estimates_line_up_with_their_month_labels():
    """Data ends 18 Jun 2025 (June incomplete). The 'July 2025' estimate of a purely seasonal
    series must equal July 2024, not June 2024 (labels were one month ahead of the values)."""
    d = pd.date_range("2023-01-01", "2025-06-18", freq="D")
    month_level = {m: 100.0 * m for m in range(1, 13)}               # strongly seasonal, same every year
    df = pd.DataFrame({"date": d, "revenue": [month_level[x.month] for x in d]})
    r = forecast_dataframe("What will revenue be for the next 3 months?", df)
    v = r["forecast"]["values"]
    assert [x["period"][:7] for x in v] == ["2025-07", "2025-08", "2025-09"]
    per_day = [month_level[7], month_level[8], month_level[9]]
    for row, level, days in zip(v, per_day, (31, 31, 30)):
        assert abs(row["value"] - level * days) < 1e-6, row


def test_plain_questions_about_the_partly_observed_current_year_are_not_forecasts():
    df = _daily(900)                                  # ends 18 Jun 2025
    for q in ("What was total revenue in 2025?", "Show revenue for 2025",
              "Show monthly revenue for 2025 as a line chart", "How many orders were there in 2025?"):
        assert not is_forecast_request(q, df), q
    for q in ("Estimate revenue for 2025", "What will revenue be in 2025?", "Forecast revenue for 2025"):
        assert is_forecast_request(q, df), q


def test_suggested_negative_value_question_is_still_answered_deterministically():
    df = pd.DataFrame({"profit_margin": [0.2, -0.1, 0.3, -0.2], "date": pd.date_range("2024-01-01", periods=4)})
    r = query_dataframe("How many profit_margin values are negative, and is that expected?", df)
    assert r is not None and "2" in r["content"]
