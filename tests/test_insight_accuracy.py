"""Regression tests for the misleading AI-insight cards (outlier / 'double' claims)."""
import numpy as np
import pandas as pd
import pytest

import backend.insight_generator as ig


@pytest.fixture
def uniform_df():
    rng = np.random.default_rng(1)
    n = 5000
    return pd.DataFrame({
        "revenue": rng.uniform(10, 50000, n).round(2),
        "discount_pct": rng.uniform(0, .4, n).round(3),
        "shipping_days": rng.integers(1, 14, n),
        "region": rng.choice(["North", "South", "East", "West"], n),
    })


def _run(monkeypatch, df, reply):
    monkeypatch.setattr(ig, "call_llm", lambda m, temperature=0.4, session_id=None: reply)
    return ig.generate_insights(df)


def test_outlier_in_title_is_rejected_when_no_outliers_exist(monkeypatch, uniform_df):
    reply = ("Shipping Delay Outlier\nFinding: Shipping days reach 13.\n"
             "Evidence: shipping_days range 1.00 to 13.00.\nInterpretation: this may indicate delays.\n")
    titles = [i["title"] for i in _run(monkeypatch, uniform_df, reply)]
    assert "Shipping Delay Outlier" not in titles


def test_model_computed_ratio_is_rejected(monkeypatch, uniform_df):
    reply = ("Discount Level\nFinding: Discounts reach 0.40, double the average discount.\n"
             "Evidence: discount_pct max 0.40.\nInterpretation: this suggests variation.\n")
    titles = [i["title"] for i in _run(monkeypatch, uniform_df, reply)]
    assert "Discount Level" not in titles


def test_extremes_wording_rejected_without_real_outliers(monkeypatch, uniform_df):
    reply = ("Revenue Range Extremes\nFinding: Revenue is wide.\n"
             "Evidence: revenue min 10.00.\nInterpretation: this suggests variation.\n")
    assert "Revenue Range Extremes" not in [i["title"] for i in _run(monkeypatch, uniform_df, reply)]


def test_outlier_wording_allowed_when_outliers_really_exist(monkeypatch):
    rng = np.random.default_rng(2)
    vals = np.concatenate([rng.normal(100, 5, 2000), [900.0, 950.0, 1000.0]])
    df = pd.DataFrame({"amount": vals, "grp": rng.choice(["a", "b"], len(vals))})
    facts, cols = ig._fact_sheet(df)
    assert cols == ["amount"] and "fall outside the 1.5xIQR fences" in facts
    assert not ig._unsupported_claims("Amount Outliers", "amount has a few extreme values", cols)


def test_fact_sheet_states_no_outliers_for_even_data(uniform_df):
    facts, cols = ig._fact_sheet(uniform_df)
    assert cols == []
    assert "every value lies inside the 1.5xIQR fences" in facts
    assert "roughly symmetric" in facts


def test_fallback_insights_are_verified_and_non_misleading(monkeypatch, uniform_df):
    out = _run(monkeypatch, uniform_df, "garbage with no structure")
    text = " ".join(i["title"] + " " + i["body"] for i in out).lower()
    assert out and "outlier" not in text.replace("no statistical outliers", "")
    assert " x the minimum" not in text and "weak positive" not in text
