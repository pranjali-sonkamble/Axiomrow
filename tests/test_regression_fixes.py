"""Regression tests for wrong answers found on the 1.4M-row sales file."""
import pandas as pd
import pytest
from backend.query_engine import query_dataframe


@pytest.fixture
def sales():
    return pd.DataFrame({
        "region": ["East", "West", "East", "West", "North", "North"],
        "sales_rep": ["Rep_1", "Rep_2", "Rep_1", "Rep_3", "Rep_2", "Rep_3"],
        "customer_segment": ["Corporate", "Consumer", "Corporate", "Consumer", "Corporate", "Consumer"],
        "revenue": [100.0, 200.0, 300.0, 50.0, 25.0, 25.0],
        "units_sold": [1, 2, 3, 4, 5, 6],
    })


def test_highest_sales_rep_is_a_rep_not_the_grand_total(sales):
    r = query_dataframe("Which sales rep has the highest revenue?", sales)
    assert r["content"].startswith("Rep_1 has the highest revenue")


def test_top_n_sales_reps(sales):
    lines = query_dataframe("Top 2 sales reps by revenue", sales)["content"].splitlines()
    assert lines[0].startswith("1. Rep_1") and lines[1].startswith("2. Rep_2")


def test_revenue_by_sales_rep_groups_by_rep_not_by_revenue(sales):
    out = query_dataframe("Show revenue by sales rep", sales)["content"].splitlines()
    assert out == ["Rep_1: 400", "Rep_2: 225", "Rep_3: 75"]


def test_how_many_sales_reps(sales):
    assert query_dataframe("How many sales reps are there?", sales)["content"] == "3"


def test_highest_customer_segment(sales):
    assert query_dataframe("Which customer_segment has the highest revenue?", sales)["content"].startswith("Corporate has the highest")


def test_order_count_question_is_not_answered_with_revenue(sales):
    assert query_dataframe("Which sales rep handled the most orders?", sales) is None


@pytest.mark.parametrize("q", ["Tell me about the revenue trend", "What is the highest revenue?",
                               "Why did revenue drop?", "What will revenue be next month?"])
def test_unhandled_structure_questions_never_return_the_grand_total(sales, q):
    r = query_dataframe(q, sales)
    assert r is None or "700" not in str(r.get("content", ""))


def test_plain_total_and_filters_still_work(sales):
    assert query_dataframe("What is the total revenue?", sales)["content"] == "700"
    assert query_dataframe("What is total units_sold for East?", sales)["content"] == "4"


def test_comparison_label_uses_the_requested_metric():
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-05", "2024-02-05"]), "units_sold": [10, 20], "revenue": [1.0, 2.0]})
    r = query_dataframe("Compare units_sold in January and February 2024", df)
    assert "units_sold" in r["content"] and "revenue" not in r["content"]


def test_numeric_stats_for_non_int64_columns():
    from backend.data_loader import get_data_profile
    df = pd.DataFrame({"a": pd.Series([1, 2, 3], dtype="int32"), "b": pd.Series([1.5, 2.5, 3.5], dtype="float32")})
    cols = {c["name"]: c for c in get_data_profile(df)["columns"]}
    assert cols["a"]["stats"]["max"] == 3 and cols["b"]["stats"]["min"] == 1.5


def test_dq_fast_paths_match_full_computation():
    from backend.data_quality import assess_data_quality
    df = pd.DataFrame({"city": ["Pune", "pune", "Mumbai"] * 10, "amount": ["1"] * 27 + ["x"] * 3,
                       "clean": ["a", "b", "c"] * 10})
    names = {i["issue"]: i["columns"] for i in assess_data_quality(df)["issues"]}
    assert names["Inconsistent categorical values"] == ["city"]
    assert names["Numeric values stored as text / mixed numeric type"] == ["amount"]
