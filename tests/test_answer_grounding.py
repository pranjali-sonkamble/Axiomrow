# tests/test_answer_grounding.py
# Regression tests for the exact failures seen in the chat screenshots,
# plus the wider class each one belongs to.

import pandas as pd
import pytest

from backend.answer_grounding import (
    build_known_values, content_is_untrustworthy, derived_counts,
    extract_numbers, find_grounding_problems, results_disagree,
    text_makes_data_claims, trim_framing_text,
)


@pytest.fixture
def df():
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=10),
        "product": ["Laptop", "Phone", "Tablet", "Laptop", "Phone",
                    "Headphones", "Laptop", "Tablet", "Phone", "Headphones"],
        "region": ["North", "South", "East", "North", "South",
                   "East", "North", "South", "East", "North"],
        "revenue": [75000, 12000, 82000, 30000, 46000, 15000, 50000, 20000, 60000, 75000],
    })


VERIFIED_PRODUCTS = (
    "Python result: Unique products (4): ['Headphones', 'Laptop', 'Phone', 'Tablet']\n"
    "SQL result: product   \nHeadphones\nLaptop    \nPhone     \nTablet    "
)


# ---- Screenshot 3: invented "Monitor" ------------------------------------
def test_invented_product_is_caught(df):
    framing = "There are 4 distinct products in the data: Laptop, Phone, Tablet, and Monitor."
    problems = find_grounding_problems(framing, VERIFIED_PRODUCTS, "which are those " + " ".join(df.columns),
                                       build_known_values(df))
    assert any("Monitor" in p for p in problems)
    assert content_is_untrustworthy(framing, VERIFIED_PRODUCTS, "which are those")


def test_correct_product_sentence_passes(df):
    framing = "There are 4 distinct products: Headphones, Laptop, Phone and Tablet."
    assert not content_is_untrustworthy(framing, VERIFIED_PRODUCTS, "which are those",
                                        build_known_values(df))


def test_count_visible_only_as_list_length_passes():
    verified = "Python result: ['Headphones', 'Laptop', 'Phone', 'Tablet']"
    assert not content_is_untrustworthy("There are 4 distinct products.", verified)


def test_real_value_omitted_from_output_is_caught(df):
    # Sentence names a real category the verified output never mentioned.
    verified = "Python result: Unique regions (2): ['East', 'North']"
    framing = "The regions are East, North and South."
    assert content_is_untrustworthy(framing, verified, "", build_known_values(df))


# ---- Screenshot 1: numbers that contradict the data ----------------------
def test_contradictory_number_is_caught():
    verified = "Python result: Max revenue: 82,000"
    assert content_is_untrustworthy("The highest single sale is $75,000.", verified)


def test_number_with_no_numbers_in_output_is_caught():
    verified = "Code ran successfully but produced no output."
    assert content_is_untrustworthy("Revenue totals 465,000.", verified)


def test_number_from_users_own_question_is_allowed():
    verified = "Python result: 12"
    assert not content_is_untrustworthy("In 2024 there were 12 sales.", verified, "sales in 2024")


def test_label_swap_still_caught():
    verified = "Laptop 30.0\nNorth 40.0"
    assert content_is_untrustworthy("Laptop 40.0 and North 30.0", verified)


def test_hedge_still_caught():
    assert content_is_untrustworthy("I don't have access to the data.", "Python result: 4")


def test_dates_do_not_create_negative_numbers():
    assert extract_numbers("2024-03-01") == [2024.0, 3.0, 1.0]
    assert extract_numbers("loss of -5.5") == [-5.5]


def test_derived_counts():
    assert 4.0 in derived_counts("['a', 'b', 'c', 'd']")
    assert 5.0 in derived_counts("h\na\nb\nc\nd")


# ---- Screenshot 1: text-only prose about the data ------------------------
@pytest.mark.parametrize("text", [
    "There are 3 distinct products: Laptop, Phone, and Tablet.",
    "You have a tiny but clean sales log with 10 distinct transactions.",
    "The products are Laptop and Phone.",
    "Revenue ranges from 12,000 to 82,000.",
    "Sales are strongest in Mumbai and Pune.",        # names not in question/columns
])
def test_prose_claims_about_data_are_detected(text, df):
    assert text_makes_data_claims(text, df, "what do you see")


@pytest.mark.parametrize("text", [
    "Hello! Ask me anything about your data.",
    "Which column would you like me to look at?",
    "I can summarise columns, run calculations and draw charts.",
])
def test_harmless_prose_is_allowed(text, df):
    assert not text_makes_data_claims(text, df, "hi")


# ---- Python vs SQL cross-check -------------------------------------------
@pytest.mark.parametrize("py, sql", [
    ("Unique products: 4", "4"),
    ("Total revenue: $465,000", "465000"),
    ("Total revenue: $465,000", "465000.4"),
    ("Share: 40.0%", "0.4"),
    ("Average: 3.14", "3.14159"),
])
def test_agreeing_results(py, sql):
    assert not results_disagree(py, sql)


def test_disagreeing_results():
    assert results_disagree("Unique products: 4", "3")


def test_tables_and_text_are_not_compared():
    assert not results_disagree("Unique products: 4", "product\nA\nB")
    assert not results_disagree("Top product: Laptop", "Laptop")


# ---- known values ---------------------------------------------------------
def test_known_values_handles_duplicate_columns_and_numbers():
    d = pd.DataFrame([["Alpha", 1, "Beta"]], columns=["x", "n", "x"])
    kv = build_known_values(d)
    assert {"alpha", "beta"} <= kv and "1" not in kv


def test_trim_keeps_multi_sentence_explanation_but_cuts_tables():
    ok = "First point. Second point. Third point."
    assert trim_framing_text(ok) == ok
    assert trim_framing_text("Summary.\n\n| a | b |") == "Summary."


# ---- §6/§12: causal/statistical overreach beyond what code establishes ---
def test_causal_claim_with_real_facts_still_caught():
    verified = ("Python result:\nOverall: Feb 120000 -> Mar 180000 (change +60000)\n"
               "By region: North +38000, South +15000, East +7000\nTop contributor: North")
    bad = "The increase was caused by a new marketing campaign in the North region."
    assert content_is_untrustworthy(bad, verified)


def test_association_language_is_allowed():
    verified = ("Python result:\nOverall: Feb 120000 -> Mar 180000 (change +60000)\n"
               "By region: North +38000\nTop contributor: North")
    good = ("Transaction value rose by 60,000 from February to March, and the increase "
           "is primarily associated with the North region, which grew by 38,000.")
    assert not content_is_untrustworthy(good, verified)


@pytest.mark.parametrize("phrase", [
    "X caused the increase", "this proves the hypothesis", "is a statistically significant outlier",
    "will definitely grow next quarter", "the reason for this is staffing",
])
def test_various_overreach_phrases_caught(phrase):
    from backend.answer_grounding import find_causal_overreach
    assert find_causal_overreach(phrase)


# ---- §13: comparison-question sentences validated same as any other ----
def test_comparison_summary_grounded():
    verified = ("Python result:\nMetric        India   UAE\nTransactions  1200    340\n"
               "Total value   950000  410000\nAvg value     791.7   1205.9")
    good = ("India has more transactions (1,200 vs 340) and a higher total value "
           "(950,000 vs 410,000), while UAE has a higher average transaction value.")
    bad = "India is clearly the winner with 3x more transactions than Japan."
    assert not content_is_untrustworthy(good, verified)
    assert content_is_untrustworthy(bad, verified)
