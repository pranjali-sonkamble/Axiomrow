"""Tests for backend/query_engine.py — the deterministic answer engine.

This formalizes the manual regression check that was run when this
file was refactored: one test per branch (11 sections), plus edge
cases for the numeric filter parser (AND / OR / BETWEEN).
"""

try:
    import pytest  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover
    pytest = None

import pandas as pd
from backend.query_engine import query_dataframe, _find_column, _word_in_text, _find_date_column


class TestWordBoundaryMatching:
    def test_exact_word_matches(self):
        assert _word_in_text("product", "show me the product list") is True

    def test_substring_inside_longer_word_does_not_match(self):
        # "product" must not match inside "production"
        assert _word_in_text("product", "the production line") is False

    def test_plural_does_not_match_singular_column_name(self):
        assert _word_in_text("product", "show me the products") is False


class TestFindColumn:
    def test_finds_exact_column_name(self, sample_df):
        assert _find_column("what is total revenue?", sample_df.columns) == "revenue"

    def test_returns_none_when_no_column_mentioned(self, sample_df):
        assert _find_column("what color is the sky?", sample_df.columns) is None

    def test_prefers_longer_column_name_on_overlap(self):
        import pandas as pd
        cols = pd.Index(["date", "update_date"])
        # "update_date" contains "date" as a substring but not a whole
        # word boundary match, so only the longer name should ever win
        assert _find_column("what is the update_date?", cols) == "update_date"


class TestFindDateColumn:
    def test_detects_column_named_date(self, sample_df):
        assert _find_date_column(sample_df) == "date"

    def test_returns_none_when_no_date_column(self):
        import pandas as pd
        df = pd.DataFrame({"a": [1], "b": [2]})
        assert _find_date_column(df) is None


class TestQueryDataframeBranches:
    """One case per section of query_dataframe, matching the manual
    baseline verification performed when this file was refactored."""

    def test_unique_values(self, sample_df):
        # NOTE: must use the exact singular column name — "regions"
        # (plural) is a known, documented limitation of _word_in_text's
        # whole-word matching (see TestPluralLimitation below).
        result = query_dataframe("What are the unique values in region?", sample_df)
        assert result is not None
        assert "North" in result["content"]

    def test_categorical_filter_single_value(self, sample_df):
        result = query_dataframe("Find rows where product is Laptop", sample_df)
        assert result is not None
        assert "Laptop" in result["content"]
        assert "Phone" not in result["content"]

    def test_categorical_filter_or(self, sample_df):
        result = query_dataframe("Find rows where product is Laptop or Headphones", sample_df)
        assert result is not None
        assert "Laptop" in result["content"]
        assert "Headphones" in result["content"]
        assert "Phone" not in result["content"]

    def test_membership_check_true(self, sample_df):
        result = query_dataframe("Is there any Laptop in the product column?", sample_df)
        assert result["content"] == "Yes."

    def test_membership_check_false(self, sample_df):
        result = query_dataframe("Is there any Tablet in the product column?", sample_df)
        assert result["content"] == "No."

    def test_numeric_filter_greater_than(self, sample_df):
        result = query_dataframe("Show rows where revenue > 50000", sample_df)
        assert result is not None
        assert "12000" not in result["content"]

    def test_numeric_filter_between(self, sample_df):
        result = query_dataframe("Show rows where revenue between 40000 and 70000", sample_df)
        assert result is not None
        assert "75000" not in result["content"]

    def test_numeric_filter_and(self, sample_df):
        result = query_dataframe("Show rows where revenue > 40000 and quantity > 30", sample_df)
        assert result is not None

    def test_numeric_filter_or(self, sample_df):
        result = query_dataframe("Show rows where revenue > 80000 or quantity >= 40", sample_df)
        assert result is not None
        assert "12000" in result["content"]  # quantity 40 row (Headphones)

    def test_highest_transaction(self, sample_df):
        result = query_dataframe("What is the highest revenue transaction?", sample_df)
        assert "75000" in result["content"]

    def test_lowest_transaction(self, sample_df):
        result = query_dataframe("What is the lowest revenue transaction?", sample_df)
        assert "12000" in result["content"]

    def test_exact_date_lookup_iso(self, sample_df):
        result = query_dataframe("What was the revenue on 2024-01-15?", sample_df)
        assert "75,000" in result["content"]

    def test_exact_date_lookup_natural(self, sample_df):
        result = query_dataframe("What was the revenue on January 18, 2024?", sample_df)
        assert "45,000" in result["content"]

    def test_grouped_total_by_category(self, sample_df):
        result = query_dataframe("Calculate total revenue by product", sample_df)
        assert result is not None
        assert "Laptop" in result["content"]

    def test_grouped_average_by_category(self, sample_df):
        result = query_dataframe("Calculate average revenue by region", sample_df)
        assert result is not None

    def test_ranking(self, sample_df):
        result = query_dataframe("Rank the products by total revenue", sample_df)
        assert result is not None
        assert "1." in result["content"]

    def test_total(self, sample_df):
        result = query_dataframe("What is the total revenue?", sample_df)
        assert result["content"] == "251,000"

    def test_total_with_month_filter(self, sample_df):
        result = query_dataframe("What is the total revenue in January 2024?", sample_df)
        assert result["content"] == "120,000"  # 75000 + 45000

    def test_average(self, sample_df):
        result = query_dataframe("What is the average revenue?", sample_df)
        assert result is not None

    def test_top_n(self, sample_df):
        result = query_dataframe("Show me the top 3 transactions by revenue", sample_df)
        assert result is not None

    def test_bottom_n(self, sample_df):
        result = query_dataframe("Show me the bottom 2 transactions by revenue", sample_df)
        assert result is not None

    def test_row_count(self, sample_df):
        result = query_dataframe("How many rows are there?", sample_df)
        assert result["content"] == "5"

    def test_unmatched_question_returns_none(self, sample_df):
        assert query_dataframe("What color is the sky?", sample_df) is None

    def test_plural_column_names_now_match(self, sample_df):
        """Regression test: plural forms ('regions') now resolve to the
        singular column ('region') via _column_tokens' pluralization
        handling. This used to be a known limitation — see git history."""
        result = query_dataframe("What are the unique regions?", sample_df)
        assert result == {"answer_type": "text", "content": "North, South, East"}

    def test_none_dataframe_returns_none(self):
        assert query_dataframe("What is the total revenue?", None) is None

    def test_empty_question_returns_none(self, sample_df):
        assert query_dataframe("   ", sample_df) is None


class TestLargeResultSetCapping:
    """Covers the fix found while stress-testing on a 300K-row dataset:
    a filter query matched 38,760 rows and tried to format all of them
    as one text blob (3.8s, and would flood the chat UI)."""

    def test_large_filter_result_is_capped(self):
        import numpy as np
        big_df = pd.DataFrame({
            "revenue": np.random.default_rng(0).integers(0, 1000, 10_000),
        })
        result = query_dataframe("Show rows where revenue > 0", big_df)
        lines = result["content"].splitlines()
        assert len(lines) < 35  # header + 30 data rows + summary line
        assert "more rows" in lines[-1]

    def test_small_filter_result_is_not_truncated(self, sample_df):
        result = query_dataframe("Show rows where revenue > 50000", sample_df)
        assert "more rows" not in result["content"]


class TestResponseShapeConsistency:
    """Every branch should return the same {'answer_type', 'content'}
    shape — this was a real inconsistency found and fixed during the
    refactor (one branch used to return a bare string instead)."""

    @pytest.mark.parametrize("question", [
        "Find rows where product is Laptop",
        "Find rows where product is Laptop or Headphones",
        "What is the total revenue?",
        "How many rows are there?",
    ])
    def test_all_matched_branches_return_dict_shape(self, sample_df, question):
        result = query_dataframe(question, sample_df)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"answer_type", "content"}
