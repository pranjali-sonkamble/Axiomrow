"""Tests for the non-network parts of backend/llm_agent.py — response
parsing, the chart-request fallback safety net, and code/SQL execution
safety. Nothing here makes a real API call (no key needed to run these)."""

import pytest
from backend.llm_agent import (
    parse_llm_response,
    parse_chart_request,
    infer_chart_from_question,
    execute_code,
    execute_sql,
    DANGEROUS_PY_KEYWORDS,
    DANGEROUS_SQL_KEYWORDS,
)


class TestParseLLMResponse:
    def test_plain_text_response(self):
        result = parse_llm_response("The total is 100.")
        assert result["answer_type"] == "text"

    def test_python_code_block(self):
        raw = "Here:\n```python\nprint(1)\n```\nDone."
        result = parse_llm_response(raw)
        assert result["answer_type"] == "code"
        assert result["code"] == "print(1)"
        assert result["sql"] is None

    def test_python_and_sql_blocks_together(self):
        raw = "```python\ntotal = df['x'].sum()\n```\n```sql\nSELECT SUM(x) FROM df\n```"
        result = parse_llm_response(raw)
        assert result["answer_type"] == "code"
        assert "sum()" in result["code"]
        assert "SELECT SUM" in result["sql"]

    def test_sql_only_block(self):
        raw = "```sql\nSELECT * FROM df\n```"
        result = parse_llm_response(raw)
        assert result["answer_type"] == "code"
        assert result["code"] is None
        assert result["sql"] == "SELECT * FROM df"

    def test_chart_request_detected(self):
        raw = "CHART_REQUEST: bar | x=product | y=revenue | title=Test\nExplanation here."
        result = parse_llm_response(raw)
        assert result["answer_type"] == "chart"
        assert result["chart_config"]["type"] == "bar"


class TestParseChartRequest:
    def test_parses_all_fields(self):
        config = parse_chart_request("CHART_REQUEST: bar | x=product | y=revenue | title=Revenue by Product")
        assert config == {"type": "bar", "x": "product", "y": "revenue", "title": "Revenue by Product"}

    def test_handles_missing_optional_fields(self):
        config = parse_chart_request("CHART_REQUEST: histogram | x=age")
        assert config["type"] == "histogram"
        assert config["x"] == "age"
        assert "y" not in config


class TestChartFallbackInference:
    """This is the safety net added after the LLM described a chart in
    prose instead of emitting CHART_REQUEST — these tests reproduce
    that exact failure scenario."""

    COLUMNS = ["date", "product", "region", "revenue", "quantity"]

    def test_reproduces_the_original_bug_scenario(self):
        # The exact question that triggered the bug in production
        config = infer_chart_from_question("Show revenue vs quantity as a scatter chart", self.COLUMNS)
        assert config is not None
        assert config["type"] == "scatter"
        assert {config["x"], config["y"]} == {"revenue", "quantity"}

    def test_histogram_single_column(self):
        config = infer_chart_from_question("Show me revenue distribution as a histogram", self.COLUMNS)
        assert config["type"] == "histogram"
        assert config["x"] == "revenue"

    def test_value_by_category_phrasing_gets_correct_axes(self):
        # "revenue share BY region" — value mentioned before "by" but
        # must still map to y, category after "by" must map to x
        config = infer_chart_from_question("What's the revenue share by region as a pie chart", self.COLUMNS)
        assert config["type"] == "pie"
        assert config["x"] == "region"
        assert config["y"] == "revenue"

    def test_trend_over_time_finds_implied_date_column(self):
        # Only "revenue" is named explicitly — the date axis is implied
        config = infer_chart_from_question("Show revenue trend over time", self.COLUMNS)
        assert config["type"] == "line"
        assert config["x"] == "date"
        assert config["y"] == "revenue"

    def test_non_chart_question_returns_none(self):
        assert infer_chart_from_question("What is the total revenue?", self.COLUMNS) is None

    def test_chart_keyword_without_matching_column_returns_none(self):
        assert infer_chart_from_question("Show me a bar chart", []) is None


class TestExecuteCodeSafety:
    @pytest.mark.parametrize("keyword", DANGEROUS_PY_KEYWORDS)
    def test_every_dangerous_keyword_is_blocked(self, sample_df, keyword):
        output, error = execute_code(keyword, sample_df)
        assert output is None
        assert "Security" in error

    def test_normal_code_still_works(self, sample_df):
        output, error = execute_code("print(df['revenue'].sum())", sample_df)
        assert error is None
        assert output == "251000"

    def test_key_error_gives_friendly_message(self, sample_df):
        output, error = execute_code("print(df['nonexistent'])", sample_df)
        assert output is None
        assert "not found" in error

    def test_zero_division_gives_friendly_message(self, sample_df):
        output, error = execute_code("print(1/0)", sample_df)
        assert "Division by zero" in error

    def test_stdout_is_restored_after_every_outcome(self, sample_df):
        import sys
        before = sys.stdout
        execute_code("print(df['revenue'].sum())", sample_df)          # success path
        execute_code("print(df['bad_col'])", sample_df)                # exception path
        execute_code("import os", sample_df)                           # security-block path
        assert sys.stdout is before

    def test_empty_code_handled_gracefully(self, sample_df):
        output, error = execute_code("", sample_df)
        assert error == "No code was generated to run."


class TestExecuteSQLSafety:
    @pytest.mark.parametrize("keyword", DANGEROUS_SQL_KEYWORDS)
    def test_every_dangerous_keyword_is_blocked(self, sample_df, keyword):
        result, error = execute_sql(f"{keyword} something", sample_df)
        assert result is None
        assert "Security" in error

    def test_scalar_result(self, sample_df):
        result, error = execute_sql("SELECT SUM(revenue) FROM df", sample_df)
        assert error is None
        assert result == "251000.0"

    def test_multi_row_result_formatted_as_table(self, sample_df):
        result, error = execute_sql("SELECT product, revenue FROM df ORDER BY revenue DESC", sample_df)
        assert error is None
        assert "\n" in result  # multi-row results are text-table formatted

    def test_bad_column_gives_error_not_crash(self, sample_df):
        result, error = execute_sql("SELECT nonexistent FROM df", sample_df)
        assert result is None
        assert error is not None

    def test_empty_sql_handled_gracefully(self, sample_df):
        result, error = execute_sql("", sample_df)
        assert error == "No SQL was generated."
