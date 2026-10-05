# tests/test_query_engine_integration.py
import re
#
# Integration tests for wiring backend/query_engine.py's query_dataframe()
# as a deterministic fast path at the top of answer_question()
# (backend/llm_agent.py). See that file for the integration itself —
# nothing in query_engine.py was modified.

import pandas as pd
import pytest

import backend.llm_agent as agent


@pytest.fixture
def df():
    return pd.DataFrame({
        "region": ["North", "South", "East", "North", "South", "East", "North"],
        "product": ["Laptop", "Phone", "Tablet", "Laptop", "Phone", "Tablet", "Laptop"],
        "revenue": [75000, 12000, 82000, 30000, 46000, 15000, 50000],
        "transaction_date": pd.to_datetime([
            "2023-01-05", "2023-01-18", "2023-02-01", "2023-02-14",
            "2023-03-01", "2023-03-10", "2023-01-22",
        ]),
    })


def _forbid_llm_call(monkeypatch):
    """Any call_llm invocation here is itself the test failure — a
    deterministically-answered question must never reach the network."""
    def fail(*a, **k):
        raise AssertionError("call_llm was invoked for a question query_engine should have answered")
    monkeypatch.setattr(agent, "call_llm", fail)


# ============================================================
# Deterministic path — the 6 required test questions
# ============================================================

def test_how_many_unique_regions(monkeypatch, df):
    # UPDATED: wording changed from a bare number ("3") to a natural
    # sentence, per direct user feedback that deterministic answers read
    # too tersely compared to the LLM path. The underlying VALUE is
    # unchanged (still df["region"].nunique()) — only the wording.
    _forbid_llm_call(monkeypatch)
    r = agent.answer_question("How many unique regions are there?", "summary", None,
                              df_columns=list(df.columns), df=df, session_id=None)
    assert r["answer_type"] == "text"
    assert str(df["region"].nunique()) in r["content"]
    assert "unique region" in r["content"]


def test_which_region_highest_revenue(monkeypatch, df):
    _forbid_llm_call(monkeypatch)
    r = agent.answer_question("Which region has the highest revenue?", "summary", None,
                              df_columns=list(df.columns), df=df, session_id=None)
    assert r["answer_type"] == "text"
    top = df.groupby("region")["revenue"].sum().idxmax()
    assert top in r["content"]


def test_top_5_products_by_revenue(monkeypatch, df):
    _forbid_llm_call(monkeypatch)
    r = agent.answer_question("What are the top 5 products by revenue?", "summary", None,
                              df_columns=list(df.columns), df=df, session_id=None)
    assert r["answer_type"] == "text"
    top_product = df.groupby("product")["revenue"].sum().idxmax()
    assert top_product in r["content"]


def test_show_transactions_above_10000(monkeypatch, df):
    # DISCOVERED LIMITATION (query_engine.py's own pattern-matching, not an
    # integration issue — not modified, per instructions): this exact
    # phrasing does not name which numeric column "10000" refers to
    # ("transactions" isn't itself numeric), so query_dataframe() correctly
    # declines to guess and returns None. "Show transactions WITH REVENUE
    # above 10000" IS recognized (confirmed separately). This test verifies
    # the safe, correct consequence of that: it falls through to the LLM
    # rather than crashing, erroring, or silently guessing a column.
    from backend.query_engine import query_dataframe
    assert query_dataframe("Show transactions above 10000.", df) is None

    calls = []
    monkeypatch.setattr(agent, "call_llm",
                        lambda messages, temperature=0.1, session_id=None: (
                            calls.append(1) or
                            "```python\nprint(df[df['revenue']>10000])\n```\n\n```sql\nSELECT * FROM df WHERE revenue>10000\n```"
                        ))
    r = agent.answer_question("Show transactions above 10000.", "summary", None,
                              df_columns=list(df.columns), df=df, session_id=None)
    assert len(calls) == 1
    assert r["answer_type"] == "code"


def test_show_transactions_above_10000_WITH_column_named_is_deterministic(monkeypatch, df):
    # The same request, worded so the numeric column is explicit, IS
    # answered deterministically — confirms the capability exists, just not
    # for the exact bare phrasing in the task's required-test list.
    _forbid_llm_call(monkeypatch)
    r = agent.answer_question("Show transactions with revenue above 10000.", "summary", None,
                              df_columns=list(df.columns), df=df, session_id=None)
    assert r["answer_type"] == "text"
    assert "75000" in r["content"] or "82000" in r["content"]


def test_how_many_transactions_in_january_2023(monkeypatch, df):
    _forbid_llm_call(monkeypatch)
    r = agent.answer_question("How many transactions happened in January 2023?", "summary", None,
                              df_columns=list(df.columns), df=df, session_id=None)
    assert r["answer_type"] == "text"
    expected = int((df["transaction_date"].dt.month == 1).sum())
    assert r["content"] == str(expected) or str(expected) in r["content"]


def test_how_many_products(monkeypatch, df):
    _forbid_llm_call(monkeypatch)
    r = agent.answer_question("How many products are there?", "summary", None,
                              df_columns=list(df.columns), df=df, session_id=None)
    assert r["answer_type"] == "text"
    assert r["content"] == str(df["product"].nunique())


# ============================================================
# LLM fallback — questions query_dataframe() does NOT recognize
# ============================================================

def test_fallback_open_ended_explanation(monkeypatch, df):
    # Confirmed via direct query_dataframe() call: returns None for this.
    from backend.query_engine import query_dataframe
    assert query_dataframe("explain the pattern in this data", df) is None

    calls = []
    monkeypatch.setattr(agent, "call_llm",
                        lambda messages, temperature=0.1, session_id=None: (
                            calls.append(1) or
                            "```python\nprint('analysis')\n```\n\n```sql\nSELECT 1\n```"
                        ))
    r = agent.answer_question("explain the pattern in this data", "summary", None,
                              df_columns=list(df.columns), df=df, session_id=None)
    assert len(calls) == 1               # the LLM WAS called — correct fallback
    assert r["answer_type"] == "code"    # existing pipeline ran unchanged


def test_fallback_root_cause_question(monkeypatch, df):
    # NOTE: "why did revenue increase in March" was tried first and found
    # to be a genuine discovered limitation, NOT usable as a clean fallback
    # example: query_dataframe() (pre-existing behavior, not touched by this
    # integration) matches it against its filtered-aggregation pattern,
    # "in March" isn't understood as a date filter without a year, and it
    # silently returns the UNFILTERED grand total (310,000) instead of
    # falling through to the LLM's root-cause reasoning. That is reported
    # separately as a limitation. This test uses a confirmed-None question
    # instead, so it actually demonstrates clean LLM fallback behavior.
    from backend.query_engine import query_dataframe
    assert query_dataframe("give me an overview of the product mix", df) is None

    calls = []
    monkeypatch.setattr(agent, "call_llm",
                        lambda messages, temperature=0.1, session_id=None: (
                            calls.append(1) or "This dataset covers three products across three regions."
                        ))
    r = agent.answer_question("give me an overview of the product mix", "summary", None,
                              df_columns=list(df.columns), df=df, session_id=None)
    assert len(calls) >= 1
    assert r["answer_type"] == "text"


def test_why_question_is_never_answered_with_the_unfiltered_grand_total(df):
    # FIXED (was test_LIMITATION_why_question_can_be_misread_as_plain_total).
    # query_dataframe() used to answer "why did revenue increase in March"
    # with the UNFILTERED grand total (310,000) because "in March" has no year.
    # Ranking / counting / trend / cause / future-intent questions that are not
    # understood are no longer answered with an overall total: the engine
    # returns None so the LLM path computes it from the data.
    from backend.query_engine import query_dataframe
    grand_total = f"{int(df['revenue'].sum()):,}"
    for q in ("why did revenue increase in March",
              "Tell me about the revenue trend",
              "predict next month revenue"):
        r = query_dataframe(q, df)
        assert r is None or grand_total not in str(r.get("content", "")), q


def test_fallback_preserves_existing_validation(monkeypatch, df):
    # A fallback question whose LLM reply makes an unverified data claim
    # must still go through the SAME grounding/retry machinery as before —
    # the deterministic integration must not have bypassed it.
    from backend.answer_grounding import UNVERIFIED_ANSWER_MESSAGE
    replies = iter(["There are 500 unique customers with 99% loyalty.",
                    "There are 500 unique customers with 99% loyalty."])
    monkeypatch.setattr(agent, "call_llm",
                        lambda messages, temperature=0.1, session_id=None: next(replies))
    r = agent.answer_question("tell me something surprising", "summary", None,
                              df_columns=list(df.columns), df=df, session_id=None)
    assert r["content"] == UNVERIFIED_ANSWER_MESSAGE
    assert r.get("unverified") is True


# ============================================================
# Error isolation
# ============================================================

def test_query_engine_exception_falls_through_to_llm(monkeypatch, df):
    def boom(question, df):
        raise RuntimeError("simulated unexpected query_engine failure")
    monkeypatch.setattr(agent, "query_dataframe", boom)

    calls = []
    monkeypatch.setattr(agent, "call_llm",
                        lambda messages, temperature=0.1, session_id=None: (
                            calls.append(1) or
                            "```python\nprint(df['revenue'].sum())\n```\n\n```sql\nSELECT SUM(revenue) FROM df\n```"
                        ))
    # "what is the total revenue" would normally be answered deterministically;
    # with query_dataframe forced to raise, the app must not crash and must
    # fall through to the existing LLM pipeline instead.
    r = agent.answer_question("what is the total revenue", "summary", None,
                              df_columns=list(df.columns), df=df, session_id=None)
    assert len(calls) == 1
    assert r["answer_type"] == "code"


def test_query_engine_not_called_when_df_is_none(monkeypatch):
    # answer_question() can be called without a live DataFrame (e.g. the
    # dataset failed to load); the fast path must not attempt to run
    # query_dataframe(None, ...) and must go straight to the LLM path.
    def fail_if_called(question, df):
        raise AssertionError("query_dataframe should not be called with df=None")
    monkeypatch.setattr(agent, "query_dataframe", fail_if_called)
    monkeypatch.setattr(agent, "call_llm",
                        lambda messages, temperature=0.1, session_id=None: "Hello! Ask me about your data.")
    r = agent.answer_question("hi", "summary", None, df_columns=None, df=None, session_id=None)
    assert r["answer_type"] == "text"


# ============================================================
# Chart-request behavior — deliberately NOT merged (see llm_agent.py comment)
# ============================================================

def test_chart_shaped_query_engine_result_falls_through_to_existing_chart_pipeline(monkeypatch, df):
    # query_dataframe() DOES recognize this as a chart request internally,
    # but its {"chart": {...}} shape is intentionally not adapted into this
    # integration (see the comment in llm_agent.answer_question). Confirm
    # it still reaches the existing, unmodified LLM+deterministic-fallback
    # chart pipeline and produces a normal chart_config-shaped answer.
    from backend.query_engine import query_dataframe
    det = query_dataframe("plot revenue by region", df)
    assert det is not None and det.get("answer_type") == "chart"   # confirms the chart branch exists

    monkeypatch.setattr(agent, "call_llm",
                        lambda messages, temperature=0.1, session_id=None: (
                            "CHART_REQUEST: bar | x=region | y=revenue | title=Revenue by Region\n"
                            "This chart plots revenue by region."
                        ))
    r = agent.answer_question("plot revenue by region", "summary", None,
                              df_columns=list(df.columns), df=df, session_id=None)
    assert r["answer_type"] == "chart"
    assert "chart_config" in r          # existing shape, not query_engine's "chart" key
    assert r["chart_config"]["x"] == "region" and r["chart_config"]["y"] == "revenue"


# ============================================================
# Fixes made this round (user-reported issues, verified against
# screenshots): top-N shortfall note, natural-language wording,
# KeyError misattribution, and code visibility on error.
# ============================================================

def test_top_n_shortfall_note_when_fewer_entities_exist(df):
    from backend.query_engine import query_dataframe
    actual_count = df["product"].nunique()
    r = query_dataframe("What are the top 99 products by revenue?", df)
    assert r is not None
    assert "only has" in r["content"] and str(actual_count) in r["content"]


def test_top_n_no_shortfall_note_when_enough_entities_exist(df):
    from backend.query_engine import query_dataframe
    r = query_dataframe("What are the top 2 products by revenue?", df)
    assert r is not None
    assert "only has" not in r["content"]


def test_highest_lowest_wording_is_a_sentence_not_a_bare_colon_pair(df):
    from backend.query_engine import query_dataframe
    high = query_dataframe("Which region has the highest revenue?", df)
    low = query_dataframe("Which product has the lowest revenue?", df)
    assert "has the highest" in high["content"]
    assert "has the lowest" in low["content"]
    # Old format was exactly "Label: value" with nothing else — confirm
    # that bare shape is gone.
    assert not re.match(r"^[\w\s]+:\s[\d,.]+$", high["content"])


def test_keyerror_from_real_column_lookup_still_says_column_not_found():
    df2 = pd.DataFrame({"revenue": [1, 2, 3]})
    _, err = agent.execute_code("print(df['nope'].sum())", df2)
    assert "not found in the dataset" in err


def test_keyerror_unrelated_to_dataframe_is_not_misreported():
    df2 = pd.DataFrame({"revenue": [1, 2, 3]})
    _, err = agent.execute_code("d = {'a': 1}\nprint(d['import'])", df2)
    assert "not found in the dataset" not in err
    assert "Code execution error" in err
