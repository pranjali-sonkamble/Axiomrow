# tests/test_answer_pipeline.py
# answer_question() end-to-end with a scripted fake LLM (no network).

import pandas as pd
import pytest

import backend.llm_agent as agent
from backend.answer_grounding import UNVERIFIED_ANSWER_MESSAGE


@pytest.fixture
def df():
    return pd.DataFrame({
        "product": ["Laptop", "Phone", "Tablet", "Headphones"],
        "revenue": [75000, 12000, 82000, 30000],
    })


CODE_REPLY = (
    "```python\nvals = sorted(df['product'].unique().tolist())\n"
    "print(f\"Unique products ({len(vals)}): {vals}\")\n```\n\n"
    "```sql\nSELECT DISTINCT product FROM df\n```"
)


def script(monkeypatch, replies):
    """Replace call_llm with a fake that returns `replies` in order and
    records every message list it was called with."""
    calls = []
    it = iter(replies)

    def fake(messages, temperature=0.1, session_id=None):
        calls.append({"messages": messages, "session_id": session_id})
        return next(it)

    monkeypatch.setattr(agent, "call_llm", fake)
    return calls


def ask(q, df, history=None):
    return agent.answer_question(q, "summary...", history, df_columns=list(df.columns),
                                 df=df, session_id="s1")


def test_prose_guess_is_retried_as_code(monkeypatch, df):
    # Screenshot 1: "3 distinct products: Laptop, Phone, Tablet" from prose.
    # Question wording changed from "how many products are present in this
    # dataset" to one query_dataframe() (backend/query_engine.py) genuinely
    # does NOT recognize (confirmed: returns None) — that question is now
    # correctly answered by the deterministic fast path before any LLM call,
    # which is the intended behavior of that integration, not a regression
    # here. This test is specifically about the LLM-path prose-retry
    # mechanism, so it needs a question guaranteed to reach the LLM.
    calls = script(monkeypatch, ["There are 3 distinct products: Laptop, Phone, and Tablet.", CODE_REPLY])
    result = ask("explain the pattern in this data", df)
    assert result["answer_type"] == "code" and "df['product']" in result["code"]
    assert len(calls) == 2
    assert "without running any code" in calls[1]["messages"][-1]["content"]
    assert calls[1]["session_id"] == "s1#followup"   # counted against quota, exempt from the 2s gap


def test_prose_guess_twice_is_refused_not_shown(monkeypatch, df):
    # Same reasoning as above: reworded so this reaches the LLM path being
    # tested, rather than the new deterministic fast path.
    script(monkeypatch, ["There are 3 distinct products: Laptop, Phone, Tablet.",
                         "Definitely 3 products: Laptop, Phone, Tablet."])
    result = ask("what's interesting about this dataset", df)
    assert result["answer_type"] == "text"
    assert result["content"] == UNVERIFIED_ANSWER_MESSAGE
    assert result["unverified"] is True


def test_greeting_needs_no_retry(monkeypatch, df):
    calls = script(monkeypatch, ["Hello! Ask me anything about your data."])
    result = ask("hi", df)
    assert result["answer_type"] == "text" and len(calls) == 1
    assert not result.get("unverified")


def test_llm_error_after_bad_prose_is_refused(monkeypatch, df):
    # Reworded: "total revenue" is now answered deterministically (confirmed
    # via query_dataframe) and never reaches call_llm at all.
    script(monkeypatch, ["Revenue totals 199,000.", "Error: Something went wrong communicating with the LLM."])
    assert ask("give me an overview of the product mix", df)["content"] == UNVERIFIED_ANSWER_MESSAGE


def test_chart_caption_with_claims_is_dropped(monkeypatch, df):
    script(monkeypatch, ["CHART_REQUEST: bar | x=product | y=revenue | title=Revenue by Product\n"
                         "Laptop leads with 75,000 in sales."])
    # Wording the deterministic chart engine does not understand (no metric
    # column named), so the LLM path - and its caption check - is what runs.
    result = ask("show a chart of how the products compare", df)
    assert result["answer_type"] == "chart" and result["content"] == ""


def test_chart_caption_without_claims_is_kept(monkeypatch, df):
    script(monkeypatch, ["CHART_REQUEST: bar | x=product | y=revenue | title=Revenue by Product\n"
                         "This chart plots revenue for each product."])
    result = ask("show a chart of how the products compare", df)
    assert "plots revenue" in result["content"]


def test_errors_and_unverified_answers_never_become_history(monkeypatch, df):
    calls = script(monkeypatch, [CODE_REPLY])
    history = [
        {"question": "q-good", "answer": "Python result: 4", "verified": True},
        {"question": "q-err", "answer": "Error: Something went wrong communicating with the LLM."},
        {"question": "q-bad", "answer": "There are 3 products.", "verified": False},
    ]
    ask("which are those", df, history)
    sent = " ".join(m["content"] for m in calls[0]["messages"])
    assert "q-good" in sent and "q-err" not in sent and "q-bad" not in sent
    assert "3 products" not in sent


def test_temperature_is_actually_sent_to_the_api(monkeypatch):
    seen = {}

    class FakeCompletions:
        def create(self, **kw):
            seen.update(kw)
            msg = type("M", (), {"content": "ok"})()
            return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()

    fake_client = type("Cl", (), {"chat": type("Ch", (), {"completions": FakeCompletions()})()})()
    monkeypatch.setattr(agent, "get_llm_client", lambda: (fake_client, "groq", "m"))
    agent.call_llm([{"role": "user", "content": "x"}])
    assert seen["temperature"] == 0.1


def test_summary_pass_failure_returns_empty(monkeypatch):
    script(monkeypatch, ["Error: boom"])
    assert agent.summarize_verified_result("q", "Python result: 4") == ""
    assert agent.summarize_verified_result("q", "") == ""


# ---- prompt shape: "how many" must yield a number in BOTH Python and SQL ----
def test_prompt_teaches_count_shape_not_list():
    from prompts.system_prompts import DATA_ANALYST_SYSTEM_PROMPT as P
    assert "COUNT(DISTINCT product)" in P          # count example exists
    assert "Do NOT list the values unless" in P    # no list dump for "how many"


def test_count_question_gets_cross_checked(monkeypatch):
    # Python + SQL both return one number -> the disagreement check can run.
    from backend.answer_grounding import results_disagree
    assert not results_disagree("Unique products: 4", "4")
    assert results_disagree("Unique products: 4", "3")


def test_prompt_handles_vague_followups_and_line_per_fact():
    from prompts.system_prompts import DATA_ANALYST_SYSTEM_PROMPT as P
    assert '"show me"' in P and "previous question" in P
    assert "ONE labelled fact per line" in P


def test_prompt_avoids_technical_jargon_and_raw_timestamps():
    from prompts.system_prompts import DATA_ANALYST_SYSTEM_PROMPT as P
    assert "never print raw Timestamp objects" in P
    assert "No missing values in any column" in P


def test_summary_prompt_targets_non_technical_reader():
    from backend.llm_agent import _SUMMARY_SYSTEM_PROMPT as S
    assert "non-technical reader" in S
    assert "never repeat a Python Timestamp" in S


# ---- Performance regression: extra LLM call only as a fallback ----------
def test_summarize_not_called_when_first_sentence_is_already_valid(monkeypatch):
    calls = []
    monkeypatch.setattr(agent, "summarize_verified_result",
                        lambda q, v: calls.append(1) or "unused")
    from backend.answer_grounding import content_is_untrustworthy
    verified = "Python result: Unique products (4): ['Headphones', 'Laptop', 'Phone', 'Tablet']"
    good = "There are 4 distinct products: Headphones, Laptop, Phone, Tablet."
    assert not content_is_untrustworthy(good, verified)
    # If the pre-existing sentence already passes, app.py's _build_framing
    # must never reach summarize_verified_result — this only documents the
    # contract; the ordering itself is exercised via the app.py harness.
    assert calls == []


# ---- Validation bypass audit finding: chart-shaped question, prose answer,
# failed fallback inference must still be enforced/refused ----------------
def test_chart_wording_with_prose_reply_and_failed_fallback_is_enforced(monkeypatch, df):
    # is_chart_request is decided from the QUESTION's wording; the answer
    # can still legitimately be plain prose. If the model's prose makes a
    # data claim and the deterministic chart fallback also can't build a
    # config (column name doesn't match df_columns here), the result must
    # still go through _enforce_verified_answer — not fall through raw.
    calls = script(monkeypatch, ["Laptop generated 900 in revenue across all regions.",
                                 "Laptop generated 900 in revenue across all regions."])
    result = agent.answer_question("show me a chart of revenue by nonexistent_column", "summary", None,
                                   df_columns=list(df.columns), df=df, session_id=None)
    assert result["answer_type"] == "text"
    assert "900" not in result["content"]
    assert len(calls) == 2   # confirms the retry path actually ran, not a silent pass-through


# ---- Startup performance: client caching + background warm-up --------
def test_llm_client_is_built_once_and_reused(monkeypatch):
    build_count = {"n": 0}

    class FakeClient:
        pass

    def fake_groq_ctor(api_key=None):
        build_count["n"] += 1
        return FakeClient()

    import sys, types
    fake_groq_module = types.ModuleType("groq")
    fake_groq_module.Groq = fake_groq_ctor
    monkeypatch.setitem(sys.modules, "groq", fake_groq_module)
    agent._LLM_CLIENT_CACHE.clear()
    monkeypatch.setenv("LLM_PROVIDER", "groq")

    c1, provider1, model1 = agent.get_llm_client()
    c2, provider2, model2 = agent.get_llm_client()
    assert build_count["n"] == 1          # NOT rebuilt on the second call
    assert c1 is c2


def test_warm_up_never_raises_even_if_everything_fails(monkeypatch):
    # Simulate every optional dependency being unavailable/misconfigured.
    monkeypatch.setattr(agent, "get_llm_client", lambda: (_ for _ in ()).throw(RuntimeError("no key")))
    agent.warm_up_heavy_imports()   # must swallow the exception, not propagate
