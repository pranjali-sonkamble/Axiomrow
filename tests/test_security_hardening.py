"""Regression tests for the round-2 security hardening.
Each case below was a REAL escape / abuse path before the fix."""
import os
import pandas as pd
import pytest

import backend.llm_agent as agent
from backend.llm_agent import execute_code, execute_sql
import backend.rate_limiter as rl


@pytest.fixture
def df():
    return pd.DataFrame({"a": [1, 2, 3], "core": [4, 5, 6]})


@pytest.mark.parametrize("code", [
    "print(pd.read_csv('/etc/hostname'))",
    "x = pd.read_csv\nprint(x)",
    "df.to_csv('/tmp/axiomrow_pwn.csv')",
    "print(np.loadtxt('/etc/hostname', dtype=str))",
    "print(pd.io.common.os.environ)",
    "print(pd.compat.sys.version)",
    "plt.plot([1]); plt.savefig('/tmp/axiomrow_x.png')",
    "s='{0.__cla'+'ss__.__mro__}'\nprint(s.format(df))",
    "print(str.format('{0.__cla'+'ss__}', df))",
    "print(df.eval('a + 1'))",
    "print(pd._libs)",
    "from numpy import loadtxt",
    "from pandas import read_csv",
    "print(np.lib.npyio)",
])
def test_sandbox_escape_attempts_are_blocked(df, code):
    out, err = execute_code(code, df)
    assert out is None and err and "Security" in err, (out, err)


def test_normal_analysis_still_works(df):
    out, err = execute_code("print(df['a'].sum())\nprint(df['core'].mean())", df)
    assert err is None and out.splitlines()[0] == "6"


def test_f_strings_and_percent_formatting_still_work(df):
    out, err = execute_code("t = df['a'].sum()\nprint(f'Total: {t:,}')\nprint('%d rows' % len(df))", df)
    assert err is None and "Total: 6" in out

@pytest.mark.skipif(
    "fork" not in __import__("multiprocessing").get_all_start_methods(),
    reason="needs fork start method (Linux/macOS only)",
)

def test_worker_environment_is_wiped(df, monkeypatch):
    # Even if a path to os.environ existed, the secrets are gone in the child.
    monkeypatch.setenv("GROQ_API_KEY", "SECRET123")
    monkeypatch.setenv("OPENAI_API_KEY", "SECRET456")
    out, err = execute_code("print(len(df))", df)
    assert err is None
    import multiprocessing as mp
    q = mp.get_context("fork").Queue()

    def probe(q):
        agent._lock_down_worker()
        q.put([k for k in os.environ if "KEY" in k.upper()])
    p = mp.get_context("fork").Process(target=probe, args=(q,)); p.start(); p.join(5)
    assert q.get(timeout=5) == []


@pytest.mark.parametrize("sql", [
    "SELECT * FROM read_csv('/etc/hostname')",
    "SELECT * FROM read_text('/etc/hostname')",
    "SELECT * FROM glob('/etc/*')",
    "SELECT * FROM '/etc/hostname'",
    "SELECT * FROM duckdb_settings()",
])
def test_sql_cannot_touch_files_or_settings(df, sql):
    out, err = execute_sql(sql, df)
    assert out is None and err


def test_sql_still_queries_the_dataframe(df):
    out, err = execute_sql("SELECT SUM(a) FROM df", df)
    assert err is None and out in ("6", "6.0")


def test_sql_runaway_query_is_interrupted(df, monkeypatch):
    monkeypatch.setattr(agent, "SQL_TIMEOUT_SECONDS", 1)
    out, err = execute_sql(
        "WITH r AS (SELECT range AS n FROM range(40000)) SELECT count(*) FROM r x, r y, r z", df)
    assert out is None and err


def test_followup_calls_count_but_skip_the_gap():
    rl._session_calls.clear(); rl._global_calls.clear()
    assert rl.check_and_register("s1", now=1000.0)[0]
    assert not rl.check_and_register("s1", now=1000.5)[0]          # gap applies
    assert rl.check_and_register("s1#followup", now=1000.5)[0]     # exempt...
    assert len(rl._session_calls["s1"]) == 2                       # ...but counted


def test_global_cap_applies_across_sessions(monkeypatch):
    rl._session_calls.clear(); rl._global_calls.clear()
    monkeypatch.setattr(rl, "GLOBAL_MAX_CALLS_PER_HOUR", 3)
    results = [rl.check_and_register(f"s{i}", now=1000.0 + i * 10)[0] for i in range(5)]
    assert results == [True, True, True, False, False]


# ---- upload limits are enforced while parsing -------------------------
def test_row_cap_enforced_during_parse(monkeypatch):
    import io
    import backend.data_loader as dl
    monkeypatch.setattr(dl, "MAX_ROWS", 5)
    with pytest.raises(ValueError, match="rows"):
        dl.load_csv(io.BytesIO(("a,b\n" + "1,2\n" * 50).encode()))


def test_column_cap_enforced_during_parse(monkeypatch):
    import io
    import backend.data_loader as dl
    monkeypatch.setattr(dl, "MAX_COLS", 3)
    with pytest.raises(ValueError, match="columns"):
        dl.load_csv(io.BytesIO(b"a,b,c,d,e\n1,2,3,4,5\n"))


def test_excel_upload_disabled_by_default():
    import io
    import backend.data_loader as dl

    class F(io.BytesIO):
        name = "x.xlsx"
    with pytest.raises(ValueError, match="not supported"):
        dl.load_uploaded_file(F(b"PK"))


def test_sample_rows_can_be_withheld_from_llm(monkeypatch):
    import backend.data_loader as dl
    d = pd.DataFrame({"name": ["Alice Smith", "Bob Jones"], "v": [1, 2]})
    monkeypatch.setenv("AXIOMROW_SEND_SAMPLE_ROWS", "0")
    assert "Alice Smith" not in dl.get_llm_context(d).split("Columns:")[1].split("sample values")[0] + ""
    assert "Sample rows" not in dl.get_llm_context(d)


def test_overlong_question_is_rejected_without_calling_the_llm(monkeypatch, df):
    monkeypatch.setattr(agent, "call_llm", lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM called")))
    r = agent.answer_question("x" * 5000, "ctx", None, df_columns=list(df.columns), df=df)
    assert "too long" in r["content"]
