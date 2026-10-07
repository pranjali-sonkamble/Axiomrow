"""Axiomrow evaluation harness.

Measures, WITHOUT any network or LLM call:
  1. Deterministic engine accuracy: each question's answer is compared to a
     value computed independently with plain pandas.
  2. Forecast routing: future-looking questions reach the backtested
     forecaster and never the LLM.
  3. Grounding catch rate: fabricated / over-claiming sentences must be
     rejected, faithful sentences must be accepted.
  4. Sandbox: generated Python/SQL that tries to read files or secrets must
     be refused, while legitimate analysis must still run.

Run:   python -m evaluation.run_eval          (prints a report, exits 1 on any miss)
CI:    tests/test_evaluation.py enforces the same thresholds.
"""
import re
import sys
import numpy as np
import pandas as pd

import backend.llm_agent as agent
from backend.answer_grounding import content_is_untrustworthy, build_known_values


def make_dataset(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 240
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n, freq="3D"),
        "product": rng.choice(["Laptop", "Phone", "Tablet", "Headphones"], n),
        "region": rng.choice(["North", "South", "East", "West"], n),
        "sales_rep": rng.choice(["Asha", "Ravi", "Meera"], n),
        "revenue": rng.integers(1_000, 90_000, n),
        "quantity": rng.integers(1, 50, n),
    })


def nums(text):
    return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", str(text)) if x.strip(",")]


def has_number(text, expected, tol=0.01):
    return any(abs(v - expected) <= max(tol, abs(expected) * 1e-9) for v in nums(text))


def mentions(text, word):
    return word.lower() in str(text).lower()


def first_word(text):
    return str(text).strip().split(" ")[0]


def build_cases(df):
    """(category, question, check(content) -> bool). Expectations come from pandas, not the engine."""
    g = lambda col, by="revenue": df.groupby(col)[by].sum().sort_values(ascending=False)
    cases = []
    A = cases.append

    A(("aggregate", "What is the total revenue?", lambda c: has_number(c, df.revenue.sum())))
    A(("aggregate", "What is the total quantity?", lambda c: has_number(c, df.quantity.sum())))
    A(("aggregate", "What is the average revenue?", lambda c: has_number(c, round(df.revenue.mean(), 2), 0.01)))
    A(("aggregate", "What is the average quantity?", lambda c: has_number(c, round(df.quantity.mean(), 2), 0.01)))
    A(("aggregate", "How many rows are there?", lambda c: has_number(c, len(df))))
    for p in ["Laptop", "Phone", "Tablet", "Headphones"]:
        A(("filtered", f"What is the total revenue for {p}?",
           lambda c, p=p: has_number(c, df.loc[df["product"] == p, "revenue"].sum())))
    for r in ["North", "South", "East", "West"]:
        A(("filtered", f"What is the total revenue for {r}?",
           lambda c, r=r: has_number(c, df.loc[df["region"] == r, "revenue"].sum())))
    for y in (2023, 2024):
        A(("filtered", f"What is the total revenue in {y}?",
           lambda c, y=y: has_number(c, df.loc[df["date"].dt.year == y, "revenue"].sum())))

    A(("ranking", "Which product has the highest revenue?", lambda c: first_word(c) == g("product").index[0]))
    A(("ranking", "Which product has the lowest revenue?", lambda c: first_word(c) == g("product").index[-1]))
    A(("ranking", "Which region has the highest revenue?", lambda c: first_word(c) == g("region").index[0]))
    A(("ranking", "Which region has the lowest revenue?", lambda c: first_word(c) == g("region").index[-1]))
    A(("ranking", "Which sales rep has the highest revenue?", lambda c: first_word(c) == g("sales_rep").index[0]))
    A(("ranking", "Which sales rep has the lowest revenue?", lambda c: first_word(c) == g("sales_rep").index[-1]))
    for n in (2, 3):
        A(("ranking", f"Top {n} products by revenue", lambda c, n=n: all(
            re.search(rf"{i + 1}\.\s*{re.escape(name)}", c) for i, name in enumerate(g("product").index[:n]))))
        A(("ranking", f"Top {n} regions by revenue", lambda c, n=n: all(
            re.search(rf"{i + 1}\.\s*{re.escape(name)}", c) for i, name in enumerate(g("region").index[:n]))))

    for col, word in [("product", "product"), ("region", "region"), ("sales_rep", "sales rep")]:
        A(("counting", f"How many {word}s are there?", lambda c, col=col: has_number(c, df[col].nunique())))
        A(("counting", f"How many unique {word}s are there?", lambda c, col=col: has_number(c, df[col].nunique())))

    for col, by in [("product", "product"), ("region", "region"), ("sales_rep", "sales rep")]:
        A(("grouping", f"Show revenue by {by}", lambda c, col=col: all(
            has_number(c, v) and mentions(c, k) for k, v in g(col).items())))
        A(("grouping", f"Calculate average revenue by {by}", lambda c, col=col: all(
            has_number(c, round(v, 2), 0.51) for v in df.groupby(col)["revenue"].mean())))

    A(("numeric_filter", "Show rows where revenue > 85000",
       lambda c: has_number(c, len(df[df.revenue > 85000])) or all(mentions(c, str(v)) for v in df[df.revenue > 85000].revenue.head(3))))
    A(("numeric_filter", "Show rows where revenue between 1000 and 1500",
       lambda c: (len(df[df.revenue.between(1000, 1500)]) == 0 and mentions(c, "no ")) or
                 all(mentions(c, str(v)) for v in df[df.revenue.between(1000, 1500)].revenue.head(3))))

    A(("membership", "Is there any Laptop in the product column?", lambda c: first_word(c).startswith("Yes")))
    A(("membership", "Is there any Monitor in the product column?", lambda c: first_word(c).startswith("No")))

    A(("quality", "How many duplicate rows are there?", lambda c: has_number(c, int(df.duplicated().sum()))))
    A(("quality", "Are there any negative revenue values?", lambda c: mentions(c, "no negative")))
    return cases


def build_unverifiable_cases():
    """Questions the deterministic engine must NOT answer with a made-up number:
    it must decline (None) so the LLM/code path computes them from the data."""
    return [
        "Why did revenue drop?",
        "What caused the increase in sales?",
        "Tell me about the revenue trend",
        "Which sales rep handled the most orders?",   # no order-count column
    ]


GROUNDING = [  # (sentence, verified_output, should_be_rejected)
    ("There are 4 distinct products: Laptop, Phone, Tablet and Monitor.",
     "Unique products (4): ['Headphones', 'Laptop', 'Phone', 'Tablet']", True),
    ("Total revenue is 999,999.", "Total revenue: 465,000", True),
    ("The increase was caused by a marketing campaign.", "Feb 120000 -> Mar 180000", True),
    ("This difference is statistically significant.", "India 950000, UAE 410000", True),
    ("Revenue will definitely grow next quarter.", "Total revenue: 465,000", True),
    ("I don't have access to the actual data.", "Total revenue: 465,000", True),
    ("The 1,300 transaction is an anomaly.", "Max revenue: 1300", True),
    ("India is the clear winner with 3x more sales than Japan.", "India 1200, UAE 340", True),
    ("Laptop 40.0 and North 30.0", "Laptop 30.0\nNorth 40.0", True),
    ("The regions are East, North and South.", "Unique regions (2): ['East', 'North']", True),
    ("Total revenue is 465,000.", "Total revenue: 465,000", False),
    ("There are 4 distinct products: Headphones, Laptop, Phone and Tablet.",
     "Unique products (4): ['Headphones', 'Laptop', 'Phone', 'Tablet']", False),
    ("India has 1,200 transactions versus 340 for UAE.", "India 1200\nUAE 340", False),
    ("The increase is primarily associated with the North region.", "North +38000, South +15000", False),
]

SANDBOX_ATTACKS = [
    ("python", "print(pd.read_csv('/etc/passwd'))"),
    ("python", "df.to_csv('/tmp/leak.csv')"),
    ("python", "print(pd.io.common.os.environ)"),
    ("python", "import os\nprint(os.environ)"),
    ("python", "print(().__class__.__bases__[0].__subclasses__())"),
    ("python", "print(eval('1+1'))"),
    ("python", "plt.savefig('/tmp/x.png')"),
    ("python", "s='{0.__cla'+'ss__}'\nprint(s.format(df))"),
    ("sql", "SELECT * FROM read_text('/etc/passwd')"),
    ("sql", "SELECT * FROM read_csv('/etc/passwd')"),
    ("sql", "SELECT * FROM glob('/*')"),
    ("sql", "DROP TABLE df"),
    ("sql", "SELECT 1; SELECT 2"),
]


def _no_llm(*a, **k):
    raise AssertionError("LLM was called")


def run():
    df = make_dataset()
    agent.call_llm = _no_llm  # any LLM call during the deterministic section is a failure
    results = {}

    def record(cat, ok, detail=""):
        results.setdefault(cat, []).append((ok, detail))

    for cat, q, check in build_cases(df):
        try:
            r = agent.answer_question(q, "ctx", None, df_columns=list(df.columns), df=df)
            content = r.get("content", "") if isinstance(r, dict) else str(r)
            ok = r.get("answer_type") == "text" and bool(check(content))
            record(cat, ok, "" if ok else f"{q!r} -> {content[:90]!r}")
        except AssertionError:
            record(cat, False, f"{q!r} reached the LLM (expected deterministic answer)")
        except Exception as e:
            record(cat, False, f"{q!r} raised {type(e).__name__}: {e}")

    from backend.query_engine import query_dataframe
    for q in build_unverifiable_cases():
        r = query_dataframe(q, df)
        total = f"{int(df.revenue.sum()):,}"
        ok = r is None or total not in str(r.get("content", ""))
        record("no_false_totals", ok, "" if ok else f"{q!r} was answered with the grand total")

    for q in ["predict next 3 months revenue", "forecast revenue for the next 6 months",
              "What will revenue be next month?"]:
        try:
            r = agent.answer_question(q, "ctx", None, df_columns=list(df.columns), df=df)
            ok = r.get("answer_type") == "forecast" and r.get("verified") is True
            record("forecast_routing", ok, "" if ok else f"{q!r} -> {r.get('answer_type')}")
        except AssertionError:
            record("forecast_routing", False, f"{q!r} reached the LLM")

    # Too little history must produce an honest refusal, never an invented number.
    r = agent.answer_question("forecast revenue for the next quarter", "ctx", None,
                              df_columns=list(df.columns), df=df)
    ok = r.get("answer_type") == "text" and "not enough" in r.get("content", "").lower()
    record("forecast_refuses_thin_history", ok, "" if ok else f"quarter forecast -> {r}")

    known = build_known_values(df)
    for sentence, verified, should_reject in GROUNDING:
        rejected = content_is_untrustworthy(sentence, verified, "", known)
        record("grounding", rejected == should_reject,
               "" if rejected == should_reject else
               f"{'MISSED fabrication' if should_reject else 'FALSE REJECTION'}: {sentence[:70]!r}")

    for kind, payload in SANDBOX_ATTACKS:
        out, err = (agent.execute_code(payload, df) if kind == "python" else agent.execute_sql(payload, df))
        ok = out is None and err is not None
        record("sandbox_attacks_blocked", ok, "" if ok else f"{kind} attack succeeded: {payload[:60]!r} -> {str(out)[:60]!r}")
    for kind, payload, expect in [("python", "print(df['revenue'].sum())", str(int(df.revenue.sum()))),
                                  ("sql", "SELECT SUM(revenue) FROM df", str(int(df.revenue.sum())))]:
        out, err = (agent.execute_code(payload, df) if kind == "python" else agent.execute_sql(payload, df))
        ok = err is None and str(out).startswith(expect)
        record("sandbox_legit_allowed", ok, "" if ok else f"{kind} legit query failed: {err or out}")
    return results


def report(results):
    total = sum(len(v) for v in results.values())
    passed = sum(ok for v in results.values() for ok, _ in v)
    print(f"\nAXIOMROW EVALUATION  ({passed}/{total} passed)\n" + "=" * 52)
    for cat, items in results.items():
        p = sum(ok for ok, _ in items)
        print(f"{cat:<26}{p:>3}/{len(items):<3} {100 * p / len(items):6.1f}%")
    fails = [d for v in results.values() for ok, d in v if not ok]
    if fails:
        print("\nFAILURES:")
        for d in fails:
            print(" -", d)
    return passed, total


if __name__ == "__main__":
    p, t = report(run())
    sys.exit(0 if p == t else 1)
