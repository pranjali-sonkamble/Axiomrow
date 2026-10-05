# Axiomrow

**Your data. Smarter insights.**

Axiomrow is an AI-assisted data analysis workspace built with Streamlit. Upload a CSV, check its quality, explore it, ask questions in plain English, and export a report. The design goal is that **a number shown to the user is always one that was computed from their data**, never one the language model made up.

> **Live demo:** _add your Streamlit Cloud URL here_
> **Screenshots:** _add 3–4 images to `assets/` and link them here (Home, Data Quality, Chat, Report)_

---

## What it does

| Area | Features |
|---|---|
| **Upload** | CSV upload with clear feedback for empty, header-only, non-CSV and oversized files |
| **Data Quality gate** | Deterministic checks for missing values, duplicates, mixed types, invalid dates, inconsistent categories, outliers and suspicious negatives. Blocking issues stop analysis until fixed, with a step-by-step cleaning plan |
| **Analysis** | Data preview with search and export, per-column profile, and automatically generated insights |
| **Chat with Data** | Natural-language questions answered by a deterministic query engine first, and by the LLM only when needed. Python and SQL results are shown with the answer |
| **Charts** | Bar, line, scatter, pie, histogram and box charts from plain-English requests |
| **Forecasting** | Backtested, deterministic forecasts for next-period questions, with incomplete periods excluded |
| **Reports** | One consistent PDF and Markdown report: overview, quality, statistics, EDA, charts, insights, recommendations and your chat analysis |

---

## How answers stay honest

LLMs are good at language and unreliable at arithmetic. Axiomrow treats the LLM as a translator, not a source of truth:

1. **Deterministic first.** `query_engine.py` answers exact questions (totals, top-N, filters, groupings, comparisons, counts) directly from the DataFrame. No model call is made.
2. **Verified code for everything else.** For other data questions the model must return Python and SQL, which are executed on the real data. The result shown to the user is the *executed output*.
3. **Two independent computations.** The Python and SQL results are compared. If they disagree, no written summary is shown.
4. **Grounding.** Any sentence the model writes next to a result is validated by `answer_grounding.py`. Numbers, names and category values not present in the verified output are rejected, as are causal, statistical-significance and forecast claims that no code established.
5. **Insights are checked too.** Auto-generated insights are validated against a fact sheet computed from every row. Titles and bodies are both checked, model-computed ratios ("double", "twice") are rejected, and words like *outlier* are only allowed when an outlier check actually found some.
6. **Refuse rather than guess.** If the model answers a data question in prose twice, the user sees "I couldn't verify an answer" instead of the guess.

---

## Architecture

```
app.py                  Streamlit UI, navigation, upload handling, session state
theme.py                Colors and global CSS
backend/
  data_loader.py        CSV loading with row/column limits, profiling, LLM context
  data_quality.py       Deterministic quality gate
  query_engine.py       Deterministic answers for exact questions
  forecasting.py        Backtested time-series estimates
  llm_agent.py          LLM client, response parsing, sandboxed Python/SQL execution
  answer_grounding.py   Validation of model-written text
  insight_generator.py  Auto-insights with fact sheet and claim checks
  chart_generator.py    Interactive Plotly charts for chat
  chart_renderer.py     Pillow-only PNG charts for reports (no browser needed)
  report_generator.py   Markdown and PDF report builder
  rate_limiter.py       Per-session and global LLM call limits
  pii_masking.py        Regex PII masking for text sent to the provider
prompts/
  system_prompts.py     Chat and insight prompts
tests/                  pytest suite
.streamlit/             config.toml and secrets.toml.example
```

**Stack:** Python 3.10, Streamlit, pandas, NumPy, DuckDB, Plotly, Pillow, fpdf2, Groq (default) or OpenAI.

---

## Security design

Axiomrow runs model-generated code on user data, so this was treated as a primary concern.

**Code execution sandbox** (`llm_agent.py`)
- Keyword filter, then AST validation, then restricted builtins and an import allowlist.
- File, network and process access through pandas, NumPy and matplotlib is blocked by name (`read_*`, `to_csv`, `loadtxt`, `savefig`, `pd.io`, `os`, `sys` and similar), including aliases.
- `str.format` is blocked because it can walk attributes at runtime.
- Code runs in a separate, killable process with a timeout. On Linux the child also gets: a wiped environment (no API keys), file writes disabled, and CPU and memory limits.

**SQL** (DuckDB)
- Read-only `SELECT`/`WITH`, single statement only.
- `enable_external_access=false` plus locked configuration, so queries cannot read files or URLs. Queries are interrupted after 10 seconds.

**LLM data handling**
- The model receives column names, types, aggregate statistics and limited sample information, never the whole dataset.
- Dataset content is labeled as untrusted data in prompts, and model output is scanned for prompt or credential disclosure.
- A regex PII masker runs on every outgoing request. For sensitive data set `AXIOMROW_SEND_SAMPLE_ROWS=0` to stop raw sample rows being sent at all.

**Abuse and resource limits**
- 2 s minimum gap and 20 LLM calls per 10 minutes per session, plus a global cap (default 600 per hour).
- Retry and summary calls count against the quota.
- Upload limit of 150 MB, 3,000,000 rows and 500 columns, enforced while parsing. Questions are limited to 2,000 characters. Excel upload is disabled by default.

**Application**
- User-controlled strings (filenames, chat titles) are HTML-escaped before rendering.
- Errors shown to users are generic. Details go to the server log, and user question text is not logged.
- XSRF protection on, static serving off, error details hidden, secrets git-ignored.

---

## Getting started

### Requirements
- Python 3.10+
- A Groq or OpenAI API key

### Install
```bash
git clone <your-repo-url>
cd Axiomrow
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### Configure
Copy `.env.example` to `.env` and fill in your key:
```
GROQ_API_KEY=your-key-here
LLM_PROVIDER=groq
```
Use `LLM_PROVIDER=openai` with `OPENAI_API_KEY` to use OpenAI instead.

### Run
```bash
streamlit run app.py
```

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` / `OPENAI_API_KEY` | none | Provider credentials |
| `LLM_PROVIDER` | `groq` | `groq` or `openai` |
| `AXIOMROW_SEND_SAMPLE_ROWS` | `1` | Set `0` to stop sending raw sample rows to the LLM |
| `AXIOMROW_GLOBAL_LLM_CALLS_PER_HOUR` | `600` | Deployment-wide LLM call cap |
| `AXIOMROW_ENABLE_XLSX` | off | Set `1` to allow Excel loading in the loader (the UI accepts CSV only) |

---

## Deploying to Streamlit Community Cloud

1. Push the repository to GitHub. Confirm `.env` and `.streamlit/secrets.toml` are not tracked.
2. Create the app, pointing at `app.py`.
3. In **Settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your-key-here"
   LLM_PROVIDER = "groq"
   ```
   `app.py` copies these into the environment before the backend loads. A `.env` file takes priority locally, and placeholder values are ignored.
4. Run one upload-to-report pass on the live URL.

---

## Tests

```bash
python -m pytest tests -q
```

The suite covers the query engine, forecasting, grounding, the data-quality gate, charts, report generation, sandbox escape attempts, upload limits and feedback, rate limiting, and insight accuracy.

Note: `test_worker_environment_is_wiped` requests the `fork` multiprocessing context, which Windows does not provide. It fails on Windows for that reason and passes on Linux.

CI: `.github/workflows/security.yml` runs `pip-audit`, `bandit` and the tests.

---

## Performance

Measured on a Windows laptop with a ~128 MB CSV of ~1.4 million rows and 13 columns:

| Step | Time |
|---|---|
| Upload, profiling, quality check and insights | ~39 s |
| Full PDF and Markdown report | ~38 s |
| Small file (under 1 KB) | ~1 s |

Time scales roughly linearly with rows.

---

## Known limitations

- **The sandbox is layered defense, not an OS-level boundary.** The name filter is a denylist. For untrusted public use, also run the app in a container as a non-root user with no outbound network except to the LLM provider. On Windows only the environment wipe applies. Kernel limits are Linux-only.
- **PII masking is pattern-based.** It catches emails, phone numbers, cards, SSN, Aadhaar and PAN formats, but not free-text names. Use `AXIOMROW_SEND_SAMPLE_ROWS=0` for sensitive data.
- **Data stays in memory for the session.** Nothing is persisted, and large files use correspondingly large memory.
- **Forecasts are simple baselines** (naive, seasonal naive, linear trend) chosen by backtest. They are estimates, not guarantees.
- **CSV only.** A text file renamed to `.csv` will load as a tiny dataset and be judged by the quality gate.
- The theme loads fonts and icons from Google Fonts and jsDelivr. Self-host them if you need zero third-party requests.

---

## License

_Add a license (for example MIT) before publishing._
