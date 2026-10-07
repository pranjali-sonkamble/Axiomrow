# Axiomrow — Implementation Notes

## Request pipeline
1. **Upload** → size/row/column limits enforced while parsing → data-quality gate (`backend/data_quality.py`).
   Blocking issues stop analysis; warnings are shown but do not block.
2. **Question** → in order: deterministic engine (`query_engine.py`) → backtested forecaster (`forecasting.py`)
   → LLM that must return Python + SQL (`llm_agent.py`).
3. **Execution** → AST-validated Python in a killable, environment-scrubbed subprocess; read-only DuckDB with
   external access disabled and configuration locked.
4. **Verification** → Python and SQL results are cross-checked; any sentence the model writes is validated by
   `answer_grounding.py` (numbers, names, causal/prediction claims). Unverifiable answers are refused, not guessed.
5. **Report** → one consistent PDF/Markdown report built from dataframe-verified statistics (`report_generator.py`).

## Testing
- `python -m pytest tests -q` — unit, integration, security and evaluation tests.
- `python -m evaluation.run_eval` — 80-check accuracy / grounding / sandbox evaluation (no network).
- One test (`test_worker_environment_is_wiped`) needs the `fork` start method and is skipped on Windows.

## Known limitations
- The sandbox is layered defence, not OS isolation; run in a container for untrusted public use.
- PII masking is regex-based. Set `AXIOMROW_SEND_SAMPLE_ROWS=0` for sensitive data.
- Forecasts are simple baselines (naive, seasonal naive, linear trend) selected by backtest.
