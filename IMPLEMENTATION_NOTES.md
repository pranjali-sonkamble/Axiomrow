# Axiomrow — NO / NOT ANALYSIS-READY Branch

## Changed files
- `app.py` — minimum integration required to place the quality gate before the existing analysis entry point.
- `backend/data_quality.py` — deterministic, read-only profiling and readiness logic.
- `tests/test_data_quality.py` — data-quality/readiness tests.
- `tests/test_quality_gate_integration.py` — checks that downstream analysis is guarded and same-name re-upload identity is supported.

## Gate behavior
1. Upload is loaded once.
2. Existing Axiomrow profile is attempted.
3. Deterministic quality profiling runs.
4. Critical/High findings block analysis.
5. Warning/Info findings are shown but do not automatically block.
6. On NOT READY, downstream LLM context, insights and quick stats are not generated, and the existing analysis tabs are not rendered.
7. On READY, the existing downstream calls are executed unchanged.
8. A new upload is detected using Streamlit's `file_id` when available, so a cleaned file can be re-uploaded even with the same filename.
9. The uploaded dataframe is never modified by the quality layer.

## Verification
- `python -m py_compile app.py` — passed.
- New quality/integration tests: **15 passed**.
- The full pre-existing project test suite could not be executed in this environment because the complete project source tree was not available as a mounted project; only the current `app.py` and selected saved project artifacts were available for implementation.
