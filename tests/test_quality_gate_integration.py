from pathlib import Path


APP = Path(__file__).resolve().parents[1] / "app.py"


def test_analysis_calls_are_inside_readiness_guard():
    """The data-quality gate is checked at each downstream call site
    individually (insights generation, quick_stats, tab rendering) rather
    than through one centralized session_state flag — confirmed via
    findstr showing 4 separate `["analysis_ready"]` checks in app.py.
    This asserts the pattern shows up more than once, since a single
    occurrence would mean only one call site is actually guarded."""
    source = APP.read_text(encoding="utf-8")
    occurrences = source.count('["analysis_ready"]')
    assert occurrences >= 3, (
        f"Expected multiple readiness checks guarding separate analysis "
        f"call sites, found {occurrences}."
    )


def test_not_ready_branch_is_before_existing_analysis_tabs():
    """The first readiness check must appear before tabs are rendered,
    so a NOT READY dataset can't reach the tab UI before being blocked."""
    source = APP.read_text(encoding="utf-8")
    first_check = source.find('["analysis_ready"]')
    first_tabs_call = source.find('st.tabs(')
    assert first_check != -1, "No readiness check found in app.py."
    assert first_tabs_call == -1 or first_check < first_tabs_call, (
        "Readiness check appears after tabs are rendered — a NOT READY "
        "dataset could reach the tab UI before being blocked."
    )


def test_upload_identity_supports_same_filename_reupload():
    """A cleaned file re-uploaded under the same filename must be
    treated as new. Matched by (name, size) rather than file_id,
    since file_id isn't reliably available across Streamlit versions.
    Known limitation: a same-size edit would still be missed — see
    handle_csv_upload docstring."""
    source = APP.read_text(encoding="utf-8")
    assert 'uploaded_file.name, uploaded_file.size' in source