"""Upload failure feedback (Gate 6). app.py is a Streamlit script and cannot be
imported, so the real handle_csv_upload source is extracted and run against a
stubbed `st` -- this tests the actual code, not a copy."""
import ast
import logging
import os
import re
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app.py"


class _Slot:
    def __init__(self): self.messages = []
    def error(self, m): self.messages.append(m)


class _St:
    def __init__(self):
        self.session_state = {}
        self.errors = []
    def error(self, m): self.errors.append(m)
    def rerun(self): raise AssertionError("rerun is only expected on success")


class _File:
    def __init__(self, name, data: bytes):
        self.name, self._d, self.size = name, data, len(data)


@pytest.fixture
def env():
    src = APP.read_text(encoding="utf-8")
    tree = ast.parse(src)
    wanted = {"handle_csv_upload"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert nodes, "handle_csv_upload not found in app.py"
    st = _St()
    calls = {"load": 0}

    def fake_load(f):
        calls["load"] += 1
        if f._d.strip() == b"name,amount,date":
            raise ValueError("The CSV file has no data rows (only headers, or completely empty).")
        raise RuntimeError("/secret/path/boom")

    ns = {"st": st, "os": os, "re": re, "logger": logging.getLogger("t"),
          "load_csv": fake_load, "MAX_UPLOAD_ROWS": 10, "MAX_UPLOAD_COLS": 10}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "app.py", "exec"), ns)
    return ns["handle_csv_upload"], st, calls


def test_empty_file_gets_a_message_not_silence(env):
    fn, st, calls = env
    slot = _Slot()
    fn(_File("empty.csv", b""), status_slot=slot)
    assert slot.messages and "empty" in slot.messages[0].lower()
    assert calls["load"] == 0


def test_non_csv_is_refused_with_clear_message_and_never_parsed(env):
    fn, st, calls = env
    slot = _Slot()
    fn(_File("report.doc", b"PK\x03\x04junk"), status_slot=slot)
    assert ".doc" in slot.messages[0] and "csv" in slot.messages[0].lower()
    assert calls["load"] == 0


def test_headers_only_message_goes_to_the_slot(env):
    fn, st, calls = env
    slot = _Slot()
    fn(_File("h.csv", b"name,amount,date"), status_slot=slot)
    assert "no data rows" in slot.messages[0]
    assert st.errors == []          # nothing leaks to the bottom of the page


def test_unexpected_error_is_generic_and_hides_internals(env):
    fn, st, calls = env
    slot = _Slot()
    fn(_File("x.csv", b"a,b\n1,2\n"), status_slot=slot)
    assert "/secret/path" not in slot.messages[0] and "boom" not in slot.messages[0]


def test_rerun_with_same_bad_file_reshows_message_without_reparsing(env):
    fn, st, calls = env
    f = _File("h.csv", b"name,amount,date")
    fn(f, status_slot=_Slot())
    slot2 = _Slot()
    fn(f, status_slot=slot2)
    assert calls["load"] == 1 and "no data rows" in slot2.messages[0]


def test_without_slot_falls_back_to_st_error_for_settings_page(env):
    fn, st, calls = env
    fn(_File("h.csv", b"name,amount,date"))
    assert "no data rows" in st.errors[0]
