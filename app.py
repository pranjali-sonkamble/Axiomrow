# app.py — Axiomrow (Dark Sidebar SaaS visual language)
import streamlit as st
import pandas as pd
import uuid
import io
import base64
import html
import os
import re
import time
import logging
import threading
from datetime import datetime

# ── PAGE CONFIG — must be the FIRST Streamlit command ─────────────
st.set_page_config(
    page_title="Axiomrow",
    page_icon=None,
    layout="wide"
)

# ── LOGGING — real errors go here (visible in Streamlit Cloud's log
#    viewer / local console); the UI only ever shows a clean, generic
#    message. Keeps internal exception text (library internals, file
#    paths, stack traces) out of what an end user sees.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("axiomrow")

# ── SECRETS BRIDGE — must run before any backend import ────────────
# backend/llm_agent.py reads its API key with plain os.getenv(), which
# works great locally via a .env file (python-dotenv). On Streamlit
# Cloud there is no .env file — secrets are entered in the dashboard
# and surface as st.secrets, NOT automatically as environment
# variables. Rather than teach every backend module about Streamlit,
# we bridge the two here, once, at the single entry point that already
# knows about both: copy any matching keys from st.secrets into
# os.environ before backend.llm_agent (which calls load_dotenv() at
# import time) is imported below.
#
# setdefault() means an existing local .env value always wins if one
# is somehow already present — st.secrets only fills gaps, never
# overrides a value that's already set.
try:
    _secrets_paths = [
        os.path.join(os.path.expanduser("~"), ".streamlit", "secrets.toml"),
        os.path.join(os.getcwd(), ".streamlit", "secrets.toml"),
    ]
    # Load .env FIRST so a real local key always wins, as documented above.
    # (load_dotenv never overrides variables that are already set, so the
    # order matters: the bridge below must not run before it.)
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()

    def _usable_secret(v) -> bool:
        """Reject empty values and the placeholders shipped in
        secrets.toml.example / .env.example."""
        v = str(v).strip().lower()
        return bool(v) and not any(p in v for p in ("your_", "api_key", "changeme", "xxxx"))

    if any(os.path.isfile(p) for p in _secrets_paths):
        for _key in ("GROQ_API_KEY", "OPENAI_API_KEY", "LLM_PROVIDER"):
            if _key in st.secrets and _usable_secret(st.secrets[_key]):
                os.environ.setdefault(_key, str(st.secrets[_key]))
except Exception:
    pass

import sys
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from backend.insight_generator import generate_insights, get_quick_stats
from backend.chart_generator import generate_chart
from backend.data_loader import load_csv, get_data_profile, get_llm_context
from backend.report_generator import (
    generate_markdown_report, generate_pdf_report, prepare_report_analysis,
    reconstruct_chat_chart, render_chart_config_png, render_forecast_png,
)
try:
    from backend.data_quality import assess_data_quality
except ImportError:
    # Fallback: data_quality.py placed next to app.py (root level)
    from data_quality import assess_data_quality



# Pre-warm heavy, otherwise lazily-imported dependencies (groq/openai,
# duckdb, fpdf2 + fontTools) in the background as soon as the app starts.
# These imports are already correctly deferred to their actual point of
# use (a real perf win on its own — it means a user who never touches
# Chat or Reports never pays for them). But that also means the FULL
# cold-import cost — measured at ~0.26s groq, ~0.37s openai, ~0.37s fpdf
# — landed entirely on whichever action the user happened to do first,
# which is what made a fresh run's first response feel slow compared to
# every response after it (once imported, Python caches the module for
# the rest of the process). Running it here overlaps that cost with the
# time the user naturally spends uploading a file and reading the Data
# Quality tab, instead of adding it to their first click in Chat.
# threading (not asyncio) because this only needs to run once in the
# background without blocking Streamlit's own script execution; daemon=True
# so it can never prevent the process from exiting. Guarded by session
# state so it fires once per session, not on every Streamlit rerun.
if "warmup_started" not in st.session_state:
    st.session_state.warmup_started = True
    from backend.llm_agent import warm_up_heavy_imports
    threading.Thread(target=warm_up_heavy_imports, daemon=True).start()

# ── UI: theme (colors + global CSS) now lives in theme.py — see that
# file's docstring for why. T is still the single source of truth for
# color values; it's just imported rather than defined here.
from theme import T, apply_theme


# ── ICON SYSTEM ─────────────────────────────────────────────────────
BOOTSTRAP_ICONS = {
    "brand": "graph-up-arrow",
    "user": "person",
    "bolt": "lightning-charge",
    "plus": "plus-lg",
    "message": "chat-left-text",
    "trending": "graph-up",
    "barchart": "bar-chart",
    "file": "file-earmark-text",
    "file-check": "file-earmark-check",
    "generate": "file-earmark-arrow-up",
    "folder": "folder",
    "trash": "trash3",
    "search": "search",
    "code": "code-slash",
    "chevron": "chevron-down",
    "chevronright": "chevron-right",
    "home": "house",
    "settings": "gear",
    "cloud": "cloud",
    "database": "database",
    "upload": "upload",
}

def icon_svg(name: str, size: int = 15, color: str = "currentColor", stroke_width: str = "1.8") -> str:
    icon = BOOTSTRAP_ICONS.get(name, name)
    return (
        f'<i class="bi bi-{icon}" style="font-size:{size}px;color:{color};'
        f'vertical-align:-.08em;line-height:1;flex-shrink:0;"></i>'
    )


BOOTSTRAP_AVATAR_PATHS = {
    "user": "M150 150Q173 150 189.5 166.5Q206 183 206 206.5Q206 230 189.5 246Q173 262 150 262.5Q127 263 110.5 246.5Q94 230 94 206.5Q94 183 110.5 166.5Q127 150 150 150ZM188 206Q188 191 177 180Q166 169 150 169Q134 169 123 180Q112 191 112.5 206.5Q113 222 123.5 233Q134 244 150 244Q166 244 177 233Q188 222 188 206ZM263 56Q263 48 258.5 44Q254 40 249 38H244H56H51Q46 40 42 44Q38 48 38 56Q38 69 46 84Q57 104 79 116Q107 131 150 131Q193 131 221 116Q243 104 254 84Q263 69 263 56ZM244 56Q244 62 241 69Q236 79 228 88Q217 98 201 105Q179 113 150 113Q121 113 99 105Q83 98 72 88Q64 79 59 69Q56 62 56 56Z",
    "brand": "M0 300H19V19H300V0H0ZM188 234Q188 238 190.5 241Q193 244 197 244H272Q276 244 278.5 241Q281 238 281 234V159Q281 155 278.5 152.5Q276 150 272 150Q268 150 265 152.5Q262 155 262 159V208L195 125Q192 122 188 122Q184 122 181 125L132 173L64 79Q61 76 57.5 75.5Q54 75 51 77Q48 79 47.5 83Q47 87 49 90L124 193Q126 197 130.5 197Q135 197 138 194L187 145L252 225H197Q193 225 190 227.5Q187 230 187 234Z",
}

def build_avatar_data_uri(icon_name: str, bg: str) -> str:
    """Builds a small rounded-square avatar image (icon on a colored
    background) as a base64 SVG data URI, for use as a chat_message
    avatar. `bg` can be a solid hex color or an SVG paint reference
    like 'url(#grad)' when a <defs> gradient is included in `bg`."""
    path = BOOTSTRAP_AVATAR_PATHS.get(icon_name, "")
    defs = ""
    fill = bg
    if bg == "gradient":
        defs = (
            '<defs><linearGradient id="g1" x1="0" y1="0" x2="1" y2="1">'
            f'<stop offset="0" stop-color="{T["primary"]}"/>'
            f'<stop offset="1" stop-color="{T["accent2"]}"/>'
            '</linearGradient></defs>'
        )
        fill = "url(#g1)"
        stroke = "white"
    else:
        stroke = T["text_secondary"]

    # Streamlit renders the avatar SVG as an image. Avoid nested SVG/currentColor
    # here: some browser/Streamlit combinations render the rounded background
    # but drop the nested icon. Put the path directly in the root SVG and give
    # it an explicit fill so both avatars are always visible.
    icon_color = "#FFFFFF" if bg == "gradient" else T["text_secondary"]
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">'
        f'{defs}<rect width="64" height="64" rx="18" fill="{fill}"/>'
        # The source icon paths use a 300x300 coordinate system. Scale them
        # into the 24x24 icon area and flip the Y-axis because these source
        # paths are authored in the opposite vertical orientation.
        f'<g transform="translate(20,20) scale(0.08,-0.08) translate(0,-300)">'
        f'<path d="{path}" fill="{icon_color}" stroke="none"/>'
        f'</g></svg>'
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("utf-8")


AVATAR_USER = build_avatar_data_uri("user", T["border_strong"])
AVATAR_ASSISTANT = build_avatar_data_uri("brand", "gradient")


def render_code_result(t: dict, code: str, code_result: str, execution_time: float = None):
    """Renders the result box for a successfully-executed Python code
    answer. The code itself is shown by the caller (render_code_boxes),
    once, regardless of whether execution succeeded or failed."""

    if code_result:
        timing_html = (
            f"<span style='color:{t['muted']};margin-left:12px;'>"
            f"Execution time: {execution_time:.4f} s</span>"
            if execution_time is not None else ""
        )

        text = str(code_result)
        if "\n" in text:
            # Multi-line output (overviews, tables, several labelled
            # values): HTML collapses newlines, which turned a whole
            # overview into one run-on line. Render it as a pre-wrapped
            # block under the label instead. Blank lines become &nbsp; so
            # markdown can't end the HTML block early.
            body = "\n".join(
                html.escape(line) if line.strip() else "&nbsp;"
                for line in text.splitlines()
            )
            result_html = (
                f"<pre style=\"margin:6px 0 0 0;padding:0;background:transparent;"
                f"white-space:pre-wrap;word-break:break-word;"
                f"font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;"
                f"font-size:0.85rem;line-height:1.55;color:{t['text']};\">{body}</pre>"
            )
        else:
            result_html = (
                f"<span style=\"color:{t['text']};margin-left:5px;\">"
                f"{html.escape(text)}</span>"
            )

        st.markdown(
            f"<div style=\"background:{t['success_bg']};"
            f"border-left:3px solid {t['success']};border-radius:0 0 6px 6px;"
            f"padding:9px 13px;margin-top:-10px;margin-bottom:24px;font-size:0.88rem;\">"
            f"<b style=\"color:{t['success']}\">Python result:</b>"
            f"{timing_html}{result_html}</div>",
            unsafe_allow_html=True
        )


def render_sql_result(t: dict, sql: str, sql_result: str, execution_time: float = None):
    """Renders SQL query + attached result box."""

    with st.expander("View SQL query"):
        st.code(sql, language="sql")

    if sql_result:
        is_table = "\n" in sql_result

        if is_table:
            # One explicitly-styled <pre> instead of st.code: st.code's
            # text lexer/theme CSS rendered the header line at a different
            # size than the rows. Blank lines are replaced with &nbsp; so
            # markdown can't end the HTML block early.
            safe = "\n".join(
                html.escape(line) if line.strip() else "&nbsp;"
                for line in str(sql_result).splitlines()
            )
            st.markdown(
                f"<pre style=\"background:{t['soft_surface']};"
                f"border-left:3px solid {t['primary']};border-radius:0 6px 6px 0;"
                f"padding:10px 13px;margin:0 0 6px 0;overflow-x:auto;"
                f"font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;"
                f"font-size:0.85rem;line-height:1.5;color:{t['text']};"
                f"white-space:pre;\">{safe}</pre>",
                unsafe_allow_html=True,
            )

            if execution_time is not None:
                st.caption(
                    f"Execution time: {execution_time:.4f} s"
                )

        else:
            timing_html = (
                f"<span style='color:{t['muted']};margin-left:12px;'>"
                f"Execution time: {execution_time:.4f} s</span>"
                if execution_time is not None else ""
            )

            st.markdown(
                f"""
                <div style="
                    background:{t['soft_surface']};
                    border-left:3px solid {t['primary']};
                    border-radius:0 0 6px 6px;
                    padding:9px 13px;
                    margin-top:-10px;
                    font-size:0.88rem;
                ">
                    <b style="color:{t['primary']}">SQL result:</b>
                    <span style="color:{t['text']};margin-left:5px;">
                        {html.escape(str(sql_result))}
                    </span>
                    {timing_html}
                </div>
                """,
                unsafe_allow_html=True
            )

# ── CUSTOM TABLE / DATA RENDERERS ──────────────────────────────────
# WHY: st.dataframe() and st.json() are canvas/widget-based components
# that read colors from the OS/browser's dark-mode preference in some
# Streamlit versions, ignoring both our CSS and config.toml. Rendering
# plain HTML instead guarantees they always match the app's theme.

# Answer-validation helpers live in backend/answer_grounding.py (pure
# Python, unit-tested). Old private names are kept as aliases so the rest
# of this file reads exactly as before.
from backend.answer_grounding import (
    content_is_untrustworthy as _content_is_untrustworthy,
    trim_framing_text as _trim_framing_text,
    build_known_values as _build_known_values,
    results_disagree as _results_disagree,
    UNTRUSTWORTHY_FRAMING_FALLBACK as _UNTRUSTWORTHY_FRAMING_FALLBACK,
    RESULTS_DISAGREE_MESSAGE as _RESULTS_DISAGREE_MESSAGE,
    NO_VERIFIED_RESULT_MESSAGE as _NO_VERIFIED_RESULT_MESSAGE,
)


def render_code_boxes(t: dict, msg: dict):
    """Draws the verified Python/SQL boxes (and any execution errors) for
    a code-type message. Used by BOTH the live render and history replay,
    so a message looks identical the first time and on every rerun."""
    if msg.get("code"):
        # The code is shown even when it failed — previously the error
        # message replaced the code entirely, so there was no way to see
        # what was actually attempted (e.g. to tell a genuine "wrong
        # column name" failure apart from an unrelated bug in the
        # generated code).
        with st.expander("View Python code"):
            st.code(msg["code"], language="python")
        if msg.get("code_error"):
            st.error(f"Python: {msg['code_error']}")
        else:
            render_code_result(t, msg["code"], msg.get("code_result"), msg.get("code_execution_time"))
    if msg.get("sql"):
        if msg.get("sql_error"):
            st.warning(f"SQL: {msg['sql_error']}")
        else:
            render_sql_result(t, msg["sql"], msg.get("sql_result"), msg.get("sql_execution_time"))


def render_framing(msg: dict, columns=None):
    """Draws the written sentence under the verified boxes. Re-validated
    at read time (cheap checks only — no DataFrame scan) as
    defense-in-depth, so a message stored without validation, or saved
    before a rule existed, still can't show a contradicted sentence."""
    framing = msg.get("framing")
    if not framing:
        return
    if msg.get("disagree"):
        st.warning(framing)
        return
    if msg.get("verified") is not False and framing != _UNTRUSTWORTHY_FRAMING_FALLBACK:
        allowed = f"{msg.get('question') or ''} " + " ".join(str(c) for c in (columns or []))
        if _content_is_untrustworthy(framing, msg.get("answer", ""), allowed):
            framing = _UNTRUSTWORTHY_FRAMING_FALLBACK
    st.caption(framing)


def _build_framing(question, model_sentence: str, verified_text: str,
                   allowed_text: str, known_values) -> str:
    """Chooses the sentence shown under a verified result.

    Try the sentence the model already wrote alongside the code FIRST —
    it cost nothing extra, it's already sitting in `result`. Only if that
    fails validation do we pay for a second LLM call that re-words the
    answer FROM the verified output. This was previously reversed (always
    calling the model a second time before even checking the first
    sentence), which doubled the network round-trip on every single data
    question — that's the "taking long to load" regression.
    Either candidate is shown only if it passes validation; if neither
    does, a neutral message is shown — never a wrong sentence, never
    silence."""
    text = _trim_framing_text(model_sentence or "")
    if text and not _content_is_untrustworthy(text, verified_text, allowed_text, known_values):
        return text

    if question:
        from backend.llm_agent import summarize_verified_result
        with st.spinner("Verifying answer..."):
            retry = _trim_framing_text(summarize_verified_result(question, verified_text, session_id=st.session_state.get('session_id')) or "")
        if retry and not _content_is_untrustworthy(retry, verified_text, allowed_text, known_values):
            return retry

    return _UNTRUSTWORTHY_FRAMING_FALLBACK



def _looks_like_chart_request(question: str) -> bool:
    q = str(question or "").strip().lower()
    return bool(re.search(r"\b(chart|plot|graph|visuali[sz]e|bar chart|line chart|pie chart|scatter plot|histogram|trend)\b", q))

def process_answer_result(raw_result, df: pd.DataFrame, t: dict, render_inline: bool,
                          question: str = None) -> dict:
    """
    The single place that turns answer_question()'s return value into
    a fully-populated chat message: normalizes the result, executes
    any Python/SQL code, builds any requested chart, and (optionally)
    renders it inline.

      render_inline=True  -> also draws the response into the current
                             script run (live chat input)
      render_inline=False -> builds the message only; the caller reruns
                             and history replay draws it (suggestion buttons)

    `question` is the user's question text; it lets the written summary
    be generated FROM the verified output (see _build_framing).

    Trust model for what a message may contain:
      * answer  — verified result text only (also the follow-up context)
      * framing — one validated sentence, or a neutral fallback
      * verified — False when nothing here should be fed back to the
                   model as fact (errors, refusals, Python/SQL mismatch)
    """
    from backend.llm_agent import execute_code, execute_sql

    if isinstance(raw_result, str):
        result = {"answer_type": "text", "content": raw_result}
    elif not isinstance(raw_result, dict):
        result = {"answer_type": "text", "content": str(raw_result)}
    else:
        result = raw_result

    answer_type = result.get("answer_type", "text")
    new_msg = {
        "question": question,  # callers also set this; kept for validation
        "answer": result.get("content", "I couldn't generate a response."),
        "framing": None,
        "verified": True,
        "disagree": False,
        "type": answer_type,
        "code": None, "code_result": None, "code_error": None, "code_execution_time": None,
        "sql": None, "sql_result": None, "sql_error": None, "sql_execution_time": None,
        "chart_config": None,
        "chart_bytes": None, "chart_title": None,
    }

    if answer_type == "text":
        content = result.get("content", "")
        if render_inline:
            st.write(content)
        new_msg["answer"] = content
        if result.get("unverified") or str(content).startswith("Error:"):
            new_msg["verified"] = False

    elif answer_type == "code":
        code_str = result.get("code")
        sql_str = result.get("sql")
        summary_parts = []

        # Execute FIRST, then display: the model's sentence used to be a
        # prediction written before any code ran. Verified results lead.
        if code_str:
            _start = time.perf_counter()
            code_output, code_error = execute_code(code_str, df)
            new_msg["code"] = code_str
            new_msg["code_execution_time"] = time.perf_counter() - _start
            if code_error:
                new_msg["code_error"] = code_error
            else:
                new_msg["code_result"] = code_output
                summary_parts.append(f"Python result: {code_output}")

        if sql_str:
            _start = time.perf_counter()
            sql_output, sql_error = execute_sql(sql_str, df)
            new_msg["sql"] = sql_str
            new_msg["sql_execution_time"] = time.perf_counter() - _start
            if sql_error:
                new_msg["sql_error"] = sql_error
            else:
                new_msg["sql_result"] = sql_output
                summary_parts.append(f"SQL result: {sql_output}")

        if render_inline:
            render_code_boxes(t, new_msg)

        verified_text = "\n".join(summary_parts)

        if not code_str and not sql_str:
            # Model promised code but sent none: its prose is unverified.
            new_msg["answer"] = _NO_VERIFIED_RESULT_MESSAGE
            new_msg["framing"] = None
            new_msg["verified"] = False
            if render_inline:
                st.warning("The model said it would generate code but didn't return any.")
                st.write(_NO_VERIFIED_RESULT_MESSAGE)

        elif not summary_parts:
            # Everything that was attempted failed to execute.
            new_msg["answer"] = _NO_VERIFIED_RESULT_MESSAGE
            new_msg["framing"] = _NO_VERIFIED_RESULT_MESSAGE
            new_msg["verified"] = False
            if render_inline:
                st.caption(_NO_VERIFIED_RESULT_MESSAGE)

        elif _results_disagree(new_msg.get("code_result"), new_msg.get("sql_result")):
            # Two independent computations of the same question disagree:
            # one of them is wrong, so no sentence and no follow-up context.
            new_msg["answer"] = verified_text
            new_msg["framing"] = _RESULTS_DISAGREE_MESSAGE
            new_msg["disagree"] = True
            new_msg["verified"] = False
            logger.warning("Python/SQL disagreement (question length %d)", len(question or ""))
            if render_inline:
                st.warning(_RESULTS_DISAGREE_MESSAGE)

        else:
            allowed_text = f"{question or ''} " + " ".join(str(c) for c in df.columns)
            framing = _build_framing(
                question, result.get("content", ""), verified_text,
                allowed_text, _build_known_values(df),
            )
            # "answer" carries ONLY the verified result (follow-up context
            # + report generation); the sentence is stored separately so an
            # imperfect one can never leak into what the model sees later.
            new_msg["answer"] = verified_text
            new_msg["framing"] = framing
            if render_inline:
                st.caption(framing)

    elif answer_type == "chart":
        if render_inline and result.get("content"):
            st.write(result["content"])

        chart_config = result.get("chart_config", {})
        new_msg["chart_config"] = chart_config
        fig, error = generate_chart(df, chart_config, theme=t)

        if error:
            if render_inline:
                st.warning(f"Could not generate chart: {error}")
        else:
            if render_inline:
                st.plotly_chart(fig, use_container_width=True)
            try:
                # Pillow renderer: no kaleido/Chrome dependency.
                img_bytes = render_chart_config_png(df, chart_config)
                if not img_bytes:
                    raise ValueError("chart image unavailable")
                if render_inline:
                    st.download_button(
                        label="Download chart as PNG",
                        data=img_bytes,
                        file_name=f"chart_{chart_config.get('type', 'chart')}.png",
                        mime="image/png"
                    )
                new_msg["chart_bytes"] = img_bytes
                new_msg["chart_title"] = chart_config.get("title", "Chart")
            except Exception:
                logger.info("PNG export unavailable for this chart", exc_info=True)

        new_msg["answer"] = result.get("content") or "Here’s the chart."

    elif answer_type == "forecast":
        content = result.get("content", "")
        new_msg["answer"] = content
        png = render_forecast_png(result)
        if png:
            new_msg["chart_bytes"] = png
            new_msg["chart_title"] = f"{(result.get('forecast') or {}).get('metric', 'Metric')} forecast"
        if render_inline:
            st.write(content.replace("\n", "  \n"))
            if png:
                st.image(png, use_container_width=True)

    else:
        if render_inline:
            st.write(new_msg["answer"])

    # FINAL CHART SAFETY NET:
    # A chart request can arrive as text/code from an imperfect LLM response
    # even though the user's question explicitly asks for a chart.  Persist a
    # deterministic PNG at message-save time so Reports never has to depend on
    # reconstructing an old/partial chart_config later.
    if question and _looks_like_chart_request(question) and not new_msg.get("chart_bytes"):
        try:
            img_bytes, img_title = reconstruct_chat_chart(df, question)
            if img_bytes:
                new_msg["chart_bytes"] = img_bytes
                new_msg["chart_title"] = img_title or "Chart"
                # Mark it as chart content for history/report consumers without
                # changing the verified answer text.
                if not new_msg.get("chart_config"):
                    new_msg["chart_config"] = {"type": "bar", "title": img_title or "Chart"}
        except Exception:
            logger.exception("Deterministic chart persistence failed for %r", question)

    return new_msg


def render_table_html(df: pd.DataFrame, t: dict, max_height: int = 520):
    columns = list(df.columns)
    n_cols = max(len(columns), 1)
    col_width = 100 / n_cols

    def _cell(value):
        if pd.isna(value):
            return ""
        return html.escape(str(value))

    headers = "".join(
        f"<th style='width:{col_width:.4f}%;'>"
        f"{html.escape(str(col))}</th>" for col in columns
    )
    rows = []
    for _, row in df.iterrows():
        cells = "".join(
            f"<td style='width:{col_width:.4f}%;'>{_cell(row[col])}</td>"
            for col in columns
        )
        rows.append(f"<tr>{cells}</tr>")

    st.markdown(f"""
    <style>
        .data-preview-table-wrap {{
            width: 100% !important; max-width: none !important;
            display: block !important; overflow-x: auto;
            max-height: {max_height}px;
            border: 1px solid #E5E7EB; border-radius: 10px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            background: #fff;
        }}
        .data-preview-table-wrap table {{
            width: 100% !important; min-width: 100% !important;
            max-width: none !important; table-layout: fixed !important;
            border-collapse: collapse !important; border-spacing: 0;
            margin: 0 !important;
        }}
        .data-preview-table-wrap th {{
            background: #F8FAFC; color: #374151;
            border-bottom: 2px solid #E5E7EB; padding: 11px 16px;
            text-align: left; font-weight: 600; font-size: 0.82rem;
            white-space: nowrap;
        }}
        .data-preview-table-wrap td {{
            border-bottom: 1px solid #F3F4F6; padding: 11px 16px;
            color: #111827; background: #FFFFFF; font-size: 0.855rem;
            vertical-align: middle; overflow: hidden;
            text-overflow: ellipsis; white-space: nowrap;
        }}
        /* Zebra striping: alternate rows for easier scanning */
        .data-preview-table-wrap tbody tr:nth-child(even) td {{
            background: #F8FAFC;
        }}
        .data-preview-table-wrap tbody tr:hover td {{
            background: #F0F7FF;
        }}
        .data-preview-table-wrap tbody tr:last-child td {{ border-bottom: 0; }}
    </style>
    <div class="data-preview-table-wrap">
        <table>
            <colgroup>
                {''.join(f"<col style='width:{col_width:.4f}%;'>" for _ in columns)}
            </colgroup>
            <thead><tr>{headers}</tr></thead>
            <tbody>{''.join(rows)}</tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)


def render_kv_list_html(data: dict, t: dict):
    rows = "".join([
        f"<div style='display:flex;justify-content:space-between;"
        f"padding:7px 14px;border-bottom:1px solid {t['border']};'>"
        f"<span style='color:{t['text_secondary']};font-size:0.85rem;'>{html.escape(str(k))}</span>"
        f"<span style='color:{t['primary']};font-weight:600;font-size:0.85rem;'>{html.escape(str(v))}</span>"
        f"</div>"
        for k, v in data.items()
    ])
    st.markdown(
        f"<div style='background:{t['surface']};border:1px solid {t['border']};"
        f"border-radius:8px;overflow:hidden;'>{rows}</div>",
        unsafe_allow_html=True
    )


# ── MULTI-CHAT SESSION HELPERS ─────────────────────────────────────
UNTITLED_CHAT_TITLE = "Untitled chat"  # distinct from the "+ New chat"
                                        # button label so the sidebar
                                        # doesn't show two identical-
                                        # looking "New chat" rows.

def create_new_chat() -> str:
    chat_id = str(uuid.uuid4())[:8]
    st.session_state.chats[chat_id] = {
        "title": UNTITLED_CHAT_TITLE,
        "messages": [],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    st.session_state.current_chat_id = chat_id
    return chat_id


def get_current_messages() -> list:
    if (st.session_state.current_chat_id is None
            or st.session_state.current_chat_id not in st.session_state.chats):
        create_new_chat()
    return st.session_state.chats[st.session_state.current_chat_id]["messages"]


def maybe_set_chat_title(chat_id: str, first_question: str):
    chat = st.session_state.chats[chat_id]
    if chat["title"] == UNTITLED_CHAT_TITLE:
        title = first_question.strip()
        chat["title"] = (title[:36] + "…") if len(title) > 36 else title


# ── SUGGESTED QUESTIONS (shared by the Chat empty-state and Reports) ──
def build_suggested_questions(profile: dict, dq_result: dict = None) -> list:
    """Builds a small set of dataset-aware example questions from the
    column profile. Shared by the Chat tab's empty state and the
    Reports tab's Suggested Questions panel, so the two stay in sync
    instead of drifting into two separate implementations.

    dq_result is the output of assess_data_quality() (st.session_state.dq_result),
    optional so existing callers/tests that only pass a profile keep working.
    When provided, two things change: columns with a known WARNING/BLOCKING
    quality issue are no longer the FIRST choice for a naive "total X" /
    "histogram X" suggestion (they're deprioritized, not hidden — still
    usable further down the list), and one suggestion is generated directly
    from the highest-severity flagged issue, so a real, known problem in
    THIS dataset gets surfaced as something to ask about, instead of the
    chat silently computing totals/charts as if the flagged issue didn't
    exist. This is what closes the gap between the Data Quality tab and
    Chat with Data — right now they're two disconnected systems."""
    import re
    row_count = profile.get("row_count", 0)

    # column name -> its single most notable issue dict (WARNING/BLOCKING only)
    flagged: dict = {}
    if dq_result:
        for item in dq_result.get("blocking_issues", []) + dq_result.get("warnings", []):
            for col in item.get("columns", []):
                # keep the first (most severe, since blocking_issues is listed
                # first) issue seen per column rather than overwriting it
                flagged.setdefault(col, item)

    def _is_identifier_like(col: dict) -> bool:
        # get_data_profile() already flags this for text/categorical
        # columns (uniqueness_ratio > 0.5), but NOT for numeric columns —
        # so a numeric id column like order_id (int64, 100% unique) sails
        # through untagged and ends up as "What is the total order_id?",
        # which is meaningless for an identifier. Apply the same
        # uniqueness heuristic here for numeric columns too, plus a
        # common id-style naming pattern as a second, independent signal.
        if col.get("is_identifier_like"):
            return True
        if row_count and col.get("unique_count", 0) / row_count > 0.5:
            return True
        name = col.get("name", "").lower()
        if re.search(r'(?:^|_)id$', name):
            return True
        return False

    def _sort_clean_first(names: list) -> list:
        # Stable sort: unflagged columns first, but flagged ones stay in the
        # list (further down) rather than being dropped entirely — a
        # flagged column can still be a legitimate, interesting thing to
        # ask about, just not the FIRST thing suggested with no caveat.
        return sorted(names, key=lambda n: 1 if n in flagged else 0)

    numeric_cols = _sort_clean_first([
        c["name"] for c in profile["columns"]
        if c["dtype"] in ["int64", "float64"] and not _is_identifier_like(c)
    ])
    cat_cols = _sort_clean_first([
        c["name"] for c in profile["columns"]
        if c["dtype"] == "object" and not _is_identifier_like(c)
    ])

    suggestions = []
    if numeric_cols:
        suggestions.append(f"What is the total {numeric_cols[0]}?")
        suggestions.append(f"Show me {numeric_cols[0]} distribution as a histogram")
    if len(numeric_cols) >= 2:
        suggestions.append(f"Show {numeric_cols[0]} vs {numeric_cols[1]} as a scatter chart")
    if cat_cols and numeric_cols:
        suggestions.append(f"Show {numeric_cols[0]} by {cat_cols[0]} as a bar chart")
        suggestions.append(f"Which {cat_cols[0]} has the highest {numeric_cols[0]}?")
    if len(cat_cols) >= 2:
        suggestions.append(f"How many unique values are in {cat_cols[0]}?")

    # Actively surface the single most severe known data-quality issue as
    # its own suggested question, at the front of the list — this is the
    # direct fix for "why suggest totaling a column you already flagged as
    # invalid, instead of asking about the flag itself".
    if flagged:
        top_col, top_issue = max(
            flagged.items(),
            key=lambda kv: (kv[1].get("severity") == "BLOCKING", kv[1].get("affected_pct") or 0)
        )
        category = top_issue.get("category", "")
        templates = {
            "validity": f"How many {top_col} values are negative, and is that expected?",
            "outlier": f"What are the extreme outlier values in {top_col}?",
            "duplicate": "Show me the duplicate rows in this dataset",
            "missing": f"Which rows are missing {top_col}?",
            "data_type": f"Which values in {top_col} can't be read as numbers?",
            "date": f"Which values in {top_col} don't look like valid dates?",
            "category": f"Which values in {top_col} might be the same category with different spelling or casing?",
        }
        quality_question = templates.get(
            category, f"What's the {top_issue.get('issue', 'data quality issue').lower()} in {top_col}?"
        )
        suggestions.insert(0, quality_question)

    if not suggestions:
        suggestions = ["What are the key insights?", "Show summary statistics", "Find the top 5 items"]

    return suggestions


def ask_suggestion(suggestion: str):
    """Sends a suggestion straight into the chat, the same way the
    live chat input does — used by both the Chat empty-state starter
    pills and the Reports tab's Suggested Questions buttons."""
    from backend.llm_agent import answer_question
    df = st.session_state.df
    data_context = st.session_state.llm_context
    messages = get_current_messages()

    with st.spinner("Thinking..."):
        raw_result = answer_question(
            suggestion, data_context, messages,
            df_columns=list(df.columns), df=df,
            session_id=st.session_state.session_id,
        )
    new_msg = process_answer_result(raw_result, df, T, render_inline=False, question=suggestion)
    new_msg["question"] = suggestion

    messages.append(new_msg)
    # Chat content changed; any previously generated report is now stale.
    # In particular, a newly-created chart must be included in the next PDF.
    st.session_state.report_md = None
    st.session_state.report_pdf = None
    st.session_state.report_analysis = None
    st.session_state.report_config = None
    maybe_set_chat_title(st.session_state.current_chat_id, suggestion)

    st.session_state.active_tab = "Chat"
    st.rerun()


# ── DATE TYPE NORMALIZATION ───────────────────────────────────────
# CSV files commonly load date columns as object/string.  Convert only
# columns whose names clearly indicate a date/time field, and only when
# the values are overwhelmingly parseable as dates.  This keeps normal
# text columns such as product/region untouched.
_DATE_DTYPE_FIX_VERSION = 1

def _normalize_date_columns(df):
    if df is None or df.empty:
        return df

    date_name_tokens = (
        "date", "datetime", "timestamp", "time",
        "created_at", "updated_at", "transaction_date",
        "purchase_date", "order_date", "invoice_date",
    )

    for col in df.columns:
        if not isinstance(col, str):
            continue

        name = col.strip().lower().replace(" ", "_")
        if not any(token in name for token in date_name_tokens):
            continue

        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            continue

        non_null = int(series.notna().sum())
        if non_null == 0:
            continue

        parsed = pd.to_datetime(series, errors="coerce")
        parse_ratio = float(parsed.notna().sum()) / non_null

        if parse_ratio >= 0.80:
            df[col] = parsed

    return df


# ── UPLOAD LIMITS ─────────────────────────────────────────────────
# config.toml caps the file at 150 MB, but a small CSV can still expand to
# a huge frame (or a very wide one) once parsed. Reject before any
# profiling / LLM / report work is attempted.
MAX_UPLOAD_ROWS = 3_000_000
MAX_UPLOAD_COLS = 500


def _safe_display_name(name) -> str:
    """Strip any path component and control characters from an uploaded
    filename before it is stored or shown."""
    base = os.path.basename(str(name).replace("\\", "/"))
    base = re.sub(r"[\x00-\x1f\x7f]", "", base).strip()
    return base[:120] or "uploaded.csv"


# ── CSV UPLOAD HANDLING (shared by the Home page and Settings) ────
def handle_csv_upload(uploaded_file, status_slot=None):
    """Loads a newly-uploaded CSV, resets everything that's derived
    from the previous dataset, and kicks off profiling/insight
    generation. Safe to call every rerun — it no-ops once the same
    file (matched by name AND size) is already loaded, so a cleaned
    file re-uploaded under the same filename is correctly treated
    as new.

    Failures are reported through `status_slot` (an st.empty() placed
    directly under the uploader) when given, so the message appears where
    the user is looking and REPLACES the static "Profiling..." card. A
    failed file's message is remembered per (name, size), so Streamlit
    reruns neither re-parse a bad file nor lose the message."""
    if uploaded_file is None:
        return

    upload_identity = (uploaded_file.name, uploaded_file.size)

    def _show(msg):
        (status_slot if status_slot is not None else st).error(msg)

    def _fail(msg):
        st.session_state["failed_upload"] = (upload_identity, msg)
        _show(msg)

    failed = st.session_state.get("failed_upload")
    if failed and failed[0] == upload_identity:
        _show(failed[1])
        return

    if st.session_state.get("loaded_upload_identity") == upload_identity:
        return

    ext = os.path.splitext(str(uploaded_file.name))[1].lower()
    if ext != ".csv":
        _fail(f"Unsupported file type ({ext[:12] or 'no extension'}). "
              "Axiomrow analyses .csv files — please export your data as CSV and upload it again.")
        return
    if uploaded_file.size == 0:
        _fail("That file is empty (0 bytes). Please upload a CSV that contains a header row and data.")
        return

    try:
        df_loaded = load_csv(uploaded_file)
        if len(df_loaded) > MAX_UPLOAD_ROWS or df_loaded.shape[1] > MAX_UPLOAD_COLS:
            raise ValueError(
                f"This file is too large to analyse safely "
                f"({len(df_loaded):,} rows x {df_loaded.shape[1]:,} columns). "
                f"Limit: {MAX_UPLOAD_ROWS:,} rows and {MAX_UPLOAD_COLS} columns."
            )
        df_loaded = _normalize_date_columns(df_loaded)
        st.session_state.df = df_loaded
        st.session_state.profile = get_data_profile(df_loaded)
        st.session_state._date_dtype_fix_version = _DATE_DTYPE_FIX_VERSION
        # Remember this upload so a widget that still holds the file
        # (e.g. the Settings uploader) doesn't reload it and wipe chats.
        st.session_state.loaded_upload_identity = upload_identity
        st.session_state.loaded_filename = _safe_display_name(uploaded_file.name)
        st.session_state.insights = None
        st.session_state.quick_stats = None
        st.session_state.report_md = None
        st.session_state.report_pdf = None
        st.session_state.report_analysis = None
        st.session_state.report_config = None
        st.session_state.csv_bytes = None
        st.session_state.dq_result = None
        st.session_state.chats = {}
        st.session_state.current_chat_id = None
        create_new_chat()

        

            # ── Step 2: run the quality gate ─────────────────────────────────
        dq = assess_data_quality(df_loaded)
        st.session_state.dq_result = dq

        if dq["analysis_ready"]:
                # ── YES path: prepare the analysis pipeline, but always land
                #    on Data Quality first after a fresh upload.
                st.session_state.llm_context = get_llm_context(
                    df_loaded, profile=st.session_state.profile
                )
                st.session_state.insights = generate_insights(
                    df_loaded, session_id=st.session_state.session_id
                )
                st.session_state.quick_stats = get_quick_stats(df_loaded)
                st.toast(
                    f"Loaded — {len(df_loaded):,} rows × {len(df_loaded.columns)} columns",
                    icon=None
                )
        else:
                # ── NO path: block analysis and land on the quality report. ──
                st.toast(
                    f"{len(dq['blocking_issues'])} blocking issue(s) found — please clean your data.",
                    icon=None
                )

        # ── Every fresh CSV upload opens Data Quality first. ────────────────
        #    This applies to both analysis-ready and blocked datasets.
        st.session_state.active_tab = "Data Quality"
        st.session_state["failed_upload"] = None

        st.rerun()

    except ValueError as e:
        _fail(str(e))
    except Exception:
        # Never show parser/library internals (paths, stack frames) to the user.
        logger.exception("CSV upload failed")
        _fail("That file could not be read. Please check it is a valid CSV and try again.")


# ── SESSION STATE SETUP ───────────────────────────────────────────
if "session_id" not in st.session_state:
    # A stable id for this browser session, used only for per-session
    # LLM rate limiting (backend/rate_limiter.py) — not tied to any
    # user account or identity, just distinguishes concurrent sessions
    # from each other so one person's usage can't affect another's.
    st.session_state.session_id = str(uuid.uuid4())
if "loaded_filename" not in st.session_state:
    st.session_state.loaded_filename = None
if "loaded_upload_identity" not in st.session_state:
    st.session_state.loaded_upload_identity = None
if "df" not in st.session_state:
    st.session_state.df = None
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "insights" not in st.session_state:
    st.session_state.insights = None
if "quick_stats" not in st.session_state:
    st.session_state.quick_stats = None
if "profile" not in st.session_state:
    st.session_state.profile = None
if "llm_context" not in st.session_state:
    st.session_state.llm_context = None
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Home"
if "last_analysis_tab" not in st.session_state:
    st.session_state.last_analysis_tab = "Data Preview"
if "report_md" not in st.session_state:
    st.session_state.report_md = None
if "report_pdf" not in st.session_state:
    st.session_state.report_pdf = None
if "report_analysis" not in st.session_state:
    st.session_state.report_analysis = None
if "report_config" not in st.session_state:
    st.session_state.report_config = None
if "csv_bytes" not in st.session_state:
    st.session_state.csv_bytes = None
if "dq_result" not in st.session_state:
    st.session_state.dq_result = None

# ── APPLY THEME ───────────────────────────────────────────────────
t = T
apply_theme(t)

ANALYSIS_SUBTABS = ["Data Preview", "Data Profile", "AI Analysis"]

ANALYSIS_SUBICONS = ["table", "bar-chart-line", "stars"]
NAV_ITEMS = [
    ("Home",           "\uf425", True),
    ("Data Quality",   "\uf52f", False),
    ("Chat with Data", "\uf252", False),
    ("Analysis",       "\uf17e", False),
    ("Reports",        "\uf38b", False),
    ("Settings",       "\uf3e5", True),
]

has_data = st.session_state.df is not None

# One-time migration for datasets that were already loaded before the date
# normalization fix was added.  Avoids re-scanning large datasets on every
# Streamlit rerun while ensuring the current session gets the corrected dtype.
if has_data and st.session_state.get("_date_dtype_fix_version") != _DATE_DTYPE_FIX_VERSION:
    st.session_state.df = _normalize_date_columns(st.session_state.df)
    st.session_state.profile = get_data_profile(st.session_state.df)
    st.session_state._date_dtype_fix_version = _DATE_DTYPE_FIX_VERSION

# Recover derived profile data if a session survived a code reload or an
# earlier reset left the dataframe loaded but the profile unset.
# All data-dependent pages rely on profile, so rebuild it before routing.
if has_data and st.session_state.get("profile") is None:
    try:
        st.session_state.profile = get_data_profile(st.session_state.df)
    except Exception:
        logger.exception("Failed to rebuild dataset profile")
        st.error(
            "Something went wrong loading your dataset. Try re-uploading the file — "
            "if it keeps happening, the file may be in an unexpected format."
        )
        st.stop()

# The landing/cover page is intentionally distraction-free: the navigation
# sidebar is hidden until a CSV is loaded. Once data exists, the normal
# application navigation appears automatically.
if not has_data:
    st.markdown("""
        <style>
            /* Hide the entire sidebar on the cover/upload screen. */
            section[data-testid="stSidebar"],
            [data-testid="stSidebar"] {
                display: none !important;
            }

            /* Reclaim the sidebar's horizontal space for the cover page. */
            [data-testid="stAppViewContainer"] > .main {
                margin-left: 0 !important;
            }
            [data-testid="stAppViewContainer"] .main .block-container {
                max-width: 1200px !important;
                margin-left: auto !important;
                margin-right: auto !important;
                padding-left: 2rem !important;
                padding-right: 2rem !important;
            }
        </style>
    """, unsafe_allow_html=True)

# Guard against stale navigation (e.g. session was just reset from
# Settings while the user was sitting on a data-dependent tab).
if not has_data and st.session_state.active_tab not in ("Home", "Settings"):
    st.session_state.active_tab = "Home"

# If data loaded but quality gate failed, redirect data-dependent tabs to quality report
dq_result = st.session_state.get("dq_result")
dq_blocked = has_data and dq_result is not None and not dq_result["analysis_ready"]
if dq_blocked and st.session_state.active_tab in ("Chat", *ANALYSIS_SUBTABS, "Reports"):
    st.session_state.active_tab = "Data Quality"

if st.session_state.active_tab in ANALYSIS_SUBTABS:
    current_nav = "Analysis"
elif st.session_state.active_tab == "Chat":
    current_nav = "Chat with Data"
elif st.session_state.active_tab == "Data Quality":
    current_nav = "Data Quality"
else:
    current_nav = st.session_state.active_tab

# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
        <div style='display:flex;align-items:center;gap:10px;margin-bottom:22px;'>
            <div style='width:32px;height:32px;border-radius:9px;flex-shrink:0;
                        background:linear-gradient(135deg,{t["primary"]},{t["accent2"]});
                        display:flex;align-items:center;justify-content:center;'>
                {icon_svg("brand", size=17, color="white")}
            </div>
            <div>
                <div style='font-weight:700;font-size:16px;letter-spacing:-0.01em;color:#f1f5f9;line-height:1.2;'>
                    Axiomrow
                </div>
                <div class='brand-tagline'>Your Data. Smarter Insights.</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # On the pre-upload "cover" screen, only Home and Settings are
    # actually usable — the rest are disabled anyway. Showing all 6
    # greyed-out items on someone's first look at the app reads as
    # clutter, so the full nav only appears once data is loaded.
    nav_visible = NAV_ITEMS if has_data else [item for item in NAV_ITEMS if item[2]]

    for label, icon_char, always_enabled in nav_visible:
        is_active = (label == current_nav)
        is_disabled = (not always_enabled) and (not has_data)
        if st.button(
            f"{icon_char}\u2002{label}",
            key=f"navbtn_{label}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
            disabled=is_disabled,
        ):
            if label == "Home":
                st.session_state.active_tab = "Home"
            elif label == "Data Quality":
                st.session_state.active_tab = "Data Quality"
            elif label == "Chat with Data":
                st.session_state.active_tab = "Chat"
            elif label == "Analysis":
                st.session_state.active_tab = st.session_state.last_analysis_tab
            elif label == "Reports":
                st.session_state.active_tab = "Reports"
            elif label == "Settings":
                st.session_state.active_tab = "Settings"
            st.rerun()

    # ── Recent chats (ChatGPT-style) ────────────────────────────
    if has_data:
        st.divider()
        st.markdown("<div class='sidebar-section-label'>Recent</div>", unsafe_allow_html=True)

        if st.button("\uf64d  New chat", use_container_width=True, type="secondary", key="new_chat_btn"):
            create_new_chat()
            st.session_state.active_tab = "Chat"
            st.rerun()

        for cid, chat in reversed(list(st.session_state.chats.items())):
            is_current = (cid == st.session_state.current_chat_id and st.session_state.active_tab == "Chat")
            # Keep recent-chat buttons the same full width as the main
            # sidebar navigation buttons. The delete control is overlaid
            # at the right edge so it does not shrink the chat-title button.
            col_select, col_delete = st.columns([1, 0.12])
            with col_select:
                chat_title = chat.get("title", "New chat")
                if st.button(
                    chat_title,
                    key=f"chatbtn_{cid}",
                    use_container_width=True,
                    type="primary" if is_current else "secondary",
                ):
                    st.session_state.current_chat_id = cid
                    st.session_state.active_tab = "Chat"
                    st.rerun()
            with col_delete:
                if len(st.session_state.chats) > 1:
                    if st.button("\uf5de", key=f"delbtn_{cid}", type="secondary"):
                        del st.session_state.chats[cid]
                        if st.session_state.current_chat_id == cid:
                            st.session_state.current_chat_id = list(
                                st.session_state.chats.keys()
                            )[-1]
                        st.rerun()

    st.divider()
    st.markdown(
        "<div style='text-align:center;color:#334155;font-size:11px;padding-bottom:4px;'>"
        "Axiomrow © 2026</div>",
        unsafe_allow_html=True
    )


# ── MAIN AREA ─────────────────────────────────────────────────────

# ── HOME ────────────────────────────────────────────────────────
if st.session_state.active_tab == "Home":

    if st.session_state.df is None:
        # ---- Landing / cover state ----
        # Inspired by the supplied reference layout, while preserving the
        # existing Axiomrow light theme, palette, typography and functionality.
        # ── COVER / LANDING ─────────────────────────────────────────────
        # Reference-inspired landing page.  The application's existing light
        # theme and upload/processing pipeline remain unchanged.
        st.markdown(f"""
        <style>
        html {{
            scroll-behavior:smooth;
            scroll-padding-top:24px;
        }}
        /* Exact reference-cover canvas */
        /* Remove Streamlit's native top chrome: the supplied reference
           cover begins at the top edge with the custom navbar. */
        .stApp:has(.cover-topbar) [data-testid="stHeader"],
        .stApp:has(.cover-topbar) header[data-testid="stHeader"],
        .stApp:has(.cover-topbar) [data-testid="stToolbar"] {{
            display:none !important;
        }}
        .stApp:has(.cover-topbar) [data-testid="stAppViewContainer"] > .main {{
            padding-top:0 !important;
            margin-top:0 !important;
        }}
        .stApp:has(.cover-topbar) {{
            background:
                radial-gradient(circle at 48% 36%, rgba(226,238,255,.52) 0, rgba(247,250,255,0) 38%),
                linear-gradient(180deg,#F8FBFF 0%,#F7FAFF 100%) !important;
        }}
        .stApp:has(.cover-topbar) .block-container {{
            max-width:1395px !important;
            width:100% !important;
            margin-left:auto !important;
            margin-right:auto !important;
            padding-top:2px !important;
            padding-left:0 !important;
            padding-right:0 !important;
            box-sizing:border-box !important;
        }}
        .cover-shell {{
            width:100%;
            max-width:1395px;
            margin:0 auto;
            padding:0;
            box-sizing:border-box;
        }}

        /* Reference hero is inset from the navbar on both sides. */
        .stApp:has(.cover-topbar) [data-testid="stHorizontalBlock"] {{
            width:calc(100% - 118px) !important;
            max-width:1290px !important;
            margin-left:auto !important;
            margin-right:auto !important;
            gap:0 !important;
        }}
        .stApp:has(.cover-topbar) [data-testid="stHorizontalBlock"] > [data-testid="column"] {{
            min-width:0 !important;
        }}

        /* Top navigation */
        .cover-topbar {{
            height:66px;
            display:flex;
            align-items:center;
            justify-content:space-between;
            padding:0 22px;
            margin-bottom:46px;
            border:1px solid {t["border_strong"]};
            border-radius:18px;
            background:rgba(255,255,255,.84);
            backdrop-filter:blur(14px);
            box-shadow:0 7px 25px rgba(15,23,42,.055);
        }}
        .cover-brand {{
            display:flex;
            align-items:center;
            gap:13px;
            color:{t["text"]};
            font-weight:800;
            font-size:1.18rem;
            letter-spacing:-.025em;
        }}
        .cover-brand-icon {{
            width:48px;
            height:48px;
            border-radius:12px;
            display:flex;
            align-items:center;
            justify-content:center;
            background:linear-gradient(135deg,{t["primary"]},{t["accent2"]});
            box-shadow:0 7px 18px rgba(37,99,235,.24);
        }}
        .cover-nav {{
            display:flex;
            align-items:center;
            gap:31px;
            color:{t["text_secondary"]};
            font-size:.78rem;
            font-weight:500;
        }}
        .cover-nav a {{
            position:relative;
            display:inline-flex;
            align-items:center;
            height:66px;
            padding:0 1px;
            color:{t["text_secondary"]};
            text-decoration:none!important;
            white-space:nowrap;
            transition:color .18s ease;
        }}
        .cover-nav a:hover {{
            color:{t["primary"]};
        }}
        .cover-nav a.active {{
            color:{t["primary"]};
            font-weight:700;
        }}
        .cover-nav a.active::after {{
            content:"";
            position:absolute;
            left:0;
            right:0;
            bottom:0;
            height:2px;
            border-radius:2px;
            background:linear-gradient(90deg,{t["primary"]},{t["accent2"]});
        }}
        .cover-nav-cta {{
            height:48px!important;
            padding:0 23px!important;
            border-radius:13px;
            color:white!important;
            background:linear-gradient(135deg,{t["primary"]},{t["accent2"]});
            box-shadow:0 6px 17px rgba(37,99,235,.20);
        }}
        .cover-nav-cta:hover {{
            color:white!important;
            transform:translateY(-1px);
            box-shadow:0 8px 20px rgba(37,99,235,.26);
        }}

        /* Hero */
        .cover-kicker {{
            display:inline-flex;
            align-items:center;
            gap:8px;
            color:{t["primary"]};
            background:{t["soft_surface"]};
            border:1px solid #D8E2FF;
            border-radius:999px;
            padding:7px 14px;
            font-size:.70rem;
            font-weight:800;
            letter-spacing:.055em;
            text-transform:uppercase;
            margin-bottom:16px;
        }}
        .cover-title {{
            margin:0;
            color:{t["text"]};
            font-size:72px;
            line-height:.98;
            font-weight:800;
            letter-spacing:-.055em;
        }}
        .cover-title-gradient {{
            background:linear-gradient(135deg,{t["primary"]},{t["accent2"]});
            -webkit-background-clip:text;
            background-clip:text;
            color:transparent;
        }}
        .cover-subtitle {{
            margin:15px 0 17px;
            max-width:790px;
            color:{t["text_secondary"]};
            font-size:18px;
            line-height:1.45;
        }}

        /* Right-side value propositions */
        .cover-benefits {{
            border-left:1px solid #C9D5E8;
            padding-left:43px;
            margin-top:0;
            padding-top:0;
        }}
        .cover-benefit {{
            display:flex;
            gap:16px;
            align-items:flex-start;
            margin:0 0 25px;
        }}
        .cover-benefit:last-child {{margin-bottom:0;}}
        .cover-benefit-icon {{
            width:48px;
            height:48px;
            border-radius:14px;
            flex:0 0 48px;
            display:flex;
            align-items:center;
            justify-content:center;
            background:{t["soft_surface"]};
            box-shadow:0 3px 10px rgba(37,99,235,.045);
        }}
        .cover-benefit-title {{
            color:{t["text"]};
            font-size:1.00rem;
            font-weight:700;
            line-height:1.25;
        }}
        .cover-benefit-sub {{
            color:{t["text_secondary"]};
            font-size:.86rem;
            margin-top:5px;
            line-height:1.35;
        }}

                /* Upload dropzone */
        .cover-upload-wrap {{
            width:100%;
            max-width:780px;
            margin-top:0;
        }}

        .cover-upload-wrap [data-testid="stFileUploaderDropzone"] {{
            min-height:110px!important;
            height:110px!important;
            display:flex!important;
            align-items:center!important;
            background:linear-gradient(180deg,#FFFFFF,{t["surface_muted"]})!important;
            border:1.5px dashed {t["border_dashed"]}!important;
            border-radius:17px!important;
            padding:15px 24px!important;
            box-shadow:0 7px 24px rgba(15,23,42,.045);
        }}

        /* Upload button — remove extra internal spacing */
        .cover-upload-wrap [data-testid="stFileUploader"] button {{
            margin: 0 !important;
            padding: 8px 14px !important;
            min-height: 40px !important;
            height: 40px !important;
        }}

        .cover-upload-wrap [data-testid="stFileUploaderDropzone"]:hover {{
            border-color:{t["primary"]}!important;
            box-shadow:0 0 0 4px {t["soft_surface"]},
                       0 9px 25px rgba(37,99,235,.08)!important;
        }}
        .cover-support {{
            color:#71809D;
            font-size:.78rem;
            margin-top:6px;
        }}

        /* Selected file card */
        .cover-file-card {{
            width:100%;
            max-width:700px;
            min-height:58px;
            margin-top:10px;
            padding:9px 13px;
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:12px;
            box-sizing:border-box;
            border:1px solid #B9E9DD;
            border-radius:14px;
            background:linear-gradient(100deg,#FFFFFF 0%,#F7FFFC 100%);
            box-shadow:
                0 8px 24px rgba(16,185,129,.09),
                0 0 20px rgba(16,185,129,.07);
        }}
        .cover-file-main {{
            display:flex;
            align-items:center;
            gap:11px;
            min-width:0;
        }}
        .cover-file-icon {{
            width:34px;
            height:34px;
            flex:0 0 34px;
            border-radius:9px;
            display:flex;
            align-items:center;
            justify-content:center;
            background:#E2F8F0;
            color:#10A37F;
        }}
        .cover-file-name {{
            color:{t["text"]};
            font-size:.77rem;
            font-weight:750;
            white-space:nowrap;
            overflow:hidden;
            text-overflow:ellipsis;
        }}
        .cover-file-meta {{
            color:{t["muted"]};
            font-size:.65rem;
            margin-top:2px;
        }}
        .cover-file-ready {{
            display:flex;
            align-items:center;
            gap:6px;
            padding:6px 10px;
            border-radius:999px;
            background:#E4F8F0;
            color:#079669;
            font-size:.65rem;
            font-weight:750;
            white-space:nowrap;
        }}
        .cover-file-ready-dot {{
            width:7px;
            height:7px;
            border-radius:50%;
            background:#10B981;
            box-shadow:0 0 7px rgba(16,185,129,.5);
        }}

        /* Processing status */
        .cover-processing {{
            width:100%;
            max-width:700px;
            min-height:64px;
            margin-top:10px;
            padding:9px 13px;
            display:flex;
            align-items:center;
            justify-content:space-between;
            box-sizing:border-box;
            border:1px solid #B8C9FF;
            border-radius:14px;
            background:linear-gradient(100deg,#F9FBFF 0%,#F5F0FF 100%);
            box-shadow:
                0 9px 26px rgba(37,99,235,.10),
                0 0 22px rgba(124,58,237,.08);
        }}
        .cover-processing-main {{
            display:flex;
            align-items:center;
            gap:12px;
        }}
        .cover-spinner {{
            width:30px;
            height:30px;
            border-radius:50%;
            border:4px solid #DCE4FF;
            border-top-color:{t["primary"]};
            border-right-color:{t["accent2"]};
            box-sizing:border-box;
            animation:cover-spin .85s linear infinite;
            box-shadow:0 0 12px rgba(79,70,229,.14);
        }}
        @keyframes cover-spin {{
            to {{transform:rotate(360deg);}}
        }}
        .cover-processing-title {{
            color:#243FA8;
            font-size:.79rem;
            font-weight:800;
            letter-spacing:-.01em;
        }}
        .cover-processing-sub {{
            color:{t["text_secondary"]};
            font-size:.65rem;
            margin-top:3px;
        }}
        .cover-processing-dots {{
            color:{t["primary"]};
            font-size:1rem;
            letter-spacing:3px;
            animation:cover-pulse 1.2s ease-in-out infinite;
        }}
        @keyframes cover-pulse {{
            0%,100% {{opacity:.35;}}
            50% {{opacity:1;}}
        }}

        /* Process section */
        .cover-process-title {{
            text-align:center;
            color:{t["text"]};
            font-size:1.34rem;
            font-weight:800;
            letter-spacing:-.025em;
            margin:10px 0 5px;
            letter-spacing:-.015em;
        }}
        .cover-process-sub {{
            text-align:center;
            color:{t["text_secondary"]};
            font-size:.82rem;
            margin-bottom:13px;
        }}
        .cover-process-grid {{
            display:grid;
            grid-template-columns:repeat(5,minmax(0,1fr));
            gap:40px;
            width:100%;
        }}
        .cover-step {{
            position:relative;
            min-height:136px;
            height:136px;
            padding:15px 19px;
            overflow:visible;
            border:1px solid {t["border_strong"]};
            border-radius:14px;
            background:rgba(255,255,255,.94);
            box-shadow:0 3px 12px rgba(15,23,42,.035);
            box-sizing:border-box;
            transition:transform .18s ease,box-shadow .18s ease,border-color .18s ease;
        }}
        .cover-step:hover {{
            transform:translateY(-2px);
            border-color:#AFC1FF;
            box-shadow:0 8px 20px rgba(37,99,235,.08);
        }}
        .cover-step:not(:last-child)::after {{
            content:"→";
            position:absolute;
            right:-30px;
            top:50%;
            transform:translateY(-50%);
            color:#7891C5;
            font-size:25px;
            font-weight:500;
            line-height:1;
            pointer-events:none;
        }}
        .cover-step-icon {{
            width:40px;
            height:40px;
            border-radius:12px;
            display:flex;
            align-items:center;
            justify-content:center;
            margin-bottom:10px;
        }}
        .cover-step-title {{
            color:{t["text"]};
            font-size:.92rem;
            font-weight:750;
        }}
        .cover-step-sub {{
            color:{t["text_secondary"]};
            font-size:.80rem;
            line-height:1.35;
            margin-top:4px;
        }}
        
        @media(max-width:900px){{
            .cover-nav{{display:none;}}
            .cover-title{{font-size:2.5rem;}}
            .cover-benefits{{margin-top:24px;border-left:0;border-top:1px solid {t["border_strong"]};
                padding:20px 0 0;}}
            .cover-process-grid{{grid-template-columns:repeat(2,minmax(0,1fr));}}
        }}
        @media(max-width:560px){{
            .cover-process-grid{{grid-template-columns:1fr;}}
        }}
        </style>

        <div class="cover-shell">
            <div class="cover-topbar" id="cover-home">
                <div class="cover-brand">
                    <div class="cover-brand-icon">{icon_svg("brand",size=18,color="white")}</div>
                    Axiomrow
                </div>
                <nav class="cover-nav" aria-label="Landing page navigation">
                    <a class="active" href="#cover-home">Home</a>
                    <a href="#cover-features">Features</a>
                    <a href="#cover-how-it-works">How it Works</a>
                    <a class="cover-nav-cta" href="#cover-get-started">Get Started&nbsp;→</a>
                </nav>
            </div>
        </div>
        """, unsafe_allow_html=True)

        hero_left, hero_right = st.columns([1.48, .82], gap="medium")

        with hero_left:
            st.markdown(f"""
                <div class="cover-kicker">
                    {icon_svg("bolt",size=11,color=t["primary"])}
                    AI-Powered Data Analysis
                </div>
                <h1 class="cover-title">
                    <b>Turn Your Data<br>into</b>
                    <span class="cover-title-gradient">Decisions.</span>
                </h1>
                <p class="cover-subtitle">
                    Axiomrow is your AI-powered data analysis workspace.
                    Upload your dataset, explore its structure, uncover patterns,
                    visualize insights, and ask questions — all in one place.
                </p>
            """, unsafe_allow_html=True)

            st.markdown('<div class="cover-upload-wrap" id="cover-get-started">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Upload Dataset",
                help="Max file size: 150MB",
                label_visibility="collapsed",
                key="home_uploader",
            )
            st.markdown("</div>", unsafe_allow_html=True)

            # Everything below the uploader lives in ONE slot, so a failure
            # message can replace the "Profiling..." card in place instead
            # of appearing at the bottom of the page under it.
            status_slot = st.empty()
            _bad_file = (uploaded_file is not None and (
                uploaded_file.size == 0
                or not str(uploaded_file.name).lower().endswith(".csv")))

            if uploaded_file is None:
                status_slot.markdown("""
                    <div class="cover-support">
                        Supports CSV files only · Up to 150MB per file · Ready for analysis
                    </div>
                """, unsafe_allow_html=True)
            elif _bad_file:
                pass  # handle_csv_upload() puts the explanation in status_slot
            else:
                # Replace Streamlit's plain selected-file row with a polished card.
                size_bytes = getattr(uploaded_file, "size", 0) or 0
                if size_bytes < 1024:
                    size_text = f"{size_bytes} B"
                elif size_bytes < 1024 * 1024:
                    size_text = f"{size_bytes / 1024:.1f} KB"
                else:
                    size_text = f"{size_bytes / (1024 * 1024):.1f} MB"

                _slot = status_slot.container()
                _slot.markdown(f"""
                    <div class="cover-file-card">
                        <div class="cover-file-main">
                            <div class="cover-file-icon">
                                {icon_svg("file",16,"#10A37F")}
                            </div>
                            <div>
                                <div class="cover-file-name">{html.escape(str(uploaded_file.name))}</div>
                                <div class="cover-file-meta">{size_text} · CSV</div>
                            </div>
                        </div>
                        <div class="cover-file-ready">
                            <span class="cover-file-ready-dot"></span>
                            File Ready
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                _slot.markdown("""
                    <div class="cover-processing">
                        <div class="cover-processing-main">
                            <div class="cover-spinner"></div>
                            <div>
                                <div class="cover-processing-title">
                                    Profiling &amp; assessing data quality...
                                </div>
                                <div class="cover-processing-sub">
                                    This may take a few seconds. Please wait while we prepare your data.
                                </div>
                            </div>
                        </div>
                        <div class="cover-processing-dots">•••</div>
                    </div>
                """, unsafe_allow_html=True)

        with hero_right:
            st.markdown(f"""
            <div class="cover-benefits" id="cover-features">
                <div class="cover-benefit">
                    <div class="cover-benefit-icon">{icon_svg("database",17,t["primary"])}</div>
                    <div>
                        <div class="cover-benefit-title">Understand Your Data</div>
                        <div class="cover-benefit-sub">Profile columns, types and quality.</div>
                    </div>
                </div>
                <div class="cover-benefit">
                    <div class="cover-benefit-icon">{icon_svg("trending",17,t["accent2"])}</div>
                    <div>
                        <div class="cover-benefit-title">Find Patterns with AI</div>
                        <div class="cover-benefit-sub">Discover trends and anomalies faster.</div>
                    </div>
                </div>
                <div class="cover-benefit">
                    <div class="cover-benefit-icon">{icon_svg("barchart",17,t["primary"])}</div>
                    <div>
                        <div class="cover-benefit-title">Create Visualizations</div>
                        <div class="cover-benefit-sub">Turn numbers into meaningful charts.</div>
                    </div>
                </div>
                <div class="cover-benefit">
                    <div class="cover-benefit-icon">{icon_svg("bolt",17,t["warning"])}</div>
                    <div>
                        <div class="cover-benefit-title">Get Actionable Insights</div>
                        <div class="cover-benefit-sub">Move from data to decisions.</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
            <div class="cover-process-title" id="cover-how-it-works">From data to discovery</div>
            <div class="cover-process-sub">
                Upload once. Analyze, visualize and ask — all in one workspace.
            </div>
        """, unsafe_allow_html=True)

        steps=[
            ("upload","1. Upload","Add your CSV dataset",t["primary"]),
            ("search","2. Profile","Understand your data",t["accent2"]),
            ("trending","3. Analyze","Discover trends & patterns",t["primary"]),
            ("barchart","4. Visualize","Create meaningful charts",t["accent2"]),
            ("bolt","5. Ask","Get answers from your data",t["success"]),
        ]

        

        step_cards = []
        for i, (icon_name, title, subtitle, accent) in enumerate(steps):
           
            icon_html = icon_svg(icon_name, 15, accent)
            step_cards.append(
                f'<div class="cover-step">'
                
                f'<div class="cover-step-icon" style="background:{accent}1A;color:{accent};">'
                f'{icon_html}'
                f'</div>'
                f'<div class="cover-step-title">{title}</div>'
                f'<div class="cover-step-sub">{subtitle}</div>'
                f'</div>'
            )

        st.markdown(
            '<div class="cover-process-grid">' + "".join(step_cards) + "</div>",
            unsafe_allow_html=True,
        )
        handle_csv_upload(uploaded_file, status_slot=status_slot)

    else:
        # ---- Loaded dashboard state ----
        df = st.session_state.df
        profile = st.session_state.profile
        _dq = st.session_state.get("dq_result")
        _dq_ready = (_dq is None) or _dq["analysis_ready"]

        _pill_color = t["success"] if _dq_ready else t["error"]
        _pill_bg    = t["success_bg"] if _dq_ready else t["error_bg"]
        _pill_dot   = f"<span style='width:5px;height:5px;border-radius:50%;background:{_pill_color};display:inline-block;'></span>"
        _pill_label = "Ready" if _dq_ready else "Quality Issues"

        st.markdown(f"""
            <div class='topbar'>
                <div style='display:flex;align-items:center;gap:7px;font-size:0.85rem;color:{t["text_secondary"]};'>
                    {icon_svg("file", size=15, color=t["muted"])}
                    <span>{html.escape(str(st.session_state.loaded_filename))} &nbsp;·&nbsp;
                    <b style='color:{t["text"]};font-weight:600;'>{profile['row_count']:,} rows</b>
                    &nbsp;·&nbsp; {profile['column_count']} columns</span>
                </div>
                <span style='display:inline-flex;align-items:center;gap:5px;font-size:0.78rem;
                             font-weight:500;color:{_pill_color};background:{_pill_bg};
                             padding:3px 10px;border-radius:100px;'>
                    {_pill_dot} {_pill_label}
                </span>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='home-section-title'>Welcome back</div>", unsafe_allow_html=True)
        st.markdown("<div class='home-section-sub'>Pick up where you left off.</div>", unsafe_allow_html=True)

        _cards = (
            [
                (None, "search",  "Data Quality",   "View blocking issues and the step-by-step cleaning plan.", "Data Quality"),
                (None, "message", "Chat with Data", "Ask questions about your dataset in plain English.", "Chat"),
                (None, "barchart","Analysis",        "Browse rows, profile columns, and read AI-generated insights.", st.session_state.last_analysis_tab),
            ] if not _dq_ready else [
                (None, "message", "Chat with Data", "Ask questions about your dataset in plain English.", "Chat"),
                (None, "barchart","Analysis",        "Browse rows, profile columns, and read AI-generated insights.", st.session_state.last_analysis_tab),
                (None, "file",    "Reports",         "Get suggested questions and export a shareable report.", "Reports"),
            ]
        )

        qc_cols = st.columns(len(_cards))
        for (col, icon_name, title, desc, dest), qcol in zip(_cards, qc_cols):
            col = qcol
            with col:
                st.markdown(f"""
                <div class='feature-card'>
                    <div style='display:flex;align-items:center;gap:10px;margin-bottom:11px;'>
                        <div class='insight-icon-box'>{icon_svg(icon_name, size=16, color=t["primary"])}</div>
                        <div class='home-card-title'>{title}</div>
                    </div>
                    <span class='home-card-desc'>{desc}</span>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Open {title}", key=f"quicklink_{title}", use_container_width=True, type="secondary"):
                    st.session_state.active_tab = dest
                    st.rerun()


# ── DATA QUALITY GATE ─────────────────────────────────────────
elif st.session_state.active_tab == "Data Quality":
    dq = st.session_state.get("dq_result")

    if dq is None:
        st.info("No quality assessment available. Please upload a CSV first.")
    else:
        ps = dq["profile_summary"]
        is_ready = dq["analysis_ready"]

        # ── Top bar ────────────────────────────────────────────────────
        status_color = t["success"] if is_ready else t["error"]
        status_bg    = t["success_bg"] if is_ready else t["error_bg"]
        status_label = "ANALYSIS READY" if is_ready else "NOT READY — CLEAN DATA FIRST"
        st.markdown(f"""
            <div class='topbar'>
                <div style='display:flex;align-items:center;gap:7px;font-size:0.85rem;color:{t["text_secondary"]};'>
                    {icon_svg("file", size=15, color=t["muted"])}
                    <span>{html.escape(str(st.session_state.loaded_filename))} &nbsp;·&nbsp;
                    <b style='color:{t["text"]};font-weight:600;'>{ps["rows"]:,} rows</b>
                    &nbsp;·&nbsp; {ps["columns"]} columns</span>
                </div>
                <span style='display:inline-flex;align-items:center;gap:5px;font-size:0.78rem;
                             font-weight:600;color:{status_color};background:{status_bg};
                             padding:3px 12px;border-radius:100px;'>
                    {status_label}
                </span>
            </div>
        """, unsafe_allow_html=True)

        # ── Pane header ────────────────────────────────────────────────
        header_bg  = "#ECFDF5" if is_ready else "#FEF2F2"
        header_bc  = "#A7F3D0" if is_ready else "#FECACA"
        header_ic  = t["success"] if is_ready else t["error"]
        st.markdown(f"""
        <div style='background:{header_bg};border:1px solid {header_bc};
                    border-radius:10px;padding:14px 18px;margin-bottom:20px;
                    display:flex;align-items:center;gap:12px;'>
            <div style='width:36px;height:36px;border-radius:8px;flex-shrink:0;
                        background:{header_ic};display:flex;align-items:center;
                        justify-content:center;'>
                {icon_svg("search", size=18, color="white")}
            </div>
            <div>
                <div style='font-size:1rem;font-weight:700;color:{t["text"]};'>
                    Data Quality Assessment
                </div>
                <div style='font-size:0.78rem;color:{t["text_secondary"]};margin-top:2px;'>
                    {html.escape(str(dq["reason"]))}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Profile summary metrics ────────────────────────────────────
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Rows",        f"{ps['rows']:,}")
        m2.metric("Columns",     ps["columns"])
        m3.metric("Numeric",     ps["numeric_columns"])
        m4.metric("Missing cells", f"{ps['missing_cells']:,}")
        m5.metric("Duplicate rows", f"{ps['duplicate_rows']:,}")

        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

        # ── Blocking issues ────────────────────────────────────────────
        if dq["blocking_issues"]:
            st.markdown(
                f"<div style='font-size:0.8rem;font-weight:600;color:{t['error']};"
                f"letter-spacing:.05em;text-transform:uppercase;margin-bottom:8px;'>"
                f"Blocking Issues — must fix before analysis</div>",
                unsafe_allow_html=True
            )
            for issue in dq["blocking_issues"]:
                cols_txt = (", ".join(f"`{html.escape(str(c))}`" for c in issue.get("columns", []))
                            if issue.get("columns") else "")
                st.markdown(f"""
                <div style='background:{t["error_bg"]};border:1px solid #FECACA;
                            border-left:4px solid {t["error"]};border-radius:8px;
                            padding:12px 16px;margin-bottom:8px;'>
                    <div style='font-weight:600;color:{t["error"]};font-size:0.9rem;
                                margin-bottom:4px;'>{html.escape(str(issue["issue"]))}</div>
                    <div style='color:{t["text"]};font-size:0.85rem;margin-bottom:6px;'>
                        {html.escape(str(issue["observed"]))}
                    </div>
                    {f"<div style='color:{t['text_secondary']};font-size:0.82rem;margin-bottom:4px;'><b>Columns:</b> {cols_txt}</div>" if cols_txt else ""}
                    <div style='color:{t["text_secondary"]};font-size:0.82rem;'>
                        <b>Why it matters:</b> {html.escape(str(issue["why"]))}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Warnings ──────────────────────────────────────────────────
        if dq["warnings"]:
            st.markdown(
                f"<div style='font-size:0.8rem;font-weight:600;color:{t['warning']};"
                f"letter-spacing:.05em;text-transform:uppercase;margin:14px 0 8px;'>"
                f"Warnings — review before analysis</div>",
                unsafe_allow_html=True
            )
            for issue in dq["warnings"]:
                cols_txt = (", ".join(f"`{html.escape(str(c))}`" for c in issue.get("columns", []))
                            if issue.get("columns") else "")
                st.markdown(f"""
                <div style='background:{t["warning_bg"]};border:1px solid #FDE68A;
                            border-left:4px solid {t["warning"]};border-radius:8px;
                            padding:12px 16px;margin-bottom:8px;'>
                    <div style='font-weight:600;color:{t["warning"]};font-size:0.9rem;
                                margin-bottom:4px;'>{html.escape(str(issue["issue"]))}</div>
                    <div style='color:{t["text"]};font-size:0.85rem;margin-bottom:6px;'>
                        {html.escape(str(issue["observed"]))}
                    </div>
                    {f"<div style='color:{t['text_secondary']};font-size:0.82rem;margin-bottom:4px;'><b>Columns:</b> {cols_txt}</div>" if cols_txt else ""}
                </div>
                """, unsafe_allow_html=True)

        # ── Cleaning plan ─────────────────────────────────────────────
        if dq["cleaning_plan"]:
            st.markdown("---")
            st.markdown(
                f"<div style='font-size:0.8rem;font-weight:600;color:{t['text_secondary']};"
                f"letter-spacing:.05em;text-transform:uppercase;margin-bottom:10px;'>"
                f"{icon_svg('bolt', size=13, color=t['primary'])}  Cleaning Plan</div>",
                unsafe_allow_html=True
            )
            for step in dq["cleaning_plan"]:
                sev_color = t["error"] if step["severity"] == "BLOCKING" else t["warning"]
                cols_txt = ", ".join(f"`{c}`" for c in step.get("columns", []))
                with st.expander(
                    f"Priority {step['priority']}  ·  {step['issue']}",
                    expanded=(step["severity"] == "BLOCKING")
                ):
                    if cols_txt:
                        st.markdown(f"**Affected columns:** {cols_txt}")
                    st.markdown(f"**Recommended action:** {step['action']}")
                    st.markdown(f"**Expected result:** {step['expected_result']}")

        # ── Info notices ──────────────────────────────────────────────
        if dq["info"]:
            st.markdown("---")
            for notice in dq["info"]:
                cols_txt = (", ".join(f"`{c}`" for c in notice.get("columns", []))
                            if notice.get("columns") else "")
                st.info(f"**{notice['issue']}** — {notice['observed']}"
                        + (f"  Columns: {cols_txt}" if cols_txt else ""))

        # ── CTA buttons ────────────────────────────────────────────────
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        if is_ready:
            if st.button("Continue to Analysis", type="primary", key="dq_continue"):
                st.session_state.active_tab = "Chat"
                st.rerun()
        else:
            st.warning(
                "Fix the blocking issues listed above, then re-upload your cleaned CSV. "
                "Warnings do not block analysis but should be reviewed."
            )
            if st.button("Re-upload a cleaned CSV", type="primary", key="dq_reupload"):
                # Reset everything so handle_csv_upload treats a new upload as fresh
                for k in ["loaded_filename", "loaded_upload_identity", "df", "dq_result", "insights", "quick_stats",
                           "profile", "llm_context", "report_md", "report_pdf",
                           "report_analysis", "report_config", "csv_bytes"]:
                    st.session_state[k] = None
                st.session_state.chats = {}
                st.session_state.current_chat_id = None
                st.session_state.active_tab = "Home"
                st.rerun()


# ── CHAT WITH DATA ─────────────────────────────────────────────
elif st.session_state.active_tab == "Chat":
    df = st.session_state.df
    profile = st.session_state.profile

    st.markdown(f"""
        <div class='topbar'>
            <div style='display:flex;align-items:center;gap:7px;font-size:0.85rem;color:{t["text_secondary"]};'>
                {icon_svg("file", size=15, color=t["muted"])}
                <span>{html.escape(str(st.session_state.loaded_filename))} &nbsp;·&nbsp;
                <b style='color:{t["text"]};font-weight:600;'>{profile['row_count']:,} rows</b>
                &nbsp;·&nbsp; {profile['column_count']} columns</span>
            </div>
            <span class='status-pill'><span class='status-dot'></span>Ready</span>
        </div>
    """, unsafe_allow_html=True)

    messages = get_current_messages()
    chat_title = st.session_state.chats[st.session_state.current_chat_id]["title"]

    st.markdown(f"""
    <div class="pane-header chat">
        <div class="pane-header-icon">
            {icon_svg('message', size=18, color='white')}
        </div>
        <div>
            <div class="pane-header-title">Chat with Data</div>
            <div class="pane-header-sub">Ask questions about your data in plain English. Get instant insights, visualizations, and summaries.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not messages:
        # ── Empty-state hero, matching the "Start a conversation" screen ──
        st.markdown(f"""
        <div style='text-align:center;padding:1.6rem 0 1.2rem;'>
            <div style='width:52px;height:52px;border-radius:14px;margin:0 auto 14px;
                        background:linear-gradient(135deg,{t["primary"]},{t["accent2"]});
                        display:flex;align-items:center;justify-content:center;'>
                {icon_svg("message", size=24, color="white", stroke_width="2")}
            </div>
            <div style='font-size:1.15rem;font-weight:700;color:{t["text"]};margin-bottom:4px;'>Start a conversation</div>
            <p style='color:{t["text_secondary"]};font-size:0.9rem;max-width:440px;margin:0 auto;'>
                Ask questions about your data, request analysis, or explore insights — all in plain English.
            </p>
        </div>
        """, unsafe_allow_html=True)

        starters = build_suggested_questions(profile, st.session_state.get("dq_result"))[:3]
        starter_cols = st.columns(len(starters))
        _starter_icons = [
            ":material/trending_up:",
            ":material/bar_chart:",
            ":material/bolt:",
        ]
        for i, (col, starter) in enumerate(zip(starter_cols, starters)):
            with col:
                if st.button(
                    starter,
                    key=f"starter_{i}",
                    use_container_width=True,
                    type="secondary",
                ):
                    ask_suggestion(starter)
    else:
        st.markdown(
            f"<div style='font-size:1rem;font-weight:600;color:{t['text']};margin-bottom:12px;'>{html.escape(str(chat_title))}</div>",
            unsafe_allow_html=True
        )

        for msg_idx, msg in enumerate(messages):
            with st.chat_message("user", avatar=AVATAR_USER):
                st.write(msg["question"])
            with st.chat_message("assistant", avatar=AVATAR_ASSISTANT):
                if msg.get("type") == "code":
                    # Identical to the live render (same two helpers), so a
                    # message looks the same the first time and on every
                    # Streamlit rerun: verified Python/SQL boxes first, then
                    # the validated sentence. render_framing re-validates at
                    # read time as defense-in-depth.
                    render_code_boxes(t, msg)
                    render_framing(msg, columns=list(df.columns))
                elif msg.get("type") == "forecast":
                    st.write(str(msg["answer"]).replace("\n", "  \n"))
                    if msg.get("chart_bytes"):
                        st.image(msg["chart_bytes"], use_container_width=True)
                else:
                    st.write(msg["answer"])
                if msg.get("type") == "chart":
                    # Rebuild the Plotly figure from the saved chart config.
                    # Do not depend on PNG/Kaleido bytes: suggested questions
                    # trigger st.rerun(), so the chart must be reproducible
                    # during chat-history replay.
                    chart_config = msg.get("chart_config")
                    if chart_config:
                        fig, chart_error = generate_chart(df, chart_config, theme=t)
                        if chart_error:
                            st.warning(f"Could not generate chart: {chart_error}")
                        else:
                            st.plotly_chart(
                                fig,
                                use_container_width=True,
                                key=f"history_chart_{st.session_state.current_chat_id}_{msg_idx}"
                            )
                    elif msg.get("chart_bytes"):
                        # Backward compatibility for older chat messages.
                        st.image(msg["chart_bytes"], use_container_width=True)

    user_question = st.chat_input("Ask a question about your data...")

    if user_question:
        from backend.llm_agent import answer_question

        with st.chat_message("user", avatar=AVATAR_USER):
            st.write(user_question)

        data_context = st.session_state.llm_context

        with st.chat_message("assistant", avatar=AVATAR_ASSISTANT):
            with st.spinner("Thinking..."):
                raw_result = answer_question(
                    user_question, data_context, messages,
                    df_columns=list(df.columns), df=df,
                    session_id=st.session_state.session_id,
                )
            new_msg = process_answer_result(raw_result, df, t, render_inline=True, question=user_question)

        new_msg["question"] = user_question
        messages.append(new_msg)
        # Invalidate report artifacts whenever chat changes. Otherwise the
        # Reports tab can legitimately show/download a PDF generated before
        # the latest chart question was asked.
        st.session_state.report_md = None
        st.session_state.report_pdf = None
        st.session_state.report_analysis = None
        st.session_state.report_config = None
        maybe_set_chat_title(st.session_state.current_chat_id, user_question)

    if messages:
        # Streamlit compatibility: older Streamlit versions used by this
        # project do not support st.container(key=...). In the Chat tab,
        # Clear Chat is the only main-area secondary button here, so scope
        # the styling to the main content and the secondary button type.
        st.markdown("""
        <style>
        /* ── Clear Chat — version-compatible styling ── */
        .main .stButton > button[kind="secondary"] {
            width: auto !important;
            min-width: 112px !important;
            height: 34px !important;
            padding: 6px 14px !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;

            background: #FFFFFF !important;
            color: #62636C !important;
            border: 1px solid #E1E3E8 !important;
            border-radius: 8px !important;

            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            font-size: 12px !important;
            font-weight: 500 !important;
            line-height: 1 !important;
            box-shadow: none !important;
            transform: none !important;
            transition: background-color .15s ease,
                        border-color .15s ease,
                        color .15s ease,
                        box-shadow .15s ease !important;
        }

        .main .stButton > button[kind="secondary"]:hover {
            background: #EEF2FF !important;
            color: #2563EB !important;
            border-color: #C7D2FE !important;
            box-shadow: 0 0 0 3px rgba(238,242,255,.85) !important;
        }

        .main .stButton > button[kind="secondary"]:focus-visible {
            background: #FFFFFF !important;
            color: #2563EB !important;
            border-color: #2563EB !important;
            box-shadow: 0 0 0 3px #EEF2FF !important;
            outline: none !important;
        }

        .main .stButton > button[kind="secondary"]:active {
            background: #E0E7FF !important;
            color: #1D4ED8 !important;
            border-color: #A5B4FC !important;
            box-shadow: none !important;
        }

        .main .stButton > button[kind="secondary"] p,
        .main .stButton > button[kind="secondary"] span,
        .main .stButton > button[kind="secondary"] div {
            color: inherit !important;
            font-family: inherit !important;
            font-size: inherit !important;
            font-weight: inherit !important;
        }
        </style>
        """, unsafe_allow_html=True)

        if st.button("Clear chat", key="clear_chat_btn", type="secondary"):
            messages.clear()
            st.rerun()


# ── ANALYSIS (Data Preview / Data Profile / AI Analysis) ───────
elif st.session_state.active_tab in ANALYSIS_SUBTABS:
    df = st.session_state.df
    profile = st.session_state.profile

    st.markdown(f"""
        <div class='topbar'>
            <div style='display:flex;align-items:center;gap:7px;font-size:0.85rem;color:{t["text_secondary"]};'>
                {icon_svg("file", size=15, color=t["muted"])}
                <span>{html.escape(str(st.session_state.loaded_filename))} &nbsp;·&nbsp;
                <b style='color:{t["text"]};font-weight:600;'>{profile['row_count']:,} rows</b>
                &nbsp;·&nbsp; {profile['column_count']} columns</span>
            </div>
            <span class='status-pill'><span class='status-dot'></span>Ready</span>
        </div>
    """, unsafe_allow_html=True)

    active_sub = st.session_state.active_tab

    st.markdown("""
    <style>
/* ── ANALYSIS TAB-PANE — NOT BUTTONS / NOT CARDS ────────────────
   Data Preview | Data Profile | AI Analysis
   The Streamlit buttons are used only as the interaction mechanism;
   visually they are a tab strip with a shared baseline and an active
   underline. No filled card, border, radius or button shadow. */

div[data-testid="stHorizontalBlock"]:has(.st-key-__analysisnav0):has(.st-key-__analysisnav1):has(.st-key-__analysisnav2) {
    gap:0 !important;
    margin:0 0 20px 0 !important;
    padding:0 !important;
    border-bottom:1px solid #E2E8F0 !important;
    background:transparent !important;
    box-shadow:none !important;
}

div[data-testid="stHorizontalBlock"]:has(.st-key-__analysisnav0):has(.st-key-__analysisnav1):has(.st-key-__analysisnav2) > div[data-testid="column"] {
    flex:1 1 0 !important;
    min-width:0 !important;
    padding:0 !important;
}

div[data-testid="stHorizontalBlock"]:has(.st-key-__analysisnav0):has(.st-key-__analysisnav1):has(.st-key-__analysisnav2) .stButton > button {
    position:relative !important;
    width:100% !important;
    height:46px !important;
    min-height:46px !important;
    padding:0 14px !important;
    margin:0 !important;

    background:transparent !important;
    color:#64748B !important;
    border:0 !important;
    border-radius:0 !important;
    box-shadow:none !important;

    font-size:.86rem !important;
    font-weight:500 !important;
    display:flex !important;
    align-items:center !important;
    justify-content:center !important;
    gap:7px !important;

    transition:color .18s ease, background-color .18s ease !important;
}

div[data-testid="stHorizontalBlock"]:has(.st-key-__analysisnav0):has(.st-key-__analysisnav1):has(.st-key-__analysisnav2) .stButton > button:hover {
    background:#F8FAFC !important;
    color:#2563EB !important;
    border:0 !important;
    border-radius:0 !important;
    box-shadow:none !important;
    transform:none !important;
}

div[data-testid="stHorizontalBlock"]:has(.st-key-__analysisnav0):has(.st-key-__analysisnav1):has(.st-key-__analysisnav2) .stButton > button[kind="primary"] {
    background:transparent !important;
    color:#2563EB !important;
    font-weight:650 !important;
    border:0 !important;
    border-radius:0 !important;
    box-shadow:none !important;
}

div[data-testid="stHorizontalBlock"]:has(.st-key-__analysisnav0):has(.st-key-__analysisnav1):has(.st-key-__analysisnav2) .stButton > button[kind="primary"]::after {
    content:"";
    position:absolute;
    left:16%;
    right:16%;
    bottom:0;
    height:3px;
    border-radius:3px 3px 0 0;
    background:linear-gradient(90deg,#2563EB,#7C3AED);
}

div[data-testid="stHorizontalBlock"]:has(.st-key-__analysisnav0):has(.st-key-__analysisnav1):has(.st-key-__analysisnav2) .stButton [data-testid="stIconMaterial"] {
    font-size:18px !important;
    color:#64748B !important;
    transition:color .18s ease !important;
}

div[data-testid="stHorizontalBlock"]:has(.st-key-__analysisnav0):has(.st-key-__analysisnav1):has(.st-key-__analysisnav2) .stButton > button[kind="primary"] [data-testid="stIconMaterial"] {
    color:#4F46E5 !important;
}

div[data-testid="stHorizontalBlock"]:has(.st-key-__analysisnav0):has(.st-key-__analysisnav1):has(.st-key-__analysisnav2) .stButton > button:hover [data-testid="stIconMaterial"] {
    color:#2563EB !important;
}
    </style>""", unsafe_allow_html=True)

    _nc = st.columns(3, gap="small")

    _analysis_tab_icons = [
        ":material/table_chart:",
        ":material/analytics:",
        ":material/auto_awesome:",
    ]

    for i, name in enumerate(ANALYSIS_SUBTABS):
        with _nc[i]:
            if st.button(
                name,
                key=f"__analysisnav{i}",
                use_container_width=True,
                type="primary" if active_sub == name else "secondary",
            ):
                st.session_state.active_tab = name
                st.session_state.last_analysis_tab = name
                st.rerun()

    # ── SUB-VIEW: DATA PREVIEW ───────────────────────────────────
    if active_sub == "Data Preview":
        # ── Data Preview pane header ─────────────────────────────────
        st.markdown(f"""
        <div class="pane-header preview">
            <div class="pane-header-icon">
                {icon_svg('barchart', size=18, color='white')}
            </div>
            <div>
                <div class="pane-header-title">Data Preview</div>
                <div class="pane-header-sub">
                    Browse, filter and explore your dataset before analysis.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Controls: row selector | filter | export buttons ─────────
        col_rows, col_search, col_export = st.columns([1, 2.5, 1.5])
        with col_rows:
            n_rows = st.selectbox("Show rows:", [5, 10, 25, 50, 100], index=0)
        with col_search:
            search = st.text_input(
                "Filter rows containing:",
                placeholder="Type to filter...",
            )
        with col_export:
            st.markdown("<div style='height:27px;'></div>", unsafe_allow_html=True)
            # Export CSV — two-step: prepare then download (existing logic preserved)
            if st.session_state.csv_bytes is None:
                if st.button(
                    "Export CSV",
                    key="prepare_csv",
                    type="secondary",
                    use_container_width=True,
                ):
                    with st.spinner("Preparing CSV export..."):
                        st.session_state.csv_bytes = df.to_csv(index=False).encode("utf-8")
                    st.rerun()
            else:
                st.download_button(
                    label="Download CSV",
                    data=st.session_state.csv_bytes,
                    file_name="data_export.csv",
                    mime="text/csv",
                    key="download_full_csv",
                    use_container_width=True,
                )

        # ── Filtering logic (unchanged) ───────────────────────────────
        display_df = df.head(n_rows)
        if search:
            # Scan column-by-column instead of materializing a second full
            # DataFrame with df.astype(str). This reduces peak memory usage
            # on large datasets while preserving generic cross-column search.
            needle = search.strip()
            if needle:
                mask = pd.Series(False, index=df.index)
                for col in df.columns:
                    try:
                        mask |= df[col].astype("string").str.contains(
                            needle, case=False, na=False, regex=False
                        )
                    except Exception:
                        continue
                display_df = df.loc[mask].head(n_rows)

        # ── Table ─────────────────────────────────────────────────────
        render_table_html(display_df, t)

        # ── Row count caption ─────────────────────────────────────────
        total_shown = len(display_df)
        total_rows = len(df)
        if search and search.strip():
            row_label = f"Showing <b>{total_shown}</b> matching rows of <b>{total_rows:,}</b> total"
        else:
            row_label = f"Showing <b>{total_shown}</b> of <b>{total_rows:,}</b> rows"
        st.markdown(f"<div class='dp-row-count'>{row_label}</div>", unsafe_allow_html=True)

    # ── SUB-VIEW: DATA PROFILE ───────────────────────────────────
    elif active_sub == "Data Profile":
        st.markdown(f"""
        <div class="pane-header profile">
            <div class="pane-header-icon">
                {icon_svg('search', size=18, color='white')}
            </div>
            <div>
                <div class="pane-header-title">Data Profile</div>
                <div class="pane-header-sub">Column types, missing values and distribution stats</div>
            </div>
        </div>""", unsafe_allow_html=True)

        # Theme-consistent metric cards — clean white surfaces with restrained accent colors.
        mem_kb = df.memory_usage(deep=True).sum() / 1024
        mem_str = f"{mem_kb:.1f} KB" if mem_kb < 1024 else f"{mem_kb/1024:.2f} MB"
        missing_total = profile['missing_values_total']
        missing_color = "#DC2626" if missing_total > 0 else t["success"]

        st.markdown(f"""
        <style>
            .profile-metrics {{
                display:grid;
                grid-template-columns:repeat(4,minmax(0,1fr));
                gap:12px;
                margin-bottom:24px;
            }}
            .profile-metric {{
                position:relative;
                overflow:hidden;
                background:linear-gradient(145deg, {t['surface']} 0%, #F8FAFC 100%);
                border:1px solid {t['border']};
                border-radius:14px;
                padding:16px 18px 18px;
                box-shadow:0 2px 8px rgba(15,23,42,.045);
                transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease;
            }}
            .profile-metric:hover {{
                transform:translateY(-2px);
                box-shadow:0 7px 18px rgba(15,23,42,.08);
                border-color:{t['border_strong']};
            }}
            .profile-metric::before {{
                content:"";
                position:absolute;
                left:0;
                top:0;
                bottom:0;
                width:4px;
                background:var(--metric-accent);
                border-radius:14px 0 0 14px;
            }}
            .profile-metric-icon {{
                width:34px;
                height:34px;
                border-radius:10px;
                display:flex;
                align-items:center;
                justify-content:center;
                margin-bottom:12px;
                background:color-mix(in srgb, var(--metric-accent) 12%, white);
                color:var(--metric-accent);
            }}
            .profile-metric::before {{
                content:"";
                position:absolute;
                left:0; right:0; top:0;
                height:3px;
                background:var(--metric-accent);
            }}
            .profile-metric:hover {{
                border-color:{t['border_strong']};
                box-shadow:0 4px 14px rgba(16,17,20,.06);
                transform:translateY(-1px);
            }}
            .profile-metric-label {{
                font-size:10px;
                font-weight:600;
                color:var(--metric-accent);
                letter-spacing:.05em;
                margin-bottom:7px;
            }}
            .profile-metric-value {{
                font-size:1.8rem;
                line-height:1.1;
                font-weight:700;
                color:{t['text']};
                letter-spacing:-.025em;
            }}
            @media (max-width: 900px) {{
                .profile-metrics {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
            }}
            @media (max-width: 520px) {{
                .profile-metrics {{ grid-template-columns:1fr; }}
            }}
        </style>
        <div class="profile-metrics">
            <div class="profile-metric" style="--metric-accent:{t['primary']};">
                <div class="profile-metric-icon">{icon_svg('table', size=17, color=t['primary'])}</div>
                <div class="profile-metric-label">ROWS</div>
                <div class="profile-metric-value">{profile['row_count']:,}</div>
            </div>
            <div class="profile-metric" style="--metric-accent:{t['accent2']};">
                <div class="profile-metric-icon">{icon_svg('columns', size=17, color=t['accent2'])}</div>
                <div class="profile-metric-label">COLUMNS</div>
                <div class="profile-metric-value">{profile['column_count']}</div>
            </div>
            <div class="profile-metric" style="--metric-accent:{missing_color};">
                <div class="profile-metric-icon">{icon_svg('warning', size=17, color=missing_color)}</div>
                <div class="profile-metric-label">MISSING</div>
                <div class="profile-metric-value">{missing_total}</div>
            </div>
            <div class="profile-metric" style="--metric-accent:#64748B;">
                <div class="profile-metric-icon">{icon_svg('database', size=17, color='#64748B')}</div>
                <div class="profile-metric-label">MEMORY</div>
                <div class="profile-metric-value">{mem_str}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
            {icon_svg('search', size=14, color=t['primary'])}
            <span style="font-size:0.85rem;font-weight:600;color:{t['text_secondary']};
                         letter-spacing:.05em;text-transform:uppercase;">Column details</span>
        </div>""", unsafe_allow_html=True)

        def _dtype_badge(dtype_str):
            s = str(dtype_str)
            if s == "object":   return f'<span class="dtype-badge dtype-object">object</span>'
            if "int" in s:      return f'<span class="dtype-badge dtype-int">{s}</span>'
            if "float" in s:    return f'<span class="dtype-badge dtype-float">{s}</span>'
            if "datetime" in s: return f'<span class="dtype-badge dtype-datetime">datetime</span>'
            if "bool" in s:     return f'<span class="dtype-badge dtype-bool">bool</span>'
            return f'<span class="dtype-badge dtype-other">{s[:10]}</span>'

        for col_info in profile["columns"]:
            dtype = str(col_info["dtype"])
            # Streamlit expander labels support inline Markdown.  Code spans
            # are styled above as compact datatype badges.
            expander_label = f"{col_info['name']}  ·  `{dtype}`"
            with st.expander(expander_label):
                c1, c2, c3 = st.columns(3)
                c1.metric("Unique values", f"{col_info['unique_count']:,}")
                c2.metric("Missing", f"{col_info['missing_count']:,}")
                missing_pct = round(col_info['missing_count'] / profile['row_count'] * 100, 1)
                c3.metric("Missing %", f"{missing_pct}%")

                if col_info.get("stats"):
                    s = col_info["stats"]
                    st.markdown(
                        f"**Min:** `{s['min']}` &nbsp;|&nbsp; "
                        f"**Max:** `{s['max']}` &nbsp;|&nbsp; "
                        f"**Mean:** `{s['mean']}` &nbsp;|&nbsp; "
                        f"**Median:** `{s['median']}`"
                    )
                elif col_info.get("all_missing"):
                    st.caption("This column has no non-missing values, so no statistics are available.")

                if "top_values" in col_info:
                    st.markdown("**Top values:**")
                    render_kv_list_html(col_info["top_values"], t)

    # ── SUB-VIEW: AI ANALYSIS ────────────────────────────────────
    elif active_sub == "AI Analysis":
        st.markdown(f"""
        <div class="pane-header analysis">
            <div class="pane-header-icon">
                {icon_svg('bolt', size=18, color='white')}
            </div>
            <div>
                <div class="pane-header-title">AI Analysis</div>
                <div class="pane-header-sub">Auto-generated insights from your dataset</div>
            </div>
        </div>""", unsafe_allow_html=True)

        if st.session_state.insights is None:
            st.info("Upload a CSV to automatically generate AI insights.")

        elif len(st.session_state.insights) == 0:
            st.warning("No insights could be generated. Try a richer dataset.")

        else:
            if st.session_state.quick_stats:
                qs = st.session_state.quick_stats
                c1, c2, c3 = st.columns(3)
                c1.metric("Numeric columns", len(qs["numeric_columns"]))
                c2.metric("Categorical columns", len(qs["categorical_columns"]))
                c3.metric("Columns with missing data", len(qs["high_missing"]))

            st.divider()
            st.markdown(
                f"<div style='font-size:0.8rem;font-weight:600;color:{t['text_secondary']};"
                f"display:flex;align-items:center;gap:6px;margin-bottom:10px;'>"
                f"{icon_svg('bolt', size=13, color=t['muted'])} KEY FINDINGS</div>",
                unsafe_allow_html=True
            )

            _insight_accents = [t["primary"], t["accent2"], t["success"], "#D97706"]
            for i, insight in enumerate(st.session_state.insights):
                accent = _insight_accents[i % len(_insight_accents)]
                st.markdown(f"""
                <div class='insight-card' style='--insight-accent-color:{accent};'>
                    <div class='insight-icon-box' style='--insight-accent:{accent}22;'>
                        {icon_svg('bolt', size=17, color=accent)}
                    </div>
                    <div style='min-width:0;flex:1;'>
                        <div class='insight-title'>{html.escape(str(insight['title']))}</div>
                        <div class='insight-body'>{html.escape(str(insight['body']))}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.divider()
            st.info("Head to the **Reports** page for suggested follow-up questions and to export this analysis as PDF or Markdown.")


# ── REPORTS ──────────────────────────────────────────────────────
elif st.session_state.active_tab == "Reports":
    df = st.session_state.df
    profile = st.session_state.profile

    st.markdown(f"""
        <div class='topbar'>
            <div style='display:flex;align-items:center;gap:7px;font-size:0.85rem;color:{t["text_secondary"]};'>
                {icon_svg("file", size=15, color=t["muted"])}
                <span>{html.escape(str(st.session_state.loaded_filename))} &nbsp;·&nbsp;
                <b style='color:{t["text"]};font-weight:600;'>{profile['row_count']:,} rows</b>
                &nbsp;·&nbsp; {profile['column_count']} columns</span>
            </div>
            <span class='status-pill'><span class='status-dot'></span>Ready</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="pane-header reports">
        <div class="pane-header-icon">
            {icon_svg('file', size=18, color='white')}
        </div>
        <div>
            <div class="pane-header-title">Reports</div>
            <div class="pane-header-sub">Suggested questions · Report export</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── SUGGESTED QUESTIONS ────────────────────────────────────────
    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:10px;margin-bottom:14px;'>
        <div style='width:32px;height:32px;border-radius:8px;flex-shrink:0;background:{t["soft_surface"]};
                    display:flex;align-items:center;justify-content:center;'>
            {icon_svg('file', size=16, color=t['primary'])}
        </div>
        <div>
            <div style='font-weight:700;font-size:0.95rem;color:{t["text"]};'>Suggested Questions</div>
            <div style='font-size:0.78rem;color:{t["text_secondary"]};'>Click on a question to ask, or explore your data further</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.insights is None:
        st.info("Upload a CSV to generate suggested questions.")
    else:
        suggestions = build_suggested_questions(profile, st.session_state.get("dq_result"))

        # Suggested-question ACTION BUTTONS. The key-scoped CSS gives them
        # the Export Report visual effect without turning them into cards.
        st.markdown(
            "<div class='suggested-question-style-anchor'></div>",
            unsafe_allow_html=True
        )

        btn_cols = st.columns(2, gap="medium")
        _question_icons = ["trending", "barchart", "bolt", "search", "trending", "file"]
        for i, suggestion in enumerate(suggestions[:6]):
            with btn_cols[i % 2]:
                icon_name = _question_icons[i % len(_question_icons)]
                if st.button(
                    suggestion,
                    key=f"suggestion_{i}",
                    use_container_width=True,
                    type="secondary",

                ):
                    ask_suggestion(suggestion)

    # ── REPORT EXPORT / BUILDER ───────────────────────────────
    st.divider()

    st.markdown(f"""
    <div class="export-section">
        <div class="export-header">
            <div class="export-header-icon">
                {icon_svg("file", size=18, color="white")}
            </div>
            <div>
                <div class="export-header-title">Export Report</div>
                <div class="export-header-sub">One complete, consistent report — no configuration required</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── FIXED REPORT FORMAT ────────────────────────────────────────
    # Reports are intentionally standardized. This is informational
    # only — there are no selectable sections or configuration cards.
    st.markdown(
        f"""
        <div class="report-blueprint">
            <div class="report-blueprint-top">
                <div>
                    <div class="report-blueprint-title">Standard report contents</div>
                    <div class="report-blueprint-sub">
                        Every generated report includes the dataset overview, data preview,
                        exploratory analysis, visualizations, AI insights, and available chat analysis.
                    </div>
                </div>
            </div>
            <div class="report-content-grid">
                <div class="report-content-item"><span class="report-content-item-icon">{icon_svg("database", size=13, color="#4F6FF5")}</span>Dataset overview</div>
                <div class="report-content-item"><span class="report-content-item-icon">{icon_svg("table", size=13, color="#4F6FF5")}</span>Data preview</div>
                <div class="report-content-item"><span class="report-content-item-icon">{icon_svg("bolt", size=13, color="#4F6FF5")}</span>EDA</div>
                <div class="report-content-item"><span class="report-content-item-icon">{icon_svg("bar-chart", size=13, color="#4F6FF5")}</span>Visualizations</div>
                <div class="report-content-item"><span class="report-content-item-icon">{icon_svg("lightbulb", size=13, color="#4F6FF5")}</span>AI insights</div>
                <div class="report-content-item"><span class="report-content-item-icon">{icon_svg("message", size=13, color="#4F6FF5")}</span>Chat analysis</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Fixed settings: the report is always complete and consistent.
    include_preview = True
    include_eda = True
    include_charts = True
    include_chat = True
    preview_rows = 5
    make_pdf = True
    make_markdown = True

    current_report_config = {
        "format": "standard_complete",
        "include_preview": True,
        "include_eda": True,
        "include_charts": True,
        "include_chat": True,
        "preview_rows": 5,
        "make_pdf": True,
        "make_markdown": True,
    }

    # If the user changes any option, invalidate the old output so an
    # accidentally stale report can never be downloaded.
    if st.session_state.report_config != current_report_config:
        st.session_state.report_md = None
        st.session_state.report_pdf = None
        st.session_state.report_analysis = None

    if st.button("Generate Report", key="generate_report", type="primary", use_container_width=True):
        progress = st.empty()
        # Carries the real uploaded filename into report_generator, which
        # only ever sees `profile` and `df` — not st.session_state. Done
        # as a shallow copy so we never mutate the shared profile dict
        # other parts of the app rely on.
        profile_for_report = {**profile, "dataset_name": st.session_state.get("loaded_filename")}
        try:
            progress.info("Step 1/3 — Calculating shared dataset analysis...")
            report_data = prepare_report_analysis(
                df,
                profile_for_report,
                st.session_state.insights,
                st.session_state.quick_stats,
                st.session_state.chats,
                include_preview=include_preview,
                include_eda=include_eda,
                include_charts=include_charts,
                include_chat=include_chat,
                preview_rows=preview_rows if include_preview else 0,
            )
            st.session_state.report_analysis = report_data

            st.session_state.report_md = None
            st.session_state.report_pdf = None

            if make_markdown:
                progress.info("Step 2/3 — Building Markdown report...")
                st.session_state.report_md = generate_markdown_report(
                    df, profile_for_report, st.session_state.insights,
                    st.session_state.quick_stats, st.session_state.chats,
                    report_data=report_data,
                )

            if make_pdf:
                progress.info("Step 3/3 — Building PDF report...")
                try:
                    st.session_state.report_pdf = generate_pdf_report(
                        df, profile_for_report, st.session_state.insights,
                        st.session_state.quick_stats, st.session_state.chats,
                        report_data=report_data,
                    )
                except Exception:
                    st.session_state.report_pdf = None
                    logger.exception("PDF generation failed")
                    st.warning(
                        "PDF generation failed. Your Markdown report is still available above — "
                        "you can try the PDF again, or try with charts disabled."
                    )

            st.session_state.report_config = current_report_config
            progress.markdown(
                f"""
                <div class="report-success-card">
                    <div class="report-success-icon">{icon_svg("check", size=15, color="#059669")}</div>
                    <div>
                        <div class="report-success-title">Report generated successfully</div>
                        <div class="report-success-sub">The same complete report is available in PDF and Markdown.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception:
            st.session_state.report_md = None
            st.session_state.report_pdf = None
            st.session_state.report_analysis = None
            logger.exception("Report generation failed")
            st.error(
                "Report generation failed. Please try again — if it keeps happening, "
                "try reducing the preview row count or disabling charts."
            )

    if st.session_state.report_config == current_report_config and (
            st.session_state.report_md is not None or st.session_state.report_pdf is not None
        ):
            st.markdown(
                f"""
                <div class="report-ready-heading">
                    <div class="report-ready-icon">{icon_svg("download", size=15, color="#059669")}</div>
                    <div>
                        <div class="report-ready-title">Report ready to download</div>
                        <div class="report-ready-sub">Download the complete report in your preferred format.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            has_md = st.session_state.report_md is not None
            has_pdf = st.session_state.report_pdf is not None

            if has_md and has_pdf:
                ex_c1, ex_c2 = st.columns(2, gap="medium")
                download_cols = [(ex_c1, "md"), (ex_c2, "pdf")]
            else:
                download_cols = [(st.container(), "md" if has_md else "pdf")]

            for col, fmt in download_cols:
                with col:
                    if fmt == "md":
                        st.download_button(
                            label="Download Markdown  ·  GitHub · Notion · Obsidian",
                            data=st.session_state.report_md.encode("utf-8"),
                            file_name="axiomrow_report.md",
                            mime="text/markdown",
                            use_container_width=True,
                            key="dl_md_final",
                        )
                    else:
                        st.download_button(
                            label="Download PDF  ·  Print-ready · Shareable",
                            data=st.session_state.report_pdf,
                            file_name="axiomrow_report.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            key="dl_pdf_final",
                        )



# ── SETTINGS ─────────────────────────────────────────────────────
elif st.session_state.active_tab == "Settings":
    st.markdown(f"""
    <div class="pane-header settings">
        <div class="pane-header-icon">
            {icon_svg('settings', size=18, color='white')}
        </div>
        <div>
            <div class="pane-header-title">Settings</div>
            <div class="pane-header-sub">Manage your dataset and session</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Dataset ────────────────────────────────────────────────
    st.markdown(f"<h4 style='color:{t['text']};margin-bottom:2px;'>Dataset</h4>", unsafe_allow_html=True)

    if st.session_state.df is not None:
        df = st.session_state.df
        profile = st.session_state.profile
        mem_kb = df.memory_usage(deep=True).sum() / 1024
        mem_str = f"{mem_kb:.1f} KB" if mem_kb < 1024 else f"{mem_kb/1024:.2f} MB"

        render_kv_list_html({
            "File name": st.session_state.loaded_filename,
            "Rows": f"{profile['row_count']:,}",
            "Columns": profile['column_count'],
            "Memory usage": mem_str,
        }, t)

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        with st.expander("Upload a different file"):
            settings_uploaded_file = st.file_uploader(
                "Choose a CSV file",
                help="Max file size: 150MB",
                label_visibility="collapsed",
                key="settings_uploader",
            )
            handle_csv_upload(settings_uploaded_file)
    else:
        st.info("No dataset loaded yet.")
        if st.button("Go to Home to upload a CSV", type="secondary", key="settings_go_home"):
            st.session_state.active_tab = "Home"
            st.rerun()

    st.divider()

    # ── Session ────────────────────────────────────────────────
    st.markdown(f"<h4 style='color:{t['text']};margin-bottom:2px;'>Session</h4>", unsafe_allow_html=True)
    st.caption("These actions apply only to your current browser session.")

    sc1, sc2 = st.columns(2)
    with sc1:
        if st.button("Clear chat history", type="secondary", use_container_width=True,
                      disabled=st.session_state.df is None, key="settings_clear_chats"):
            st.session_state.chats = {}
            st.session_state.current_chat_id = None
            create_new_chat()
            st.toast("Chat history cleared.", )
            st.rerun()
    with sc2:
        if st.button("Reset everything", type="secondary", use_container_width=True, key="settings_reset_all"):
            for key in [
                "loaded_filename", "loaded_upload_identity", "dq_result", "df", "chats", "current_chat_id", "insights",
                "quick_stats", "profile", "llm_context", "report_md", "report_pdf",
                "report_analysis", "report_config", "csv_bytes",
            ]:
                st.session_state[key] = None
            st.session_state.chats = {}
            st.session_state.active_tab = "Home"
            st.toast("Session reset.", )
            st.rerun()

    st.divider()

    # ── About ──────────────────────────────────────────────────
    st.markdown(f"<h4 style='color:{t['text']};margin-bottom:2px;'>About</h4>", unsafe_allow_html=True)
    render_kv_list_html({
        "App": "Axiomrow",
        "Stack": "Streamlit + Python + Groq",
        "Version": "1.0",
    }, t)
