# theme.py
"""
Axiomrow's visual theme — the color palette and the global CSS injector
— pulled out of app.py so the two things that were previously
interleaved (application logic and ~1,300 lines of CSS) can each be
read, and changed, on their own.

Nothing here has any application logic in it: T is a plain dict, and
apply_theme() just writes that dict into a <style> block once at
startup. app.py imports both and threads T through render functions
exactly as it did before this split (most functions still take a
`t: dict` parameter) — this file changes where the definitions live,
not how the rest of the app uses them.
"""

import streamlit as st

T = {
    "primary":        "#2563EB",
    "primary_hover":  "#1D4ED8",
    "secondary":      "#1E40AF",
    "accent2":        "#7C3AED",   # the ONE extra hue — used for the brand/AI gradient signature

    "bg":             "#F1F4F9",
    "surface":        "#FFFFFF",
    "surface_muted":  "#F8FAFC",
    "soft_surface":   "#EEF2FF",

    "border":         "#E7E9F0",
    "border_strong":  "#DDE1EA",
    "border_dashed":  "#C7D2E8",

    "text":           "#101114",
    "text_secondary": "#62636C",
    "muted":          "#9A9BA3",
    "disabled":       "#C4C6CC",

    "success":        "#059669",
    "success_bg":     "#ECFDF5",
    "warning":        "#D97706",
    "warning_bg":     "#FFFBEB",
    "error":          "#DC2626",
    "error_bg":       "#FEF2F2",

    "plotly_template": "plotly_white",
    "plotly_colors": ["#2563EB", "#7C3AED", "#1E40AF", "#60A5FA", "#334155"],
}


def apply_theme(t: dict):
    st.markdown(f"""
    <style>
   @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
   @import url('https://cdn.jsdelivr.net/npm/bootstrap-icons@1.13.1/font/bootstrap-icons.css');
   @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,300..700,0..1,-50..200&display=swap');

    .stApp, .stApp * {{
    font-family: 'Inter', -apple-system, sans-serif;
}}

    code, pre, pre * {{
        font-family: 'SF Mono', 'Fira Code', monospace !important;
    }}

    [data-testid="stIconMaterial"],
    span[data-testid="stIconMaterial"] {{
        font-family: 'Material Symbols Rounded' !important;
        font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24 !important;
    }}

    .stApp {{ background-color: {t['bg']}; }}
    .block-container {{ padding-top: 1.6rem !important; }}

    /* ── Sidebar — dark navy, matches the reference product shots ── */
    [data-testid="stSidebar"] {{
        background-color: #0b1120 !important;
        border-right: 1px solid #1e293b !important;
    }}
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] li {{
        color: #94a3b8 !important;
        opacity: 1 !important;
    }}
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: #475569 !important;
        opacity: 1 !important;
    }}
    [data-testid="stSidebar"] button p::first-letter {{
        font-family: "bootstrap-icons" !important;
    }}

    /* ── Recent chat buttons — fixed, stable size ───────────────────
       Keep chat rows from stretching/shrinking with the sidebar or the
       adjacent delete column.  This also prevents Streamlit from
       treating the title as a dynamically-sized/truncated control. */
    [data-testid="stSidebar"] [class*="st-key-chatbtn_"] {{
        width: 280px !important;
        min-width: 280px !important;
        max-width: 280px !important;
        flex: 0 0 280px !important;
    }}
    [data-testid="stSidebar"] [class*="st-key-chatbtn_"] > div {{
        width: 280px !important;
        min-width: 280px !important;
        max-width: 280px !important;
    }}
    [data-testid="stSidebar"] [class*="st-key-chatbtn_"] button {{
        width: 280px !important;
        min-width: 280px !important;
        max-width: 280px !important;
        height: 40px !important;
        min-height: 40px !important;
        max-height: 40px !important;
        padding: 8px 12px !important;
        overflow: hidden !important;
        white-space: nowrap !important;
        text-overflow: ellipsis !important;
        box-sizing: border-box !important;
    }}
    /* Recent-chat row: keep the chat selector the same 280px width as the
       sidebar navigation buttons.  The delete control stays compact at the
       far right without shrinking the chat selector. */
    [data-testid="stSidebar"] [class*="st-key-chatbtn_"] {{
        margin-right: -54px !important;
        position: relative !important;
        z-index: 2 !important;
    }}
    [data-testid="stSidebar"] [class*="st-key-delbtn_"] {{
        position: relative !important;
        z-index: 3 !important;
    }}

    [data-testid="stSidebar"] [class*="st-key-chatbtn_"] button p {{
        overflow: hidden !important;
        white-space: nowrap !important;
        text-overflow: ellipsis !important;
        margin: 0 !important;
        line-height: 1.2 !important;
    }}

    /* ── ALL sidebar buttons (nav items + recent-chat rows) ── */
    [data-testid="stSidebar"] button {{
        background-color: transparent !important;
        color: #94a3b8 !important;
        border: none !important;
        text-align: left !important;
        font-weight: 500 !important;
        width: 100% !important;
        padding: 8px 11px !important;
        border-radius: 8px !important;
        transition: background 0.15s, color 0.15s !important;
    }}
    [data-testid="stSidebar"] button * {{
        color: #94a3b8 !important;
    }}
    [data-testid="stSidebar"] button:hover:not(:disabled) {{
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
    }}
    [data-testid="stSidebar"] button:hover:not(:disabled) * {{
        color: #e2e8f0 !important;
    }}
    /* Active nav item / active chat row — solid gradient pill,
       matching the reference's blue→violet active-state highlight. */
    [data-testid="stSidebar"] button[kind="primary"] {{
        background: linear-gradient(135deg, {t['primary']}, {t['accent2']}) !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 10px rgba(37,99,235,0.35) !important;
    }}
    [data-testid="stSidebar"] button[kind="primary"] * {{
        color: #ffffff !important;
    }}
    [data-testid="stSidebar"] button[kind="primary"]:hover {{
        filter: brightness(1.05) !important;
    }}
    /* Disabled nav items (data-dependent pages before a CSV is loaded) */
    [data-testid="stSidebar"] button:disabled {{
        background-color: transparent !important;
        color: #334155 !important;
        opacity: 0.65 !important;
        cursor: not-allowed !important;
    }}
    [data-testid="stSidebar"] button:disabled * {{
        color: #334155 !important;
    }}
    /* Divider inside sidebar */
    [data-testid="stSidebar"] hr {{
        border-color: #1e293b !important;
        margin: 14px 0 !important;
    }}
    /* File uploader on dark sidebar/backgrounds (Settings re-upload) */
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
        background-color: #1e293b !important;
        border-color: #334155 !important;
    }}
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]:hover {{
        background-color: #1e293b !important;
        border-color: #3b82f6 !important;
    }}
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] div,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] span {{
        color: #64748b !important;
    }}
    /* Caption at bottom */
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] .stCaption * {{
        color: #334155 !important;
    }}

    /* ── Kill every white/light background inside sidebar ── */
    [data-testid="stSidebar"] section,
    [data-testid="stSidebar"] [data-testid="stSidebarContent"],
    [data-testid="stSidebar"] [data-testid="stSidebarUserContent"],
    [data-testid="stSidebarUserContent"],
    [data-testid="stSidebarContent"] {{
        background-color: #0b1120 !important;
    }}
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebar"] > div > div,
    [data-testid="stSidebar"] > div > div > div {{
        background-color: #0b1120 !important;
    }}
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"],
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: #0b1120 !important;
        border: none !important;
        box-shadow: none !important;
    }}
    /* Success/info alert on dark sidebar — subtle, not a bright box */
    [data-testid="stSidebar"] [data-testid="stAlert"],
    [data-testid="stSidebar"] .stAlert,
    [data-testid="stSidebar"] [data-testid="stAlertContainer"],
    [data-testid="stSidebar"] [class*="AlertContainer"],
    [data-testid="stSidebar"] [class*="alert"] {{
        background-color: #0f2d1f !important;
        border: 1px solid #14532d !important;
        color: #34d399 !important;
        border-radius: 6px !important;
        box-shadow: none !important;
    }}
    [data-testid="stSidebar"] [data-testid="stAlert"] *,
    [data-testid="stSidebar"] .stAlert * {{
        color: #34d399 !important;
        background-color: transparent !important;
    }}
    /* File uploader wrapper cards */
    [data-testid="stSidebar"] [data-testid="stFileUploader"],
    [data-testid="stSidebar"] [data-testid="stFileUploaderFile"],
    [data-testid="stSidebar"] [class*="uploadedFile"] {{
        background-color: #1e293b !important;
        border-color: #334155 !important;
    }}
    [data-testid="stSidebar"] [data-testid="stFileUploaderFile"] * {{
        color: #94a3b8 !important;
    }}

    /* ── Suggested-question pills (Reports) — list style with a
       trailing chevron, matching the reference's "› " rows.        */
    .stButton > button[kind="secondary"][data-testid*="suggestion"] {{
        background-color: #ffffff !important;
        color: {t['secondary']} !important;
        border: 1.5px solid #bfdbfe !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        text-align: left !important;
        justify-content: space-between !important;
        transition: all 0.15s !important;
        box-shadow: 0 1px 3px rgba(37,99,235,0.08) !important;
    }}
    .stButton > button[kind="secondary"][data-testid*="suggestion"]:hover {{
        background-color: {t['soft_surface']} !important;
        border-color: {t['primary']} !important;
        color: {t['primary_hover']} !important;
        box-shadow: 0 2px 8px rgba(37,99,235,0.15) !important;
        transform: translateY(-1px) !important;
    }}
    /* ── Starter pills (Chat empty state) — rounded, centered look */
    .stButton > button[kind="secondary"][data-testid*="starter"] {{
        background-color: {t['surface_muted']} !important;
        color: {t['text_secondary']} !important;
        border: 1px solid {t['border_strong']} !important;
        border-radius: 999px !important;
        font-weight: 500 !important;
        font-size: 0.82rem !important;
    }}
    .stButton > button[kind="secondary"][data-testid*="starter"]:hover {{
        border-color: {t['primary']} !important;
        color: {t['primary']} !important;
        background-color: {t['soft_surface']} !important;
    }}
    /* ── Quick-link cards on the loaded Home dashboard ── */
    .stButton > button[kind="secondary"][data-testid*="quicklink"] {{
        background-color: {t['surface']} !important;
        color: {t['primary']} !important;
        border: 1px solid {t['border_strong']} !important;
        font-weight: 600 !important;
    }}
    .stButton > button[kind="secondary"][data-testid*="quicklink"]:hover {{
        border-color: {t['primary']} !important;
        box-shadow: 0 0 0 3px {t['soft_surface']} !important;
    }}

    /* ── Metric cards ── */
    [data-testid="stMetric"] {{
        background-color: {t['surface']};
        border: 1px solid {t['border']};
        border-radius: 10px;
        padding: 14px 16px !important;
        transition: border-color 0.15s;
    }}
    [data-testid="stMetric"]:hover {{ border-color: {t['border_strong']}; }}
    [data-testid="stMetricValue"] {{
        color: {t['text']} !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
    }}
    [data-testid="stMetricLabel"] p {{
        color: {t['text_secondary']} !important;
        opacity: 1 !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
    }}

    /* replace the generic rule with: */
    [data-testid="stVerticalBlockBorderWrapper"]:not(:has(> [data-testid="stVerticalBlock"] [data-testid="stVerticalBlockBorderWrapper"])):has(> div > [data-testid="stVerticalBlock"] .stMarkdown) {{ }}

    /* ── Chat ── */
    [data-testid="stChatMessage"] {{
        background-color: {t['surface']};
        border-radius: 10px;
        border: 1px solid {t['border']};
        margin-bottom: 8px;
    }}
    .stApp:has(.cover-topbar) [data-testid="stHorizontalBlock"] [data-testid="stVerticalBlockBorderWrapper"] {{
        border: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
    }}
    [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessageContent"] span {{
        color: {t['text']} !important;
        opacity: 1 !important;
        font-size: 0.9rem !important;
    }}
    [data-testid="stChatInput"] {{
        border-radius: 14px !important;
    }}
    [data-testid="stChatInput"] textarea {{
        background-color: {t['surface']} !important;
        border: 1px solid {t['border_strong']} !important;
        color: {t['text']} !important;
        border-radius: 14px !important;
    }}
    /* Subtle gradient glow ring on focus — the ONLY other place the
       gradient signature appears besides the AI avatar/active nav. */
    [data-testid="stChatInput"]:focus-within textarea {{
        border-color: transparent !important;
        box-shadow: 0 0 0 2px {t['bg']}, 0 0 0 4px {t['primary']}55 !important;
    }}
    [data-testid="stChatInputSubmitButton"] {{ color: {t['primary']} !important; }}
    [data-testid="stChatInputSubmitButton"]:hover {{ background-color: {t['soft_surface']} !important; }}

    /* ── Buttons ── */
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, {t['primary']}, {t['accent2']});
        color: #ffffff;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.15s;
    }}
    .stButton > button[kind="primary"]:hover {{
        filter: brightness(1.05);
        box-shadow: 0 2px 10px {t['primary']}44;
    }}
    .stButton > button[kind="secondary"] {{
        background-color: {t['surface']};
        color: {t['text_secondary']};
        border: 1px solid {t['border_strong']};
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.15s;
    }}
    .stButton > button[kind="secondary"]:hover {{
        border-color: {t['primary']};
        color: {t['primary']} !important;
        box-shadow: 0 0 0 3px {t['soft_surface']};
    }}
    .stButton > button {{
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.15s;
    }}
    .stButton > button:disabled {{
        background: {t['border']} !important;
        color: {t['disabled']} !important;
        border-color: {t['border']} !important;
    }}
    [data-testid="stDownloadButton"] button {{
        background-color: {t['surface']};
        color: {t['text_secondary']};
        border: 1px solid {t['border_strong']};
        border-radius: 8px;
    }}
    [data-testid="stDownloadButton"] button:hover {{
        border-color: {t['primary']};
        color: {t['primary']} !important;
        box-shadow: 0 0 0 3px {t['soft_surface']};
    }}

    /* ── STRICT SIDEBAR NAV OVERRIDE ──────────────────────────────
       Sidebar navigation is NAV UI, never a card. This late rule
       intentionally wins over the generic button styles below.
       It also keeps Chat with Data visually identical on Home,
       Analysis, Reports, etc. unless it is the active destination. */
    [data-testid="stSidebar"] .stButton > button[kind="secondary"] {{
        background:transparent !important;
        color:#94a3b8 !important;
        border:0 !important;
        border-radius:8px !important;
        box-shadow:none !important;
        font-weight:500 !important;
        transform:none !important;
    }}

    [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {{
        background:#1e293b !important;
        color:#e2e8f0 !important;
        border:0 !important;
        box-shadow:none !important;
        transform:none !important;
    }}

    [data-testid="stSidebar"] .stButton > button[kind="primary"] {{
        background:linear-gradient(135deg,#2563EB,#7C3AED) !important;
        color:#ffffff !important;
        border:0 !important;
        border-radius:8px !important;
        box-shadow:0 2px 10px rgba(37,99,235,.35) !important;
        font-weight:600 !important;
    }}

    [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {{
        filter:brightness(1.05) !important;
        transform:none !important;
    }}

    /* Recent-chat title buttons must match the exact width of the main
       sidebar navigation buttons. The delete column is only a control
       overlay and must never make the title button narrower. */
    [data-testid="stSidebar"] [class*="st-key-chatbtn-"] .stButton > button {{
        width: calc(100% + 14.5%) !important;
        max-width: none !important;
    }}
    [data-testid="stSidebar"] [class*="st-key-delbtn-"] .stButton > button {{
        position: relative !important;
        z-index: 5 !important;
        width: 28px !important;
        min-width: 28px !important;
        padding: 0 !important;
    }}

    /* ── Top toolbar (native "Deploy" menu bar) ──────────────────
       NOTE: we only override the background here. Forcing `fill`
       on every descendant paints over the menu icon's invisible
       hover-background rect too, turning it into a solid square.
       config.toml's base="light" already colors the icon correctly
       on its own — no CSS override needed for that part. */
    [data-testid="stHeader"] {{ background-color: {t['bg']} !important; }}
    [data-testid="stToolbar"] {{ background-color: transparent !important; }}
    [data-testid="stDecoration"] {{
        background-image: linear-gradient(90deg, {t['primary']}, {t['accent2']}) !important;
    }}

    [data-stale="true"] {{
        opacity: 0.85 !important;
        transition: opacity 0.15s ease !important;
    }}

    /* ── Expanders ── */
    [data-testid="stExpander"] {{
        background-color: {t['surface']};
        border: 1px solid {t['border']} !important;
        border-radius: 10px !important;
    }}
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary div {{
        color: {t['text']} !important;
        opacity: 1 !important;
        font-size: 0.88rem !important;
    }}
    [data-testid="stExpander"] summary:hover {{ color: {t['primary']} !important; }}
    [data-testid="stExpander"] div[data-testid="stExpanderDetails"] p,
    [data-testid="stExpander"] div[data-testid="stExpanderDetails"] span {{
        color: {t['text']} !important;
        opacity: 1 !important;
    }}

    /* ── Generated report status + downloads ── */
    .report-success-card {{
        display:flex; align-items:center; gap:10px;
        padding:11px 14px; margin:4px 0 14px;
        background:linear-gradient(135deg,#ECFDF5,#F0FDFA);
        border:1px solid #B7E4D2; border-radius:11px;
        box-shadow:0 2px 8px rgba(16,185,129,.055);
    }}
    .report-success-icon, .report-ready-icon {{
        width:30px; height:30px; flex:0 0 30px; border-radius:9px;
        display:flex; align-items:center; justify-content:center;
        background:#D1FAE5;
    }}
    .report-success-title, .report-ready-title {{
        color:#047857; font-size:.82rem; font-weight:700; line-height:1.3;
    }}
    .report-success-sub, .report-ready-sub {{
        color:#4B7A6A; font-size:.71rem; margin-top:2px; line-height:1.35;
    }}
    .report-ready-heading {{
        display:flex; align-items:center; gap:10px; margin:4px 0 9px;
    }}
    .report-ready-icon {{ background:#DCFCE7; }}
    .report-ready-title {{ color:{t['text']}; font-size:.9rem; }}
    .report-ready-sub {{ color:{t['muted']}; }}

    .st-key-dl_md_final .stDownloadButton > button,
    .st-key-dl_pdf_final .stDownloadButton > button {{
        min-height:52px !important; border-radius:13px !important;
        border:1px solid #C8D5F2 !important;
        background:linear-gradient(135deg,#FFFFFF 0%,#F7F9FF 55%,#FBF8FF 100%) !important;
        color:#273A70 !important; font-size:.82rem !important;
        font-weight:650 !important;
        box-shadow:0 3px 9px rgba(37,99,235,.06) !important;
        transition:transform .18s ease, box-shadow .18s ease,
                    border-color .18s ease !important;
    }}
    .st-key-dl_md_final .stDownloadButton > button:hover,
    .st-key-dl_pdf_final .stDownloadButton > button:hover {{
        transform:translateY(-2px) !important; border-color:#8199FF !important;
        box-shadow:0 8px 20px rgba(37,99,235,.13),
                   0 0 16px rgba(124,58,237,.10) !important;
    }}
    .st-key-dl_md_final [data-testid="stIconMaterial"],
    .st-key-dl_pdf_final [data-testid="stIconMaterial"] {{
        color:#4F6FF5 !important;
        filter:drop-shadow(0 0 5px rgba(79,111,245,.3)) !important;
    }}
    .st-key-regenerate_report .stButton > button {{
        margin-top:10px !important; min-height:40px !important;
        border-radius:10px !important; font-size:.76rem !important;
    }}

    /* ── Export Report header (gradient icon tile) ── */
    .export-section {{ margin-bottom: 14px; }}
    .export-header {{
        display: flex; align-items: center; gap: 12px;
    }}
    .export-header-icon {{
        width: 36px; height: 36px; border-radius: 8px; flex-shrink: 0;
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        display: flex; align-items: center; justify-content: center;
    }}
    .export-header-title {{
        font-size: 1rem; font-weight: 700; color: #1e1b4b;
        letter-spacing: -0.01em; line-height: 1.2;
    }}
    .export-header-sub {{
        font-size: 0.75rem; color: #6366f1; margin-top: 3px;
    }}

    /* ── Fixed report contents — matches the requested reference ── */
    .report-blueprint {{
        margin: 2px 0 18px;
        padding: 18px 16px 16px;
        background: #FFFFFF;
        border: 1px solid #E1E7F0;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(15,23,42,.025);
    }}
    .report-blueprint-top {{
        display:flex;
        align-items:center;
        justify-content:flex-start;
        gap:0;
        margin-bottom:11px;
    }}
    .report-blueprint-title {{
        font-size:.88rem;
        font-weight:700;
        color:#111827;
        line-height:1.25;
    }}
    .report-blueprint-sub {{
        margin-top:5px;
        font-size:.72rem;
        color:#64748B;
        line-height:1.4;
        max-width:none;
    }}
    .report-standard-badge {{
        display:none;
    }}
    .report-content-grid {{
        display:flex;
        flex-wrap:wrap;
        align-items:center;
        gap:7px;
    }}
    .report-content-item {{
        display:inline-flex;
        align-items:center;
        gap:6px;
        min-height:27px;
        padding:4px 10px;
        background:#FFFFFF;
        border:1px solid #DCE3EC;
        border-radius:999px;
        color:#334155;
        font-size:.68rem;
        font-weight:600;
        box-shadow:none;
        transition:all .16s ease;
    }}
    .report-content-item:hover {{
        transform:translateY(-1px);
        background:#F8FAFF;
        border-color:#B8C6FF;
        box-shadow:0 3px 10px rgba(79,111,245,.09);
    }}
    .report-content-item-icon {{
        width:auto;
        height:auto;
        flex:0 0 auto;
        border:0;
        border-radius:0;
        display:inline-flex;
        align-items:center;
        justify-content:center;
        background:transparent;
        color:#4F6FF5;
        box-shadow:none;
    }}
    .report-content-item:nth-child(2n) .report-content-item-icon,
    .report-content-item:nth-child(3n) .report-content-item-icon {{
        background:transparent;
        color:#4F6FF5;
    }}

    /* Primary report action — clean gradient button with restrained glow. */
    .st-key-generate_report .stButton > button {{
        position:relative !important;
        min-height:40px !important;
        height:40px !important;
        border-radius:8px !important;
        border:0 !important;
        background:linear-gradient(100deg,#2563EB 0%,#4F46E5 52%,#7C3AED 100%) !important;
        color:#FFFFFF !important;
        font-size:.80rem !important;
        font-weight:650 !important;
        box-shadow:0 4px 12px rgba(79,70,229,.16) !important;
        transition:transform .16s ease, box-shadow .16s ease, filter .16s ease !important;
    }}
    .st-key-generate_report .stButton > button:hover {{
        transform:translateY(-1px) !important;
        filter:brightness(1.035) !important;
        box-shadow:
            0 7px 18px rgba(37,99,235,.22),
            0 0 16px rgba(124,58,237,.10) !important;
    }}
    .st-key-generate_report .stButton > button [data-testid="stIconMaterial"] {{
        font-size:16px !important;
        color:#FFFFFF !important;
        filter:drop-shadow(0 0 5px rgba(255,255,255,.30));
    }}

    /* ── Pane headers ── */
    .pane-header {{
        display: flex; align-items: center; gap: 12px;
        padding: 14px 18px; border-radius: 10px;
        margin-bottom: 20px; border: 1px solid;
    }}
    .pane-header-icon {{
        width: 36px; height: 36px; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
    }}
    .pane-header-title {{
        font-size: 1rem; font-weight: 700; letter-spacing: -0.01em;
        margin-bottom: 1px;
    }}
    .pane-header-sub {{
        font-size: 0.78rem; opacity: 0.75;
    }}
    /* Chat pane — blue */
    .pane-header.chat {{
        background: #eff6ff; border-color: #bfdbfe; color: #1e40af;
    }}
    .pane-header.chat .pane-header-icon {{ background: #2563eb; }}
    .pane-header.chat .pane-header-title {{ color: #1e3a8a; }}
    .pane-header.chat .pane-header-sub {{ color: #3b82f6; }}
    /* Preview pane — indigo/violet */
    .pane-header.preview {{
        background: linear-gradient(135deg, #eef2ff 0%, #faf5ff 100%);
        border-color: #c7d2fe; color: #3730a3;
    }}
    .pane-header.preview .pane-header-icon {{
        background: linear-gradient(135deg, #2563eb, #6366f1);
    }}
    .pane-header.preview .pane-header-title {{ color: #1e1b4b; }}
    .pane-header.preview .pane-header-sub {{ color: #6366f1; }}
    /* Profile pane — violet */
    .pane-header.profile {{
        background: #f5f3ff; border-color: #ddd6fe; color: #5b21b6;
    }}
    .pane-header.profile .pane-header-icon {{ background: #7c3aed; }}
    .pane-header.profile .pane-header-title {{ color: #4c1d95; }}
    .pane-header.profile .pane-header-sub {{ color: #7c3aed; }}
    /* AI Analysis pane — indigo/gradient */
    .pane-header.analysis {{
        background: linear-gradient(135deg, #eef2ff 0%, #faf5ff 100%);
        border-color: #c7d2fe; color: #3730a3;
    }}
    .pane-header.analysis .pane-header-icon {{
        background: linear-gradient(135deg, #2563eb, #7c3aed);
    }}
    .pane-header.analysis .pane-header-title {{ color: #1e1b4b; }}
    .pane-header.analysis .pane-header-sub {{ color: #6366f1; }}
    /* Reports pane — blue (document icon) */
    .pane-header.reports {{
        background: #eff6ff; border-color: #bfdbfe; color: #1e40af;
    }}
    .pane-header.reports .pane-header-icon {{ background: #2563eb; }}
    .pane-header.reports .pane-header-title {{ color: #1e3a8a; }}
    .pane-header.reports .pane-header-sub {{ color: #3b82f6; }}
    /* Settings pane — slate (gear icon) */
    .pane-header.settings {{
        background: #f8fafc; border-color: #e2e8f0; color: #334155;
    }}
    .pane-header.settings .pane-header-icon {{ background: #334155; }}
    .pane-header.settings .pane-header-title {{ color: #0f172a; }}
    .pane-header.settings .pane-header-sub {{ color: #64748b; }}

    /* ── dtype badges ── */
    .dtype-badge {{
        display: inline-flex; align-items: center; justify-content: center;
        font-size: 11px; font-weight: 600;
        padding: 3px 9px; border-radius: 999px; margin-left: 8px;
        vertical-align: middle; letter-spacing: 0.02em;
        line-height: 1.2;
    }}
    /* Datatype badges shown directly inside Column Details expander labels. */
    [data-testid="stExpander"] summary code {{
        border-radius: 999px !important;
        padding: 3px 9px !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        border: 0 !important;
        margin-left: 5px !important;
    }}
    [data-testid="stExpander"] summary code[data-badge-type="object"] {{
        background: #f3e8ff !important; color: #6b21a8 !important;
    }}
    [data-testid="stExpander"] summary code[data-badge-type="numeric"] {{
        background: #dbeafe !important; color: #1e40af !important;
    }}
    [data-testid="stExpander"] summary code[data-badge-type="datetime"] {{
        background: #d1fae5 !important; color: #065f46 !important;
    }}
    [data-testid="stExpander"] summary code[data-badge-type="bool"] {{
        background: #fef3c7 !important; color: #92400e !important;
    }}
    .dtype-object   {{ background: #f3e8ff; color: #6b21a8; }}
    .dtype-int      {{ background: #dbeafe; color: #1e40af; }}
    .dtype-float    {{ background: #dbeafe; color: #1e40af; }}
    .dtype-datetime {{ background: #d1fae5; color: #065f46; }}
    .dtype-bool     {{ background: #fef3c7; color: #92400e; }}
    .dtype-other    {{ background: #f1f5f9; color: #475569; }}

    .stSelectbox > div > div,
    [data-baseweb="select"] > div {{
        background-color: {t['surface']} !important;
        border-color: {t['border_strong']} !important;
        color: {t['text']} !important;
    }}
    [data-baseweb="select"]:focus-within > div {{
        border-color: {t['primary']} !important;
        box-shadow: 0 0 0 3px {t['soft_surface']} !important;
    }}
    [data-baseweb="menu"] li:hover, [role="option"]:hover {{ background-color: {t['soft_surface']} !important; }}
    [data-baseweb="select"] div {{ color: {t['text']} !important; }}

    /* Card shell around the home-page uploader — gives it a designed,
       intentional footprint instead of a bare dropzone floating on the page. */
    .upload-card-wrap {{
        max-width: 720px;
        margin: 0 auto;
        padding: 0;
        background: transparent;
        border: 0;
        border-radius: 0;
        box-shadow: none;
    }}
    [data-testid="stFileUploader"] {{ background-color: transparent; border-radius: 10px; padding: 6px; }}
    [data-testid="stFileUploaderDropzone"] {{
        border: 2px dashed {t['border_dashed']} !important;
        background: linear-gradient(180deg, {t['surface_muted']}, {t['soft_surface']}) !important;
        border-radius: 14px !important;
        padding: 14px !important;
        transition: border-color .15s ease, background .15s ease, box-shadow .15s ease !important;
    }}
    [data-testid="stFileUploaderDropzone"]:hover {{
        border-color: {t['primary']} !important;
        background: linear-gradient(180deg, {t['soft_surface']}, {t['soft_surface']}) !important;
        box-shadow: 0 0 0 4px {t['soft_surface']} !important;
    }}
    [data-testid="stFileUploaderDropzoneInstructions"] div,
    [data-testid="stFileUploaderDropzoneInstructions"] span {{
        color: {t['text_secondary']} !important;
        opacity: 1 !important;
    }}
    /* Higher-specificity selector (matches [kind="secondary"] too)
       so this beats the generic secondary-button gray-text rule —
       that specificity conflict was making "Browse files" render
       as washed-out gray text on the blue button. */
       [data-testid="stFileUploader"] button[kind="secondary"],
     [data-testid="stFileUploader"] button {{
    background: linear-gradient(135deg, {t['primary']}, {t['accent2']}) !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 600 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}}

[data-testid="stFileUploader"] button span,
[data-testid="stFileUploader"] button p,
[data-testid="stFileUploader"] button div {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #ffffff !important;
}}

[data-testid="stFileUploader"] button svg {{
    display: block !important;
    width: 18px !important;
    height: 18px !important;
    color: #ffffff !important;
    fill: none !important;
}}

[data-testid="stFileUploader"] button span[data-testid="stIconMaterial"] {{
    font-family: 'Material Symbols Rounded' !important;
}}
    
    [data-testid="stFileUploader"] button:hover {{ filter: brightness(1.05) !important; }}
    [data-testid="stFileUploaderFile"] * {{ color: {t['text']} !important; opacity: 1 !important; }}

    [data-testid="stAlert"] {{ border-radius: 8px; }}
    [data-testid="stAlert"] p, [data-testid="stAlert"] span, [data-testid="stAlert"] div {{ opacity: 1 !important; }}

    input, textarea {{ color: {t['text']} !important; }}
    input:focus, textarea:focus,
    [data-baseweb="input"]:focus-within,
    [data-baseweb="base-input"]:focus-within,
    [data-baseweb="textarea"]:focus-within {{
        border-color: {t['primary']} !important;
        box-shadow: 0 0 0 3px {t['soft_surface']} !important;
    }}
    input::placeholder, textarea::placeholder {{ color: {t['muted']} !important; opacity: 0.9 !important; }}
    [data-testid="stTextInput"] > div {{
        background-color: {t['surface']} !important;
        border: 1px solid {t['border_strong']} !important;
        border-radius: 8px !important;
    }}
    [data-testid="stTextInput"] input {{ background-color: {t['surface']} !important; color: {t['text']} !important; }}
    [data-testid="stNumberInput"] > div {{ background-color: {t['surface']} !important; border: 1px solid {t['border_strong']} !important; }}

    a, a:visited {{ color: {t['primary']} !important; }}
    a:hover {{ color: {t['primary_hover']} !important; }}

    .stSpinner > div {{ border-top-color: {t['primary']} !important; }}

    [data-baseweb="slider"] [role="slider"] {{ background-color: {t['primary']} !important; border-color: {t['primary']} !important; }}
    /* ── Report option cards ── */
    .report-options-heading {{
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:16px;
        margin:2px 0 3px;
    }}
    .report-options-title {{
        font-size:.98rem;
        font-weight:700;
        color:{t['text']};
    }}
    .report-options-hint {{
        font-size:.72rem;
        color:{t['muted']};
    }}
    .report-options-caption {{
        color:{t['text_secondary']};
        font-size:.77rem;
        margin:0 0 14px;
        line-height:1.45;
    }}

    .st-key-report_include_preview_card .stButton > button,
    .st-key-report_include_eda_card .stButton > button,
    .st-key-report_include_charts_card .stButton > button,
    .st-key-report_include_chat_card .stButton > button,
    .st-key-report_make_pdf_card .stButton > button,
    .st-key-report_make_markdown_card .stButton > button {{
        min-height:58px !important;
        border-radius:14px !important;
        padding:10px 15px !important;
        justify-content:flex-start !important;
        text-align:left !important;
        font-size:.83rem !important;
        font-weight:650 !important;
        border:1px solid #DCE3F0 !important;
        box-shadow:0 2px 8px rgba(15,23,42,.035) !important;
        transition:transform .18s ease, box-shadow .18s ease,
                    border-color .18s ease, background .18s ease !important;
    }}

    .st-key-report_include_preview_card .stButton > button:hover,
    .st-key-report_include_eda_card .stButton > button:hover,
    .st-key-report_include_charts_card .stButton > button:hover,
    .st-key-report_include_chat_card .stButton > button:hover,
    .st-key-report_make_pdf_card .stButton > button:hover,
    .st-key-report_make_markdown_card .stButton > button:hover {{
        transform:translateY(-2px) !important;
        border-color:#8EA7FF !important;
        box-shadow:
            0 7px 18px rgba(37,99,235,.11),
            0 0 15px rgba(124,58,237,.07) !important;
    }}

    /* Selected cards: subtle brand gradient + glow, but keep text readable. */
    .st-key-report_include_preview_card .stButton > button[kind="primary"],
    .st-key-report_include_eda_card .stButton > button[kind="primary"],
    .st-key-report_include_charts_card .stButton > button[kind="primary"],
    .st-key-report_include_chat_card .stButton > button[kind="primary"],
    .st-key-report_make_pdf_card .stButton > button[kind="primary"],
    .st-key-report_make_markdown_card .stButton > button[kind="primary"] {{
        background:linear-gradient(135deg,#EEF4FF 0%,#F5F0FF 100%) !important;
        color:#263E78 !important;
        border:1.5px solid #829BFF !important;
        box-shadow:
            0 4px 14px rgba(37,99,235,.10),
            0 0 0 2px rgba(99,102,241,.06),
            0 0 18px rgba(124,58,237,.07) !important;
    }}

    .st-key-report_include_preview_card .stButton > button[kind="primary"] [data-testid="stIconMaterial"],
    .st-key-report_include_eda_card .stButton > button[kind="primary"] [data-testid="stIconMaterial"],
    .st-key-report_include_charts_card .stButton > button[kind="primary"] [data-testid="stIconMaterial"],
    .st-key-report_include_chat_card .stButton > button[kind="primary"] [data-testid="stIconMaterial"],
    .st-key-report_make_pdf_card .stButton > button[kind="primary"] [data-testid="stIconMaterial"],
    .st-key-report_make_markdown_card .stButton > button[kind="primary"] [data-testid="stIconMaterial"] {{
        color:#4F6FF5 !important;
        filter:drop-shadow(0 0 5px rgba(79,111,245,.28)) !important;
    }}

    .report-control-label {{
        display:flex;
        align-items:center;
        justify-content:space-between;
        margin:16px 0 7px;
        font-size:.78rem;
        font-weight:650;
        color:{t['text']};
    }}
    .report-control-label span:last-child {{
        font-size:.69rem;
        font-weight:500;
        color:{t['muted']};
    }}

    /* Streamlit keeps a hidden label wrapper even with collapsed labels. */
    .report-control-label + div[data-testid="stSelectbox"] {{
        margin-top:0 !important;
    }}

    ::selection {{ background-color: {t['primary']}33; color: {t['text']}; }}
    *:focus-visible {{ outline-color: {t['primary']} !important; }}
    button, [role="button"], [role="tab"], summary, .stSelectbox {{ cursor: pointer !important; }}

    h1, h2, h3, h4, h5, h6 {{ color: {t['text']}; font-weight: 600; letter-spacing: -0.01em; }}

    code {{
        background-color: {t['soft_surface']} !important;
        color: {t['secondary']} !important;
        border: 1px solid {t['border']} !important;
        border-radius: 4px !important;
        padding: 2px 6px !important;
        font-size: 0.85em !important;
    }}
    pre code {{ background-color: transparent !important; color: inherit !important; border: none !important; padding: 0 !important; }}

    /* ── SUGGESTED QUESTIONS — SHARED GLOWING ACTION STYLE ─────────
       The Chat starter questions and Reports suggested questions use
       the SAME component appearance:
       left aligned text + leading icon circle + trailing arrow +
       blue/violet glow + soft shadow.
       This block intentionally does not style cards, normal buttons,
       or the Analysis tab pane. */
    .st-key-suggestion_0 .stButton > button,
    .st-key-suggestion_1 .stButton > button,
    .st-key-suggestion_2 .stButton > button,
    .st-key-suggestion_3 .stButton > button,
    .st-key-suggestion_4 .stButton > button,
    .st-key-suggestion_5 .stButton > button,
    .st-key-starter_0 .stButton > button,
    .st-key-starter_1 .stButton > button,
    .st-key-starter_2 .stButton > button {{
        position:relative !important;
        width:100% !important;
        min-height:54px !important;
        height:54px !important;
        padding:8px 52px 8px 14px !important;

        background:linear-gradient(135deg,#FFFFFF 0%,#FAFBFF 52%,#FBF8FF 100%) !important;
        color:#263E78 !important;
        border:1px solid #B9C8F5 !important;
        border-radius:11px !important;

        font-size:.84rem !important;
        font-weight:600 !important;
        text-align:left !important;
        justify-content:flex-start !important;
        box-shadow:
            0 4px 16px rgba(37,99,235,.13),
            0 0 17px rgba(124,58,237,.09) !important;

        transition:
            transform .18s ease,
            box-shadow .18s ease,
            border-color .18s ease,
            background .18s ease !important;
    }}

    /* Keep the question text visually left-aligned. */
    .st-key-suggestion_0 .stButton > button p,
    .st-key-suggestion_1 .stButton > button p,
    .st-key-suggestion_2 .stButton > button p,
    .st-key-suggestion_3 .stButton > button p,
    .st-key-suggestion_4 .stButton > button p,
    .st-key-suggestion_5 .stButton > button p,
    .st-key-starter_0 .stButton > button p,
    .st-key-starter_1 .stButton > button p,
    .st-key-starter_2 .stButton > button p {{
        text-align:left !important;
    }}

   
    /* Trailing arrow circle. */
    .st-key-suggestion_0 .stButton > button::after,
    .st-key-suggestion_1 .stButton > button::after,
    .st-key-suggestion_2 .stButton > button::after,
    .st-key-suggestion_3 .stButton > button::after,
    .st-key-suggestion_4 .stButton > button::after,
    .st-key-suggestion_5 .stButton > button::after,
    .st-key-starter_0 .stButton > button::after,
    .st-key-starter_1 .stButton > button::after,
    .st-key-starter_2 .stButton > button::after {{
        content:"→";
        position:absolute !important;
        right:12px !important;
        top:50% !important;
        transform:translateY(-50%) !important;

        width:30px !important;
        height:30px !important;
        border-radius:50% !important;
        display:flex !important;
        align-items:center !important;
        justify-content:center !important;

        background:linear-gradient(135deg,#F0F4FF 0%,#F1E9FF 100%) !important;
        color:#5B5CF0 !important;
        font-size:15px !important;
        font-weight:600 !important;
        box-shadow:0 0 10px rgba(124,58,237,.10) !important;
    }}

    /* Same glowing hover treatment on both pages. */
    .st-key-suggestion_0 .stButton > button:hover,
    .st-key-suggestion_1 .stButton > button:hover,
    .st-key-suggestion_2 .stButton > button:hover,
    .st-key-suggestion_3 .stButton > button:hover,
    .st-key-suggestion_4 .stButton > button:hover,
    .st-key-suggestion_5 .stButton > button:hover,
    .st-key-starter_0 .stButton > button:hover,
    .st-key-starter_1 .stButton > button:hover,
    .st-key-starter_2 .stButton > button:hover {{
        transform:translateY(-2px) !important;
        background:linear-gradient(135deg,#EEF4FF 0%,#F5F0FF 100%) !important;
        border-color:#829BFF !important;
        color:#1E3A8A !important;
        box-shadow:
            0 8px 22px rgba(37,99,235,.18),
            0 0 22px rgba(124,58,237,.13) !important;
    }}

    .st-key-suggestion_0 .stButton > button:hover [data-testid="stIconMaterial"],
    .st-key-suggestion_1 .stButton > button:hover [data-testid="stIconMaterial"],
    .st-key-suggestion_2 .stButton > button:hover [data-testid="stIconMaterial"],
    .st-key-suggestion_3 .stButton > button:hover [data-testid="stIconMaterial"],
    .st-key-suggestion_4 .stButton > button:hover [data-testid="stIconMaterial"],
    .st-key-suggestion_5 .stButton > button:hover [data-testid="stIconMaterial"],
    .st-key-starter_0 .stButton > button:hover [data-testid="stIconMaterial"],
    .st-key-starter_1 .stButton > button:hover [data-testid="stIconMaterial"],
    .st-key-starter_2 .stButton > button:hover [data-testid="stIconMaterial"] {{
        transform:scale(1.05) !important;
        filter:drop-shadow(0 0 7px rgba(79,111,245,.34)) !important;
    }}

    /* ── AI Analysis insight cards ── */
    .insight-card {{
        position:relative;
        overflow:hidden;
        background:{t['surface']};
        border:1px solid {t['border']};
        border-left:4px solid var(--insight-accent-color, {t['primary']});
        border-radius:14px;
        padding:16px 18px 16px 15px;
        margin-bottom:12px;
        display:flex;
        gap:14px;
        align-items:flex-start;
        box-shadow:0 3px 10px rgba(15,23,42,.055);
        transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease;
    }}
    .insight-card:hover {{
        transform:translateY(-2px);
        border-color:var(--insight-accent-color, {t['primary']});
        box-shadow:0 9px 24px rgba(15,23,42,.10);
    }}
    .insight-icon-box {{
        width:38px;
        height:38px;
        border-radius:11px;
        flex:0 0 38px;
        background:var(--insight-accent, {t['soft_surface']}) !important;
        display:flex;
        align-items:center;
        justify-content:center;
        box-shadow:0 2px 6px rgba(15,23,42,.06);
    }}
    .insight-card .insight-title {{
        font-weight:700;
        font-size:.93rem;
        color:{t['text']};
        margin:1px 0 5px;
        line-height:1.35;
    }}
    .insight-card .insight-body {{
        color:{t['text_secondary']};
        font-size:.84rem;
        line-height:1.6;
    }}

    /* ── AI Analysis metric cards ── */
    [data-testid="stMetric"] {{
        position:relative !important;
        overflow:hidden !important;
        background:linear-gradient(145deg,#FFFFFF 0%,#F8FAFC 100%) !important;
        border:1px solid {t['border']} !important;
        border-radius:14px !important;
        padding:16px 18px 17px !important;
        box-shadow:0 3px 10px rgba(15,23,42,.05) !important;
        transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease !important;
    }}
    [data-testid="stMetric"]::before {{
        content:"";
        position:absolute;
        left:0;
        top:0;
        bottom:0;
        width:4px;
        background:linear-gradient(180deg,{t['primary']},{t['accent2']});
    }}
    [data-testid="stMetric"]:hover {{
        transform:translateY(-2px);
        border-color:{t['primary']} !important;
        box-shadow:0 9px 22px rgba(37,99,235,.10) !important;
    }}
    [data-testid="stMetricValue"] {{
        color:{t['text']} !important;
        font-size:1.65rem !important;
        font-weight:700 !important;
    }}
    [data-testid="stMetricLabel"] p {{
        color:{t['text_secondary']} !important;
        opacity:1 !important;
        font-size:.8rem !important;
        font-weight:600 !important;
    }}

    /* ── Loaded Home dashboard cards — match the current cover language ── */
    .feature-card {{
        position: relative;
        overflow: hidden;
        background: linear-gradient(145deg,#FFFFFF 0%,#FBFCFF 100%);
        border: 1px solid #E2E8F4;
        border-radius: 15px;
        padding: 19px 18px 18px;
        height: 190px;
        min-height: 190px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        transition: border-color .18s ease, box-shadow .18s ease, transform .18s ease;
        box-shadow: 0 3px 12px rgba(37,99,235,.045);
    }}
    .feature-card::before {{
        content:"";
        position:absolute;
        left:0; right:0; top:0;
        height:3px;
        background:linear-gradient(90deg,#2563EB,#7C3AED);
        opacity:.85;
    }}
    .feature-card > span {{
        display: block;
        flex: 1;
        overflow: hidden;
        color: {t['text_secondary']} !important;
    }}
    .feature-card:hover {{
        border-color:#C9D6F4;
        box-shadow:0 10px 26px rgba(37,99,235,.10);
        transform:translateY(-2px);
    }}
    .feature-card .insight-icon-box {{
        width:38px !important;
        height:38px !important;
        border-radius:11px !important;
        background:linear-gradient(135deg,#EEF4FF,#F4EEFF) !important;
        border:1px solid #E0E7FF;
        box-shadow:none !important;
    }}
    .home-section-title {{
        font-size:1.65rem;
        line-height:1.15;
        font-weight:750;
        letter-spacing:-.035em;
        color:{t['text']};
        margin:8px 0 4px;
    }}
    .home-section-sub {{
        color:{t['muted']};
        font-size:.86rem;
        margin-bottom:18px;
    }}
    .home-card-title {{
        font-size:.98rem;
        font-weight:700;
        color:{t['text']};
        letter-spacing:-.01em;
    }}
    .home-card-desc {{
        display:block;
        color:{t['text_secondary']} !important;
        font-size:.82rem;
        line-height:1.55;
        margin-top:2px;
    }}
    /* Action buttons directly below each card form one connected card group. */
    .stButton > button[data-testid*="quicklink"] {{
        min-height:40px !important;
        border-radius:0 0 12px 12px !important;
        border:1px solid #E2E8F4 !important;
        border-top:0 !important;
        background:#FFFFFF !important;
        color:#52617A !important;
        font-size:.78rem !important;
        font-weight:600 !important;
        box-shadow:0 3px 12px rgba(37,99,235,.035) !important;
        margin-top:-1px !important;
    }}
    .stButton > button[data-testid*="quicklink"]:hover {{
        color:#2563EB !important;
        background:#F8FAFF !important;
        border-color:#C9D6F4 !important;
        box-shadow:0 7px 18px rgba(37,99,235,.08) !important;
    }}
    /* Landing-page card row: stretch all 4 columns to equal height so
       shorter cards don't look mismatched against longer ones. */
    div[data-testid="stHorizontalBlock"]:has(.feature-card) {{
        align-items: stretch !important;
    }}
    div[data-testid="stHorizontalBlock"]:has(.feature-card) > div {{
        display: flex !important;
    }}
    div[data-testid="stHorizontalBlock"]:has(.feature-card) > div > div {{
        display: flex !important;
        width: 100% !important;
    }}

    /* ── Loaded-app content offset ─────────────────────────────────
       Streamlit's native top header is fixed over the main content.
       The first dataset topbar was therefore partially hidden behind
       the Deploy/menu chrome. Keep the cover untouched, but push every
       loaded-data view down far enough for the topbar to be completely
       visible. */
    .stApp:has(.topbar) .block-container {{
        padding-top: 4.5rem !important;
    }}

    /* ── Top bar (dataset identity + status pill) — card style ── */
    .topbar {{
        display: flex; align-items: center; justify-content: space-between;
        padding: 10px 18px; margin-bottom: 16px;
        background: {t['surface']};
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        font-size: 0.855rem;
    }}
    .status-pill {{
        display: inline-flex; align-items: center; gap: 6px;
        font-size: 0.78rem; font-weight: 600; color: {t['success']};
        background: {t['success_bg']};
        padding: 4px 12px;
        border-radius: 100px;
        border: 1px solid #A7F3D0;
        letter-spacing: 0.01em;
    }}
    .status-dot {{ width: 6px; height: 6px; border-radius: 50%; background: {t['success']}; }}

    /* ── Sidebar brand lockup ── */
    .brand-tagline {{
        font-size: 11px; color: #64748b; line-height: 1.2; margin-top: 1px;
    }}
    .sidebar-section-label {{
        font-size: 11px; font-weight: 600; letter-spacing: 0.05em;
        text-transform: uppercase; color: #475569; margin: 4px 0 6px;
    }}
    </style>
    """, unsafe_allow_html=True)
