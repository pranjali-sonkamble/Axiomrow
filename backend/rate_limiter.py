# backend/rate_limiter.py
"""
Per-session rate limiting for LLM (Groq/OpenAI) API calls.

WHY THIS EXISTS: every LLM call costs real money against a real API
key. Once this app is deployed with a public link, anyone can hit it —
without a limit, a single person (accidentally, via rapid clicking, or
deliberately) could run up unexpected API costs. The deterministic
query engine is unaffected: this only guards the paths that actually
reach the network.

DESIGN: kept deliberately framework-agnostic (no `import streamlit`)
so it can be unit-tested directly with plain function calls, the same
way llm_agent.py's execute_code()/execute_sql() are. Session storage
is a plain in-memory dict keyed by a caller-supplied session_id —
app.py owns generating and passing that id (see the top of app.py).

Two layers, both per-session:
  - MIN_SECONDS_BETWEEN_CALLS: stops literal rapid-fire/double-click
    spam — a single click can't be throttled by a sliding window alone
    since the window only fills up after several calls have already
    landed.
  - MAX_CALLS_PER_WINDOW / WINDOW_SECONDS: stops slower, sustained
    abuse (e.g. a script firing one request every 3 seconds, which a
    2-second minimum gap alone wouldn't catch).

These numbers are deliberately generous — sized so a recruiter or
reviewer actually using the demo never notices them, while still
bounding worst-case cost from misuse.
"""

import time

MIN_SECONDS_BETWEEN_CALLS = 2.0
MAX_CALLS_PER_WINDOW = 20
WINDOW_SECONDS = 600  # 10 minutes

# Process-wide, in-memory store: {session_id: [timestamp, timestamp, ...]}.
# Resets on app restart — acceptable here since the cost being guarded
# against (a runaway API bill) resets the same way a restart would
# reset any other in-memory session state this app already relies on.
_session_calls: dict = {}

# Process-wide backstop. The per-session limits above are keyed by a session
# id that a browser refresh regenerates, so on their own they bound a
# polite user, not an abuser. This caps TOTAL provider spend per hour for
# the whole deployment, whoever is asking. Override via environment.
import os as _os
GLOBAL_MAX_CALLS_PER_HOUR = int(_os.getenv("AXIOMROW_GLOBAL_LLM_CALLS_PER_HOUR", "600"))
_GLOBAL_WINDOW = 3600
_global_calls: list = []
_MAX_TRACKED_SESSIONS = 5000


def check_and_register(session_id: str, now: float = None) -> tuple:
    """
    Checks whether a new LLM call is allowed for this session, and — if
    allowed — registers it immediately (so a caller can't check twice
    and slip two calls through between check and use).

    Returns (allowed: bool, reason: str | None).
    """
    if now is None:
        now = time.time()

    # "<id>#followup": a retry / summary call for a question whose first call
    # was already admitted. Counted, but exempt from the minimum-gap rule.
    session_id, _, tag = str(session_id).partition("#")
    is_followup = bool(tag)

    # Global backstop + bounded memory.
    global _global_calls
    _global_calls = [ts for ts in _global_calls if now - ts < _GLOBAL_WINDOW]
    if len(_global_calls) >= GLOBAL_MAX_CALLS_PER_HOUR:
        return False, ("The service is handling a lot of requests right now. "
                       "Please try again in a few minutes.")
    if len(_session_calls) > _MAX_TRACKED_SESSIONS:
        for sid in [k for k, v in _session_calls.items()
                    if not v or now - v[-1] >= WINDOW_SECONDS]:
            _session_calls.pop(sid, None)

    timestamps = _session_calls.get(session_id, [])
    # Prune anything outside the sliding window on every check, so this
    # dict doesn't grow unbounded over a long session.
    timestamps = [ts for ts in timestamps if now - ts < WINDOW_SECONDS]

    if not is_followup and timestamps and (now - timestamps[-1]) < MIN_SECONDS_BETWEEN_CALLS:
        wait = MIN_SECONDS_BETWEEN_CALLS - (now - timestamps[-1])
        _session_calls[session_id] = timestamps
        return False, f"Please wait {wait:.1f}s before asking another question."

    if len(timestamps) >= MAX_CALLS_PER_WINDOW:
        oldest = timestamps[0]
        wait_seconds = WINDOW_SECONDS - (now - oldest)
        wait_minutes = max(1, int(wait_seconds // 60) + 1)
        _session_calls[session_id] = timestamps
        return False, (
            f"You've reached the limit of {MAX_CALLS_PER_WINDOW} AI requests per "
            f"{WINDOW_SECONDS // 60} minutes. Try again in about {wait_minutes} "
            "minute(s) — or ask a specific question (totals, counts, top/bottom "
            "N, filters) that the deterministic query engine can answer "
            "instantly without using the AI at all."
        )

    timestamps.append(now)
    _global_calls.append(now)
    _session_calls[session_id] = timestamps
    return True, None


def get_usage(session_id: str, now: float = None) -> tuple:
    """
    Read-only — reports current usage for display purposes without
    registering a call or mutating state. Returns (calls_used,
    calls_remaining) for the current sliding window.
    """
    if now is None:
        now = time.time()
    timestamps = [ts for ts in _session_calls.get(session_id, []) if now - ts < WINDOW_SECONDS]
    used = len(timestamps)
    return used, max(0, MAX_CALLS_PER_WINDOW - used)