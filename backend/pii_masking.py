# backend/pii_masking.py
"""
Minimal, pattern-based PII masking for text sent to the LLM provider (Groq).

Scope and honesty about limits, up front: this is regex-based
defense-in-depth, not a comprehensive PII detection system. It catches
common, well-structured PII formats — it will NOT catch every PII
format (e.g. free-text names, unusual phone formats, non-US/non-Indian
ID formats), and it can occasionally false-positive on structured
non-PII data that happens to match a pattern (e.g. a delimited numeric
ID that looks like a phone number). That tradeoff is intentional: a
strict pattern set with some false negatives is safer to ship than a
loose one that mangles legitimate data or gives false confidence.

Applied once, at call_llm()'s single chokepoint (the same place the
rate limiter already lives), so it covers every caller — answer_question,
generate_insights, or anything added later — without each one needing
to know masking exists.
"""

import re

# Order matters: more specific / less ambiguous patterns first, so a
# looser pattern later doesn't "steal" a match a stricter one should own.
_PATTERNS = [
    # Email addresses
    (re.compile(r'[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}'), '[REDACTED_EMAIL]'),

    # Credit/debit card numbers — 4 groups of 4 digits, space or dash
    # separated. Deliberately does NOT match 16 bare contiguous digits
    # (too easy to collide with other numeric IDs); requires the
    # human-readable grouping most card numbers are actually written in.
    (re.compile(r'\b\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{4}\b'), '[REDACTED_CARD]'),

    # US Social Security Numbers
    (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), '[REDACTED_SSN]'),

    # Indian Aadhaar numbers — 12 digits, conventionally grouped 4-4-4
    (re.compile(r'\b\d{4}\s\d{4}\s\d{4}\b'), '[REDACTED_AADHAAR]'),

    # Indian PAN (tax ID) — 5 letters, 4 digits, 1 letter, e.g. ABCDE1234F
    (re.compile(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b'), '[REDACTED_PAN]'),

    # Phone numbers — common delimited formats. Conservative on purpose:
    # requires a separator (dash, dot, or space), so it will NOT catch a
    # bare 10-digit numeric ID with no delimiters at all.
    (re.compile(r'\(\d{3}\)\s?\d{3}[-.\s]\d{4}'), '[REDACTED_PHONE]'),
    (re.compile(r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'), '[REDACTED_PHONE]'),
    # International, with a leading '+': covers both 3+3+4-style groupings
    # (e.g. +1-415-555-0132) and 5+5-style groupings common in India
    # (e.g. +91-98765-43210), plus a bare '+<countrycode><number>' run.
    (re.compile(r'\+\d{1,3}[-.\s]?\d{4,5}[-.\s]?\d{4,5}\b'), '[REDACTED_PHONE]'),
    (re.compile(r'\+\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{0,4}\b'), '[REDACTED_PHONE]'),
]


def mask_pii(text: str) -> tuple:
    """
    Scrubs common PII patterns from a single string.
    Returns (masked_text, redaction_count). The count is for logging
    only — logging *how many* redactions happened, never the actual
    values that got redacted, so debugging doesn't reintroduce the leak
    this module exists to prevent.
    """
    if not text:
        return text, 0

    count = 0
    for pattern, placeholder in _PATTERNS:
        text, n = pattern.subn(placeholder, text)
        count += n

    return text, count


def mask_messages(messages: list) -> tuple:
    """
    Applies mask_pii() to every message's 'content' field in an LLM
    `messages` list (the format call_llm() already takes).
    Returns (masked_messages, total_redaction_count). Never mutates the
    input list or dicts in place — returns new ones.
    """
    total = 0
    masked = []
    for msg in messages:
        content = msg.get("content", "") or ""
        masked_content, n = mask_pii(content)
        total += n
        new_msg = dict(msg)
        new_msg["content"] = masked_content
        masked.append(new_msg)
    return masked, total
