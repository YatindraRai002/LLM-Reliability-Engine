"""
Centralized input sanitization module.
All user-facing inputs should pass through sanitize_prompt() before
being forwarded to any model or API.
"""
import re
import logging

logger = logging.getLogger(__name__)

MAX_PROMPT_LENGTH = 10_000

_INJECTION_PATTERNS = [
    re.compile(r"(ignore|disregard|forget)\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)", re.I),
    re.compile(r"you\s+are\s+now\s+(DAN|in\s+developer\s+mode)", re.I),
    re.compile(r"<\s*script\b", re.I),
    re.compile(r"javascript\s*:", re.I),
]

_EXCESSIVE_REPEAT = re.compile(r"([^\w\s])\1{9,}")


def sanitize_prompt(raw: str) -> str:
    """
    Clean and validate a user prompt.

    1. Truncate to MAX_PROMPT_LENGTH characters.
    2. Strip leading/trailing whitespace.
    3. Collapse excessive repeated special characters.
    4. Flag (log) any known injection patterns but do NOT block—transparency
       over silent rejection.

    Returns the sanitised string.
    """
    if not isinstance(raw, str):
        logger.warning("Non-string prompt received, coercing to empty string")
        return ""

    text = raw[:MAX_PROMPT_LENGTH].strip()

    text = _EXCESSIVE_REPEAT.sub(r"\1\1\1", text)

    for pat in _INJECTION_PATTERNS:
        if pat.search(text):
            logger.warning("Potential prompt-injection pattern detected in user input")
            break

    return text
