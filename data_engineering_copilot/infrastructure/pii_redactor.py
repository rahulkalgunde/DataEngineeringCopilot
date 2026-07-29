"""PII detection and redaction for user queries and LLM outputs."""

from __future__ import annotations

import re
from enum import Enum


class RedactionMode(Enum):
    FULL = "full"
    MASKED = "masked"
    NONE = "none"


_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_PHONE_RE = re.compile(r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CREDIT_CARD_RE = re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b")
_IP_RE = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")

_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": _EMAIL_RE,
    "credit_card": _CREDIT_CARD_RE,
    "ssn": _SSN_RE,
    "phone": _PHONE_RE,
    "ip_address": _IP_RE,
}


def _mask_email(match: re.Match[str]) -> str:
    original = match.group()
    local, _, domain = original.partition("@")
    masked_local = "*" if len(local) <= 1 else local[0] + "***"
    domain_parts = domain.split(".")
    masked_domain = "***." + domain_parts[-1] if len(domain_parts) >= 2 else "***"
    return f"{masked_local}@{masked_domain}"


def _mask_phone(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group())
    if len(digits) >= 7:
        return f"***-***-{digits[-4:]}"
    return "***-***-****"


def _mask_generic(match: re.Match[str]) -> str:
    original = match.group()
    if len(original) <= 4:
        return "*" * len(original)
    return original[:2] + "*" * (len(original) - 4) + original[-2:]


_MASKERS: dict[str, re.Pattern[str]] = {
    "email": _EMAIL_RE,
    "phone": _PHONE_RE,
    "ssn": _SSN_RE,
    "credit_card": _CREDIT_CARD_RE,
    "ip_address": _IP_RE,
}

_MASK_FUNCTIONS: dict[str, callable] = {
    "email": _mask_email,
    "phone": _mask_phone,
    "ssn": _mask_generic,
    "credit_card": _mask_generic,
    "ip_address": _mask_generic,
}


class PiiRedactor:
    """Regex-based PII detector and redactor.

    Detects emails, phone numbers, SSNs, credit card numbers, and IP addresses.
    Supports full redaction (``[REDACTED_TYPE]``) or partial masking.
    """

    def __init__(self, mode: RedactionMode = RedactionMode.FULL) -> None:
        self._mode = mode

    def detect(self, text: str) -> list[dict[str, str | int]]:
        """Return list of detected PII spans.

        Each dict contains ``type``, ``start``, ``end``, and ``original``.
        """
        findings: list[dict[str, str | int]] = []
        for pii_type, pattern in _PATTERNS.items():
            for match in pattern.finditer(text):
                findings.append(
                    {
                        "type": pii_type,
                        "start": match.start(),
                        "end": match.end(),
                        "original": match.group(),
                    }
                )
        findings.sort(key=lambda f: f["start"])
        return findings

    def redact(self, text: str) -> tuple[str, list[str]]:
        """Redact PII from text.

        Returns ``(redacted_text, list_of_pii_types_found)``.
        """
        if self._mode == RedactionMode.NONE:
            return text, []

        types_found: list[str] = []

        if self._mode == RedactionMode.FULL:
            result = text
            for pii_type, pattern in _PATTERNS.items():
                matches = list(pattern.finditer(result))
                if matches:
                    types_found.append(pii_type)
                for match in reversed(matches):
                    replacement = f"[REDACTED_{pii_type.upper()}]"
                    result = result[: match.start()] + replacement + result[match.end() :]
            return result, types_found

        # MASKED mode
        result = text
        for pii_type, pattern in _MASKERS.items():
            mask_fn = _MASK_FUNCTIONS[pii_type]
            matches = list(pattern.finditer(result))
            if matches:
                types_found.append(pii_type)
            for match in reversed(matches):
                masked = mask_fn(match)
                result = result[: match.start()] + masked + result[match.end() :]
        return result, types_found
