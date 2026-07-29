"""Unit tests for PII redaction layer."""

from __future__ import annotations

from data_engineering_copilot.infrastructure.pii_redactor import PiiRedactor, RedactionMode


class TestPiiDetection:
    def test_detects_email(self) -> None:
        redactor = PiiRedactor()
        findings = redactor.detect("Contact me at alice@example.com for details")
        assert len(findings) == 1
        assert findings[0]["type"] == "email"
        assert findings[0]["original"] == "alice@example.com"

    def test_detects_phone_number(self) -> None:
        redactor = PiiRedactor()
        findings = redactor.detect("Call me at (555) 123-4567")
        assert any(f["type"] == "phone" for f in findings)

    def test_detects_ssn(self) -> None:
        redactor = PiiRedactor()
        findings = redactor.detect("SSN: 123-45-6789")
        assert len(findings) == 1
        assert findings[0]["type"] == "ssn"

    def test_detects_credit_card(self) -> None:
        redactor = PiiRedactor()
        findings = redactor.detect("Card: 4111-1111-1111-1111")
        assert len(findings) == 1
        assert findings[0]["type"] == "credit_card"

    def test_detects_ip_address(self) -> None:
        redactor = PiiRedactor()
        findings = redactor.detect("Server at 192.168.1.100 is down")
        assert len(findings) == 1
        assert findings[0]["type"] == "ip_address"

    def test_detects_multiple_pii_types(self) -> None:
        redactor = PiiRedactor()
        text = "Email alice@test.com, SSN 123-45-6789, card 4111111111111111"
        findings = redactor.detect(text)
        types = {f["type"] for f in findings}
        assert "email" in types
        assert "ssn" in types
        assert "credit_card" in types

    def test_no_pii_returns_empty(self) -> None:
        redactor = PiiRedactor()
        findings = redactor.detect("How does Delta Lake time travel work?")
        assert findings == []


class TestFullRedaction:
    def test_redacts_email(self) -> None:
        redactor = PiiRedactor(mode=RedactionMode.FULL)
        result, types = redactor.redact("Email alice@example.com for info")
        assert "[REDACTED_EMAIL]" in result
        assert "alice@example.com" not in result
        assert "email" in types

    def test_redacts_ssn(self) -> None:
        redactor = PiiRedactor(mode=RedactionMode.FULL)
        result, types = redactor.redact("SSN: 123-45-6789")
        assert "[REDACTED_SSN]" in result
        assert "123-45-6789" not in result
        assert "ssn" in types

    def test_redacts_multiple(self) -> None:
        redactor = PiiRedactor(mode=RedactionMode.FULL)
        result, types = redactor.redact("Email a@b.com SSN 123-45-6789")
        assert "[REDACTED_EMAIL]" in result
        assert "[REDACTED_SSN]" in result
        assert len(types) == 2

    def test_preserves_clean_text(self) -> None:
        redactor = PiiRedactor(mode=RedactionMode.FULL)
        result, types = redactor.redact("How does Spark Structured Streaming work?")
        assert result == "How does Spark Structured Streaming work?"
        assert types == []


class TestMaskedRedaction:
    def test_masks_email(self) -> None:
        redactor = PiiRedactor(mode=RedactionMode.MASKED)
        result, types = redactor.redact("Email alice@example.com")
        assert "alice@example.com" not in result
        assert "***" in result
        assert "@example.com" not in result or "***" in result
        assert "email" in types

    def test_masks_phone(self) -> None:
        redactor = PiiRedactor(mode=RedactionMode.MASKED)
        result, types = redactor.redact("Call (555) 123-4567")
        assert "(555) 123-4567" not in result
        assert "4567" in result
        assert "phone" in types

    def test_masks_credit_card(self) -> None:
        redactor = PiiRedactor(mode=RedactionMode.MASKED)
        result, types = redactor.redact("Card 4111-1111-1111-1111")
        assert "4111-1111-1111-1111" not in result
        assert "11" in result  # first two digits kept
        assert "credit_card" in types


class TestNoRedaction:
    def test_none_mode_passthrough(self) -> None:
        redactor = PiiRedactor(mode=RedactionMode.NONE)
        result, types = redactor.redact("Email alice@test.com SSN 123-45-6789")
        assert result == "Email alice@test.com SSN 123-45-6789"
        assert types == []


class TestEdgeCases:
    def test_empty_string(self) -> None:
        redactor = PiiRedactor(mode=RedactionMode.FULL)
        result, types = redactor.redact("")
        assert result == ""
        assert types == []

    def test_no_false_positives_on_code(self) -> None:
        redactor = PiiRedactor(mode=RedactionMode.FULL)
        result, types = redactor.redact("Use function(192, 168, 1, 100) for the IP")
        # Should not match "192, 168, 1, 100" as IP (commas break the pattern)
        assert "192, 168, 1, 100" in result

    def test_preserves_whitespace(self) -> None:
        redactor = PiiRedactor(mode=RedactionMode.FULL)
        result, _ = redactor.redact("  spaces  ")
        assert result.startswith("  ")
        assert result.endswith("  ")
