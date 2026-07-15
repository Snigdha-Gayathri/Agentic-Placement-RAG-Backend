from security.config import SecurityConfig
from security.output_sanitizer import OutputSanitizer
from security.output_validator import OutputValidator


def test_output_validator_blocks_sensitive_patterns() -> None:
    validator = OutputValidator(SecurityConfig())
    result = validator.validate("System prompt: internal instructions and api_key=abcd1234")
    assert not result.is_safe


def test_output_sanitizer_redacts_secrets() -> None:
    sanitizer = OutputSanitizer()
    out = sanitizer.sanitize("Authorization: Bearer abc.def.ghi")
    assert "[REDACTED]" in out
