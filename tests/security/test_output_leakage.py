from security.config import SecurityConfig
from security.output_validator import OutputValidator


def test_output_validator_detects_prompt_leakage() -> None:
    validator = OutputValidator(SecurityConfig())
    result = validator.validate("Here is the system prompt and internal instructions...")
    assert not result.is_safe


def test_output_validator_detects_private_path_and_trace() -> None:
    validator = OutputValidator(SecurityConfig())
    result = validator.validate("Traceback (most recent call last): File C:\\secret\\app.py")
    assert not result.is_safe
