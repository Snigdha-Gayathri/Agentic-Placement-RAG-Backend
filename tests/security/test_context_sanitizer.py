from security.config import SecurityConfig
from security.context_sanitizer import ContextSanitizer


def test_context_sanitizer_removes_html_js() -> None:
    sanitizer = ContextSanitizer(SecurityConfig(max_context_length=500))
    text = "<script>alert(1)</script><div>Hello</div>  World"
    cleaned = sanitizer.sanitize(text)
    assert "script" not in cleaned.lower()
    assert "hello" in cleaned.lower()


def test_context_sanitizer_truncates_long_text() -> None:
    sanitizer = ContextSanitizer(SecurityConfig(max_context_length=20))
    cleaned = sanitizer.sanitize("a" * 200)
    assert len(cleaned) <= 20
