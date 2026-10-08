import json

from security.config import load_config
from security.logger import SecurityLogger


def test_logger_redacts_secret(caplog) -> None:
    logger = SecurityLogger()
    with caplog.at_level("INFO"):
        logger.info("event_test", value="api_key=supersecretvalue")
    assert "event_test" in caplog.text
    assert "[REDACTED]" in caplog.text
    assert "supersecretvalue" not in caplog.text


def test_config_reads_env_values(monkeypatch) -> None:
    monkeypatch.setenv("MAX_QUERY_LENGTH", "2500")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.2")
    cfg = load_config()
    assert cfg.max_query_length == 2500
    assert cfg.similarity_threshold == 0.2


def test_config_falls_back_on_invalid_env(monkeypatch) -> None:
    monkeypatch.setenv("MAX_OUTPUT_TOKENS", "invalid")
    cfg = load_config()
    assert cfg.max_output_tokens > 0
