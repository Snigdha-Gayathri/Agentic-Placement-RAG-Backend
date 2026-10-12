"""Pytest configuration ensuring unit tests run offline without live API side-effects."""

import os
import pytest

@pytest.fixture(autouse=True, scope="session")
def disable_live_jev_for_tests():
    """Ensure tests run deterministically against the heuristic backend unless RUN_LIVE_API_TEST=true."""
    original_jev_enabled = os.environ.get("JEV_ENABLED")
    original_gemini_key = os.environ.get("GEMINI_API_KEY")
    if os.environ.get("RUN_LIVE_API_TEST") != "true":
        os.environ["JEV_ENABLED"] = "false"
    if "GEMINI_API_KEY" not in os.environ or not os.environ["GEMINI_API_KEY"]:
        os.environ["GEMINI_API_KEY"] = "mock-test-key-for-offline-ci"
    yield
    if original_jev_enabled is not None:
        os.environ["JEV_ENABLED"] = original_jev_enabled
    else:
        os.environ.pop("JEV_ENABLED", None)
    if original_gemini_key is not None:
        os.environ["GEMINI_API_KEY"] = original_gemini_key
    else:
        os.environ.pop("GEMINI_API_KEY", None)
