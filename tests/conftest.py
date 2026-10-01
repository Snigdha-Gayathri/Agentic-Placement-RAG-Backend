"""Pytest configuration ensuring unit tests run offline without live API side-effects."""

import os
import pytest

@pytest.fixture(autouse=True, scope="session")
def disable_live_jev_for_tests():
    """Ensure tests run deterministically against the heuristic backend unless RUN_LIVE_API_TEST=true."""
    original_jev_enabled = os.environ.get("JEV_ENABLED")
    if os.environ.get("RUN_LIVE_API_TEST") != "true":
        os.environ["JEV_ENABLED"] = "false"
    yield
    if original_jev_enabled is not None:
        os.environ["JEV_ENABLED"] = original_jev_enabled
    else:
        os.environ.pop("JEV_ENABLED", None)
