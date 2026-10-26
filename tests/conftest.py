"""Pytest configuration and fixtures for PlantPilot tests."""

import pytest

from plantpilot.config import reset_config
from plantpilot.storage import reset_database


@pytest.fixture(autouse=True)
def reset_state():
    """Reset configuration and database state before each test."""
    reset_config()
    reset_database()
    yield
    reset_config()
    reset_database()