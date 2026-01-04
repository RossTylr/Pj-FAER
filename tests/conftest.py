"""Pytest fixtures for Pj FAER tests."""

import pytest


@pytest.fixture
def default_seed() -> int:
    """Default random seed for reproducible tests."""
    return 42


@pytest.fixture
def short_run_length() -> float:
    """Short run length (minutes) for quick tests."""
    return 60.0  # 1 hour
