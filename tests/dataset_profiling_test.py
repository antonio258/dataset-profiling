"""Tests for the dataset_profiling package."""

from dataset_profiling.cli import app


def test_hello():
    """Tests the hello command."""
    result = app.invoke("hello", ["--name", "Test"])
    assert result.exit_code == 0
