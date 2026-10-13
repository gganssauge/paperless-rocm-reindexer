"""Tests for paperless_rocm_reindexer package initialization."""

import paperless_rocm_reindexer


def test_version() -> None:
    """Verify package version is defined."""
    assert paperless_rocm_reindexer.__version__ == "0.1.0"
