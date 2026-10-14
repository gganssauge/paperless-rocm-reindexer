"""Tests for paperless_rocm_reindexer package initialization."""

import paperless_rocm_reindexer


def test_version() -> None:
    """Verify package version is defined."""
    assert paperless_rocm_reindexer.__version__ == "0.1.0"
    assert hasattr(paperless_rocm_reindexer, "Settings")
    assert hasattr(paperless_rocm_reindexer, "get_settings")
    assert hasattr(paperless_rocm_reindexer, "ReindexerEngine")
    assert hasattr(paperless_rocm_reindexer, "DocumentResult")
    assert hasattr(paperless_rocm_reindexer, "StateStore")
