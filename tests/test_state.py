"""Tests for StateStore."""

from __future__ import annotations

from pathlib import Path

import pytest

from paperless_rocm_reindexer.state import StateStore


@pytest.mark.asyncio
async def test_state_store_in_memory_initialization() -> None:
    """Test initializing StateStore with in-memory database."""
    async with StateStore(":memory:") as store:
        summary = await store.get_summary()
        assert summary == {
            "pending": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "total": 0,
        }


@pytest.mark.asyncio
async def test_state_store_wal_mode_on_disk(tmp_path: Path) -> None:
    """Test that WAL mode is enabled for disk databases and schema is created."""
    db_file = tmp_path / "subdir" / "test_state.db"
    store = StateStore(db_file)
    await store.initialize()
    await store.initialize()  # idempotent call

    assert store._db is not None
    async with store._db.execute("PRAGMA journal_mode;") as cursor:
        row = await cursor.fetchone()
        assert row is not None
        assert row[0].lower() == "wal"

    # Verify tables exist
    async with store._db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='reindex_log';"
    ) as cursor:
        row = await cursor.fetchone()
        assert row is not None
        assert row["name"] == "reindex_log"

    # Verify index exists
    async with store._db.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_reindex_status';"
    ) as cursor:
        row = await cursor.fetchone()
        assert row is not None
        assert row["name"] == "idx_reindex_status"

    await store.close()


@pytest.mark.asyncio
async def test_state_store_uninitialized_error() -> None:
    """Test that calling methods before initialize raises RuntimeError."""
    store = StateStore(":memory:")
    with pytest.raises(RuntimeError, match="StateStore is not initialized"):
        await store.is_processed(1)


@pytest.mark.asyncio
async def test_state_store_record_lifecycle() -> None:
    """Test record_pending, record_success, record_failure, record_skipped."""
    async with StateStore(":memory:") as store:
        assert await store.get_status(1) is None
        assert not await store.is_processed(1)

        # 1. Record pending
        await store.record_pending(document_id=1, title="Test Doc 1")
        assert await store.get_status(1) == "pending"
        assert not await store.is_processed(1)

        rec = await store.get_record(1)
        assert rec is not None
        assert rec["document_id"] == 1
        assert rec["title"] == "Test Doc 1"
        assert rec["status"] == "pending"
        assert rec["processed_at"] is not None

        # 2. Record success (transition from pending)
        await store.record_success(
            document_id=1,
            title="Test Doc 1 Updated",
            page_count=5,
            original_size_bytes=1024,
            archive_size_bytes=2048,
            old_checksum="old123",
            new_checksum="new456",
            inference_time_ms=123.45,
        )
        assert await store.get_status(1) == "success"
        assert await store.is_processed(1)

        rec = await store.get_record(1)
        assert rec is not None
        assert rec["title"] == "Test Doc 1 Updated"
        assert rec["page_count"] == 5
        assert rec["original_size_bytes"] == 1024
        assert rec["archive_size_bytes"] == 2048
        assert rec["old_checksum"] == "old123"
        assert rec["new_checksum"] == "new456"
        assert rec["inference_time_ms"] == 123.45
        assert rec["status"] == "success"
        assert rec["error_message"] is None

        # 3. Record failure
        await store.record_failure(
            document_id=2,
            error_message="OCR HIP out of memory",
            title="Doc 2",
        )
        assert await store.get_status(2) == "failed"
        assert not await store.is_processed(2)

        rec2 = await store.get_record(2)
        assert rec2 is not None
        assert rec2["document_id"] == 2
        assert rec2["status"] == "failed"
        assert rec2["error_message"] == "OCR HIP out of memory"

        # 4. Record skipped
        await store.record_skipped(
            document_id=3,
            reason="Already has OCR tag",
            title="Doc 3",
        )
        assert await store.get_status(3) == "skipped"
        assert await store.is_processed(3)

        rec3 = await store.get_record(3)
        assert rec3 is not None
        assert rec3["document_id"] == 3
        assert rec3["status"] == "skipped"
        assert rec3["error_message"] == "Already has OCR tag"

        # Summary check
        summary = await store.get_summary()
        assert summary == {
            "pending": 0,
            "success": 1,
            "failed": 1,
            "skipped": 1,
            "total": 3,
        }


@pytest.mark.asyncio
async def test_state_store_idempotency_and_updates() -> None:
    """Test idempotency: re-running or updating existing entries."""
    async with StateStore(":memory:") as store:
        # Initial failure
        await store.record_failure(document_id=10, error_message="Network error")
        assert await store.get_status(10) == "failed"
        assert not await store.is_processed(10)

        # Retry: set to pending
        await store.record_pending(document_id=10)
        assert await store.get_status(10) == "pending"
        rec = await store.get_record(10)
        assert rec is not None
        assert rec["error_message"] is None

        # Retry succeeds
        await store.record_success(
            document_id=10,
            page_count=2,
            new_checksum="abc",
            inference_time_ms=50.0,
        )
        assert await store.get_status(10) == "success"
        assert await store.is_processed(10)

        # Re-recording success does not duplicate rows or crash
        await store.record_success(
            document_id=10,
            inference_time_ms=45.0,
        )
        summary = await store.get_summary()
        assert summary["total"] == 1
        assert summary["success"] == 1

        rec = await store.get_record(10)
        assert rec is not None
        assert rec["page_count"] == 2  # preserved
        assert rec["inference_time_ms"] == 45.0  # updated


@pytest.mark.asyncio
async def test_get_record_nonexistent() -> None:
    """Test get_record returns None for non-existent document."""
    async with StateStore(":memory:") as store:
        assert await store.get_record(9999) is None
