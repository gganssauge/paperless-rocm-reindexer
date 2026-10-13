"""SQLite state store and checkpoint manager."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import aiosqlite

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS reindex_log (
    document_id INTEGER PRIMARY KEY,
    title TEXT,
    page_count INTEGER,
    original_size_bytes INTEGER,
    archive_size_bytes INTEGER,
    old_checksum TEXT,
    new_checksum TEXT,
    inference_time_ms REAL,
    status TEXT CHECK(status IN ('pending', 'success', 'failed', 'skipped')),
    error_message TEXT,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_reindex_status ON reindex_log(status);
"""


class StateStore:
    """Persistent state tracking to avoid duplicate processing and allow resume."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Initialize database connection, enable WAL mode, and create schema."""
        if self._db is not None:
            return

        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row

        # Set WAL journal mode (note: in-memory DBs ignore or return memory for WAL)
        if self.db_path != ":memory:":
            await self._db.execute("PRAGMA journal_mode=WAL;")

        await self._db.execute(CREATE_TABLE_SQL)
        await self._db.execute(CREATE_INDEX_SQL)
        await self._db.commit()

    async def close(self) -> None:
        """Close database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def __aenter__(self) -> StateStore:
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.close()

    def _ensure_connected(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError(
                "StateStore is not initialized. Call initialize() or use async with context."
            )
        return self._db

    async def is_processed(self, document_id: int) -> bool:
        """Check if document has been successfully processed or skipped."""
        db = self._ensure_connected()
        async with db.execute(
            "SELECT status FROM reindex_log WHERE document_id = ?",
            (document_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return False
            return row["status"] in ("success", "skipped")

    async def get_status(self, document_id: int) -> str | None:
        """Get the current processing status of a document."""
        db = self._ensure_connected()
        async with db.execute(
            "SELECT status FROM reindex_log WHERE document_id = ?",
            (document_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return str(row["status"])

    async def record_pending(self, document_id: int, title: str | None = None) -> None:
        """Record or update a document as pending."""
        db = self._ensure_connected()
        sql = """
        INSERT INTO reindex_log (document_id, title, status, processed_at)
        VALUES (?, ?, 'pending', CURRENT_TIMESTAMP)
        ON CONFLICT(document_id) DO UPDATE SET
            title = COALESCE(excluded.title, reindex_log.title),
            status = 'pending',
            error_message = NULL,
            processed_at = CURRENT_TIMESTAMP;
        """
        await db.execute(sql, (document_id, title))
        await db.commit()

    async def record_success(
        self,
        document_id: int,
        title: str | None = None,
        page_count: int | None = None,
        original_size_bytes: int | None = None,
        archive_size_bytes: int | None = None,
        old_checksum: str | None = None,
        new_checksum: str | None = None,
        inference_time_ms: float | None = None,
    ) -> None:
        """Record a successful document reindex."""
        db = self._ensure_connected()
        sql = """
        INSERT INTO reindex_log (
            document_id, title, page_count, original_size_bytes,
            archive_size_bytes, old_checksum, new_checksum,
            inference_time_ms, status, error_message, processed_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'success', NULL, CURRENT_TIMESTAMP)
        ON CONFLICT(document_id) DO UPDATE SET
            title = COALESCE(excluded.title, reindex_log.title),
            page_count = COALESCE(excluded.page_count, reindex_log.page_count),
            original_size_bytes = COALESCE(excluded.original_size_bytes, reindex_log.original_size_bytes),
            archive_size_bytes = COALESCE(excluded.archive_size_bytes, reindex_log.archive_size_bytes),
            old_checksum = COALESCE(excluded.old_checksum, reindex_log.old_checksum),
            new_checksum = COALESCE(excluded.new_checksum, reindex_log.new_checksum),
            inference_time_ms = COALESCE(excluded.inference_time_ms, reindex_log.inference_time_ms),
            status = 'success',
            error_message = NULL,
            processed_at = CURRENT_TIMESTAMP;
        """
        await db.execute(
            sql,
            (
                document_id,
                title,
                page_count,
                original_size_bytes,
                archive_size_bytes,
                old_checksum,
                new_checksum,
                inference_time_ms,
            ),
        )
        await db.commit()

    async def record_failure(
        self,
        document_id: int,
        error_message: str,
        title: str | None = None,
    ) -> None:
        """Record a failed document reindex."""
        db = self._ensure_connected()
        sql = """
        INSERT INTO reindex_log (document_id, title, status, error_message, processed_at)
        VALUES (?, ?, 'failed', ?, CURRENT_TIMESTAMP)
        ON CONFLICT(document_id) DO UPDATE SET
            title = COALESCE(excluded.title, reindex_log.title),
            status = 'failed',
            error_message = excluded.error_message,
            processed_at = CURRENT_TIMESTAMP;
        """
        await db.execute(sql, (document_id, title, error_message))
        await db.commit()

    async def record_skipped(
        self,
        document_id: int,
        reason: str | None = None,
        title: str | None = None,
    ) -> None:
        """Record a skipped document."""
        db = self._ensure_connected()
        sql = """
        INSERT INTO reindex_log (document_id, title, status, error_message, processed_at)
        VALUES (?, ?, 'skipped', ?, CURRENT_TIMESTAMP)
        ON CONFLICT(document_id) DO UPDATE SET
            title = COALESCE(excluded.title, reindex_log.title),
            status = 'skipped',
            error_message = excluded.error_message,
            processed_at = CURRENT_TIMESTAMP;
        """
        await db.execute(sql, (document_id, title, reason))
        await db.commit()

    async def get_summary(self) -> dict[str, int]:
        """Get summary counts of document statuses."""
        db = self._ensure_connected()
        summary = {
            "pending": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "total": 0,
        }
        async with db.execute(
            "SELECT status, COUNT(*) as count FROM reindex_log GROUP BY status"
        ) as cursor:
            async for row in cursor:
                status = row["status"]
                count = row["count"]
                if status in summary:
                    summary[status] = count
                summary["total"] += count

        return summary

    async def get_record(self, document_id: int) -> dict[str, Any] | None:
        """Get full record for a document."""
        db = self._ensure_connected()
        async with db.execute(
            "SELECT * FROM reindex_log WHERE document_id = ?",
            (document_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return dict(row)
