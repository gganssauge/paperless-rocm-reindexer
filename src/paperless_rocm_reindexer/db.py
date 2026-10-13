"""Database adapter for Paperless PostgreSQL database."""

from typing import Any

import asyncpg


class PaperlessDatabaseAdapter:
    """Adapter for interacting with the Paperless-ngx PostgreSQL database."""

    def __init__(self, db_url: str, min_size: int = 1, max_size: int = 10) -> None:
        """Initialize adapter.

        Args:
            db_url: PostgreSQL connection URI (e.g. postgresql://user:pass@host:5432/paperless).
            min_size: Minimum connection pool size.
            max_size: Maximum connection pool size.
        """
        self.db_url = db_url
        self.min_size = min_size
        self.max_size = max_size
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Create asyncpg connection pool if not already connected."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                dsn=self.db_url,
                min_size=self.min_size,
                max_size=self.max_size,
            )

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def __aenter__(self) -> "PaperlessDatabaseAdapter":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    @property
    def pool(self) -> asyncpg.Pool:
        """Get active connection pool, raising RuntimeError if not connected."""
        if self._pool is None:
            raise RuntimeError("Database pool is not connected. Call connect() first.")
        return self._pool

    async def get_document_archive_info(
        self, document_id: int
    ) -> tuple[str, str] | None:
        """Fetch document archive details from Paperless documents_document table.

        Args:
            document_id: Primary key ID of the document.

        Returns:
            Tuple of (archive_filename, archive_checksum) or None if document not found.
            Note that archive_filename or archive_checksum can be empty/None in DB;
            they are returned as string or empty string.
        """
        query = """
        SELECT archive_filename, archive_checksum
        FROM documents_document
        WHERE id = $1;
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, document_id)
            if row is None:
                return None
            return (row["archive_filename"], row["archive_checksum"])

    async def update_archive_checksum(
        self, document_id: int, archive_checksum: str
    ) -> bool:
        """Update archive_checksum for a document in Paperless documents_document table.

        Args:
            document_id: Primary key ID of the document.
            archive_checksum: New checksum hex string.

        Returns:
            True if document was found and updated, False otherwise.
        """
        query = """
        UPDATE documents_document
        SET archive_checksum = $1
        WHERE id = $2;
        """
        async with self.pool.acquire() as conn:
            status = await conn.execute(query, archive_checksum, document_id)
            # asyncpg returns status string e.g. "UPDATE 1" or "UPDATE 0"
            if status and status.startswith("UPDATE "):
                count_str = status.split(" ")[1]
                return int(count_str) > 0
            return False
