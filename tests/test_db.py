from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from paperless_rocm_reindexer.db import PaperlessDatabaseAdapter


@pytest.fixture
def db_adapter():
    return PaperlessDatabaseAdapter(
        db_url="postgresql://user:pass@localhost:5432/paperless"
    )


def test_db_adapter_init(db_adapter):
    assert db_adapter.db_url == "postgresql://user:pass@localhost:5432/paperless"
    assert db_adapter.min_size == 1
    assert db_adapter.max_size == 10
    with pytest.raises(RuntimeError, match="Database pool is not connected"):
        _ = db_adapter.pool


@pytest.mark.asyncio
async def test_connect_and_close(db_adapter):
    mock_pool = MagicMock()
    mock_pool.close = AsyncMock()
    with patch(
        "asyncpg.create_pool", new_callable=AsyncMock, return_value=mock_pool
    ) as mock_create_pool:
        await db_adapter.connect()
        mock_create_pool.assert_awaited_once_with(
            dsn=db_adapter.db_url,
            min_size=db_adapter.min_size,
            max_size=db_adapter.max_size,
        )
        assert db_adapter.pool is mock_pool

        # Calling connect again doesn't recreate pool
        await db_adapter.connect()
        mock_create_pool.assert_awaited_once()

        await db_adapter.close()
        mock_pool.close.assert_awaited_once()
        with pytest.raises(RuntimeError, match="Database pool is not connected"):
            _ = db_adapter.pool


@pytest.mark.asyncio
async def test_context_manager(db_adapter):
    mock_pool = MagicMock()
    mock_pool.close = AsyncMock()
    with patch("asyncpg.create_pool", new_callable=AsyncMock, return_value=mock_pool):
        async with db_adapter as adapter:
            assert adapter.pool is mock_pool
        mock_pool.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_document_archive_info_found(db_adapter):
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_conn.fetchrow.return_value = {
        "archive_filename": "0000123.pdf",
        "archive_checksum": "abcdef0123456789abcdef0123456789",
    }
    # Mock acquire context manager
    acquire_ctx = AsyncMock()
    acquire_ctx.__aenter__.return_value = mock_conn
    acquire_ctx.__aexit__.return_value = None
    mock_pool.acquire.return_value = acquire_ctx

    db_adapter._pool = mock_pool

    info = await db_adapter.get_document_archive_info(123)
    assert info == ("0000123.pdf", "abcdef0123456789abcdef0123456789")

    mock_conn.fetchrow.assert_awaited_once()
    query = mock_conn.fetchrow.call_args[0][0]
    assert "SELECT archive_filename, archive_checksum" in query
    assert "WHERE id = $1" in query
    assert mock_conn.fetchrow.call_args[0][1] == 123


@pytest.mark.asyncio
async def test_get_document_archive_info_not_found(db_adapter):
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_conn.fetchrow.return_value = None

    acquire_ctx = AsyncMock()
    acquire_ctx.__aenter__.return_value = mock_conn
    acquire_ctx.__aexit__.return_value = None
    mock_pool.acquire.return_value = acquire_ctx

    db_adapter._pool = mock_pool

    info = await db_adapter.get_document_archive_info(999)
    assert info is None
    mock_conn.fetchrow.assert_awaited_once_with(mock_conn.fetchrow.call_args[0][0], 999)


@pytest.mark.asyncio
async def test_update_archive_checksum_success(db_adapter):
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_conn.execute.return_value = "UPDATE 1"

    acquire_ctx = AsyncMock()
    acquire_ctx.__aenter__.return_value = mock_conn
    acquire_ctx.__aexit__.return_value = None
    mock_pool.acquire.return_value = acquire_ctx

    db_adapter._pool = mock_pool

    res = await db_adapter.update_archive_checksum(123, "newhash123")
    assert res is True

    mock_conn.execute.assert_awaited_once()
    query = mock_conn.execute.call_args[0][0]
    assert "UPDATE documents_document" in query
    assert "SET archive_checksum = $1" in query
    assert "WHERE id = $2" in query
    assert mock_conn.execute.call_args[0][1] == "newhash123"
    assert mock_conn.execute.call_args[0][2] == 123


@pytest.mark.asyncio
async def test_update_archive_checksum_not_found(db_adapter):
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_conn.execute.return_value = "UPDATE 0"

    acquire_ctx = AsyncMock()
    acquire_ctx.__aenter__.return_value = mock_conn
    acquire_ctx.__aexit__.return_value = None
    mock_pool.acquire.return_value = acquire_ctx

    db_adapter._pool = mock_pool

    res = await db_adapter.update_archive_checksum(999, "newhash123")
    assert res is False


@pytest.mark.asyncio
async def test_update_archive_checksum_unexpected_status(db_adapter):
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_conn.execute.return_value = ""

    acquire_ctx = AsyncMock()
    acquire_ctx.__aenter__.return_value = mock_conn
    acquire_ctx.__aexit__.return_value = None
    mock_pool.acquire.return_value = acquire_ctx

    db_adapter._pool = mock_pool

    res = await db_adapter.update_archive_checksum(123, "newhash123")
    assert res is False
