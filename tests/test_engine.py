"""Unit tests for the ReindexerEngine and ReindexWorkerPool."""

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from paperless_rocm_reindexer.engine import (
    DocumentResult,
    ReindexerEngine,
    ReindexWorkerPool,
)
from paperless_rocm_reindexer.exceptions import (
    OcrClientError,
    OcrServiceUnavailableError,
    PaperlessApiError,
)
from paperless_rocm_reindexer.state import StateStore
from paperless_rocm_reindexer.storage import ArchiveFileManager


@pytest.fixture
def temp_media_root(tmp_path: Path) -> Path:
    """Fixture providing a temporary media root directory."""
    media_dir = tmp_path / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    return media_dir


@pytest.fixture
async def state_store() -> AsyncGenerator[StateStore, None]:
    """Fixture providing an initialized in-memory StateStore."""
    store = StateStore(":memory:")
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
def file_manager(temp_media_root: Path) -> ArchiveFileManager:
    """Fixture providing an ArchiveFileManager pointing to temp media root."""
    return ArchiveFileManager(media_root=temp_media_root, default_checksum_type="md5")


@pytest.fixture
def mock_paperless_client() -> AsyncMock:
    """Fixture providing a mocked PaperlessClient."""
    client = AsyncMock()
    client.download_original.return_value = b"%PDF-1.4 raw document bytes"
    client.get_or_create_tag.side_effect = lambda name: (
        100 if name == "ocr:rocm-v4" else 200
    )
    client.patch_document.return_value = {"id": 1, "tags": [100]}
    return client


@pytest.fixture
def mock_rocm_client() -> AsyncMock:
    """Fixture providing a mocked RocmOcrClient."""
    client = AsyncMock()
    client.generate_sandwich.return_value = (
        b"%PDF-1.4 sandwich searchable content",
        {
            "X-Inference-Time-Ms": "142.5",
            "X-Page-Count": "2",
            "X-Extracted-Chars": "850",
            "X-Extracted-Text": "Extracted OCR text content",
        },
    )
    client.set_concurrency.return_value = {"status": "ok", "concurrency": 4}
    return client


@pytest.fixture
def mock_db_adapter() -> AsyncMock:
    """Fixture providing a mocked PaperlessDatabaseAdapter."""
    adapter = AsyncMock()
    adapter.get_document_archive_info.return_value = ("0000001.pdf", "old_checksum_hex")
    adapter.update_archive_checksum.return_value = True
    return adapter


@pytest.fixture
def engine(
    mock_paperless_client: AsyncMock,
    mock_rocm_client: AsyncMock,
    mock_db_adapter: AsyncMock,
    file_manager: ArchiveFileManager,
    state_store: StateStore,
) -> ReindexerEngine:
    """Fixture providing a configured ReindexerEngine instance."""
    return ReindexerEngine(
        paperless_client=mock_paperless_client,
        rocm_client=mock_rocm_client,
        db_adapter=mock_db_adapter,
        file_manager=file_manager,
        state_store=state_store,
        concurrency=4,
        tag_success="ocr:rocm-v4",
        tag_failure="ocr:failed",
        auto_setup_rocm=True,
    )


class TestReindexerEngine:
    """Test suite for ReindexerEngine."""

    @pytest.mark.asyncio
    async def test_reindex_worker_pool_alias(self) -> None:
        """Verify ReindexWorkerPool is an alias of ReindexerEngine."""
        assert ReindexWorkerPool is ReindexerEngine

    @pytest.mark.asyncio
    async def test_successful_pipeline_flow(
        self,
        engine: ReindexerEngine,
        mock_paperless_client: AsyncMock,
        mock_rocm_client: AsyncMock,
        mock_db_adapter: AsyncMock,
        file_manager: ArchiveFileManager,
        state_store: StateStore,
    ) -> None:
        """Test full successful processing of a single document."""
        doc = {
            "id": 1,
            "title": "Invoice 2024",
            "original_file_name": "inv2024.pdf",
            "tags": [5, 200],  # Includes pre-existing failure tag 200 to test removal
        }

        result = await engine.process_document(doc)

        assert result.is_success is True
        assert result.is_failed is False
        assert result.is_skipped is False
        assert result.document_id == 1
        assert result.title == "Invoice 2024"
        assert result.retried is False
        assert result.page_count == 2
        assert result.inference_time_ms == 142.5
        assert result.old_checksum == "old_checksum_hex"
        assert result.original_size_bytes == len(b"%PDF-1.4 raw document bytes")
        assert result.archive_size_bytes == len(b"%PDF-1.4 sandwich searchable content")

        # Verify client calls
        mock_paperless_client.download_original.assert_awaited_once_with(1)
        mock_rocm_client.generate_sandwich.assert_awaited_once_with(
            b"%PDF-1.4 raw document bytes", "inv2024.pdf"
        )

        # Verify file written to disk
        archive_path = file_manager.get_archive_path("0000001.pdf")
        assert archive_path.exists()
        assert archive_path.read_bytes() == b"%PDF-1.4 sandwich searchable content"

        # Verify PostgreSQL updated with computed checksum
        expected_checksum = file_manager.calculate_checksum(
            b"%PDF-1.4 sandwich searchable content"
        )
        assert result.new_checksum == expected_checksum
        mock_db_adapter.update_archive_checksum.assert_awaited_once_with(
            1, expected_checksum
        )

        # Verify Paperless patched with tags and fulltext, removing failure tag 200
        mock_paperless_client.patch_document.assert_awaited_once_with(
            1,
            content="Extracted OCR text content",
            tags=[5, 100],  # Existing 5 + success tag 100 (tag 200 removed)
        )

        # Verify StateStore recorded success
        stored_record = await state_store.get_record(1)
        assert stored_record is not None
        assert stored_record["status"] == "success"
        assert stored_record["new_checksum"] == expected_checksum
        assert stored_record["page_count"] == 2

    @pytest.mark.asyncio
    async def test_default_archive_filename_when_empty(
        self,
        engine: ReindexerEngine,
        mock_db_adapter: AsyncMock,
        file_manager: ArchiveFileManager,
    ) -> None:
        """Test default filename generation when database archive_filename is None."""
        mock_db_adapter.get_document_archive_info.return_value = (None, None)

        doc = {"id": 42, "title": "Doc without archive filename"}
        result = await engine.process_document(doc)

        assert result.status == "success"
        archive_path = file_manager.get_archive_path("0000042.pdf")
        assert archive_path.exists()

    @pytest.mark.asyncio
    async def test_missing_id_in_document_raises_value_error(
        self,
        engine: ReindexerEngine,
    ) -> None:
        """Test document dictionary without id raises ValueError."""
        with pytest.raises(ValueError, match="missing 'id' key"):
            await engine.process_document({"title": "No ID"})

    @pytest.mark.asyncio
    async def test_skip_already_processed_document(
        self,
        engine: ReindexerEngine,
        mock_paperless_client: AsyncMock,
        mock_rocm_client: AsyncMock,
        state_store: StateStore,
    ) -> None:
        """Test skipping documents that have already succeeded in the state store."""
        await state_store.record_success(document_id=42, title="Pre-processed Doc")

        doc = {"id": 42, "title": "Pre-processed Doc", "tags": []}
        result = await engine.process_document(doc)

        assert result.status == "skipped"
        assert result.is_skipped is True
        mock_paperless_client.download_original.assert_not_called()
        mock_rocm_client.generate_sandwich.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_marked_skipped_in_state_store(
        self,
        engine: ReindexerEngine,
        mock_paperless_client: AsyncMock,
        mock_rocm_client: AsyncMock,
        state_store: StateStore,
    ) -> None:
        """Test skipping documents marked skipped in state store."""
        await state_store.record_skipped(document_id=77, reason="Manual skip")

        doc = {"id": 77, "title": "Skipped Doc"}
        result = await engine.process_document(doc)

        assert result.status == "skipped"
        assert result.is_skipped is True
        mock_paperless_client.download_original.assert_not_called()

    @pytest.mark.asyncio
    async def test_503_adaptive_backoff_and_successful_retry(
        self,
        engine: ReindexerEngine,
        mock_paperless_client: AsyncMock,
        mock_rocm_client: AsyncMock,
        state_store: StateStore,
    ) -> None:
        """Test adaptive backoff scaling rocm concurrency down to 1 on 503 and retrying."""
        # First call raises 503, second call succeeds
        mock_rocm_client.generate_sandwich.side_effect = [
            OcrServiceUnavailableError("GPU OOM / HIP error"),
            (
                b"%PDF-1.4 retried sandwich",
                {"X-Inference-Time-Ms": "250.0", "X-Page-Count": "1"},
            ),
        ]

        doc = {"id": 10, "title": "OOM Document", "tags": [1]}
        result = await engine.process_document(doc)

        assert result.status == "success"
        assert result.retried is True

        # Verify concurrency scaling calls:
        # 1. engine.initialize() -> set_concurrency(4)
        # 2. Backoff retry -> set_concurrency(1)
        # 3. Post-retry restoration -> set_concurrency(4)
        assert mock_rocm_client.set_concurrency.await_count >= 3
        calls = [
            call.args[0] for call in mock_rocm_client.set_concurrency.await_args_list
        ]
        assert 1 in calls
        assert calls[-1] == 4  # Concurrency restored back to 4

        # Verify state store recorded success
        record = await state_store.get_record(10)
        assert record is not None
        assert record["status"] == "success"

    @pytest.mark.asyncio
    async def test_503_retry_failure_tags_failed_and_records_state(
        self,
        engine: ReindexerEngine,
        mock_paperless_client: AsyncMock,
        mock_rocm_client: AsyncMock,
        state_store: StateStore,
    ) -> None:
        """Test 503 retry failure causes document to be tagged ocr:failed without crashing."""
        # Both initial call and retry raise 503
        mock_rocm_client.generate_sandwich.side_effect = [
            OcrServiceUnavailableError("GPU OOM initial"),
            OcrServiceUnavailableError("GPU OOM retry failed"),
        ]

        doc = {"id": 20, "title": "Permanently Failing Doc", "tags": [5, 100]}
        result = await engine.process_document(doc)

        assert result.status == "failed"
        assert result.is_failed is True
        assert result.retried is True
        assert "OCR failed" in (result.error or "")

        # Verify rocm concurrency was not ramped back up on failure (remains at isolated 1)
        last_concurrency_call = mock_rocm_client.set_concurrency.await_args_list[
            -1
        ].args[0]
        assert last_concurrency_call == 1

        # Verify failure tag was applied on Paperless (tag 200 is ocr:failed, tag 100 removed)
        mock_paperless_client.patch_document.assert_awaited_once_with(
            20,
            tags=[5, 200],
        )

        # Verify StateStore recorded failure
        record = await state_store.get_record(20)
        assert record is not None
        assert record["status"] == "failed"
        assert "OCR failed" in record["error_message"]

    @pytest.mark.asyncio
    async def test_non_503_ocr_error_fails_without_retry(
        self,
        engine: ReindexerEngine,
        mock_rocm_client: AsyncMock,
        state_store: StateStore,
    ) -> None:
        """Test HTTP 400 or other non-503 OCR errors fail immediately without retry."""
        mock_rocm_client.generate_sandwich.side_effect = OcrClientError(
            "Corrupt image file", status_code=400
        )

        doc = {"id": 55, "title": "Corrupt Format"}
        result = await engine.process_document(doc)

        assert result.status == "failed"
        assert result.retried is False
        assert mock_rocm_client.generate_sandwich.await_count == 1

        record = await state_store.get_record(55)
        assert record is not None
        assert record["status"] == "failed"

    @pytest.mark.asyncio
    async def test_missing_document_in_database(
        self,
        engine: ReindexerEngine,
        mock_paperless_client: AsyncMock,
        mock_db_adapter: AsyncMock,
        state_store: StateStore,
    ) -> None:
        """Test handling when document is not found in PostgreSQL documents_document table."""
        mock_db_adapter.get_document_archive_info.return_value = None

        doc = {"id": 99, "title": "Ghost Doc"}
        result = await engine.process_document(doc)

        assert result.status == "failed"
        assert "not found in Paperless PostgreSQL" in (result.error or "")

        record = await state_store.get_record(99)
        assert record is not None
        assert record["status"] == "failed"

    @pytest.mark.asyncio
    async def test_download_original_failure(
        self,
        engine: ReindexerEngine,
        mock_paperless_client: AsyncMock,
        state_store: StateStore,
    ) -> None:
        """Test handling when download_original throws an exception."""
        mock_paperless_client.download_original.side_effect = PaperlessApiError(
            "Original file missing", status_code=404
        )

        doc = {"id": 33, "title": "Corrupt Doc"}
        result = await engine.process_document(doc)

        assert result.status == "failed"
        assert "Failed to download original" in (result.error or "")

        record = await state_store.get_record(33)
        assert record is not None
        assert record["status"] == "failed"

    @pytest.mark.asyncio
    async def test_commit_success_failure_handled(
        self,
        engine: ReindexerEngine,
        mock_db_adapter: AsyncMock,
        state_store: StateStore,
    ) -> None:
        """Test that failure during commit phase records failure cleanly."""
        mock_db_adapter.update_archive_checksum.side_effect = RuntimeError(
            "DB connection dropped"
        )

        doc = {"id": 88, "title": "Doc DB Crash"}
        result = await engine.process_document(doc)

        assert result.status == "failed"
        assert "Failed committing OCR output" in (result.error or "")

        record = await state_store.get_record(88)
        assert record is not None
        assert record["status"] == "failed"

    @pytest.mark.asyncio
    async def test_failure_handling_tolerates_client_and_store_errors(
        self,
        engine: ReindexerEngine,
        mock_paperless_client: AsyncMock,
        mock_rocm_client: AsyncMock,
        state_store: StateStore,
    ) -> None:
        """Test _record_failure does not crash if StateStore or Paperless patch also fail."""
        mock_rocm_client.generate_sandwich.side_effect = RuntimeError("Fatal OCR bug")
        mock_paperless_client.patch_document.side_effect = PaperlessApiError(
            "API offline"
        )

        with patch.object(
            state_store, "record_failure", side_effect=RuntimeError("SQLite disk error")
        ):
            doc = {"id": 105, "title": "Double Disaster"}
            result = await engine.process_document(doc)

            assert result.status == "failed"
            assert "Fatal OCR bug" in (result.error or "")

    @pytest.mark.asyncio
    async def test_initialize_idempotency_and_error_handling(
        self,
        mock_paperless_client: AsyncMock,
        mock_rocm_client: AsyncMock,
        mock_db_adapter: AsyncMock,
        file_manager: ArchiveFileManager,
        state_store: StateStore,
    ) -> None:
        """Test initialize is idempotent and logs errors without crashing."""
        mock_rocm_client.set_concurrency.side_effect = RuntimeError("OCR setup failed")
        mock_paperless_client.get_or_create_tag.side_effect = RuntimeError(
            "Paperless tag API offline"
        )

        engine = ReindexerEngine(
            paperless_client=mock_paperless_client,
            rocm_client=mock_rocm_client,
            db_adapter=mock_db_adapter,
            file_manager=file_manager,
            state_store=state_store,
        )

        await engine.initialize()
        assert engine._initialized is True

        # Second call returns early
        await engine.initialize()
        assert engine._initialized is True

    @pytest.mark.asyncio
    async def test_get_tag_id_uncached(
        self,
        engine: ReindexerEngine,
        mock_paperless_client: AsyncMock,
    ) -> None:
        """Test tag resolution when tag IDs are not pre-cached."""
        engine._tag_success_id = None
        engine._tag_failure_id = None
        mock_paperless_client.get_or_create_tag.side_effect = lambda name: (
            111 if name == "ocr:rocm-v4" else 222
        )

        success_id = await engine._get_tag_id("ocr:rocm-v4")
        failure_id = await engine._get_tag_id("ocr:failed")

        assert success_id == 111
        assert failure_id == 222
        assert engine._tag_success_id == 111
        assert engine._tag_failure_id == 222

    @pytest.mark.asyncio
    async def test_set_concurrency_error_during_backoff_retry(
        self,
        engine: ReindexerEngine,
        mock_rocm_client: AsyncMock,
    ) -> None:
        """Test set_concurrency exception during 503 backoff is handled gracefully."""
        mock_rocm_client.generate_sandwich.side_effect = [
            OcrServiceUnavailableError("GPU 503"),
            (b"%PDF-1.4 sandwich", {}),
        ]
        # set_concurrency fails
        mock_rocm_client.set_concurrency.side_effect = RuntimeError(
            "Failed to set concurrency"
        )

        doc = {"id": 110, "title": "Set Concurrency Fail"}
        result = await engine.process_document(doc)

        assert result.status == "success"
        assert result.retried is True

    @pytest.mark.asyncio
    async def test_invalid_header_metrics_fallback_to_defaults(
        self,
        engine: ReindexerEngine,
        mock_rocm_client: AsyncMock,
    ) -> None:
        """Test unparseable header values fallback to safe defaults."""
        mock_rocm_client.generate_sandwich.return_value = (
            b"%PDF-1.4 content",
            {
                "X-Inference-Time-Ms": "not_a_float",
                "X-Page-Count": "not_an_int",
            },
        )

        doc = {"id": 115, "title": "Invalid Headers Doc"}
        result = await engine.process_document(doc)

        assert result.status == "success"
        assert result.inference_time_ms == 0.0
        assert result.page_count == 1

    @pytest.mark.asyncio
    async def test_concurrency_limiting_in_batch(
        self,
        engine: ReindexerEngine,
        mock_rocm_client: AsyncMock,
    ) -> None:
        """Test that worker concurrency never exceeds configured semaphore limit."""
        engine.concurrency = 2
        engine._semaphore = asyncio.Semaphore(2)

        active_count = 0
        max_active = 0
        lock = asyncio.Lock()

        async def slow_sandwich(*args, **kwargs):
            nonlocal active_count, max_active
            async with lock:
                active_count += 1
                if active_count > max_active:
                    max_active = active_count
            await asyncio.sleep(0.02)
            async with lock:
                active_count -= 1
            return b"%PDF-1.4 dummy", {"X-Page-Count": "1"}

        mock_rocm_client.generate_sandwich.side_effect = slow_sandwich

        docs = [{"id": i, "title": f"Doc {i}"} for i in range(1, 6)]
        results = await engine.process_batch(docs)

        assert len(results) == 5
        assert max_active <= 2
        assert all(r.status == "success" for r in results)

    @pytest.mark.asyncio
    async def test_process_batch_with_progress_callback(
        self,
        engine: ReindexerEngine,
    ) -> None:
        """Test batch runner notifies on_progress callback."""
        docs = [{"id": 1, "title": "Doc 1"}, {"id": 2, "title": "Doc 2"}]
        progress_calls: list[DocumentResult] = []

        # Also test on_progress exception does not disrupt batch
        def throwing_callback(res: DocumentResult) -> None:
            progress_calls.append(res)
            if res.document_id == 1:
                raise ValueError("UI callback bug")

        results = await engine.process_batch(docs, on_progress=throwing_callback)

        assert len(results) == 2
        assert len(progress_calls) == 2
        assert {r.document_id for r in progress_calls} == {1, 2}

    @pytest.mark.asyncio
    async def test_process_batch_empty_list(
        self,
        engine: ReindexerEngine,
    ) -> None:
        """Test process_batch handles empty list safely."""
        results = await engine.process_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_process_batch_cancellation(
        self,
        engine: ReindexerEngine,
        mock_rocm_client: AsyncMock,
    ) -> None:
        """Test process_batch propagates CancelledError and sets stopping flag."""

        async def slow_sandwich(*args, **kwargs):
            await asyncio.sleep(1.0)
            return b"%PDF-1.4 dummy", {}

        mock_rocm_client.generate_sandwich.side_effect = slow_sandwich

        docs = [{"id": 201, "title": "Cancelled Doc"}]
        task = asyncio.create_task(engine.process_batch(docs))
        await asyncio.sleep(0.01)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert engine._stopping is True

    @pytest.mark.asyncio
    async def test_engine_graceful_stop(
        self,
        engine: ReindexerEngine,
        mock_rocm_client: AsyncMock,
    ) -> None:
        """Test that calling stop() drains currently active item and skips subsequent queue items."""
        engine.concurrency = 1
        engine._semaphore = asyncio.Semaphore(1)

        async def pause_sandwich(*args, **kwargs):
            # Stop the engine while item 1 is being processed
            engine.stop()
            return b"%PDF-1.4 dummy", {"X-Page-Count": "1"}

        mock_rocm_client.generate_sandwich.side_effect = pause_sandwich

        docs = [
            {"id": 100, "title": "First"},
            {"id": 101, "title": "Second"},
            {"id": 102, "title": "Third"},
        ]
        results = await engine.process_batch(docs)

        # Item 100 finishes, but 101 and 102 were not processed because of stop()
        assert len(results) == 1
        assert results[0].document_id == 100

    @pytest.mark.asyncio
    async def test_context_manager_lifecycle(
        self,
        mock_paperless_client: AsyncMock,
        mock_rocm_client: AsyncMock,
        mock_db_adapter: AsyncMock,
        file_manager: ArchiveFileManager,
        state_store: StateStore,
    ) -> None:
        """Test async context manager initializes and closes cleanly."""
        engine = ReindexerEngine(
            paperless_client=mock_paperless_client,
            rocm_client=mock_rocm_client,
            db_adapter=mock_db_adapter,
            file_manager=file_manager,
            state_store=state_store,
            concurrency=3,
        )

        async with engine as active_engine:
            assert active_engine._initialized is True
            mock_rocm_client.set_concurrency.assert_awaited_with(3)

        assert engine._stopping is True

    def test_signal_handlers_fallback(
        self,
        mock_paperless_client: AsyncMock,
        mock_rocm_client: AsyncMock,
        mock_db_adapter: AsyncMock,
        file_manager: ArchiveFileManager,
        state_store: StateStore,
    ) -> None:
        """Test signal handler registration when loop raises RuntimeError or NotImplementedError."""
        engine = ReindexerEngine(
            paperless_client=mock_paperless_client,
            rocm_client=mock_rocm_client,
            db_adapter=mock_db_adapter,
            file_manager=file_manager,
            state_store=state_store,
        )

        # 1. No running loop
        engine.register_signal_handlers()
        engine.unregister_signal_handlers()

        # 2. Loop raises NotImplementedError
        fake_loop = MagicMock()
        fake_loop.add_signal_handler.side_effect = NotImplementedError
        fake_loop.remove_signal_handler.side_effect = NotImplementedError
        with patch("asyncio.get_running_loop", return_value=fake_loop):
            engine.register_signal_handlers()
            engine.unregister_signal_handlers()

    def test_engine_concurrency_capping(
        self,
        mock_paperless_client: AsyncMock,
        mock_rocm_client: AsyncMock,
        mock_db_adapter: AsyncMock,
        file_manager: ArchiveFileManager,
        state_store: StateStore,
    ) -> None:
        """Test engine clamps concurrency between 1 and 8."""
        engine_high = ReindexerEngine(
            paperless_client=mock_paperless_client,
            rocm_client=mock_rocm_client,
            db_adapter=mock_db_adapter,
            file_manager=file_manager,
            state_store=state_store,
            concurrency=20,
        )
        assert engine_high.concurrency == 8

        engine_low = ReindexerEngine(
            paperless_client=mock_paperless_client,
            rocm_client=mock_rocm_client,
            db_adapter=mock_db_adapter,
            file_manager=file_manager,
            state_store=state_store,
            concurrency=0,
        )
        assert engine_low.concurrency == 1

    @pytest.mark.asyncio
    async def test_extract_text_from_pdf_success(self) -> None:
        """Test extract_text_from_pdf using a synthesized PDF."""
        import io

        import pypdf

        from paperless_rocm_reindexer.models import extract_text_from_pdf

        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=100, height=100)
        buf = io.BytesIO()
        writer.write(buf)
        empty_pdf_bytes = buf.getvalue()

        # Empty page returns None (falling back to headers)
        assert extract_text_from_pdf(empty_pdf_bytes) is None

        # Corrupt bytes return None gracefully
        assert extract_text_from_pdf(b"not a valid pdf") is None
