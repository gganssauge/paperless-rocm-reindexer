"""Reindexer worker engine and concurrency orchestrator."""

from __future__ import annotations

import asyncio
import logging
import signal
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal

from paperless_rocm_reindexer.exceptions import (
    OcrServiceUnavailableError,
)

if TYPE_CHECKING:
    from paperless_rocm_reindexer.clients.paperless import PaperlessClient
    from paperless_rocm_reindexer.clients.rocm_ocr import RocmOcrClient
    from paperless_rocm_reindexer.db import PaperlessDatabaseAdapter
    from paperless_rocm_reindexer.state import StateStore
    from paperless_rocm_reindexer.storage import ArchiveFileManager

logger = logging.getLogger(__name__)


@dataclass
class DocumentResult:
    """Result of processing a single document through the reindexer engine."""

    document_id: int
    status: Literal["success", "failed", "skipped"]
    title: str | None = None
    old_checksum: str | None = None
    new_checksum: str | None = None
    original_size_bytes: int | None = None
    archive_size_bytes: int | None = None
    page_count: int | None = None
    inference_time_ms: float | None = None
    error: str | None = None
    retried: bool = False

    @property
    def is_success(self) -> bool:
        """Check if document processing succeeded."""
        return self.status == "success"

    @property
    def is_failed(self) -> bool:
        """Check if document processing failed."""
        return self.status == "failed"

    @property
    def is_skipped(self) -> bool:
        """Check if document was skipped."""
        return self.status == "skipped"


class ReindexerEngine:
    """Orchestrator for reindexing Paperless-ngx documents with rocm-ocr."""

    def __init__(
        self,
        paperless_client: PaperlessClient,
        rocm_client: RocmOcrClient,
        db_adapter: PaperlessDatabaseAdapter,
        file_manager: ArchiveFileManager,
        state_store: StateStore,
        concurrency: int = 4,
        tag_success: str = "ocr:rocm-v4",
        tag_failure: str = "ocr:failed",
        auto_setup_rocm: bool = True,
    ) -> None:
        """Initialize ReindexerEngine.

        Args:
            paperless_client: Client for Paperless-ngx REST API.
            rocm_client: Client for rocm-ocr service.
            db_adapter: Adapter for Paperless PostgreSQL database.
            file_manager: Manager for local archive files.
            state_store: Local SQLite state tracker.
            concurrency: Maximum concurrent document processing tasks.
            tag_success: Tag name applied upon successful OCR.
            tag_failure: Tag name applied upon permanent OCR failure.
            auto_setup_rocm: If True, configures initial concurrency on rocm-ocr service.
        """
        self.paperless_client = paperless_client
        self.rocm_client = rocm_client
        self.db_adapter = db_adapter
        self.file_manager = file_manager
        self.state_store = state_store
        self.concurrency = max(1, concurrency)
        self.tag_success = tag_success
        self.tag_failure = tag_failure
        self.auto_setup_rocm = auto_setup_rocm

        self._semaphore = asyncio.Semaphore(self.concurrency)
        self._isolation_lock = asyncio.Lock()
        self._backoff_event = asyncio.Event()
        self._backoff_event.set()

        self._tag_success_id: int | None = None
        self._tag_failure_id: int | None = None
        self._initialized: bool = False
        self._stopping: bool = False
        self._active_tasks: set[asyncio.Task[Any]] = set()

    async def initialize(self) -> None:
        """Initialize engine, ensure state store is ready, and configure OCR concurrency."""
        if self._initialized:
            return

        await self.state_store.initialize()

        if self.auto_setup_rocm:
            try:
                await self.rocm_client.set_concurrency(self.concurrency)
                logger.info("Initialized rocm-ocr concurrency to %d", self.concurrency)
            except Exception as e:
                logger.warning("Could not set initial rocm-ocr concurrency: %s", e)

        try:
            self._tag_success_id = await self.paperless_client.get_or_create_tag(
                self.tag_success
            )
            self._tag_failure_id = await self.paperless_client.get_or_create_tag(
                self.tag_failure
            )
        except Exception as e:
            logger.warning("Could not pre-resolve tag IDs from Paperless: %s", e)

        self._initialized = True

    async def close(self) -> None:
        """Clean up engine resources and unregister signal handlers."""
        self.unregister_signal_handlers()
        self._stopping = True

    async def __aenter__(self) -> ReindexerEngine:
        """Enter async context, initializing engine and registering signal handlers."""
        await self.initialize()
        self.register_signal_handlers()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context and close resources."""
        await self.close()

    def stop(self) -> None:
        """Signal the engine to stop taking new work and drain in-flight workers."""
        logger.info("Graceful shutdown requested for ReindexerEngine")
        self._stopping = True

    def register_signal_handlers(self) -> None:
        """Register SIGINT and SIGTERM handlers for graceful shutdown."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.stop)
            except (NotImplementedError, RuntimeError):
                pass

    def unregister_signal_handlers(self) -> None:
        """Unregister previously registered signal handlers."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.remove_signal_handler(sig)
            except (NotImplementedError, RuntimeError):
                pass

    async def _get_tag_id(self, tag_name: str) -> int:
        """Resolve tag ID, utilizing cached IDs if possible."""
        if tag_name == self.tag_success and self._tag_success_id is not None:
            return self._tag_success_id
        if tag_name == self.tag_failure and self._tag_failure_id is not None:
            return self._tag_failure_id

        tag_id = await self.paperless_client.get_or_create_tag(tag_name)
        if tag_name == self.tag_success:
            self._tag_success_id = tag_id
        elif tag_name == self.tag_failure:
            self._tag_failure_id = tag_id
        return tag_id

    async def _execute_ocr_with_backoff(
        self,
        original_bytes: bytes,
        filename: str,
        doc_id: int,
    ) -> tuple[bytes, dict[str, str], bool]:
        """Execute OCR request with 503 adaptive backoff retry.

        Returns:
            Tuple of (sandwich_pdf_bytes, headers_dict, retried_boolean).
        """
        await self._backoff_event.wait()
        try:
            content, headers = await self.rocm_client.generate_sandwich(
                original_bytes, filename
            )
            return content, headers, False
        except OcrServiceUnavailableError as exc:
            logger.warning(
                "Document %d received HTTP 503 from rocm-ocr: %s. Entering adaptive backoff.",
                doc_id,
                exc,
            )

        # 503 adaptive backoff: run retried document in isolation
        async with self._isolation_lock:
            self._backoff_event.clear()
            try:
                try:
                    await self.rocm_client.set_concurrency(1)
                    logger.info(
                        "Scaled rocm-ocr concurrency down to 1 for document %d retry",
                        doc_id,
                    )
                except Exception as e:
                    logger.warning("Failed to scale rocm-ocr concurrency to 1: %s", e)

                logger.info("Retrying OCR in isolation for document %d", doc_id)
                content, headers = await self.rocm_client.generate_sandwich(
                    original_bytes, filename
                )
                logger.info("Isolated retry succeeded for document %d", doc_id)
                return content, headers, True
            finally:
                try:
                    await self.rocm_client.set_concurrency(self.concurrency)
                    logger.info(
                        "Restored rocm-ocr concurrency to %d after retry",
                        self.concurrency,
                    )
                except Exception as e:
                    logger.warning("Failed to restore rocm-ocr concurrency: %s", e)
                self._backoff_event.set()

    async def _commit_success(
        self,
        doc_id: int,
        title: str | None,
        archive_filename: str,
        sandwich_bytes: bytes,
        old_checksum: str | None,
        original_size_bytes: int,
        headers: dict[str, str],
        document: dict[str, Any],
        retried: bool,
    ) -> DocumentResult:
        """Atomically persist sandwich PDF, update DB checksum, patch Paperless, and record success."""
        # 1. Write sandwich PDF to disk atomically
        self.file_manager.write_archive_file(archive_filename, sandwich_bytes)
        archive_size_bytes = len(sandwich_bytes)

        # 2. Update PostgreSQL archive_checksum
        new_checksum = self.file_manager.calculate_checksum(
            sandwich_bytes, existing_checksum=old_checksum
        )
        await self.db_adapter.update_archive_checksum(doc_id, new_checksum)

        # 3. Patch Paperless document tags and fulltext content if available
        success_tag_id = await self._get_tag_id(self.tag_success)
        failure_tag_id = await self._get_tag_id(self.tag_failure)

        existing_tags = set(document.get("tags") or [])
        if failure_tag_id in existing_tags:
            existing_tags.remove(failure_tag_id)
        existing_tags.add(success_tag_id)

        extracted_text = headers.get("X-Extracted-Text") or headers.get(
            "x-extracted-text"
        )

        await self.paperless_client.patch_document(
            doc_id,
            content=extracted_text,
            tags=sorted(existing_tags),
        )

        # 4. Extract metrics from headers
        inference_time_ms = 0.0
        for key in ("X-Inference-Time-Ms", "x-inference-time-ms"):
            if key in headers:
                try:
                    inference_time_ms = float(headers[key])
                    break
                except (ValueError, TypeError):
                    pass

        page_count = 1
        for key in ("X-Page-Count", "x-page-count", "X-Total-Pages", "x-total-pages"):
            if key in headers:
                try:
                    page_count = int(headers[key])
                    break
                except (ValueError, TypeError):
                    pass

        # 5. Record success in StateStore
        await self.state_store.record_success(
            document_id=doc_id,
            title=title,
            page_count=page_count,
            original_size_bytes=original_size_bytes,
            archive_size_bytes=archive_size_bytes,
            old_checksum=old_checksum,
            new_checksum=new_checksum,
            inference_time_ms=inference_time_ms,
        )

        return DocumentResult(
            document_id=doc_id,
            status="success",
            title=title,
            old_checksum=old_checksum,
            new_checksum=new_checksum,
            original_size_bytes=original_size_bytes,
            archive_size_bytes=archive_size_bytes,
            page_count=page_count,
            inference_time_ms=inference_time_ms,
            retried=retried,
        )

    async def _record_failure(
        self,
        doc_id: int,
        title: str | None,
        error_msg: str,
        document: dict[str, Any],
    ) -> None:
        """Record document failure in StateStore and tag with failure tag in Paperless."""
        try:
            await self.state_store.record_failure(
                doc_id, error_message=error_msg, title=title
            )
        except Exception as e:
            logger.error("Failed recording failure state for doc %d: %s", doc_id, e)

        try:
            failure_tag_id = await self._get_tag_id(self.tag_failure)
            success_tag_id = await self._get_tag_id(self.tag_success)

            existing_tags = set(document.get("tags") or [])
            if success_tag_id in existing_tags:
                existing_tags.remove(success_tag_id)
            existing_tags.add(failure_tag_id)

            await self.paperless_client.patch_document(
                doc_id, tags=sorted(existing_tags)
            )
        except Exception as e:
            logger.error("Failed patching failure tag for doc %d: %s", doc_id, e)

    async def _process_document_internal(
        self, document: dict[str, Any]
    ) -> DocumentResult:
        """Internal processing pipeline logic for a single document."""
        if "id" not in document:
            raise ValueError("Document dict missing 'id' key")

        doc_id = int(document["id"])
        title = document.get("title")

        # 1. Skip if already processed in state store
        if await self.state_store.is_processed(doc_id):
            logger.info("Document %d already processed/skipped, skipping", doc_id)
            return DocumentResult(
                document_id=doc_id,
                status="skipped",
                title=title,
                error="Already processed in state store",
            )

        # Record pending in state store
        await self.state_store.record_pending(doc_id, title=title)

        # 2. Fetch document archive info from Paperless database
        archive_info = await self.db_adapter.get_document_archive_info(doc_id)
        if archive_info is None:
            error_msg = f"Document {doc_id} not found in Paperless PostgreSQL database"
            logger.error(error_msg)
            await self._record_failure(doc_id, title, error_msg, document)
            return DocumentResult(
                document_id=doc_id,
                status="failed",
                title=title,
                error=error_msg,
            )

        archive_filename, old_checksum = archive_info
        if not archive_filename:
            archive_filename = f"{doc_id:07d}.pdf"

        # 3. Download original document bytes
        try:
            original_bytes = await self.paperless_client.download_original(doc_id)
        except Exception as exc:
            error_msg = f"Failed to download original document {doc_id}: {exc}"
            logger.error(error_msg)
            await self._record_failure(doc_id, title, error_msg, document)
            return DocumentResult(
                document_id=doc_id,
                status="failed",
                title=title,
                error=error_msg,
            )

        original_size_bytes = len(original_bytes)
        filename = document.get("original_file_name") or f"doc_{doc_id}.pdf"

        # 4. Perform OCR with adaptive backoff
        try:
            sandwich_bytes, headers, retried = await self._execute_ocr_with_backoff(
                original_bytes, filename, doc_id
            )
        except Exception as exc:
            error_msg = f"OCR failed for document {doc_id}: {exc}"
            logger.error(error_msg)
            await self._record_failure(doc_id, title, error_msg, document)
            return DocumentResult(
                document_id=doc_id,
                status="failed",
                title=title,
                error=error_msg,
                retried=isinstance(exc, OcrServiceUnavailableError),
            )

        # 5. On successful OCR, write file, update DB, patch Paperless, and record success
        try:
            return await asyncio.shield(
                self._commit_success(
                    doc_id=doc_id,
                    title=title,
                    archive_filename=archive_filename,
                    sandwich_bytes=sandwich_bytes,
                    old_checksum=old_checksum,
                    original_size_bytes=original_size_bytes,
                    headers=headers,
                    document=document,
                    retried=retried,
                )
            )
        except Exception as exc:
            error_msg = f"Failed committing OCR output for document {doc_id}: {exc}"
            logger.error(error_msg)
            await self._record_failure(doc_id, title, error_msg, document)
            return DocumentResult(
                document_id=doc_id,
                status="failed",
                title=title,
                error=error_msg,
                retried=retried,
            )

    async def process_document(self, document: dict[str, Any]) -> DocumentResult:
        """Process a single document through the reindexing pipeline.

        Acquires concurrency semaphore slot.
        """
        if not self._initialized:
            await self.initialize()

        async with self._semaphore:
            return await self._process_document_internal(document)

    async def process_batch(
        self,
        documents: list[dict[str, Any]],
        on_progress: Callable[[DocumentResult], Any] | None = None,
    ) -> list[DocumentResult]:
        """Process a batch of documents through the worker engine.

        Maintains order of results corresponding to the input documents.
        Handles cancellation and stop signals gracefully.
        """
        if not documents:
            return []

        if not self._initialized:
            await self.initialize()

        queue: asyncio.Queue[tuple[int, dict[str, Any]]] = asyncio.Queue()
        for idx, doc in enumerate(documents):
            queue.put_nowait((idx, doc))

        results: list[DocumentResult | None] = [None] * len(documents)
        worker_count = min(self.concurrency, len(documents))

        async def worker() -> None:
            while not self._stopping:
                try:
                    idx, doc = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                try:
                    res = await self.process_document(doc)
                    results[idx] = res
                    if on_progress is not None:
                        try:
                            on_progress(res)
                        except Exception:
                            logger.exception("Error in on_progress callback")
                finally:
                    queue.task_done()

        tasks = [asyncio.create_task(worker()) for _ in range(worker_count)]
        self._active_tasks.update(tasks)
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self.stop()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            self._active_tasks.difference_update(tasks)

        return [r for r in results if r is not None]


# Alias for ticket and spec terminology compatibility
ReindexWorkerPool = ReindexerEngine
