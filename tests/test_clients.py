"""Tests for PaperlessClient and RocmOcrClient HTTP integrations."""

import pytest
import respx

from paperless_rocm_reindexer.clients.paperless import PaperlessClient
from paperless_rocm_reindexer.clients.rocm_ocr import RocmOcrClient
from paperless_rocm_reindexer.exceptions import (
    DocumentNotFoundError,
    OcrClientError,
    OcrServiceUnavailableError,
    PaperlessApiError,
)

PAPERLESS_URL = "http://paperless.test:8000"
ROCM_URL = "http://rocm-ocr.test:8000"


class TestPaperlessClient:
    """Test suite for PaperlessClient."""

    @pytest.mark.asyncio
    async def test_get_document_success(self) -> None:
        async with respx.mock(base_url=PAPERLESS_URL) as respx_mock:
            respx_mock.get("/api/documents/42/").respond(
                status_code=200,
                json={"id": 42, "title": "Invoice 2026", "tags": [1, 2]},
            )

            async with PaperlessClient(PAPERLESS_URL, token="secret-token") as client:
                doc = await client.get_document(42)

            assert doc["id"] == 42
            assert doc["title"] == "Invoice 2026"

    @pytest.mark.asyncio
    async def test_get_document_not_found(self) -> None:
        async with respx.mock(base_url=PAPERLESS_URL) as respx_mock:
            respx_mock.get("/api/documents/999/").respond(status_code=404)

            async with PaperlessClient(PAPERLESS_URL, token="secret-token") as client:
                with pytest.raises(DocumentNotFoundError) as exc_info:
                    await client.get_document(999)
                assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_document_unauthorized(self) -> None:
        async with respx.mock(base_url=PAPERLESS_URL) as respx_mock:
            respx_mock.get("/api/documents/42/").respond(
                status_code=401, json={"detail": "Invalid token."}
            )

            async with PaperlessClient(PAPERLESS_URL, token="bad-token") as client:
                with pytest.raises(PaperlessApiError) as exc_info:
                    await client.get_document(42)
                assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_get_document_server_error(self) -> None:
        async with respx.mock(base_url=PAPERLESS_URL) as respx_mock:
            respx_mock.get("/api/documents/42/").respond(
                status_code=500, text="Internal Server Error"
            )

            async with PaperlessClient(PAPERLESS_URL, token="secret-token") as client:
                with pytest.raises(PaperlessApiError) as exc_info:
                    await client.get_document(42)
                assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_download_original_success(self) -> None:
        fake_bytes = b"%PDF-1.4 test original content"
        async with respx.mock(base_url=PAPERLESS_URL) as respx_mock:
            respx_mock.get("/api/documents/42/download/").respond(
                status_code=200, content=fake_bytes
            )

            async with PaperlessClient(PAPERLESS_URL, token="secret-token") as client:
                data = await client.download_original(42)

            assert data == fake_bytes

    @pytest.mark.asyncio
    async def test_download_original_not_found(self) -> None:
        async with respx.mock(base_url=PAPERLESS_URL) as respx_mock:
            respx_mock.get("/api/documents/999/download/").respond(status_code=404)

            async with PaperlessClient(PAPERLESS_URL, token="secret-token") as client:
                with pytest.raises(DocumentNotFoundError):
                    await client.download_original(999)

    @pytest.mark.asyncio
    async def test_patch_document_success(self) -> None:
        async with respx.mock(base_url=PAPERLESS_URL) as respx_mock:
            route = respx_mock.patch("/api/documents/42/").respond(
                status_code=200,
                json={"id": 42, "content": "Recognized text", "tags": [1, 5]},
            )

            async with PaperlessClient(PAPERLESS_URL, token="secret-token") as client:
                updated = await client.patch_document(
                    42, content="Recognized text", tags=[1, 5]
                )

            assert route.called
            assert updated["content"] == "Recognized text"
            assert updated["tags"] == [1, 5]

    @pytest.mark.asyncio
    async def test_get_or_create_tag_existing(self) -> None:
        async with respx.mock(base_url=PAPERLESS_URL) as respx_mock:
            respx_mock.get("/api/tags/").respond(
                status_code=200,
                json={"results": [{"id": 10, "name": "ocr:rocm-v4"}]},
            )

            async with PaperlessClient(PAPERLESS_URL, token="secret-token") as client:
                tag_id = await client.get_or_create_tag("ocr:rocm-v4")

            assert tag_id == 10

    @pytest.mark.asyncio
    async def test_get_or_create_tag_create_new(self) -> None:
        async with respx.mock(base_url=PAPERLESS_URL) as respx_mock:
            respx_mock.get("/api/tags/").respond(
                status_code=200,
                json={"results": []},
            )
            create_route = respx_mock.post("/api/tags/").respond(
                status_code=201,
                json={"id": 15, "name": "ocr:rocm-v4"},
            )

            async with PaperlessClient(PAPERLESS_URL, token="secret-token") as client:
                tag_id = await client.get_or_create_tag("ocr:rocm-v4")

            assert create_route.called
            assert tag_id == 15

    @pytest.mark.asyncio
    async def test_list_unprocessed_documents_with_pagination_and_filtering(
        self,
    ) -> None:
        async with respx.mock(base_url=PAPERLESS_URL) as respx_mock:
            # Tag queries for exclusion check
            respx_mock.get(
                path="/api/tags/", params={"name__iexact": "ocr:rocm-v4"}
            ).respond(
                status_code=200,
                json={"results": [{"id": 100, "name": "ocr:rocm-v4"}]},
            )
            respx_mock.get(
                path="/api/tags/", params={"name__iexact": "ocr:failed"}
            ).respond(
                status_code=200,
                json={"results": [{"id": 101, "name": "ocr:failed"}]},
            )

            # Page 1: docs 1 (unprocessed), 2 (already processed tag 100)
            respx_mock.get(path="/api/documents/", params={"page_size": "100"}).respond(
                status_code=200,
                json={
                    "next": "/api/documents/?page=2",
                    "results": [
                        {"id": 1, "title": "Doc 1", "tags": [5]},
                        {"id": 2, "title": "Doc 2", "tags": [100]},
                    ],
                },
            )
            # Page 2: doc 3 (failed tag 101), doc 4 (unprocessed)
            respx_mock.get(path="/api/documents/", params={"page": "2"}).respond(
                status_code=200,
                json={
                    "next": None,
                    "results": [
                        {"id": 3, "title": "Doc 3", "tags": [101]},
                        {"id": 4, "title": "Doc 4", "tags": []},
                    ],
                },
            )

            async with PaperlessClient(PAPERLESS_URL, token="secret-token") as client:
                docs = await client.list_unprocessed_documents()

            assert len(docs) == 2
            assert [d["id"] for d in docs] == [1, 4]

    @pytest.mark.asyncio
    async def test_list_unprocessed_documents_with_limit(self) -> None:
        async with respx.mock(base_url=PAPERLESS_URL) as respx_mock:
            respx_mock.get("/api/tags/").respond(
                status_code=200,
                json={"results": []},
            )
            respx_mock.get(path="/api/documents/", params={"page_size": "100"}).respond(
                status_code=200,
                json={
                    "next": "/api/documents/?page=2",
                    "results": [
                        {"id": 1, "title": "Doc 1", "tags": []},
                        {"id": 2, "title": "Doc 2", "tags": []},
                    ],
                },
            )

            async with PaperlessClient(PAPERLESS_URL, token="secret-token") as client:
                docs = await client.list_unprocessed_documents(limit=1)

            assert len(docs) == 1
            assert docs[0]["id"] == 1

    @pytest.mark.asyncio
    async def test_list_unprocessed_documents_with_server_tag_filtering(self) -> None:
        async with respx.mock(base_url=PAPERLESS_URL) as respx_mock:
            respx_mock.get("/api/tags/").respond(
                status_code=200,
                json={"results": [{"id": 42, "name": "ocr:rocm-v4"}]},
            )
            route = respx_mock.get(
                path="/api/documents/",
                params={"page_size": "100", "tags__id__none": "42"},
            ).respond(
                status_code=200,
                json={
                    "next": None,
                    "results": [
                        {"id": 99, "title": "Unprocessed Doc", "tags": []},
                    ],
                },
            )

            async with PaperlessClient(PAPERLESS_URL, token="secret-token") as client:
                docs = await client.list_unprocessed_documents(
                    tag_exclude="ocr:rocm-v4",
                    tag_exclude_fail="",
                    server_filter=True,
                )

            assert route.called
            assert len(docs) == 1
            assert docs[0]["id"] == 99


class TestRocmOcrClient:
    """Test suite for RocmOcrClient."""

    @pytest.mark.asyncio
    async def test_generate_sandwich_success(self) -> None:
        fake_pdf = b"%PDF-1.4 sandwich binary content"
        response_headers = {
            "Content-Type": "application/pdf",
            "X-Inference-Time-Ms": "145.2",
            "X-Extracted-Chars": "1200",
            "X-Total-Pages": "3",
            "X-Execution-Provider": "ROCMExecutionProvider",
        }

        async with respx.mock(base_url=ROCM_URL) as respx_mock:
            route = respx_mock.post("/ocr/v1/sandwich").respond(
                status_code=200,
                content=fake_pdf,
                headers=response_headers,
            )

            async with RocmOcrClient(ROCM_URL, api_key="test-ocr-key") as client:
                data, headers = await client.generate_sandwich(
                    b"%PDF-1.4 raw doc", "test.pdf"
                )

            assert route.called
            assert (
                route.calls.last.request.headers["authorization"]
                == "Bearer test-ocr-key"
            )
            assert data == fake_pdf
            assert headers.get("X-Extracted-Chars") == "1200"
            assert headers.get("X-Inference-Time-Ms") == "145.2"
            assert (
                headers.get("X-Total-Pages") == "3"
                or headers.get("X-Page-Count") == "3"
            )

    @pytest.mark.asyncio
    async def test_generate_sandwich_unauthorized_401(self) -> None:
        async with respx.mock(base_url=ROCM_URL) as respx_mock:
            respx_mock.post("/ocr/v1/sandwich").respond(
                status_code=401, json={"detail": "Invalid or missing API key"}
            )

            async with RocmOcrClient(ROCM_URL, api_key="wrong-key") as client:
                with pytest.raises(OcrClientError) as exc_info:
                    await client.generate_sandwich(b"pdf data", "test.pdf")
                assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_generate_sandwich_bad_request_404_or_400(self) -> None:
        async with respx.mock(base_url=ROCM_URL) as respx_mock:
            respx_mock.post("/ocr/v1/sandwich").respond(
                status_code=404, json={"detail": "Not found"}
            )

            async with RocmOcrClient(ROCM_URL) as client:
                with pytest.raises(OcrClientError) as exc_info:
                    await client.generate_sandwich(b"pdf data", "test.pdf")
                assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_generate_sandwich_server_error_500(self) -> None:
        async with respx.mock(base_url=ROCM_URL) as respx_mock:
            respx_mock.post("/ocr/v1/sandwich").respond(
                status_code=500, text="Internal crash"
            )

            async with RocmOcrClient(ROCM_URL) as client:
                with pytest.raises(OcrClientError) as exc_info:
                    await client.generate_sandwich(b"pdf data", "test.pdf")
                assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_generate_sandwich_gpu_oom_503(self) -> None:
        async with respx.mock(base_url=ROCM_URL) as respx_mock:
            respx_mock.post("/ocr/v1/sandwich").respond(
                status_code=503,
                json={
                    "detail": "HIP out of memory",
                    "provider_attempted": "ROCMExecutionProvider",
                },
            )

            async with RocmOcrClient(ROCM_URL) as client:
                with pytest.raises(OcrServiceUnavailableError) as exc_info:
                    await client.generate_sandwich(b"pdf data", "test.pdf")
                assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_set_concurrency_success(self) -> None:
        async with respx.mock(base_url=ROCM_URL) as respx_mock:
            respx_mock.post("/ocr/v1/setup").respond(
                status_code=200,
                json={"status": "updated", "concurrency": 2, "active_tasks": 0},
            )

            async with RocmOcrClient(ROCM_URL) as client:
                resp = await client.set_concurrency(2)

            assert resp["status"] == "updated"
            assert resp["concurrency"] == 2

    @pytest.mark.asyncio
    async def test_set_concurrency_error(self) -> None:
        async with respx.mock(base_url=ROCM_URL) as respx_mock:
            respx_mock.post("/ocr/v1/setup").respond(
                status_code=503, text="GPU Unavailable"
            )

            async with RocmOcrClient(ROCM_URL) as client:
                with pytest.raises(OcrServiceUnavailableError) as exc_info:
                    await client.set_concurrency(1)
                assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_set_concurrency_clamping(self) -> None:
        async with respx.mock(base_url=ROCM_URL) as respx_mock:
            route = respx_mock.post("/ocr/v1/setup").respond(
                status_code=200,
                json={"status": "updated", "concurrency": 8},
            )

            async with RocmOcrClient(ROCM_URL) as client:
                # Concurrency 16 should be clamped to 8
                await client.set_concurrency(16)
                assert route.calls.last.request.content == b'{"concurrency":8}'

                # Concurrency 0 should be clamped to 1
                await client.set_concurrency(0)
                assert route.calls.last.request.content == b'{"concurrency":1}'

    @pytest.mark.asyncio
    async def test_get_setup_success(self) -> None:
        async with respx.mock(base_url=ROCM_URL) as respx_mock:
            respx_mock.get("/ocr/v1/setup").respond(
                status_code=200,
                json={"concurrency": 4, "active_tasks": 0, "device": "cuda:0"},
            )

            async with RocmOcrClient(ROCM_URL) as client:
                setup_data = await client.get_setup()

            assert setup_data["concurrency"] == 4
            assert setup_data["device"] == "cuda:0"

    @pytest.mark.asyncio
    async def test_get_setup_fallback_on_404_or_405(self) -> None:
        async with respx.mock(base_url=ROCM_URL) as respx_mock:
            respx_mock.get("/ocr/v1/setup").respond(status_code=405)

            async with RocmOcrClient(ROCM_URL) as client:
                setup_data = await client.get_setup()

            assert setup_data == {}
