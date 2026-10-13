"""Asynchronous HTTP client for rocm-ocr microservice."""

from typing import Any

import httpx

from paperless_rocm_reindexer.exceptions import (
    OcrClientError,
    OcrServiceUnavailableError,
)


class RocmOcrClient:
    """Async client interacting with rocm-ocr microservice."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        client: httpx.AsyncClient | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self._external_client = client is not None
        self._client = client or httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client session."""
        if not self._external_client:
            await self._client.aclose()

    async def __aenter__(self) -> "RocmOcrClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.aclose()

    def _handle_response_error(self, response: httpx.Response, action: str) -> None:
        """Raise appropriate exceptions based on response status."""
        if response.status_code == 503:
            raise OcrServiceUnavailableError(
                f"rocm-ocr service unavailable (GPU OOM / HIP error) during {action}: {response.text}"
            )
        if response.is_error:
            raise OcrClientError(
                f"rocm-ocr error during {action} (HTTP {response.status_code}): {response.text}",
                status_code=response.status_code,
            )

    async def generate_sandwich(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> tuple[bytes, dict[str, str]]:
        """Submit PDF or image to rocm-ocr and receive searchable Sandwich PDF.

        Args:
            file_bytes: Raw binary content of document.
            filename: Original file name.

        Returns:
            Tuple of (sandwich_pdf_bytes, metadata_headers_dict).
            Headers include:
            - X-Extracted-Chars
            - X-Inference-Time-Ms
            - X-Total-Pages (or X-Page-Count)
            - X-Execution-Provider
        """
        files = {
            "file": (filename, file_bytes, "application/pdf"),
        }
        response = await self._client.post(
            "/ocr/v1/sandwich",
            files=files,
        )
        self._handle_response_error(response, f"generate_sandwich({filename})")

        headers = {k: v for k, v in response.headers.items()}
        # Ensure canonical casing and normalized aliases
        for k, v in list(headers.items()):
            canonical = "-".join(part.capitalize() for part in k.split("-"))
            headers[canonical] = v

        if "x-total-pages" in headers:
            headers["x-page-count"] = headers["x-total-pages"]
            headers["X-Page-Count"] = headers["x-total-pages"]

        return response.content, headers

    async def set_concurrency(self, target_concurrency: int) -> dict[str, Any]:
        """Dynamically adjust concurrency capacity of rocm-ocr engine.

        Args:
            target_concurrency: Desired concurrency slot count (>= 1).

        Returns:
            Setup response payload dictionary.
        """
        response = await self._client.post(
            "/ocr/v1/setup",
            json={"concurrency": target_concurrency},
        )
        self._handle_response_error(response, f"set_concurrency({target_concurrency})")
        data: dict[str, Any] = response.json()
        return data
