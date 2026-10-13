"""Asynchronous HTTP client for Paperless-ngx REST API."""

from typing import Any

import httpx

from paperless_rocm_reindexer.exceptions import (
    DocumentNotFoundError,
    PaperlessApiError,
)


class PaperlessClient:
    """Async client interacting with Paperless-ngx REST API."""

    def __init__(
        self,
        base_url: str,
        token: str,
        client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        headers = {
            "Authorization": f"Token {self.token}",
            "Accept": "application/json; version=5",
        }
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

    async def __aenter__(self) -> "PaperlessClient":
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
        if response.status_code == 404:
            raise DocumentNotFoundError(
                f"Resource not found ({action}): {response.status_code}"
            )
        if response.is_error:
            raise PaperlessApiError(
                f"Paperless API error during {action} (HTTP {response.status_code}): {response.text}",
                status_code=response.status_code,
            )

    async def get_document(self, document_id: int) -> dict[str, Any]:
        """Fetch metadata for a single document by ID.

        Args:
            document_id: Numeric ID of document.

        Returns:
            Dictionary containing document representation from Paperless.
        """
        response = await self._client.get(f"/api/documents/{document_id}/")
        self._handle_response_error(response, f"get_document({document_id})")
        data: dict[str, Any] = response.json()
        return data

    async def download_original(self, document_id: int) -> bytes:
        """Download pristine original binary file for a document.

        Args:
            document_id: Numeric ID of document.

        Returns:
            Raw bytes of the original file.
        """
        response = await self._client.get(
            f"/api/documents/{document_id}/download/",
            params={"original": "true"},
        )
        self._handle_response_error(response, f"download_original({document_id})")
        return response.content

    async def patch_document(
        self,
        document_id: int,
        content: str | None = None,
        tags: list[int] | None = None,
    ) -> dict[str, Any]:
        """Patch document content or tags.

        Args:
            document_id: Numeric ID of document.
            content: New extracted fulltext content.
            tags: List of tag IDs associated with document.

        Returns:
            Updated document metadata dictionary.
        """
        payload: dict[str, Any] = {}
        if content is not None:
            payload["content"] = content
        if tags is not None:
            payload["tags"] = tags

        response = await self._client.patch(
            f"/api/documents/{document_id}/",
            json=payload,
        )
        self._handle_response_error(response, f"patch_document({document_id})")
        data: dict[str, Any] = response.json()
        return data

    async def get_or_create_tag(self, tag_name: str) -> int:
        """Retrieve existing tag ID or create a new tag if not found.

        Args:
            tag_name: The name of the tag (e.g. 'ocr:rocm-v4').

        Returns:
            Tag ID integer.
        """
        search_resp = await self._client.get(
            "/api/tags/",
            params={"name__iexact": tag_name},
        )
        self._handle_response_error(search_resp, f"search_tag({tag_name})")
        search_data = search_resp.json()
        results: list[dict[str, Any]] = search_data.get("results", [])

        for tag in results:
            if tag.get("name", "").lower() == tag_name.lower():
                tag_id: int = tag["id"]
                return tag_id

        create_resp = await self._client.post(
            "/api/tags/",
            json={"name": tag_name},
        )
        self._handle_response_error(create_resp, f"create_tag({tag_name})")
        create_data: dict[str, Any] = create_resp.json()
        new_tag_id: int = create_data["id"]
        return new_tag_id

    async def list_unprocessed_documents(
        self,
        tag_exclude: str = "ocr:rocm-v4",
        tag_exclude_fail: str = "ocr:failed",
        limit: int = 0,
    ) -> list[dict[str, Any]]:
        """List documents needing OCR processing, excluding processed/failed tags.

        Handles pagination transparently until limit is reached or all pages are read.

        Args:
            tag_exclude: Tag name marking successfully processed documents.
            tag_exclude_fail: Tag name marking failed documents.
            limit: Maximum number of documents to return (0 = all).

        Returns:
            List of document dictionaries.
        """
        exclude_tag_ids: set[int] = set()

        for tag_name in (tag_exclude, tag_exclude_fail):
            if not tag_name:
                continue
            resp = await self._client.get(
                "/api/tags/",
                params={"name__iexact": tag_name},
            )
            if not resp.is_error:
                data = resp.json()
                for tag in data.get("results", []):
                    if tag.get("name", "").lower() == tag_name.lower():
                        exclude_tag_ids.add(tag["id"])

        documents: list[dict[str, Any]] = []
        url: str | None = "/api/documents/"
        params: dict[str, Any] | None = {"page_size": 100}

        while url:
            response = await self._client.get(url, params=params)
            self._handle_response_error(response, "list_unprocessed_documents")
            data = response.json()
            results: list[dict[str, Any]] = data.get("results", [])

            for doc in results:
                doc_tags = set(doc.get("tags", []))
                if not (doc_tags & exclude_tag_ids):
                    documents.append(doc)
                    if 0 < limit <= len(documents):
                        return documents

            url = data.get("next")
            params = None  # URL contains next query parameters

        return documents
