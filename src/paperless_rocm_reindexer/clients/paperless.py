from typing import Any, Final

import httpx

from paperless_rocm_reindexer.exceptions import (
    DocumentNotFoundError,
    PaperlessApiError,
)

TAGS_ENDPOINT: Final[str] = "/api/tags/"


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
            "Accept": "application/json",
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
        return response.json()

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
        return response.json()

    async def get_or_create_tag(self, tag_name: str) -> int:
        """Retrieve existing tag ID or create a new tag if not found.

        Args:
            tag_name: The name of the tag (e.g. 'ocr:rocm-v4').

        Returns:
            Tag ID integer.
        """
        search_resp = await self._client.get(
            TAGS_ENDPOINT,
            params={"name__iexact": tag_name},
        )
        self._handle_response_error(search_resp, f"search_tag({tag_name})")
        for tag in search_resp.json().get("results", []):
            if tag.get("name", "").lower() == tag_name.lower():
                return tag["id"]

        create_resp = await self._client.post(
            TAGS_ENDPOINT,
            json={"name": tag_name},
        )
        self._handle_response_error(create_resp, f"create_tag({tag_name})")
        return create_resp.json()["id"]

    async def _resolve_tag_ids(self, tag_names: tuple[str, ...]) -> set[int]:
        """Resolve numeric tag IDs for specified tag names."""
        exclude_tag_ids: set[int] = set()
        for name in tag_names:
            if not name:
                continue
            resp = await self._client.get(TAGS_ENDPOINT, params={"name__iexact": name})
            if not resp.is_error:
                for tag in resp.json().get("results", []):
                    if tag.get("name", "").lower() == name.lower():
                        exclude_tag_ids.add(tag["id"])
        return exclude_tag_ids

    def _build_list_params(
        self,
        exclude_tag_ids: set[int],
        server_filter: bool,
    ) -> dict[str, Any]:
        """Build query parameters including optional server-side tag exclusion."""
        params: dict[str, Any] = {"page_size": 100}
        if server_filter and exclude_tag_ids:
            params["tags__id__none"] = ",".join(str(t) for t in sorted(exclude_tag_ids))
        return params

    def _filter_page_documents(
        self,
        docs: list[dict[str, Any]],
        exclude_tag_ids: set[int],
        limit: int,
        documents: list[dict[str, Any]],
    ) -> bool:
        """Filter and collect documents, returning True if limit is reached."""
        for doc in docs:
            doc_tags = set(doc.get("tags", []))
            if not (doc_tags & exclude_tag_ids):
                documents.append(doc)
                if 0 < limit <= len(documents):
                    return True
        return False

    async def list_unprocessed_documents(
        self,
        tag_exclude: str = "ocr:rocm-v4",
        tag_exclude_fail: str = "ocr:failed",
        limit: int = 0,
        server_filter: bool = True,
    ) -> list[dict[str, Any]]:
        """List documents needing OCR processing, excluding processed/failed tags.

        Handles pagination transparently until limit is reached or all pages are read.
        Supports server-side tag exclusion filtering via query params when available.

        Args:
            tag_exclude: Tag name marking successfully processed documents.
            tag_exclude_fail: Tag name marking failed documents.
            limit: Maximum number of documents to return (0 = all).
            server_filter: If True, send tags__id__none filter parameter to Paperless API.

        Returns:
            List of document dictionaries.
        """
        exclude_ids = await self._resolve_tag_ids((tag_exclude, tag_exclude_fail))
        documents: list[dict[str, Any]] = []
        url: str | None = "/api/documents/"
        params: dict[str, Any] | None = self._build_list_params(
            exclude_ids, server_filter
        )

        while url:
            response = await self._client.get(url, params=params)
            self._handle_response_error(response, "list_unprocessed_documents")
            data = response.json()
            if self._filter_page_documents(
                data.get("results", []), exclude_ids, limit, documents
            ):
                return documents

            url = data.get("next")
            params = None

        return documents
