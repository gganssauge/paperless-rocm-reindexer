"""Custom exceptions for API clients and reindexing operations."""


class ReindexerError(Exception):
    """Base exception for paperless-rocm-reindexer."""


class PaperlessApiError(ReindexerError):
    """Raised when Paperless-ngx REST API returns an unexpected error."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class DocumentNotFoundError(PaperlessApiError):
    """Raised when a requested document does not exist in Paperless-ngx (HTTP 404)."""

    def __init__(self, message: str = "Document not found in Paperless-ngx") -> None:
        super().__init__(message, status_code=404)


class OcrClientError(ReindexerError):
    """Base exception for rocm-ocr client interactions."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class OcrServiceUnavailableError(OcrClientError):
    """Raised when rocm-ocr service returns HTTP 503 (GPU OOM / HIP failure)."""

    def __init__(
        self,
        message: str = "rocm-ocr service is unavailable (GPU OOM or provider failure)",
    ) -> None:
        super().__init__(message, status_code=503)
