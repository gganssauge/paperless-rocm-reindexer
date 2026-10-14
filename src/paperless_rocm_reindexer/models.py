"""Domain models and records for OCR processing."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Any, Mapping

import pypdf

logger = logging.getLogger(__name__)


@dataclass
class MetricRecord:
    """Metrics extracted from rocm-ocr response headers."""

    page_count: int = 1
    inference_time_ms: float = 0.0
    extracted_chars: int = 0
    extracted_text: str | None = None
    execution_provider: str = "RapidOCR / ROCm PP-OCRv4"


def parse_ocr_headers(
    headers: Mapping[str, str],
    default_elapsed_ms: float | None = None,
) -> MetricRecord:
    """Parse and normalize rocm-ocr metadata headers into a MetricRecord.

    Args:
        headers: Case-insensitive or dictionary headers from response.
        default_elapsed_ms: Fallback inference time in ms if not present in headers.

    Returns:
        Structured MetricRecord.
    """
    norm = {k.lower(): v for k, v in headers.items()}
    text = norm.get("x-extracted-text")
    inf_time = _parse_float(norm.get("x-inference-time-ms"), default_elapsed_ms or 0.0)
    pages = _parse_int(norm.get("x-page-count") or norm.get("x-total-pages"), 1)
    chars = _parse_int(norm.get("x-extracted-chars"), len(text) if text else 0)
    provider = norm.get("x-execution-provider", "RapidOCR / ROCm PP-OCRv4")

    return MetricRecord(
        page_count=pages,
        inference_time_ms=inf_time,
        extracted_chars=chars,
        extracted_text=text,
        execution_provider=provider,
    )


def extract_text_from_pdf(pdf_bytes: bytes) -> str | None:
    """Extract text layer from PDF bytes using pypdf.

    Args:
        pdf_bytes: Raw binary content of PDF.

    Returns:
        Extracted text string if successful and non-empty, otherwise None.
    """
    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        pages_text = [page.extract_text() or "" for page in reader.pages]
        combined = "\n".join(pages_text).strip()
        return combined if combined else None
    except Exception as exc:
        logger.debug("pypdf text extraction skipped: %s", exc)
        return None


def _parse_float(value: Any, default: float) -> float:
    """Safely parse float value with fallback."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _parse_int(value: Any, default: int) -> int:
    """Safely parse int value with fallback."""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


@dataclass
class SuccessRecord:
    """Record containing data required to commit a successful OCR job."""

    doc_id: int
    title: str | None
    archive_filename: str
    sandwich_bytes: bytes
    old_checksum: str | None
    original_size_bytes: int
    metrics: MetricRecord
    document: dict[str, Any]
    retried: bool = False
