"""Clients package for paperless-rocm-reindexer."""

from paperless_rocm_reindexer.clients.paperless import PaperlessClient
from paperless_rocm_reindexer.clients.rocm_ocr import RocmOcrClient

__all__ = ["PaperlessClient", "RocmOcrClient"]
