"""paperless-rocm-reindexer package."""

from paperless_rocm_reindexer.config import Settings, get_settings
from paperless_rocm_reindexer.engine import (
    DocumentResult,
    ReindexerEngine,
    ReindexWorkerPool,
)
from paperless_rocm_reindexer.models import (
    MetricRecord,
    SuccessRecord,
    parse_ocr_headers,
)
from paperless_rocm_reindexer.state import StateStore

__version__ = "0.1.0"
__all__ = [
    "DocumentResult",
    "MetricRecord",
    "ReindexerEngine",
    "ReindexWorkerPool",
    "Settings",
    "StateStore",
    "SuccessRecord",
    "__version__",
    "get_settings",
    "parse_ocr_headers",
]
