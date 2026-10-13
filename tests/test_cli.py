"""Unit and integration tests for Paperless ROCm Reindexer CLI and configuration."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from paperless_rocm_reindexer.cli import (
    app,
    display_benchmarks,
    display_diff,
    run_async,
)
from paperless_rocm_reindexer.config import Settings, get_settings
from paperless_rocm_reindexer.engine import DocumentResult
from paperless_rocm_reindexer.state import StateStore


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing Typer CliRunner."""
    return CliRunner()


# --- Configuration Tests ---


def test_settings_defaults() -> None:
    """Verify default Settings values comply with spec."""
    s = Settings()
    assert s.PAPERLESS_BASE_URL == "http://paperless-web:8000"
    assert s.PAPERLESS_API_TOKEN == ""
    assert s.PAPERLESS_MEDIA_ROOT == Path("/media")
    assert s.PAPERLESS_DB_URL == ""
    assert s.ROCM_OCR_BASE_URL == "http://rocm-ocr:8000"
    assert s.ROCM_OCR_API_KEY == ""
    assert s.OCR_CONCURRENCY == 4
    assert s.BATCH_LIMIT == 0
    assert s.TAG_SUCCESS == "ocr:rocm-v4"
    assert s.TAG_FAILURE == "ocr:failed"
    assert s.DAEMON_INTERVAL_SEC == 0
    assert s.DRY_RUN is False
    assert s.STATE_DB_PATH == Path("/data/reindexer_state.db")


def test_settings_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Settings loads overrides from environment variables."""
    monkeypatch.setenv("PAPERLESS_BASE_URL", "http://custom-paperless:9000")
    monkeypatch.setenv("PAPERLESS_API_TOKEN", "secret-token-123")
    monkeypatch.setenv("OCR_CONCURRENCY", "8")
    monkeypatch.setenv("BATCH_LIMIT", "50")
    monkeypatch.setenv("DRY_RUN", "true")
    monkeypatch.setenv("STATE_DB_PATH", "/tmp/custom_state.db")

    s = Settings()
    assert s.PAPERLESS_BASE_URL == "http://custom-paperless:9000"
    assert s.PAPERLESS_API_TOKEN == "secret-token-123"
    assert s.OCR_CONCURRENCY == 8
    assert s.BATCH_LIMIT == 50
    assert s.DRY_RUN is True
    assert s.STATE_DB_PATH == Path("/tmp/custom_state.db")


def test_get_settings_cached() -> None:
    """Verify get_settings returns a valid Settings instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert isinstance(s1, Settings)
    assert s1 is s2


# --- CLI Runner Helper Tests ---


def test_run_async_in_sync_context() -> None:
    """Verify run_async works when no event loop is active."""

    async def sample() -> int:
        return 42

    result = run_async(sample())
    assert result == 42


async def test_run_async_in_async_context() -> None:
    """Verify run_async works when called from inside an active event loop."""

    async def sample() -> str:
        await asyncio.sleep(0.01)
        return "async_ok"

    result = run_async(sample())
    assert result == "async_ok"


# --- CLI Help & Validation Tests ---


def test_cli_help(runner: CliRunner) -> None:
    """Verify root help command displays all subcommands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "batch" in result.stdout
    assert "inspect" in result.stdout
    assert "daemon" in result.stdout
    assert "stats" in result.stdout


@pytest.mark.parametrize("subcmd", ["batch", "inspect", "daemon", "stats"])
def test_cli_subcommand_help(runner: CliRunner, subcmd: str) -> None:
    """Verify help message for each subcommand."""
    result = runner.invoke(app, [subcmd, "--help"])
    assert result.exit_code == 0


def test_inspect_missing_doc_id(runner: CliRunner) -> None:
    """Verify inspect command fails when --doc-id is omitted."""
    result = runner.invoke(app, ["inspect"])
    assert result.exit_code == 1
    assert "Error: --doc-id is required" in result.stdout


# --- CLI Inspect Tests ---


def test_inspect_success(runner: CliRunner, tmp_path: Path) -> None:
    """Verify inspect command displays visual diff and inference benchmarks."""
    mock_doc = {
        "id": 42,
        "title": "Invoice #42",
        "content": "Line 1: Old invoice text\nLine 2: Subtotal $100",
        "original_file_name": "invoice_42.pdf",
    }
    mock_headers = {
        "X-Extracted-Text": "Line 1: Old invoice text\nLine 2: Subtotal $100\nLine 3: Total $119",
        "X-Inference-Time-Ms": "48.5",
        "X-Extracted-Chars": "65",
        "X-Page-Count": "1",
        "X-Execution-Provider": "RapidOCR ROCm",
    }
    mock_pdf_bytes = b"%PDF-1.4 sandwich binary content"

    mock_p_client = AsyncMock()
    mock_p_client.get_document.return_value = mock_doc
    mock_p_client.download_original.return_value = b"%PDF-1.4 original"

    mock_r_client = AsyncMock()
    mock_r_client.generate_sandwich.return_value = (mock_pdf_bytes, mock_headers)

    save_path = tmp_path / "sandwich_out.pdf"

    with (
        patch("paperless_rocm_reindexer.cli.PaperlessClient") as p_cls,
        patch("paperless_rocm_reindexer.cli.RocmOcrClient") as r_cls,
    ):
        p_cls.return_value.__aenter__.return_value = mock_p_client
        r_cls.return_value.__aenter__.return_value = mock_r_client

        result = runner.invoke(
            app,
            [
                "inspect",
                "--doc-id",
                "42",
                "--save-sandwich",
                str(save_path),
            ],
        )

    assert result.exit_code == 0
    assert "Invoice #42" in result.stdout
    assert "Inference Benchmarks" in result.stdout
    assert "48.5" in result.stdout
    assert "RapidOCR ROCm" in result.stdout
    assert save_path.exists()
    assert save_path.read_bytes() == mock_pdf_bytes


def test_inspect_identical_content(runner: CliRunner) -> None:
    """Verify inspect displays matching message when text is identical."""
    matching_text = "Sample unchanged content"
    mock_doc = {
        "id": 10,
        "title": "Doc 10",
        "content": matching_text,
    }
    mock_headers = {
        "X-Extracted-Text": matching_text,
        "X-Inference-Time-Ms": "15.0",
        "X-Extracted-Chars": str(len(matching_text)),
        "X-Page-Count": "1",
    }

    mock_p_client = AsyncMock()
    mock_p_client.get_document.return_value = mock_doc
    mock_p_client.download_original.return_value = b"%PDF-1.4"

    mock_r_client = AsyncMock()
    mock_r_client.generate_sandwich.return_value = (b"%PDF-1.4", mock_headers)

    with (
        patch("paperless_rocm_reindexer.cli.PaperlessClient") as p_cls,
        patch("paperless_rocm_reindexer.cli.RocmOcrClient") as r_cls,
    ):
        p_cls.return_value.__aenter__.return_value = mock_p_client
        r_cls.return_value.__aenter__.return_value = mock_r_client

        result = runner.invoke(app, ["inspect", "--doc-id", "10"])

    assert result.exit_code == 0
    assert "No textual diff" in result.stdout


def test_inspect_error_handling(runner: CliRunner) -> None:
    """Verify inspect exits with code 1 when Paperless raises an error."""
    mock_p_client = AsyncMock()
    mock_p_client.get_document.side_effect = RuntimeError("Paperless unreachable")

    with patch("paperless_rocm_reindexer.cli.PaperlessClient") as p_cls:
        p_cls.return_value.__aenter__.return_value = mock_p_client

        result = runner.invoke(app, ["inspect", "--doc-id", "999"])

    assert result.exit_code == 1
    assert "Inspect error:" in result.stdout


# --- CLI Batch Tests ---


def test_batch_no_unprocessed_documents(runner: CliRunner) -> None:
    """Verify batch exits gracefully when no documents require reindexing."""
    mock_p_client = AsyncMock()
    mock_p_client.list_unprocessed_documents.return_value = []

    mock_r_client = AsyncMock()

    with (
        patch("paperless_rocm_reindexer.cli.PaperlessClient") as p_cls,
        patch("paperless_rocm_reindexer.cli.RocmOcrClient") as r_cls,
    ):
        p_cls.return_value.__aenter__.return_value = mock_p_client
        r_cls.return_value.__aenter__.return_value = mock_r_client

        result = runner.invoke(app, ["batch"])

    assert result.exit_code == 0
    assert "No unprocessed documents found" in result.stdout


def test_batch_dry_run(runner: CliRunner) -> None:
    """Verify batch --dry-run processes through OCR without touching DB or files."""
    docs = [
        {"id": 1, "title": "Doc 1", "original_file_name": "doc1.pdf"},
        {"id": 2, "title": "Doc 2", "original_file_name": "doc2.pdf"},
    ]
    mock_headers = {
        "X-Extracted-Text": "Extracted text sample",
        "X-Extracted-Chars": "21",
        "X-Page-Count": "1",
        "X-Inference-Time-Ms": "32.0",
    }

    mock_p_client = AsyncMock()
    mock_p_client.list_unprocessed_documents.return_value = docs
    mock_p_client.download_original.return_value = b"%PDF-1.4 original"

    mock_r_client = AsyncMock()
    mock_r_client.generate_sandwich.return_value = (b"%PDF-sandwich", mock_headers)

    with (
        patch("paperless_rocm_reindexer.cli.PaperlessClient") as p_cls,
        patch("paperless_rocm_reindexer.cli.RocmOcrClient") as r_cls,
        patch("paperless_rocm_reindexer.cli.PaperlessDatabaseAdapter") as db_cls,
    ):
        p_cls.return_value.__aenter__.return_value = mock_p_client
        r_cls.return_value.__aenter__.return_value = mock_r_client

        result = runner.invoke(app, ["batch", "--dry-run", "--concurrency", "2"])

        # Database adapter must NOT be called in dry run mode
        db_cls.assert_not_called()

    assert result.exit_code == 0
    assert "Executing DRY-RUN" in result.stdout
    assert "Dry-Run Execution Results" in result.stdout
    assert "Doc 1" in result.stdout
    assert "Doc 2" in result.stdout
    assert "Dry-run finished" in result.stdout


def test_batch_normal_run(runner: CliRunner) -> None:
    """Verify batch normal run executes ReindexerEngine and reports progress summary."""
    docs = [
        {"id": 101, "title": "Doc 101", "original_file_name": "doc101.pdf"},
        {"id": 102, "title": "Doc 102", "original_file_name": "doc102.pdf"},
    ]

    mock_p_client = AsyncMock()
    mock_p_client.list_unprocessed_documents.return_value = docs

    mock_r_client = AsyncMock()

    mock_db = AsyncMock()
    mock_store = AsyncMock()

    results = [
        DocumentResult(document_id=101, status="success", title="Doc 101"),
        DocumentResult(document_id=102, status="success", title="Doc 102"),
    ]

    async def mock_process_batch(docs_in, on_progress=None):
        for res in results:
            if on_progress:
                on_progress(res)
        return results

    mock_engine = MagicMock()
    mock_engine.__aenter__ = AsyncMock(return_value=mock_engine)
    mock_engine.__aexit__ = AsyncMock(return_value=None)
    mock_engine.close = AsyncMock()
    mock_engine.process_batch = AsyncMock(side_effect=mock_process_batch)

    with (
        patch("paperless_rocm_reindexer.cli.PaperlessClient") as p_cls,
        patch("paperless_rocm_reindexer.cli.RocmOcrClient") as r_cls,
        patch("paperless_rocm_reindexer.cli.PaperlessDatabaseAdapter") as db_cls,
        patch("paperless_rocm_reindexer.cli.StateStore") as store_cls,
        patch("paperless_rocm_reindexer.cli.ReindexerEngine") as engine_cls,
    ):
        p_cls.return_value.__aenter__.return_value = mock_p_client
        r_cls.return_value.__aenter__.return_value = mock_r_client
        db_cls.return_value = mock_db
        store_cls.return_value = mock_store
        engine_cls.return_value = mock_engine

        result = runner.invoke(app, ["batch", "--concurrency", "2", "--limit", "10"])

    assert result.exit_code == 0
    assert "Starting batch reindex" in result.stdout
    assert "Batch Reindex Summary" in result.stdout
    assert "Total Processed" in result.stdout


# --- CLI Stats Tests ---


async def test_stats_command(runner: CliRunner, tmp_path: Path) -> None:
    """Verify stats command reads StateStore and prints formatted table."""
    db_file = tmp_path / "test_stats.db"
    store = StateStore(db_file)
    await store.initialize()
    await store.record_success(1, title="Success 1")
    await store.record_success(2, title="Success 2")
    await store.record_failure(3, error_message="OOM", title="Fail 3")
    await store.record_skipped(4, reason="already done")
    await store.record_pending(5, title="Pending 5")
    await store.close()

    result = runner.invoke(app, ["stats", "--state-db", str(db_file)])

    assert result.exit_code == 0
    assert "Reindex Processing State Summary" in result.stdout
    assert "Success" in result.stdout
    assert "Failed" in result.stdout
    assert "Skipped" in result.stdout
    assert "Pending" in result.stdout
    assert "Total" in result.stdout


# --- CLI Daemon Tests ---


def test_daemon_command(runner: CliRunner) -> None:
    """Verify daemon executes polling iterations and stops cleanly."""
    with patch("paperless_rocm_reindexer.cli.run_batch") as mock_batch:
        mock_batch.return_value = None

        result = runner.invoke(
            app,
            [
                "daemon",
                "--interval-sec",
                "1",
                "--concurrency",
                "2",
                "--max-iterations",
                "2",
            ],
        )

    assert result.exit_code == 0
    assert "Starting Reindexer Daemon" in result.stdout
    assert "Daemon polling iteration #1" in result.stdout
    assert "Daemon polling iteration #2" in result.stdout
    assert "Daemon stopped" in result.stdout


# --- Direct Helper Functions Tests ---


def test_display_diff_and_benchmarks() -> None:
    """Verify display_diff and display_benchmarks execute without exceptions."""
    import io

    from rich.console import Console

    buf = io.StringIO()
    test_console = Console(file=buf, force_terminal=False)

    display_diff(
        old_text="Old invoice text\nAmount: $10",
        new_text="New invoice text\nAmount: $10\nTax: $2",
        doc_id=55,
        out_console=test_console,
    )

    display_benchmarks(
        doc_id=55,
        title="Test Doc",
        headers={
            "X-Inference-Time-Ms": "42.0",
            "X-Extracted-Chars": "35",
            "X-Page-Count": "1",
        },
        elapsed_ms=42.0,
        out_console=test_console,
    )

    output = buf.getvalue()
    assert "Text Overview - Document 55" in output
    assert "Visual Text Diff" in output
    assert "Inference Benchmarks - Document 55" in output


def test_inspect_positional_argument(runner: CliRunner) -> None:
    """Verify inspect supports passing doc_id as positional argument."""
    mock_doc = {"id": 88, "title": "Doc 88", "content": "Text"}
    mock_headers = {"X-Extracted-Text": "Text", "X-Inference-Time-Ms": "10"}

    mock_p_client = AsyncMock()
    mock_p_client.get_document.return_value = mock_doc
    mock_p_client.download_original.return_value = b"%PDF-1.4"

    mock_r_client = AsyncMock()
    mock_r_client.generate_sandwich.return_value = (b"%PDF", mock_headers)

    with (
        patch("paperless_rocm_reindexer.cli.PaperlessClient") as p_cls,
        patch("paperless_rocm_reindexer.cli.RocmOcrClient") as r_cls,
    ):
        p_cls.return_value.__aenter__.return_value = mock_p_client
        r_cls.return_value.__aenter__.return_value = mock_r_client

        result = runner.invoke(app, ["inspect", "88"])

    assert result.exit_code == 0
    assert "Doc 88" in result.stdout


def test_batch_dry_run_with_ocr_error(runner: CliRunner) -> None:
    """Verify dry-run captures document OCR failure and continues."""
    docs = [
        {"id": 1, "title": "Doc 1", "original_file_name": "doc1.pdf"},
        {"id": 2, "title": "Doc 2", "original_file_name": "doc2.pdf"},
    ]

    mock_p_client = AsyncMock()
    mock_p_client.list_unprocessed_documents.return_value = docs
    mock_p_client.download_original.side_effect = [
        b"%PDF",
        RuntimeError("Corrupt original"),
    ]

    mock_r_client = AsyncMock()
    mock_r_client.generate_sandwich.return_value = (
        b"%PDF-sandwich",
        {"X-Extracted-Chars": "10", "X-Inference-Time-Ms": "20"},
    )

    with (
        patch("paperless_rocm_reindexer.cli.PaperlessClient") as p_cls,
        patch("paperless_rocm_reindexer.cli.RocmOcrClient") as r_cls,
    ):
        p_cls.return_value.__aenter__.return_value = mock_p_client
        r_cls.return_value.__aenter__.return_value = mock_r_client

        result = runner.invoke(app, ["batch", "--dry-run"])

    assert result.exit_code == 0
    assert "failed: Corrupt original" in result.stdout
    assert "Dry-Run Execution Results" in result.stdout


def test_batch_command_error_handling(runner: CliRunner) -> None:
    """Verify batch command catches exceptions and exits with code 1."""
    with patch("paperless_rocm_reindexer.cli.run_batch") as mock_batch:
        mock_batch.side_effect = RuntimeError("Database connection failed")
        result = runner.invoke(app, ["batch"])

    assert result.exit_code == 1
    assert "Batch error: Database connection failed" in result.stdout


def test_stats_command_empty(runner: CliRunner, tmp_path: Path) -> None:
    """Verify stats command on empty database calculates 0.0% cleanly."""
    db_file = tmp_path / "empty_stats.db"
    result = runner.invoke(app, ["stats", "--state-db", str(db_file)])

    assert result.exit_code == 0
    assert "Reindex Processing State Summary" in result.stdout
    assert "Total" in result.stdout


def test_stats_command_error_handling(runner: CliRunner) -> None:
    """Verify stats command catches exceptions and exits with code 1."""
    with patch("paperless_rocm_reindexer.cli.run_stats") as mock_stats:
        mock_stats.side_effect = RuntimeError("Corrupt state DB")
        result = runner.invoke(app, ["stats"])

    assert result.exit_code == 1
    assert "Stats error: Corrupt state DB" in result.stdout


def test_daemon_command_error_handling(runner: CliRunner) -> None:
    """Verify daemon command catches exceptions and exits with code 1."""
    with patch("paperless_rocm_reindexer.cli.run_daemon") as mock_daemon:
        mock_daemon.side_effect = RuntimeError("Fatal daemon failure")
        result = runner.invoke(app, ["daemon"])

    assert result.exit_code == 1
    assert "Daemon error: Fatal daemon failure" in result.stdout


def test_daemon_iteration_error_tolerated(runner: CliRunner) -> None:
    """Verify daemon logs iteration errors and proceeds without crashing."""
    with patch("paperless_rocm_reindexer.cli.run_batch") as mock_batch:
        mock_batch.side_effect = [RuntimeError("Temporary API drop"), None]
        result = runner.invoke(
            app, ["daemon", "--interval-sec", "1", "--max-iterations", "2"]
        )

    assert result.exit_code == 0
    assert "Error during daemon iteration #1: Temporary API drop" in result.stdout
    assert "Daemon polling iteration #2" in result.stdout
    assert "Daemon stopped" in result.stdout
