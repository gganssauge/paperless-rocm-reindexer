"""Command Line Interface for Paperless ROCm Reindexer."""

from __future__ import annotations

import asyncio
import concurrent.futures
import difflib
import logging
import signal
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.syntax import Syntax
from rich.table import Table

from paperless_rocm_reindexer.clients.paperless import PaperlessClient
from paperless_rocm_reindexer.clients.rocm_ocr import RocmOcrClient
from paperless_rocm_reindexer.config import Settings, get_settings
from paperless_rocm_reindexer.db import PaperlessDatabaseAdapter
from paperless_rocm_reindexer.engine import DocumentResult, ReindexerEngine
from paperless_rocm_reindexer.models import parse_ocr_headers
from paperless_rocm_reindexer.state import StateStore
from paperless_rocm_reindexer.storage import ArchiveFileManager

logger = logging.getLogger(__name__)

console = Console()
app = typer.Typer(help="Paperless ROCm Reindexer CLI")


def run_async(coro: Any) -> Any:
    """Run an async coroutine from synchronous contexts safely."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


def _build_text_preview_table(old_text: str, new_text: str, doc_id: int) -> Table:
    """Construct preview table comparing existing content with newly extracted text."""
    table = Table(
        title=f"Text Overview - Document {doc_id}",
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )
    table.add_column("Paperless Content (Existing)", style="dim", ratio=1)
    table.add_column("ROCm-OCR Content (Extracted)", style="cyan", ratio=1)

    old_prev = (
        old_text
        if len(old_text) <= 500
        else old_text[:500] + "\n[dim]... [truncated][/dim]"
    )
    new_prev = (
        new_text
        if len(new_text) <= 500
        else new_text[:500] + "\n[dim]... [truncated][/dim]"
    )
    table.add_row(
        old_prev if old_text.strip() else "[dim](empty)[/dim]",
        new_prev if new_text.strip() else "[dim](empty)[/dim]",
    )
    return table


def _render_unified_diff(
    old_lines: Sequence[str],
    new_lines: Sequence[str],
    doc_id: int,
    out_console: Console,
) -> None:
    """Render unified diff panel or identical text notification."""
    diff = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"doc_{doc_id}_paperless_existing",
            tofile=f"doc_{doc_id}_rocm_ocr_extracted",
            lineterm="",
        )
    )
    if not diff:
        out_console.print(
            "[green]No textual diff: extracted text matches existing content.[/green]"
        )
        return

    out_console.print(
        Panel(
            Syntax("\n".join(diff), "diff", theme="ansi_dark", line_numbers=True),
            title=f"Visual Text Diff (Unified) - Document {doc_id}",
            border_style="blue",
        )
    )


def display_diff(
    old_text: str,
    new_text: str,
    doc_id: int,
    out_console: Console | None = None,
) -> None:
    """Display side-by-side overview and unified diff between old and new text."""
    c = out_console or console
    c.print(_build_text_preview_table(old_text, new_text, doc_id))
    old_lines = (old_text or "").splitlines(keepends=True)
    new_lines = (new_text or "").splitlines(keepends=True)
    _render_unified_diff(old_lines, new_lines, doc_id, c)


def display_benchmarks(
    doc_id: int,
    title: str | None,
    headers: dict[str, str],
    elapsed_ms: float,
    out_console: Console | None = None,
) -> None:
    """Display OCR inference benchmark metrics in a table."""
    c = out_console or console
    metrics = parse_ocr_headers(headers, default_elapsed_ms=elapsed_ms)
    bench_table = Table(
        title=f"Inference Benchmarks - Document {doc_id}",
        show_header=True,
        header_style="bold green",
    )
    bench_table.add_column("Metric", style="bold")
    bench_table.add_column("Value", style="cyan")

    bench_table.add_row("Document ID", str(doc_id))
    bench_table.add_row("Title", title or "(none)")
    bench_table.add_row("Inference Time (ms)", f"{metrics.inference_time_ms:.2f}")
    bench_table.add_row("Extracted Characters", str(metrics.extracted_chars))
    bench_table.add_row("Total Pages", str(metrics.page_count))
    bench_table.add_row("Execution Provider", metrics.execution_provider)

    c.print(bench_table)


def _save_scratch_pdf(doc_id: int, pdf_bytes: bytes, scratch_dir: Path | None) -> Path:
    """Save sandwich PDF to scratch folder or default /tmp path."""
    dest_dir = scratch_dir or Path("/tmp")
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / f"sandwich_test_{doc_id}.pdf"
    out_path.write_bytes(pdf_bytes)
    return out_path


async def _process_dry_run_doc(
    doc: dict[str, Any],
    sem: asyncio.Semaphore,
    paperless_client: PaperlessClient,
    rocm_client: RocmOcrClient,
    save_scratch: bool,
    scratch_dir: Path | None,
) -> dict[str, Any]:
    """Execute dry-run OCR on a single document."""
    doc_id = doc["id"]
    title = doc.get("title", "")
    filename = doc.get("original_file_name") or f"doc_{doc_id}.pdf"
    start = time.perf_counter()
    try:
        async with sem:
            orig_bytes = await paperless_client.download_original(doc_id)
            sandwich_bytes, headers = await rocm_client.generate_sandwich(
                orig_bytes, filename
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            metrics = parse_ocr_headers(headers, default_elapsed_ms=elapsed_ms)

            if save_scratch:
                _save_scratch_pdf(doc_id, sandwich_bytes, scratch_dir)

            return {
                "id": doc_id,
                "title": title,
                "status": "success",
                "chars": metrics.extracted_chars,
                "pages": metrics.page_count,
                "time_ms": metrics.inference_time_ms,
                "size": len(sandwich_bytes),
            }
    except Exception as exc:
        return {"id": doc_id, "title": title, "status": "failed", "error": str(exc)}


def _render_dry_run_table(results: list[dict[str, Any]], out_console: Console) -> None:
    """Render summary table for dry-run batch execution."""
    table = Table(
        title="Dry-Run Execution Results",
        show_header=True,
        header_style="bold yellow",
    )
    table.add_column("Doc ID", style="bold")
    table.add_column("Title")
    table.add_column("Status")
    table.add_column("Chars", justify="right")
    table.add_column("Pages", justify="right")
    table.add_column("Time (ms)", justify="right")

    for r in results:
        if r["status"] == "success":
            table.add_row(
                str(r["id"]),
                r.get("title", "") or "",
                "[green]success[/green]",
                str(r.get("chars", 0)),
                str(r.get("pages", 1)),
                f"{r.get('time_ms', 0):.1f}",
            )
        else:
            table.add_row(
                str(r["id"]),
                r.get("title", "") or "",
                f"[red]failed: {r.get('error')}[/red]",
                "-",
                "-",
                "-",
            )
    out_console.print(table)
    out_console.print(
        "[yellow]Dry-run finished. No storage files or database records were modified.[/yellow]"
    )


async def _run_dry_run_batch(
    docs: list[dict[str, Any]],
    eff_concurrency: int,
    paperless_client: PaperlessClient,
    rocm_client: RocmOcrClient,
    out_console: Console,
    save_scratch: bool = False,
    scratch_dir: Path | None = None,
) -> None:
    """Execute dry-run processing for all documents."""
    out_console.print(
        f"[bold yellow]Executing DRY-RUN on {len(docs)} document(s) (concurrency={eff_concurrency})...[/bold yellow]"
    )
    sem = asyncio.Semaphore(eff_concurrency)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=out_console,
    ) as progress:
        task = progress.add_task("[yellow]Dry-run OCR[/yellow]", total=len(docs))

        async def tracked_doc(d: dict[str, Any]) -> dict[str, Any]:
            try:
                return await _process_dry_run_doc(
                    d, sem, paperless_client, rocm_client, save_scratch, scratch_dir
                )
            finally:
                progress.advance(task)

        results = await asyncio.gather(*(tracked_doc(d) for d in docs))

    _render_dry_run_table(results, out_console)


def _render_batch_summary(
    results: list[DocumentResult],
    counts: dict[str, int],
    out_console: Console,
) -> None:
    """Render summary table for real batch execution."""
    summary_table = Table(
        title="Batch Reindex Summary",
        show_header=True,
        header_style="bold cyan",
    )
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Count", justify="right")
    summary_table.add_row("Total Processed", str(len(results)))
    summary_table.add_row("[green]Success[/green]", str(counts["success"]))
    summary_table.add_row("[red]Failed[/red]", str(counts["failed"]))
    summary_table.add_row("[yellow]Skipped[/yellow]", str(counts["skipped"]))
    out_console.print(summary_table)


async def _run_active_batch(
    docs: list[dict[str, Any]],
    eff_concurrency: int,
    cfg: Settings,
    paperless_client: PaperlessClient,
    rocm_client: RocmOcrClient,
    out_console: Console,
) -> None:
    """Run real reindexing execution persisting updates to Paperless and disk."""
    out_console.print(
        f"[bold green]Starting batch reindex of {len(docs)} document(s) (concurrency={eff_concurrency})...[/bold green]"
    )
    db_adapter = PaperlessDatabaseAdapter(db_url=cfg.PAPERLESS_DB_URL)
    file_manager = ArchiveFileManager(media_root=cfg.PAPERLESS_MEDIA_ROOT)
    state_store = StateStore(db_path=cfg.STATE_DB_PATH)

    await db_adapter.connect()
    try:
        engine = ReindexerEngine(
            paperless_client=paperless_client,
            rocm_client=rocm_client,
            db_adapter=db_adapter,
            file_manager=file_manager,
            state_store=state_store,
            concurrency=eff_concurrency,
            tag_success=cfg.TAG_SUCCESS,
            tag_failure=cfg.TAG_FAILURE,
        )
        async with engine:
            counts = {"success": 0, "failed": 0, "skipped": 0}
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                TextColumn("{task.fields[status_counts]}"),
                console=out_console,
            ) as progress:
                task = progress.add_task(
                    "Reindexing documents",
                    total=len(docs),
                    status_counts="Success: 0 | Failed: 0 | Skipped: 0",
                )

                def on_progress(res: DocumentResult) -> None:
                    counts[res.status] = counts.get(res.status, 0) + 1
                    progress.update(
                        task,
                        advance=1,
                        status_counts=f"Success: {counts['success']} | Failed: {counts['failed']} | Skipped: {counts['skipped']}",
                    )

                results = await engine.process_batch(docs, on_progress=on_progress)
            _render_batch_summary(results, counts, out_console)
    finally:
        await db_adapter.close()
        await state_store.close()


async def run_batch(
    concurrency: int | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    settings: Settings | None = None,
    out_console: Console | None = None,
    save_scratch: bool = False,
    scratch_dir: Path | None = None,
) -> None:
    """Execute batch processing of unprocessed documents."""
    cfg = settings or get_settings()
    c = out_console or console

    eff_concurrency = concurrency if concurrency is not None else cfg.OCR_CONCURRENCY
    eff_limit = limit if limit is not None else cfg.BATCH_LIMIT
    is_dry_run = dry_run or cfg.DRY_RUN

    async with (
        PaperlessClient(
            base_url=cfg.PAPERLESS_BASE_URL, token=cfg.PAPERLESS_API_TOKEN
        ) as paperless_client,
        RocmOcrClient(
            base_url=cfg.ROCM_OCR_BASE_URL, api_key=cfg.ROCM_OCR_API_KEY
        ) as rocm_client,
    ):
        c.print(
            f"[cyan]Querying unprocessed documents from {cfg.PAPERLESS_BASE_URL}...[/cyan]"
        )
        docs = await paperless_client.list_unprocessed_documents(
            tag_exclude=cfg.TAG_SUCCESS,
            tag_exclude_fail=cfg.TAG_FAILURE,
            limit=eff_limit,
        )

        if not docs:
            c.print("[yellow]No unprocessed documents found.[/yellow]")
            return

        if is_dry_run:
            await _run_dry_run_batch(
                docs,
                eff_concurrency,
                paperless_client,
                rocm_client,
                c,
                save_scratch,
                scratch_dir,
            )
            return

        await _run_active_batch(
            docs, eff_concurrency, cfg, paperless_client, rocm_client, c
        )


async def run_inspect(
    doc_id: int,
    save_sandwich: Path | None = None,
    settings: Settings | None = None,
    out_console: Console | None = None,
    save_scratch: bool = False,
    scratch_dir: Path | None = None,
) -> None:
    """Inspect OCR extraction and text diff against Paperless for a single document."""
    cfg = settings or get_settings()
    c = out_console or console

    c.print(
        f"[cyan]Inspecting document {doc_id} from {cfg.PAPERLESS_BASE_URL}...[/cyan]"
    )
    async with (
        PaperlessClient(
            base_url=cfg.PAPERLESS_BASE_URL, token=cfg.PAPERLESS_API_TOKEN
        ) as paperless_client,
        RocmOcrClient(
            base_url=cfg.ROCM_OCR_BASE_URL, api_key=cfg.ROCM_OCR_API_KEY
        ) as rocm_client,
    ):
        document = await paperless_client.get_document(doc_id)
        old_content = document.get("content") or ""
        title = document.get("title")
        filename = document.get("original_file_name") or f"doc_{doc_id}.pdf"

        c.print(f"[cyan]Downloading original file '{filename}'...[/cyan]")
        orig_bytes = await paperless_client.download_original(doc_id)

        c.print(f"[cyan]Running OCR via {cfg.ROCM_OCR_BASE_URL}...[/cyan]")
        start = time.perf_counter()
        sandwich_bytes, headers = await rocm_client.generate_sandwich(
            orig_bytes, filename
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        metrics = parse_ocr_headers(headers, default_elapsed_ms=elapsed_ms)
        display_diff(old_content, metrics.extracted_text or "", doc_id, out_console=c)
        display_benchmarks(doc_id, title, headers, elapsed_ms, out_console=c)

        target_save = save_sandwich
        if not target_save and save_scratch:
            target_save = (scratch_dir or Path("/tmp")) / f"sandwich_test_{doc_id}.pdf"

        if target_save:
            target_save.parent.mkdir(parents=True, exist_ok=True)
            target_save.write_bytes(sandwich_bytes)
            c.print(
                f"[bold green]Saved sandwich PDF ({len(sandwich_bytes)} bytes) to {target_save}[/bold green]"
            )


def _setup_daemon_signals(request_stop: Callable[[], None]) -> None:
    """Register shutdown signal handlers for daemon loop."""
    try:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, request_stop)
            except (NotImplementedError, RuntimeError):
                pass
    except RuntimeError:
        pass


async def _run_daemon_iteration(
    iteration: int, cfg: Settings, eff_concurrency: int, c: Console
) -> None:
    """Execute a single polling iteration for the daemon."""
    c.print(f"[dim]Daemon polling iteration #{iteration}...[/dim]")
    try:
        await run_batch(
            concurrency=eff_concurrency,
            limit=cfg.BATCH_LIMIT,
            dry_run=False,
            settings=cfg,
            out_console=c,
        )
    except Exception as e:
        c.print(f"[red]Error during daemon iteration #{iteration}: {e}[/red]")


async def run_daemon(
    interval_sec: int | None = None,
    concurrency: int | None = None,
    max_iterations: int | None = None,
    settings: Settings | None = None,
    out_console: Console | None = None,
) -> None:
    """Run continuous reindexing loop polling Paperless periodically."""
    cfg = settings or get_settings()
    c = out_console or console
    eff_interval = (
        interval_sec
        if (interval_sec is not None and interval_sec > 0)
        else (cfg.DAEMON_INTERVAL_SEC or 60)
    )
    eff_concurrency = concurrency if concurrency is not None else cfg.OCR_CONCURRENCY

    c.print(
        f"[bold cyan]Starting Reindexer Daemon (interval={eff_interval}s, concurrency={eff_concurrency})...[/bold cyan]"
    )
    stop_event = asyncio.Event()

    def _request_stop() -> None:
        c.print("\n[yellow]Daemon stop requested, shutting down...[/yellow]")
        stop_event.set()

    _setup_daemon_signals(_request_stop)

    iteration = 0
    while not stop_event.is_set():
        iteration += 1
        await _run_daemon_iteration(iteration, cfg, eff_concurrency, c)

        if max_iterations is not None and iteration >= max_iterations:
            c.print(
                f"[dim]Reached max iterations ({max_iterations}), stopping daemon.[/dim]"
            )
            break

        if stop_event.is_set():
            break

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=float(eff_interval))
        except (asyncio.TimeoutError, TimeoutError):
            pass

    c.print("[green]Daemon stopped.[/green]")


async def run_stats(
    state_db: Path | str | None = None,
    settings: Settings | None = None,
    out_console: Console | None = None,
) -> None:
    """Read state database and render processing summary statistics."""
    cfg = settings or get_settings()
    c = out_console or console
    db_path = state_db or cfg.STATE_DB_PATH
    state_store = StateStore(db_path=db_path)
    await state_store.initialize()
    try:
        summary = await state_store.get_summary()
    finally:
        await state_store.close()

    table = Table(
        title=f"Reindex Processing State Summary ({db_path})",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Status", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")

    total = summary.get("total", 0)
    for status in ("success", "failed", "skipped", "pending"):
        count = summary.get(status, 0)
        pct = (count / total * 100) if total > 0 else 0.0
        color = {
            "success": "green",
            "failed": "red",
            "skipped": "yellow",
            "pending": "blue",
        }.get(status, "white")
        table.add_row(
            f"[{color}]{status.capitalize()}[/{color}]", str(count), f"{pct:.1f}%"
        )

    table.add_section()
    table.add_row("[bold]Total[/bold]", f"[bold]{total}[/bold]", "100.0%")
    c.print(table)


# Typer Command Callbacks


@app.command(name="batch")
def batch_command(
    concurrency: int | None = typer.Option(
        None, "--concurrency", "-c", help="Concurrent document processing tasks"
    ),
    limit: int | None = typer.Option(
        None, "--limit", "-l", help="Maximum documents to process (0 = all)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Run OCR without modifying Paperless files or database"
    ),
    save_scratch: bool = typer.Option(
        False,
        "--save-scratch",
        help="Save sandwich PDFs generated during dry-run to /tmp or scratch dir",
    ),
    scratch_dir: Path | None = typer.Option(
        None, "--scratch-dir", help="Directory for saving dry-run sandwich PDFs"
    ),
) -> None:
    """Process a batch of unprocessed documents."""
    try:
        run_async(
            run_batch(
                concurrency=concurrency,
                limit=limit,
                dry_run=dry_run,
                save_scratch=save_scratch,
                scratch_dir=scratch_dir,
            )
        )
    except Exception as exc:
        console.print(f"[bold red]Batch error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command(name="inspect")
def inspect_command(
    doc_id: int | None = typer.Option(
        None, "--doc-id", "-d", help="Target document ID to inspect"
    ),
    doc_id_arg: int | None = typer.Argument(None, help="Target document ID to inspect"),
    save_sandwich: Path | None = typer.Option(
        None, "--save-sandwich", "-s", help="Path to export the generated sandwich PDF"
    ),
    save_scratch: bool = typer.Option(
        False, "--save-scratch", help="Save sandwich PDF to /tmp/sandwich_test_{id}.pdf"
    ),
    scratch_dir: Path | None = typer.Option(
        None,
        "--scratch-dir",
        help="Directory for saving sandwich PDF (defaults to /tmp)",
    ),
) -> None:
    """Inspect OCR extraction and text diff against Paperless for a document."""
    target_id = doc_id if doc_id is not None else doc_id_arg
    if target_id is None:
        console.print("[red]Error: --doc-id is required.[/red]")
        raise typer.Exit(code=1)
    try:
        run_async(
            run_inspect(
                doc_id=target_id,
                save_sandwich=save_sandwich,
                save_scratch=save_scratch,
                scratch_dir=scratch_dir,
            )
        )
    except Exception as exc:
        console.print(f"[bold red]Inspect error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command(name="daemon")
def daemon_command(
    interval_sec: int | None = typer.Option(
        None, "--interval-sec", "-i", help="Polling interval in seconds"
    ),
    concurrency: int | None = typer.Option(
        None, "--concurrency", "-c", help="Concurrent document processing tasks"
    ),
    max_iterations: int | None = typer.Option(
        None, "--max-iterations", help="Exit after N polling iterations"
    ),
) -> None:
    """Run continuously polling Paperless for new documents."""
    try:
        run_async(
            run_daemon(
                interval_sec=interval_sec,
                concurrency=concurrency,
                max_iterations=max_iterations,
            )
        )
    except Exception as exc:
        console.print(f"[bold red]Daemon error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command(name="stats")
def stats_command(
    state_db: Path | None = typer.Option(
        None, "--state-db", help="Path to SQLite state database"
    ),
) -> None:
    """Display processing summary statistics from the state store."""
    try:
        run_async(run_stats(state_db=state_db))
    except Exception as exc:
        console.print(f"[bold red]Stats error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
