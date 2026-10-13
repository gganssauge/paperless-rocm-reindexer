"""Command Line Interface for Paperless ROCm Reindexer."""

from __future__ import annotations

import asyncio
import concurrent.futures
import difflib
import logging
import signal
import time
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


def display_diff(
    old_text: str,
    new_text: str,
    doc_id: int,
    out_console: Console | None = None,
) -> None:
    """Display side-by-side overview and unified diff between old and new text."""
    c = out_console or console
    old_lines = (old_text or "").splitlines(keepends=True)
    new_lines = (new_text or "").splitlines(keepends=True)

    # 1. Side-by-side text preview table
    preview_table = Table(
        title=f"Text Overview - Document {doc_id}",
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )
    preview_table.add_column("Paperless Content (Existing)", style="dim", ratio=1)
    preview_table.add_column("ROCm-OCR Content (Extracted)", style="cyan", ratio=1)

    old_preview = (
        old_text
        if len(old_text) <= 500
        else old_text[:500] + "\n[dim]... [truncated][/dim]"
    )
    new_preview = (
        new_text
        if len(new_text) <= 500
        else new_text[:500] + "\n[dim]... [truncated][/dim]"
    )
    preview_table.add_row(
        old_preview if old_text.strip() else "[dim](empty)[/dim]",
        new_preview if new_text.strip() else "[dim](empty)[/dim]",
    )
    c.print(preview_table)

    # 2. Unified diff panel
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
        c.print(
            "[green]No textual diff: extracted text matches existing content.[/green]"
        )
    else:
        diff_text = "\n".join(diff)
        c.print(
            Panel(
                Syntax(diff_text, "diff", theme="ansi_dark", line_numbers=True),
                title=f"Visual Text Diff (Unified) - Document {doc_id}",
                border_style="blue",
            )
        )


def display_benchmarks(
    doc_id: int,
    title: str | None,
    headers: dict[str, str],
    elapsed_ms: float,
    out_console: Console | None = None,
) -> None:
    """Display OCR inference benchmark metrics in a table."""
    c = out_console or console
    bench_table = Table(
        title=f"Inference Benchmarks - Document {doc_id}",
        show_header=True,
        header_style="bold green",
    )
    bench_table.add_column("Metric", style="bold")
    bench_table.add_column("Value", style="cyan")

    inf_time = (
        headers.get("X-Inference-Time-Ms")
        or headers.get("x-inference-time-ms")
        or f"{elapsed_ms:.2f}"
    )
    chars = (
        headers.get("X-Extracted-Chars")
        or headers.get("x-extracted-chars")
        or str(
            len(
                headers.get("X-Extracted-Text", "")
                or headers.get("x-extracted-text", "")
            )
        )
    )
    pages = (
        headers.get("X-Page-Count")
        or headers.get("x-page-count")
        or headers.get("X-Total-Pages")
        or headers.get("x-total-pages")
        or "1"
    )
    provider = (
        headers.get("X-Execution-Provider")
        or headers.get("x-execution-provider")
        or "RapidOCR / ROCm PP-OCRv4"
    )

    bench_table.add_row("Document ID", str(doc_id))
    bench_table.add_row("Title", title or "(none)")
    bench_table.add_row("Inference Time (ms)", str(inf_time))
    bench_table.add_row("Extracted Characters", str(chars))
    bench_table.add_row("Total Pages", str(pages))
    bench_table.add_row("Execution Provider", str(provider))

    c.print(bench_table)


async def run_batch(
    concurrency: int | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    settings: Settings | None = None,
    out_console: Console | None = None,
) -> None:
    """Execute batch processing of unprocessed documents."""
    cfg = settings or get_settings()
    c = out_console or console

    eff_concurrency = concurrency if concurrency is not None else cfg.OCR_CONCURRENCY
    eff_limit = limit if limit is not None else cfg.BATCH_LIMIT
    is_dry_run = dry_run or cfg.DRY_RUN

    async with (
        PaperlessClient(
            base_url=cfg.PAPERLESS_BASE_URL,
            token=cfg.PAPERLESS_API_TOKEN,
        ) as paperless_client,
        RocmOcrClient(
            base_url=cfg.ROCM_OCR_BASE_URL,
            api_key=cfg.ROCM_OCR_API_KEY,
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
            c.print(
                f"[bold yellow]Executing DRY-RUN on {len(docs)} document(s) (concurrency={eff_concurrency})...[/bold yellow]"
            )
            sem = asyncio.Semaphore(eff_concurrency)
            results: list[dict[str, Any]] = []

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=c,
            ) as progress:
                task = progress.add_task(
                    "[yellow]Dry-run OCR[/yellow]", total=len(docs)
                )

                async def process_one(doc: dict[str, Any]) -> dict[str, Any]:
                    doc_id = doc["id"]
                    title = doc.get("title", "")
                    filename = doc.get("original_file_name") or f"doc_{doc_id}.pdf"
                    start = time.perf_counter()
                    try:
                        async with sem:
                            orig_bytes = await paperless_client.download_original(
                                doc_id
                            )
                            (
                                sandwich_bytes,
                                headers,
                            ) = await rocm_client.generate_sandwich(
                                orig_bytes, filename
                            )
                            elapsed_ms = (time.perf_counter() - start) * 1000
                            chars = (
                                headers.get("X-Extracted-Chars")
                                or headers.get("x-extracted-chars")
                                or str(len(headers.get("X-Extracted-Text", "")))
                            )
                            pages = (
                                headers.get("X-Page-Count")
                                or headers.get("x-page-count")
                                or "1"
                            )
                            inf_time = (
                                headers.get("X-Inference-Time-Ms")
                                or headers.get("x-inference-time-ms")
                                or f"{elapsed_ms:.1f}"
                            )
                            return {
                                "id": doc_id,
                                "title": title,
                                "status": "success",
                                "chars": int(chars),
                                "pages": int(pages),
                                "time_ms": float(inf_time),
                                "size": len(sandwich_bytes),
                            }
                    except Exception as exc:
                        return {
                            "id": doc_id,
                            "title": title,
                            "status": "failed",
                            "error": str(exc),
                        }
                    finally:
                        progress.advance(task)

                tasks = [process_one(d) for d in docs]
                results = await asyncio.gather(*tasks)

            # Print dry run summary table
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
            c.print(table)
            c.print(
                "[yellow]Dry-run finished. No storage files or database records were modified.[/yellow]"
            )
            return

        # Normal run
        c.print(
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
                    console=c,
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

                # Batch summary
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
                summary_table.add_row(
                    "[yellow]Skipped[/yellow]", str(counts["skipped"])
                )
                c.print(summary_table)
        finally:
            await db_adapter.close()
            await state_store.close()


async def run_inspect(
    doc_id: int,
    save_sandwich: Path | None = None,
    settings: Settings | None = None,
    out_console: Console | None = None,
) -> None:
    """Inspect OCR extraction and text diff against Paperless for a single document."""
    cfg = settings or get_settings()
    c = out_console or console

    c.print(
        f"[cyan]Inspecting document {doc_id} from {cfg.PAPERLESS_BASE_URL}...[/cyan]"
    )

    async with (
        PaperlessClient(
            base_url=cfg.PAPERLESS_BASE_URL,
            token=cfg.PAPERLESS_API_TOKEN,
        ) as paperless_client,
        RocmOcrClient(
            base_url=cfg.ROCM_OCR_BASE_URL,
            api_key=cfg.ROCM_OCR_API_KEY,
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

        new_content = (
            headers.get("X-Extracted-Text") or headers.get("x-extracted-text") or ""
        )

        # Display text diff
        display_diff(old_content, new_content, doc_id, out_console=c)

        # Display benchmarks
        display_benchmarks(doc_id, title, headers, elapsed_ms, out_console=c)

        if save_sandwich:
            save_path = Path(save_sandwich)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_bytes(sandwich_bytes)
            c.print(
                f"[bold green]Saved sandwich PDF ({len(sandwich_bytes)} bytes) to {save_path}[/bold green]"
            )


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

    eff_interval = interval_sec if interval_sec is not None else cfg.DAEMON_INTERVAL_SEC
    if eff_interval <= 0:
        eff_interval = 60
    eff_concurrency = concurrency if concurrency is not None else cfg.OCR_CONCURRENCY

    c.print(
        f"[bold cyan]Starting Reindexer Daemon (interval={eff_interval}s, concurrency={eff_concurrency})...[/bold cyan]"
    )

    stop_event = asyncio.Event()

    def request_stop() -> None:
        c.print("\n[yellow]Daemon stop requested, shutting down gracefully...[/yellow]")
        stop_event.set()

    try:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, request_stop)
            except (NotImplementedError, RuntimeError):
                pass
    except RuntimeError:
        pass

    iteration = 0
    while not stop_event.is_set():
        iteration += 1
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
            f"[{color}]{status.capitalize()}[/{color}]",
            str(count),
            f"{pct:.1f}%",
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
) -> None:
    """Process a batch of unprocessed documents."""
    try:
        run_async(
            run_batch(
                concurrency=concurrency,
                limit=limit,
                dry_run=dry_run,
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
) -> None:
    """Inspect OCR extraction and text diff against Paperless for a document."""
    target_id = doc_id if doc_id is not None else doc_id_arg
    if target_id is None:
        console.print("[red]Error: --doc-id is required.[/red]")
        raise typer.Exit(code=1)
    try:
        run_async(run_inspect(doc_id=target_id, save_sandwich=save_sandwich))
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
