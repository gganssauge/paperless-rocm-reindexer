# Implementation Tickets: `paperless-rocm-reindexer`

**Epic:** Build, verify, and package the `paperless-rocm-reindexer` automation service for batch and continuous document re-OCR using `rocm-ocr`.
- **Specification:** [spec.md](spec.md)

---

## Ticket Overview

- [x] **PRO-00:** Epic Planning & Architecture Alignment (Completed via Grilling)
- [x] **PRO-01:** Project Setup, Tooling & CI/CD Pipeline
- [x] **PRO-02:** Paperless-ngx & `rocm-ocr` API Clients
- [x] **PRO-03:** Local SQLite State Store & Checkpoint Manager
- [x] **PRO-04:** Storage Sync, Checksum Calculator & Postgres Database Adapter
- [x] **PRO-05:** Worker Engine, Dynamic Concurrency & Adaptive Backoff
- [x] **PRO-06:** Typer CLI & Dry-Run / Diff Inspection Suite
- [x] **PRO-07:** Containerization, Kubernetes Manifests & Pilot Run

---

### Ticket PRO-01: Project Setup, Tooling & CI/CD Pipeline

- **Type:** Task / Infrastructure
- **Estimated Effort:** 1.5 h
- **Goal:** Initialize repository `paperless-rocm-reindexer` with standard Astral `uv` toolchain, linting, formatting, and GitHub Actions matching repo standards (`paperless-custom-classifier` / `rocm-ocr`).

#### Checklist
- [ ] Initialize repository with `uv init --python 3.14`.
- [ ] Configure `pyproject.toml` with dependencies (`httpx`, `typer[all]`, `asyncpg` or `psycopg`, `pydantic`, `rich`).
- [ ] Configure dev dependencies (`pytest`, `pytest-asyncio`, `ruff`, `ty`).
- [ ] Add pre-commit hooks (Ruff format & lint).
- [ ] Configure GitHub Actions workflow:
  - Run linting and unit test suites.
  - SonarCloud scan and quality gate enforcement.
  - Trivy vulnerability scanner on container builds.

#### Acceptance Criteria
- [ ] `uv sync` installs cleanly in a fresh environment.
- [ ] `ruff check .` and `ty check` pass with zero warnings.
- [ ] CI pipeline runs successfully on PR creation.

---

### Ticket PRO-02: Paperless-ngx & `rocm-ocr` API Clients

- **Type:** Feature / Integration
- **Estimated Effort:** 2.5 h
- **Goal:** Provide asynchronous HTTP clients for Paperless-ngx REST API and `rocm-ocr` inference service.

#### Checklist
- [ ] Implement `PaperlessClient`:
  - `list_unprocessed_documents(tag_filter, limit)` (pagination support).
  - `download_original(document_id) -> bytes`.
  - `patch_document(document_id, content, tags)`.
  - `get_or_create_tag(tag_name) -> int`.
- [ ] Implement `RocmOcrClient`:
  - `generate_sandwich(file_bytes, filename) -> (pdf_bytes, headers)`.
  - Extract text characters from `X-Extracted-Chars` and inference time from `X-Inference-Time-Ms`.
  - `set_concurrency(target_concurrency: int)`.
  - Handle optional Bearer token authentication.

#### Acceptance Criteria
- [ ] Unit tests with `pytest-httpx` / `respx` mocking Paperless and `rocm-ocr` responses.
- [ ] HTTP 401, 404, 500, and 503 error handling paths are covered with custom exceptions (`OcrServiceUnavailableError`, `PaperlessApiError`).

---

### Ticket PRO-03: Local SQLite State Store & Checkpoint Manager

- **Type:** Feature / Storage
- **Estimated Effort:** 1.5 h
- **Goal:** Implement persistent state tracking to guarantee pause/resume capability and avoid duplicate processing without hammering Paperless API.

#### Checklist
- [ ] Implement `StateStore` with SQLite schema:
  - Columns: `document_id`, `title`, `page_count`, `checksum_old`, `checksum_new`, `status`, `duration_ms`, `error_message`, `processed_at`.
- [ ] Implement query methods:
  - `is_processed(document_id) -> bool`
  - `record_success(doc_id, metadata)`
  - `record_failure(doc_id, error_str)`
  - `get_summary() -> dict` (counts of success, failed, remaining).
- [ ] Ensure SQLite WAL mode (`PRAGMA journal_mode=WAL;`) for concurrent async access.

#### Acceptance Criteria
- [ ] Transactions are ACID compliant.
- [ ] Process interruption mid-batch allows immediate resume without reprocessing successful entries.

---

### Ticket PRO-04: Storage Sync, Checksum Calculator & Postgres Database Adapter

- **Type:** Feature / Storage & DB
- **Estimated Effort:** 2.5 h
- **Goal:** Safely replace the archive PDF on the mounted volume and update `archive_checksum` in Paperless PostgreSQL.

#### Checklist
- [ ] Implement `ArchiveFileManager`:
  - Resolve path: `{media_root}/documents/archive/{archive_filename}`.
  - Write new sandwich PDF atomically using temporary file `{path}.tmp` + `os.replace`.
  - Calculate SHA256 (or MD5 depending on Paperless configuration) of the new binary.
- [ ] Implement `PaperlessDatabaseAdapter`:
  - Connect to Paperless PostgreSQL via `asyncpg` / `psycopg`.
  - Run SQL update:
    ```sql
    UPDATE documents_document
    SET archive_checksum = $1
    WHERE id = $2;
    ```
- [ ] Verify permissions / umask handling to match Paperless container user permissions.

#### Acceptance Criteria
- [ ] In-place replacement leaves no orphaned `.tmp` files.
- [ ] `archive_checksum` matches the hash of the written PDF file exactly.
- [ ] Paperless `document_sanity_checker` validates without warning on modified documents.

---

### Ticket PRO-05: Worker Engine, Dynamic Concurrency & Adaptive Backoff

- **Type:** Feature / Core Engine
- **Estimated Effort:** 3.0 h
- **Goal:** Orchestrate the pipeline with parallel task execution and dynamic hardware failure recovery.

#### Checklist
- [ ] Implement `ReindexWorkerPool` using `asyncio.Semaphore(max_concurrency)`.
- [ ] Implement execution loop for each document:
  1. Download original.
  2. Send to `rocm-ocr` `/ocr/v1/sandwich`.
  3. On HTTP 503 (HIP / GPU OOM):
     - Trigger adaptive backoff: set `rocm-ocr` concurrency to 1 via `POST /ocr/v1/setup`.
     - Retry document once.
     - On second failure: log error, tag document `ocr:failed`, update SQLite status `failed`.
  4. Write sandwich PDF to disk.
  5. Update Postgres `archive_checksum`.
  6. Patch Paperless `content` and add `ocr:rocm-v4` tag.
  7. Record `success` in SQLite.
- [ ] Graceful shutdown on `SIGINT` / `SIGTERM` (drain active workers cleanly).

#### Acceptance Criteria
- [ ] Concurrent document processing achieves high GPU utilization without deadlocks.
- [ ] 503 errors trigger backoff without crashing the entire batch process.

---

### Ticket PRO-06: Typer CLI & Dry-Run / Diff Inspection Suite

- **Type:** Feature / CLI
- **Estimated Effort:** 2.0 h
- **Goal:** Provide rich CLI commands with inspection, sample runs, text-diffing, and progress reporting.

#### Checklist
- [ ] Setup Typer CLI with commands:
  - `reindex batch [--concurrency N] [--limit N] [--dry-run]`
  - `reindex inspect --doc-id <ID> [--save-sandwich <PATH>]`
  - `reindex daemon [--interval-sec N]`
  - `reindex stats` (summary of processed/pending documents).
- [ ] Implement Dry-Run mode:
  - Fetches original and performs OCR without modifying Paperless files or database.
  - Displays side-by-side terminal diff of old Paperless text vs. new `rocm-ocr` extracted text using `rich`.
  - Optionally exports the sandwich PDF to `/tmp/sandwich_test.pdf`.

#### Acceptance Criteria
- [ ] Running `reindex inspect --doc-id 123` prints visual diff and inference benchmarks.
- [ ] Batch execution shows rich live progress bar (`rich.progress`) with ETA and throughput.

---

### Ticket PRO-07: Containerization, Kubernetes Manifests & Pilot Run

- **Type:** Task / Deployment & QA
- **Estimated Effort:** 2.5 h
- **Goal:** Package the application as a production container image and deploy Kubernetes manifests for pilot execution.

#### Checklist
- [ ] Create multi-stage `Dockerfile` (distroless/debian-slim, non-root user).
- [ ] Author Kubernetes manifests:
  - `Job` for one-shot batch migration.
  - `CronJob` or `Deployment` for continuous daemon operation.
  - PVC mounts for Paperless `media/` volume.
  - Secret definitions for Paperless API token, DB credentials, and OCR API key.
- [ ] Execute pilot test run against 10–20 documents in staging/homelab:
  - Verify document searchability in Paperless Web UI.
  - Verify text layer rendering in PDF viewer.
  - Run Paperless `python manage.py document_sanity_checker`.

#### Acceptance Criteria
- [ ] Container image scans with 0 high/critical CVEs via Trivy.
- [ ] Batch job processes pilot documents seamlessly.
- [ ] Paperless search index accurately reflects newly recognized text.
