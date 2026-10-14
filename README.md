# paperless-rocm-reindexer

[![Python: 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![uv: Astral](https://img.shields.io/badge/package%20manager-uv-purple.svg)](https://docs.astral.sh/uv/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![RapidOCR: PP-OCRv4](https://img.shields.io/badge/OCR-RapidOCR%20PP--OCRv4-orange.svg)](https://github.com/RapidAI/RapidOCR)
[![Hardware: AMD ROCm](https://img.shields.io/badge/Hardware-AMD%20ROCm-red.svg)](https://rocm.docs.amd.com/)

**Batch & Continuous Reindexing Engine for Paperless-ngx powered by `rocm-ocr`.**

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture & Processing Pipeline](#architecture--processing-pipeline)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
  - [`reindex batch`](#reindex-batch)
  - [`reindex inspect`](#reindex-inspect)
  - [`reindex daemon`](#reindex-daemon)
  - [`reindex stats`](#reindex-stats)
- [Configuration](#configuration)
- [Deployment](#deployment)
  - [Kubernetes (Kustomize)](#kubernetes-kustomize)
  - [Docker / Container](#docker--container)
  - [Docker Compose](#docker-compose)
- [Integrity & Verification](#integrity--verification)
- [Development & Testing](#development--testing)
- [Technical Documentation](#technical-documentation)
- [License](#license)

---

## Overview

`paperless-rocm-reindexer` is a high-performance orchestration service that re-processes and re-OCRs documents stored in [Paperless-ngx](https://docs.paperless-ngx.com/) using a dedicated, GPU-accelerated [`rocm-ocr`](https://github.com/RapidAI/RapidOCR) microservice running on AMD ROCm (tested on AMD Radeon RX 7800 XT).

Standard CPU-based OCR pipelines (such as Tesseract) can be slow, resource-heavy, and produce sub-optimal text extraction on scanned documents, receipts, and multi-column invoices. `paperless-rocm-reindexer` bridges Paperless-ngx and `rocm-ocr` to upgrade your archive with high-fidelity, searchable Sandwich PDFs and enriched fulltext search indices without interrupting your document management workflow.

---

## Key Features

- **Clean Extraction (`original=true`)**: Always downloads the pristine uploaded original file from Paperless-ngx, avoiding generational quality loss or OCR artifact accumulation.
- **Dual-Layer Upgrade**:
  1. Generates and replaces the physical visual archive PDF on disk (`/media/documents/archive/...`) with a high-fidelity searchable Sandwich PDF containing embedded font text layers.
  2. Updates Paperless-ngx's internal fulltext search index by patching document `content` via the Paperless REST API (`PATCH /api/documents/{id}/`).
- **Database & Sanity Checker Integrity**:
  - Automatically identifies whether your Paperless deployment uses MD5 or SHA-256 archive checksums.
  - Computes cryptographic hashes of the newly written archive PDFs.
  - Directly updates PostgreSQL (`documents_document.archive_checksum`), keeping Paperless-ngx's `python manage.py document_sanity_checker` 100% clean and compliant.
- **Adaptive Hardware Concurrency & VRAM Protection**:
  - Dynamically configures OCR inference concurrency via `POST /ocr/v1/setup` (default 4 workers, up to 8).
  - Automatically catches ROCm/HIP VRAM exhaustion (HTTP 503).
  - Throttles concurrency to 1 on memory pressure, isolates and retries the document, and ramps concurrency back up once the bottleneck clears.
- **Persistent State Store (SQLite WAL)**:
  - Tracks every document's lifecycle in a local SQLite database (`reindex_state.db`) using Write-Ahead Logging (`WAL`) mode.
  - Ensures seamless resumption after interruptions without repeatedly querying the Paperless API.
- **Zero In-Place Corruption**: Writes sandwich PDFs to a temporary sibling file (`{filename}.tmp`) and applies an atomic filesystem swap (`os.replace`).
- **Dry-Run & Visual Diffing**: Built-in CLI inspector with rich side-by-side terminal diffs, character metrics, and inference latency benchmarks.

---

## Architecture & Processing Pipeline

The `rocm-ocr` microservice remains strictly domain-agnostic and stateless. `paperless-rocm-reindexer` acts as the orchestrator bridging the Paperless storage/API boundary and the inference engine:

```mermaid
flowchart TD
    subgraph Storage ["Persistent Storage & DB"]
        MediaArchive[("Paperless Media Archive<br/>/media/documents/archive")]
        PG[("Paperless PostgreSQL<br/>table: documents_document")]
    end

    subgraph Service ["paperless-rocm-reindexer"]
        CLI["CLI / Daemon Engine<br/>Typer / Asyncio"]
        StateDB[("Local State Store<br/>SQLite: reindex_state.db")]
        ClientPaperless["Paperless REST Client"]
        ClientROCM["ROCm-OCR HTTP Client"]
    end

    subgraph Paperless ["Paperless-ngx Core"]
        PaperlessAPI["Paperless REST API"]
    end

    subgraph OCR ["rocm-ocr Microservice"]
        ROCMService["FastAPI Inference Engine<br/>PP-OCRv4 / ROCm HIP"]
    end

    CLI -->|1. Query Docs & Tags| ClientPaperless
    ClientPaperless -->|GET /api/documents/| PaperlessAPI
    PaperlessAPI -->|Stream Original| CLI

    CLI -->|2. POST /ocr/v1/sandwich| ClientROCM
    ClientROCM --> ROCMService
    ROCMService -->|Return Sandwich PDF| ClientROCM

    CLI -->|3. Atomic PDF Write| MediaArchive
    CLI -->|4. UPDATE archive_checksum| PG
    CLI -->|5. PATCH /api/documents/:id/| ClientPaperless
    CLI -->|6. Log Checkpoint| StateDB
```

### Document Processing Sequence

For each eligible document (documents lacking the `ocr:rocm-v4` tag and not marked `ocr:failed`):

```mermaid
sequenceDiagram
    autonumber
    participant R as Reindexer Worker
    participant P_API as Paperless API
    participant ROCM as rocm-ocr
    participant FS as Archive Storage
    participant DB as Paperless Postgres
    participant S as SQLite State

    R->>P_API: GET /api/documents/ (filter unprocessed)
    R->>P_API: GET /api/documents/:id/download/?original=true
    P_API-->>R: Pristine Original File (PDF / Image)

    R->>ROCM: POST /ocr/v1/sandwich (multipart file)
    alt HTTP 503 GPU VRAM Exhaustion
        R->>ROCM: POST /ocr/v1/setup (concurrency=1)
        R->>ROCM: POST /ocr/v1/sandwich (retry single doc)
    end
    ROCM-->>R: Binary Sandwich PDF + X-Headers

    R->>FS: Atomic write to media/documents/archive/:archive_filename
    R->>R: Calculate new MD5 / SHA256 checksum
    R->>DB: UPDATE documents_document SET archive_checksum = new_hash WHERE id = doc_id
    R->>P_API: PATCH /api/documents/:id/ (update content and add ocr:rocm-v4 tag)
    R->>S: Log status to reindex_log (status = success)
```


---

## Prerequisites

Before running `paperless-rocm-reindexer`, ensure you have:

1. **Paperless-ngx Instance**: Running with an active REST API. Obtain an API token from your Paperless admin account profile (`/admin/authtoken/tokenproxy/`).
2. **`rocm-ocr` Microservice**: Running and reachable over HTTP (e.g., `http://rocm-ocr:8000`).
3. **Paperless Media Volume Access**: Direct POSIX filesystem read/write access to Paperless's media directory (specifically `/media/documents/archive/`).
4. **Paperless PostgreSQL Access**: Network connectivity and credentials to Paperless's PostgreSQL database to update `documents_document.archive_checksum`.
5. **Python 3.12+** or a **Container Runtime** (Docker / Podman / Kubernetes).

---

## Quick Start

### 1. Clone & Install with `uv`

```bash
git clone https://github.com/gorki120/paperless-rocm-reindexer.git
cd paperless-rocm-reindexer

# Install dependencies using uv
uv sync
```

### 2. Configure Environment

Create a `.env` file in the project root:

```bash
cat << 'EOF' > .env
# Paperless REST API
PAPERLESS_BASE_URL=http://paperless.local:8000
PAPERLESS_API_TOKEN=your_paperless_api_token_here

# Direct storage & database access
PAPERLESS_MEDIA_ROOT=/path/to/paperless/media
PAPERLESS_DB_URL=postgresql://paperless:secret@localhost:5432/paperless

# ROCm-OCR service
ROCM_OCR_BASE_URL=http://rocm-ocr.local:8000
ROCM_OCR_API_KEY=

# Execution parameters
OCR_CONCURRENCY=4
STATE_DB_PATH=./reindexer_state.db
EOF
```

### 3. Inspect a Single Document (Safe, Read-Only)

Test OCR on document ID `42` without making any modifications:

```bash
uv run reindex inspect --doc-id 42
```

This will:
- Download the original document.
- Run ROCm OCR inference.
- Print a side-by-side terminal diff between Paperless's existing text and the new OCR text.
- Print inference time, character count, and page metrics.

### 4. Run a Dry-Run Batch

Simulate batch processing on 5 documents without modifying files or the database:

```bash
uv run reindex batch --dry-run --limit 5
```

### 5. Run Live Batch Reindex

Execute the reindexing process:

```bash
uv run reindex batch --concurrency 4
```

---

## CLI Reference

The CLI entrypoint is available as both `paperless-rocm-reindexer` and the shortcut alias `reindex`.

```text
Usage: reindex [OPTIONS] COMMAND [ARGS]...

  Paperless ROCm Reindexer CLI

Commands:
  batch    Process a batch of unprocessed documents.
  inspect  Inspect OCR extraction and text diff against Paperless for a document.
  daemon   Run continuously polling Paperless for new documents.
  stats    Display processing summary statistics from the state store.
```

### `reindex batch`

Process unprocessed documents lacking the completion tag.

```bash
reindex batch [OPTIONS]
```

| Option | Flag | Default | Description |
| :--- | :--- | :--- | :--- |
| `--concurrency` | `-c` | `4` (or `OCR_CONCURRENCY`) | Number of concurrent async workers (1–8). |
| `--limit` | `-l` | `0` (or `BATCH_LIMIT`) | Max documents to process in this run (`0` = unlimited). |
| `--dry-run` | | `false` | Run OCR inference without writing to disk or DB. |
| `--save-scratch` | | `false` | In dry-run mode, saves generated sandwich PDFs to scratch dir. |
| `--scratch-dir` | | `/tmp` | Target directory for `--save-scratch`. |

**Examples:**

```bash
# Process all unprocessed documents with 6 parallel workers
reindex batch --concurrency 6

# Process next 20 documents in dry-run mode and inspect PDFs in /tmp/test_ocr/
reindex batch --dry-run --limit 20 --save-scratch --scratch-dir /tmp/test_ocr
```

### `reindex inspect`

Inspect a single document, view terminal diffs, and benchmark inference.

```bash
reindex inspect [OPTIONS] [DOC_ID_ARG]
```

| Option | Flag | Default | Description |
| :--- | :--- | :--- | :--- |
| `--doc-id` | `-d` | Required | Document ID to inspect. Can also be passed as positional argument. |
| `--save-sandwich`| `-s` | None | File path to export the generated sandwich PDF. |
| `--save-scratch` | | `false` | Save generated PDF to `/tmp/sandwich_test_{id}.pdf`. |
| `--scratch-dir` | | `/tmp` | Directory for `--save-scratch`. |

**Example:**

```bash
reindex inspect 128 --save-sandwich ./output_128.pdf
```

### `reindex daemon`

Run continuously as a background service, periodically polling Paperless for new or newly modified documents.

```bash
reindex daemon [OPTIONS]
```

| Option | Flag | Default | Description |
| :--- | :--- | :--- | :--- |
| `--interval-sec`| `-i` | `60` (or `DAEMON_INTERVAL_SEC`)| Polling interval in seconds between passes. |
| `--concurrency` | `-c` | `4` | Worker concurrency per pass. |
| `--max-iterations`| | Unlimited | Exit cleanly after N iterations (useful for testing or batch cycles). |

The daemon registers signal handlers for `SIGINT` and `SIGTERM` to allow in-flight document processing to drain cleanly before terminating.

### `reindex stats`

Display an overview table of all processed, failed, skipped, and pending documents recorded in the SQLite state store.

```bash
reindex stats [--state-db /path/to/reindexer_state.db]
```

**Example Output:**

```text
             Reindex Processing State Summary (/data/reindexer_state.db)
┏━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┓
┃ Status   ┃ Count ┃ Percentage ┃
┡━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━┩
│ Success  │  1420 │      98.6% │
│ Failed   │     8 │       0.6% │
│ Skipped  │    12 │       0.8% │
│ Pending  │     0 │       0.0% │
├──────────┼───────┼────────────┤
│ Total    │  1440 │     100.0% │
└──────────┴───────┴────────────┘
```

---

## Configuration

All configuration settings are loaded via environment variables or a `.env` file via `pydantic-settings`.

| Environment Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `PAPERLESS_BASE_URL` | String | `http://paperless-web:8000` | Base URL of the Paperless-ngx instance. |
| `PAPERLESS_API_TOKEN` | String | `""` (*Required*) | API Token for Paperless REST authentication. |
| `PAPERLESS_MEDIA_ROOT` | Path | `/media` | Path to mounted Paperless media storage volume. |
| `PAPERLESS_DB_URL` | String | `""` (*Required*) | PostgreSQL connection string (`postgresql://user:pass@host:5432/paperless`). |
| `ROCM_OCR_BASE_URL` | String | `http://rocm-ocr:8000` | Base URL of the `rocm-ocr` microservice. |
| `ROCM_OCR_API_KEY` | String | `""` | Optional Bearer token for `rocm-ocr` authentication. |
| `OCR_CONCURRENCY` | Int | `4` | Default worker concurrency (clamped between 1 and 8). |
| `BATCH_LIMIT` | Int | `0` | Max documents to process per batch (`0` = all). |
| `TAG_SUCCESS` | String | `ocr:rocm-v4` | Tag applied in Paperless upon successful reindexing. |
| `TAG_FAILURE` | String | `ocr:failed` | Tag applied in Paperless upon permanent OCR failure. |
| `DAEMON_INTERVAL_SEC` | Int | `0` | Default polling interval in seconds for daemon mode. |
| `DRY_RUN` | Bool | `false` | If `true`, OCR runs without writing files or updating Paperless. |
| `STATE_DB_PATH` | Path | `/data/reindexer_state.db` | Path to the local SQLite state database file. |

---

## Deployment

### Kubernetes (Kustomize)

The repository provides production-ready Kubernetes manifests in the `k8s/` directory.

#### Structure

```text
k8s/
├── configmap.yaml       # Non-sensitive runtime configuration
├── secret-sample.yaml   # Sample secret (copy to secret.yaml or seal)
├── pvc.yaml             # PVC for state DB persistence (OpenEBS ZFS / Local PV)
├── job-batch.yaml       # One-shot migration Job
├── deployment.yaml      # Continuous Daemon Deployment
├── cronjob.yaml         # Nightly CronJob (runs at 02:00)
└── kustomization.yaml   # Kustomize aggregation
```

#### Step-by-Step Deployment

1. **Configure Secrets**:
   Copy `k8s/secret-sample.yaml` and enter your credentials:
   ```bash
   cp k8s/secret-sample.yaml k8s/secret.yaml
   # Edit k8s/secret.yaml with real token & postgres credentials
   ```

2. **Adjust Volume Mounts**:
   Ensure `k8s/job-batch.yaml`, `k8s/deployment.yaml`, or `k8s/cronjob.yaml` mounts the same `media` volume used by your Paperless pod:
   ```yaml
   volumeMounts:
     - name: paperless-media
       mountPath: /media
     - name: reindexer-data
       mountPath: /data
   ```

3. **Deploy One-Shot Batch Migration Job**:
   ```bash
   kubectl apply -f k8s/job-batch.yaml
   kubectl logs -n paperless -f job/paperless-rocm-reindexer-batch
   ```

4. **Or Deploy Continuous Daemon**:
   ```bash
   kubectl apply -f k8s/deployment.yaml
   kubectl logs -n paperless -f deployment/paperless-rocm-reindexer
   ```

5. **Or Deploy Nightly CronJob**:
   ```bash
   kubectl apply -f k8s/cronjob.yaml
   ```

### Docker / Container

A multi-stage, rootless container image is defined in the `Dockerfile`.

Build the image locally:

```bash
docker build -t paperless-rocm-reindexer:latest .
```

Run a one-off batch reindex:

```bash
docker run --rm -it \
  --env-file .env \
  -v /path/to/paperless/media:/media \
  -v $(pwd)/data:/data \
  paperless-rocm-reindexer:latest batch --concurrency 4
```

### Docker Compose

You can integrate `paperless-rocm-reindexer` directly into your existing Paperless Docker Compose setup:

```yaml
services:
  reindexer:
    image: ghcr.io/gorki120/paperless-rocm-reindexer:latest
    restart: unless-stopped
    command: ["daemon", "--interval-sec", "120"]
    environment:
      - PAPERLESS_BASE_URL=http://webserver:8000
      - PAPERLESS_API_TOKEN=your_token_here
      - PAPERLESS_MEDIA_ROOT=/media
      - PAPERLESS_DB_URL=postgresql://paperless:secret@db:5432/paperless
      - ROCM_OCR_BASE_URL=http://rocm-ocr:8000
      - OCR_CONCURRENCY=4
      - STATE_DB_PATH=/data/reindexer_state.db
    volumes:
      - media:/media
      - reindexer_data:/data
    depends_on:
      - webserver
      - db
```

---

## Integrity & Verification

### Paperless Sanity Check

`paperless-rocm-reindexer` ensures complete cryptographic integrity. After processing documents, verify that Paperless-ngx confirms archive consistency:

```bash
# Inside your Paperless container / pod:
python manage.py document_sanity_checker
```

If the reindexer updated the files and database correctly, `document_sanity_checker` will report zero errors or warnings.

### Atomic Replacement

Files are never written directly in-place. The engine:
1. Streams OCR output to `{archive_path}.tmp`.
2. Computes the checksum on the completed file.
3. Performs an atomic `os.replace` to overwrite the existing archive PDF.
4. Updates PostgreSQL with the new checksum.

This prevents partially-written files or corrupted states during power loss or container crashes.

### Handling Failed Documents

When a document permanently fails OCR (e.g., corrupt PDF, unsupported font, persistent timeout):
1. It is tagged with `ocr:failed` in Paperless-ngx.
2. It is logged in `reindexer_state.db` with status `failed` and the corresponding error message.
3. The engine continues processing subsequent documents without stalling.
4. To inspect the failure:
   ```bash
   reindex inspect <DOC_ID>
   ```

---

## Development & Testing

### Running Tests

The test suite includes comprehensive mocked tests using `respx`, `pytest-asyncio`, and in-memory SQLite/asyncpg mock fixtures:

```bash
# Run full test suite with coverage
uv run pytest --cov=paperless_rocm_reindexer --cov-report=term-missing
```

### Code Quality & Formatting

The project enforces strict code quality checks with `ruff` and `ty`:

```bash
# Run linting and code formatting checks
uv run ruff check .
uv run ruff format --check .

# Run type checker
uv run ty check
```

---

## Technical Documentation

For deeper architecture internals, design decisions, and implementation history:
- [Technical Specification](docs/spec.md) — Architectural boundaries, domain isolation, database contracts, and sequence diagrams.
- [Implementation Tickets](docs/tickets.md) — Implementation roadmap, task breakdown, and acceptance criteria.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
