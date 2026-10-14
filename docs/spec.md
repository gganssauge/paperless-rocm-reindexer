# Specification: `paperless-rocm-reindexer`

- **Project:** `paperless-rocm-reindexer`
- **Component:** Batch & Continuous Reindexing Engine for Paperless-ngx powered by `rocm-ocr`
- **Target Architecture:** Python 3.14, Astral `uv`, PostgreSQL, Paperless-ngx REST API, Kubernetes
- **Inference Engine:** `rocm-ocr` (AMD ROCm / RapidOCR PP-OCRv4 on AMD Radeon RX 7800 XT)
- **Implementation Tickets:** [tickets.md](tickets.md)

---

## 1. Executive Summary & Objective

`paperless-rocm-reindexer` is a standalone, decoupled utility service designed to re-process and re-OCR documents stored in Paperless-ngx using the dedicated `rocm-ocr` microservice.

### Key Capabilities

1. **Clean Extraction:** Pulls pristine original document files (`original=true`) to avoid OCR artifact accumulation.
2. **Dual-Layer Upgrade:**
   - Replaces the visual archive PDF on disk with high-fidelity searchable Sandwich PDFs (`POST /ocr/v1/sandwich`).
   - Updates the fulltext index (`content`) via Paperless REST API (`PATCH /api/documents/{id}/`).
3. **Database Integrity:** Computes cryptographic hashes (SHA256/MD5) of new archive files and synchronizes Paperless PostgreSQL `documents_document.archive_checksum`, keeping `document_sanity_checker` 100% compliant.
4. **Resilient Hardware Concurrency:** Leverages dynamic concurrency scaling via `POST /ocr/v1/setup` and performs adaptive backoff on GPU VRAM exhaustion (HTTP 503).
5. **Idempotent Dual-Operation:** Operates either as a one-shot CLI migration tool or as an autonomous Kubernetes Daemon / CronJob.

---

## 2. Architectural Boundaries & Domain Isolation

To preserve universal service invariants:
- **`rocm-ocr`** remains strictly domain-agnostic and stateless. It has zero knowledge of Paperless-ngx schemas or APIs.
- **`paperless-rocm-reindexer`** acts as the orchestrator bridging the Paperless storage/API boundary and the `rocm-ocr` inference service.

```mermaid
flowchart TD
    subgraph Storage ["Persistent Storage & DB"]
        MediaArchive[("Paperless Media Archive\n/usr/src/paperless/media/documents/archive")]
        PG[("Paperless PostgreSQL\ntable: documents_document")]
    end

    subgraph Service ["paperless-rocm-reindexer"]
        CLI["CLI / Daemon Engine (Typer / Async)"]
        StateDB[("Local State Store\nSQLite: reindex_state.db")]
        ClientPaperless["Paperless REST Client"]
        ClientROCM["ROCm-OCR HTTP Client"]
    end

    subgraph OCR ["rocm-ocr Microservice"]
        ROCMService["FastAPI Inference Engine\n(PP-OCRv4 / RX 7800 XT)"]
    end

    CLI -->|Query Docs & Tag| ClientPaperless
    ClientPaperless -->|GET /api/documents/| PaperlessAPI["Paperless REST API"]
    PaperlessAPI -->|Stream Original| CLI

    CLI -->|POST /ocr/v1/sandwich| ClientROCM
    ClientROCM --> ROCMService
    ROCMService -->|Return Sandwich PDF| ClientROCM

    CLI -->|1. Overwrite PDF| MediaArchive
    CLI -->|2. UPDATE archive_checksum| PG
    CLI -->|3. PATCH /api/documents/{id}/| ClientPaperless
    CLI -->|4. Log Checkpoint| StateDB
```

---

## 3. Interfaces & Data Contracts

### 3.1 External Dependencies & Connection Requirements

| Component | Protocol | Auth / Mechanism | Target Role |
| :--- | :--- | :--- | :--- |
| **Paperless-ngx API** | HTTP/REST | `Authorization: Token <TOKEN>` | Fetch docs, download originals, patch `content`, manage tags |
| **`rocm-ocr` Service** | HTTP/REST | Optional Bearer `API_KEY` | Generate sandwich PDFs (`/ocr/v1/sandwich`), scale concurrency (`/ocr/v1/setup`) |
| **Paperless Media Volume**| POSIX FS | Direct Mount (`rw`) | Overwrite `archive_filename` PDF binaries |
| **Paperless PostgreSQL** | TCP/SQL | Postgres Credentials | Update `archive_checksum` on `documents_document` |

### 3.2 Dynamic Concurrency & Adaptive Backoff

1. **Initial Ramp-Up:** Before initiating bulk operations, the reindexer queries `/ocr/v1/setup` or sets initial batch concurrency (default: 4, configurable up to 8).
2. **Error Interception:**
   - If `rocm-ocr` returns **HTTP 503** (ROCm GPU inference failure / HIP out of memory):
     - The reindexer dynamically posts `{"concurrency": 1}` to `rocm-ocr/setup`.
     - Retries the failed document once in isolation.
     - If the retry succeeds, concurrency is gradually ramped back up.
     - If the retry fails, the document is flagged `ocr:failed` in Paperless and recorded as `failed` in SQLite. The pipeline continues uninterrupted.

### 3.3 State Tracking Schema (Local SQLite)

Database location: `/data/reindexer_state.db` (or configurable via `STATE_DB_PATH`).

```sql
CREATE TABLE IF NOT EXISTS reindex_log (
    document_id INTEGER PRIMARY KEY,
    title TEXT,
    page_count INTEGER,
    original_size_bytes INTEGER,
    archive_size_bytes INTEGER,
    old_checksum TEXT,
    new_checksum TEXT,
    inference_time_ms REAL,
    status TEXT CHECK(status IN ('pending', 'success', 'failed', 'skipped')),
    error_message TEXT,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_reindex_status ON reindex_log(status);
```

---

## 4. Processing Pipeline Sequence

For each eligible document (documents lacking the `ocr:rocm-v4` tag and not marked `ocr:failed`):

```mermaid
sequenceDiagram
    autonumber
    participant R as Reindexer Worker
    participant P_API as Paperless API
    participant ROCM as rocm-ocr
    participant FS as Archive Mount
    participant DB as Paperless Postgres
    participant S as SQLite State

    R->>P_API: GET /api/documents/?tags__name__iexact!=ocr:rocm-v4
    R->>P_API: GET /api/documents/{id}/download/?original=true
    P_API-->>R: Binary Original (PDF/Image)
    
    R->>ROCM: POST /ocr/v1/sandwich (multipart file)
    alt HTTP 503 GPU OOM
        R->>ROCM: POST /ocr/v1/setup (concurrency=1)
        R->>ROCM: POST /ocr/v1/sandwich (retry)
    end
    ROCM-->>R: Binary Sandwich PDF + X-Headers
    
    R->>FS: Write binary to media/documents/archive/{archive_filename}
    R->>R: Calculate SHA256 / MD5 checksum
    R->>DB: UPDATE documents_document SET archive_checksum = $hash WHERE id = $id
    R->>P_API: PATCH /api/documents/{id}/ { content: $extracted_text, tags: [...tags, ocr:rocm-v4] }
    R->>S: INSERT INTO reindex_log (status='success')
```

---

## 5. Configuration & Environment Variables

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `PAPERLESS_BASE_URL` | String | `http://paperless-web:8000` | Base URL of Paperless-ngx |
| `PAPERLESS_API_TOKEN` | Secret | Required | API Token for Paperless authentication |
| `PAPERLESS_MEDIA_ROOT` | Path | `/media` | Path to mounted Paperless media folder |
| `PAPERLESS_DB_URL` | Secret | Required | PostgreSQL connection string (`postgresql://user:pass@host:5432/paperless`) |
| `ROCM_OCR_BASE_URL` | String | `http://rocm-ocr:8000` | Base URL of `rocm-ocr` service |
| `ROCM_OCR_API_KEY` | Secret | `""` | Optional Bearer token for `rocm-ocr` |
| `OCR_CONCURRENCY` | Int | `4` | Number of parallel worker threads / async slots |
| `BATCH_LIMIT` | Int | `0` | Max documents to process (0 = all) |
| `TAG_SUCCESS` | String | `ocr:rocm-v4` | Tag applied on successful reindex |
| `TAG_FAILURE` | String | `ocr:failed` | Tag applied on permanent failure |
| `DAEMON_INTERVAL_SEC` | Int | `0` | If > 0, runs indefinitely polling every N seconds |
| `DRY_RUN` | Bool | `false` | If true, processes files locally without writing to Paperless |

---

## 6. Safety Gates & Verification

1. **`--dry-run` & Inspection:** Allows running against a single document (`--doc-id 42`) or small batch (`--limit 5`). Dumps extracted text diffs and outputs the generated sandwich PDF to a local scratch folder for visual inspection.
2. **Paperless Sanity Check:** Verification suite runs `python manage.py document_sanity_checker` inside Paperless to ensure zero archive discrepancies.
3. **Atomic File Replacement:** Sandwich PDF is written to a temporary sibling file (`{filename}.tmp`) and atomically renamed (`os.replace`) to prevent corrupt states during unexpected power or process termination.
