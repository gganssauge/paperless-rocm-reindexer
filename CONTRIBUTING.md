# Contributing to Paperless ROCm Reindexer

Thank you for your interest in contributing to `paperless-rocm-reindexer`!

## Code of Conduct

All contributors are expected to uphold our [Code of Conduct](CODE_OF_CONDUCT.md). Please report any unacceptable behavior to `ganssauge@gorki120.de`.

## Development Setup

The project uses [`uv`](https://docs.astral.sh/uv/) for Python package and virtual environment management.

1. **Clone the repository:**

   ```bash
   git clone https://github.com/gganssauge/paperless-rocm-reindexer.git
   cd paperless-rocm-reindexer
   ```

2. **Set up Python 3.14 and dependencies:**

   ```bash
   uv python install 3.14
   uv sync --dev
   ```

3. **Install Pre-Commit Hooks:**
   We use `prek` (configured via `.pre-commit-config.yaml`):

   ```bash
   uv run prek install
   ```

## Development Workflow

- **Branching Strategy:** Direct pushes to `main` are strictly forbidden. Always create a feature branch:

  ```bash
  git checkout -b feature/<descriptive-name>
  ```

- **Commit Messages:** Follow the [Conventional Commits](https://www.conventionalcommits.org/) format:
  - `feat: support batch OCR reprocessing`
  - `fix: handle paperless rate limit error`
  - `docs: update operational runbook`
  - `chore: update dependencies`

## Code Standards & Quality Gates

- **Formatting & Linting:**

  ```bash
  uv run ruff check .
  uv run ruff format --check .
  ```

- **Type Checking:**

  ```bash
  uv run ty check
  ```

- **Testing & Coverage:**

  ```bash
  uv run pytest --cov=paperless_rocm_reindexer --cov-report=term-missing
  ```

- **Function Metrics:**
  - Keep functions focused and concise (maximum 30 logical lines).
  - Maximum cognitive complexity per function/method: $\le 15$ (SonarCloud standard).

## Pull Request Guidelines

1. Ensure all tests and lint checks pass locally before opening a PR.
2. Link the relevant issue in the PR description (e.g., `Closes #12`).
3. Keep PRs focused on a single responsibility.
