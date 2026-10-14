"""Storage and file operations for Paperless documents."""

import hashlib
from pathlib import Path
from typing import Literal


class ArchiveFileManager:
    """Manages archive PDF files under Paperless media directory."""

    def __init__(
        self,
        media_root: str | Path,
        default_checksum_type: Literal["md5", "sha256"] = "md5",
        default_file_mode: int | None = 0o644,
    ) -> None:
        """Initialize ArchiveFileManager.

        Args:
            media_root: Path to Paperless media root directory.
            default_checksum_type: Default checksum algorithm ('md5' or 'sha256').
            default_file_mode: Default file permissions mode (0o644).
        """
        self.media_root = Path(media_root)
        self.default_checksum_type = default_checksum_type
        self.default_file_mode = default_file_mode

    def get_archive_path(self, archive_filename: str) -> Path:
        """Resolve archive file path under {media_root}/documents/archive/{archive_filename}.

        Args:
            archive_filename: Relative archive filename from documents_document.

        Returns:
            Resolved absolute or relative Path to the archive file.
        """
        return self.media_root / "documents" / "archive" / archive_filename

    def calculate_checksum(
        self,
        data: bytes,
        algorithm: Literal["md5", "sha256"] | None = None,
        existing_checksum: str | None = None,
    ) -> str:
        """Calculate hexadecimal checksum of data.

        Auto-detects algorithm based on existing_checksum length (32 chars -> MD5, 64 chars -> SHA256)
        if algorithm is not explicitly provided. Otherwise falls back to default_checksum_type.

        Args:
            data: Binary content.
            algorithm: Explicit algorithm ('md5' or 'sha256').
            existing_checksum: Optional existing checksum string to infer type from.

        Returns:
            Hexadecimal checksum string (lowercase).
        """
        algo_name = algorithm
        if algo_name is None:
            if existing_checksum is not None:
                cleaned = existing_checksum.strip()
                if len(cleaned) == 32:
                    algo_name = "md5"
                elif len(cleaned) == 64:
                    algo_name = "sha256"

        if algo_name is None:
            algo_name = self.default_checksum_type

        if algo_name == "md5":
            return hashlib.md5(data).hexdigest()
        elif algo_name == "sha256":
            return hashlib.sha256(data).hexdigest()
        else:
            raise ValueError(f"Unsupported checksum algorithm: {algo_name}")

    def write_archive_file(
        self,
        archive_filename: str,
        content: bytes,
        file_mode: int | None = 0o644,
    ) -> Path:
        """Atomically write binary sandwich PDF to archive file path.

        Writes to sibling temporary file `{path}.tmp` and replaces atomically with `temp_path.replace`.
        Ensures parent directory exists and cleans up temporary file on failure.

        Args:
            archive_filename: Relative archive filename.
            content: Binary PDF content.
            file_mode: Optional file permissions mode (defaults to 0o644).

        Returns:
            Path to written archive file.
        """
        target_path = self.get_archive_path(archive_filename)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        temp_path = target_path.with_name(f"{target_path.name}.tmp")
        effective_mode = file_mode if file_mode is not None else self.default_file_mode
        try:
            temp_path.write_bytes(content)
            if effective_mode is not None:
                temp_path.chmod(effective_mode)
            temp_path.replace(target_path)
            return target_path
        except Exception:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise
