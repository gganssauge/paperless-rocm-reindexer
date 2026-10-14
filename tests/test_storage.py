import hashlib
from pathlib import Path

import pytest

from paperless_rocm_reindexer.storage import ArchiveFileManager


def test_archive_file_manager_init(tmp_path: Path):
    manager = ArchiveFileManager(media_root=tmp_path)
    assert manager.media_root == tmp_path
    assert manager.default_checksum_type == "md5"

    manager_sha = ArchiveFileManager(
        media_root=str(tmp_path), default_checksum_type="sha256"
    )
    assert manager_sha.media_root == tmp_path
    assert manager_sha.default_checksum_type == "sha256"


def test_get_archive_path(tmp_path: Path):
    manager = ArchiveFileManager(media_root=tmp_path)
    path = manager.get_archive_path("2023/05/doc-0001.pdf")
    expected = tmp_path / "documents" / "archive" / "2023/05/doc-0001.pdf"
    assert path == expected


def test_calculate_checksum(tmp_path: Path):
    manager = ArchiveFileManager(media_root=tmp_path)
    content = b"Sample PDF document binary content"

    md5_expected = hashlib.md5(content).hexdigest()
    sha256_expected = hashlib.sha256(content).hexdigest()

    # Default md5
    assert manager.calculate_checksum(content) == md5_expected

    # Explicit algo
    assert manager.calculate_checksum(content, algorithm="md5") == md5_expected
    assert manager.calculate_checksum(content, algorithm="sha256") == sha256_expected

    # Auto-detect from existing 32 chars (md5)
    dummy_md5 = "a" * 32
    assert (
        manager.calculate_checksum(content, existing_checksum=dummy_md5) == md5_expected
    )

    # Auto-detect from existing 64 chars (sha256)
    dummy_sha256 = "b" * 64
    assert (
        manager.calculate_checksum(content, existing_checksum=dummy_sha256)
        == sha256_expected
    )

    # Fallback to default if existing checksum is invalid length
    assert (
        manager.calculate_checksum(content, existing_checksum="invalid") == md5_expected
    )

    # Invalid algo explicitly passed
    with pytest.raises(ValueError, match="Unsupported checksum algorithm"):
        manager.calculate_checksum(content, algorithm="sha1")  # type: ignore


def test_write_archive_file_atomic(tmp_path: Path):
    manager = ArchiveFileManager(media_root=tmp_path)
    rel_path = "sub/dir/doc-123.pdf"
    content = b"%PDF-1.4 dummy pdf bytes"

    target_path = manager.write_archive_file(rel_path, content)

    assert target_path.exists()
    assert target_path.read_bytes() == content
    assert not target_path.with_name("doc-123.pdf.tmp").exists()


def test_write_archive_file_overwrite(tmp_path: Path):
    manager = ArchiveFileManager(media_root=tmp_path)
    rel_path = "doc-456.pdf"
    old_content = b"old content"
    new_content = b"new sandwich pdf content"

    manager.write_archive_file(rel_path, old_content)
    assert manager.get_archive_path(rel_path).read_bytes() == old_content

    manager.write_archive_file(rel_path, new_content)
    assert manager.get_archive_path(rel_path).read_bytes() == new_content
    assert not (tmp_path / "documents" / "archive" / "doc-456.pdf.tmp").exists()


def test_write_archive_file_cleanup_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manager = ArchiveFileManager(media_root=tmp_path)
    rel_path = "doc-err.pdf"
    content = b"content"

    def faulty_replace(self, target):
        raise OSError("Disk failure during replace")

    monkeypatch.setattr(Path, "replace", faulty_replace)

    with pytest.raises(OSError, match="Disk failure during replace"):
        manager.write_archive_file(rel_path, content)

    target_path = manager.get_archive_path(rel_path)
    assert not target_path.exists()
    assert not target_path.with_name("doc-err.pdf.tmp").exists()


def test_write_archive_file_permissions(tmp_path: Path):
    manager = ArchiveFileManager(media_root=tmp_path)
    rel_path = "perm-doc.pdf"
    content = b"data"

    # Default file_mode should be 0o644
    target_path = manager.write_archive_file(rel_path, content)
    assert target_path.exists()
    mode = target_path.stat().st_mode & 0o777
    assert mode == 0o644

    # Explicit file_mode override
    rel_path_custom = "perm-doc-custom.pdf"
    target_custom = manager.write_archive_file(
        rel_path_custom, content, file_mode=0o600
    )
    assert target_custom.exists()
    assert (target_custom.stat().st_mode & 0o777) == 0o600
