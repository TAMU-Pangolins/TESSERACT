"""Download, verify, and cache the hosted HFB level-density dataset."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from .config import resolve_data_root

DATA_BASE_URL_ENV = "NUCRES_DATA_BASE_URL"
DEFAULT_DATA_BASE_URL = (
    "https://raw.githubusercontent.com/aldusv/TESSERACT-data/main/hfb"
)
DATASET_ARCHIVE_NAME = "hfb-level-densities-v1.zip"
DATASET_ARCHIVE_SHA256 = (
    "8dc17d749013da8d38fe48ca7d665577d91eef5b7e6f097b8dfd3cab9702275e"
)
DATASET_SENTINEL = ".tesseract-hfb-v1"
USER_AGENT = "THICC-nucres-data/1.0"
DOWNLOAD_TIMEOUT_SECONDS = 120
TAB_Z_RANGE = range(8, 110)
COR_Z_RANGE = range(9, 101)


def data_base_url() -> str:
    """Return the configured base URL for hosted HFB data."""
    return os.environ.get(DATA_BASE_URL_ENV, DEFAULT_DATA_BASE_URL).rstrip("/")


def expected_hfb_filenames() -> tuple[str, ...]:
    """Return all filenames in the hosted HFB dataset."""
    tabs = tuple(f"z{Z:03d}.tab" for Z in TAB_Z_RANGE)
    corrections = tuple(f"z{Z:03d}.cor" for Z in COR_Z_RANGE)
    return tabs + corrections


def _dataset_complete(root: Path) -> bool:
    return all((root / name).is_file() for name in expected_hfb_filenames())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    url: str,
    destination: Path,
    *,
    sha256: str,
    show_progress: bool = True,
) -> None:
    """Download one file atomically and verify its SHA-256 digest."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    digest = hashlib.sha256()

    try:
        if show_progress:
            print(f"Downloading {destination.name} ...")
        with urllib.request.urlopen(
            request, timeout=DOWNLOAD_TIMEOUT_SECONDS
        ) as response, temporary.open("wb") as handle:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)
                if show_progress and total:
                    percent = min(100, downloaded * 100 // total)
                    print(
                        f"\r  {percent:3d}%  "
                        f"{downloaded / 1e6:.1f}/{total / 1e6:.1f} MB",
                        end="",
                        flush=True,
                    )
        if show_progress and total:
            print()
    except (OSError, urllib.error.URLError) as exc:
        temporary.unlink(missing_ok=True)
        raise FileNotFoundError(f"Unable to download HFB data from {url}: {exc}") from exc

    actual_sha256 = digest.hexdigest()
    if actual_sha256 != sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA-256 mismatch for {destination.name}: "
            f"expected {sha256}, got {actual_sha256}"
        )
    temporary.replace(destination)


def _safe_extract_archive(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive, "r") as bundle:
        for member in bundle.infolist():
            file_type = (member.external_attr >> 16) & 0o170000
            if file_type == stat.S_IFLNK:
                raise ValueError(f"Archive link is not allowed: {member.filename}")
            target = (destination / member.filename).resolve()
            if (
                target != destination_resolved
                and destination_resolved not in target.parents
            ):
                raise ValueError(
                    f"Archive entry would escape the destination: {member.filename}"
                )
        bundle.extractall(destination)


def _write_sentinel(root: Path) -> None:
    try:
        (root / DATASET_SENTINEL).write_text(
            f"{DATASET_ARCHIVE_NAME} {DATASET_ARCHIVE_SHA256}\n",
            encoding="ascii",
        )
    except OSError:
        # A complete, read-only user-provided dataset remains usable.
        pass


def ensure_hfb_dataset(
    data_root: str | Path | None = None,
    *,
    force: bool = False,
    show_progress: bool = True,
    archive_url: str | None = None,
    archive_sha256: str = DATASET_ARCHIVE_SHA256,
) -> Path:
    """Ensure the complete hosted HFB table and correction set is installed."""
    root = Path(data_root).expanduser() if data_root is not None else resolve_data_root()
    if not force and _dataset_complete(root):
        _write_sentinel(root)
        return root

    root.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = root.parent / ".nucres-download-cache"
    archive = cache_dir / DATASET_ARCHIVE_NAME
    url = archive_url or f"{data_base_url()}/{DATASET_ARCHIVE_NAME}"

    archive_is_valid = (
        archive.is_file() and _sha256_file(archive) == archive_sha256
    )
    if force or not archive_is_valid:
        download_file(
            url,
            archive,
            sha256=archive_sha256,
            show_progress=show_progress,
        )

    with tempfile.TemporaryDirectory(
        prefix=".hfb-extract-", dir=root.parent
    ) as temporary_dir:
        extracted = Path(temporary_dir)
        _safe_extract_archive(archive, extracted)
        if not _dataset_complete(extracted):
            raise RuntimeError(
                f"{DATASET_ARCHIVE_NAME} does not contain the complete HFB dataset"
            )

        root.mkdir(parents=True, exist_ok=True)
        for name in expected_hfb_filenames():
            shutil.move(str(extracted / name), root / name)

    _write_sentinel(root)
    return root
