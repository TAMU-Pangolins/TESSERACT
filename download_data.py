#!/usr/bin/env python3
"""Install the complete hosted HFB dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from nucres.config import resolve_data_root, store_data_root
from nucres.data import ensure_hfb_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download, verify, and extract all HFB level-density tables and "
            "correction files from aldusv/TESSERACT-data."
        )
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination directory (default: active nucres data directory).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download and reinstall the dataset even when it is complete.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress download progress.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    destination = args.dest.expanduser() if args.dest else resolve_data_root()
    installed = ensure_hfb_dataset(
        destination,
        force=args.force,
        show_progress=not args.quiet,
    )
    store_data_root(installed)
    print(f"HFB dataset ready: {installed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
