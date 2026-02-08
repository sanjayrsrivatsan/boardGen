#!/usr/bin/env python3
"""
Stage 1.1: Database Acquisition

Downloads SQLite databases for supported climbing boards using boardlib.
Each database contains climbs, placements, holes, and metadata.

Usage:
    python data/download.py --boards tension kilter moonboard
    python data/download.py --boards tension  # single board
"""

import argparse
import subprocess
import sys
from pathlib import Path

SUPPORTED_BOARDS = ["tension", "kilter", "moonboard"]


def download_database(board: str, output_dir: Path) -> Path:
    """Download database for a single board using boardlib."""
    output_path = output_dir / f"{board}.sqlite"

    print(f"Downloading {board} database to {output_path}...")

    try:
        result = subprocess.run(
            ["boardlib", "database", board, str(output_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"  Successfully downloaded {board} database")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"  Error downloading {board}: {e.stderr}")
        raise
    except FileNotFoundError:
        print("Error: boardlib not found. Install with: pip install boardlib")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download climbing board databases")
    parser.add_argument(
        "--boards",
        nargs="+",
        choices=SUPPORTED_BOARDS,
        default=SUPPORTED_BOARDS,
        help="Boards to download (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for databases",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for board in args.boards:
        try:
            download_database(board, args.output_dir)
        except Exception as e:
            print(f"Failed to download {board}: {e}")
            continue

    print("\nDownload complete!")


if __name__ == "__main__":
    main()
