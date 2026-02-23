#!/usr/bin/env python3
"""Download the UCI Online Retail dataset into data/raw."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
UCI_XLSX_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
)
ZIP_NAME = "online_retail.zip"
XLSX_NAME = "Online Retail.xlsx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download UCI Online Retail dataset into data/raw."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to store downloaded raw data files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files if present.",
    )
    return parser.parse_args()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def extract_zip(zip_path: Path, out_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        xlsx_members = [member for member in members if member.endswith(".xlsx")]
        if not xlsx_members:
            raise RuntimeError("No .xlsx file found inside downloaded zip archive.")
        member = xlsx_members[0]
        target_path = out_dir / Path(member).name
        with zf.open(member) as src, target_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return target_path


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = out_dir / ZIP_NAME
    xlsx_path = out_dir / XLSX_NAME

    if xlsx_path.exists() and not args.force:
        print(f"Dataset already exists: {xlsx_path}")
        return

    try:
        print(f"Downloading zip archive from: {UCI_ZIP_URL}")
        download(UCI_ZIP_URL, zip_path)
        extracted = extract_zip(zip_path, out_dir)
        print(f"Saved dataset: {extracted}")
    except Exception as err:
        print(f"Zip download failed ({err}). Falling back to direct xlsx download.")
        print(f"Downloading xlsx file from: {UCI_XLSX_URL}")
        download(UCI_XLSX_URL, xlsx_path)
        print(f"Saved dataset: {xlsx_path}")


if __name__ == "__main__":
    main()
"""Dataset download utilities (planned for Day 2)."""
