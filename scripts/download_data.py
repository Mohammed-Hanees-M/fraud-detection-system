#!/usr/bin/env python3
"""
download_data.py
────────────────
Downloads the Credit Card Fraud Detection dataset from Kaggle.

Usage:
    python scripts/download_data.py

Prerequisites:
    1. Install the kaggle CLI:   pip install kaggle
    2. Create a Kaggle API token at https://www.kaggle.com/settings → API → Create New Token
    3. Place the downloaded kaggle.json in ~/.kaggle/kaggle.json
       (Linux/Mac) or %USERPROFILE%\.kaggle\kaggle.json (Windows)
    4. Ensure permissions: chmod 600 ~/.kaggle/kaggle.json  (Linux/Mac only)
"""

import os
import subprocess
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET = "mlg-ulb/creditcardfraud"
CSV_NAME = "creditcard.csv"


def download():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / CSV_NAME

    if csv_path.exists():
        print(f"[✓] Dataset already present at {csv_path}")
        return

    print(f"[→] Downloading '{DATASET}' from Kaggle …")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(DATA_DIR)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("[✗] Kaggle download failed:\n", result.stderr)
        print(
            "\nManual alternative:\n"
            "  1. Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "  2. Click 'Download' → save the zip\n"
            f"  3. Extract creditcard.csv into  {DATA_DIR}/\n"
        )
        raise SystemExit(1)

    # Unzip
    zip_path = DATA_DIR / "creditcardfraud.zip"
    if zip_path.exists():
        print("[→] Extracting …")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)
        zip_path.unlink()

    print(f"[✓] Dataset ready at {csv_path}")


if __name__ == "__main__":
    download()
