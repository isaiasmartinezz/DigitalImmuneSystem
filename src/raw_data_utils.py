# src/raw_data_utils.py

from __future__ import annotations

from pathlib import Path
import requests
import sys


DATA_DIR = Path("data")

# 🔒 FILL THESE IN WITH YOUR REAL LINKS
CDC_WASTEWATER_URL = "https://drive.google.com/file/d/1efHwA47W0n7nyPkzcPjz9LmvGvhw14-M/view?usp=sharing"
NHCS_STATA_URL = "https://drive.google.com/file/d/18YpIs6VUTZ4yeu25ulpXqiSWGlCk7UPC/view?usp=sharing"  # or .rds if you prefer

CDC_WASTEWATER_PATH = DATA_DIR / "CDC_Wastewater_Data_for_SARS-CoV-2.csv"
NHCS_STATA_PATH = DATA_DIR / "nhcs2021ip_stata.dta"


def _download_if_missing(url: str, dest: Path) -> Path:
    """
    Download a file from `url` to `dest` if it does not already exist.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"✅ Using existing file: {dest}")
        return dest

    if not url or "your-storage" in url:
        raise RuntimeError(
            f"No valid download URL configured for {dest.name}.\n"
            "Edit src/raw_data_utils.py and set the CDC_WASTEWATER_URL / NHCS_STATA_URL "
            "to real links (e.g., Google Drive direct link, S3, etc.)."
        )

    print(f"🌐 Downloading {dest.name} from {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print(f"💾 Saved to {dest} ({dest.stat().st_size / (1024**2):.1f} MB)")
    return dest


def ensure_cdc_wastewater() -> Path:
    """
    Ensure the CDC wastewater CSV is present locally.
    Returns the Path to the CSV.
    """
    return _download_if_missing(CDC_WASTEWATER_URL, CDC_WASTEWATER_PATH)


def ensure_nhcs_stata() -> Path:
    """
    Ensure the NHCS Stata (.dta) file is present locally.
    Returns the Path to the .dta file.
    """
    return _download_if_missing(NHCS_STATA_URL, NHCS_STATA_PATH)


if __name__ == "__main__":
    # quick manual test
    print("Python executable:", sys.executable)
    ensure_cdc_wastewater()
    ensure_nhcs_stata()
    print("\n✅ All raw data files are present.")