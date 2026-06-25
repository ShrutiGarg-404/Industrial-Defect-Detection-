"""
download_models.py
Downloads trained PatchCore model files from Google Drive on first run.
Called automatically by streamlit_app.py if models are missing.
"""

import os
from pathlib import Path

MODEL_FILES = {
    "patchcore_leather_full.pth":   "1lhNQH93IxNG912V3pLguV0hxjzE9PL1a",
    "patchcore_tile_full.pth":      "1NoR3YZgjEfIGWHPCOLoszWyJQS2oxokG",
    "patchcore_metal_nut_full.pth": "1oM0wVKelaDjzRTCncqthBUI0PzHy1P0z",
    "patchcore_wood_full.pth":      "1ZYoTGhYfJCBBORmkH7gyZ8uOpUZhQYuK",
}


def download_models(models_dir: Path):
    """Download all model files if not already present."""
    models_dir.mkdir(parents=True, exist_ok=True)

    try:
        import gdown
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
        import gdown

    for filename, file_id in MODEL_FILES.items():
        dest = models_dir / filename
        if dest.exists():
            print(f"  {filename} — already exists, skipping")
            continue
        print(f"  Downloading {filename}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(dest), quiet=False)
        print(f"  {filename} — done")


if __name__ == "__main__":
    models_dir = Path(__file__).parent / "outputs" / "models"
    print("Downloading DefectLens model files...")
    download_models(models_dir)
    print("All models ready.")