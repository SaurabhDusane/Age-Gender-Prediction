#!/usr/bin/env python3
"""
Download pre-trained models for the demographic analysis system
"""
import os
import urllib.request
from pathlib import Path
from tqdm import tqdm


MODEL_URLS = {
    "yolov8n-face.pt": "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt",
}


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: str):
    """Download a file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def main():
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("Downloading pre-trained models...")
    
    for model_name, url in MODEL_URLS.items():
        output_path = models_dir / model_name
        
        if output_path.exists():
            print(f"✓ {model_name} already exists, skipping...")
            continue
        
        try:
            print(f"Downloading {model_name}...")
            download_file(url, str(output_path))
            print(f"✓ Downloaded {model_name}")
        except Exception as e:
            print(f"✗ Failed to download {model_name}: {e}")
    
    print("\nNote: Some models require manual download due to licensing:")
    print("- DEX age estimation: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/")
    print("- FairFace: https://github.com/dchen236/FairFace")
    print("- ArcFace: https://github.com/deepinsight/insightface")
    print("\nPlace downloaded models in the 'models/' directory")


if __name__ == "__main__":
    main()
