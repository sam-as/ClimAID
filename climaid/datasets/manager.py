#climaid/climaid/datasets/manager.py

"""
    Dataset Manager Module for ClimAID.

    This module handles:
    - Remote dataset downloading (HuggingFace / Zenodo)
    - Local caching of large datasets
    - Transparent reuse of cached data

    Datasets are downloaded only once and stored in:
        ~/.climaid/datasets/

    Subsequent calls reuse the cached file (offline support).

"""

import pooch
from pathlib import Path
from .registry import DATASETS

CACHE_DIR = Path.home() / ".climaid" / "datasets"

print("LOADED REGISTRY FROM...", __file__)
print("DATASETS CONTENT...", DATASETS)

class DatasetManager:
    """
    Manages retrieval and caching of external datasets.

    This class ensures that:

    - datasets are downloaded only once
    - files are stored locally for reuse
    - users can work offline after first download
    
    """

    def __init__(self):
        self.base_dir = CACHE_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, dataset_name: str):

        if dataset_name not in DATASETS:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        meta = DATASETS[dataset_name]

        dataset_dir = self.base_dir / dataset_name / meta["version"]
        dataset_dir.mkdir(parents=True, exist_ok=True)

        file_path = dataset_dir / meta["filename"]

        # already cached
        if file_path.exists():
            return file_path

        print(f"\nDownloading {dataset_name}...")
        print(f"Saving to: {file_path}\n")

        path = pooch.retrieve(
            url=meta["url"],
            fname=meta["filename"],
            path=dataset_dir,
            progressbar=True,
            known_hash=None,
        )

        return Path(path)