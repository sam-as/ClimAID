"""
Dataset Registry for ClimAID.

This file defines metadata for all available datasets.

Each dataset entry includes:
    - version: dataset version (for cache separation)
    - url: remote download location
    - filename: local filename

Notes
-----
- Dataset names (keys) are used throughout the library.
- These must match calls in DatasetManager.fetch().
- Changing a version will trigger re-download into a new cache folder.
"""


DATASETS = {
    "cmip6_india": {
        "version": "test",
        "url": "https://huggingface.co/datasets/sam-as/CMIP6-India/resolve/main/India_projections.parquet",
        "filename": "India_projections.parquet",
    },
    "cmip6_south_asia": {
        "version": "test",
        "url": "https://huggingface.co/datasets/sam-as/CMIP-SouthAsia/resolve/main/SouthAsia_projections.parquet",
        "filename": "SouthAsia_projections.parquet",
    }
}