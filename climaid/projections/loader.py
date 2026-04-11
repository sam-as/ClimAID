# climaid/projections/loader.py
"""
Projection Data Loader for ClimAID.

Provides a user-friendly API to load CMIP6 climate projections
for different regions.

Internally:
- uses DatasetManager for downloading/caching
- loads Parquet datasets into pandas DataFrames
"""


import pandas as pd
from climaid.datasets.manager import DatasetManager

_manager = DatasetManager()


def load_cmip6(region="india", columns=None):
    """
    Retrieve a dataset by name.

    If the dataset is not already cached locally, it will be downloaded
    from the remote source (e.g., HuggingFace or Zenodo).

    Parameters
    ----------

    dataset_name : str 
        Key identifying the dataset (must exist in registry).

    Returns
    -------

    Path : pathlib.Path 
        Local file path to the dataset.

    Raises
    ------

    ValueError :
        If dataset_name is not defined in the registry.

    Notes
    -----

    - Datasets are cached under ~/.climaid/datasets/
    - Subsequent calls return the cached file without re-downloading
    """

    mapping = {
        "india": "cmip6_india",
        "south_asia": "cmip6_south_asia"
    }

    if region not in mapping:
        raise ValueError("region must be 'india' or 'south_asia'")

    dataset_name = mapping[region]

    path = _manager.fetch(dataset_name)

    return pd.read_parquet(path, columns=columns)