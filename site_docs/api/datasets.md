# Datasets

Utilities for accessing and managing climate datasets in ClimAID.

---

## Load Data

High-level function for loading climate projection datasets.

::: climaid.projections.loader.load_cmip6
    options:
      heading_level: 3
      filters:
        - "!^_"

---

## Dataset Management (Advanced)

Handles dataset downloading, caching, and retrieval.

::: climaid.datasets.manager.DatasetManager
    options:
      heading_level: 3
      filters:
        - "!^_"

---

## Available Datasets (Internal)

Defines dataset metadata such as source URLs and versions.

::: climaid.datasets.registry.DATASETS