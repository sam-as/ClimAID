"""
Utility functions for the ClimAID package.

This module provides a collection of helper utilities for handling
climate and epidemiological datasets used throughout ClimAID.

It includes functionality for:
- Data ingestion and validation
- Cleaning and preprocessing of datasets
- Temporal train/test splitting for modeling workflows
- Basic normalization and summary statistics

These utilities are designed to support reproducible and consistent
data preparation across the ClimAID pipeline.

Notes
-----
- Functions in this module are independent of modeling and reporting layers.
- Intended for internal use, but can be used externally for custom workflows.

Author
------
Avik Sam

Created
-------
November 2025
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# -----------------------------------------------------------
# 1. File handling and validation
# -----------------------------------------------------------

def load_csv_safe(filepath, parse_dates=["time"]):
    """
    Safely load a CSV file and parse datetime columns if present.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    try:
        df = pd.read_csv(filepath)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
        return df
    except Exception as e:
        raise RuntimeError(f"Error loading {filepath}: {e}")


def ensure_directory(path):
    """
    Create directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)
    return path


# -----------------------------------------------------------
# 2. Data cleaning and transformation
# -----------------------------------------------------------

def clean_numeric_column(series):
    """
    Clean messy numeric data like '7.71-09' or '295.2005/092' and convert to float.
    """
    s = series.astype(str)
    s = s.str.replace(r"[^0-9eE.\-]", "", regex=True)
    s = s.str.replace(r'(?<=\d)-(?=\d{2,}$)', 'e-', regex=True)
    return pd.to_numeric(s, errors="coerce")


def normalize_features(df, cols):
    """
    Normalize selected numeric columns to [0, 1] range using MinMaxScaler.
    """
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df


# -----------------------------------------------------------
# 3. Time-based operations
# -----------------------------------------------------------

def split_train_test(df, date_col="time", cutoff_year=2020):
    """
    Split a dataset into training and testing subsets by year.
    - Training: all data before `cutoff_year`
    - Testing: all data after `cutoff_year`
    """
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["year"] = df[date_col].dt.year

    df_train = df[df["year"] < cutoff_year].copy()
    df_test = df[df["year"] > cutoff_year].copy()
    return df_train, df_test


# -----------------------------------------------------------
# 4. Diagnostics and summaries
# -----------------------------------------------------------

def print_summary(df, label="Data"):
    """
    Print summary statistics and missing value info.
    """
    print(f"\n--- {label} Summary ---")
    print(f"Shape: {df.shape}")
    if "time" in df.columns:
        print(f"Date range: {df['time'].min()} → {df['time'].max()}")
    print(f"Missing values:\n{df.isna().sum()}")
    print("---------------------------")

# -----------------------------------------------------------
# 5. Data consistency
# -----------------------------------------------------------
def check_data_consistency(df, key_cols=["District", "time"]):
    """
    Verify that key identifiers (e.g., District, time) are unique and complete.
    """
    duplicates = df.duplicated(subset=key_cols).sum()
    if duplicates > 0:
        print(f"Warning: {duplicates} duplicate entries found based on {key_cols}")
    missing = df[key_cols].isna().sum().sum()
    if missing > 0:
        print(f"Warning: {missing} missing values in key columns {key_cols}")
    else:
        print("Data consistency check passed")

# -----------------------------------------------------------
# 6. Matplotlib backend support
# -----------------------------------------------------------
def use_gui_backend():
    import matplotlib
    matplotlib.use("TkAgg")

def use_headless_backend():
    import matplotlib
    matplotlib.use("Agg")

# -----------------------------------------------------------
# 7. JSON Helper for Numpy Floats
# -----------------------------------------------------------
def _json_safe_numbers(obj):
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")


# -----------------------------------------------------------
# 8. JSON Helper for rounding up
# -----------------------------------------------------------
def _round_numeric(self, value):
    """Round floats and float-like strings to 2 decimals."""

    import json
    import re
    from typing import Any, Dict
        
    # Case 1: float
    if isinstance(value, float):
        return round(value, 2)

    # Case 2: string containing float
    if isinstance(value, str):
        try:
            num = float(value)
            return str(round(num, 2))
        except ValueError:
            return value

    # Case 3: dictionary (recursive)
    if isinstance(value, dict):
        return {k: self._round_numeric(v) for k, v in value.items()}

    # Case 4: list / tuple
    if isinstance(value, (list, tuple)):
        return [self._round_numeric(v) for v in value]

    return value


# -----------------------------------------------------------
# 9. Helper for country_names
# -----------------------------------------------------------

COUNTRY_NAMES = {
    "IND": "India",
    "AFG": "Afghanistan",
    "BGD": "Bangladesh",
    "BTN": "Bhutan",
    "LKA": "Sri Lanka",
    "MMR": "Myanmar",
    "NPL": "Nepal",
    "PAK": "Pakistan",
}


def pretty_country(code: str) -> str:
    """
    Convert ISO3 country code to human readable name.

    Example
    -------
    >>> pretty_country("IND")
    'India'
    """

    if not code:
        return code

    code = code.upper()

    return COUNTRY_NAMES.get(code, code)


# -----------------------------------------------------------
# 10. Helper for country, state and district wizard
# -----------------------------------------------------------
from collections import defaultdict


def build_district_tree(district_list):
    """
    Convert district keys like:
    IND_pune_maharashtra

    into hierarchical structure:
    country → state → district
    """

    tree = defaultdict(lambda: defaultdict(list))

    for d in district_list:

        parts = d.split("_")

        if len(parts) >= 3:
            country = parts[0].upper()
            district = parts[1].title()
            state = parts[2].title()
        elif len(parts) == 2:
            country = "UNKNOWN"
            district = parts[0].title()
            state = parts[1].title()
        else:
            continue

        tree[country][state].append((district, d))

    return tree

# -----------------------------------------------------------
# 11. Utils for Mapping Columns names (only for Projections Data)
# Climate data is handled seperately inside the DiseaseModel module. 
# -----------------------------------------------------------
def _map_columns(df):
    """
    Helper function for Mapping Columns names (only for Projections Data)
        - Climate data is handled seperately inside the DiseaseModel module. 
    """
    from difflib import get_close_matches

    original_cols = df.columns.tolist()
    df.columns = df.columns.str.strip().str.lower()

    standard_vars = {
        "mean_Rain": ["rain", "rainfall", "precipitation", "precip", "pr"],
        "mean_temperature": ["temp", "temperature", "tas", "t2m"],
        "mean_SH": ["humidity", "specific_humidity", "huss", "hus", "rh"],
        "Nino_anomaly": ["nino", "enso", "nino34", "nino_anomaly"],
        "time": ["date", "datetime", "time"],
        "Dist_States": ["diststates", "dist_states", "dist_state", "district"]
    }

    rename_dict = {}
    mapping_log = []
    unmatched_cols = []

    for col in df.columns:
        matched = False

        for standard, aliases in standard_vars.items():

            # exact match
            if col == standard.lower() or col in aliases:
                rename_dict[col] = standard
                mapping_log.append((col, standard, "exact"))
                matched = True
                break

            # fuzzy match
            matches = get_close_matches(col, aliases, n=1, cutoff=0.8)
            if matches:
                rename_dict[col] = standard
                mapping_log.append((col, standard, "fuzzy"))
                matched = True
                break

        if not matched:
            unmatched_cols.append(col)

    df = df.rename(columns=rename_dict)

    # 🔍 Debug (optional but VERY useful)
    print("\n--- COLUMN MAPPING REPORT ---")
    for old, new, mtype in mapping_log:
        print(f"{old} -> {new} ({mtype})")

    if unmatched_cols:
        print("\nUnmapped columns:", unmatched_cols)

    return df