"""
districts.py
------------

Provides helper functions for listing and exploring available districts
in the ClimAID dataset.

Author: Avik Kumar Sam
Created: November 2025
"""

# import the key libraries
import pandas as pd
from pathlib import Path
from .utils import load_csv_safe

# reading the data
DATA_DIR = Path(__file__).resolve().parent / "data"
CLIMATE_FILE = DATA_DIR / "SouthAsia_weather_data.csv"   


def get_available_districts(filepath: str = None):
    """
    Load available district names from the climate data file.

    Parameters
    ----------
    filepath : str, optional
        Custom path to the data file. If None, defaults to 'data/historical.csv'.

    Returns
    -------
    list
        Sorted list of district names (e.g., ['Pune_Maharashtra', 'Delhi_Delhi', ...]).
    """
    if filepath is None:
        filepath = CLIMATE_FILE

    df = load_csv_safe(filepath)
    if "Dist_States" not in df.columns:
        raise KeyError("Column 'Dist_States' not found in dataset.")

    districts = sorted(df["Dist_States"].dropna().unique().tolist())
    return districts


def print_districts(filepath: str = None, n: int = 20):
    """
    Print a sample or full list of available districts.

    Parameters
    ----------
    filepath : str, optional
        Custom CSV file to read from.
    n : int, optional
        Number of districts to display. Set to -1 to show all.
    """
    districts = get_available_districts(filepath)

    print("\n Available Districts in ClimAID Dataset")
    print("=" * 45)
    if n == -1 or n >= len(districts):
        for name in districts:
            print(name)
    else:
        for name in districts[:n]:
            print(name)
        print(f"... ({len(districts) - n} more districts not shown)")

    print(f"\nTotal districts available: {len(districts)}\n")
    return districts