"""
data_loader.py
--------------
Handles loading the earthquake dataset from CSV and separating
features from the target variable.

Supports configurable column dropping via the config's drop_columns list.
"""

import pandas as pd
from typing import List, Optional, Tuple


def extract_event_id(filename: str) -> str:
    """Extract event timestamp identifier (YYMMDDHHMM) from seismic filename.

    In K-NET filenames (e.g. YMG0190509041906.UD), the event timestamp is
    the 10 digits (YYMMDDHHMM) immediately preceding the channel extension (.UD).
    """
    base = str(filename).split(".")[0]
    return base[-10:]


def load_dataset(
    csv_path: str,
    target_column: str = "magnitude",
    drop_columns: Optional[List[str]] = None,
    return_groups: bool = False,
) -> Tuple:
    """Load the earthquake dataset and separate features from target.

    Reads the CSV file, extracts event_id groups if filename is present,
    drops specified columns, and splits the data into input features (X)
    and target (y).

    Args:
        csv_path: Path to the earthquake CSV file.
        target_column: Name of the target column (default 'magnitude').
        drop_columns: List of column names to drop before feature extraction.
        return_groups: If True, also returns the event_id Series (groups).

    Returns:
        (X, y) if return_groups is False, else (X, y, groups)
    """
    df = pd.read_csv(csv_path)

    # Extract event groups before dropping filename
    groups = None
    if "filename" in df.columns:
        groups = df["filename"].apply(extract_event_id)
        groups.name = "event_id"

    # Default: drop 'filename' if no explicit list provided
    if drop_columns is None:
        drop_columns = ["filename"]

    # Drop specified columns (skip any that don't exist in the dataset)
    existing_drops = [c for c in drop_columns if c in df.columns]
    if existing_drops:
        df = df.drop(columns=existing_drops)
        print(f"[Data Loader] Dropped columns: {existing_drops}")

    # Separate features and target
    if target_column not in df.columns:
        raise KeyError(
            f"'{target_column}' column not found in dataset. "
            "Available columns: " + ", ".join(df.columns.tolist())
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    print(f"[Data Loader] Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    if groups is not None:
        print(f"[Data Loader] Unique earthquake events extracted: {groups.nunique()}")
    print(f"[Data Loader] Target range: {y.min():.2f} - {y.max():.2f}")
    print(f"[Data Loader] Features: {X.columns.tolist()}")

    if return_groups:
        return X, y, groups
    return X, y
