"""
data_loader.py
--------------
Handles loading the earthquake dataset from CSV and separating
features from the target variable.

Supports configurable column dropping via the config's drop_columns list.
"""

import pandas as pd
from typing import List, Optional, Tuple


def load_dataset(
    csv_path: str,
    target_column: str = "magnitude",
    drop_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the earthquake dataset and separate features from target.

    Reads the CSV file, drops specified columns (e.g., non-predictive
    identifiers and degenerate features), and splits the data into
    input features (X) and target (y).

    Args:
        csv_path: Path to the earthquake CSV file.
        target_column: Name of the target column (default 'magnitude').
        drop_columns: List of column names to drop before feature
                      extraction. If None, only 'filename' is dropped.

    Returns:
        A tuple (X, y) where:
            X: DataFrame of input features.
            y: Series of target values.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        KeyError: If the target column is missing.
    """
    df = pd.read_csv(csv_path)

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
    print(f"[Data Loader] Target range: {y.min():.2f} - {y.max():.2f}")
    print(f"[Data Loader] Features: {X.columns.tolist()}")

    return X, y
