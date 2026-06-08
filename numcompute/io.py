import numpy as np
from pathlib import Path
from typing import Union, Tuple


def load_csv(
        filepath: Union[str, Path],
        delimiter: str = ",",
        missing_strategy: str = "fill",
        fill_value: float = np.nan,
        skip_rows: int = 1,
        target_column: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a CSV file into NumPy arrays for features (X) and labels (y).

    Parameters
    filepath : Union[str, Path]
        Path to the CSV file.
    delimiter : str, optional
        Field delimiter used in the file (default is ',').
    missing_strategy : {"fill", "skip"}, optional
        Strategy for handling missing values in feature columns:
        - "fill": Replace missing values with `fill_value`.
        - "skip": Remove rows containing any missing values in features or target.
    fill_value : float, optional
        Value used to replace missing entries when `missing_strategy="fill"`.
        If set to np.nan, missing values remain unchanged.
    skip_rows : int, optional
        Number of rows to skip at the beginning of the file
        (default is 1, typically to skip a header row).
    target_column : int, optional
        Index of the column to be used as target (y). Default is -1 (last column).

    Returns
    Tuple[np.ndarray, np.ndarray]
        X: 2D NumPy array of features (float64).
        y: 1D NumPy array of targets (string or numeric).

    Raises
    ValueError
        If `missing_strategy` is not one of {"fill", "skip"}.
    """

    data = np.genfromtxt(
        filepath,
        delimiter=delimiter,
        skip_header=skip_rows,
        dtype=None,
        encoding="utf-8"
    )

    if data.ndim == 0:
        data = data.reshape(1)

    num_cols = len(data[0]) if data.ndim > 0 else 1

    if target_column < 0:
        target_column = num_cols + target_column

    y = np.array([row[target_column] for row in data])


    feature_indices = [i for i in range(num_cols) if i != target_column]
    
    X_list = []
    for row in data:
        feat_row = []
        for i in feature_indices:
            val = row[i]

            try:
                feat_row.append(float(val) if val != "" else np.nan)
            except (ValueError, TypeError):
                feat_row.append(np.nan)
        X_list.append(feat_row)
    
    X = np.array(X_list, dtype=np.float64)

    if missing_strategy == "skip":

        y_is_missing = (y == "") | (y == "nan") | (y == None)
        mask = (~np.isnan(X).any(axis=1)) & (~y_is_missing)
        X = X[mask]
        y = y[mask]
    elif missing_strategy == "fill":
        if not np.isnan(fill_value):
            X[np.isnan(X)] = fill_value
    else:
        raise ValueError(f"Invalid missing_strategy: {missing_strategy}. Use 'fill' or 'skip'.")

    return X, y
