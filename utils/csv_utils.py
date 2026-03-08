from __future__ import annotations

from typing import Optional

import pandas as pd


def load_date_indexed_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV whose date index may be stored in several legacy formats.

    Supported layouts:
    - Date as first named column: ``Date,...``
    - Date as a later column in the file
    - Unnamed first index column exported from ``DataFrame.to_csv(index=True)``
    """
    header = pd.read_csv(file_path, nrows=0)
    columns = [str(col) for col in header.columns]

    index_col: Optional[str | int]
    if "Date" in columns:
        index_col = "Date"
    else:
        index_col = 0

    df = pd.read_csv(file_path, index_col=index_col, parse_dates=True)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)
    df.index.name = "Date"
    return df