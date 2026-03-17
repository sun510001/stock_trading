"""Incremental FRED (Federal Reserve Economic Data) downloader.

Mirrors the design of :class:`~data_loader.yahoo_downloader.YahooIncrementalLoader`
so that it can be used in the same batch-download workflow.

Requires a free FRED API key from https://fred.stlouisfed.org/docs/api/api_key.html
Set it via the environment variable ``FRED_API_KEY`` or pass it explicitly.

Dependencies:
    pip install fredapi

Example usage::

    from data_loader.fred_downloader import FREDIncrementalLoader, MACRO_SERIES

    loader = FREDIncrementalLoader(
        api_key="your_api_key_here",  # or set FRED_API_KEY env var
        storage_path="./data/macro",
    )
    loader.download_batch(MACRO_SERIES)
    loader.build_aligned_macro_csv(
        series=MACRO_SERIES,
        output_path="data_processed/macro_indicators.csv",
        fill_method="ffill",
    )
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from logger import logger
from utils.decorators import ExecutionDecorators
from utils.naming import sanitize_filename

# ---------------------------------------------------------------------------
# Project-root-relative key file path
# ---------------------------------------------------------------------------
_LOADER_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_LOADER_DIR)
_DEFAULT_KEY_FILE = os.path.join(_PROJECT_ROOT, ".private_data", "key", "fred.txt")


def _load_api_key_from_file(key_file: str = _DEFAULT_KEY_FILE) -> Optional[str]:
    """Read the FRED API key from a plain-text file (one key per file).

    Args:
        key_file: Absolute path to the key file.  Defaults to
            ``.private_data/key/fred.txt`` inside the project root.

    Returns:
        Stripped key string, or ``None`` if the file does not exist or is empty.
    """
    if not os.path.exists(key_file):
        return None
    try:
        with open(key_file, "r", encoding="utf-8") as f:
            key = f.read().strip()
        return key if key else None
    except Exception as exc:
        logger.warning(f"[FRED] Could not read key file {key_file}: {exc}")
        return None

# ---------------------------------------------------------------------------
# Predefined macro series for the WindowTransformer pipeline
# ---------------------------------------------------------------------------

MACRO_SERIES: List[Dict[str, Any]] = [
    {
        "series_id": "BAMLH0A0HYM2",
        "name": "HY_OAS",
        "description": "ICE BofA US High Yield Index Option-Adjusted Spread (bps)",
        "start_date": "1997-01-01",
        "frequency": "d",  # daily
    },
    {
        "series_id": "NAPM",
        "name": "ISM_PMI",
        "description": "ISM Manufacturing: PMI Composite Index (monthly)",
        "start_date": "1950-01-01",
        "frequency": "m",  # monthly
    },
    {
        "series_id": "IC4WSA",
        "name": "JOBLESS_CLAIMS",
        "description": "4-Week Moving Average of Initial Claims (weekly, thousands)",
        "start_date": "1967-01-01",
        "frequency": "w",  # weekly
    },
    {
        "series_id": "T5YIE",
        "name": "TIPS_5Y_BE",
        "description": "5-Year Breakeven Inflation Rate (daily, %)",
        "start_date": "2003-01-01",
        "frequency": "d",
    },
    {
        "series_id": "T10YIE",
        "name": "T10YIE",
        "description": "10-Year Breakeven Inflation Rate (daily, %)",
        "start_date": "2003-01-01",
        "frequency": "d",
    },
    {
        "series_id": "T10Y2Y",
        "name": "T10Y2Y",
        "description": "10-Year minus 2-Year Treasury Yield Spread (daily, %)",
        "start_date": "1976-06-01",
        "frequency": "d",
    },
    {
        "series_id": "M2SL",
        "name": "M2_YoY",
        "description": "M2 Money Stock (monthly, billions USD) — compute YoY% externally",
        "start_date": "1959-01-01",
        "frequency": "m",
    },
]


# ---------------------------------------------------------------------------
# Helper: lazy fredapi import
# ---------------------------------------------------------------------------

def _get_fred(api_key: str) -> Any:
    """Lazily import and instantiate the fredapi.Fred client.

    Args:
        api_key: FRED API key string.

    Returns:
        fredapi.Fred instance.

    Raises:
        ImportError: If the ``fredapi`` package is not installed.
    """
    try:
        from fredapi import Fred  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "fredapi is required: pip install fredapi"
        ) from exc
    return Fred(api_key=api_key)


def _expand_low_frequency_to_daily_with_ffill(values: pd.Series) -> pd.Series:
    """Expand low-frequency macro series to calendar-daily values via forward fill.

    This method is strictly causal at each timestamp and avoids look-ahead bias
    introduced by full-sample smoothing filters.

    Args:
        values: Low-frequency time series indexed by observation date.

    Returns:
        Calendar-daily series filled by last known observation.

    Notes:
        This is an intermediate expansion step used to preserve weekend-dated
        macro release anchors. Final model and backtest datasets are later
        aligned onto the reference trading calendar in DataProcessor.
    """
    if values.empty:
        return values

    cleaned = values.copy()
    cleaned.index = pd.to_datetime(cleaned.index)
    cleaned = cleaned[cleaned.index.notna()]
    cleaned = cleaned.sort_index().dropna()
    if cleaned.empty:
        return cleaned

    # Use calendar-daily range (freq="D") so that weekend-dated observations
    # (e.g. JOBLESS_CLAIMS releases on Saturdays) are included as anchor
    # points before forward-filling into weekdays.  A business-day range
    # (freq="B") would skip those Saturday dates entirely, yielding all-NaN.
    daily_idx = pd.date_range(cleaned.index.min(), cleaned.index.max(), freq="D")
    expanded = cleaned.reindex(daily_idx).ffill().bfill()
    expanded.name = cleaned.name
    return expanded


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FREDIncrementalLoader:
    """Incremental downloader for FRED economic data series.

    Each series is stored as a separate CSV file under ``storage_path``
    with columns ``Date,Value``.  On subsequent runs only the delta since
    the last saved observation is fetched.

    Args:
        api_key: FRED API key.  Falls back to the ``FRED_API_KEY`` environment
            variable if not provided.
        storage_path: Directory where per-series CSV files are stored.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        storage_path: str = "./data/macro",
    ) -> None:
        # Key resolution order: explicit arg → key file → env var
        self.api_key: str = (
            api_key
            or _load_api_key_from_file()
            or os.environ.get("FRED_API_KEY", "")
        )
        if not self.api_key:
            raise ValueError(
                "FRED API key is required.  Options:\n"
                "  1. Save key to .private_data/key/fred.txt\n"
                "  2. Pass api_key= argument\n"
                "  3. Set FRED_API_KEY environment variable"
            )
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _file_path(self, name: str) -> str:
        """Return the absolute CSV path for a given series name.

        Args:
            name: Human-readable series name (will be sanitized).

        Returns:
            Absolute path string.
        """
        safe = sanitize_filename(name)
        return os.path.join(self.storage_path, f"{safe}.csv")

    def _load_existing(self, file_path: str) -> pd.DataFrame:
        """Load existing CSV.  Returns empty DataFrame if missing or corrupt.

        Args:
            file_path: Absolute path to the CSV file.

        Returns:
            DataFrame with DatetimeIndex and a ``Value`` column, or an empty
            DataFrame if the file does not exist or cannot be parsed.
        """
        if not os.path.exists(file_path):
            return pd.DataFrame()
        try:
            df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
            df = df[df.index.notna()]
            df = df[~df.index.duplicated(keep="last")]
            df.sort_index(inplace=True)
            return df
        except Exception as exc:
            logger.warning(f"[FRED] Corrupt file {file_path}, starting fresh: {exc}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @ExecutionDecorators.retry(max_retries=3, delay=10)
    def download_series(
        self,
        series_id: str,
        name: str,
        start_date_fallback: str = "1990-01-01",
        frequency: Optional[str] = None,
    ) -> bool:
        """Incrementally download a single FRED series and append to local CSV.

        Mirrors :meth:`YahooIncrementalLoader.download_symbol`.

        Args:
            series_id: FRED series identifier (e.g. ``"BAMLH0A0HYM2"``).
            name: Human-readable name used as the CSV filename.
            start_date_fallback: Start date if no local data exists (YYYY-MM-DD).
            frequency: Optional frequency override passed to fredapi
                (``'d'``, ``'w'``, ``'m'``, etc.).  ``None`` uses the native
                series frequency.

        Returns:
            True on success, False on failure.
        """
        file_path = self._file_path(name)
        df_old = self._load_existing(file_path)

        # Determine start date for fetch (re-fetch last row to refresh partials)
        if not df_old.empty:
            start_fetch = df_old.index[-1].strftime("%Y-%m-%d")
        else:
            start_fetch = start_date_fallback

        logger.info(f"[FRED] Downloading {series_id} ({name}) from {start_fetch}...")

        try:
            fred = _get_fred(self.api_key)
            kwargs: Dict[str, Any] = {
                "observation_start": start_fetch,
            }
            if frequency:
                kwargs["frequency"] = frequency
            raw: pd.Series = fred.get_series(series_id, **kwargs)
        except Exception as exc:
            logger.error(f"[FRED] Download failed for {series_id}: {exc}")
            return False

        if raw is None or raw.empty:
            logger.warning(f"[FRED] Empty response for {series_id} from {start_fetch}.")
            return False

        # Convert to DataFrame with standard column name
        df_new = pd.DataFrame({"Value": raw})
        df_new.index.name = "Date"
        df_new.index = pd.to_datetime(df_new.index)

        # Drop NaN rows (FRED sometimes returns .nan for unreleased periods)
        df_new = df_new.dropna(subset=["Value"])
        df_new = df_new[df_new.index.notna()]

        logger.info(
            f"[FRED] {series_id}: received {len(df_new)} observations "
            f"from {start_fetch}"
        )

        if not df_old.empty:
            # Drop the old last row (may be partial/revised) then merge
            df_base = df_old.iloc[:-1] if len(df_old) > 1 else pd.DataFrame()
            df_final = pd.concat([df_base, df_new]) if not df_base.empty else df_new
            df_final = df_final[~df_final.index.duplicated(keep="last")]
            df_final.sort_index(inplace=True)
        else:
            df_final = df_new.sort_index()

        # Raw low-frequency data is stored as-is.  Daily expansion via
        # forward-fill is applied later in DataProcessor.build_aligned_dataframe()
        # when the user clicks "Process Data", to avoid any look-ahead bias
        # during download time.
        df_final.index.name = "Date"
        df_final.to_csv(file_path, index=True)
        logger.info(f"[FRED] Saved {series_id} → {file_path} ({len(df_final)} rows)")
        return True

    def download_batch(self, series: List[Dict[str, Any]]) -> None:
        """Download a list of FRED series definitions.

        Each element should have at minimum ``series_id`` and ``name`` keys.
        Optional keys: ``start_date``, ``frequency``.

        Args:
            series: List of series configuration dicts.  Typically pass
                :data:`MACRO_SERIES` or a custom subset.
        """
        logger.info("=" * 40)
        logger.info(f"Starting FRED Batch Download ({len(series)} series)")
        logger.info("=" * 40)

        for item in series:
            sid = item["series_id"]
            name = item["name"]
            start = item.get("start_date", "1990-01-01")
            freq = item.get("frequency")
            self.download_series(sid, name, start_date_fallback=start, frequency=freq)
            time.sleep(1)  # Respect FRED rate limits (120 req/min)

        logger.info("FRED Batch Download Complete.\n")

    def build_aligned_macro_csv(
        self,
        series: List[Dict[str, Any]],
        output_path: str,
        fill_method: str = "ffill",
        limit: Optional[int] = 10,
    ) -> pd.DataFrame:
        """Merge multiple per-series CSVs into one aligned panel CSV.

        Low-frequency series (weekly/monthly) are forward-filled to match
        daily frequency when combined with daily series.

        Args:
            series: List of series dicts (same format as :meth:`download_batch`).
            output_path: Path for the merged output CSV.
            fill_method: How to handle missing daily values for
                low-frequency series.  ``'ffill'`` (default) carries the last
                observation forward; ``'interpolate'`` linearly interpolates.
            limit: Maximum number of consecutive NaN periods to fill (``ffill``
                only).  ``None`` fills all.

        Returns:
            Aligned DataFrame (DatetimeIndex, one column per series).
        """
        frames: Dict[str, pd.Series] = {}

        for item in series:
            name = item["name"]
            file_path = self._file_path(name)
            df = self._load_existing(file_path)
            if df.empty:
                logger.warning(f"[FRED] {name}: no local data found, skipping from macro CSV.")
                continue
            if "Value" not in df.columns:
                logger.warning(f"[FRED] {name}: 'Value' column missing, skipping.")
                continue
            frames[name] = df["Value"]

        if not frames:
            raise RuntimeError("No series data loaded; run download_batch() first.")

        # Build a union-index aligned DataFrame
        combined = pd.DataFrame(frames)
        combined = combined.sort_index()

        # Reindex to daily calendar and fill gaps
        daily_idx = pd.date_range(
            start=combined.index.min(), end=combined.index.max(), freq="B"
        )
        combined = combined.reindex(daily_idx)
        combined.index.name = "Date"

        if fill_method == "ffill":
            combined = combined.ffill(limit=limit)
        elif fill_method == "interpolate":
            combined = combined.interpolate(method="time")

        # Drop rows where ALL columns are still NaN
        combined = combined.dropna(how="all")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        combined.to_csv(output_path, index=True)
        logger.info(
            f"[FRED] Macro CSV saved: {output_path}  "
            f"shape={combined.shape}  cols={list(combined.columns)}"
        )
        return combined

    def probe_series_info(self, series_id: str) -> Optional[Dict[str, Any]]:
        """Fetch metadata for a FRED series (title, frequency, units, dates).

        Args:
            series_id: FRED series identifier.

        Returns:
            Dict with series info, or None if lookup fails.
        """
        try:
            fred = _get_fred(self.api_key)
            info = fred.get_series_info(series_id)
            return dict(info)
        except Exception as exc:
            logger.warning(f"[FRED] Could not probe {series_id}: {exc}")
            return None
