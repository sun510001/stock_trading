import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any
from logger import logger
from data_loader.yahoo_downloader import YahooIncrementalLoader
from data_loader.fred_downloader import _expand_low_frequency_to_daily_with_ffill
from utils.naming import sanitize_filename

class DataProcessor:
    """
    [ETL Module]
    Responsible for processing raw CSV data into a clean, synchronized portfolio matrix.
    
    This class includes financial engineering engines to convert raw market yields into 
    synthetic total return price series and handles temporal alignment across multiple assets.
    """

    def __init__(self, raw_path: str = "./data", processed_path: str = "./data_processed") -> None:
        """
        Initialize the DataProcessor with paths for raw and processed data.

        Args:
            raw_path (str): Directory containing raw asset CSV files. Defaults to "./data".
            processed_path (str): Directory where processed artifacts will be saved. 
                                 Defaults to "./data_processed".
        """
        self.raw_path: str = raw_path
        self.processed_path: str = processed_path
        if not os.path.exists(self.processed_path):
            os.makedirs(self.processed_path)

    def _load_raw(self, safe_name: str) -> pd.Series:
        """
        Load the 'Close' price column from a raw asset CSV file.

        Args:
            safe_name (str): The sanitized filename (without extension).

        Returns:
            pd.Series: A time series of closing prices indexed by Date.

        Raises:
            FileNotFoundError: If the specified CSV file does not exist.
        """
        path = os.path.join(self.raw_path, f"{safe_name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Raw data file not found: {path}")
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        df = df[df.index.notna()]
        df = df[~df.index.duplicated(keep="last")]
        df.sort_index(inplace=True)
        if "Close" in df.columns:
            return df["Close"]
        return df.iloc[:, 0]

    @staticmethod
    def _third_thursday_of_month(ts: pd.Timestamp) -> pd.Timestamp:
        """Compute the third Thursday date for the month of a timestamp.

        Args:
            ts: Any timestamp within the target month.

        Returns:
            pd.Timestamp: Calendar date of the third Thursday in the same month.
        """
        month_start = pd.Timestamp(year=ts.year, month=ts.month, day=1)
        thursday_weekday = 3  # Monday=0 ... Sunday=6
        days_to_first_thu = (thursday_weekday - month_start.weekday()) % 7
        first_thursday = month_start + pd.Timedelta(days=days_to_first_thu)
        return first_thursday + pd.Timedelta(days=14)

    def _apply_macro_availability_rules(
        self,
        name: str,
        values: pd.Series,
        asset: Dict[str, Any],
    ) -> pd.Series:
        """Align macro observations to realistic market availability timestamps.

        This method addresses release-lag look-ahead risks by remapping each
        macro observation date to a conservative release date before forward
        filling to daily frequency.

        Supported rules:
            - ``none``: keep original observation dates.
            - ``next_thursday``: move each value to the next Thursday.
            - ``third_thursday_same_month``: move each value to the third
              Thursday of its observation month.
            - ``calendar_lag``: shift by ``release_lag_days`` calendar days.

        Args:
            name: Macro series display name.
            values: Raw macro series indexed by observation dates.
            asset: Asset configuration dictionary from ``assets.json``.

        Returns:
            pd.Series: Re-indexed series using release-date timestamps.
        """
        if values.empty:
            return values

        cleaned = values.copy()
        cleaned.index = pd.to_datetime(cleaned.index)
        cleaned = cleaned[cleaned.index.notna()].sort_index().dropna()
        if cleaned.empty:
            return cleaned

        freq = str(asset.get("frequency", "")).strip().lower()
        release_rule = str(asset.get("release_rule", "")).strip().lower()
        lag_days_raw = asset.get("release_lag_days", 0)

        if not release_rule:
            default_rules: Dict[str, Dict[str, Any]] = {
                "JOBLESS_CLAIMS": {"rule": "next_thursday", "lag_days": 0},
                "PhillyFed": {"rule": "third_thursday_same_month", "lag_days": 0},
                "M2_YoY": {"rule": "calendar_lag", "lag_days": 35},
            }
            default_cfg = default_rules.get(name, {})
            release_rule = str(default_cfg.get("rule", "none"))
            lag_days_raw = default_cfg.get("lag_days", lag_days_raw)

        try:
            lag_days = int(lag_days_raw)
        except (TypeError, ValueError):
            lag_days = 0

        idx = cleaned.index
        if release_rule == "next_thursday":
            thursday_weekday = 3
            offsets = (thursday_weekday - idx.weekday) % 7
            release_idx = idx + pd.to_timedelta(offsets, unit="D")
        elif release_rule == "third_thursday_same_month":
            release_idx = pd.DatetimeIndex([self._third_thursday_of_month(ts) for ts in idx])
        elif release_rule == "calendar_lag":
            release_idx = idx + pd.to_timedelta(max(0, lag_days), unit="D")
        else:
            release_idx = idx

        aligned = pd.Series(cleaned.values, index=release_idx, name=cleaned.name)
        aligned = aligned[~aligned.index.duplicated(keep="last")].sort_index()

        if name in {"JOBLESS_CLAIMS", "PhillyFed", "M2_YoY"}:
            logger.warning(
                f"Macro '{name}' uses release-date alignment rule='{release_rule}' "
                f"(lag_days={lag_days}). Note: historical revisions are not removed "
                f"without ALFRED real-time vintages."
            )
        else:
            logger.info(
                f"Macro '{name}' availability aligned with rule='{release_rule}' "
                f"(freq={freq}, lag_days={lag_days})."
            )
        return aligned

    def bond_pricing_engine(self, yield_series: pd.Series, duration: float = 20.0, initial_price: float = 100.0) -> pd.Series:
        y = yield_series / 100.0
        dy = y.diff().fillna(0)
        interest_income = y.shift(1).fillna(y.iloc[0]) / 252.0
        capital_gain = -duration * dy
        total_daily_return = interest_income + capital_gain
        price_series = initial_price * (1 + total_daily_return).cumprod()
        return price_series

    def cash_pricing_engine(self, yield_series: pd.Series, initial_price: float = 100.0) -> pd.Series:
        y = yield_series / 100.0
        daily_ret = y.shift(1).fillna(y.iloc[0]) / 252.0
        price_series = initial_price * (1 + daily_ret).cumprod()
        return price_series

    def build_aligned_dataframe(self, assets: List[Dict[str, Any]]) -> pd.DataFrame:
        """Build aligned price matrix for given assets and return as DataFrame.

        全量对齐矩阵中允许存在 NaN；只去掉整行全空的日期。具体的“木桶式裁剪”
        会在回测阶段按本次使用的资产子集进行。"""
        logger.info("Starting Multi-Asset Data Processing & Alignment (in-memory)...")
        prices: Dict[str, pd.Series] = {}

        for asset in assets:
            name = asset["name"]
            kind = asset.get("kind", "price")
            engine = asset.get("engine")

            # ── Macro indicators (source=fred / kind=macro) ───────────────────
            # Loaded from data/macro/, expanded to business-daily via ffill.
            if kind == "macro" or asset.get("source") == "fred":
                safe_name = sanitize_filename(name)
                macro_path = os.path.join(self.raw_path, "macro", f"{safe_name}.csv")
                if not os.path.exists(macro_path):
                    logger.warning(f"Macro CSV not found, skipping '{name}': {macro_path}")
                    continue
                try:
                    df_macro = pd.read_csv(macro_path, index_col="Date", parse_dates=True)
                    df_macro = df_macro[df_macro.index.notna()]
                    df_macro = df_macro[~df_macro.index.duplicated(keep="last")]
                    df_macro.sort_index(inplace=True)
                    raw_series = df_macro.iloc[:, 0]  # first column (typically "Value")
                    raw_series = self._apply_macro_availability_rules(name, raw_series, asset)
                    freq = (asset.get("frequency") or "").strip().lower()
                    if freq in {"w", "m", "q", "a"}:
                        logger.info(
                            f"Expanding macro '{name}' ({freq}) to daily via forward-fill..."
                        )
                        prices[name] = _expand_low_frequency_to_daily_with_ffill(raw_series)
                    else:
                        # Already daily (freq=='d') or unknown — use as-is.
                        prices[name] = raw_series
                    logger.info(f"Macro indicator '{name}' added to aligned matrix.")
                except Exception as exc:
                    logger.warning(f"Failed to load macro '{name}': {exc}")
                continue

            duration: float = 20.0
            if kind == "yield" and engine == "bond":
                raw_duration = asset.get("duration", 20.0)
                if raw_duration is not None:
                    duration = float(raw_duration)

            safe_name = sanitize_filename(name)
            logger.info(f"Processing asset '{name}' (source: {safe_name}.csv)...")

            raw_series = self._load_raw(safe_name)

            if kind == "price":
                price_series = raw_series
            elif kind == "yield":
                if engine == "bond":
                    price_series = self.bond_pricing_engine(raw_series, duration=duration)
                elif engine == "cash":
                    price_series = self.cash_pricing_engine(raw_series)
                else:
                    raise ValueError(f"Unknown engine '{engine}' for asset '{name}'")
            else:
                raise ValueError(f"Unsupported asset kind '{kind}' for asset '{name}'")

            prices[name] = price_series

        if not prices:
            raise ValueError("No assets were successfully processed.")

        portfolio_df = pd.DataFrame(prices)
        # Ensure the index is a proper DatetimeIndex, monotonically increasing,
        # and free of duplicates before any further operations.
        portfolio_df.index = pd.to_datetime(portfolio_df.index)
        # Drop NaT rows (trailing empty rows from Yahoo downloads)
        portfolio_df = portfolio_df[portfolio_df.index.notna()]
        portfolio_df = portfolio_df[~portfolio_df.index.duplicated(keep="last")]
        portfolio_df.sort_index(inplace=True)
        original_len = len(portfolio_df)
        # 只丢弃整行全为空的日期，保留部分资产缺失的数据，
        # 以便在回测阶段按具体资产子集再做裁剪。
        portfolio_df.dropna(how="all", inplace=True)
        final_len = len(portfolio_df)

        logger.info(f"Alignment complete. Rows: {original_len} -> {final_len}")
        logger.info(
            f"Available range: {portfolio_df.index.min().date()} to {portfolio_df.index.max().date()}"
        )
        return portfolio_df

    def process_and_align(
        self,
        assets: List[Dict[str, Any]],
        output_filename: str = "aligned_assets.csv",
    ) -> str:
        """Full ETL pipeline: build aligned DataFrame and persist to CSV.

        TermSpread is no longer auto-derived here — the FRED series T10Y2Y
        (10Y-2Y spread) is loaded directly as a macro asset and provides a
        cleaner, official version of the same signal.

        Args:
            assets: Asset configuration list.
            output_filename: Target CSV filename under processed_path.

        Returns:
            str: Full path to the written CSV file.
        """
        try:
            portfolio_df = self.build_aligned_dataframe(assets)
            output_file = os.path.join(self.processed_path, output_filename)
            portfolio_df.to_csv(output_file)
            logger.info(f"Aligned assets saved successfully to: {output_file}")
            return output_file
        except Exception as e:
            logger.exception(f"Data pipeline processing failed: {str(e)}")
            raise


if __name__ == "__main__":
    sample_assets = [
        {"name": "Stocks", "ticker": "^NDX", "kind": "price"},
        {"name": "Gold", "ticker": "^XAU", "kind": "price"},
        {"name": "Bonds", "ticker": "^TYX", "kind": "yield", "engine": "bond", "duration": 20.0},
        {"name": "Cash", "ticker": "^IRX", "kind": "yield", "engine": "cash"},
    ]
    processor = DataProcessor()
    processor.process_and_align(sample_assets)
