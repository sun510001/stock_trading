import akshare as ak
import pandas as pd
import os
from typing import List, Dict, Optional, Any
from logger import logger
from utils.decorators import retry
from utils.naming import sanitize_filename

class USMarketLoader:
    """
    A class to download, clean, and store US stock data using AkShare.
    Designed to prepare data for VectorBT backtesting.
    """

    def __init__(self, storage_path: str = "./data"):
        """
        Initialize the loader.

        Args:
            storage_path (str): Directory where CSV files will be saved.
        """
        self.storage_path = storage_path
        self._ensure_storage_exists()

    def _ensure_storage_exists(self) -> None:
        """Create the storage directory if it does not exist."""
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
            logger.info(f"Created storage directory at: {self.storage_path}")

    @retry(max_retries=5, delay=3)
    def _fetch_single_symbol(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical data for a single symbol from AkShare.

        Note: AkShare 'stock_us_daily' returns Chinese column names by default.
        We need to map them to English for standardization.

        Args:
            symbol (str): The ticker symbol (e.g., 'SPY').
            start_date (str): Format 'YYYYMMDD'.
            end_date (str): Format 'YYYYMMDD'.

        Returns:
            pd.DataFrame: Cleaned DataFrame with standard OHLCV columns and datetime index.
        """
        logger.info(f"Fetching data for {symbol}...")

        df = ak.stock_us_daily(symbol=symbol, adjust="qfq")

        if df is None or df.empty:
            raise ValueError(f"No data returned for {symbol}")

        column_map = {
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }

        df.columns = [c.lower() for c in df.columns]
        df.rename(columns=column_map, inplace=True)

        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)

        mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
        df = df.loc[mask]

        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def download_batch(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Download data for multiple symbols and save them to CSV.

        Args:
            symbols (List[str]): List of ticker symbols.
            start_date (str): Start date 'YYYYMMDD'.
            end_date (str): End date 'YYYYMMDD'.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbol to its DataFrame.
        """
        results = {}
        for symbol in symbols:
            try:
                df = self._fetch_single_symbol(symbol, start_date, end_date)

                file_path = os.path.join(self.storage_path, f"{symbol}.csv")
                df.to_csv(file_path)
                logger.info(f"Successfully saved {symbol} to {file_path} (Records: {len(df)})")

                results[symbol] = df
            except Exception as e:
                logger.error(f"Failed to process {symbol}: {e}")

        return results

    def load_aligned_close_price(self, symbols: List[str]) -> pd.DataFrame:
        """
        Load locally saved CSVs and combine them into a single DataFrame
        containing only 'Close' prices, aligned by Date index.

        This is the format VectorBT prefers for simple portfolio backtesting.

        Args:
            symbols (List[str]): List of symbols to load.

        Returns:
            pd.DataFrame: Index=Date, Columns=Symbols (Close Prices).
        """
        close_data = pd.DataFrame()

        for symbol in symbols:
            file_path = os.path.join(self.storage_path, f"{symbol}.csv")
            if not os.path.exists(file_path):
                logger.warning(f"File not found for {symbol}, skipping alignment.")
                continue

            df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
            close_data[symbol] = df["Close"]

        original_len = len(close_data)
        close_data.dropna(inplace=True)
        new_len = len(close_data)

        if original_len != new_len:
            logger.info(f"Aligned data: Dropped {original_len - new_len} rows due to mismatched dates.")

        return close_data


class AkshareIncrementalLoader:
    """Incremental loader using AkShare, designed to be compatible with YahooIncrementalLoader interface."""

    def __init__(self, storage_path: str = "./data") -> None:
        self.storage_path: str = storage_path
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
        self._us_loader = USMarketLoader(storage_path=self.storage_path)

    def _get_existing_data(self, file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            return pd.DataFrame()
        try:
            df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
            return df
        except Exception as e:
            logger.warning(f"Corrupt file found at {file_path}, starting fresh. Error: {e}")
            return pd.DataFrame()

    def download_symbol(self, ticker: str, name: str, start_date_fallback: str = "1985-01-01") -> pd.DataFrame:
        """Download a single symbol via AkShare, returning DataFrame and saving to CSV."""
        safe_name = sanitize_filename(name)
        file_path = os.path.join(self.storage_path, f"{safe_name}.csv")
        df_old = self._get_existing_data(file_path)

        is_update = False
        if not df_old.empty:
            last_date = df_old.index[-1].date()
            import datetime as _dt

            start_download_date = last_date + _dt.timedelta(days=1)
            is_update = True
        else:
            import datetime as _dt

            start_download_date = _dt.datetime.strptime(start_date_fallback, "%Y-%m-%d").date()

        import datetime as _dt

        today = _dt.date.today()
        if start_download_date >= today:
            logger.info(f"[{name}] AkShare already up to date ({start_download_date}).")
            return df_old

        start_str = start_download_date.strftime("%Y%m%d")
        end_str = today.strftime("%Y%m%d")

        try:
            df_new = self._us_loader._fetch_single_symbol(ticker, start_str, end_str)
        except Exception as e:
            logger.error(f"[{name}] AkShare download failed: {e}")
            return df_old

        if df_new.empty:
            logger.info(f"[{name}] AkShare returned no new data.")
            return df_old

        df_new.index.name = "Date"

        if is_update:
            df_final = pd.concat([df_old, df_new])
            df_final = df_final[~df_final.index.duplicated(keep="last")]
        else:
            df_final = df_new

        df_final.sort_index(inplace=True)
        df_final.to_csv(file_path)

        status = "Updated" if is_update else "Created"
        logger.info(f"[{name}] AkShare {status} successfully. Rows: {len(df_final)}")

        return df_final

    def download_batch(self, assets: List[Dict[str, Any]], start_year: int = 1985) -> None:
        import datetime as _dt

        logger.info("=" * 40)
        logger.info(f"Starting AkShare Batch Download (Default Start: {start_year})")
        logger.info("=" * 40)

        for asset in assets:
            name = asset["name"]
            ticker = asset["ticker"]
            asset_start_date = asset.get("initial_start_date") or asset.get("start_date")
            start_fallback = asset_start_date if asset_start_date else f"{start_year}-01-01"

            self.download_symbol(ticker, name, start_fallback)

        logger.info("AkShare Batch Download Complete.\n")
