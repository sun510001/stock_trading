import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Optional
import re
import time
from typing import Dict, List, Any, Optional
from logger import logger
from utils.decorators import ExecutionDecorators
from utils.naming import sanitize_filename

class YahooIncrementalLoader:
    """
    [Data ETL Class]
    Handles incremental downloading of financial data from Yahoo Finance.
    
    This class manages the lifecycle of market data acquisition, ensuring that only 
    missing data is fetched and that all local files remain synchronized and healthy.
    """

    def __init__(self, storage_path: str = "./data") -> None:
        """
        Initialize the downloader with a target storage path.

        Args:
            storage_path (str): The directory where CSV files will be stored. Defaults to "./data".
        """
        self.storage_path: str = storage_path
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)

    def _get_existing_data(self, file_path: str) -> pd.DataFrame:
        """
        Reads existing CSV. Returns empty DataFrame if file doesn't exist.
        Handles both formats:
        - Correct: Date as index (Date,Open,High,Low,Close,Volume)
        - Legacy broken: Date as last column (Open,High,Low,Close,Volume,Date)

        Args:
            file_path (str): The absolute path to the CSV file.

        Returns:
            pd.DataFrame: Loaded data or an empty DataFrame if loading fails.
        """
        if not os.path.exists(file_path):
            return pd.DataFrame()
        try:
            # Peek at header to detect broken "Date as last column" format
            with open(file_path, 'r') as f:
                header = f.readline().strip().split(',')

            if header[0] != 'Date' and 'Date' in header:
                # Legacy broken format: Date is not the first column
                df = pd.read_csv(file_path)
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
            else:
                df = pd.read_csv(file_path, index_col='Date', parse_dates=True)

            # Sanitise index
            df = df[df.index.notna()]
            df = df[~df.index.duplicated(keep='last')]
            df.sort_index(inplace=True)
            return df
        except Exception as e:
            logger.warning(f"Corrupt file found at {file_path}, starting fresh. Error: {e}")
            return pd.DataFrame()

    @ExecutionDecorators.retry(max_retries=3, delay=5)
    def download_symbol(self, ticker: str, name: str, start_date_fallback: str = "1985-01-01") -> bool:
        """
        Smart downloader that preserves OHLCV data and handles yfinance quirks.
        It detects existing local data, drops the last recorded day to avoid 
        incomplete intraday data, and fetches the incremental delta up to today.

        Args:
            ticker (str): Yahoo Finance symbol (e.g., '^NDX').
            name (str): Logical name for the asset.
            start_date_fallback (str): Start date if no local data exists (YYYY-MM-DD).
            
        Returns:
            bool: True if data was successfully downloaded/merged or verified as up-to-date, False if it failed.
        """
        safe_name = sanitize_filename(name)
        file_path = os.path.join(self.storage_path, f"{safe_name}.csv")
        df_old = self._get_existing_data(file_path)

        # Determine the start date for the incremental download.
        # We always re-fetch from the last recorded date so that the potentially
        # incomplete intraday candle is refreshed.  The actual truncation of the
        # old last row happens AFTER a successful download, during the merge step,
        # so that no data is lost if the download fails.
        if not df_old.empty:
            start_download_date = df_old.index[-1].date()
        else:
            start_download_date = datetime.strptime(start_date_fallback, "%Y-%m-%d").date()

        logger.info(f"[{name}] Downloading {ticker} from {start_download_date}...")

        try:
            # auto_adjust=True returns adjusted OHLCV
            df_new = yf.download(ticker, start=start_download_date, progress=False, auto_adjust=True)
        except Exception as e:
            logger.error(f"[{name}] Download failed: {e}")
            return False

        if df_new.empty:
            logger.warning(f"YahooFinance returned empty frame for {ticker} from {start_download_date} or dropped due to intraday.")
            return False

        # Handle MultiIndex column structures from yfinance
        if isinstance(df_new.columns, pd.MultiIndex):
            try:
                if ticker in df_new.columns.get_level_values(1):
                    df_new = df_new.xs(ticker, axis=1, level=1)
                elif 'Close' in df_new.columns.get_level_values(0):
                    df_new.columns = df_new.columns.get_level_values(0)
            except Exception as e:
                logger.warning(f"[{name}] MultiIndex parsing warning: {e}. Flattening columns.")
                df_new.columns = df_new.columns.get_level_values(0)

        # Standardize OHLCV columns
        desired_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in desired_cols:
            if col not in df_new.columns:
                df_new[col] = np.nan

        df_new = df_new[desired_cols]
        df_new.index.name = 'Date'

        # Fix zero values or NaNs by filling with Close price
        for col in ['Open', 'High', 'Low']:
            df_new[col] = df_new[col].replace(0, np.nan).fillna(df_new['Close'])

        df_new['Volume'] = df_new['Volume'].fillna(0)

        # Drop any rows where the Date index is NaT (yfinance trailing empty rows)
        df_new = df_new[df_new.index.notna()]

        logger.info(f"Appending {len(df_new)} rows using Yahoo source from {start_download_date} for {ticker}")

        if not df_old.empty:
            # Download succeeded: now safe to drop the old last row (potentially
            # incomplete intraday candle) before merging with fresh data.
            df_base = df_old.iloc[:-1] if len(df_old) > 1 else pd.DataFrame()
            if not df_base.empty:
                df_final = pd.concat([df_base, df_new])
            else:
                df_final = df_new
            df_final = df_final[~df_final.index.duplicated(keep='last')]
            df_final.sort_index(inplace=True)
        else:
            df_final = df_new.sort_index()

        df_final.index.name = 'Date'
        df_final.to_csv(file_path, index=True)
        logger.info(f"Updated {ticker} to {file_path}")
        return True

    def probe_earliest_date(self, ticker: str) -> Optional[datetime]:
        """
        Probe the earliest available historical data date for a ticker on Yahoo Finance.
        Uses a lightweight monthly history request to minimize data overhead.

        Args:
            ticker (str): The ticker symbol to probe.

        Returns:
            Optional[datetime]: The earliest date available, or None if the symbol cannot be probed.
        """
        try:
            # yfinance doesn't natively support querying just metadata, so we download monthly to minimize bytes.
            df = yf.download(ticker, period="max", interval="1mo", progress=False)
            if df.empty:
                return None
            return df.index[0]
        except Exception as e:
            logger.warning(f"Failed to probe Yahoo Finance for {ticker}: {str(e)}")
            return None

    def download_batch(self, assets: List[Dict[str, Any]], start_year: int = 1985) -> None:
        """
        Process a batch of assets defined in a configuration list.

        Args:
            assets (List[Dict[str, Any]]): List of asset configurations.
            start_year (int): Default start year if no data exists. Defaults to 1985.
        """
        logger.info("="*40)
        logger.info(f"Starting Batch Download (Default Start: {start_year})")
        logger.info("="*40)
        
        for asset in assets:
            time.sleep(5)
            name = asset["name"]
            ticker = asset["ticker"]
            asset_start_date = asset.get("initial_start_date") or asset.get("start_date")
            start_fallback = asset_start_date if asset_start_date else f"{start_year}-01-01"
            
            self.download_symbol(ticker, name, start_fallback)
        
        logger.info("Batch Download Complete.\n")
