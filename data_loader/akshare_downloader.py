import akshare as ak
import pandas as pd
import os
import datetime as _dt
from typing import List, Dict, Optional, Any
from datetime import datetime
from contextlib import contextmanager

from logger import logger
from utils.naming import sanitize_filename
from utils.decorators import ExecutionDecorators

@contextmanager
def no_proxy():
    """Context manager to temporarily disable proxy settings for domestic API calls."""
    proxy_keys = ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]
    saved_proxies = {key: os.environ.get(key) for key in proxy_keys}
    
    for key in proxy_keys:
        if key in os.environ:
            del os.environ[key]
            
    try:
        yield
    finally:
        for key, value in saved_proxies.items():
            if value is not None:
                os.environ[key] = value

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

    @ExecutionDecorators.retry(max_retries=5, delay=3)
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

        with no_proxy():
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
    """Incremental loader using AkShare, supporting multiple domestic asset types.

    Supported ``source`` values in assets.json:

    * ``"akshare"``          – US-listed stocks via ``stock_us_daily`` (original behaviour).
    * ``"akshare_index"``    – A-share indices via ``index_zh_a_hist``
                               (e.g. CSI2000 932000, SSE50 000016, STAR50 000688).
    * ``"akshare_etf"``      – A-share ETFs via ``fund_etf_hist_em``
                               (e.g. 30Y Treasury ETF 511090, CSI Banks ETF 512800).
    * ``"akshare_gold"``     – Shanghai Gold Exchange spot gold via ``spot_gold_hist_sge``
                               (e.g. AU9999).
    * ``"akshare_hk_index"`` – Hong Kong indices via ``stock_hk_index_daily_em``
                               (e.g. HSTECH).
    """

    # ── Column normalisation maps per source type ─────────────────────────────

    # index_zh_a_hist returns: 日期 开盘 收盘 最高 最低 成交量 成交额 振幅 涨跌幅 涨跌额 换手率
    _INDEX_COL_MAP: Dict[str, str] = {
        "日期": "Date", "开盘": "Open", "收盘": "Close",
        "最高": "High", "最低": "Low", "成交量": "Volume",
    }
    # fund_etf_hist_em returns: 日期 开盘 收盘 最高 最低 成交量 成交额 振幅 涨跌幅 涨跌额 换手率
    _ETF_COL_MAP: Dict[str, str] = {
        "日期": "Date", "开盘": "Open", "收盘": "Close",
        "最高": "High", "最低": "Low", "成交量": "Volume",
    }
    # spot_gold_hist_sge: 日期 开盘价 最高价 最低价 收盘价 成交量(千克) 成交金额(元)
    _GOLD_COL_MAP: Dict[str, str] = {
        "日期": "Date", "开盘价": "Open", "收盘价": "Close",
        "最高价": "High", "最低价": "Low", "成交量(千克)": "Volume",
    }
    # stock_hk_index_daily_em: 日期 开盘 收盘 最高 最低 成交量 成交额 涨跌幅
    _HK_INDEX_COL_MAP: Dict[str, str] = {
        "日期": "Date", "开盘": "Open", "收盘": "Close",
        "最高": "High", "最低": "Low", "成交量": "Volume",
    }

    def __init__(self, storage_path: str = "./data") -> None:
        """Initialise the loader.

        Args:
            storage_path: Directory where CSV files will be saved.
        """
        self.storage_path: str = storage_path
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
        self._us_loader = USMarketLoader(storage_path=self.storage_path)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_existing_data(self, file_path: str) -> pd.DataFrame:
        """Load existing CSV data, returning empty DataFrame on error.

        Handles both the canonical format (Date as first column / index) and
        the legacy broken format where Date was written as the last column.

        Args:
            file_path: Absolute path to the CSV file.

        Returns:
            Loaded DataFrame with DatetimeIndex, or empty DataFrame if file
            absent / corrupt.
        """
        if not os.path.exists(file_path):
            return pd.DataFrame()
        try:
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

    _STANDARD_COLS: List[str] = ["Open", "High", "Low", "Close", "Volume"]

    @staticmethod
    def _normalise(df: pd.DataFrame, col_map: Dict[str, str]) -> pd.DataFrame:
        """Rename Chinese columns, set Date index, cast numeric types, and
        retain only standard OHLCV columns to match Yahoo Finance CSV format.

        Any extra columns returned by AkShare (e.g. 成交额, 振幅, 涨跌幅,
        涨跌额, 换手率) are dropped so that all downstream CSV files share
        a consistent schema: Date, Open, High, Low, Close, Volume.

        Args:
            df: Raw DataFrame from an AkShare call.
            col_map: Mapping from Chinese column names to English equivalents.

        Returns:
            Cleaned DataFrame with a DatetimeIndex named ``Date`` and only
            the standard OHLCV columns present.
        """
        df = df.rename(columns=col_map)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        standard_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in standard_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # Keep only the standard OHLCV columns that are present; drop all extras
        keep = [c for c in standard_cols if c in df.columns]
        return df[keep]

    # ── Source-specific fetch methods ─────────────────────────────────────────

    @ExecutionDecorators.retry(max_retries=5, delay=3)
    def _fetch_cn_index(self, ticker: str, start_str: str, end_str: str) -> pd.DataFrame:
        """Fetch A-share index data via ``index_zh_a_hist``.

        Supports CSI / SSE / SZSE indices identified by 6-digit or mixed
        alphanumeric codes (e.g. 932000, 000016, H30269, H30533).

        Args:
            ticker: Index code (e.g. ``"932000"``).
            start_str: Start date in ``YYYYMMDD`` format.
            end_str: End date in ``YYYYMMDD`` format.

        Returns:
            Normalised DataFrame with OHLCV columns and DatetimeIndex.

        Raises:
            ValueError: If the API returns empty data.
        """
        logger.info(f"[CN Index] Fetching {ticker} via index_zh_a_hist ...")
        with no_proxy():
            df = ak.index_zh_a_hist(
                symbol=ticker,
                period="daily",
                start_date=start_str,
                end_date=end_str,
            )
        if df is None or df.empty:
            raise ValueError(f"No data returned for CN index {ticker}")
        return self._normalise(df, self._INDEX_COL_MAP)

    @ExecutionDecorators.retry(max_retries=5, delay=3)
    def _fetch_cn_etf(self, ticker: str, start_str: str, end_str: str) -> pd.DataFrame:
        """Fetch A-share ETF data via ``fund_etf_hist_em``.

        Uses backward-adjusted (hfq) prices to account for distributions and
        splits.

        Args:
            ticker: ETF code without exchange suffix (e.g. ``"511090"``).
            start_str: Start date in ``YYYYMMDD`` format.
            end_str: End date in ``YYYYMMDD`` format.

        Returns:
            Normalised DataFrame with OHLCV columns and DatetimeIndex.

        Raises:
            ValueError: If the API returns empty data.
        """
        logger.info(f"[CN ETF] Fetching {ticker} via fund_etf_hist_em ...")
        with no_proxy():
            df = ak.fund_etf_hist_em(
                symbol=ticker,
                period="daily",
                start_date=start_str,
                end_date=end_str,
                adjust="hfq",
            )
        if df is None or df.empty:
            raise ValueError(f"No data returned for CN ETF {ticker}")
        return self._normalise(df, self._ETF_COL_MAP)

    @ExecutionDecorators.retry(max_retries=5, delay=3)
    def _fetch_cn_gold(self, ticker: str, start_str: str, end_str: str) -> pd.DataFrame:
        """Fetch Shanghai Gold Exchange spot gold via ``spot_gold_hist_sge``.

        Args:
            ticker: SGE product code (e.g. ``"AU9999"``).
            start_str: Start date in ``YYYYMMDD`` format.
            end_str: End date in ``YYYYMMDD`` format.

        Returns:
            Normalised DataFrame with OHLCV columns and DatetimeIndex.

        Raises:
            ValueError: If the API returns empty data.
        """
        logger.info(f"[CN Gold] Fetching {ticker} via spot_gold_hist_sge ...")
        with no_proxy():
            df = ak.spot_gold_hist_sge(symbol=ticker)
        if df is None or df.empty:
            raise ValueError(f"No data returned for gold spot {ticker}")
        df = self._normalise(df, self._GOLD_COL_MAP)
        # Filter to requested date range
        mask = (df.index >= pd.to_datetime(start_str)) & (df.index <= pd.to_datetime(end_str))
        return df.loc[mask]

    @ExecutionDecorators.retry(max_retries=5, delay=3)
    def _fetch_hk_index(self, ticker: str, start_str: str, end_str: str) -> pd.DataFrame:
        """Fetch Hong Kong index data via ``stock_hk_index_daily_em``.

        Args:
            ticker: HK index code as recognised by EastMoney (e.g. ``"HSTECH"``).
            start_str: Start date in ``YYYYMMDD`` format.
            end_str: End date in ``YYYYMMDD`` format.

        Returns:
            Normalised DataFrame with OHLCV columns and DatetimeIndex.

        Raises:
            ValueError: If the API returns empty data.
        """
        logger.info(f"[HK Index] Fetching {ticker} via stock_hk_index_daily_em ...")
        with no_proxy():
            df = ak.stock_hk_index_daily_em(
                symbol=ticker,
                start_date=start_str,
                end_date=end_str,
            )
        if df is None or df.empty:
            raise ValueError(f"No data returned for HK index {ticker}")
        return self._normalise(df, self._HK_INDEX_COL_MAP)

    # ── Public interface ──────────────────────────────────────────────────────

    def download_symbol(
        self,
        ticker: str,
        name: str,
        start_date_fallback: str = "1985-01-01",
        source: str = "akshare",
    ) -> bool:
        """Download a single symbol via the appropriate AkShare interface.

        Routing logic based on ``source``:

        * ``"akshare"``          → US stocks via ``stock_us_daily``
        * ``"akshare_index"``    → A-share indices via ``index_zh_a_hist``
        * ``"akshare_etf"``      → A-share ETFs via ``fund_etf_hist_em``
        * ``"akshare_gold"``     → SGE spot gold via ``spot_gold_hist_sge``
        * ``"akshare_hk_index"`` → HK indices via ``stock_hk_index_daily_em``

        Args:
            ticker: Exchange ticker / product code.
            name: Human-readable asset name used as the CSV filename base.
            start_date_fallback: Earliest date to request if no local data
                exists yet (``YYYY-MM-DD``).
            source: One of the source strings listed above.

        Returns:
            ``True`` if data was saved successfully, ``False`` otherwise.
        """
        safe_name = sanitize_filename(name)
        file_path = os.path.join(self.storage_path, f"{safe_name}.csv")
        df_old = self._get_existing_data(file_path)

        # Determine the incremental download start date.
        # Re-fetch from the last recorded date so the potentially incomplete
        # intraday candle is refreshed.  Truncation of the old last row happens
        # AFTER a successful download during the merge step, so no data is lost
        # if the download fails.
        if not df_old.empty:
            start_download_date = df_old.index[-1].date()
        else:
            start_download_date = _dt.datetime.strptime(start_date_fallback, "%Y-%m-%d").date()

        today = _dt.date.today()
        start_str = start_download_date.strftime("%Y%m%d")
        end_str = today.strftime("%Y%m%d")

        # Route to the correct fetch method
        try:
            if source == "akshare_index":
                df_new = self._fetch_cn_index(ticker, start_str, end_str)
            elif source == "akshare_etf":
                df_new = self._fetch_cn_etf(ticker, start_str, end_str)
            elif source == "akshare_gold":
                df_new = self._fetch_cn_gold(ticker, start_str, end_str)
            elif source == "akshare_hk_index":
                df_new = self._fetch_hk_index(ticker, start_str, end_str)
            else:
                # Default: original US-stock path
                df_new = self._us_loader._fetch_single_symbol(ticker, start_str, end_str)
        except Exception as e:
            logger.error(f"[{name}] AkShare download failed (source={source}): {e}")
            return False

        if df_new is None or df_new.empty:
            logger.info(f"[{name}] AkShare returned no new data.")
            return False

        df_new.index.name = "Date"

        if not df_old.empty:
            # Download succeeded: now safe to drop the old last row before merging.
            df_base = df_old.iloc[:-1] if len(df_old) > 1 else pd.DataFrame()
            if not df_base.empty:
                df_final = pd.concat([df_base, df_new])
            else:
                df_final = df_new
            df_final = df_final[~df_final.index.duplicated(keep="last")]
        else:
            df_final = df_new

        df_final.sort_index(inplace=True)
        df_final.index.name = "Date"
        df_final.to_csv(file_path, index=True)

        status = "Updated" if not df_old.empty else "Created"
        logger.info(f"[{name}] AkShare {status} successfully. Rows: {len(df_final)}")
        return True

    def probe_earliest_date(self, ticker: str) -> Optional[datetime]:
        """Probe the earliest available historical data date for an ETF ticker.

        Uses ``fund_etf_hist_em`` as the probe interface; suitable for ETF
        codes only.  For other asset types use ``download_symbol`` directly.

        Args:
            ticker: ETF code to probe (e.g. ``"511090"``).

        Returns:
            Earliest available date, or ``None`` if the request fails.
        """
        try:
            with no_proxy():
                df = ak.fund_etf_hist_em(
                    symbol=ticker,
                    period="daily",
                    start_date="19800101",
                    end_date=datetime.now().strftime("%Y%m%d"),
                    adjust="hfq",
                )
            if df.empty:
                return None
            df["Date"] = pd.to_datetime(df["日期"])
            return df["Date"].iloc[0]
        except Exception as e:
            logger.warning(f"Failed to probe AkShare for {ticker}: {str(e)}")
            return None

    def download_batch(self, assets: List[Dict[str, Any]], start_year: int = 1985) -> None:
        """Download a batch of assets, routing each by its ``source`` field.

        Args:
            assets: List of asset dicts, each containing at minimum ``name``,
                ``ticker``, and ``source`` keys.  An optional
                ``initial_start_date`` (``YYYY-MM-DD``) overrides the default
                fallback start date.
            start_year: Default start year used when ``initial_start_date`` is
                absent from an asset entry.
        """
        logger.info("=" * 40)
        logger.info(f"Starting AkShare Batch Download (Default Start: {start_year})")
        logger.info("=" * 40)

        for asset in assets:
            name = asset["name"]
            ticker = asset["ticker"]
            source = asset.get("source", "akshare")
            asset_start_date = asset.get("initial_start_date") or asset.get("start_date")
            start_fallback = asset_start_date if asset_start_date else f"{start_year}-01-01"

            self.download_symbol(ticker, name, start_fallback, source=source)

        logger.info("AkShare Batch Download Complete.\n")

