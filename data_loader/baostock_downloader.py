import os
import re
from typing import Dict, Any, List, Optional
import datetime as _dt
import pandas as pd

import baostock as bs

from logger import logger
from utils.naming import sanitize_filename
from utils.decorators import ExecutionDecorators

# Tickers that Baostock cannot handle at all (non-A-share, SGE, HK indices, etc.)
# - AU9999, HSTECH: non-A-share / SGE gold / HK tech index
# - 931722: GUOXIN HK SOE Dividend Index (HK-connect cross-border, not in Baostock)
# - 932000: CSI2000 (published 2023, not yet in Baostock historical database)
# - H30269, H30533: CSI custom indices with H-prefix, not available via Baostock
_BS_UNSUPPORTED: frozenset = frozenset({
    "AU9999", "HSTECH",
    "931722",          # GUOXIN HK SOE Dividend (HK-connect index)
    "932000",          # CSI2000 (too new for Baostock)
    "H30269",          # CSI Dividend Low Volatility (custom H-prefix index)
    "H30533",          # China Internet 50 (custom H-prefix index)
})

# A-share index / ETF codes that belong to Shanghai exchange (SSE)
# 000016 / 000300 / 000688 / 000699 → SSE composite/sector indices listed on sh
# 9xxxxx codes (e.g. 932000, 931722, 980092) → CSI/SZSE cross-market → sz prefix
_SSE_INDEX_PREFIXES = re.compile(r"^00[0-9]")   # 000xxx are SSE indices
_SZSE_CODES = re.compile(r"^(9[0-9]{5}|H[0-9A-Z]{5}|[135][0-9]{5})$")


class BaostockIncrementalLoader:
    """
    Incremental loader using Baostock as a data source.

    This class provides a minimal, filesystem-backed downloader compatible
    with the project's existing incremental loaders.

    Methods closely mirror AkshareIncrementalLoader's public surface so the
    rest of the codebase can switch to this loader with minimal changes.
    """

    def __init__(self, storage_path: str = "./data") -> None:
        """
        Initialize the loader.

        Args:
            storage_path: Directory where CSV files will be saved.
        """
        self.storage_path = storage_path
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)

    # ── Ticker normalisation ──────────────────────────────────────────────────

    @staticmethod
    def _to_baostock_ticker(raw: str) -> Optional[str]:
        """Convert a bare ticker code to Baostock's ``exchange.code`` format.

        Routing rules:
        * Known-unsupported tickers (AU9999, HSTECH, …) → ``None`` (skip).
        * 6-digit pure numeric codes:
            - First digit 6 → Shanghai (``sh.``)
            - First digit 0, 1, 2, 3, 5, 9 → Shenzhen (``sz.``)
        * 6-char alphanumeric starting with ``H`` (e.g. H30269, H30533)
          → These are CSI cross-market indices; Baostock serves them under
          ``sz.HXXXXX``.
        * Already in ``sh.`` / ``sz.`` format → returned as-is.
        * Anything else → ``None`` (unsupported).

        Args:
            raw: The bare ticker string from assets.json (e.g. ``"932000"``).

        Returns:
            Baostock-formatted ticker string, or ``None`` if not supported.
        """
        if not raw:
            return None

        # Already formatted
        if raw.startswith("sh.") or raw.startswith("sz."):
            return raw

        # Explicitly unsupported
        if raw.upper() in _BS_UNSUPPORTED:
            logger.debug(f"[Baostock] Ticker {raw!r} is not supported by Baostock; skipping.")
            return None

        # 6-digit pure numeric
        if re.match(r"^\d{6}$", raw):
            if raw.startswith("6"):
                return f"sh.{raw}"
            else:
                # 0, 1, 2, 3, 5, 9xxxxx all live on SZSE in Baostock
                return f"sz.{raw}"

        # 6-char starting with H  (CSI cross-market indices like H30269, H30533)
        if re.match(r"^H\d{5}$", raw, re.IGNORECASE):
            return f"sz.{raw.upper()}"

        # Unknown format
        logger.debug(f"[Baostock] Cannot map ticker {raw!r} to Baostock format; skipping.")
        return None

    def _get_existing_data(self, file_path: str) -> pd.DataFrame:
        """
        Load existing CSV file if present, otherwise return empty DataFrame.

        Handles both the canonical format (Date as first column / index) and
        the legacy broken format where Date was written as the last column.

        Args:
            file_path: Path to CSV file.

        Returns:
            DataFrame with DatetimeIndex named 'Date', or empty DataFrame.
        """
        if not os.path.exists(file_path):
            return pd.DataFrame()
        try:
            with open(file_path, 'r') as f:
                header = f.readline().strip().split(',')

            if header[0] != 'Date' and 'Date' in header:
                # Legacy broken format
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
            logger.warning(f"Corrupt file at {file_path}, starting fresh: {e}")
            return pd.DataFrame()

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        """Normalise baostock DataFrame to standard OHLCV with Date index.

        Renames lowercase Baostock column names to PascalCase, sets a
        DatetimeIndex, coerces values to numeric, and retains **only** the
        standard OHLCV columns so the output schema matches Yahoo / AkShare
        CSV files: ``Date, Open, High, Low, Close, Volume``.

        Args:
            df: Raw DataFrame from Baostock.

        Returns:
            Normalised DataFrame with index named 'Date' and only standard
            OHLCV columns present.
        """
        if "date" in df.columns:
            df = df.rename(columns={"date": "Date"})
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        # Rename lowercase baostock columns to canonical PascalCase
        col_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        standard_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in standard_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # Keep only standard OHLCV columns; drop any extra columns
        keep = [c for c in standard_cols if c in df.columns]
        return df[keep]

    @ExecutionDecorators.retry(max_retries=3, delay=2)
    def _fetch_history(self, ticker: str, start_str: str, end_str: str) -> pd.DataFrame:
        """Fetch historical daily bars for a ticker using Baostock.

        ``ticker`` must already be in Baostock ``exchange.code`` format
        (e.g. ``sh.600000``); callers are responsible for conversion via
        :meth:`_to_baostock_ticker`.

        Args:
            ticker: Baostock-formatted symbol (e.g. ``'sz.000016'``).
            start_str: Start date in ``YYYYMMDD`` format.
            end_str: End date in ``YYYYMMDD`` format.

        Returns:
            Normalised DataFrame with OHLCV columns and DatetimeIndex.

        Raises:
            ValueError: If Baostock returns an error or empty data.
        """
        logger.info(f"[Baostock] Fetching {ticker} from {start_str} to {end_str} ...")
        lg = bs.login()
        try:
            fields = "date,open,high,low,close,volume"
            rs = bs.query_history_k_data_plus(
                ticker,
                fields,
                start_date=start_str,
                end_date=end_str,
                frequency="d",
                adjustflag="1",
            )
            if rs.error_code != "0":
                raise ValueError(f"Baostock error: {rs.error_msg}")

            rows: List[List[str]] = []
            while rs.next():
                rows.append(rs.get_row_data())

            if not rows:
                raise ValueError(f"No data returned for {ticker}")

            df = pd.DataFrame(rows, columns=rs.fields)
            df = self._normalise(df)
            return df
        finally:
            try:
                bs.logout()
            except Exception:
                pass

    def download_symbol(
        self,
        ticker: str,
        name: str,
        start_date_fallback: str = "1985-01-01",
        source: str = "baostock",
    ) -> bool:
        """Download a single symbol and save to CSV.

        Automatically converts ``ticker`` to Baostock format via
        :meth:`_to_baostock_ticker`.  Returns ``False`` immediately for
        tickers that Baostock cannot serve (e.g. ``AU9999``, ``HSTECH``).

        Args:
            ticker: Raw ticker from assets.json (e.g. ``'932000'``).
            name: Human-friendly name used for the CSV filename.
            start_date_fallback: Fallback start date ``YYYY-MM-DD``.
            source: Source label (unused; kept for interface compatibility).

        Returns:
            ``True`` if data was saved successfully, ``False`` otherwise.
        """
        bs_ticker = self._to_baostock_ticker(ticker)
        if bs_ticker is None:
            logger.warning(f"[{name}] Ticker {ticker!r} is not supported by Baostock. Skipping.")
            return False

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
        start_str = start_download_date.strftime("%Y-%m-%d")
        end_str = today.strftime("%Y-%m-%d")

        try:
            df_new = self._fetch_history(bs_ticker, start_str, end_str)
        except Exception as e:
            logger.error(f"[{name}] Baostock download failed: {e}")
            return False

        if df_new is None or df_new.empty:
            logger.info(f"[{name}] Baostock returned no new data.")
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
        logger.info(f"[{name}] Baostock {status} successfully. Rows: {len(df_final)}")
        return True

    def probe_earliest_date(self, ticker: str) -> Optional[_dt.datetime]:
        """Probe the earliest available date for a symbol using Baostock.

        Args:
            ticker: Raw ticker from assets.json (will be auto-converted).

        Returns:
            Earliest available datetime, or ``None`` on failure / unsupported.
        """
        bs_ticker = self._to_baostock_ticker(ticker)
        if bs_ticker is None:
            return None
        try:
            bs.login()
            fields = "date"
            rs = bs.query_history_k_data_plus(
                bs_ticker,
                fields,
                start_date="1990-01-01",
                end_date=_dt.datetime.now().strftime("%Y-%m-%d"),
                frequency="d",
                adjustflag="1",
            )
            if rs.error_code != "0":
                return None
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                return None
            date_str = rows[0][0]
            return _dt.datetime.strptime(date_str, "%Y-%m-%d")
        except Exception as e:
            logger.warning(f"Failed to probe Baostock for {ticker}: {e}")
            return None
        finally:
            try:
                bs.logout()
            except Exception:
                pass

    def download_batch(self, assets: List[Dict[str, Any]], start_year: int = 1985) -> None:
        """
        Download a batch of assets using Baostock routing by 'ticker' and save CSVs.

        Args:
            assets: List of dicts with keys 'name' and 'ticker'.
            start_year: Default fallback start year.
        """
        logger.info("Starting Baostock batch download...")
        for asset in assets:
            name = asset.get("name")
            ticker = asset.get("ticker")
            start_fallback = asset.get("initial_start_date") or asset.get("start_date") or f"{start_year}-01-01"
            self.download_symbol(ticker, name, start_fallback, source="baostock")
