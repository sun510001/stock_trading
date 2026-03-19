from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from logger import logger
from utils.naming import sanitize_filename


_EASTMONEY_KLINE_URLS: tuple[str, ...] = (
    "https://push2his.eastmoney.com/api/qt/stock/kline/get",
    "https://push2.eastmoney.com/api/qt/stock/kline/get",
)
_EASTMONEY_SUGGEST_URL = "https://searchapi.eastmoney.com/api/suggest/get"
_EASTMONEY_FUND_HISTORY_URL = "https://api.fund.eastmoney.com/f10/lsjz"
_EASTMONEY_FUND_JS_URL = "https://fund.eastmoney.com/pingzhongdata/{symbol}.js"
_EASTMONEY_SEARCH_TOKEN = "D43BF722C8E33BDC906FB84D85E326E8"
_SINA_SUGGEST_URL = "https://suggest3.sinajs.cn/suggest/type=11,12,13,14,15,31,41&key={query}"
_SINA_KLINE_URL = "https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20_sina_kline="
_SINA_KLINE_LENGTHS: tuple[int, ...] = (1000, 260, 60)
_KLINE_UT_TOKENS: tuple[str, ...] = (
    "fa5fd1943c7b386f172d6893dbfba10b",
    "7eea3edcaed734bea9cbfc24409ed989",
)
_DEFAULT_MARKETS: tuple[int, ...] = (1, 0, 2, 47)
_KLINE_PERIOD_MAP = {"daily": "101", "weekly": "102", "monthly": "103"}
_KLINE_COLUMNS = [
    "Date",
    "Open",
    "Close",
    "High",
    "Low",
    "Volume",
    "Turnover",
    "Amplitude",
    "PctChange",
    "Change",
    "TurnoverRate",
]
_STANDARD_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
_FUND_HISTORY_PAGE_SIZE = 500


@dataclass(frozen=True)
class EastmoneyDownloadConfig:
    symbol: str
    name: str
    start_date: str
    end_date: str
    output_dir: str
    period: str = "daily"
    markets: tuple[int, ...] = _DEFAULT_MARKETS
    adjust: int = 0
    timeout: float = 15.0
    pause_seconds: float = 0.35


class EastmoneyIndexDownloader:
    """Download China index OHLCV data from Eastmoney raw HTTP endpoints."""

    def __init__(self, config: EastmoneyDownloadConfig) -> None:
        self.config = config
        self.session = self._build_session()

    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        session.trust_env = False
        session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json,text/plain,*/*",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "Origin": "https://quote.eastmoney.com",
                "Referer": "https://quote.eastmoney.com/",
                "Connection": "close",
            }
        )
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.8,
            allowed_methods=frozenset({"GET"}),
            status_forcelist=(429, 500, 502, 503, 504),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _build_params(self, secid: str, ut_token: str) -> dict[str, str]:
        return {
            "secid": secid,
            "ut": ut_token,
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "klt": _KLINE_PERIOD_MAP[self.config.period],
            "fqt": str(self.config.adjust),
            "beg": self.config.start_date.replace("-", ""),
            "end": self.config.end_date.replace("-", ""),
        }

    def _build_fund_history_params(self, page_index: int) -> dict[str, str]:
        return {
            "fundCode": self.config.symbol,
            "pageIndex": str(page_index),
            "pageSize": str(_FUND_HISTORY_PAGE_SIZE),
            "startDate": self.config.start_date,
            "endDate": self.config.end_date,
        }

    def _fallback_secids(self) -> list[str]:
        return [f"{market}.{self.config.symbol}" for market in self.config.markets]

    def _discover_secids(self) -> list[str]:
        params = {
            "input": self.config.symbol,
            "type": "14",
            "token": _EASTMONEY_SEARCH_TOKEN,
            "count": "10",
        }
        try:
            response = self.session.get(
                _EASTMONEY_SUGGEST_URL,
                params=params,
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            table = payload.get("QuotationCodeTable", {})
            data = table.get("Data")
            if not isinstance(data, list):
                return []
            secids: list[str] = []
            symbol_upper = self.config.symbol.upper()
            for item in data:
                if not isinstance(item, dict):
                    continue
                code = str(item.get("Code", "")).strip().upper()
                unified_code = str(item.get("UnifiedCode", "")).strip().upper()
                quote_id = str(item.get("QuoteID", "")).strip()
                if quote_id and (code == symbol_upper or unified_code == symbol_upper):
                    secids.append(quote_id)
            if secids:
                logger.info(
                    "[Eastmoney] Suggest API resolved %s -> %s",
                    self.config.symbol,
                    secids,
                )
            return list(dict.fromkeys(secids))
        except Exception as exc:
            logger.warning(
                "[Eastmoney] Suggest API failed for %s: %s",
                self.config.symbol,
                exc,
            )
            return []

    def _looks_like_cn_fund_symbol(self) -> bool:
        return bool(re.fullmatch(r"\d{6}", self.config.symbol))

    def _looks_like_cn_index_symbol(self) -> bool:
        return bool(re.fullmatch(r"\d{6}", self.config.symbol))

    @staticmethod
    def _build_price_only_frame(
        dates: pd.Series,
        close_values: pd.Series,
    ) -> pd.DataFrame:
        frame = pd.DataFrame({"Date": dates, "Close": close_values})
        frame = frame.dropna(subset=["Date", "Close"]).drop_duplicates(subset=["Date"])
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
        frame = frame.sort_values("Date").set_index("Date")
        frame["Open"] = frame["Close"]
        frame["High"] = frame["Close"]
        frame["Low"] = frame["Close"]
        frame["Volume"] = 0.0
        return frame[_STANDARD_COLUMNS]

    def _clip_requested_range(self, frame: pd.DataFrame) -> pd.DataFrame:
        start = pd.Timestamp(self.config.start_date)
        end = pd.Timestamp(self.config.end_date)
        clipped = frame.loc[(frame.index >= start) & (frame.index <= end)]
        return clipped.sort_index()

    def _fetch_fund_history_page(self, page_index: int) -> dict[str, Any]:
        response = self.session.get(
            _EASTMONEY_FUND_HISTORY_URL,
            params=self._build_fund_history_params(page_index),
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    def _fetch_fund_lsjz_frame(self) -> pd.DataFrame | None:
        rows: list[dict[str, Any]] = []
        page_index = 1
        total_count = 0
        try:
            while True:
                payload = self._fetch_fund_history_page(page_index)
                total_count = int(payload.get("TotalCount") or total_count or 0)
                data = payload.get("Data")
                if not isinstance(data, dict):
                    break
                page_rows = data.get("LSJZList")
                if not isinstance(page_rows, list) or not page_rows:
                    break
                rows.extend(item for item in page_rows if isinstance(item, dict))
                if len(rows) >= total_count or len(page_rows) < _FUND_HISTORY_PAGE_SIZE:
                    break
                page_index += 1
                time.sleep(self.config.pause_seconds)
        except Exception as exc:
            logger.warning(
                "[EastmoneyFund] lsjz failed for %s: %s",
                self.config.symbol,
                exc,
            )
            return None

        if not rows:
            return None

        dates = pd.to_datetime([row.get("FSRQ") for row in rows], errors="coerce")
        close_values = pd.to_numeric(
            [row.get("DWJZ") for row in rows],
            errors="coerce",
        )
        frame = self._build_price_only_frame(dates=dates, close_values=close_values)
        frame = self._clip_requested_range(frame)
        if frame.empty:
            logger.info(
                "[EastmoneyFund] lsjz returned no rows in requested range for %s",
                self.config.symbol,
            )
            return pd.DataFrame(columns=_STANDARD_COLUMNS)
        logger.info(
            "[EastmoneyFund] Downloaded %s via lsjz (%s rows)",
            self.config.symbol,
            len(frame),
        )
        return frame

    @staticmethod
    def _extract_js_var(text: str, var_name: str) -> Any | None:
        pattern = rf"var\s+{re.escape(var_name)}\s*=\s*(.*?);\s*(?:/\*|var\s+)"
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            return None
        raw_value = match.group(1).strip()
        try:
            return json.loads(raw_value)
        except json.JSONDecodeError:
            return None

    def _fetch_fund_pingzhongdata_frame(self) -> pd.DataFrame | None:
        url = _EASTMONEY_FUND_JS_URL.format(symbol=self.config.symbol)
        try:
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
        except Exception as exc:
            logger.warning(
                "[EastmoneyFund] pingzhongdata failed for %s: %s",
                self.config.symbol,
                exc,
            )
            return None

        trend_data = self._extract_js_var(response.text, "Data_netWorthTrend")
        if not isinstance(trend_data, list) or not trend_data:
            return None

        dates = pd.to_datetime(
            [item.get("x") for item in trend_data if isinstance(item, dict)],
            unit="ms",
            errors="coerce",
        )
        close_values = pd.to_numeric(
            [item.get("y") for item in trend_data if isinstance(item, dict)],
            errors="coerce",
        )
        frame = self._build_price_only_frame(dates=dates, close_values=close_values)
        frame = self._clip_requested_range(frame)
        if frame.empty:
            logger.info(
                "[EastmoneyFund] pingzhongdata returned no rows in requested range for %s",
                self.config.symbol,
            )
            return pd.DataFrame(columns=_STANDARD_COLUMNS)
        logger.info(
            "[EastmoneyFund] Downloaded %s via pingzhongdata (%s rows)",
            self.config.symbol,
            len(frame),
        )
        return frame

    def _fetch_fund_frame(self) -> pd.DataFrame | None:
        if not self._looks_like_cn_fund_symbol():
            return None
        frame = self._fetch_fund_lsjz_frame()
        if frame is not None:
            return frame
        return self._fetch_fund_pingzhongdata_frame()

    def _discover_sina_symbol(self) -> str | None:
        try:
            response = self.session.get(
                _SINA_SUGGEST_URL.format(query=self.config.symbol),
                timeout=self.config.timeout,
                headers={"Referer": "https://finance.sina.com.cn/"},
            )
            response.raise_for_status()
        except Exception as exc:
            logger.warning("[Sina] Suggest failed for %s: %s", self.config.symbol, exc)
            return None

        match = re.search(r'var\s+suggestvalue="(.*)";?$', response.text.strip())
        if not match:
            return None
        payload = match.group(1).strip()
        if not payload:
            return None
        for item in payload.split(";"):
            fields = item.split(",")
            if len(fields) < 4:
                continue
            code = str(fields[2]).strip().upper()
            symbol = str(fields[3]).strip().lower()
            if code == self.config.symbol.upper() and symbol:
                return symbol
        return None

    @staticmethod
    def _parse_sina_kline_payload(text: str) -> list[dict[str, Any]]:
        match = re.search(r'var\s+_sina_kline=\((.*)\);?$', text.strip(), re.DOTALL)
        if not match:
            return []
        raw_payload = match.group(1).strip()
        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            return []
        return payload if isinstance(payload, list) else []

    def _fetch_sina_index_frame(self) -> pd.DataFrame | None:
        if not self._looks_like_cn_index_symbol():
            return None
        sina_symbol = self._discover_sina_symbol()
        if not sina_symbol:
            return None

        rows: list[dict[str, Any]] = []
        for data_len in _SINA_KLINE_LENGTHS:
            url = (
                f"{_SINA_KLINE_URL}/CN_MarketDataService.getKLineData?"
                f"symbol={sina_symbol}&scale=240&ma=no&datalen={data_len}"
            )
            try:
                response = self.session.get(
                    url,
                    timeout=self.config.timeout,
                    headers={"Referer": "https://finance.sina.com.cn/"},
                )
                response.raise_for_status()
            except Exception as exc:
                logger.warning(
                    "[Sina] Kline failed for %s via %s len=%s: %s",
                    self.config.symbol,
                    sina_symbol,
                    data_len,
                    exc,
                )
                continue
            rows = self._parse_sina_kline_payload(response.text)
            if rows:
                break
        if not rows:
            return None

        frame = pd.DataFrame(rows)
        if frame.empty or "day" not in frame.columns:
            return None
        frame["Date"] = pd.to_datetime(frame["day"], errors="coerce")
        frame["Open"] = pd.to_numeric(frame.get("open"), errors="coerce")
        frame["High"] = pd.to_numeric(frame.get("high"), errors="coerce")
        frame["Low"] = pd.to_numeric(frame.get("low"), errors="coerce")
        frame["Close"] = pd.to_numeric(frame.get("close"), errors="coerce")
        frame["Volume"] = pd.to_numeric(frame.get("volume"), errors="coerce")
        frame = frame.dropna(subset=["Date", "Open", "High", "Low", "Close"])
        frame = frame.set_index("Date").sort_index()[_STANDARD_COLUMNS]
        frame = self._clip_requested_range(frame)
        if frame.empty:
            return pd.DataFrame(columns=_STANDARD_COLUMNS)
        logger.info(
            "[Sina] Downloaded %s via %s (%s rows)",
            self.config.symbol,
            sina_symbol,
            len(frame),
        )
        return frame

    @staticmethod
    def _parse_jsonp_payload(text: str) -> dict[str, Any] | None:
        match = re.search(r"^[^(]+\((.*)\)\s*;?\s*$", text, re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def _extract_klines(self, payload: Any) -> list[str] | None:
        if not isinstance(payload, dict):
            return None
        data = payload.get("data")
        if not isinstance(data, dict):
            return None
        klines = data.get("klines")
        return list(klines) if isinstance(klines, list) and klines else None

    def _request_kline_once(
        self,
        url: str,
        secid: str,
        ut_token: str,
        use_jsonp: bool,
    ) -> list[str] | None:
        params = self._build_params(secid=secid, ut_token=ut_token)
        if use_jsonp:
            params["cb"] = "jQuery1123"
        response = self.session.get(url, params=params, timeout=self.config.timeout)
        response.raise_for_status()
        payload: Any
        if use_jsonp:
            payload = self._parse_jsonp_payload(response.text)
        else:
            payload = response.json()
        return self._extract_klines(payload)

    def _fetch_klines(self) -> tuple[str, list[str]]:
        last_error: Exception | None = None
        secids = self._discover_secids() or self._fallback_secids()
        attempts = [
            (url, secid, ut_token, use_jsonp)
            for secid in secids
            for url in _EASTMONEY_KLINE_URLS
            for ut_token in _KLINE_UT_TOKENS
            for use_jsonp in (False, True)
        ]
        for url, secid, ut_token, use_jsonp in attempts:
            try:
                klines = self._request_kline_once(
                    url=url,
                    secid=secid,
                    ut_token=ut_token,
                    use_jsonp=use_jsonp,
                )
                if klines:
                    logger.info(
                        "[Eastmoney] Downloaded %s via %s secid=%s jsonp=%s (%s rows)",
                        self.config.symbol,
                        url,
                        secid,
                        use_jsonp,
                        len(klines),
                    )
                    return secid, klines
                logger.info(
                    "[Eastmoney] %s returned no klines for %s via secid=%s jsonp=%s",
                    url,
                    self.config.symbol,
                    secid,
                    use_jsonp,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "[Eastmoney] %s failed for %s via secid=%s jsonp=%s ut=%s: %s",
                    url,
                    self.config.symbol,
                    secid,
                    use_jsonp,
                    ut_token,
                    exc,
                )
            time.sleep(self.config.pause_seconds)

        if last_error is not None:
            raise RuntimeError(
                f"Eastmoney request failed for {self.config.symbol}: {last_error}"
            )
        raise RuntimeError(f"Eastmoney returned no data for {self.config.symbol}")

    @staticmethod
    def _to_frame(klines: Sequence[str]) -> pd.DataFrame:
        frame = pd.DataFrame([item.split(",") for item in klines], columns=_KLINE_COLUMNS)
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame.dropna(subset=["Date"]).set_index("Date").sort_index()
        for column in _KLINE_COLUMNS[1:]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame[_STANDARD_COLUMNS]

    def download(self) -> pd.DataFrame:
        fund_frame = self._fetch_fund_frame()
        if fund_frame is not None:
            if fund_frame.empty:
                raise RuntimeError(
                    f"Eastmoney fund source returned no data in requested range for {self.config.symbol}"
                )
            return fund_frame
        sina_frame = self._fetch_sina_index_frame()
        if sina_frame is not None:
            if sina_frame.empty:
                raise RuntimeError(
                    f"Sina index source returned no data in requested range for {self.config.symbol}"
                )
            return sina_frame
        _, klines = self._fetch_klines()
        return self._to_frame(klines)

    def save(self, frame: pd.DataFrame) -> str:
        os.makedirs(self.config.output_dir, exist_ok=True)
        filename = f"{sanitize_filename(self.config.name)}.csv"
        output_path = os.path.join(self.config.output_dir, filename)
        frame.to_csv(output_path, index=True, index_label="Date")
        logger.info("[Eastmoney] Saved %s rows to %s", len(frame), output_path)
        return output_path


def _parse_markets(raw: str) -> tuple[int, ...]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    return tuple(int(value) for value in values) if values else _DEFAULT_MARKETS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download index OHLCV data from Eastmoney raw HTTP endpoints.",
    )
    parser.add_argument("--symbol", default="H30269", help="Eastmoney index code")
    parser.add_argument(
        "--name",
        default="Dividend_Low_Volatility",
        help="Output asset name used for the CSV filename",
    )
    parser.add_argument(
        "--start-date",
        default="2013-01-04",
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end-date",
        default=dt.date.today().isoformat(),
        help="End date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--output-dir",
        default="./data",
        help="Directory where the CSV will be written",
    )
    parser.add_argument(
        "--period",
        choices=sorted(_KLINE_PERIOD_MAP.keys()),
        default="daily",
        help="K-line period",
    )
    parser.add_argument(
        "--markets",
        default=",".join(str(item) for item in _DEFAULT_MARKETS),
        help="Comma-separated Eastmoney market fallbacks, e.g. 1,0,2,47",
    )
    parser.add_argument(
        "--adjust",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="Eastmoney fqt adjustment flag: 0=none, 1=forward, 2=backward",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    config = EastmoneyDownloadConfig(
        symbol=args.symbol,
        name=args.name,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        period=args.period,
        markets=_parse_markets(args.markets),
        adjust=args.adjust,
    )
    downloader = EastmoneyIndexDownloader(config)
    frame = downloader.download()
    output_path = downloader.save(frame)
    logger.info("[Eastmoney] Finished download for %s -> %s", config.symbol, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())