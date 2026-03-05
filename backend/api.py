from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import os
import sys
import time
from typing import List, Optional, Dict, Any
import pandas as pd

# Ensure localhost / 127.0.0.1 requests are never routed through a system proxy.
# This fixes the common issue where tools like Clash/V2Ray set http_proxy globally
# and break fetch() calls from the UI to the local API server.
_NO_PROXY_HOSTS = "127.0.0.1,localhost,::1"
for _var in ("no_proxy", "NO_PROXY"):
    existing = os.environ.get(_var, "")
    merged = ",".join(filter(None, [existing, _NO_PROXY_HOSTS]))
    os.environ[_var] = merged

# Ensure project root is in path so we can import backend.service
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.service import BacktestConfig, BacktestResult, BacktestService
from backend.assets_config import AssetConfigManager
from data_loader.yahoo_downloader import YahooIncrementalLoader
from data_loader.akshare_downloader import AkshareIncrementalLoader
from data_loader.baostock_downloader import BaostockIncrementalLoader
from data_loader.data_processor import DataProcessor
from data_loader.fred_downloader import FREDIncrementalLoader, MACRO_SERIES
from utils.naming import sanitize_filename
from logger import logger

app = FastAPI(title="Backtest API", version="1.0.0")

_ALLOWED_RELEASE_RULES = {
    "none",
    "next_thursday",
    "third_thursday_same_month",
    "calendar_lag",
}

_DEFAULT_MACRO_RELEASE_CONFIG: Dict[str, Dict[str, Any]] = {
    "JOBLESS_CLAIMS": {"release_rule": "next_thursday", "release_lag_days": 0},
    "PhillyFed": {"release_rule": "third_thursday_same_month", "release_lag_days": 0},
    "M2_YoY": {"release_rule": "calendar_lag", "release_lag_days": 35},
}


class AssetModels:
    class AssetConfig(BaseModel):
        name: str = Field(..., description="Logical name of the asset")
        ticker: str = Field(..., description="Yahoo Finance ticker symbol")
        kind: str = Field(..., description="Asset data type: 'price' or 'yield'")
        frequency: Optional[str] = Field(
            None,
            description=(
                "Data frequency code: 'd' (daily), 'm' (monthly), "
                "'q' (quarterly), 'a' (annual), optionally 'w' (weekly)."
            ),
        )
        engine: Optional[str] = Field(
            None, description="Pricing engine for yields: 'bond' or 'cash'"
        )
        duration: Optional[float] = Field(
            None, description="Bond duration for bond pricing engine"
        )
        initial_start_date: Optional[str] = Field(
            "1985-01-02", description="Initial download start date (YYYY-MM-DD)"
        )
        description: Optional[str] = Field(
            "", description="A short introduction or description for the asset"
        )
        release_rule: Optional[str] = Field(
            None,
            description=(
                "Macro availability alignment rule: 'none', 'next_thursday', "
                "'third_thursday_same_month', or 'calendar_lag'."
            ),
        )
        release_lag_days: Optional[int] = Field(
            None,
            description=(
                "Calendar-day publication lag used by 'calendar_lag'. "
                "Ignored by other rules."
            ),
        )

    class AssetWithMeta(AssetConfig):
        data_start_date: Optional[str] = Field(
            None, description="Earliest date in local CSV file"
        )
        data_end_date: Optional[str] = Field(
            None, description="Latest date in local CSV file"
        )
        source: Optional[str] = Field(
            None, description="The origin of data (e.g., 'yahoo', 'akshare')"
        )
        processed: bool = Field(
            False, description="Whether the asset exists in the aligned_assets.csv"
        )
        derived: bool = Field(
            False,
            description="Whether this asset is a derived/virtual column (not editable/downloadable raw source)",
        )


class APIManager:
    def __init__(self) -> None:
        self.service = BacktestService()
        self.last_download_ts: float = 0.0
        self.download_cooldown: int = 60
        self._ui_html: Optional[str] = None

    def get_ui_html(self) -> str:
        ui_path = os.path.join(project_root, "backend", "ui.html")
        with open(ui_path, "r", encoding="utf-8") as f:
            return f.read()


manager = APIManager()


def _is_macro_asset(asset: Dict[str, Any]) -> bool:
    """Check whether an asset should be treated as a macro indicator.

    Args:
        asset: Asset configuration dictionary.

    Returns:
        bool: True if the asset is macro/FRED, otherwise False.
    """
    return asset.get("kind") == "macro" or asset.get("source") == "fred"


def _normalize_macro_release_fields(asset: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize and validate macro release-alignment fields.

    This function ensures API writes remain consistent even if UI omits macro
    alignment fields. For non-macro assets, these fields are removed.

    Args:
        asset: Raw asset dictionary to be persisted.

    Returns:
        Dict[str, Any]: A normalized copy ready for persistence.

    Raises:
        HTTPException: If release rule or lag value is invalid.
    """
    normalized = dict(asset)
    if not _is_macro_asset(normalized):
        normalized.pop("release_rule", None)
        normalized.pop("release_lag_days", None)
        return normalized

    defaults = _DEFAULT_MACRO_RELEASE_CONFIG.get(str(normalized.get("name", "")), {})
    raw_rule = normalized.get("release_rule", defaults.get("release_rule", "none"))
    rule = str(raw_rule).strip().lower() if raw_rule is not None else "none"

    if rule not in _ALLOWED_RELEASE_RULES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid release_rule '{raw_rule}'. "
                f"Allowed: {sorted(_ALLOWED_RELEASE_RULES)}"
            ),
        )

    raw_lag = normalized.get("release_lag_days", defaults.get("release_lag_days", 0))
    if raw_lag in (None, ""):
        lag_days = 0
    else:
        try:
            lag_days = int(raw_lag)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"release_lag_days must be an integer, got: {raw_lag}",
            ) from exc

    if lag_days < 0:
        raise HTTPException(
            status_code=400,
            detail="release_lag_days must be >= 0",
        )

    normalized["release_rule"] = rule
    normalized["release_lag_days"] = lag_days
    return normalized


_NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate",
    "Pragma": "no-cache",
}


@app.get("/", response_class=HTMLResponse)
def root_endpoint() -> HTMLResponse:
    return HTMLResponse(content=manager.get_ui_html(), headers=_NO_CACHE_HEADERS)


@app.get("/ui", response_class=HTMLResponse)
def ui_endpoint() -> HTMLResponse:
    return HTMLResponse(content=manager.get_ui_html(), headers=_NO_CACHE_HEADERS)


@app.get("/api/algorithms")
def list_algorithms() -> List[Dict[str, str]]:
    return [
        {"key": k, "label": v["label"], "description": v["description"]}
        for k, v in manager.service.algorithm_map.items()
    ]


def _resolve_asset_csv_path(asset: Dict[str, Any], data_dir: str, macro_dir: str) -> str:
    """Resolve the local CSV path for an asset based on its source/kind metadata.

    Args:
        asset: Asset configuration dictionary loaded from ``assets.json``.
        data_dir: Absolute path to the standard asset CSV directory.
        macro_dir: Absolute path to the macro/FRED CSV directory.

    Returns:
        Absolute CSV file path for the asset.
    """
    safe_name = sanitize_filename(asset["name"])
    is_macro = asset.get("kind") == "macro" or asset.get("source") == "fred"
    base_dir = macro_dir if is_macro else data_dir
    return os.path.join(base_dir, f"{safe_name}.csv")


@app.get("/api/assets", response_model=List[AssetModels.AssetWithMeta])
def get_assets() -> List[AssetModels.AssetWithMeta]:
    assets = AssetConfigManager.load_assets()
    results: List[AssetModels.AssetWithMeta] = []
    data_dir = os.path.join(project_root, "data")
    macro_dir = os.path.join(project_root, "data", "macro")

    # 尝试读取已处理对齐矩阵，以标记哪些资产已进入 aligned_assets.csv
    aligned_path = os.path.join(project_root, "data_processed", "aligned_assets.csv")
    aligned_cols: List[str] = []
    if os.path.exists(aligned_path):
        try:
            aligned_df = pd.read_csv(aligned_path, nrows=1)
            aligned_cols = [str(c) for c in aligned_df.columns if c != "Date"]
        except Exception:
            aligned_cols = []

    for asset in assets:
        csv_path = _resolve_asset_csv_path(asset, data_dir=data_dir, macro_dir=macro_dir)
        d_start, d_end = None, None
        source = asset.get("source", None)
        if os.path.exists(csv_path):
            try:
                with open(csv_path, 'r') as _f:
                    _header = _f.readline().strip().split(',')
                if _header[0] != 'Date' and 'Date' in _header:
                    # Legacy broken format: Date as last column
                    _df = pd.read_csv(csv_path)
                    _df['Date'] = pd.to_datetime(_df['Date'])
                    df = _df.set_index('Date')
                else:
                    df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
                df = df[df.index.notna()]
                if not df.empty:
                    d_start = df.index.min().date().isoformat()
                    d_end = df.index.max().date().isoformat()
            except Exception:
                pass

        processed = asset["name"] in aligned_cols

        # Remove "source" from the asset dictionary if it exists to avoid multiple values error
        # when we pass it explicitly a few lines below.
        asset_kwargs = asset.copy()
        asset_kwargs.pop("source", None)

        results.append(
            AssetModels.AssetWithMeta(
                **asset_kwargs,
                data_start_date=d_start,
                data_end_date=d_end,
                source=source,
                processed=processed,
                derived=False,
            )
        )

    # 将派生列 TermSpread 暴露在 Asset Management 列表中（虚拟资产，不写入 assets.json）
    if "TermSpread" in aligned_cols and not any(a.name == "TermSpread" for a in results):
        results.append(
            AssetModels.AssetWithMeta(
                name="TermSpread",
                ticker="DERIVED",
                kind="derived",
                engine=None,
                duration=None,
                initial_start_date=None,
                data_start_date=None,
                data_end_date=None,
                source="derived",
                processed=True,
                derived=True,
            )
        )
    return results


@app.post("/api/assets", response_model=AssetModels.AssetWithMeta)
def create_asset(asset: AssetModels.AssetConfig) -> AssetModels.AssetWithMeta:
    assets = AssetConfigManager.load_assets()
    if any(a["name"] == asset.name for a in assets):
        raise HTTPException(
            status_code=400, detail=f"Asset '{asset.name}' already exists."
        )
    normalized = _normalize_macro_release_fields(asset.model_dump())
    assets.append(normalized)
    AssetConfigManager.save_assets(assets)
    return AssetModels.AssetWithMeta(**normalized)


@app.put("/api/assets/{name}", response_model=AssetModels.AssetWithMeta)
def update_asset(name: str, asset: AssetModels.AssetConfig) -> AssetModels.AssetWithMeta:
    assets = AssetConfigManager.load_assets()
    if asset.name != name and any(a["name"] == asset.name for a in assets):
        raise HTTPException(
            status_code=400, detail=f"Conflict: Name '{asset.name}' already exists."
        )

    for i, a in enumerate(assets):
        if a["name"] == name:
            new_data = asset.model_dump()
            # Preserve existing source if not overwritten by client, but clear it if ticker changed!
            if "source" in a:
                if a["ticker"] == new_data["ticker"]:
                    new_data["source"] = a["source"]
                else:
                    new_data["source"] = None
            
            # Preserve existing description if the update doesn't provide a new one
            if "description" in a and not new_data.get("description"):
                new_data["description"] = a["description"]

            # Preserve existing frequency if client omitted it.
            if "frequency" in a and not new_data.get("frequency"):
                new_data["frequency"] = a["frequency"]

            # Preserve existing macro release alignment fields if omitted by UI.
            if "release_rule" in a and not new_data.get("release_rule"):
                new_data["release_rule"] = a.get("release_rule")
            if "release_lag_days" in a and new_data.get("release_lag_days") is None:
                new_data["release_lag_days"] = a.get("release_lag_days")

            new_data = _normalize_macro_release_fields(new_data)

            assets[i] = new_data
            AssetConfigManager.save_assets(assets)
            return AssetModels.AssetWithMeta(**new_data)
    raise HTTPException(status_code=404, detail="Asset configuration not found.")


@app.delete("/api/assets/{name}")
def delete_asset(name: str) -> Dict[str, str]:
    assets = AssetConfigManager.load_assets()
    new_assets = [a for a in assets if a["name"] != name]
    if len(new_assets) == len(assets):
        raise HTTPException(status_code=404, detail="Asset not found.")
    AssetConfigManager.save_assets(new_assets)
    return {"detail": f"Asset '{name}' deleted."}


class DownloadRequest(BaseModel):
    names: Optional[List[str]] = Field(
        None,
        description="Optional list of asset names to download; if omitted, all assets are processed.",
    )
    downloader: Optional[str] = Field(
        None,
        description=(
            "Force a specific download backend: 'yahoo', 'akshare', 'baostock', or None/'auto' "
            "for the default smart-routing behaviour."
        ),
    )


class MacroDownloadRequest(BaseModel):
    """Request body for FRED macro series download."""

    names: Optional[List[str]] = Field(
        None,
        description="Series names to download (as in MACRO_SERIES[*]['name']); if None, download all.",
    )
    storage_path: Optional[str] = Field(
        None,
        description="Override storage path for per-series CSVs; defaults to data/macro.",
    )
    build_csv: bool = Field(
        True,
        description="Whether to merge downloaded series into aligned macro CSV after download.",
    )
    output_path: str = Field(
        "data_processed/macro_indicators.csv",
        description="Output path for the merged macro CSV (relative to project root).",
    )


def _build_aligned_filename_from_assets(assets: List[Dict[str, Any]]) -> str:
    names_sorted = sorted(a["name"] for a in assets)
    key = "_".join(names_sorted)
    safe_key = sanitize_filename(key)
    return f"aligned_{safe_key}.csv"


@app.post("/api/assets/download")
def download_assets_endpoint(req: DownloadRequest) -> Dict[str, str]:
    now = time.time()
    if now - manager.last_download_ts < manager.download_cooldown:
        remaining = int(manager.download_cooldown - (now - manager.last_download_ts))
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit: Wait {remaining}s before next download.",
        )

    assets = AssetConfigManager.load_assets()
    if not assets:
        raise HTTPException(
            status_code=400, detail="No assets configured for download."
        )

    if req.names:
        selected = [a for a in assets if a["name"] in req.names]
        if not selected:
            raise HTTPException(
                status_code=400,
                detail="No matching assets found for the provided names.",
            )
        assets = selected
    else:
        raise HTTPException(status_code=400, detail="请先勾选资产")

    try:
        from logger import logger

        logger.info(">>> HTTP TRIGGER: SMART DUAL-SOURCE DOWNLOAD START <<<")
        data_path = os.path.join(project_root, "data")
        yahoo_loader = YahooIncrementalLoader(storage_path=data_path)
        akshare_loader = AkshareIncrementalLoader(storage_path=data_path)
        baostock_loader = BaostockIncrementalLoader(storage_path=data_path)

        # Forced downloader override: 'yahoo' | 'akshare' | 'baostock' | 'fred' | 'auto'
        forced_dl = (req.downloader or "auto").strip().lower()
        if forced_dl not in ("yahoo", "akshare", "baostock", "fred", "auto"):
            forced_dl = "auto"
        logger.info(f"Download mode: {forced_dl.upper()}")

        updated_assets = False

        # Initialise FRED loader lazily (only if fred assets are in the selection)
        _fred_loader = None
        def _get_fred_loader() -> FREDIncrementalLoader:
            nonlocal _fred_loader
            if _fred_loader is None:
                macro_path = os.path.join(project_root, "data", "macro")
                _fred_loader = FREDIncrementalLoader(storage_path=macro_path)
            return _fred_loader

        for asset in assets:
            name = asset["name"]
            ticker = asset["ticker"]

            # Start date logic
            asset_start_date = asset.get("initial_start_date") or asset.get("start_date")
            start_fallback = asset_start_date if asset_start_date else "1985-01-01"

            known_source = asset.get("source")
            success = False

            # ── FRED MACRO ASSETS ─────────────────────────────────────────────
            is_macro = known_source == "fred" or asset.get("kind") == "macro"
            if is_macro or forced_dl == "fred":
                if not is_macro and forced_dl == "fred":
                    # 用户强制选择 FRED 模式，但当前资产不是宏观指标，直接跳过
                    logger.info(f"[FRED] Skipping non-macro asset '{name}' in FRED-only mode.")
                    continue
                logger.info(f"[FRED] Downloading macro series: {name} ({ticker})")
                # Priority: asset-level setting > MACRO_SERIES default > native frequency.
                freq = (
                    str(asset.get("frequency")).strip().lower()
                    if asset.get("frequency")
                    else next(
                        (s.get("frequency") for s in MACRO_SERIES if s["name"] == name),
                        None,
                    )
                )
                if freq in {"day", "daily"}:
                    freq = "d"
                elif freq in {"month", "monthly"}:
                    freq = "m"
                elif freq in {"quarter", "quarterly"}:
                    freq = "q"
                elif freq in {"year", "yearly", "annual"}:
                    freq = "a"
                elif freq in {"week", "weekly"}:
                    freq = "w"
                elif freq not in {"d", "w", "m", "q", "a", None}:
                    freq = None
                try:
                    _get_fred_loader().download_series(
                        series_id=ticker,
                        name=name,
                        start_date_fallback=start_fallback,
                        frequency=freq,
                    )
                    success = True
                except Exception as fred_err:
                    logger.warning(f"FRED download failed for {name} ({ticker}): {fred_err}")
                    success = False
                if not success:
                    logger.warning(f"Failed to download FRED series {name} ({ticker}).")
                continue  # 跳过 yahoo/akshare/baostock 路由

            # ── FORCED MODE ───────────────────────────────────────────────────
            if forced_dl != "auto":
                logger.info(f"[FORCED:{forced_dl.upper()}] Downloading {name} ({ticker})")
                if forced_dl == "yahoo":
                    success = yahoo_loader.download_symbol(ticker, name, start_fallback)
                elif forced_dl == "akshare":
                    # Preserve the fine-grained source type (akshare_index etc.) if available
                    ak_source = known_source if (known_source and known_source.startswith("akshare")) else "akshare"
                    success = akshare_loader.download_symbol(ticker, name, start_fallback, source=ak_source)
                elif forced_dl == "baostock":
                    success = baostock_loader.download_symbol(ticker, name, start_fallback)

                if success and asset.get("source") != forced_dl:
                    # Only overwrite source when it's a genuinely different backend
                    if forced_dl != "akshare" or not (known_source and known_source.startswith("akshare")):
                        asset["source"] = forced_dl
                        updated_assets = True

            # ── AUTO MODE (smart routing with fallbacks) ──────────────────────
            elif known_source == "yahoo":
                logger.info(f"Downloading known Yahoo asset: {name} ({ticker})")
                success = yahoo_loader.download_symbol(ticker, name, start_fallback)
                if not success:
                    logger.warning(f"Yahoo failed for {name}. Attempting Akshare fallback...")
                    success = akshare_loader.download_symbol(ticker, name, start_fallback)
                    if success:
                        asset["source"] = "akshare"
                        updated_assets = True
                    else:
                        logger.info(f"Akshare fallback failed for {name}. Attempting Baostock fallback...")
                        try:
                            success = baostock_loader.download_symbol(ticker, name, start_fallback)
                            if success:
                                asset["source"] = "baostock"
                                updated_assets = True
                        except Exception:
                            success = False

            elif known_source is not None and known_source.startswith("akshare"):
                # Handles: "akshare", "akshare_index", "akshare_etf",
                #           "akshare_gold", "akshare_hk_index"
                logger.info(f"Downloading known AkShare asset: {name} ({ticker}) [source={known_source}]")
                success = akshare_loader.download_symbol(
                    ticker, name, start_fallback, source=known_source
                )
                if not success:
                    logger.warning(f"Akshare failed for {name}. Attempting Yahoo fallback...")
                    success = yahoo_loader.download_symbol(ticker, name, start_fallback)
                    if success:
                        asset["source"] = "yahoo"
                        updated_assets = True
                    else:
                        logger.info(f"Yahoo fallback failed for {name}. Attempting Baostock fallback...")
                        try:
                            success = baostock_loader.download_symbol(ticker, name, start_fallback)
                            if success:
                                asset["source"] = "baostock"
                                updated_assets = True
                        except Exception:
                            success = False

            else:
                # Smart Probing Flow for unknown assets
                logger.info(f"Unknown source for {name} ({ticker}). Probing Yahoo, Akshare, and Baostock...")
                d_yf = yahoo_loader.probe_earliest_date(ticker)
                d_ak = akshare_loader.probe_earliest_date(ticker)
                try:
                    d_ba = baostock_loader.probe_earliest_date(ticker)
                except Exception:
                    d_ba = None

                if d_yf is None and d_ak is None and d_ba is None:
                    logger.error(f"Asset {name} ({ticker}) could not be resolved on Yahoo, Akshare, or Baostock.")
                    continue

                available = {}
                if d_yf is not None:
                    available["yahoo"] = d_yf
                if d_ak is not None:
                    available["akshare"] = d_ak
                if d_ba is not None:
                    available["baostock"] = d_ba

                chosen_source = min(available, key=lambda k: available[k])
                logger.info(f"Probe complete. Yahoo: {d_yf}, Akshare: {d_ak}, Baostock: {d_ba}. Chose: {chosen_source} for {name}")
                asset["source"] = chosen_source
                updated_assets = True

                if chosen_source == "yahoo":
                    success = yahoo_loader.download_symbol(ticker, name, start_fallback)
                elif chosen_source == "akshare":
                    success = akshare_loader.download_symbol(ticker, name, start_fallback)
                elif chosen_source == "baostock":
                    success = baostock_loader.download_symbol(ticker, name, start_fallback)

            if not success:
                logger.warning(f"Failed to download/update data for {name} ({ticker}).")

        if updated_assets:
            # Reconstruct the full assets list keeping the modifications since we only got a view for selected
            full_assets = AssetConfigManager.load_assets()
            for full_a in full_assets:
                for sel_a in assets:
                    if full_a["name"] == sel_a["name"] and "source" in sel_a:
                        full_a["source"] = sel_a["source"]
            AssetConfigManager.save_assets(full_assets)

    except Exception as e:
        logger.error(f"Asset download failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "ok"}


@app.get("/api/macro/series")
def list_macro_series_endpoint() -> List[Dict[str, Any]]:
    """List all pre-defined FRED macro series with local data coverage info.

    Returns:
        List of dicts, each containing series metadata plus:
        - data_start_date: earliest date in local CSV (or None)
        - data_end_date:   latest date in local CSV (or None)
        - row_count:       number of rows in local CSV (or 0)
    """
    macro_storage = os.path.join(project_root, "data", "macro")
    result = []
    for s in MACRO_SERIES:
        entry: Dict[str, Any] = dict(s)
        csv_path = os.path.join(macro_storage, f"{sanitize_filename(s['name'])}.csv")
        if os.path.isfile(csv_path):
            try:
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                entry["data_start_date"] = str(df.index.min().date()) if not df.empty else None
                entry["data_end_date"] = str(df.index.max().date()) if not df.empty else None
                entry["row_count"] = len(df)
            except Exception:
                entry["data_start_date"] = None
                entry["data_end_date"] = None
                entry["row_count"] = 0
        else:
            entry["data_start_date"] = None
            entry["data_end_date"] = None
            entry["row_count"] = 0
        result.append(entry)
    return result


@app.post("/api/macro/download")
def download_macro_endpoint(req: MacroDownloadRequest) -> Dict[str, Any]:
    """Download selected FRED macro series and optionally build aligned macro CSV.

    Args:
        req: MacroDownloadRequest with names, storage_path, build_csv, output_path.

    Returns:
        Dict with 'downloaded', 'skipped', 'output_path' (if build_csv) keys.
    """
    try:
        storage_path = (
            os.path.join(project_root, req.storage_path)
            if req.storage_path
            else os.path.join(project_root, "data", "macro")
        )
        loader = FREDIncrementalLoader(storage_path=storage_path)

        if req.names:
            series_to_dl = [s for s in MACRO_SERIES if s["name"] in req.names]
            if not series_to_dl:
                raise HTTPException(
                    status_code=400,
                    detail=f"None of the requested names matched MACRO_SERIES: {req.names}",
                )
        else:
            series_to_dl = MACRO_SERIES

        logger.info(f"FRED download triggered for: {[s['name'] for s in series_to_dl]}")
        loader.download_batch(series_to_dl)

        response: Dict[str, Any] = {"downloaded": [s["name"] for s in series_to_dl]}

        if req.build_csv:
            output_abs = (
                os.path.join(project_root, req.output_path)
                if not os.path.isabs(req.output_path)
                else req.output_path
            )
            df = loader.build_aligned_macro_csv(series_to_dl, output_path=output_abs)
            response["output_path"] = output_abs
            response["rows"] = len(df)
            logger.info(f"Aligned macro CSV saved to {output_abs} ({len(df)} rows)")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Macro download failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/assets/process")
def process_assets_endpoint(req: DownloadRequest) -> Dict[str, str]:
    """Process and align data for **all** configured assets.

    说明:
    - 先删除已有的 aligned_assets.csv，再从所有已下载数据重新构建。
    - 忽略 UI 传入的勾选资产列表，始终处理全部配置资产。
    - 若 US30Y 和 US3M 均存在，自动派生 TermSpread 列。
    """
    assets = AssetConfigManager.load_assets()
    if not assets:
        raise HTTPException(
            status_code=400, detail="No assets configured for processing."
        )

    aligned_path = os.path.join(project_root, "data_processed", "aligned_assets.csv")
    if os.path.exists(aligned_path):
        os.remove(aligned_path)

    try:
        processor = DataProcessor(
            raw_path=os.path.join(project_root, "data"),
            processed_path=os.path.join(project_root, "data_processed"),
        )
        full_path = processor.process_and_align(
            assets,
            output_filename="aligned_assets.csv",
        )

        return {
            "detail": "Alignment complete.",
            "aligned_filename": os.path.basename(full_path),
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"System error: {str(e)}")


@app.get("/api/trend_models")
def list_trend_models() -> List[Dict[str, str]]:
    """Scan .private_data/models for .pkl and .pt files."""
    models_dir = os.path.join(project_root, ".private_data", "models")
    if not os.path.isdir(models_dir):
        return []

    results: List[Dict[str, str]] = []
    for fname in os.listdir(models_dir):
        if not (fname.endswith(".pkl") or fname.endswith(".pt")):
            continue
        rel_path = os.path.join(".private_data", "models", fname)
        results.append(
            {
                "key": rel_path,
                "label": fname,
            }
        )
    return results


@app.post("/api/backtest", response_model=BacktestResult)
def backtest_endpoint(req: BacktestConfig) -> BacktestResult:
    try:
        return manager.service.run_job(req)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Internal backtest error: {e}"
        )


@app.get("/api/configs/best")
def list_best_configs() -> List[str]:
    """List all saved configuration files in data_processed/configs/best."""
    configs_dir = os.path.join(project_root, "data_processed", "configs", "best")
    if not os.path.isdir(configs_dir):
        return []
    files = [f for f in os.listdir(configs_dir) if f.endswith(".yaml")]
    files.sort(reverse=True)
    return files


@app.get("/api/configs/best/{filename}")
def get_best_config(filename: str) -> Dict[str, Any]:
    """Get the contents of a specific saved configuration."""
    configs_dir = os.path.join(project_root, "data_processed", "configs", "best")
    file_path = os.path.join(configs_dir, filename)
    import yaml

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Config file not found")
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@app.post("/api/configs/best")
def save_best_config(req: BacktestConfig) -> Dict[str, str]:
    """Save the current configuration to data_processed/configs/best."""
    from datetime import datetime
    import yaml

    configs_dir = os.path.join(project_root, "data_processed", "configs", "best")
    os.makedirs(configs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    algo_short = req.algorithm.replace("_rebalance", "")
    model_info = f"_{req.trend_model_type}" if req.use_trend_model else ""
    yaml_filename = f"config_{timestamp}_{algo_short}{model_info}.yaml"
    yaml_path = os.path.join(configs_dir, yaml_filename)

    try:
        cfg_dict = req.model_dump() if hasattr(req, "model_dump") else req.dict()
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg_dict, f, allow_unicode=True, sort_keys=False)
        return {"detail": "Config saved successfully", "filename": yaml_filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save config YAML: {e}")


results_dir = os.path.join(project_root, "data_processed")
if os.path.exists(results_dir):
    app.mount("/results", StaticFiles(directory=results_dir), name="results")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
