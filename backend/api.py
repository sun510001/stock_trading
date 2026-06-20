from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from glob import glob
import json
import os
import re
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

from backend.service import BacktestBatchResult, BacktestConfig, BacktestResult, BacktestService
from backend.assets_config import AssetConfigManager
from data_loader.yahoo_downloader import YahooIncrementalLoader
from data_loader.akshare_downloader import AkshareIncrementalLoader
from data_loader.baostock_downloader import BaostockIncrementalLoader
from data_loader.eastmoney_index_downloader import (
    EastmoneyDownloadConfig,
    EastmoneyIndexDownloader,
)
from data_loader.data_processor import DataProcessor
from data_loader.fred_downloader import FREDIncrementalLoader, MACRO_SERIES
from data_loader.regime_dataset_builder import RegimeDatasetBuilder
from utils.naming import sanitize_filename
from utils.csv_utils import load_date_indexed_csv
from logger import logger

app = FastAPI(title="Backtest API", version="1.0.0")

_ALLOWED_RELEASE_RULES = {
    "none",
    "next_thursday",
    "next_friday",
    "third_thursday_same_month",
    "calendar_lag",
}

_DEFAULT_MACRO_RELEASE_CONFIG: Dict[str, Dict[str, Any]] = {
    "JOBLESS_CLAIMS": {"release_rule": "next_thursday", "release_lag_days": 0},
    "PhillyFed": {"release_rule": "third_thursday_same_month", "release_lag_days": 0},
    "M2_YoY": {"release_rule": "calendar_lag", "release_lag_days": 35},
}

_TREND_MODEL_DIRS: Dict[str, List[str]] = {
    "kmeans_window": ["kmeans_window"],
    "random_forest": ["random_forest"],
    "torch_mlp": ["search"],
    "window_transformer": ["search"],
    "torch_regression": ["search"],
    "bottom_signal_overlay": [],
    "regime_horizon_router": [],
}

_TREND_MODEL_STATIC_FILES: Dict[str, List[str]] = {
    "regime_horizon_router": [os.path.join(".private_data", "plan", "regime_horizon_router.json")],
    "bottom_signal_overlay": [os.path.join(".private_data", "plan", "bottom_signal_overlay_default.json")],
}

_TREND_MODEL_EXTS: Dict[str, tuple[str, ...]] = {
    "kmeans_window": (".pkl",),
    "random_forest": (".pkl",),
    "torch_mlp": (".pt",),
    "window_transformer": (".pt",),
    "torch_regression": (".pt",),
    "bottom_signal_overlay": (".json",),
    "regime_horizon_router": (".json",),
}


def _read_training_run_manifest(run_dir: str) -> Optional[Dict[str, Any]]:
    """Load run metadata for a directory-based model artifact."""
    manifest_path = os.path.join(run_dir, "training_run.json")
    if not os.path.isfile(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _manifest_matches_model_type(manifest: Dict[str, Any], model_type: str) -> bool:
    """Return whether a run manifest is compatible with the requested UI filter."""
    family = str(manifest.get("model_family", "")).strip()
    if model_type in ("torch_mlp", "window_transformer"):
        return family in {"torch_mlp", "window_transformer"}
    if model_type == "torch_regression":
        return family == "window_regression"
    if model_type == "regime_horizon_router":
        return family == "regime_horizon_router"
    return family == model_type


def _list_dynamic_router_json_files() -> List[str]:
    plan_dir = os.path.join(project_root, ".private_data", "plan")
    if not os.path.isdir(plan_dir):
        return []

    rel_paths: List[str] = []
    for abs_path in sorted(glob(os.path.join(plan_dir, "regime_horizon_router*.json"))):
        if not os.path.isfile(abs_path):
            continue
        rel_paths.append(os.path.relpath(abs_path, project_root))
    return rel_paths


def _list_dynamic_bottom_overlay_json_files() -> List[str]:
    plan_dir = os.path.join(project_root, ".private_data", "plan")
    if not os.path.isdir(plan_dir):
        return []

    rel_paths: List[str] = []
    for abs_path in sorted(glob(os.path.join(plan_dir, "bottom_signal_overlay*.json"))):
        if not os.path.isfile(abs_path):
            continue
        rel_paths.append(os.path.relpath(abs_path, project_root))
    return rel_paths


def _resolve_project_relative_path(rel_path: str) -> str:
    abs_path = os.path.normpath(os.path.join(project_root, rel_path))
    if os.path.commonpath([project_root, abs_path]) != project_root:
        raise HTTPException(status_code=400, detail="Path escapes project root")
    return abs_path


def _load_json_artifact(rel_path: str) -> Dict[str, Any]:
    abs_path = _resolve_project_relative_path(rel_path)
    if not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail=f"Artifact not found: {rel_path}")
    try:
        with open(abs_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON artifact: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Artifact JSON must be an object")
    return payload


def _summarize_trend_model_artifact(model_type: str, model_path: str) -> Dict[str, Any]:
    allowed_exts = _TREND_MODEL_EXTS.get(model_type)
    if not allowed_exts or not model_path.endswith(allowed_exts):
        raise HTTPException(status_code=400, detail="Unsupported model artifact type")

    if not model_path.endswith(".json"):
        return {
            "model_type": model_type,
            "model_path": model_path,
            "artifact_type": model_type,
        }

    payload = _load_json_artifact(model_path)
    summary: Dict[str, Any] = {
        "model_type": model_type,
        "model_path": model_path,
        "artifact_type": payload.get("artifact_type") or payload.get("base_model_type") or model_type,
    }

    if model_type == "bottom_signal_overlay":
        summary.update(
            {
                "target_asset": payload.get("target_asset"),
                "breadth_assets": payload.get("breadth_assets") or [],
                "breadth_asset_count": len(payload.get("breadth_assets") or []),
                "zscore_window": payload.get("zscore_window"),
                "ma_window": payload.get("ma_window"),
                "momentum_window": payload.get("momentum_window"),
                "breadth_lag": payload.get("breadth_lag"),
                "router_neutral_min": payload.get("router_neutral_min"),
                "router_neutral_max": payload.get("router_neutral_max"),
                "activation_threshold": payload.get("activation_threshold"),
                "overlay_weight": payload.get("overlay_weight"),
                "base_model_type": payload.get("base_model_type"),
                "base_model_path": payload.get("base_model_path"),
            }
        )
        base_model_path = payload.get("base_model_path")
        if isinstance(base_model_path, str) and base_model_path.endswith(".json"):
            base_payload = _load_json_artifact(base_model_path)
            summary["base_summary"] = {
                "ui_preset": (base_payload.get("ui_preset") or {}).get("name"),
                "benchmark_interval": (base_payload.get("ui_preset") or {}).get("benchmark_interval") or {},
                "signal_formula": base_payload.get("signal_formula"),
                "horizon_weights": base_payload.get("horizon_weights") or {},
                "ui_threshold": (base_payload.get("ui_preset") or {}).get("model_threshold"),
            }
        return summary

    if model_type == "regime_horizon_router":
        summary.update(
            {
                "ui_preset": (payload.get("ui_preset") or {}).get("name"),
                "benchmark_interval": (payload.get("ui_preset") or {}).get("benchmark_interval") or {},
                "signal_formula": payload.get("signal_formula"),
                "horizon_weights": payload.get("horizon_weights") or {},
                "ui_threshold": (payload.get("ui_preset") or {}).get("model_threshold"),
                "rolling_oos": payload.get("rolling_oos") or {},
            }
        )
        return summary

    return summary


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
        downloader: Optional[str] = Field(
            None,
            description=(
                "Canonical download backend: 'yahoo', 'akshare', 'baostock', 'eastmoney', or 'fred'. "
                "When set, auto-mode uses this directly without probing."
            ),
        )
        fallback_downloads: Optional[List[Dict[str, Any]]] = Field(
            None,
            description=(
                "Ordered fallback download definitions. Each item may set downloader, "
                "ticker, source, and frequency while keeping the canonical asset name."
            ),
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
            None, description="The origin of data (e.g., 'yahoo', 'akshare', 'eastmoney')"
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
    downloader = str(asset.get("downloader") or "").strip().lower()
    source = str(asset.get("source") or "").strip().lower()
    kind = str(asset.get("kind") or "").strip().lower()
    return kind == "macro" or source == "fred" or downloader == "fred"


def _resolve_known_downloader(asset: Dict[str, Any]) -> str:
    """Resolve the effective canonical downloader for an asset.

    Args:
        asset: Asset configuration dictionary.

    Returns:
        Canonical downloader key or an empty string if unresolved.
    """
    known_dl = str(asset.get("downloader") or "").strip().lower()
    if known_dl:
        return known_dl

    known_source = str(asset.get("source") or "").strip().lower()
    if known_source == "yahoo":
        return "yahoo"
    if known_source.startswith("akshare"):
        return "akshare"
    if known_source in {"baostock", "eastmoney", "fred"}:
        return known_source
    return ""


def _download_with_eastmoney(
    data_path: str,
    ticker: str,
    name: str,
    start_fallback: str,
) -> bool:
    """Download a CN quote via the raw Eastmoney kline endpoint."""
    config = EastmoneyDownloadConfig(
        symbol=str(ticker).strip(),
        name=str(name).strip(),
        start_date=str(start_fallback).strip(),
        end_date=pd.Timestamp.today().strftime("%Y-%m-%d"),
        output_dir=data_path,
    )
    downloader = EastmoneyIndexDownloader(config)
    frame = downloader.download()
    if frame.empty:
        return False
    downloader.save(frame)
    return True


def _can_try_akshare_us_fallback(ticker: str) -> bool:
    """Return whether a Yahoo ticker is compatible with AkShare's US path.

    AkShare's ``stock_us_daily`` endpoint works for plain US stock/ETF-style
    symbols. Yahoo-specific index, FX, crypto, and exchange-suffixed symbols
    should be rejected early to avoid noisy retry loops.

    Args:
        ticker: Raw asset ticker.

    Returns:
        True when the ticker looks compatible with AkShare US symbols.
    """
    return bool(re.fullmatch(r"[A-Z0-9]{1,10}", str(ticker or "").strip().upper()))


def _can_try_baostock_fallback(ticker: str) -> bool:
    """Return whether a ticker can be mapped to Baostock format.

    Args:
        ticker: Raw asset ticker.

    Returns:
        True when Baostock has a supported symbol mapping.
    """
    return BaostockIncrementalLoader._to_baostock_ticker(str(ticker or "").strip()) is not None


def _download_configured_fallbacks(
    *,
    asset: Dict[str, Any],
    name: str,
    ticker: str,
    start_fallback: str,
    data_path: str,
    akshare_loader: AkshareIncrementalLoader,
    baostock_loader: BaostockIncrementalLoader,
) -> bool:
    """Try per-asset fallback downloaders without changing the canonical asset name."""
    fallbacks = asset.get("fallback_downloads") or []
    if not isinstance(fallbacks, list):
        return False

    for fallback in fallbacks:
        if not isinstance(fallback, dict):
            continue
        fallback_downloader = str(fallback.get("downloader") or "").strip().lower()
        fallback_ticker = str(fallback.get("ticker") or ticker).strip()
        fallback_source = str(fallback.get("source") or fallback_downloader).strip().lower()
        fallback_frequency = fallback.get("frequency") or asset.get("frequency")

        logger.info(
            f"Trying configured fallback for {name}: "
            f"downloader={fallback_downloader}, ticker={fallback_ticker}, source={fallback_source}"
        )

        try:
            if fallback_downloader == "fred":
                # Store FRED fallback under data/<asset-name>.csv so price/yield
                # assets keep the same downstream processing path and column name.
                success = FREDIncrementalLoader(storage_path=data_path).download_series(
                    series_id=fallback_ticker,
                    name=name,
                    start_date_fallback=start_fallback,
                    frequency=str(fallback_frequency).strip().lower() if fallback_frequency else None,
                )
                if success:
                    return True
            if fallback_downloader == "akshare":
                success = akshare_loader.download_symbol(
                    fallback_ticker,
                    name,
                    start_fallback,
                    source=fallback_source or "akshare",
                )
                if success:
                    return True
            if fallback_downloader == "baostock":
                success = baostock_loader.download_symbol(fallback_ticker, name, start_fallback)
                if success:
                    return True
            if fallback_downloader == "eastmoney":
                success = _download_with_eastmoney(data_path, fallback_ticker, name, start_fallback)
                if success:
                    return True
        except Exception as exc:
            logger.warning(
                f"Configured fallback failed for {name} "
                f"({fallback_downloader}:{fallback_ticker}): {exc}"
            )

    return False


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
    is_macro = _is_macro_asset(asset)
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
            aligned_df = load_date_indexed_csv(aligned_path).head(1)
            aligned_cols = [str(c) for c in aligned_df.columns]
        except Exception:
            aligned_cols = []

    for asset in assets:
        csv_path = _resolve_asset_csv_path(asset, data_dir=data_dir, macro_dir=macro_dir)
        d_start, d_end = None, None
        source = asset.get("source", None)
        if os.path.exists(csv_path):
            try:
                df = load_date_indexed_csv(csv_path)
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

    # 将派生列暴露在 Asset Management 列表中（虚拟资产，不写入 assets.json）
    _DERIVED_COLS = {
        "TermSpread": "利率期限利差（US30Y − US3M），策略内部派生。",
        "Net_Liquidity": "美联储净流动性（WALCL − WTREGEN − RRPONTSYD），自动派生。",
    }
    for derived_name, derived_desc in _DERIVED_COLS.items():
        if derived_name in aligned_cols and not any(a.name == derived_name for a in results):
            results.append(
                AssetModels.AssetWithMeta(
                    name=derived_name,
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
                    description=derived_desc,
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
            # Preserve existing downloader if not explicitly set by client
            if not new_data.get("downloader") and "downloader" in a:
                new_data["downloader"] = a["downloader"]

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

        # Forced downloader override: 'yahoo' | 'akshare' | 'baostock' | 'eastmoney' | 'fred' | 'auto'
        forced_dl = (req.downloader or "auto").strip().lower()
        if forced_dl not in ("yahoo", "akshare", "baostock", "eastmoney", "fred", "auto"):
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
            is_macro = _is_macro_asset(asset)
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
                elif forced_dl == "eastmoney":
                    success = _download_with_eastmoney(data_path, ticker, name, start_fallback)

                if success and asset.get("source") != forced_dl:
                    # Only overwrite source when it's a genuinely different backend
                    if forced_dl != "akshare" or not (known_source and known_source.startswith("akshare")):
                        asset["source"] = forced_dl
                        asset["downloader"] = forced_dl
                        updated_assets = True

            # ── AUTO MODE (smart routing with fallbacks) ──────────────────────
            else:
                # Resolve effective downloader: prefer explicit `downloader` field, fall back to `source`
                known_dl = _resolve_known_downloader(asset)
                logger.info(f"[AUTO] Resolved downloader='{known_dl}' for {name} (source={known_source})")

                if known_dl == "yahoo":
                    logger.info(f"Downloading known Yahoo asset: {name} ({ticker})")
                    success = yahoo_loader.download_symbol(ticker, name, start_fallback)
                    if not success:
                        logger.info(f"Yahoo failed for {name}. Trying configured fallbacks.")
                        success = _download_configured_fallbacks(
                            asset=asset,
                            name=name,
                            ticker=ticker,
                            start_fallback=start_fallback,
                            data_path=data_path,
                            akshare_loader=akshare_loader,
                            baostock_loader=baostock_loader,
                        )
                    if not success:
                        if _can_try_baostock_fallback(ticker):
                            logger.info(f"Attempting Baostock fallback for {name}...")
                            try:
                                success = baostock_loader.download_symbol(ticker, name, start_fallback)
                                if success:
                                    asset["source"] = "baostock"
                                    asset["downloader"] = "baostock"
                                    updated_assets = True
                            except Exception:
                                success = False
                        else:
                            logger.info(f"Baostock fallback not available for {name} ({ticker!r}).")

                elif known_dl == "akshare":
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
                            asset["downloader"] = "yahoo"
                            updated_assets = True
                        else:
                            logger.info(f"Yahoo fallback failed for {name}. Attempting Baostock fallback...")
                            try:
                                success = baostock_loader.download_symbol(ticker, name, start_fallback)
                                if success:
                                    asset["source"] = "baostock"
                                    asset["downloader"] = "baostock"
                                    updated_assets = True
                            except Exception:
                                success = False

                elif known_dl == "baostock":
                    logger.info(f"Downloading known Baostock asset: {name} ({ticker})")
                    try:
                        success = baostock_loader.download_symbol(ticker, name, start_fallback)
                    except Exception as bs_err:
                        logger.warning(f"Baostock failed for {name}: {bs_err}")
                        success = False

                elif known_dl == "eastmoney":
                    logger.info(f"Downloading known Eastmoney asset: {name} ({ticker})")
                    try:
                        success = _download_with_eastmoney(data_path, ticker, name, start_fallback)
                    except Exception as em_err:
                        logger.warning(f"Eastmoney failed for {name}: {em_err}")
                        success = False

                elif known_dl == "fred":
                    logger.info(f"[AUTO:FRED] Downloading macro series: {name} ({ticker})")
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
                    asset["downloader"] = chosen_source
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
                    if full_a["name"] == sel_a["name"]:
                        if "source" in sel_a:
                            full_a["source"] = sel_a["source"]
                        if "downloader" in sel_a:
                            full_a["downloader"] = sel_a["downloader"]
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
    - 输出统一交易日历且仅在资产首个真实观测之后做前向填充。
    - 额外生成独立的 regime_daily_dataset.csv 供训练与标签审计使用。
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
        regime_path = os.path.join(
            project_root,
            "data_processed",
            "regime_daily_dataset.csv",
        )
        RegimeDatasetBuilder().build_from_aligned_csv(full_path, regime_path)

        return {
            "detail": "Alignment complete. Canonical regime dataset refreshed in data_processed.",
            "aligned_filename": os.path.basename(full_path),
            "aligned_output_path": os.path.relpath(full_path, project_root),
            "regime_filename": os.path.basename(regime_path),
            "regime_output_path": os.path.relpath(regime_path, project_root),
            "regime_dataset_role": "canonical_base_dataset",
            "plan_artifacts_dir": os.path.join(".private_data", "plan"),
            "plan_artifacts_note": "Audit and derived regime artifacts are generated separately under .private_data/plan.",
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"System error: {str(e)}")


@app.get("/api/trend_models")
def list_trend_models(
    model_type: Optional[str] = Query(
        default=None,
        description="Optional trend model type used to filter by mapped subdirectory.",
    ),
) -> List[Dict[str, str]]:
    """List trend model artifacts available to the UI.

    ``history_models`` is intentionally excluded from UI-visible results.
    Directory-based runs are preferred and shown by folder name, while
    non-training artifacts can be exposed from static plan paths.
    """
    models_dir = os.path.join(project_root, ".private_data", "models")
    if not os.path.isdir(models_dir):
        return []

    requested_type = (model_type or "").strip()
    if requested_type:
        folder_names = _TREND_MODEL_DIRS.get(requested_type, [])
        static_files = _TREND_MODEL_STATIC_FILES.get(requested_type, [])
        if requested_type == "regime_horizon_router":
            static_files = sorted(set([*static_files, *_list_dynamic_router_json_files()]))
        if requested_type == "bottom_signal_overlay":
            static_files = sorted(set([*static_files, *_list_dynamic_bottom_overlay_json_files()]))
        allowed_exts = _TREND_MODEL_EXTS.get(requested_type)
        if (not folder_names and not static_files) or not allowed_exts:
            return []
    else:
        folder_names = sorted(
            {name for names in _TREND_MODEL_DIRS.values() for name in names}
        )
        static_files = [
            path
            for paths in _TREND_MODEL_STATIC_FILES.values()
            for path in paths
        ]
        static_files.extend(_list_dynamic_router_json_files())
        static_files.extend(_list_dynamic_bottom_overlay_json_files())
        allowed_exts = (".pkl", ".pt", ".json")

    results_by_key: Dict[str, Dict[str, str]] = {}

    for rel_path in static_files:
        abs_path = os.path.join(project_root, rel_path)
        if not os.path.isfile(abs_path) or not abs_path.endswith(allowed_exts):
            continue
        results_by_key.setdefault(
            rel_path,
            {
                "key": rel_path,
                "label": os.path.basename(rel_path),
            },
        )

    for root, dirs, files in os.walk(models_dir):
        rel_root = os.path.relpath(root, models_dir)
        parts = rel_root.split(os.sep)
        if "history_models" in parts:
            dirs[:] = []
            continue

        if "training_run.json" not in files:
            continue

        manifest = _read_training_run_manifest(root)
        if not manifest:
            continue
        if requested_type and not _manifest_matches_model_type(manifest, requested_type):
            continue

        primary_model = str(manifest.get("primary_model", "")).strip()
        if not primary_model or not primary_model.endswith(allowed_exts):
            continue

        rel_path = os.path.relpath(root, project_root)
        results_by_key[rel_path] = {
            "key": rel_path,
            "label": os.path.basename(root),
        }
        dirs[:] = []

    for folder_name in folder_names:
        folder_path = os.path.join(models_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        for root, _, files in os.walk(folder_path):
            rel_root = os.path.relpath(root, models_dir)
            if "history_models" in rel_root.split(os.sep):
                continue
            if root != folder_path and "training_run.json" in files:
                rel_path = os.path.relpath(root, project_root)
                results_by_key.setdefault(
                    rel_path,
                    {
                        "key": rel_path,
                        "label": os.path.basename(root),
                    },
                )
                continue
            for fname in files:
                if not fname.endswith(allowed_exts):
                    continue
                if re.search(r"_fold\d+\.[^.]+$", fname):
                    continue
                abs_path = os.path.join(root, fname)
                rel_path = os.path.relpath(abs_path, project_root)
                rel_from_models = os.path.relpath(abs_path, models_dir)
                results_by_key.setdefault(
                    rel_path,
                    {
                        "key": rel_path,
                        "label": rel_from_models,
                    },
                )

    results = list(results_by_key.values())
    results.sort(key=lambda item: item["label"].lower())
    return results


@app.get("/api/trend_model_artifact")
def get_trend_model_artifact(
    model_type: str = Query(..., description="Trend model type"),
    model_path: str = Query(..., description="Project-relative path to artifact"),
) -> Dict[str, Any]:
    return _summarize_trend_model_artifact(
        model_type=model_type.strip(),
        model_path=model_path.strip(),
    )


@app.post("/api/backtest", response_model=BacktestBatchResult)
def backtest_endpoint(req: BacktestConfig) -> BacktestBatchResult:
    try:
        selected_algorithms = [str(req.algorithm).strip()]
        if req.strategy_selector_enabled:
            requested = [
                str(name).strip()
                for name in (req.selected_algorithms or [])
                if str(name).strip()
            ]
            if not requested:
                raise HTTPException(status_code=400, detail="At least one strategy must be selected")
            selected_algorithms = requested

        results: List[BacktestResult] = []
        for algorithm in selected_algorithms:
            run_cfg = req.model_copy(update={"algorithm": algorithm})
            results.append(manager.service.run_job(run_cfg))

        return BacktestBatchResult(
            results=results,
            selected_algorithms=selected_algorithms,
        )
    except HTTPException:
        raise
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
    selected_algorithms = [
        str(name).strip()
        for name in (req.selected_algorithms or [])
        if str(name).strip()
    ]
    if req.strategy_selector_enabled and len(selected_algorithms) > 1:
        algo_short = "multi_strategy"
    else:
        algo_key = selected_algorithms[0] if selected_algorithms else req.algorithm
        algo_short = algo_key.replace("_rebalance", "")
    model_info = f"_{req.trend_model_type}" if req.use_trend_model else ""
    yaml_filename = f"config_{timestamp}_{algo_short}{model_info}.yaml"
    yaml_path = os.path.join(configs_dir, yaml_filename)

    try:
        cfg_dict = (
            req.model_dump(by_alias=True)
            if hasattr(req, "model_dump")
            else req.dict(by_alias=True)
        )
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
