from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import os
import sys
import time
from typing import List, Optional, Dict, Any
import pandas as pd

# Ensure project root is in path so we can import backend.service
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.service import BacktestConfig, BacktestResult, BacktestService
from backend.assets_config import AssetConfigManager
from data_loader.yahoo_downloader import YahooIncrementalLoader
from data_loader.data_processor import DataProcessor
from utils.naming import sanitize_filename

app = FastAPI(title="Backtest API", version="1.0.0")


class AssetModels:
    class AssetConfig(BaseModel):
        name: str = Field(..., description="Logical name of the asset")
        ticker: str = Field(..., description="Yahoo Finance ticker symbol")
        kind: str = Field(..., description="Asset data type: 'price' or 'yield'")
        engine: Optional[str] = Field(
            None, description="Pricing engine for yields: 'bond' or 'cash'"
        )
        duration: Optional[float] = Field(
            None, description="Bond duration for bond pricing engine"
        )
        initial_start_date: Optional[str] = Field(
            "1985-01-02", description="Initial download start date (YYYY-MM-DD)"
        )

    class AssetWithMeta(AssetConfig):
        data_start_date: Optional[str] = Field(
            None, description="Earliest date in local CSV file"
        )
        data_end_date: Optional[str] = Field(
            None, description="Latest date in local CSV file"
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
        if self._ui_html is None:
            ui_path = os.path.join(project_root, "backend", "ui.html")
            with open(ui_path, "r", encoding="utf-8") as f:
                self._ui_html = f.read()
        return self._ui_html


manager = APIManager()


@app.get("/", response_class=HTMLResponse)
def root_endpoint() -> str:
    return manager.get_ui_html()


@app.get("/ui", response_class=HTMLResponse)
def ui_endpoint() -> str:
    return manager.get_ui_html()


@app.get("/api/algorithms")
def list_algorithms() -> List[Dict[str, str]]:
    return [
        {"key": k, "label": v["label"], "description": v["description"]}
        for k, v in manager.service.algorithm_map.items()
    ]


@app.get("/api/assets", response_model=List[AssetModels.AssetWithMeta])
def get_assets() -> List[AssetModels.AssetWithMeta]:
    assets = AssetConfigManager.load_assets()
    results: List[AssetModels.AssetWithMeta] = []
    data_dir = os.path.join(project_root, "data")

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
        safe_name = sanitize_filename(asset["name"])
        csv_path = os.path.join(data_dir, f"{safe_name}.csv")
        d_start, d_end = None, None
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
                if not df.empty:
                    d_start = df.index.min().date().isoformat()
                    d_end = df.index.max().date().isoformat()
            except Exception:
                pass

        processed = asset["name"] in aligned_cols

        results.append(
            AssetModels.AssetWithMeta(
                **asset,
                data_start_date=d_start,
                data_end_date=d_end,
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
    assets.append(asset.model_dump())
    AssetConfigManager.save_assets(assets)
    return AssetModels.AssetWithMeta(**asset.model_dump())


@app.put("/api/assets/{name}", response_model=AssetModels.AssetWithMeta)
def update_asset(name: str, asset: AssetModels.AssetConfig) -> AssetModels.AssetWithMeta:
    assets = AssetConfigManager.load_assets()
    if asset.name != name and any(a["name"] == asset.name for a in assets):
        raise HTTPException(
            status_code=400, detail=f"Conflict: Name '{asset.name}' already exists."
        )

    for i, a in enumerate(assets):
        if a["name"] == name:
            assets[i] = asset.model_dump()
            AssetConfigManager.save_assets(assets)
            return AssetModels.AssetWithMeta(**asset.model_dump())
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

        logger.info(">>> HTTP TRIGGER: INCREMENTAL DOWNLOAD START <<<")
        downloader = YahooIncrementalLoader(
            storage_path=os.path.join(project_root, "data")
        )
        downloader.download_batch(assets, start_year=1985)
        manager.last_download_ts = time.time()
        return {"detail": "Batch download complete."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"System error: {str(e)}")


@app.post("/api/assets/process")
def process_assets_endpoint(req: DownloadRequest) -> Dict[str, str]:
    """Process and align data for **all** configured assets.

    说明:
    - 对齐矩阵始终基于 config/assets.json 的全部资产构建。
    - 若 UI 传入勾选资产 names，且其中同时包含 US30Y 与 US3M，
      则自动在 aligned_assets.csv 中派生 TermSpread 列。
    """
    assets = AssetConfigManager.load_assets()
    if not assets:
        raise HTTPException(
            status_code=400, detail="No assets configured for processing."
        )

    selected_names: Optional[set[str]] = None
    if req.names:
        selected_names = set(req.names)

    try:
        processor = DataProcessor(
            raw_path=os.path.join(project_root, "data"),
            processed_path=os.path.join(project_root, "data_processed"),
        )
        full_path = processor.process_and_align(
            assets,
            output_filename="aligned_assets.csv",
            selected_asset_names=selected_names,
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


results_dir = os.path.join(project_root, "data_processed")
if os.path.exists(results_dir):
    app.mount("/results", StaticFiles(directory=results_dir), name="results")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
