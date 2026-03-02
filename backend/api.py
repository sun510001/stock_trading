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
from data_loader.akshare_downloader import AkshareIncrementalLoader
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
        description: Optional[str] = Field(
            "", description="A short introduction or description for the asset"
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
        source = asset.get("source", None)
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
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
        
        updated_assets = False

        for asset in assets:
            name = asset["name"]
            ticker = asset["ticker"]
            
            # Start date logic
            asset_start_date = asset.get("initial_start_date") or asset.get("start_date")
            start_fallback = asset_start_date if asset_start_date else "1985-01-01"
            
            # Known source flow
            known_source = asset.get("source")
            success = False
            
            if known_source == "yahoo":
                logger.info(f"Downloading known Yahoo asset: {name} ({ticker})")
                success = yahoo_loader.download_symbol(ticker, name, start_fallback)
                if not success:
                    logger.warning(f"Yahoo failed for {name}. Attempting Akshare fallback...")
                    success = akshare_loader.download_symbol(ticker, name, start_fallback)
                    if success:
                        asset["source"] = "akshare"
                        updated_assets = True

            elif known_source == "akshare":
                logger.info(f"Downloading known Akshare asset: {name} ({ticker})")
                success = akshare_loader.download_symbol(ticker, name, start_fallback)
                if not success:
                    logger.warning(f"Akshare failed for {name}. Attempting Yahoo fallback...")
                    success = yahoo_loader.download_symbol(ticker, name, start_fallback)
                    if success:
                        asset["source"] = "yahoo"
                        updated_assets = True
            else:
                # Smart Probing Flow for unknown assets
                logger.info(f"Unknown source for {name} ({ticker}). Probing both APIs...")
                d_yf = yahoo_loader.probe_earliest_date(ticker)
                d_ak = akshare_loader.probe_earliest_date(ticker)
                
                if d_yf is None and d_ak is None:
                    logger.error(f"Asset {name} ({ticker}) could not be resolved on either Yahoo or Akshare.")
                    continue
                
                # Compare and decide
                chosen_source = None
                if d_yf is not None and d_ak is None:
                    chosen_source = "yahoo"
                elif d_ak is not None and d_yf is None:
                    chosen_source = "akshare"
                else:
                    # Both are valid, choose the one with the earlier date
                    if d_yf <= d_ak:
                        chosen_source = "yahoo"
                    else:
                        chosen_source = "akshare"
                
                logger.info(f"Probe complete. Yahoo: {d_yf}, Akshare: {d_ak}. Chose: {chosen_source} for {name}")
                asset["source"] = chosen_source
                updated_assets = True
                
                # Perform the actual download
                if chosen_source == "yahoo":
                    success = yahoo_loader.download_symbol(ticker, name, start_fallback)
                elif chosen_source == "akshare":
                    success = akshare_loader.download_symbol(ticker, name, start_fallback)

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

        manager.last_download_ts = time.time()
        return {"detail": "Smart batch download complete."}
    except Exception as e:
        logger.exception("Download fault")
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
