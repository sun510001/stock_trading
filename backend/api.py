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

app = FastAPI(title="Backtest API", version="1.0.0")


class AssetModels:
    """Namespace for Asset-related Pydantic models."""

    class AssetConfig(BaseModel):
        """Configuration model for a single asset entry."""

        name: str = Field(..., description="Logical name of the asset")
        ticker: str = Field(..., description="Yahoo Finance ticker symbol")
        kind: str = Field(
            ..., description="Asset data type: 'price' or 'yield'"
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

    class AssetWithMeta(AssetConfig):
        """Asset configuration extended with local data metadata."""

        data_start_date: Optional[str] = Field(
            None, description="Earliest date in local CSV file"
        )
        data_end_date: Optional[str] = Field(
            None, description="Latest date in local CSV file"
        )


class APIManager:
    """Manager class responsible for handling high-level API logic and coordinating services."""

    def __init__(self) -> None:
        self.service = BacktestService()
        self.last_download_ts: float = 0.0
        self.download_cooldown: int = 60
        self._ui_html: Optional[str] = None

    def get_ui_html(self) -> str:
        """Return the static HTML for the web dashboard, loaded from backend/ui.html."""
        if self._ui_html is None:
            ui_path = os.path.join(project_root, "backend", "ui.html")
            with open(ui_path, "r", encoding="utf-8") as f:
                self._ui_html = f.read()
        return self._ui_html


# Application instance management
manager = APIManager()


@app.get("/", response_class=HTMLResponse)
def root_endpoint() -> str:
    """Entry point rendering the web dashboard."""
    return manager.get_ui_html()


@app.get("/ui", response_class=HTMLResponse)
def ui_endpoint() -> str:
    """Alternative route for the interactive dashboard."""
    return manager.get_ui_html()


@app.get("/api/algorithms")
def list_algorithms() -> List[Dict[str, str]]:
    """Return a list of all available rebalancing algorithms."""
    return [
        {"key": k, "label": v["label"], "description": v["description"]}
        for k, v in manager.service.algorithm_map.items()
    ]


@app.get("/api/assets", response_model=List[AssetModels.AssetWithMeta])
def get_assets() -> List[AssetModels.AssetWithMeta]:
    """Retrieve all configured assets with their current data availability ranges."""
    assets = AssetConfigManager.load_assets()
    results: List[AssetModels.AssetWithMeta] = []
    data_dir = os.path.join(project_root, "data")

    for asset in assets:
        safe_name = YahooIncrementalLoader.sanitize_filename(asset["name"])
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

        results.append(
            AssetModels.AssetWithMeta(
                **asset, data_start_date=d_start, data_end_date=d_end
            )
        )
    return results


@app.post("/api/assets", response_model=AssetModels.AssetWithMeta)
def create_asset(asset: AssetModels.AssetConfig) -> AssetModels.AssetWithMeta:
    """Add a new asset configuration to the system."""
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
    """Update an existing asset configuration, supporting name changes."""
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
    """Permanently remove an asset configuration."""
    assets = AssetConfigManager.load_assets()
    new_assets = [a for a in assets if a["name"] != name]
    if len(new_assets) == len(assets):
        raise HTTPException(status_code=404, detail="Asset not found.")
    AssetConfigManager.save_assets(new_assets)
    return {"detail": f"Asset '{name}' deleted."}


@app.post("/api/assets/download")
def download_assets_endpoint() -> Dict[str, str]:
    """Trigger incremental data updates for all assets with a safety cooldown."""
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

    try:
        from logger import logger

        logger.info(">>> HTTP TRIGGER: INCREMENTAL DOWNLOAD START <<<")
        downloader = YahooIncrementalLoader(
            storage_path=os.path.join(project_root, "data")
        )
        downloader.download_batch(assets, start_year=1985)

        processor = DataProcessor(
            raw_path=os.path.join(project_root, "data"),
            processed_path=os.path.join(project_root, "data_processed"),
        )
        processor.process_and_align(assets)
        manager.last_download_ts = time.time()
        return {"detail": "Batch update and alignment complete."}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"System error: {str(e)}"
        )


@app.post("/api/backtest", response_model=BacktestResult)
def backtest_endpoint(req: BacktestConfig) -> BacktestResult:
    """Execute a backtest synchronously based on the provided configuration."""
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


# Mount data directory for static chart serving
results_dir = os.path.join(project_root, "data_processed")
if os.path.exists(results_dir):
    app.mount("/results", StaticFiles(directory=results_dir), name="results")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
