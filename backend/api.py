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
        kind: str = Field(..., description="Asset data type: 'price' or 'yield'")
        engine: Optional[str] = Field(None, description="Pricing engine for yields: 'bond' or 'cash'")
        duration: Optional[float] = Field(None, description="Bond duration for bond pricing engine")
        initial_start_date: Optional[str] = Field("1985-01-02", description="Initial download start date (YYYY-MM-DD)")

    class AssetWithMeta(AssetConfig):
        """Asset configuration extended with local data metadata."""
        data_start_date: Optional[str] = Field(None, description="Earliest date in local CSV file")
        data_end_date: Optional[str] = Field(None, description="Latest date in local CSV file")

class APIManager:
    """Manager class responsible for handling high-level API logic and coordinating services."""
    
    def __init__(self) -> None:
        """Initialize the API manager with backtest service and configuration."""
        self.service = BacktestService()
        self.last_download_ts: float = 0.0
        self.download_cooldown: int = 60

    def get_ui_html(self) -> str:
        """Generate and return the interactive HTML dashboard for the backtest console."""
        return """<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='UTF-8'>
  <title>Backtest Console</title>
  <meta name='viewport' content='width=device-width, initial-scale=1.0'>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body { background:#111; color:#eee; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; margin:0; padding:20px; }
    .panel { background:#1e1e1e; border:1px solid #333; border-radius:8px; padding:16px; margin-bottom:16px; }
    label { display:block; font-size:12px; color:#aaa; margin-bottom:4px; }
    input, select { width:100%; padding:6px 8px; border-radius:4px; border:1px solid #444; background:#222; color:#eee; box-sizing:border-box; }
    button { padding:8px 12px; border:none; border-radius:4px; background:#0af; color:#111; font-weight:bold; cursor:pointer; }
    button:disabled { opacity:0.6; cursor:not-allowed; }
    .grid-stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:8px; }
    .stat-card { background:#181818; border:1px solid #333; border-radius:6px; padding:8px; }
    .stat-label { font-size:12px; color:#888; }
    .stat-value { font-size:18px; font-weight:bold; }
    .error { background:#611; border:1px solid #a33; border-radius:6px; padding:8px; margin-top:8px; display:none; }
    iframe { width:100%; height:600px; border:none; border-radius:6px; background:#000; }
    @media (min-width: 900px) {
      .main-layout { display:grid; grid-template-columns:360px 1fr; gap:16px; }
      .right-col { display:flex; flex-direction:column; gap:16px; }
    }
    table { width:100%; border-collapse:collapse; margin-top:8px; font-size:12px; }
    th, td { border:1px solid #333; padding:4px 6px; text-align:left; }
    th { background:#222; }
  </style>
</head>
<body>
  <div class='max-w-[1400px] mx-auto'>
    <h1 class="text-3xl font-bold mb-2">Backtest Console</h1>
    <p class="text-gray-500 mb-6">Manage assets, configure strategy parameters, and analyze performance.</p>
    
    <div class='main-layout'>
      <!-- LEFT COLUMN: Configuration Panel -->
      <div class='panel self-start'>
        <h2 class="text-lg font-bold mb-4 border-b border-gray-700 pb-2">Configuration</h2>
        <form id='backtest-form' class="space-y-4">
          <div>
            <label>Algorithm</label>
            <select id='algorithm' name='algorithm'></select>
          </div>
          <div class="flex gap-2">
            <div class="flex-1">
              <label>Start Date</label>
              <input type='date' id='start_date' value='2016-01-01'>
            </div>
            <div class="flex-1">
              <label>End Date</label>
              <input type='date' id='end_date' value='2026-01-01'>
            </div>
          </div>
          <div>
            <label>Initial Capital</label>
            <input type='number' id='initial_capital' value='100000'>
          </div>
          <div>
            <label>Rebalance Interval (days, optional)</label>
            <input type='number' id='rebalance_interval_days' placeholder='e.g. 30'>
          </div>
          <div>
            <label>Strategy Assets (comma separated, empty = all)</label>
            <input type='text' id='asset_cols' placeholder='e.g. Nasdaq100, GoldIndex, US30Y, US3M'>
          </div>
          <div>
            <label>Benchmarks (comma separated)</label>
            <input type='text' id='benchmark_cols' value='Nasdaq100, GoldIndex, US30Y, US3M'>
          </div>
          <div class="border-t border-gray-700 pt-3 mt-3">
            <label class="flex items-center gap-2 text-sm">
              <input type='checkbox' id='use_trend_model' />
              <span>Use Trend Model</span>
            </label>
            <div class="mt-2 grid grid-cols-2 gap-2">
              <div>
                <label>Trend Model Type</label>
                <select id='model_type'>
                  <option value='kmeans'>kmeans</option>
                  <option value='autoencoder'>autoencoder</option>
                  <option value='hmm'>hmm</option>
                </select>
              </div>
              <div>
                <label>Lookback Days (fixed window)</label>
                <input type='number' id='model_lookback_days' value='60'>
              </div>
              <div>
                <label>Trend Threshold [0,1]</label>
                <input type='number' id='model_threshold' value='0.5' step='0.05' min='0' max='1'>
              </div>
            </div>
          </div>
          <button type='submit' id='run-btn' class="w-full mt-2">Run Backtest</button>
          <div id='error-msg' class='error'></div>
        </form>
      </div>

      <!-- RIGHT COLUMN: Assets & Results -->
      <div class='right-col'>
        <!-- Assets Section -->
        <div class='panel'>
          <h2 class="text-lg font-bold mb-4 border-b border-gray-700 pb-2">Assets Management</h2>
          <div class="flex gap-2 mb-4">
            <button type='button' id='refresh-assets-btn' class="bg-gray-700">Refresh List</button>
            <button type='button' id='download-assets-btn' class="bg-orange-600 hover:bg-orange-500">Download & Process Data</button>
          </div>
          <table id='assets-table'>
            <thead>
              <tr>
                <th>Name</th>
                <th>Ticker</th>
                <th>Kind</th>
                <th>Engine</th>
                <th>Data Start</th>
                <th>Data End</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody></tbody>
          </table>

          <h3 class="text-md font-bold mt-6 mb-2 text-blue-400">Add / Edit Asset</h3>
          <form id='asset-form' class="grid grid-cols-2 gap-3">
            <input type='hidden' id='asset_original_name'>
            <div><label>Name</label><input type='text' id='asset_name'></div>
            <div><label>Ticker</label><input type='text' id='asset_ticker'></div>
            <div>
              <label>Kind</label>
              <select id='asset_kind'>
                <option value='price'>price</option>
                <option value='yield'>yield</option>
              </select>
            </div>
            <div>
              <label>Engine</label>
              <select id='asset_engine'>
                <option value=''>None</option>
                <option value='bond'>bond</option>
                <option value='cash'>cash</option>
              </select>
            </div>
            <div><label>Initial Start Date</label><input type='date' id='asset_initial_start_date' value='1985-01-02'></div>
            <div><label>Duration (for bond)</label><input type='number' step='0.1' id='asset_duration' placeholder="20.0"></div>
            <div class="col-span-2 flex gap-2 pt-2">
              <button type='submit' id='save-asset-btn' class="flex-1">Save Asset</button>
              <button type='button' id='new-asset-btn' class="flex-1 bg-gray-700">Clear Form</button>
            </div>
          </form>
        </div>

        <!-- Backtest Execution View -->
        <div id='loading' class='panel hidden text-center py-12'>
          <div class="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500 mb-4"></div>
          <div>Processing backtest simulation, please wait...</div>
        </div>

        <div id='results-panel' class="hidden space-y-4">
          <div class='panel'>
            <h2 class="text-lg font-bold mb-4">Statistics</h2>
            <div id='stats-grid' class='grid-stats'></div>
          </div>
          <div class='panel'>
            <h2 class="text-lg font-bold mb-4">Performance Chart</h2>
            <iframe id='plot-frame' src=''></iframe>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    async function loadAlgorithms() {
      try {
        const res = await fetch('/api/algorithms');
        const algos = await res.json();
        const select = document.getElementById('algorithm');
        select.innerHTML = '';
        algos.forEach(a => {
          const opt = document.createElement('option');
          opt.value = a.key;
          opt.textContent = a.label;
          select.appendChild(opt);
        });
      } catch (e) { console.error('Failed to load algorithms', e); }
    }

    async function loadAssets() {
      try {
        const res = await fetch('/api/assets');
        const assets = await res.json();
        const tbody = document.querySelector('#assets-table tbody');
        tbody.innerHTML = '';
        assets.forEach(a => {
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td>${a.name}</td>
            <td>${a.ticker}</td>
            <td>${a.kind}</td>
            <td>${a.engine || '-'}</td>
            <td>${a.data_start_date || '-'}</td>
            <td>${a.data_end_date || '-'}</td>
            <td class="flex gap-1">
              <button type='button' class='edit-asset bg-gray-700 hover:bg-gray-600 px-2 py-1 text-[10px]'>EDIT</button>
              <button type='button' class='delete-asset bg-red-900 hover:bg-red-800 px-2 py-1 text-[10px]'>DEL</button>
            </td>
          `;
          tr.querySelector('.edit-asset').onclick = () => {
            document.getElementById('asset_original_name').value = a.name;
            document.getElementById('asset_name').value = a.name;
            document.getElementById('asset_ticker').value = a.ticker;
            document.getElementById('asset_kind').value = a.kind;
            document.getElementById('asset_engine').value = a.engine || '';
            document.getElementById('asset_duration').value = a.duration || '';
            document.getElementById('asset_initial_start_date').value = a.initial_start_date || '1985-01-02';
          };
          tr.querySelector('.delete-asset').onclick = async () => {
            if (!confirm(`Delete asset '${a.name}'?`)) return;
            const res = await fetch(`/api/assets/${encodeURIComponent(a.name)}`, { method: 'DELETE' });
            if (res.ok) loadAssets(); else alert('Delete failed');
          };
          tbody.appendChild(tr);
        });
      } catch (e) { console.error('Failed to load assets', e); }
    }

    async function saveAsset(e) {
      e.preventDefault();
      const originalName = document.getElementById('asset_original_name').value || null;
      const payload = {
        name: document.getElementById('asset_name').value.trim(),
        ticker: document.getElementById('asset_ticker').value.trim(),
        kind: document.getElementById('asset_kind').value,
        engine: document.getElementById('asset_engine').value || null,
        duration: document.getElementById('asset_duration').value ? parseFloat(document.getElementById('asset_duration').value) : null,
        initial_start_date: document.getElementById('asset_initial_start_date').value || "1985-01-02"
      };

      if (!payload.name || !payload.ticker) { alert('Name and Ticker are required'); return; }

      const url = originalName ? `/api/assets/${encodeURIComponent(originalName)}` : '/api/assets';
      const method = originalName ? 'PUT' : 'POST';

      const res = await fetch(url, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (res.ok) {
        document.getElementById('asset_original_name').value = payload.name;
        loadAssets();
      } else {
        const err = await res.json();
        alert(err.detail || 'Save failed');
      }
    }

    async function downloadAssets() {
      if (!confirm('Start incremental data update for all assets?')) return;
      const btn = document.getElementById('download-assets-btn');
      const originalText = btn.textContent;
      btn.disabled = true;
      btn.textContent = 'Updating...';
      
      try {
        const res = await fetch('/api/assets/download', { method: 'POST' });
        const data = await res.json();
        if (res.status === 429) {
            alert(data.detail);
        } else if (!res.ok) {
            alert('Failed: ' + (data.detail || 'Unknown error'));
        } else {
            alert('Success: Data updated and aligned.');
            loadAssets();
        }
      } catch (e) { alert('Download trigger failed'); }
      finally {
        btn.disabled = false;
        btn.textContent = originalText;
      }
    }

    async function runBacktest(e) {
      e.preventDefault();
      const runBtn = document.getElementById('run-btn');
      const loading = document.getElementById('loading');
      const resultsPanel = document.getElementById('results-panel');
      const errorMsg = document.getElementById('error-msg');
      const statsGrid = document.getElementById('stats-grid');
      const plotFrame = document.getElementById('plot-frame');

      errorMsg.style.display = 'none';
      resultsPanel.classList.add('hidden');
      runBtn.disabled = true;
      loading.classList.remove('hidden');

      const payload = {
        algorithm: document.getElementById('algorithm').value,
        start_date: document.getElementById('start_date').value || null,
        end_date: document.getElementById('end_date').value || null,
        initial_capital: parseFloat(document.getElementById('initial_capital').value),
        asset_cols: document.getElementById('asset_cols').value
          .split(',')
          .map(s => s.trim())
          .filter(Boolean),
        benchmark_cols: document.getElementById('benchmark_cols').value
          .split(',')
          .map(s => s.trim())
          .filter(Boolean),
        rebalance_interval_days: document.getElementById('rebalance_interval_days').value
          ? parseInt(document.getElementById('rebalance_interval_days').value)
          : null,
        use_trend_model: document.getElementById('use_trend_model').checked,
        model_type: document.getElementById('model_type').value,
        model_lookback_days: document.getElementById('model_lookback_days').value
          ? parseInt(document.getElementById('model_lookback_days').value)
          : 60,
        model_threshold: document.getElementById('model_threshold').value
          ? parseFloat(document.getElementById('model_threshold').value)
          : 0.5,
      };

      // If user leaves strategy assets empty, send null so backend uses all columns
      if (payload.asset_cols.length === 0) {
        payload.asset_cols = null;
      }

      try {
        const res = await fetch('/api/backtest', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });

        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || 'Backtest failed');
        }

        const data = await res.json();
        const s = data.stats;
        const metrics = [
          { label: 'Total Return', value: (s['Total Return'] * 100).toFixed(2) + '%' },
          { label: 'CAGR', value: (s['CAGR'] * 100).toFixed(2) + '%' },
          { label: 'Sharpe Ratio', value: s['Sharpe Ratio'].toFixed(2) },
          { label: 'Max Drawdown', value: (s['Max Drawdown'] * 100).toFixed(2) + '%' },
          { label: 'Volatility', value: (s['Volatility'] * 100).toFixed(2) + '%' },
          { label: 'Years', value: s['Years'].toFixed(1) },
        ];

        statsGrid.innerHTML = '';
        metrics.forEach(m => {
          const card = document.createElement('div');
          card.className = 'stat-card';
          card.innerHTML = `<div class='stat-label'>${m.label}</div><div class='stat-value'>${m.value}</div>`;
          statsGrid.appendChild(card);
        });

        plotFrame.src = data.result_url + '?t=' + new Date().getTime();
        resultsPanel.classList.remove('hidden');
      } catch (err) {
        errorMsg.textContent = err.message;
        errorMsg.style.display = 'block';
      } finally {
        loading.classList.add('hidden');
        runBtn.disabled = false;
      }
    }

    document.getElementById('backtest-form').onsubmit = runBacktest;
    document.getElementById('refresh-assets-btn').onclick = loadAssets;
    document.getElementById('download-assets-btn').onclick = downloadAssets;
    document.getElementById('asset-form').onsubmit = saveAsset;
    document.getElementById('new-asset-btn').onclick = () => {
      document.getElementById('asset_original_name').value = '';
      document.getElementById('asset_name').value = '';
      document.getElementById('asset_ticker').value = '';
      document.getElementById('asset_kind').value = 'price';
      document.getElementById('asset_engine').value = '';
      document.getElementById('asset_duration').value = '';
      document.getElementById('asset_initial_start_date').value = '1985-01-02';
    };

    loadAlgorithms();
    loadAssets();
  </script>
</body>
</html>"""

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

        results.append(AssetModels.AssetWithMeta(**asset, data_start_date=d_start, data_end_date=d_end))
    return results

@app.post("/api/assets", response_model=AssetModels.AssetWithMeta)
def create_asset(asset: AssetModels.AssetConfig) -> AssetModels.AssetWithMeta:
    """Add a new asset configuration to the system."""
    assets = AssetConfigManager.load_assets()
    if any(a["name"] == asset.name for a in assets):
        raise HTTPException(status_code=400, detail=f"Asset '{asset.name}' already exists.")
    assets.append(asset.model_dump())
    AssetConfigManager.save_assets(assets)
    return AssetModels.AssetWithMeta(**asset.model_dump())

@app.put("/api/assets/{name}", response_model=AssetModels.AssetWithMeta)
def update_asset(name: str, asset: AssetModels.AssetConfig) -> AssetModels.AssetWithMeta:
    """Update an existing asset configuration, supporting name changes."""
    assets = AssetConfigManager.load_assets()
    if asset.name != name and any(a["name"] == asset.name for a in assets):
        raise HTTPException(status_code=400, detail=f"Conflict: Name '{asset.name}' already exists.")
    
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
        raise HTTPException(status_code=429, detail=f"Rate limit: Wait {remaining}s before next download.")

    assets = AssetConfigManager.load_assets()
    if not assets:
        raise HTTPException(status_code=400, detail="No assets configured for download.")

    try:
        from logger import logger
        logger.info(">>> HTTP TRIGGER: INCREMENTAL DOWNLOAD START <<<")
        downloader = YahooIncrementalLoader(storage_path=os.path.join(project_root, "data"))
        downloader.download_batch(assets, start_year=1985)
        
        processor = DataProcessor(
            raw_path=os.path.join(project_root, "data"),
            processed_path=os.path.join(project_root, "data_processed")
        )
        processor.process_and_align(assets)
        manager.last_download_ts = time.time()
        return {"detail": "Batch update and alignment complete."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"System error: {str(e)}")

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
        raise HTTPException(status_code=500, detail=f"Internal backtest error: {e}")

# Mount data directory for static chart serving
results_dir = os.path.join(project_root, "data_processed")
if os.path.exists(results_dir):
    app.mount("/results", StaticFiles(directory=results_dir), name="results")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
