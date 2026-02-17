# Updates

- **2026-02-17**
  - Added UI controls for configuring an unsupervised trend model (model type, fixed lookback window, threshold) and wiring it through the backtest engine.
  - Introduced a new `signal_weighted_rebalance` algorithm option that tilts portfolio weights based on per-asset trend signals (implementation details remain private).
  - Enhanced the Performance Chart to display Total Return, CAGR, Sharpe Ratio, and Max Drawdown for both the strategy and all selected benchmarks.
  - Added a second subplot below the performance chart that shows the portfolio's asset weights over time as a stacked area plot.

---

# Stock Trading Backtest System

A Python-based multi-asset backtesting system featuring automated data downloading, yield-to-price pricing engines, a core backtest engine with pluggable rebalancing algorithms, and an interactive FastAPI-based web console.

The system is designed for backtesting long-term asset allocation strategies (e.g., Permanent Portfolio) with support for arbitrary asset configurations.

---

## Project Structure

```text
project_root/
├── backend/
│   ├── api.py           # FastAPI implementation and interactive Web UI (/ui)
│   └── service.py       # Backtest service layer: Request/Result models and core orchestration
│
├── data/                # Raw asset data (CSV files named using sanitized asset names)
├── data_processed/
│   ├── aligned_assets.csv              # Synchronized price matrix for backtesting
│   └── backtest_results_*.html         # Interactive Plotly charts generated from backtest runs
│
├── data_loader/
│   ├── yahoo_downloader.py  # Incremental OHLCV data downloader using yfinance
│   └── data_processor.py    # Yield-to-price conversion and multi-asset alignment
│
├── strategies/
│   ├── backtest_engine.py      # Core simulation engine (Time loop + algorithm execution)
│   ├── algorithms.py           # Library of rebalancing strategies (Private algorithms)
|   └── algorithms_template.py  # A template for algorithms.py
│
├── utils/
│   ├── decorators.py        # Generic decorators (e.g., @retry)
│   └── tools.py             # Timezone and date utility functions
│
├── logs/app.log             # System runtime logs
├── logger.py                # Global logging configuration
├── main_download.py         # Entry point: Data synchronization and processing
├── main_backtest.py         # Entry point: CLI-based backtest execution
├── requirements.txt
└── README.md
```

---

## Module Functionality

### 1. `backend/`: API and Service Layer

#### `backend/service.py` (BacktestService Class)
- **Algorithm Management**: Automatically discovers rebalancing strategies from the `algorithms` module.
- **Job Execution**: Orchestrates the full backtest workflow, including path resolution, engine instantiation, and result aggregation.
- **Models**: Defines `BacktestConfig` and `BacktestResult` using Pydantic for robust data validation.

#### `backend/api.py` (APIManager Class)
- **Web Interface**: Provides an interactive dashboard at `/ui` using Tailwind CSS.
- **REST Endpoints**:
  - `GET /api/algorithms`: Lists available rebalancing strategies.
  - `GET /api/assets`: Retrieves configured assets and their local data availability.
  - `POST /api/assets`: Manages asset configurations.
  - `POST /api/assets/download`: Triggers data updates with a 1-minute rate limit.
  - `POST /api/backtest`: Executes backtest simulations synchronously.

---

### 2. `data_loader/`: ETL Pipeline

#### `data_loader/yahoo_downloader.py` (YahooIncrementalLoader Class)
- **Incremental Sync**: Only downloads missing data based on local CSV history.
- **Data Sanitization**: Standardizes yfinance output into consistent OHLCV formats.
- **File Safety**: Sanitizes asset names for cross-platform filesystem compatibility.

#### `data_loader/data_processor.py` (DataProcessor Class)
- **Financial Engineering**: Includes engines to convert Treasury yields into synthetic total return price series (Bond and Cash engines).
- **Temporal Alignment**: Performs inner joins across multiple assets to ensure a synchronized timeline for backtesting.

---

### 3. `strategies/`: Core Engine and Algorithms

#### `strategies/algorithms.py` (RebalanceAlgorithms Class)
- **Static Strategies**: Contains pure mathematical implementations of rebalancing logic.
- **Pluggability**: Methods follow a strict signature allowing them to be swapped dynamically by the engine.

#### `strategies/backtest_engine.py` (BacktestEngine Class)
- **Vectorized Calculations**: Uses NumPy for high-performance portfolio value tracking.
- **Simulation Logic**: Handles initial capital allocation, transaction fees, and rebalancing triggers (Calendar frequency or fixed day intervals).
- **Visualization**: Generates standalone Plotly HTML charts with normalized performance comparisons.

---

## Installation

### 1. Create Virtual Environment (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage Guide

### 1. Data Setup

1. Define assets in `config/assets.json` or via the Web UI.
2. Run data synchronization:

   ```bash
   python main_download.py
   ```

### 2. Running Simulations

```bash
mv strategies/algorithms_template.py strategies/algorithms.py
```

#### Via CLI

```bash
python main_backtest.py
```

#### Via Web UI

1. Start the server:

   ```bash
   python backend/api.py
   ```

2. Navigate to `http://127.0.0.1:8000/ui`.
3. Configure parameters and click **Run Backtest**.

---

## FAQ

- **Rate Limiting**: The Yahoo Finance downloader is subject to API rate limits. The system implements a 60-second cooldown on the download endpoint.
- **Adding Algorithms**: New strategies can be added as static methods ending in `_rebalance` within `strategies/algorithms.py`. They will be automatically detected by the system.
