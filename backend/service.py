import os
import sys
import inspect
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# Ensure project root is in path so we can import strategies
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from strategies.backtest_engine import BacktestEngine
from strategies.algorithms import RebalanceAlgorithms
from utils.naming import sanitize_filename

class BacktestConfig(BaseModel):
    """Configuration model for backtest execution."""

    data_file: str = Field(
        default="data_processed/aligned_assets.csv",
        description="Path to aligned data CSV relative to project root",
    )
    output_dir: str = Field(
        default="data_processed",
        description="Directory to save backtest HTML results",
    )
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    initial_capital: float = Field(100_000.0, description="Initial portfolio capital")
    fees: float = Field(0.0005, description="Transaction fee rate")
    rebalance_freq: str = Field("QE", description="Calendar frequency (e.g., QE, YE)")
    benchmark_cols: List[str] = Field(
        default_factory=lambda: ["Nasdaq100", "GoldIndex", "US30Y", "US3M"],
        description="Benchmark columns for comparison",
    )
    rebalance_interval_days: Optional[int] = Field(
        None,
        description="Fixed interval in days for rebalancing",
    )
    algorithm: str = Field(
        default="permanent_portfolio_rebalance",
        description="Algorithm function name ending with '_rebalance'",
    )
    asset_cols: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional list of columns to use as strategy assets; "
            "if None, all columns in the data file will be used"
        ),
    )
    use_trend_model: bool = Field(
        default=False,
        description="Whether to enable the unsupervised trend model to gate rebalancing",
    )
    model_type: str = Field(
        default="kmeans",
        description="Trend model type: 'kmeans', 'autoencoder', or 'hmm'",
    )
    model_lookback_days: int = Field(
        default=60,
        description="Number of past days used as the fixed window for the trend model",
    )
    model_threshold: float = Field(
        default=0.5,
        description="Trend score threshold in [0,1]; only rebalance when score >= threshold",
    )


class BacktestResult(BaseModel):
    """Response model for backtest results."""

    stats: Dict[str, Any]
    result_html_path: str
    result_url: str
    algorithm: str


class BacktestService:
    """Service class responsible for orchestrating backtest simulations."""

    def __init__(self) -> None:
        self.algorithm_map: Dict[str, Dict[str, Any]] = self._discover_algorithms()

    def _discover_algorithms(self) -> Dict[str, Dict[str, Any]]:
        """Discover rebalance algorithms from the RebalanceAlgorithms class."""
        algo_map: Dict[str, Dict[str, Any]] = {}

        for name, obj in inspect.getmembers(RebalanceAlgorithms, inspect.isfunction):
            if not name.endswith("_rebalance"):
                continue

            algo_map[name] = {
                "fn": obj,
                "label": name.replace("_", " ").title(),
                "description": f"Auto-discovered algorithm: {name}",
            }
        return algo_map

    def _resolve_path(self, path: str) -> str:
        """Resolve a relative path against the project root."""
        return os.path.join(project_root, path)

    def _build_aligned_filename_from_asset_cols(self, asset_cols: Optional[List[str]]) -> str:
        """Build aligned CSV filename based on strategy asset universe.

        If asset_cols is None or empty, fall back to the default global file.
        """
        if not asset_cols:
            return "data_processed/aligned_assets.csv"

        names_sorted = sorted(asset_cols)
        key = "_".join(names_sorted)
        safe_key = sanitize_filename(key)
        return os.path.join("data_processed", f"aligned_{safe_key}.csv")

    def run_job(self, cfg: BacktestConfig) -> BacktestResult:
        """Run a complete backtest simulation job."""
        # 1) 优先使用全局对齐文件 data_processed/aligned_assets.csv，
        #    其中可能包含比本次策略资产更多的列，供 Benchmarks 使用。
        global_rel = cfg.data_file or "data_processed/aligned_assets.csv"
        global_abs = self._resolve_path(global_rel)

        data_file_abs: str
        if os.path.exists(global_abs):
            data_file_abs = global_abs
        else:
            # 2) 如果找不到全局文件，则退回到按资产集合推导专属对齐文件的旧逻辑
            data_file_rel = self._build_aligned_filename_from_asset_cols(cfg.asset_cols)
            data_file_abs = self._resolve_path(data_file_rel)

        if not os.path.exists(data_file_abs):
            # 如果既没有全局文件也没有资产集合专属文件，提示用户先执行下载与处理
            raise FileNotFoundError("请先执行 Download & Process 生成 aligned_assets.csv 或该资产集合的对齐文件")

        output_dir_abs = self._resolve_path(cfg.output_dir)
        os.makedirs(output_dir_abs, exist_ok=True)

        algo_info = self.algorithm_map.get(cfg.algorithm)
        if not algo_info:
            raise RuntimeError(f"Unsupported algorithm: {cfg.algorithm}")

        rebalance_fn = algo_info["fn"]

        strategy = BacktestEngine(data_file_abs, cfg.initial_capital)
        strategy.run_backtest(
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            rebalance_freq=cfg.rebalance_freq,
            fees=cfg.fees,
            rebalance_fn=rebalance_fn,
            rebalance_interval_days=cfg.rebalance_interval_days,
            asset_cols=cfg.asset_cols,
            use_trend_model=cfg.use_trend_model,
            model_lookback_days=cfg.model_lookback_days,
            model_threshold=cfg.model_threshold,
            model_type=cfg.model_type,
        )

        stats = strategy.get_performance_stats()
        if not stats:
            raise RuntimeError("Backtest simulation produced no statistics.")

        html_path = strategy.plot_results(
            output_dir=output_dir_abs,
            benchmark_cols=cfg.benchmark_cols,
        )
        if not html_path:
            raise RuntimeError("Failed to generate performance chart.")

        rel_html_path = os.path.relpath(html_path, project_root)
        result_url = f"/results/{os.path.basename(html_path)}"

        return BacktestResult(
            stats=stats,
            result_html_path=rel_html_path,
            result_url=result_url,
            algorithm=cfg.algorithm,
        )
