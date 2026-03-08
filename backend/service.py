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
    benchmark_cols: List[str] = Field(
        default_factory=lambda: ["Nasdaq100", "GoldIndex", "20Y_Treasury_ETF", "US3M"],
        description="Benchmark columns for comparison",
    )
    rebalance_interval_days: int = Field(
        30,
        description="Fixed interval in days for rebalancing",
    )
    algorithm: str = Field(
        default="permanent_portfolio_rebalance",
        description="Algorithm function name ending with '_rebalance'",
    )
    candidate_assets: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional list of columns to use as candidate assets; "
            "if None, all columns in the data file will be used"
        ),
    )
    safe_assets: Optional[List[str]] = Field(
        default=["20Y_Treasury_ETF", "GoldIndex", "US3M"],
        description="Optional list of assets to shift to when risk-off (if None, shift to Cash)",
    )
    top_k: int = Field(
        default=3,
        description="Number of top candidate assets to select based on trend score",
    )
    target_volatility: float = Field(
        default=0.10,
        description="Target annualized portfolio volatility (e.g., 0.10 for 10%)",
    )
    vol_lookback: int = Field(
        default=60,
        description="Number of past days to compute inverse volatility and portfolio volatility",
    )
    max_leverage: float = Field(
        default=1.0,
        description="Maximum leverage ratio (1.0 means no leverage)",
    )
    max_asset_weight: float = Field(
        default=1.0,
        description="Hard cap on the weight of any single asset (e.g. 0.30 = 30%). Excess is redistributed among other selected assets.",
    )
    vol_scale_lookback: int = Field(
        default=0,
        description="Short trailing window (days) used exclusively for vol estimation in the scaling layer. 0 = use full vol_lookback window.",
    )
    momentum_threshold: float = Field(
        default=0.0,
        description="Minimum cumulative return for an asset to pass the absolute-momentum filter. 0.0 = original hard-zero.",
    )
    use_sharpe_weighting: bool = Field(
        default=False,
        description="If True, allocation weights are proportional to Sharpe-proxy (return / vol) instead of raw cumulative return.",
    )
    min_blend: float = Field(
        default=0.0,
        description="Minimum momentum blend ratio [0,1]. Even at the lowest trend score the portfolio holds at least this fraction of momentum weights. Guards against out-of-sample model false-negatives. 0.0 = unconstrained soft-gate.",
    )
    fill_residual_with_safe: bool = Field(
        default=True,
        description="If True, route unused portfolio weight into the configured safe assets instead of leaving it as cash.",
    )
    use_trend_model: bool = Field(
        default=False,
        description="Whether to enable the unsupervised trend model to gate rebalancing",
    )
    trend_model_type: str = Field(
        default="kmeans_simple",
        description=(
            "Trend model type: 'kmeans_simple', 'kmeans_window', 'random_forest', "
            "'torch_mlp', 'window_transformer', or 'torch_regression'"
        ),
    )
    model_path: Optional[str] = Field(
        default=None,
        description="Relative path to a persisted trend model file or run folder under project root",
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

    def _build_aligned_filename_from_candidate_assets(self, candidate_assets: Optional[List[str]]) -> str:
        """Build aligned CSV filename based on candidate asset universe.

        If candidate_assets is None or empty, fall back to the default global file.
        """
        if not candidate_assets:
            return "data_processed/aligned_assets.csv"

        names_sorted = sorted(candidate_assets)
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
            data_file_rel = self._build_aligned_filename_from_candidate_assets(cfg.candidate_assets)
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

        # Always work on a *copy* so we never mutate the caller's cfg object.
        actual_candidate_assets = list(cfg.candidate_assets) if cfg.candidate_assets is not None else None

        strategy = BacktestEngine(data_file_abs, cfg.initial_capital)
        strategy.run_backtest(
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            fees=cfg.fees,
            rebalance_fn=rebalance_fn,
            rebalance_interval_days=cfg.rebalance_interval_days,
            candidate_assets=actual_candidate_assets,
            use_trend_model=cfg.use_trend_model,
            model_lookback_days=cfg.model_lookback_days,
            model_threshold=cfg.model_threshold,
            model_type=cfg.trend_model_type,
            model_path=cfg.model_path,
            top_k=cfg.top_k,
            target_volatility=cfg.target_volatility,
            vol_lookback=cfg.vol_lookback,
            max_leverage=cfg.max_leverage,
            safe_assets=cfg.safe_assets,
            max_asset_weight=cfg.max_asset_weight,
            vol_scale_lookback=cfg.vol_scale_lookback,
            momentum_threshold=cfg.momentum_threshold,
            use_sharpe_weighting=cfg.use_sharpe_weighting,
            min_blend=cfg.min_blend,
            fill_residual_with_safe=cfg.fill_residual_with_safe,
        )

        stats = strategy.get_performance_stats(
            benchmark_col=cfg.benchmark_cols[0] if cfg.benchmark_cols else "SP500"
        )
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
