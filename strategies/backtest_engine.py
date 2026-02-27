import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, Union, Optional, List, Callable
from logger import logger

from strategies.algorithms import RebalanceAlgorithms


class BacktestEngine:
    """Core backtest engine handling simulation and visualization."""

    def __init__(self, data_path: str, initial_capital: float = 10000.0) -> None:
        self.data_path: str = data_path
        self.initial_capital: float = initial_capital
        self.data: Optional[pd.DataFrame] = None
        self.portfolio_value: Optional[pd.Series] = None
        self.drawdown: Optional[pd.Series] = None
        self.asset_weights: Optional[pd.DataFrame] = None

        self._load_data()

    def _load_data(self) -> None:
        """Load price data from the specified CSV file."""
        try:
            logger.info(f"Loading data from {self.data_path}...")
            df = pd.read_csv(self.data_path, index_col="Date", parse_dates=True)
            self.data = df.astype(float)
            logger.info(f"Data loaded. Shape: {self.data.shape}")
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise

    @staticmethod
    def _risk_leverage_from_score(
        trend_score: float,
        score_lo: float,
        score_hi: float,
        leverage_min: float,
        leverage_max: float,
    ) -> float:
        """Map trend score in [0, 1] to a leverage value in [leverage_min, leverage_max]."""
        score = float(np.clip(trend_score, 0.0, 1.0))
        lev_min = float(np.clip(leverage_min, 0.0, 1.0))
        lev_max = float(np.clip(leverage_max, 0.0, 1.0))
        if lev_max < lev_min:
            lev_min, lev_max = lev_max, lev_min

        lo = float(np.clip(score_lo, 0.0, 1.0))
        hi = float(np.clip(score_hi, 0.0, 1.0))
        if hi <= lo:
            return lev_max if score >= hi else lev_min

        t = (score - lo) / (hi - lo)
        t = float(np.clip(t, 0.0, 1.0))
        return lev_min + t * (lev_max - lev_min)

    @staticmethod
    def _compute_intragroup_weights(
        rebalance_fn: Callable,
        group_units: np.ndarray,
        group_prices: np.ndarray,
        max_tilt: float,
        group_signals: Optional[np.ndarray],
    ) -> np.ndarray:
        """Infer within-group target weights from the selected rebalance function."""
        n_assets = len(group_prices)
        if n_assets == 0:
            return np.array([])

        group_val = float(np.sum(group_units * group_prices))
        if group_val <= 0:
            return np.full(n_assets, 1.0 / n_assets)

        is_signal_weighted = rebalance_fn is RebalanceAlgorithms.signal_weighted_rebalance
        if is_signal_weighted:
            if group_signals is None or len(group_signals) != n_assets:
                return np.full(n_assets, 1.0 / n_assets)
            new_units, _ = rebalance_fn(
                current_units=group_units,
                prices=group_prices,
                fees=0.0,
                signals=group_signals,
                max_tilt=max_tilt,
            )
        else:
            new_units, _ = rebalance_fn(
                current_units=group_units,
                prices=group_prices,
                fees=0.0,
            )

        target_vals = np.clip(new_units * group_prices, 0.0, None)
        total = float(target_vals.sum())
        if total <= 0:
            return np.full(n_assets, 1.0 / n_assets)
        return target_vals / total

    @staticmethod
    def _compute_safe_group_weights(
        safe_asset_names: List[str],
        safe_allocation_mode: str,
        safe_fixed_weights: Optional[Dict[str, float]],
        safe_single_asset: Optional[str],
    ) -> np.ndarray:
        """Build within-safe-bucket weights based on selected allocation mode."""
        n_assets = len(safe_asset_names)
        if n_assets == 0:
            return np.array([])

        mode = (safe_allocation_mode or "equal_weight").strip().lower()
        if mode == "equal_weight":
            return np.full(n_assets, 1.0 / n_assets)

        if mode == "single_asset":
            selected = safe_single_asset or safe_asset_names[0]
            if selected not in safe_asset_names:
                raise RuntimeError(
                    "safe_single_asset must be one of safe_asset_cols. "
                    f"Got '{selected}', safe_asset_cols={safe_asset_names}"
                )
            weights = np.zeros(n_assets, dtype=float)
            weights[safe_asset_names.index(selected)] = 1.0
            return weights

        if mode == "fixed_weight":
            weights_map = safe_fixed_weights or {}
            unknown = sorted(set(weights_map.keys()) - set(safe_asset_names))
            if unknown:
                raise RuntimeError(
                    "safe_fixed_weights contains assets not in safe_asset_cols: "
                    f"{unknown}"
                )

            values = np.array(
                [float(weights_map.get(name, 0.0)) for name in safe_asset_names],
                dtype=float,
            )
            if np.any(values < 0):
                raise RuntimeError("safe_fixed_weights cannot contain negative values.")
            total = float(values.sum())
            if total <= 0:
                raise RuntimeError(
                    "safe_fixed_weights total must be > 0 for fixed_weight mode."
                )
            return values / total

        raise RuntimeError(
            "Unsupported safe_allocation_mode. "
            "Use 'equal_weight', 'fixed_weight', or 'single_asset'."
        )

    def run_backtest(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        rebalance_freq: str = "QE",
        fees: float = 0.0005,
        rebalance_fn: Callable = RebalanceAlgorithms.permanent_portfolio_rebalance,
        rebalance_interval_days: Optional[int] = None,
        asset_cols: Optional[List[str]] = None,
        use_trend_model: bool = False,
        model_lookback_days: int = 60,
        model_threshold: float = 0.5,
        model_type: str = "kmeans_simple",
        model_path: Optional[str] = None,
        max_tilt: float = 0.1,
        risk_asset_cols: Optional[List[str]] = None,
        safe_asset_cols: Optional[List[str]] = None,
        safe_allocation_mode: str = "equal_weight",
        safe_fixed_weights: Optional[Dict[str, float]] = None,
        safe_single_asset: Optional[str] = None,
        risk_leverage_enabled: bool = False,
        risk_leverage_min: float = 0.2,
        risk_leverage_max: float = 1.0,
        risk_leverage_score_lo: float = 0.2,
        risk_leverage_score_hi: float = 0.8,
    ) -> None:
        """Execute the backtest simulation."""
        logger.info(
            f"Preparing simulation for range: {start_date or 'Start'} to {end_date or 'End'}..."
        )
        logger.info(
            f"BacktestEngine.run_backtest params | use_trend_model={use_trend_model}, "
            f"model_type={model_type}, model_lookback_days={model_lookback_days}, "
            f"model_threshold={model_threshold}, max_tilt={max_tilt}, "
            f"risk_leverage_enabled={risk_leverage_enabled}, "
            f"safe_allocation_mode={safe_allocation_mode}"
        )

        trend_model = None
        if use_trend_model:
            from strategies.trend_models import get_trend_model

            trend_model = get_trend_model(
                model_type=model_type,
                model_path=model_path,
            )

        if self.data is None:
            logger.error("No data available for backtest.")
            return

        # 1. Slice Data (Time Filtering)
        df_slice = self.data.copy()
        if start_date:
            df_slice = df_slice.loc[start_date:]
        if end_date:
            df_slice = df_slice.loc[:end_date]

        if df_slice.empty:
            logger.error("No data found for the specified date range!")
            return

        # Optionally restrict to a subset of asset columns (strategy asset universe)
        if asset_cols:
            missing = [c for c in asset_cols if c not in df_slice.columns]
            if missing:
                raise RuntimeError(f"Missing asset columns in data: {missing}")
            df_slice = df_slice[asset_cols]

        # 在策略资产子集上做严格裁剪，丢弃含 NaN 的日期，
        # 避免全局对齐矩阵中的短历史资产影响其它资产的可用区间。
        before_rows = len(df_slice)
        df_slice = df_slice.dropna(how="any")
        after_rows = len(df_slice)
        if df_slice.empty:
            logger.error("No data left after dropping NaNs for selected assets!")
            return
        if after_rows < before_rows:
            logger.info(
                f"Dropped {before_rows - after_rows} rows with NaNs for selected strategy assets."
            )

        actual_start = df_slice.index[0].strftime("%Y-%m-%d")
        actual_end = df_slice.index[-1].strftime("%Y-%m-%d")
        logger.info(
            f"Actual Backtest Range: {actual_start} to {actual_end} ({len(df_slice)} days)"
        )

        # === 若使用 kmeans_window, 先基于训练时的 feature_cols 校验列名一致性 ===
        trend_feature_cols: Optional[List[str]] = None
        if use_trend_model and trend_model is not None and hasattr(trend_model, "feature_cols"):
            trend_feature_cols = getattr(trend_model, "feature_cols") or None
            if trend_feature_cols:
                missing_feat = [c for c in trend_feature_cols if c not in self.data.columns]
                if missing_feat:
                    raise RuntimeError(
                        "当前 aligned_assets.csv 缺少模型训练所需列: "
                        f"{missing_feat}. 请使用相同列集合重新训练该模型, "
                        "或选择与当前数据匹配的模型文件。"
                    )
                logger.info(
                    f"Trend model feature_cols validated against data columns: {trend_feature_cols}"
                )

        # 2. Setup Variables
        prices = df_slice.values
        dates = df_slice.index
        n_days, n_assets = prices.shape
        asset_names = list(df_slice.columns)
        name_to_idx = {name: idx for idx, name in enumerate(asset_names)}

        # Resolve risk/safe buckets. If both are empty, all strategy assets are treated as risk.
        risk_input = list(dict.fromkeys(risk_asset_cols or []))
        safe_input = list(dict.fromkeys(safe_asset_cols or []))
        all_bucket_cols = set(risk_input + safe_input)
        missing_bucket_cols = [c for c in all_bucket_cols if c not in name_to_idx]
        if missing_bucket_cols:
            raise RuntimeError(
                f"Risk/Safe bucket columns are not in strategy assets: {missing_bucket_cols}"
            )
        overlap_cols = sorted(set(risk_input).intersection(set(safe_input)))
        if overlap_cols:
            raise RuntimeError(
                f"Assets cannot be in both risk and safe buckets: {overlap_cols}"
            )

        if not risk_input and not safe_input:
            risk_indices = list(range(n_assets))
            safe_indices: List[int] = []
        else:
            risk_indices = [name_to_idx[c] for c in risk_input]
            safe_indices = [name_to_idx[c] for c in safe_input]
            assigned = set(risk_indices + safe_indices)
            unassigned = [idx for idx in range(n_assets) if idx not in assigned]
            if unassigned:
                # Keep behavior robust: any strategy assets not explicitly bucketed default to risk.
                risk_indices.extend(unassigned)
                logger.info(
                    "Some strategy assets are not bucketed and will default to risk assets: "
                    f"{[asset_names[j] for j in unassigned]}"
                )

        if risk_leverage_enabled and not use_trend_model:
            logger.warning(
                "risk_leverage_enabled=True but use_trend_model=False. "
                "Leverage will remain at max (no score-based reduction)."
            )

        safe_asset_names = [asset_names[j] for j in safe_indices]
        safe_group_weights = self._compute_safe_group_weights(
            safe_asset_names=safe_asset_names,
            safe_allocation_mode=safe_allocation_mode,
            safe_fixed_weights=safe_fixed_weights,
            safe_single_asset=safe_single_asset,
        )

        # Identify rebalance indices
        if rebalance_interval_days is not None and rebalance_interval_days > 0:
            rb_indices = set(range(0, n_days, rebalance_interval_days))
        else:
            rb_dates = df_slice.index.to_series().resample(rebalance_freq).last().index
            rb_indices = set(df_slice.index.get_indexer(rb_dates, method="ffill"))

        # 3. Initialization
        portfolio_history = np.zeros(n_days)
        weights_history = np.zeros((n_days, n_assets))

        start_prices = prices[0]
        target_allocation = self.initial_capital / n_assets
        current_units = (target_allocation / start_prices) * (1 - fees)
        initial_invested = float(np.sum(current_units * start_prices))
        cash_balance = self.initial_capital - initial_invested
        portfolio_history[0] = initial_invested + cash_balance
        weights_history[0] = (current_units * start_prices) / portfolio_history[0]

        # 4. Time Loop
        for i in range(1, n_days):
            today_prices = prices[i]
            current_asset_vals = current_units * today_prices
            current_val = float(np.sum(current_asset_vals) + cash_balance)

            if i in rb_indices:
                signals = None
                trend_score = 1.0

                if use_trend_model and model_lookback_days > 0 and trend_model is not None:
                    if i < model_lookback_days:
                        portfolio_history[i] = current_val
                        weights_history[i] = (
                            (current_units * today_prices) / current_val
                            if current_val > 0
                            else np.zeros(n_assets)
                        )
                        continue

                    window = df_slice.iloc[i - model_lookback_days : i]
                    if window.empty:
                        portfolio_history[i] = current_val
                        weights_history[i] = (
                            (current_units * today_prices) / current_val
                            if current_val > 0
                            else np.zeros(n_assets)
                        )
                        continue

                    window_arr = window.values.astype(float)

                    # === 统一计算全局趋势分数 ===
                    if trend_feature_cols:
                        # 按模型训练时的 feature_cols 从全局数据中取出对应列的窗口
                        # 注意: 使用 self.data 而非 df_slice, 避免策略资产子集影响特征列选择
                        full_window = self.data.loc[window.index, trend_feature_cols].dropna(how="any")
                        if full_window.empty or len(full_window) < model_lookback_days:
                            trend_score = 0.5
                        else:
                            # 使用多列矩阵接口, 复刻训练时多列拼特征的逻辑
                            mat = full_window.values.astype(float)
                            if hasattr(trend_model, "predict_latest_score_from_matrix"):
                                trend_score = float(
                                    trend_model.predict_latest_score_from_matrix(mat)
                                )
                            else:
                                # 兼容性兜底: 退回单列接口
                                close_series = full_window.iloc[:, 0].values.astype(float)
                                trend_score = float(
                                    trend_model.predict_latest_score_from_series(
                                        close=close_series
                                    )
                                )
                    elif hasattr(trend_model, "predict_latest_score_from_series"):
                        # 兼容旧模型: 使用策略资产子集的第一列作为趋势判断基准
                        close_series = window.iloc[:, 0].values.astype(float)
                        trend_score = float(
                            trend_model.predict_latest_score_from_series(
                                close=close_series
                            )
                        )
                    else:
                        # 兼容更旧的简单模型(直接基于价格矩阵打分)
                        trend_score = float(trend_model.predict_score(window_arr))

                    trend_score = max(0.0, min(1.0, trend_score))
                    logger.info(
                        f"Trend model | day_idx={i}, lookback={model_lookback_days}, "
                        f"score={trend_score:.3f}, model_type={model_type}"
                    )
                    if trend_score < model_threshold and not risk_leverage_enabled:
                        logger.info(
                            f"Rebalance skipped at index {i} due to low trend_score={trend_score:.3f} "
                            f"(threshold={model_threshold:.3f})"
                        )
                        portfolio_history[i] = current_val
                        weights_history[i] = (
                            (current_units * today_prices) / current_val
                            if current_val > 0
                            else np.zeros(n_assets)
                        )
                        continue

                    # 每个资产的信号, 供信号加权算法使用
                    signals = trend_model.predict_asset_scores(window_arr)

                # Risk leverage controls total risk-bucket exposure.
                risk_budget = 1.0
                safe_budget = 0.0
                if risk_leverage_enabled:
                    risk_budget = self._risk_leverage_from_score(
                        trend_score=trend_score,
                        score_lo=risk_leverage_score_lo,
                        score_hi=risk_leverage_score_hi,
                        leverage_min=risk_leverage_min,
                        leverage_max=risk_leverage_max,
                    )
                    if safe_indices:
                        safe_budget = max(0.0, 1.0 - risk_budget)
                    else:
                        safe_budget = 0.0

                # If only safe bucket exists, allocate all investable capital to safe bucket.
                if not risk_indices and safe_indices:
                    risk_budget = 0.0
                    safe_budget = 1.0

                # Compose target weights (cash is the residual: 1 - sum(target_weights)).
                target_weights = np.zeros(n_assets)
                if risk_indices and risk_budget > 0:
                    risk_signals = (
                        np.asarray(signals)[risk_indices]
                        if signals is not None
                        else None
                    )
                    risk_group_weights = self._compute_intragroup_weights(
                        rebalance_fn=rebalance_fn,
                        group_units=current_units[risk_indices],
                        group_prices=today_prices[risk_indices],
                        max_tilt=max_tilt,
                        group_signals=risk_signals,
                    )
                    target_weights[risk_indices] = risk_group_weights * risk_budget

                if safe_indices and safe_budget > 0:
                    target_weights[safe_indices] = safe_group_weights * safe_budget

                # Normalize if numerical drift pushes sum above 1.0.
                target_sum = float(target_weights.sum())
                if target_sum > 1.0 and target_sum > 0:
                    target_weights = target_weights / target_sum
                    target_sum = float(target_weights.sum())

                target_vals = target_weights * current_val
                diffs = target_vals - current_asset_vals
                trade_volume = float(np.sum(np.abs(diffs)))
                total_fees = trade_volume * fees
                current_val_after_fees = max(0.0, current_val - total_fees)

                target_vals_after_fees = target_weights * current_val_after_fees
                cash_balance = current_val_after_fees - float(np.sum(target_vals_after_fees))
                current_units = np.divide(
                    target_vals_after_fees,
                    today_prices,
                    out=np.zeros_like(target_vals_after_fees),
                    where=today_prices > 0,
                )
                current_val = current_val_after_fees

                if (
                    rebalance_fn is RebalanceAlgorithms.signal_weighted_rebalance
                    and signals is not None
                ):
                    logger.info(
                        f"Rebalance (signal_weighted) | idx={i}, max_tilt={max_tilt:.3f}, "
                        f"signals={np.round(signals, 3).tolist()}, risk_budget={risk_budget:.3f}, "
                        f"safe_budget={safe_budget:.3f}, cash_budget={max(0.0, 1.0 - target_sum):.3f}"
                    )
                else:
                    logger.info(
                        f"Rebalance | idx={i}, risk_budget={risk_budget:.3f}, "
                        f"safe_budget={safe_budget:.3f}, cash_budget={max(0.0, 1.0 - target_sum):.3f}"
                    )

            portfolio_history[i] = current_val
            weights_history[i] = (
                (current_units * today_prices) / current_val
                if current_val > 0
                else np.zeros(n_assets)
            )

        # 5. Finalize
        self.portfolio_value = pd.Series(
            portfolio_history, index=dates, name="Portfolio Value"
        )
        self.asset_weights = pd.DataFrame(
            weights_history, index=dates, columns=df_slice.columns
        )
        running_max = self.portfolio_value.cummax()
        self.drawdown = (self.portfolio_value / running_max) - 1

        logger.info("Simulation complete.")

    def get_performance_stats(self) -> Dict[str, float]:
        """Calculate performance statistics for the backtest."""
        if self.portfolio_value is None or self.portfolio_value.empty:
            return {}

        returns = self.portfolio_value.pct_change(fill_method=None).dropna()
        start_val = self.portfolio_value.iloc[0]
        end_val = self.portfolio_value.iloc[-1]

        n_years = (
            self.portfolio_value.index[-1] - self.portfolio_value.index[0]
        ).days / 365.25
        cagr = (end_val / start_val) ** (1 / n_years) - 1
        sharpe = (
            (returns.mean() / returns.std()) * np.sqrt(252)
            if returns.std() != 0
            else 0
        )
        max_dd = self.drawdown.min() if self.drawdown is not None else 0
        vol = returns.std() * np.sqrt(252)

        return {
            "Start Value": float(start_val),
            "End Value": float(end_val),
            "Total Return": float((end_val / start_val) - 1),
            "CAGR": float(cagr),
            "Sharpe Ratio": float(sharpe),
            "Max Drawdown": float(max_dd),
            "Volatility": float(vol),
            "Years": float(n_years),
        }

    def plot_results(
        self, output_dir: str, benchmark_cols: Union[str, List[str]] = "Stocks"
    ) -> str:
        """Generate a comparative plot of strategy vs benchmarks."""
        if self.portfolio_value is None or self.portfolio_value.empty:
            logger.warning("Portfolio value is empty. Cannot plot.")
            return ""

        filename = "backtest_results.html"
        full_path = os.path.join(output_dir, filename)

        logger.info(f"Generating comparative plot to {full_path}...")

        start_date = self.portfolio_value.index[0]
        end_date = self.portfolio_value.index[-1]
        strategy_norm = (self.portfolio_value / self.portfolio_value.iloc[0]) - 1

        if isinstance(benchmark_cols, str):
            benchmark_list = [benchmark_cols]
        else:
            benchmark_list = list(benchmark_cols)

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.65, 0.35],
            subplot_titles=(
                f"Performance: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                "Portfolio Weights",
            ),
        )

        # Strategy stats
        strat_rets = self.portfolio_value.pct_change(fill_method=None).dropna()
        strat_years = (end_date - start_date).days / 365.25
        strat_cagr = (
            (self.portfolio_value.iloc[-1] / self.portfolio_value.iloc[0])
            ** (1 / strat_years)
            - 1
        )
        strat_sharpe = (
            (strat_rets.mean() / strat_rets.std()) * np.sqrt(252)
            if strat_rets.std() != 0
            else 0
        )
        strat_running_max = self.portfolio_value.cummax()
        strat_dd = (self.portfolio_value / strat_running_max) - 1
        strat_max_dd = strat_dd.min()
        strat_total_ret = strategy_norm.iloc[-1]

        fig.add_trace(
            go.Scatter(
                x=strategy_norm.index,
                y=strategy_norm,
                mode="lines",
                name=(
                    "Strategy<br>"
                    f"Total: {strat_total_ret:.1%}, CAGR: {strat_cagr:.1%}<br>"
                    f"Sharpe: {strat_sharpe:.2f}, MaxDD: {strat_max_dd:.1%}"
                ),
                line=dict(color="#00CC96", width=2),
            ),
            row=1,
            col=1,
        )

        default_colors = [
            "#EF553B",
            "#AB63FA",
            "#19D3F3",
            "#636EFA",
            "#FFA15A",
            "#FF6692",
        ]
        for idx, col in enumerate(benchmark_list):
            if self.data is not None and col in self.data.columns:
                bench_series = self.data[col].loc[start_date:end_date]
                if not bench_series.empty:
                    bench_norm = (bench_series / bench_series.iloc[0]) - 1
                    bench_total_ret = bench_norm.iloc[-1]

                    bench_rets = bench_series.pct_change(fill_method=None).dropna()
                    years = (bench_series.index[-1] - bench_series.index[0]).days / 365.25
                    cagr = (
                        (bench_series.iloc[-1] / bench_series.iloc[0])
                        ** (1 / years)
                        - 1
                    )
                    sharpe = (
                        (bench_rets.mean() / bench_rets.std()) * np.sqrt(252)
                        if bench_rets.std() != 0
                        else 0
                    )
                    running_max = bench_series.cummax()
                    dd = (bench_series / running_max) - 1
                    max_dd = dd.min()

                    color = default_colors[idx % len(default_colors)]
                    fig.add_trace(
                        go.Scatter(
                            x=bench_norm.index,
                            y=bench_norm,
                            mode="lines",
                            name=(
                                f"Benchmark ({col})<br>"
                                f"Total: {bench_total_ret:.1%}, CAGR: {cagr:.1%}<br>"
                                f"Sharpe: {sharpe:.2f}, MaxDD: {max_dd:.1%}"
                            ),
                            line=dict(color=color, width=1, dash="dot"),
                        ),
                        row=1,
                        col=1,
                    )

        # Weights subplot (stacked area)
        if self.asset_weights is not None and not self.asset_weights.empty:
            w_df = self.asset_weights.loc[start_date:end_date]
            for idx, col in enumerate(w_df.columns):
                color = default_colors[idx % len(default_colors)]
                fig.add_trace(
                    go.Scatter(
                        x=w_df.index,
                        y=w_df[col],
                        mode="lines",
                        name=f"Weight {col}",
                        stackgroup="weights",
                        line=dict(width=0.5, color=color),
                        opacity=0.8,
                    ),
                    row=2,
                    col=1,
                )

        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(
            title_text="Cumulative Return (%)", tickformat=".0%", row=1, col=1
        )
        fig.update_yaxes(
            title_text="Portfolio Weights", tickformat=".0%", row=2, col=1
        )

        fig.update_layout(
            title=(
                f"Strategy & Benchmarks with Weights | "
                f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            ),
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1.0,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(0,0,0,0.5)",
            ),
        )

        os.makedirs(output_dir, exist_ok=True)
        fig.write_html(full_path)
        return full_path
