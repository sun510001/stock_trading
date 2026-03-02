import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, Union, Optional, List, Callable
from logger import logger

from strategies.algorithms import RebalanceAlgorithms, RebalanceContext, RebalanceResult


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

    def run_backtest(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        fees: float = 0.0005,
        rebalance_fn: Optional[Callable] = None,
        rebalance_interval_days: int = 30,
        candidate_assets: Optional[List[str]] = None,
        use_trend_model: bool = False,
        model_lookback_days: int = 60,
        model_threshold: float = 0.5,
        model_type: str = "kmeans_simple",
        model_path: Optional[str] = None,
        top_k: int = 3,
        target_volatility: float = 0.10,
        vol_lookback: int = 60,
        max_leverage: float = 1.0,
        safe_assets: Optional[List[str]] = None,
    ) -> None:
        """Execute the backtest simulation by delegating trade logic to ``rebalance_fn``.

        BacktestEngine is responsible **only** for:
        - Slicing and pre-processing data for the requested date range.
        - Loading and validating the optional ML trend model.
        - Iterating through trading days (the Time Loop).
        - On rebalance days: gathering context (prices, returns windows, trend
          score) and delegating weight/unit decisions to ``rebalance_fn``.
        - Applying the returned unit positions and recording portfolio history.

        All trading-strategy logic lives inside ``rebalance_fn``.  If
        ``rebalance_fn`` is None the engine falls back to
        :meth:`~strategies.algorithms.RebalanceAlgorithms.permanent_portfolio_rebalance`
        (equal-weight), which matches the default algorithm exposed by
        ``BacktestConfig`` in the backend service.

        Args:
            start_date: ISO date string for backtest start (inclusive).  If
                None, uses the first available date in the data.
            end_date: ISO date string for backtest end (inclusive).  If None,
                uses the last available date in the data.
            fees: Per-trade transaction fee as a fraction of traded value
                (e.g. 0.0005 for 5 bps).
            rebalance_fn: Callable with signature
                ``(ctx: RebalanceContext) -> RebalanceResult``.
                Receives all market context and returns target positions.
            rebalance_interval_days: Number of calendar days between
                consecutive rebalance events (e.g. 30 for monthly).
            candidate_assets: Optional list of column names to restrict the
                asset universe; if None all columns are used.
            use_trend_model: Whether to activate the ML trend overlay model.
            model_lookback_days: Number of past days fed to the trend model as
                its feature window.
            model_threshold: Trend score threshold passed to ``rebalance_fn``
                via :class:`~strategies.algorithms.RebalanceContext`.
            model_type: Identifier string for the trend model flavour.
            model_path: Optional path to a persisted model file (.pkl / .pt).
            top_k: Maximum number of assets to select; forwarded to
                ``RebalanceContext``.
            target_volatility: Target annualised portfolio volatility;
                forwarded to ``RebalanceContext``.
            vol_lookback: Lookback window (days) for momentum and covariance
                estimation; forwarded to ``RebalanceContext``.
            max_leverage: Hard cap on gross exposure; forwarded to
                ``RebalanceContext``.
            safe_assets: Column names of safe-haven assets to hold during
                risk-off periods; forwarded to ``RebalanceContext``.
        """
        logger.info(
            f"Preparing simulation for range: {start_date or 'Start'} to {end_date or 'End'}..."
        )
        logger.info(
            f"BacktestEngine.run_backtest params | rebalance_fn={getattr(rebalance_fn, '__name__', rebalance_fn)}, "
            f"top_k={top_k}, target_volatility={target_volatility:.2f}, "
            f"vol_lookback={vol_lookback}, max_leverage={max_leverage:.2f}"
        )

        # Default strategy: use equal-weight, matching BacktestConfig.algorithm default.
        if rebalance_fn is None:
            rebalance_fn = RebalanceAlgorithms.permanent_portfolio_rebalance
            logger.info(
                "rebalance_fn not provided; defaulting to permanent_portfolio_rebalance (equal-weight)."
            )

        # ── Load optional ML trend model ───────────────────────────────────────
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

        # ── 1. Slice & pre-process data ────────────────────────────────────────
        df_slice = self.data.copy()
        if start_date:
            df_slice = df_slice.loc[start_date:]
        if end_date:
            df_slice = df_slice.loc[:end_date]

        if df_slice.empty:
            logger.error("No data found for the specified date range!")
            return

        if candidate_assets:
            missing = [c for c in candidate_assets if c not in df_slice.columns]
            if missing:
                raise RuntimeError(f"Missing asset columns in data: {missing}")
            df_slice = df_slice[candidate_assets]
        else:
            df_slice = df_slice.dropna(axis=1, how="all")

        df_slice = df_slice.dropna(how="all").ffill().bfill()
        data_filled = self.data.ffill().bfill()

        actual_start = df_slice.index[0].strftime("%Y-%m-%d")
        actual_end = df_slice.index[-1].strftime("%Y-%m-%d")
        logger.info(
            f"Actual Backtest Range: {actual_start} to {actual_end} ({len(df_slice)} days)"
        )

        # Validate trend model feature columns against loaded data
        trend_feature_cols: Optional[List[str]] = None
        if use_trend_model and trend_model is not None and hasattr(trend_model, "feature_cols"):
            trend_feature_cols = getattr(trend_model, "feature_cols") or None
            if trend_feature_cols:
                missing_feat = [c for c in trend_feature_cols if c not in self.data.columns]
                if missing_feat:
                    raise RuntimeError(
                        "aligned_assets.csv is missing columns required by the trend model: "
                        f"{missing_feat}. Re-train the model with the current data columns, "
                        "or select a model file that matches the current dataset."
                    )
                logger.info(
                    f"Trend model feature_cols validated: {trend_feature_cols}"
                )

        # ── 2. Setup arrays ────────────────────────────────────────────────────
        prices = df_slice.values
        dates = df_slice.index
        n_days, n_assets = prices.shape
        col_names: List[str] = list(df_slice.columns)

        returns_arr = df_slice.pct_change(fill_method=None).fillna(0).values

        interval = rebalance_interval_days if rebalance_interval_days and rebalance_interval_days > 0 else 30
        rb_indices = set(range(0, n_days, interval))

        # Pre-compute safe asset index positions (relative to candidate universe)
        safe_asset_indices: List[int] = []
        if safe_assets:
            for col in safe_assets:
                if col in col_names:
                    safe_asset_indices.append(col_names.index(col))

        # Minimum warm-up period before first rebalance
        warmup = max(model_lookback_days if use_trend_model else 0, vol_lookback)

        # ── 3. Initialise portfolio ────────────────────────────────────────────
        portfolio_history = np.zeros(n_days)
        weights_history = np.zeros((n_days, n_assets))
        current_units = np.zeros(n_assets)
        cash_balance = self.initial_capital

        # ── 4. Time Loop ───────────────────────────────────────────────────────
        for i in range(n_days):
            today_prices = prices[i]
            current_asset_vals = current_units * today_prices
            current_val = float(np.sum(current_asset_vals) + cash_balance)

            if i in rb_indices:
                # Skip rebalance until warm-up period has elapsed
                if i < warmup:
                    portfolio_history[i] = current_val
                    weights_history[i] = (
                        current_asset_vals / current_val
                        if current_val > 0
                        else np.zeros(n_assets)
                    )
                    continue

                # ── Compute ML trend score ─────────────────────────────────────
                trend_score = 1.0
                if use_trend_model and trend_model is not None and model_lookback_days > 0:
                    window = df_slice.iloc[i - model_lookback_days : i]
                    if not window.empty:
                        if trend_feature_cols:
                            full_window = data_filled.loc[
                                window.index, trend_feature_cols
                            ].dropna(how="any")
                            if (
                                not full_window.empty
                                and len(full_window) >= model_lookback_days
                            ):
                                mat = full_window.values.astype(float)
                                if hasattr(
                                    trend_model, "predict_latest_score_from_matrix"
                                ):
                                    trend_score = float(
                                        trend_model.predict_latest_score_from_matrix(mat)
                                    )
                                else:
                                    close_series = full_window.iloc[:, 0].values.astype(
                                        float
                                    )
                                    trend_score = float(
                                        trend_model.predict_latest_score_from_series(
                                            close=close_series
                                        )
                                    )
                        elif hasattr(trend_model, "predict_latest_score_from_series"):
                            close_series = window.iloc[:, 0].values.astype(float)
                            trend_score = float(
                                trend_model.predict_latest_score_from_series(
                                    close=close_series
                                )
                            )
                        else:
                            trend_score = float(
                                trend_model.predict_score(window.values.astype(float))
                            )
                    logger.info(
                        f"Rebalance | idx={i}, Market Trend Score {trend_score:.3f}"
                    )

                # ── Build price / return windows for the rebalance function ────
                lookback_start = max(0, i - vol_lookback)
                price_window = prices[lookback_start:i]    # [lookback, n_assets]
                returns_window = returns_arr[lookback_start:i]  # [lookback, n_assets]

                # ── Assemble context and delegate to rebalance_fn ──────────────
                ctx = RebalanceContext(
                    current_units=current_units.copy(),
                    today_prices=today_prices,
                    fees=fees,
                    cash_balance=cash_balance,
                    price_window=price_window,
                    returns_window=returns_window,
                    trend_score=trend_score,
                    model_threshold=model_threshold,
                    top_k=top_k,
                    target_volatility=target_volatility,
                    max_leverage=max_leverage,
                    safe_asset_indices=safe_asset_indices,
                    col_names=col_names,
                )

                result: RebalanceResult = rebalance_fn(ctx)

                # Apply result
                current_units = result.new_units
                cash_balance = result.new_cash
                current_val = float(np.sum(current_units * today_prices) + cash_balance)

                total_exposure = float(np.sum(current_units * today_prices)) / current_val if current_val > 0 else 0.0
                logger.info(
                    f"Rebalance | idx={i}, exposure={total_exposure:.2f}, "
                    f"cash_budget={max(0.0, 1.0 - total_exposure):.2f}"
                )

            portfolio_history[i] = current_val
            weights_history[i] = (
                (current_units * today_prices) / current_val
                if current_val > 0
                else np.zeros(n_assets)
            )

        # ── 5. Finalise ────────────────────────────────────────────────────────
        self.portfolio_value = pd.Series(
            portfolio_history, index=dates, name="Portfolio Value"
        )
        self.asset_weights = pd.DataFrame(
            weights_history, index=dates, columns=df_slice.columns
        )
        running_max = self.portfolio_value.cummax()
        self.drawdown = (self.portfolio_value / running_max) - 1

        logger.info("Simulation complete.")

    def get_performance_stats(self, benchmark_col: str = "SP500") -> Dict[str, float]:
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

        # New metrics calculation
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0

        # Max Recovery Days: longest calendar-day span from a peak to the next
        # point where the portfolio fully recovers to (or above) that peak.
        # Algorithm:
        #   1. Find every date that sets a new all-time high (peak).
        #   2. For each consecutive pair of peaks (peak_i, peak_{i+1}), the
        #      recovery period length = (peak_{i+1} - peak_i).days.
        #      If the portfolio never fully recovers after the last peak, that
        #      ongoing drawdown period stretches to the final date.
        #   3. max_recovery_days = max over all such periods.
        running_max = self.portfolio_value.cummax()
        max_recovery_days = 0
        if not running_max.empty:
            # Dates where the portfolio value equals the running maximum (new high)
            peak_dates = self.portfolio_value.index[
                self.portfolio_value >= running_max - running_max * 1e-9
            ]
            if len(peak_dates) >= 2:
                # Gaps between consecutive new-high dates
                for i in range(1, len(peak_dates)):
                    gap = (peak_dates[i] - peak_dates[i - 1]).days
                    if gap > max_recovery_days:
                        max_recovery_days = gap
            # Also account for an ongoing drawdown that has not yet recovered
            last_peak_date = peak_dates[-1] if len(peak_dates) > 0 else self.portfolio_value.index[0]
            ongoing = (self.portfolio_value.index[-1] - last_peak_date).days
            if ongoing > max_recovery_days:
                max_recovery_days = ongoing

        win_rate = float((returns > 0).mean()) if not returns.empty else 0.0

        avg_win = returns[returns > 0].mean() if not returns[returns > 0].empty else 0.0
        avg_loss = returns[returns < 0].mean() if not returns[returns < 0].empty else 0.0
        pl_ratio = float(avg_win / abs(avg_loss)) if avg_loss != 0 else 0.0

        # Alpha, Beta, Information Ratio w.r.t Benchmark
        alpha = 0.0
        beta = 0.0
        info_ratio = 0.0

        if self.data is not None and benchmark_col in self.data.columns:
            # Align dates
            bench_prices = self.data[benchmark_col].loc[self.portfolio_value.index].dropna()
            bench_rets = bench_prices.pct_change(fill_method=None).dropna()

            # Match lengths
            aligned_rets = pd.concat([returns, bench_rets], axis=1).dropna()
            aligned_rets.columns = ['Strategy', 'Benchmark']

            strat_rets_align = aligned_rets['Strategy']
            bench_rets_align = aligned_rets['Benchmark']

            if not bench_rets_align.empty and bench_rets_align.std() != 0:
                covar = strat_rets_align.cov(bench_rets_align)
                ben_var = bench_rets_align.var()
                beta = covar / ben_var if ben_var != 0 else 0

                annual_strat_ret = strat_rets_align.mean() * 252
                annual_bench_ret = bench_rets_align.mean() * 252
                alpha = annual_strat_ret - beta * annual_bench_ret

                active_rets = strat_rets_align - bench_rets_align
                if active_rets.std() != 0:
                    info_ratio = (active_rets.mean() / active_rets.std()) * np.sqrt(252)

        return {
            "Start Value": float(start_val),
            "End Value": float(end_val),
            "Total Return": float((end_val / start_val) - 1),
            "CAGR": float(cagr),
            "Sharpe Ratio": float(sharpe),
            "Max Drawdown": float(max_dd),
            "Volatility": float(vol),
            "Years": float(n_years),
            "Alpha": float(alpha),
            "Beta": float(beta),
            "Information Ratio": float(info_ratio),
            "Calmar Ratio": float(calmar),
            "Max Recovery Days": int(max_recovery_days),
            "Win Rate": float(win_rate),
            "P/L Ratio": float(pl_ratio),
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
            hoverlabel=dict(namelength=-1),
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
