# =============================================================================
# algorithms_template.py  —  Custom Strategy Template
#
# This file shows how to add a custom rebalance function to TrendAlloc.
#
# HOW IT WORKS
# ------------
# BacktestEngine calls your function on every rebalance day with a
# RebalanceContext object, and expects a RebalanceResult back.
#
#   rebalance_fn(ctx: RebalanceContext) -> RebalanceResult
#
# RebalanceContext gives you everything you need:
#   ctx.current_units     — units currently held  (n_assets,)
#   ctx.today_prices      — prices today           (n_assets,)
#   ctx.cash_balance      — uninvested cash
#   ctx.fees              — transaction fee rate   (e.g. 0.0005)
#   ctx.price_window      — price history          (lookback, n_assets)
#   ctx.returns_window    — return history         (lookback, n_assets)
#   ctx.trend_score       — ML model score ∈ [0,1] (1.0 = no model)
#   ctx.model_threshold   — score threshold for safe-asset switch
#   ctx.top_k             — max assets to hold
#   ctx.target_volatility — annualised vol target
#   ctx.safe_asset_indices— indices of safe-haven assets
#   ctx.col_names         — asset column names
#
# RebalanceResult requires:
#   new_units  (np.ndarray) — target units to hold after rebalance
#   new_cash   (float)      — remaining cash after trades & fees
#
# DISCOVERY
# ---------
# Any @staticmethod whose name ends with "_rebalance" is auto-discovered
# by BacktestService and shown in the UI dropdown.  Keep that suffix.
#
# QUICK START
# -----------
# 1. Copy the `my_custom_rebalance` stub below into .private_data/algorithms.py
# 2. Implement your logic
# 3. Run: python main_backtest.py
# =============================================================================

from __future__ import annotations

import numpy as np

from strategies.algorithms import RebalanceAlgorithms, RebalanceContext, RebalanceResult


class MyCustomAlgorithms:
    """Template class showing how to add custom rebalance strategies.

    Methods whose names end with ``_rebalance`` are auto-discovered by
    :class:`~backend.service.BacktestService` and appear in the UI strategy
    dropdown.  All methods must follow the ``(ctx: RebalanceContext) ->
    RebalanceResult`` signature — see module docstring for full field list.
    """

    @staticmethod
    def my_custom_rebalance(ctx: RebalanceContext) -> RebalanceResult:
        """Example custom rebalance strategy — equal-weight delegation.

        Replace this body with your own logic.  The example below simply
        delegates to the built-in equal-weight algorithm so the file is
        runnable out of the box.

        Args:
            ctx: Rebalance context provided by BacktestEngine each period.

        Returns:
            RebalanceResult with new unit holdings and updated cash balance.
        """
        # --- insert your custom logic here ---
        # Example: delegate to the built-in equal-weight strategy
        return RebalanceAlgorithms.permanent_portfolio_rebalance(ctx)
