import numpy as np

class RebalanceAlgorithms:
    """
    Collection of portfolio rebalancing algorithms.
    
    Each method implements a specific strategy to redistribute asset weights
    and returns the new unit holdings and the updated portfolio value.
    """

    @staticmethod
    def permanent_portfolio_rebalance(
        current_units: np.ndarray,
        prices: np.ndarray,
        fees: float,
    ) -> tuple[np.ndarray, float]:
        """
        Standard Equal-Weight Rebalance for the Permanent Portfolio.
        
        Args:
            current_units (np.ndarray): Current number of units held for each asset.
            prices (np.ndarray): Today's prices for each asset.
            fees (float): Transaction fee rate (e.g., 0.0005 for 5bps).
            
        Returns:
            tuple[np.ndarray, float]: A tuple containing:
                - new_units (np.ndarray): Updated number of units for each asset.
                - current_val_after_fees (float): Final portfolio value after deducting fees.
        """
        # 1. Calculate current market value
        current_val = float(np.sum(current_units * prices))
        n_assets = len(prices)
        
        # 2. Target allocation (Equal weight)
        target_val_per_asset = current_val / n_assets
        current_asset_vals = current_units * prices
        
        # 3. Calculate trade volume and fees
        diffs = target_val_per_asset - current_asset_vals
        trade_volume = np.sum(np.abs(diffs))
        total_fees = trade_volume * fees
        
        # 4. Final portfolio value after fees
        current_val_after_fees = current_val - total_fees
        
        # 5. Determine new units to hold
        new_target_val = current_val_after_fees / n_assets
        new_units = new_target_val / prices
        
        return new_units, current_val_after_fees

    @staticmethod
    def my_new_rebalance(
        current_units: np.ndarray,
        prices: np.ndarray,
        fees: float,
    ) -> tuple[np.ndarray, float]:
        """
        An example custom rebalance function.
        
        Currently delegates to the standard permanent portfolio rebalance logic.
        
        Args:
            current_units (np.ndarray): Current number of units held for each asset.
            prices (np.ndarray): Today's prices for each asset.
            fees (float): Transaction fee rate.
            
        Returns:
            tuple[np.ndarray, float]: (new_units, new_portfolio_value)
        """
        return RebalanceAlgorithms.permanent_portfolio_rebalance(current_units, prices, fees)
