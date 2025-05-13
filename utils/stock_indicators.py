import pandas_ta as ta
import pandas as pd
from logger import logger
from typing import Dict


class StockIndicators:
    @staticmethod
    def calculate_sma(df: pd.DataFrame, k: int) -> float:
        """
        Calculate the last value of the Simple Moving Average (SMA).

        Parameters:
            df (pd.DataFrame): Must contain a 'close' column.
            k (int): The window size.

        Returns:
            float or None: Last SMA value, or None if not enough data.
        """
        sma_series = df["close"].rolling(window=k).mean()
        return sma_series.iloc[-1] if not sma_series.empty else None

    @staticmethod
    def calculate_ema(df: pd.DataFrame, k: int) -> float:
        """
        Calculate the last value of the Exponential Moving Average (EMA).

        Parameters:
            df (pd.DataFrame): Must contain a 'close' column.
            k (int): The span for EMA.

        Returns:
            float or None: Last EMA value, or None if not enough data.
        """
        ema_series = df["close"].ewm(span=k, adjust=False).mean()
        return ema_series.iloc[-1] if not ema_series.empty else None

    @staticmethod
    def cal_rsi(df: pd.DataFrame, k: int = 14) -> float:
        """
        Calculate the Relative Strength Index (RSI) using pandas_ta.
        Parameters:
            df (pd.DataFrame): DataFrame containing 'close' column.
            k (int, optional): The number of periods to use for the RSI calculation. Defaults to 14.
        Returns:
            float: The latest RSI value, or None if the DataFrame is empty or RSI cannot be calculated.
        """
        df_close = pd.DataFrame({"close": df["close"]})
        df_close["rsi"] = ta.rsi(df_close["close"], length=k)
        if df_close.empty:
            return None
        else:
            latest_rsi = df_close["rsi"].iloc[-1]
            return latest_rsi

    @staticmethod
    def cal_kdj(df: pd.DataFrame, k: int = 14, d: int = 3) -> Dict:
        """
        Calculate the KDJ indicator using pandas_ta.
        Parameters:
            df (pd.DataFrame): DataFrame containing 'high', 'low', and 'close' columns.
            k (int, optional): The number of periods to use for the KDJ calculation. Defaults to 14.
            d (int, optional): The number of periods for the D smoothing. Defaults to 3.
        Returns:
            dict: A dictionary containing the latest K, D, and J values.
        """
        try:
            stoch = ta.stoch(
                high=df["high"], low=df["low"], close=df["close"], k=k, d=d
            )
            latest_k = stoch[f"STOCHk_{k}_{d}_3"].iloc[-1]
            latest_d = stoch[f"STOCHd_{k}_{d}_3"].iloc[-1]
            j = 3 * latest_k - 2 * latest_d
            return {"K": latest_k, "D": latest_d, "J": j}
        except Exception as e:
            logger.error(f"Error calculating KDJ: {e}")
            return {"K": None, "D": None, "J": None}

    @staticmethod
    def cal_mfi(df: pd.DataFrame, k: int = 14) -> float:
        """
        Calculate the Money Flow Index (MFI) using pandas_ta.
        Parameters:
            df (pd.DataFrame): DataFrame containing 'high', 'low', 'close', and 'volume' columns.
            k (int, optional): The number of periods to use for the MFI calculation. Defaults to 14.
        Returns:
            float: The latest MFI value, or None if the DataFrame is empty or MFI cannot be calculated.
        """
        mfi_series = ta.mfi(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            volume=df["volume"],
            length=k,
            append=False,
        )
        if mfi_series.dropna().empty:
            return None
        else:
            latest_mfi = float(mfi_series.dropna().iloc[-1])
            return latest_mfi

    @staticmethod
    def get_adx_signal_dict(row: pd.DataFrame, k: int = 14, adx_threshold: float = 25) -> dict:
        """
        Generate a dictionary with ADX, +DI (DMP), -DI (DMN), and signal.

        Parameters:
            row (pd.Series): A row from a DataFrame containing 'ADX_14', 'DMP_14', 'DMN_14'.
            k (int): The number of periods to use for the ADX calculation.
            adx_threshold (float): Threshold to consider trend as strong.

        Returns:
            dict: A dictionary with ADX values and the signal.
        """
        adx = row.get(f"ADX_{k}")
        dmp = row.get(f"DMP_{k}")
        dmn = row.get(f"DMN_{k}")
        signal = 0

        # Check for NaNs
        if pd.notna(adx) and pd.notna(dmp) and pd.notna(dmn):
            if dmp > dmn and adx > adx_threshold:
                signal = 1
            elif dmn > dmp and adx > adx_threshold:
                signal = 2
        return {"ADX": f"{adx:.2f}", "DMP": f"{dmp:.2f}", "DMN": f"{dmn:.2f}", "signal": signal}

    def cal_adx(self, df: pd.DataFrame, k: int = 14) -> Dict:
        """
        Calculates the ADX (Average Directional Index) using pandas_ta.

        Parameters:
            df (pd.DataFrame): Must include 'high', 'low', 'close' columns.
            k (int): The number of periods to use for the ADX calculation.

        Returns:
            Dict: signal with ADX, +DI, and -DI columns added.
        """
        # Ensure input DataFrame contains required columns
        required_cols = {"high", "low", "close"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        # Use pandas_ta to compute ADX, +DI, and -DI
        adx_df = ta.adx(
            high=df["high"], low=df["low"], close=df["close"], length=k
        )
        if adx_df.empty:
            return None
        else:
            signal = self.get_adx_signal_dict(adx_df.iloc[-1], k=k)
            return signal
