import pandas_ta as ta
import pandas as pd
from logger import logger

class StockIndicators:
    def calculate_sma(self, df, period):
        return df['close'].rolling(window=period).mean()

    def calculate_ema(self, df, period):
        return df['close'].ewm(span=period, adjust=False).mean()

    @staticmethod
    def cal_rsi(df, k=14):
        df_close = pd.DataFrame({"close": df["close"]})
        df_close["rsi"] = ta.rsi(df_close["close"], length=k)
        if df_close.empty:
            return None
        else:
            latest_rsi = df_close["rsi"].iloc[-1]
            return latest_rsi

    @staticmethod
    def cal_kdj(df, k=14, d=3):
        try:
            stoch = ta.stoch(high=df["high"], low=df["low"], close=df["close"], k=k, d=d)
            latest_k = stoch[f"STOCHk_{k}_{d}_3"].iloc[-1]
            latest_d = stoch[f"STOCHd_{k}_{d}_3"].iloc[-1]
            j = 3 * latest_k - 2 * latest_d
            return {"K": latest_k, "D": latest_d, "J": j}
        except Exception as e:
            logger.error(f"Error calculating KDJ: {e}")
            return {"K": None, "D": None, "J": None}

    @staticmethod
    def cal_mfi(df, k=14):
        mfi_series = ta.mfi(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            volume=df["volume"],
            length=k,
            append=False,
        )
        latest_mfi = mfi_series.iloc[-1]
        if mfi_series.dropna().empty:
            return None
        else:
            latest_mfi = mfi_series.dropna().iloc[-1]
            return latest_mfi