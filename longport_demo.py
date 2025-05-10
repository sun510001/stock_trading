import asyncio
import pandas as pd
import pandas_ta as ta
from time import sleep
from typing import Dict
from logger import logger
from longport.openapi import Config, QuoteContext, Period, AdjustType
from utils import get_time_difference_from_ny
from datetime import datetime

# from send_email import GmailStockNotifier
from send_telegram_bot import TelegramNotifier
from tokens.telegram import token, chat_id


class LongPortSubscribe:
    def __init__(self, symbol: str, element_dict: Dict[str, int]):

        self.symbol = symbol
        self.element_dict = element_dict
        self.config = Config.from_env()  # Load configuration from environment variables
        self.quote_ctx = QuoteContext(self.config)

    def get_info(self, window_size=25):
        candlesticks = self.quote_ctx.candlesticks(
            self.symbol, Period.Min_1, window_size, AdjustType.ForwardAdjust
        )
        data = {
            "close": [k.close for k in candlesticks],
            "high": [k.high for k in candlesticks],
            "low": [k.low for k in candlesticks],
            "volume": [k.volume for k in candlesticks],
            "timestamp": [k.timestamp.timestamp() for k in candlesticks],
        }
        return pd.DataFrame(data).astype(
            {
                "close": "float64",
                "high": "float64",
                "low": "float64",
                "volume": "float64",
                "timestamp": "float64",
            }
        )

    @staticmethod
    def cal_rsi(df, k=14):
        df_close = pd.DataFrame({"close": df["close"]})
        df_close["rsi"] = ta.rsi(df_close["close"], length=k)
        latest_rsi = df_close["rsi"].iloc[-1]
        return latest_rsi

    @staticmethod
    def cal_kdj(df, k=14, d=3):
        stoch = ta.stoch(high=df["high"], low=df["low"], close=df["close"], k=k, d=d)
        latest_k = stoch[f"STOCHk_{k}_{d}_3"].iloc[-1]
        latest_d = stoch[f"STOCHd_{k}_{d}_3"].iloc[-1]
        j = 3 * latest_k - 2 * latest_d
        return {"K": latest_k, "D": latest_d, "J": j}

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
            return "NaN"
        else:
            latest_mfi = mfi_series.dropna().iloc[-1]
            return latest_mfi

    def cal_indicators(self):
        min_offset = get_time_difference_from_ny()
        max_key = max(self.element_dict.values())
        max_period = max_key * 3
        min_period = max_key + 1
        if min_period <= min_offset <= 390:
            # if True:
            df = self.get_info(window_size=max_period)
            last_stock_info = df.iloc[-1].to_dict()
            dt_object = datetime.fromtimestamp(last_stock_info.get("timestamp", 0))
            if dt_object:
                last_stock_info["timestamp"] = dt_object.strftime("%Y-%m-%d %H:%M:%S")
            rsi_value = self.cal_rsi(df, self.element_dict["rsi"])
            mfi_value = self.cal_mfi(df, self.element_dict["mfi"])
            kdj_values = self.cal_kdj(df, self.element_dict["kdj"])
            logger.info(f"RSI: {rsi_value}")
            logger.info(f"MFI: {mfi_value}")
            logger.info(f"KDJ: {kdj_values}")
            return {
                "RSI": rsi_value,
                "MFI": mfi_value,
                "KDJ": kdj_values,
            }, last_stock_info
        else:
            logger.info(f"[{min_offset}/{min_period}] On tracking...")
            return None, None

    @staticmethod
    def decision_func(mfi, kdj):
        if mfi > 79 and kdj["K"] > 79 and kdj["D"] > 79:
            return "SELL!"
        elif mfi < 19 and kdj["K"] < 19 and kdj["D"] < 19:
            return "BUY!"
        else:
            return "HOLD!"


def main():
    stock_id = "UVXY.US"
    element_dict = {"rsi": 12, "mfi": 9, "kdj": 9}

    lp = LongPortSubscribe(symbol=stock_id, element_dict=element_dict)

    previous_decision = "HOLD!"
    while True:
        try:
            results, last_stock_info = lp.cal_indicators()
        except Exception as e:
            logger.error(f"Error occurred: {e}")
            results, last_stock_info = None, None
        if results:
            decision = lp.decision_func(mfi=results["MFI"], kdj=results["KDJ"])
            logger.info(f"Decision: {decision}")

            if decision != previous_decision:
                try:
                    notifier = TelegramNotifier(token, chat_id)
                    asyncio.run(
                        notifier.send_message(
                            stock_id, last_stock_info, results, decision
                        )
                    )
                    previous_decision = decision
                except Exception as e:
                    logger.error(f"Error sending message: {e}")
                # sleep(4)  # Sleep to avoid spamming
            else:
                logger.info(f"Decision remains the same: {decision}")

        sleep(1.05)


if __name__ == "__main__":
    main()
