import asyncio
import threading
import pandas as pd
from time import sleep
from typing import Dict
from logger import logger
from longport.openapi import (
    Config,
    QuoteContext,
    Period,
    AdjustType,
)
from utils.tools import get_time_difference_from_ny, get_time_offset
from datetime import datetime

# from send_email import GmailStockNotifier
from utils.send_telegram_bot import TelegramNotifier
from tokens.telegram import token, chat_id
from utils.stock_indicators import StockIndicators

# from utils.fixed_queue_df import FixedQueueDF
from utils.global_param import shared_message_list, list_lock, Message, stop_event


class LongPortSubscribe:
    def __init__(
        self,
        symbol: str,
        element_dict: Dict[str, int],
        sell_threshold: int,
        buy_threshold: int,
    ):
        """
        Initialize the LongPortSubscribe class.
        :param symbol: The stock symbol to subscribe to.
        :param element_dict: A dictionary containing the parameters for RSI, MFI, and KDJ.
        :param sell_threshold: The threshold for selling.
        :param buy_threshold: The threshold for buying.
        """

        self.symbol = symbol
        self.element_dict = element_dict
        self.sell_threshold = sell_threshold
        self.buy_threshold = buy_threshold

        # Load configuration from environment variables
        self.config = Config.from_env()  # Load configuration from environment variables
        self.quote_ctx = QuoteContext(self.config)

        max_key = max(self.element_dict.values())
        self.max_period = max_key * 3
        self.min_period = max_key + 1

        self.stock_indicators = StockIndicators()

    def get_info(self, window_size=25):
        """
        Get stock information for the given symbol.
        :param window_size: The number of candlesticks to retrieve.
        :return: A DataFrame containing the stock information.
        """
        # Get candlestick data for the specified symbol and period
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

    def cal_indicators(self):
        min_offset = get_time_difference_from_ny()

        if self.min_period <= min_offset <= 390:
            # if True:
            df = self.get_info(window_size=self.max_period)
            last_stock_info = df.iloc[-1].to_dict()
            dt_object = datetime.fromtimestamp(last_stock_info.get("timestamp", 0))
            offset_time = get_time_offset(dt_object)
            if offset_time > 10:
                logger.info(f"Time offset: {offset_time} seconds. Outside of trading hours. skipped.")
                return None, None
            
            if dt_object:
                last_stock_info["timestamp"] = dt_object.strftime("%Y-%m-%d %H:%M:%S")
            rsi_value = self.stock_indicators.cal_rsi(df, self.element_dict["rsi"])
            mfi_value = self.stock_indicators.cal_mfi(df, self.element_dict["mfi"])
            kdj_values = self.stock_indicators.cal_kdj(df, self.element_dict["kdj"])
            logger.info(f"RSI: {rsi_value}")
            logger.info(f"MFI: {mfi_value}")
            logger.info(f"KDJ: {kdj_values}")
            return {
                "RSI": rsi_value,
                "MFI": mfi_value,
                "KDJ": kdj_values,
            }, last_stock_info
        else:
            logger.info(f"[{min_offset}/{self.min_period}] On tracking...")
            return None, None

    def decision_func(self, mfi, kdj):
        if (
            mfi > self.sell_threshold
            and kdj["K"] > self.sell_threshold
            and kdj["D"] > self.sell_threshold
        ):
            return 2  # SELL!
        elif (
            mfi < self.buy_threshold
            and kdj["K"] < self.buy_threshold
            and kdj["D"] < self.buy_threshold
        ):
            return 1  # BUY!
        else:
            return 0  # HOLD!


def pull_quote_thread(
    symbol: str,
    element_dict: Dict[str, int],
    decision_dict: Dict[int, str],
    stock_id: str,
    sell_threshold: int,
    buy_threshold: int,
):
    global shared_message_list
    # Initialize the LongPortSubscribe class
    lp = LongPortSubscribe(
        symbol=symbol,
        element_dict=element_dict,
        sell_threshold=sell_threshold,
        buy_threshold=buy_threshold,
    )

    previous_decision = 0
    while not stop_event.is_set():
        try:
            results, last_stock_info = lp.cal_indicators()
        except Exception as e:
            logger.error(f"Error occurred: {e}")
            results, last_stock_info = None, None

        if results:
            decision = lp.decision_func(mfi=results["MFI"], kdj=results["KDJ"])
            logger.info(f"Decision: {decision_dict[decision]}")

            if decision != previous_decision:
                with list_lock:
                    shared_message_list.append(
                        Message(
                            stock_id=stock_id,
                            current_info=last_stock_info,
                            indicators=results,
                            suggestion=decision_dict[decision],
                        )
                    )
                previous_decision = decision
            else:
                logger.info(f"Decision remains the same: {decision_dict[decision]}")
        sleep(3)


async def async_task(notifier: TelegramNotifier):
    global shared_message_list
    while not stop_event.is_set():
        with list_lock:
            if shared_message_list:
                message = shared_message_list.pop(0)
            else:
                message = None
        if message:
            # Send the message using the notifier
            logger.info(f"Sending message: {message}")
            try:
                await notifier.send_message(message.stock_id, message.current_info, message.indicators, message.suggestion)
            except Exception as e:
                logger.error(f"Error sending message: {e}")
        await asyncio.sleep(0.1)


def send_message_thread(notifier: TelegramNotifier):
    asyncio.run(async_task(notifier))


def main():
    symbols = ["TQQQ.US", "SQQQ.US"]
    element_dict = {"rsi": 12, "mfi": 9, "kdj": 9}
    decision_dict = {0: "HOLD!", 1: "BUY!", 2: "SELL!"}
    sell_threshold = 79
    buy_threshold = 19
    notifier = TelegramNotifier(token, chat_id)

    send_message_t = threading.Thread(target=send_message_thread, args=(notifier,))
    send_message_t.start()

    symbol_threads = []
    for symbol in symbols:
        t = threading.Thread(
            target=pull_quote_thread,
            args=(
                symbol,
                element_dict,
                decision_dict,
                symbol,
                sell_threshold,
                buy_threshold,
            ),
        )
        t.start()
        symbol_threads.append(t)

    try:
        while True:
            sleep(1)
    except KeyboardInterrupt:
        logger.info("Received exit signal, shutting down...")
        stop_event.set()
        sleep(1)
        for t in symbol_threads:
            t.join()
        send_message_t.join()
        logger.info("All threads have finished.")


if __name__ == "__main__":
    main()
