'''
Author: sun510001
Date: 2025-05-06 23:39:42
LastEditors: sun510001 sqf121@gmail.com
LastEditTime: 2025-05-07 19:58:32
FilePath: lp_get_info.py
Description: 
Copyright 2025 OBKoro1, All Rights Reserved. 
2025-05-06 23:39:42
'''

import pandas as pd
import pandas_ta as ta
from time import sleep
from typing import List
from logger import logger
from longport.openapi import Config, QuoteContext, Period, AdjustType
from utils import get_time_difference_from_4am_ny


class LongPortSubscribe():
    def __init__(self, symbol: str, rsi_length_list: List[int]):
        self.symbol = symbol
        self.rsi_length_list = rsi_length_list
        self.config = Config.from_env()  # Load configuration from environment variables
        self.quote_ctx = QuoteContext(self.config)

    def get_info(self, window_size=25):
        # TODO: close不太对吧?
        candlesticks = self.quote_ctx.candlesticks(self.symbol, Period.Min_1, window_size, AdjustType.NoAdjust)
        close_prices = [k.close for k in candlesticks]
        return close_prices

    @staticmethod
    def cal_rsi(close_prices, periods=[6,12,24]):
        df = pd.DataFrame({'close': close_prices})
        rsi_dict = {}
        for index, length in enumerate(periods):
            df[f'rsi{index+1}'] = ta.rsi(df['close'], length=length)
            latest_rsi = df[f'rsi{index+1}'].iloc[-1]
            rsi_dict[length] = f"{latest_rsi:.3f}"
        logger.debug(df)
        return rsi_dict

    def __call__(self):
        min_offset = get_time_difference_from_4am_ny()
        max_period = max(self.rsi_length_list) * 3
        min_period = min(self.rsi_length_list) * 2
        # max_period = 100
        if min_period <= min_offset <= 720:
            close_prices = self.get_info(window_size=max_period)
            rsi_dict = self.cal_rsi(close_prices, self.rsi_length_list)
            logger.info(f"RSI: {rsi_dict}")
        elif 0 <= min_offset < min_period:
            logger.info(f"[{min_offset}/{min_period}] On tracking...")


def main():
    stock_id = "UVXY.US"
    rsi_length_list = [6,12,24]
    lp = LongPortSubscribe(symbol=stock_id, rsi_length_list=rsi_length_list)
    while True:
        lp()
        sleep(3)


if __name__ == "__main__":
    main()
