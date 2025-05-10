"""
Author: sun510001 sqf121@gmail.com
Date: 2025-05-09 18:07:58
LastEditors: sun510001 sqf121@gmail.com
LastEditTime: 2025-05-09 18:07:58
FilePath: /stock_trading/utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

import pytz
from datetime import datetime, time


def get_time_difference_from_ny(hour=9, minute=30):
    ny_tz = pytz.timezone("America/New_York")
    now_ny = datetime.now(ny_tz)
    four_am_ny = ny_tz.localize(datetime.combine(now_ny.date(), time(hour, minute)))
    time_difference = now_ny - four_am_ny
    return int(time_difference.total_seconds() / 60)
