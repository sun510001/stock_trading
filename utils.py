import pytz
from datetime import datetime, time


def get_time_difference_from_4am_ny():
    ny_tz = pytz.timezone('America/New_York')
    now_ny = datetime.now(ny_tz)
    four_am_ny = ny_tz.localize(datetime.combine(now_ny.date(), time(4, 0)))
    time_difference = now_ny - four_am_ny
    return int(time_difference.total_seconds() / 60)