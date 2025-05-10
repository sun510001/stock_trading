import pytz
from datetime import datetime, time


def get_time_difference_from_ny(hour=9, minute=30):
    ny_tz = pytz.timezone("America/New_York")
    now_ny = datetime.now(ny_tz)
    four_am_ny = ny_tz.localize(datetime.combine(now_ny.date(), time(hour, minute)))
    time_difference = now_ny - four_am_ny
    return int(time_difference.total_seconds() / 60)

def get_time_offset(datetime_obj: datetime):
    """
    Get the time offset in minutes from a specific timezone.

    :param datetime_obj: The datetime object to convert.
    :param timezone_str: The timezone string (e.g., 'America/New_York').
    :return: Time offset in minutes.
    """
    local_time = datetime.now()
    time_difference = local_time - datetime_obj
    return time_difference.total_seconds()