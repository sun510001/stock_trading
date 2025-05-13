'''
Author: sun510001 sqf121@gmail.com
Date: 2025-05-14 00:26:20
LastEditors: sun510001 sqf121@gmail.com
LastEditTime: 2025-05-14 00:27:44
FilePath: /stock_trading/utils/global_param.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import threading
from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class Message:
    """
    stock_id="TSLA",
    current_info={"price": 650.55},
    indicators={
        "RSI": 47.2,
        "MFI": 52.6,
        "KDJ": {"K": 60.0, "D": 55.0, "J": 70.0}
    },
    suggestions={"mfi+kdj": "Hold", "adx": "Hold"}
    """

    stock_id: str
    current_info: Dict[str, float]
    indicators: Dict[str, Any]
    suggestions: Dict[str, str]


# A global variable that defines a list of message types used for sending messages.
shared_message_list: List[Message] = []
list_lock = threading.Lock()

# A global variable that defines a stop event for thread termination.
stop_event = threading.Event()
