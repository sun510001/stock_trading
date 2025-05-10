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
    suggestion="Hold"
    """

    stock_id: str
    current_info: Dict[str, float]
    indicators: Dict[str, Any]
    suggestion: str


# A global variable that defines a list of message types used for sending messages.
shared_message_list: List[Message] = []
list_lock = threading.Lock()

# A global variable that defines a stop event for thread termination.
stop_event = threading.Event()
