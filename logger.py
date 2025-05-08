'''
Author: sun510001 sqf121@gmail.com
Date: 2025-05-08 00:46:17
LastEditors: sun510001 sqf121@gmail.com
LastEditTime: 2025-05-08 20:17:14
FilePath: /home_process/stock_trading/logger.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
"""
Author: sun510001
Date: 2025-05-07 00:54:27
LastEditors: sun510001 sqf121@gmail.com
LastEditTime: 2025-05-07 20:51:39
FilePath: logger.py
Description:
Copyright 2025 OBKoro1, All Rights Reserved.
2025-05-07 00:54:27
"""

import logging


# 创建日志格式
log_format = logging.Formatter(
    "%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s", datefmt="%Y-%m-%d_%H-%M-%S"
)

# 创建根 logger，设置最低级别
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 文件处理器：记录到文件
file_handler = logging.FileHandler("stock_trading/logs/app.log", mode="a")
file_handler.setFormatter(log_format)
logger.addHandler(file_handler)

# 终端处理器：输出到控制台
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_format)
logger.addHandler(console_handler)

# 示例
logger.info("Logger started.")
