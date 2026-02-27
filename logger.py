import logging
import os
from logging.handlers import RotatingFileHandler

class LogConfigurator:
    """
    Utility class to configure and initialize the application logging system.
    
    This class sets up a global logger with both console and rotating file handlers,
    ensuring consistent log formatting across the entire project.
    """

    @staticmethod
    def setup_logger(log_file: str = "logs/app.log", level: int = logging.INFO) -> logging.Logger:
        """
        Configure the root logger with standard formatting and handlers.

        Args:
            log_file (str): Path to the log file. Defaults to "logs/app.log".
            level (int): Logging level. Defaults to logging.INFO.

        Returns:
            logging.Logger: The configured root logger instance.
        """
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Define log format
        log_format = logging.Formatter(
            "[%(asctime)s.%(msecs)03d][%(threadName)s][%(levelname)s] %(message)s",
            datefmt="%Y.%m.%d-%H:%M:%S"
        )

        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Clear existing handlers if any
        if root_logger.hasHandlers():
            root_logger.handlers.clear()

        # Rotating file handler: 500MB per file, max 20 backup files
        file_handler = RotatingFileHandler(
            filename=log_file,
            mode="a",
            maxBytes=500 * 1024 * 1024,
            backupCount=20,
            encoding="utf-8"
        )
        file_handler.setFormatter(log_format)
        root_logger.addHandler(file_handler)

        # Console handler for terminal output
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        root_logger.addHandler(console_handler)

        import multiprocessing
        if multiprocessing.current_process().name == 'MainProcess':
            root_logger.info("Logging system initialized.")
            
        return root_logger

# Initialize logger instance at module level for global access
logger = LogConfigurator.setup_logger()
