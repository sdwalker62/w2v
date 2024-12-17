"""
A basic logger utility that contains a log factory.
"""

import logging
import shutil
import sys
from datetime import datetime
from typing import Optional


class CustomFormatter(logging.Formatter):
    """Custom formatter that aligns timestamps to the right and adds colors for warnings and above"""

    # ANSI escape sequences for colors and formatting
    COLORS = {
        logging.WARNING: "\033[93m",  # Yellow
        logging.ERROR: "\033[91m",  # Red
        logging.CRITICAL: "\033[95m",  # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def __init__(self):
        super().__init__()
        # Get terminal width for right alignment
        self.term_width = shutil.get_terminal_size().columns

    def format(self, record: logging.LogRecord) -> str:
        # Create the left part of the message with bold module and function names
        module_func = f"[{self.BOLD}{record.module}.{record.funcName}{self.RESET}]"
        left_part = f"{module_func} {record.msg}"

        # Create the timestamp part with HH:mm:ss format
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        timestamp_part = f"at {timestamp}"

        # Calculate padding needed for right alignment
        # Need to account for ANSI escape sequences not taking up visual space
        visual_length = (
            len(f"[{record.module} > {record.funcName}]") + 1 + len(record.msg)
        )  # +1 for the space
        padding = self.term_width - visual_length - len(timestamp_part)
        padding = max(1, padding)  # Ensure at least one space

        # Add color if level is warning or above
        if record.levelno in self.COLORS:
            colored_msg = (
                f"{module_func} {self.COLORS[record.levelno]}{record.msg}{self.RESET}"
            )
            return f"{colored_msg}{' ' * padding}{timestamp_part}"

        return f"{left_part}{' ' * padding}{timestamp_part}"


def logger_factory(name: Optional[str] = None) -> logging.Logger:
    """Set up and return a logger with the custom formatter"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CustomFormatter())
    logger.addHandler(handler)

    return logger
