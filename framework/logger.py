"""Logging configuration and utilities for the application."""
import datetime
import json
import logging
import os
import sys
from functools import lru_cache, partial
from logging import Logger
from logging.config import dictConfig
from os import path
from types import MethodType
from typing import Any, Optional, cast

from framework.kernel_gen_config import config

# Environment variable configuration
APP_CONFIGURE_LOGGING = os.environ.get("APP_CONFIGURE_LOGGING", "1") == "1"
APP_LOGGING_CONFIG_PATH = os.environ.get("APP_LOGGING_CONFIG_PATH", "")
APP_LOGGING_LEVEL = os.environ.get("APP_LOGGING_LEVEL", "INFO")
APP_LOGGING_PREFIX = os.environ.get("APP_LOGGING_PREFIX", "[AKB] ")

# Logging format configuration
_FORMAT = "[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"

# Default logging configuration
DEFAULT_LOGGING_CONFIG = {
    "formatters": {
        "app": {
            "class": "logging.Formatter",
            "datefmt": _DATE_FORMAT,
            "format": _FORMAT,
        },
    },
    "handlers": {
        "app": {
            "class": "logging.StreamHandler",
            "formatter": "app",
            "level": APP_LOGGING_LEVEL,
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "app": {
            "handlers": ["app"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
    "version": 1,
    "disable_existing_loggers": False
}


class NewLineFormatter(logging.Formatter):
    """Formatter that supports multi-line log messages with proper indentation."""

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        # Maintain consistent indentation for multi-line messages
        if "\n" in message:
            parts = message.split("\n")
            subsequent_indent = " " * (len(parts[0]) - len(parts[0].lstrip()))
            return "\n".join([parts[0]] + [subsequent_indent + p for p in parts[1:]])
        return message


@lru_cache
def _print_info_once(logger: Logger, msg: str) -> None:
    """Print info message once (cached to avoid duplicates)."""
    logger.info(msg, stacklevel=2)


@lru_cache
def _print_warning_once(logger: Logger, msg: str) -> None:
    """Print warning message once (cached to avoid duplicates)."""
    logger.warning(msg, stacklevel=2)


@lru_cache
def _print_error_once(logger: Logger, msg: str) -> None:
    """Print error message once (cached to avoid duplicates)."""
    logger.error(msg, stacklevel=2)


class _AppLogger(Logger):
    """
    Extended logger class with additional utility methods.

    Note:
        This class is only used for type information.
        We actually patch methods directly on logging.Logger instances
        to avoid conflicts with other libraries.
    """

    def info_once(self, msg: str) -> None:
        """Log info message once - subsequent calls with same message are ignored."""
        _print_info_once(self, msg)

    def warning_once(self, msg: str) -> None:
        """Log warning message once - subsequent calls with same message are ignored."""
        _print_warning_once(self, msg)

    def error_once(self, msg: str) -> None:
        """Log error message once - subsequent calls with same message are ignored."""
        _print_error_once(self, msg)


def _configure_app_root_logger() -> None:
    """Configure the root application logger."""
    logging_config = dict[str, Any]()

    if not APP_CONFIGURE_LOGGING and APP_LOGGING_CONFIG_PATH:
        raise RuntimeError(
            "APP_CONFIGURE_LOGGING is disabled but APP_LOGGING_CONFIG_PATH is provided. "
            "Either enable APP_CONFIGURE_LOGGING or remove APP_LOGGING_CONFIG_PATH."
        )

    if APP_CONFIGURE_LOGGING:
        logging_config = DEFAULT_LOGGING_CONFIG

    if APP_LOGGING_CONFIG_PATH:
        if not path.exists(APP_LOGGING_CONFIG_PATH):
            raise RuntimeError(f"Logging config file not found: {APP_LOGGING_CONFIG_PATH}")

        with open(APP_LOGGING_CONFIG_PATH, encoding="utf-8") as file:
            custom_config = json.loads(file.read())

        if not isinstance(custom_config, dict):
            raise ValueError(f"Invalid logging config. Expected dict, got {type(custom_config).__name__}")
        logging_config = custom_config

    # Update NewLineFormatter class paths
    for formatter in logging_config.get("formatters", {}).values():
        if formatter.get("class", "").endswith("NewLineFormatter"):
            formatter["class"] = "NewLineFormatter"

    if logging_config:
        dictConfig(logging_config)


def init_logger(name: str) -> _AppLogger:
    """
    Initialize and get logger with enhanced functionality.

    This function ensures the root app logger is properly configured
    and adds utility methods to the logger instance.

    Args:
        name: Logger name, typically the module name

    Returns:
        Configured logger instance with additional methods
    """
    # Ensure proper logger hierarchy
    if name.startswith("app."):
        logger = logging.getLogger(name)
    else:
        logger = logging.getLogger(f"app.{name}")

    # Add enhanced methods to logger
    methods_to_patch = {
        "info_once": _print_info_once,
        "warning_once": _print_warning_once,
        "error_once": _print_error_once,
    }

    for method_name, method in methods_to_patch.items():
        setattr(logger, method_name, MethodType(partial(method, logger), logger))

    return cast(_AppLogger, logger)


def add_file_handler(log_file_path: str, level: str = "INFO", logger=None):
    """Add file handler to logger for persistent logging."""
    if logger is None:
        logger = logging.getLogger("app")

    # Skip if handlers already exist
    if logger.handlers:
        return logger.handlers

    file_handler = logging.FileHandler(log_file_path, mode='w', encoding=config.encoding)
    file_handler.setLevel(getattr(logging, level))
    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return file_handler


def switch_handler(log_file_path, old_handler, level: str = "INFO"):
    """Switch logging handler to a new file."""
    root_logger = logging.getLogger("app")
    root_logger.removeHandler(old_handler)
    return add_file_handler(log_file_path, level)


def _trace_calls(log_path, root_dir, frame, event, arg=None):
    """Function call tracer for debugging purposes."""
    if event in ['call', 'return']:
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        func_name = frame.f_code.co_name

        # Only trace functions within root_dir
        if not filename.startswith(root_dir):
            return

        try:
            last_frame = frame.f_back
            if last_frame is not None:
                last_filename = last_frame.f_code.co_filename
                last_lineno = last_frame.f_lineno
                last_func_name = last_frame.f_code.co_name
            else:
                last_filename = ""
                last_lineno = 0
                last_func_name = ""

            with open(log_path, 'a', encoding=config.encoding) as f:
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                if event == 'call':
                    f.write(f"{ts} Call to {func_name} in {filename}:{lineno} "
                            f"from {last_func_name} in {last_filename}:{last_lineno}\n")
                else:
                    f.write(f"{ts} Return from {func_name} in {filename}:{lineno} "
                            f"to {last_func_name} in {last_filename}:{last_lineno}\n")
        except NameError:
            # Module deleted during shutdown
            pass

    return partial(_trace_calls, log_path, root_dir)


def enable_trace_function_call(log_file_path: str, root_dir: Optional[str] = None):
    """
    Enable tracing of every function call within root_dir.
    Useful for debugging hangs or crashes.

    Args:
        log_file_path: Path to write trace log
        root_dir: Root directory to trace (defaults to app root)

    Note:
        This is thread-level tracing. Only the calling thread will be traced.
        This significantly slows down execution and should only be used for debugging.
    """
    logger = init_logger(__name__)
    logger.warning(
        "Function call tracing enabled. This will log every Python function "
        "execution and significantly slow down performance. Use only for debugging."
    )
    logger.info("Trace log saved to %s", log_file_path)

    if root_dir is None:
        root_dir = os.path.dirname(os.path.dirname(__file__))

    sys.settrace(partial(_trace_calls, log_file_path, root_dir))


# Initialize root logger on module import
# This is thread-safe as modules are imported once, guaranteed by Python GIL
_configure_app_root_logger()

# Common logger for this module
common_logger = init_logger(__name__)