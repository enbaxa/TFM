#!/usr/bin/env python3
# Author: Enric Basso
"""
This module, `set_logger.py`, is responsible for setting up the logging
configuration for the application.

It provides functions to initialize a logger with a specific name,
set the logging level, and specify the format of the log messages.

This allows for consistent logging practices across the application,
making it easier to debug issues and track application flow.
"""
import logging
import json
from datetime import datetime as dt
from color import Color


class _PrettierRecord(logging.Filter):
    """
    A logging filter that dynamically adds color and module name attributes to log
    records based on their severity level. This approach avoids the need for specifying
    these attributes in each log message manually. The filter always returns
    True to ensure it does not interfere with the log records' processing.

    Attributes:
        color_levels (dict): A mapping of log levels to color codes.
    """
    color_levels = {
        "DEBUG": Color.cyan,
        "INFO": Color.green,
        "WARNING": Color.yellow,
        "ERROR": Color.red,
        "CRITICAL": Color.magenta
    }

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Adds color and module name attributes to the log record.

        Parameters:
            record (logging.LogRecord): The log record to process.

        Returns:
            bool: Always returns True to ensure the record is not filtered out.
        """
        record.color = self.color_levels.get(record.levelname, '')
        record.reset = Color.reset
        return True


class Filter_name(logging.Filter):
    """
    A logging filter that will determine if a log record should be processed
    based on which logger emitted the record. This filter is used to separate
    log records based on the logger name, allowing for different handling
    of log records based on their source. Useful to attach it to a handler

    Attributes:
        name (str): The name of the logger to filter by. If the logger name
            matches this value, the log record will be processed.
    """
    def __init__(self, allowed: str | list = None):
        """
        Initializes the filter with the specified logger name.

        If no name is provided, the filter will block all log records.

        Parameters:
            name (str or list): The names of the loggers to filter by.
        """
        super().__init__()
        self.allowed = allowed
        if self.allowed is not None and not isinstance(self.allowed, list):
            self.allowed = [self.allowed]

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Determines if the log record should be processed based on the logger name.

        Parameters:
            record (logging.LogRecord): The log record to process.

        Returns:
            bool: True if the log record should be processed, False otherwise.
        """
        if self.allowed is None:
            return False
        return record.name in self.allowed


class PrettyScreenHandler(logging.StreamHandler):
    """
    A logging handler that formats log messages for display on the screen
    with colors and delimiters for enhanced readability. It uses a custom
      format that includes the log level, logger name, function name,
      line number, and the log message itself.

    Inherits from logging.StreamHandler.
    """
    def __init__(self):
        super().__init__()
        self.addFilter(_PrettierRecord())
        self.setFormatter(Formatters.pretty_formatter())
        self.setLevel(logging.NOTSET)


class SimpleScreenHandler(logging.StreamHandler):
    """
    A logging handler designed for simple, uncolored output to the screen.
    Suitable for most applications where minimal log information is sufficient.
    Formats log messages to include only the log level and the message itself.

    Inherits from logging.StreamHandler.
    """
    def __init__(self):
        super().__init__()
        self.addFilter(_PrettierRecord())
        self.setFormatter(Formatters.simple_formatter())
        self.setLevel(logging.NOTSET)


class DetailedScreenHandler(logging.StreamHandler):
    """
    A logging handler that outputs detailed log messages to the screen without
    color coding. It includes information such as severity level, module name,
    function name, and line number, providing a comprehensive context for each log entry.
    This handler is ideal for environments where color coding is not supported or required.

    Inherits from logging.StreamHandler.
    """
    def __init__(self):
        super().__init__()
        self.addFilter(_PrettierRecord())
        self.setFormatter(Formatters.detailed_formatter())
        self.setLevel(logging.NOTSET)


class ColoredDetailedScreenHandler(logging.StreamHandler):
    """
    A logging handler designed for outputting detailed log messages to the screen,
    including severity level, module name, function name, and line number,
    all with appropriate color coding for enhanced readability.
    This handler is suitable for applications requiring detailed context in logs,
    particularly useful for debugging and monitoring.

    Inherits from logging.StreamHandler.
    """
    def __init__(self):
        super().__init__()
        self.addFilter(_PrettierRecord())
        self.setFormatter(Formatters.colored_detailed_formatter())
        self.setLevel(logging.NOTSET)


class DetailedFileHandler(logging.FileHandler):
    """
    A file-based logging handler that records detailed log messages,
    including severity level, module name, function name, and line number.
    This handler is tailored for persistent logging to a file, where detailed
    contextual information is essential for post-mortem analysis
    and archiving purposes.

    Inherits from logging.FileHandler.
    """
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        super().__init__(filename, mode, encoding, delay)
        self.addFilter(_PrettierRecord())
        self.setFormatter(Formatters.detailed_formatter())
        self.setLevel(logging.NOTSET)


class PrinterScreenHandler(logging.StreamHandler):
    """
    A logging handler that outputs log messages as plain text to the screen,
    without any color coding or additional contextual information.
    Designed for the most straightforward logging needs, this handler is useful
    for applications where logs are intended to be minimal and unobtrusive.

    Inherits from logging.StreamHandler.
    """
    def __init__(self):
        super().__init__()
        self.addFilter(_PrettierRecord())
        self.setFormatter(Formatters.print_formatter())
        self.setLevel(logging.NOTSET)


class ColoredPrinterScreenHandler(logging.StreamHandler):
    """
    A logging handler for outputting log messages as plain text with severity
    level-based color coding to the screen.
    This handler is optimized for simplicity and visual differentiation
    of log messages, making it suitable for quick debugging tasks
    where minimal contextual information is required.

    Inherits from logging.StreamHandler.
    """
    def __init__(self):
        super().__init__()
        self.addFilter(_PrettierRecord())
        self.setFormatter(Formatters.colored_print_formatter())
        self.setLevel(logging.NOTSET)


class PersistentLogHandler(logging.FileHandler):
    """
    A file-based logging handler that records log messages as plain text.
    This handler is suitable for persistent logging to a file, where
    all data in the log message can be stored for future reference.
    The log messages are formatted as json objects for easy parsing.

    Inherits from logging.FileHandler.
    """
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        super().__init__(filename, mode, encoding, delay)
        self.setFormatter(Formatters.json_formatter())
        self.setLevel(logging.NOTSET)


class PersistentLogHandlerStructured(logging.FileHandler):
    """
    A file-based logging handler that records log messages as plain text.
    This handler is suitable for persistent logging to a file, where
    all data in the log message can be stored for future reference.
    The log messages are formatted as json objects for easy parsing.

    Inherits from logging.FileHandler.
    """
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        super().__init__(filename, mode, encoding, delay)
        self.setFormatter(Formatters.json_formatter(structured=True))
        self.setLevel(logging.NOTSET)


class JsonFormatter(logging.Formatter):
    """
    Creates a formatter for logging messages as json

    Returns:
        logging.Formatter: An instance configured for json log messages.
    """

    def __init__(self, structured: bool = False):
        super().__init__()
        self.indent = 2 if structured else None

    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            'message': record.getMessage(),
            'level': record.levelname,
            'module': record.filename,
            'function': record.funcName,
            'line': record.lineno,
            'time': dt.fromtimestamp(record.created).isoformat()
        }
        return json.dumps(log_record, indent=self.indent)


class Formatters:
    """
    Provides static methods for creating various logging.Formatter objects.
    This class acts as a namespace and is not intended to be instantiated.
    It encapsulates formatter definitions for ease of use and consistency
    across different logging handlers.
    """

    @staticmethod
    def simple_formatter() -> logging.Formatter:
        """
        Creates a formatter for simple logging.
        Formats log messages to include the log level and the message.

        Returns:
            logging.Formatter: An instance configured for simple log messages.
        """
        return logging.Formatter("{levelname}: {message}\n", style="{")

    @staticmethod
    def colored_simple_formatter() -> logging.Formatter:
        """
        Creates a formatter for logging with minimal information,
        adding color coding according to the severity level.
        Suitable for concise and visually distinguishable log messages.

        Returns:
            logging.Formatter: An instance configured for simple, colored log messages.
        """
        return logging.Formatter("{color}{levelname}: {reset}{message}\n", style="{")

    @staticmethod
    def detailed_formatter() -> logging.Formatter:
        """
        Creates a formatter for logging detailed messages without color coding.
        Includes log level, module name, function name, and line number,
        providing a comprehensive context for each log message.

        Returns:
            logging.Formatter: An instance configured for detailed log messages.
        """
        return logging.Formatter(
            "{levelname}: {filename}:"
            " in line {lineno}:\n {message}\n", style="{"
            )

    @staticmethod
    def colored_detailed_formatter() -> logging.Formatter:
        """
        Generates a formatter that provides detailed logging information,
        including log level, logger name, module name, function name,
        and line number, all colored based on the severity level.
        Ideal for in-depth debugging where context is crucial.

        Returns:
            logging.Formatter: An instance configured for detailed, colored log messages.
        """
        return logging.Formatter(
            "{color}{levelname} from {name}: "
            "{filename}: in line {lineno}: {reset}\n{message}\n",
            style="{"
            )

    @staticmethod
    def pretty_formatter() -> logging.Formatter:
        """
        Produces a formatter that enhances log readability by using
        delimiters and newlines, along with color coding by severity level.
        Designed to make critical log messages stand out visually in busy logs.

        Returns:
            logging.Formatter: An instance configured for visually enhanced log messages.
        """
        divisor = "-" * 60
        return logging.Formatter(
            "{color}"
            f"{divisor}"
            "\n{levelname} from {name} "
            "logger in function \"{funcName}\" line {lineno}:\n"
            "{reset}\n{message}\n{color}"
            f"{divisor}"
            "{reset}\n",
            style="{"
            )

    @staticmethod
    def print_formatter() -> logging.Formatter:
        """
        Generates a formatter for logging messages as plain text.
        This formatter is intended for simple output scenarios where
        only the message content is of interest, without any
        additional log metadata.

        Returns:
            logging.Formatter: An instance configured for plain text log messages.
        """
        return logging.Formatter("{message}", style="{")

    @staticmethod
    def colored_print_formatter() -> logging.Formatter:
        """
        Creates a formatter for logging messages as plain text with color
        coding according to the severity level.
        It simplifies output while retaining visual distinction of log
        severity for quick scanning.

        Returns:
            logging.Formatter: An instance configured for plain text, colored log messages.
        """
        return logging.Formatter("{color}{message}{reset}", style="{")

    @staticmethod
    def json_formatter(structured=False) -> logging.Formatter:
        """
        Creates a formatter for logging messages as json objects.
        This formatter is suitable for persistent logging to a file,
        where structured data is required for further processing.

        Returns:
            logging.Formatter: An instance configured for json log messages.

        """
        return JsonFormatter(structured=structured)
