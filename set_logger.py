
#!/usr/bin/env python3
# Author: Enric Basso

import logging
from Color import Colors


class _PrettierRecord(logging.Filter):
    """
    This is a filter to choose whether to process a record sent to
    the logger.

    This is not doing that though, it is a trick:
    Since this is executed whenever a record is sent to the logger,
    it used as a chance to append keywords in the dictionary of
    the record, that can then be accessed by the handler.

        - Avoids having to give them as arguments for every call
        - filter is always true so does not affect record selection

    In this case this is adding colors, that will be used to make
    a specific choice according to the severity level of the record.
    """
    color_levels ={"DEBUG": Color.cyan,
                   "INFO": Color.green,
                   "WARNING": Color.yellow,
                   "ERROR": Color.red,
                   "CRITICAL": Color.magenta
                   }

    def filter(self, record: logging.LogRecord) -> bool:
        record.color = self.color_levels[record.levelname]
        record.module_name = record.pathname.split("/")[-1]
        record.reset = Color.reset
        return True


class PrettyScreenHandler(logging.StreamHandler):
    """
    This Handler will send messages to the screen, properly colored
    according to severity level and taking up a lot of space. Includes
    delimiters.

    The message will be of the shape
    --------------------------------------------------
    {name} in function {funcName} in line {lineno}: {levelname}:
     {msg}
    --------------------------------------------------

    """
    def __init__(self):
        super().__init__()
        self.addFilter(_PrettierRecord())
        _fmt = Formatters.pretty_formatter()
        self.setFormatter(_fmt)
        self.setLevel(logging.NOTSET)

class SimpleScreenHandler(logging.StreamHandler):
    """
    This Handler will send messages to the screen, properly colored
    according to severity level. This is enough for most applications,
    or for top level logging.

    The message will be of the shape
    {levelname}: {msg}

    """
    def __init__(self):
        super().__init__()
        self.addFilter(_PrettierRecord())
        _fmt = Formatters.colored_simple_formatter()
        self.setFormatter(_fmt)
        self.setLevel(logging.NOTSET)

class DetailedScreenHandler(logging.StreamHandler):
    """
    This Handler will send messages to the screen, properly colored
    according to severity level. It adds some contextual information.
    This is enough for most applications, or for top level logging.
    This can be considered a good standard.

    The message will be of the shape
    {levelname}:{module_name}:{funcName} in line {lineno}: {msg}
    """
    def __init__(self):
        super().__init__()
        self.addFilter(_PrettierRecord())
        _fmt = Formatters.colored_detailed_formatter()
        self.setFormatter(_fmt)
        self.setLevel(logging.NOTSET)

class DetailedFileHandler(logging.FileHandler):
    """
    This Handler will send messages to the screen, properly colored
    according to severity level. It adds some contextual information.
    This is enough for most applications, or for top level logging.
    This can be considered a good standard.

    The message will be of the shape
    {levelname}:{module_name}:{funcName} in line {lineno}: {msg}
    """
    def __init__(self, filename, mode='w', encoding=None, delay=False, errors=None):
        super().__init__(filename=filename,
                         mode=mode,
                         encoding=encoding,
                         delay=delay, #errors=errors   # errors is only available on python 3.9+
                         )
        self.addFilter(_PrettierRecord())
        _fmt = Formatters.detailed_formatter()
        self.setFormatter(_fmt)
        self.setLevel(logging.NOTSET)

class ColoredPrinterScreenHandler(logging.StreamHandler):
    """
    This Handler will send messages to the screen, properly colored
    according to severity level. It adds NO contextual information.
    The message will be of the shape
    {msg}

    Suggestion: If you give this handler to a logger and you forbid
    it to propagate to the root logger, you can use it a simple color printing
    class, while retaining flexibility to add more handlers to also send the
    messages elsewhere (i.e. a file).
    """
    def __init__(self):
        super().__init__()
        self.addFilter(_PrettierRecord())
        _fmt = Formatters.colored_print_formatter()
        self.setFormatter(_fmt)
        self.setLevel(logging.NOTSET)

class PrinterScreenHandler(logging.StreamHandler):
    """
    This Handler will send messages to the screen, properly colored
    according to severity level. It adds NO contextual information.
    The message will be of the shape
    {msg}

    Suggestion: If you give this handler to a logger and you forbid
    it to propagate to the root logger, you can use it a simple color printing
    class, while retaining flexibility to add more handlers to also send the
    messages elsewhere (i.e. a file).
    """
    def __init__(self):
        super().__init__()
        self.addFilter(_PrettierRecord())
        _fmt = Formatters.print_formatter()
        self.setFormatter(_fmt)
        self.setLevel(logging.NOTSET)

class Formatters():
    """
    This is a class just meant to act as a container for functions that return
    formatters for logger Handlers. It is not meant to be instatiated, and it
    does not have any functionality other than enclosing the formatters
    """

    @staticmethod
    def colored_simple_formatter() -> logging.Formatter:
        """
        Method to get a formatter to give to a logger instance.
        Minimal information.
        Coloring according to severity level.

        parameters:
            None

        Returns:
            logging.Formatter instance
        """
        _fmt = logging.Formatter(("{color}"
                                  "{levelname}: "
                                  "{reset}{msg}\n"
                                  ), style="{"
                             )
        return _fmt

    @staticmethod
    def colored_detailed_formatter() -> logging.Formatter:
        """
        Method to get a formatter to give to a logger instance.
        Detailed information.
        Coloring according to severity level.

        parameters:
            None

        Returns:
            logging.Formatter instance
        """
        _fmt = logging.Formatter(("{color}{levelname} from {name}: "
                                  "{module_name}:{funcName} in line {lineno}: "
                                  "{reset}\n{msg}\n"
                                  ), style="{"
                             )
        return _fmt

    @staticmethod
    def pretty_formatter() -> logging.Formatter:
        """
        Method to get a formatter to give to a logger instance.
        Detailed information.
        Coloring according to severity level.
        Uses messages delimiters and newlines to take more space for
        visual effect.

        parameters:
            None

        Returns:
            logging.Formatter instance
        """
        _divisor = "-" * 60
        _fmt = logging.Formatter(("{color}" + _divisor + "\n"
                              "{levelname} from {name} logger in function \"{funcName}\""
                              " line {lineno}:\n"
                              "{reset}\n{msg}\n"
                              "{color}" + _divisor + "{reset}\n"
                              ), style="{"
                             )
        return _fmt

    @staticmethod
    def detailed_formatter() -> logging.Formatter:
        """
        Method to get a formatter to give to a logger instance.
        Detailed information. No coloring according to severity level.

        parameters:
            None

        Returns:
            logging.Formatter instance
        """
        _divisor = "-" * 60
        _fmt = logging.Formatter(("{levelname}: "
                                  "{module_name}: {funcName} in line {lineno}: "
                                  "{msg}\n"
                                  ), style="{"
                             )
        return _fmt

    @staticmethod
    def print_formatter() -> logging.Formatter:
        """
        Method to get a formatter to give to a logger instance.
        Just prints the message without information.
        This is useful to print on screen, but through the logger instance.
        That way the message can also go elsewhere with other handlers,
        if so desired.

        parameters:
            None

        Returns:
            logging.Formatter instance
        """
        _fmt = logging.Formatter("{msg}", style="{")
        return _fmt

    @staticmethod
    def colored_print_formatter() -> logging.Formatter:
        """
        Method to get a formatter to give to a logger instance.
        Just prints the message without information.
        This is useful to print on screen, but through the logger instance.
        That way the message can also go elsewhere with other handlers,
        if so desired.
        It will color according to the severity level given at the call.

        parameters:
            None

        Returns:
            logging.Formatter instance
        """
        _fmt = logging.Formatter("{color}{msg}{reset}", style="{")
        return _fmt

