import logging
import logging.handlers
import sys

class CustomFormatter(logging.Formatter):
    """Custom logging format"""
    grey = "\x1b[38;21m"
    green = "\x1b[32;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "[%(levelname)s] %(asctime)s - %(name)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(CustomFormatter())
    return console_handler

def get_file_handler(log_file):
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    return file_handler

def get_logger(logger_name, log_file=None, log_level=logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    logger.addHandler(get_console_handler())

    if log_file:
        logger.addHandler(get_file_handler(log_file))

    # Prevent the logger from propagating to the root logger
    logger.propagate = False

    return logger