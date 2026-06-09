import logging
from logging.handlers import RotatingFileHandler

import yaml
import os

from datetime import datetime
import time
import numpy as np
import sys
import re
import traceback

LOGGER_NAME = 'main'
logger = logging.getLogger(LOGGER_NAME)
logger = logging.LoggerAdapter(logger, extra={'script': __name__})

class UnicodeFilter(logging.Filter):
    """
    A custom filter to remove non-ASCII characters from log messages.
    """
    def filter(self, record):
        # Ensure the message is a string and replace non-ASCII characters
        if isinstance(record.msg, str):
            record.msg = record.msg.encode('ascii', 'replace').decode('ascii')
        return True


# _initialized_loggers = {}

# def setup_logger(config=None, name=__name__, level='INFO', temp=False, log_dir=None, log_file=None, log_path=None, enable_error_forwarding=False):
#     """
#     Set up a logger with file, console, and optional error forwarding handlers.
#
#     Args:
#         config (dict): Configuration dictionary for directory structure.
#         name (str): Logger name.
#         level (str or int): Logging level (e.g., 'INFO', 'DEBUG').
#         temp (bool): Use a temporary filename.
#         log_dir (str): Directory to store logs.
#         log_file (str): Specific log filename.
#         log_path (str): Full path to log file (overrides log_dir/log_file).
#         enable_error_forwarding (bool): Also log WARNING+ to shared error files.
#     """
#
#     # if is_logger_initialized(name):
#     #     close_logger(name)
#
#     # Convert string level to logging constant
#     level = getattr(logging, level.upper(), logging.INFO) if isinstance(level, str) else level
#
#     # Avoid reinitializing if logger already configured
#     if name in _initialized_loggers:
#         return _initialized_loggers[name]
#
#     # Determine log path
#     if log_path:
#         log_dir = os.path.dirname(log_path)
#         log_file = os.path.basename(log_path)
#     else:
#         log_dir = log_dir or get_directory(config, 'log', long_path=False) if config else 'log'
#         if log_file:
#             log_path = os.path.join(log_dir, log_file)
#         elif temp:
#             log_path = os.path.join(log_dir, "temp.log")
#         else:
#             timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#             log_path = os.path.join(log_dir, f"training_{timestamp}.log")
#
#     os.makedirs(log_dir, exist_ok=True)
#
#     # Create the logger
#     logger = logging.getLogger(name)
#     logger.setLevel(level)
#     logger.propagate = False
#
#     # Remove all existing handlers (reset clean)
#     logger.handlers.clear()
#
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#
#     # File handler
#     file_handler = logging.FileHandler(log_path, encoding='utf-8')
#     file_handler.setFormatter(formatter)
#     logger.addHandler(file_handler)
#
#     # Console handler
#     stream_handler = logging.StreamHandler(sys.stdout)
#     stream_handler.setFormatter(formatter)
#     logger.addHandler(stream_handler)
#
#     # Optional error forwarding
#     if enable_error_forwarding:
#         class ErrorFileHandler(logging.Handler):
#             def emit(self, record):
#                 if record.levelno >= logging.WARNING:
#                     detailed_log_path = os.path.join('log', 'error_preprocess_detailed.log')
#                     concise_log_path = os.path.join('log', 'error_preprocess_concise.log')
#                     os.makedirs(os.path.dirname(detailed_log_path), exist_ok=True)
#
#                     message = self.format(record)
#                     raw_message = record.getMessage()
#
#                     if record.levelno >= logging.ERROR and record.exc_info:
#                         trace = ''.join(traceback.format_exception(*record.exc_info))
#                         message += f"\nTraceback:\n{trace}"
#
#                     with open(detailed_log_path, 'a', encoding='utf-8') as f:
#                         f.write(message + '\n')
#
#                     with open(concise_log_path, 'a', encoding='utf-8') as f:
#                         f.write(raw_message + '\n')
#
#         error_handler = ErrorFileHandler()
#         error_handler.setLevel(logging.WARNING)
#         error_handler.setFormatter(formatter)
#         logger.addHandler(error_handler)
#
#     logger.info(f"Logger initialized at level {logging.getLevelName(level)} (to: {log_path})")
#
#     # Cache the logger so it’s not re-created
#     _initialized_loggers[name] = logger
#
#     return logger


def setup_logger(config=None, name=__name__, level='INFO', log_path=None, warning_log_path=None, error_log_path=None):
    """
    Sets up a logger with console output, full log file, optional warning and error log files.

    Args:
        config (dict): Configuration dictionary containing session metadata.
        name (str): Script/module name to include in log context (default: __name__).
        level (str or int): Logging level ('DEBUG', 'INFO', etc. or corresponding int).
        log_path (str): Path to the main full log file.
        warning_log_path (str): Path to a warning-only log file.
        error_log_path (str): Path to an error-only log file.
    """
    logger = logging.getLogger(LOGGER_NAME)

    # Avoid duplicate handlers if re-initializing
    if logger.hasHandlers():
        logger.handlers.clear()

    # Convert level string to numeric value
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False  # Prevent messages from going to root logger

    session_id = (config.get('metadata') or {}).get('session', '') if config else ''
    log_name = None
    if config is not None and log_path is None:
        # Use config to determine log path if not provided (lazy import to avoid circular import with utils.helpers)
        from utils.helpers import get_directory
        log_dir = get_directory(config, 'log', long_path=False)
        # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_name = f"{session_id}.log"
        log_path = os.path.join(log_dir, log_name)
    elif log_path:
        # Extract log name from provided log_path
        log_name = os.path.basename(log_path)

    # Create log directory if needed
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    if warning_log_path:
        os.makedirs(os.path.dirname(warning_log_path), exist_ok=True)

    # Formatter for regular logs and console
    standard_formatter = logging.Formatter(
        '%(asctime)s - %(script)-15s - %(levelname)-7s - %(message)s'
    )

    # Formatter for warning logs with session id
    special_formatter = logging.Formatter(
        f'%(asctime)s - %(script)-15s - %(levelname)-7s - {session_id} - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(standard_formatter)
    logger.addHandler(console_handler)

    # Full log (all levels) file handler
    if log_path:
        file_handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(standard_formatter)
        logger.addHandler(file_handler)

    # Warning log (WARNING and above) file handler (optional)
    if warning_log_path:
        warning_handler = RotatingFileHandler(warning_log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8')
        warning_handler.setLevel(logging.WARNING)
        warning_handler.setFormatter(special_formatter)
        logger.addHandler(warning_handler)

    # Error log (ERROR and above) file handler (optional)
    if error_log_path:
        error_handler = RotatingFileHandler(error_log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(special_formatter)
        logger.addHandler(error_handler)

    logger = get_logger(name)
    logger.info("Logger initialized.")
    return logger


def setup_temp_logger(name=__name__, level='INFO'):
    """
    Sets up a temporary logger with console output.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    return logger

def get_logger(name=__name__):
    """
    Retrieves a LoggerAdapter with the script/module name included.

    Args:
        script_name (str): The calling script/module name (usually use __name__).

    Returns:
        LoggerAdapter: A logger with the 'script' field injected.
    """
    base_logger = logging.getLogger(LOGGER_NAME)
    return logging.LoggerAdapter(base_logger, extra={'script': name or '__main__'})


def is_logger_initialized():
    """
    Checks whether the main logger has any handlers.

    Returns:
        bool: True if logger has handlers, False otherwise.
    """
    return bool(logging.getLogger(LOGGER_NAME).handlers)


def close_logger():
    """
    Flushes, closes, and removes all handlers from the main logger.
    """
    logger = logging.getLogger(LOGGER_NAME)
    handlers = logger.handlers[:]
    for handler in handlers:
        try:
            handler.flush()
            handler.close()
        except Exception as e:
            print(f"Error closing handler: {e}")
        logger.removeHandler(handler)


def log_welcome_message_from_config(config,logger, params, title="NeuroPixel 2 Rooms Experiment"):
    """
    Log a welcome message using the provided configuration.
    Args:
        config (dict): Configuration dictionary containing session metadata and paths information.
        logger (logging.Logger): Logger instance to log the welcome message.
        params (dict): Parameters to be included in the welcome message.
        title (str): Title of the experiment. Defaults to "NeuroPixel 2 Rooms Experiment".
    """
    metadata = config.get('metadata') or {}
    experiment_name = metadata.get('session', '')
    group = metadata.get('group', None)
    version = metadata.get('version', None)
    subdir = metadata.get('subdir', None)
    model_name = config['model']['name']
    hash_params = config.get('hash', {}).get('params', None)
    # Log the welcome message
    log_welcome_message(experiment_name, title, model_name, group, version, subdir, params, hash_params)

from datetime import datetime

def log_welcome_message(experiment_name, title="NeuroPixel 2 Rooms Experiment",
                        model_name=None, group=None, version=None, subdir=None, params=None, hash_params=None, title_length=48):
    import sys
    
    # Use ASCII characters on Windows to avoid encoding issues with console capture
    use_unicode = sys.platform != 'win32'

    def fit_string_to_width(text: str, max_length: int) -> str:
        """Truncate the string to fit within max_length, appending '...' if needed."""
        ellipsis = '... '
        if len(text) > max_length:
            return text[:max_length - len(ellipsis)] + ellipsis
        return text

    # Get the current date and time
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Adjust title_length if title is longer
    if len(title) > title_length:
        title_length = min(len(title) + 2, 60)
    # Calculate total box width
    # Base padding: emoji + space + title + space + emoji + fixed left/right border paddings
    inner_width = title_length + 25  # emojis + spaces + borders
    
    if use_unicode:
        border_char = "═"
        top_left = "╔"
        top_right = "╗"
        bottom_left = "╚"
        bottom_right = "╝"
        vertical = "║"
        emoji_star = "🌟✨"
        emoji_rocket = "🚀"
        emoji_calendar = "📅"
        emoji_people = "👥"
        emoji_package = "📦"
        emoji_robot = "🤖"
        emoji_wrench = "🔧"
        emoji_folder = "📂"
        emoji_hash = "🔑"
    else:
        border_char = "="
        top_left = "+"
        top_right = "+"
        bottom_left = "+"
        bottom_right = "+"
        vertical = "|"
        emoji_star = "***"
        emoji_rocket = "[*]"
        emoji_calendar = "[*]"
        emoji_people = "[*]"
        emoji_package = "[*]"
        emoji_robot = "[*]"
        emoji_wrench = "[*]"
        emoji_folder = "[*]"
        emoji_hash = "[*]"
    
    border_line = border_char * inner_width

    # Construct the welcome message
    lines = [
        f"\n",
        f"    {top_left}{border_line}{top_right}",
        f"    {vertical}    {emoji_star:<5} {title:^{title_length}} {emoji_star:>5}    {vertical}",
        f"    {vertical}{border_char * inner_width}{vertical}",
        f"    {vertical}{' ' * inner_width}{vertical}"
    ]

    # Optional fields
    optional_fields = [
        (emoji_rocket, "Session", experiment_name),
        (emoji_calendar, "Date", current_date),
        (emoji_people, "Group", group),
        (emoji_package, "Version", version),
        (emoji_folder, "Subdir", subdir),
        (emoji_robot, "Model", model_name),
        (emoji_hash, "Hash", hash_params),
    ]

    for emoji, label, value in optional_fields:
        if value:
            pad = (len(label) if label else 0) + (len(emoji) if label else 0) + 13  # emoji + label + padding
            lines.append(
                f"    {vertical}        {emoji}  {label}: {fit_string_to_width(value, inner_width - pad):<{inner_width - pad}}{vertical}"
            )

    # Params block
    if params:
        emoji, label = emoji_wrench, 'Params'
        pad = (len(label) if label else 0) + (len(emoji) if label else 0) + 13  # emoji + label + padding
        lines.append(f"    {vertical}        {emoji}  {label}: {' ' * (inner_width - pad)}{vertical}")

        for key, val in params.items():
            if key.startswith('*'):
                continue
            param_line = f"{key}: {val}"
            pad = 15
            lines.append(f"    {vertical}             - {fit_string_to_width(param_line, inner_width - pad):<{inner_width - pad}}{vertical}")

    # Close the border of the message
    lines.append(f"    {vertical}{' ' * inner_width}{vertical}")
    lines.append(f"    {bottom_left}{border_line}{bottom_right}")

    # Log the welcome message with error handling for encoding issues
    try:
        logger.info("\n".join(lines))
    except UnicodeEncodeError:
        # Fallback to ASCII if Unicode fails
        logger.info("\n".join(lines).encode("ascii", errors="replace").decode("ascii"))



def log_completion_message(start_time):
    import sys
    
    # Use ASCII characters on Windows to avoid encoding issues with console capture
    use_unicode = sys.platform != 'win32'
    
    # Get the current date and time
    end_time = time.time()
    duration = end_time - start_time
    formatted_duration = format_time_difference(duration) if duration>0 else " "

    if use_unicode:
        border_char = "═"
        top_left = "╔"
        top_right = "╗"
        bottom_left = "╚"
        bottom_right = "╝"
        vertical = "║"
        checkmark = "✅"
        clock = "🕑"
    else:
        border_char = "="
        top_left = "+"
        top_right = "+"
        bottom_left = "+"
        bottom_right = "+"
        vertical = "|"
        checkmark = "[OK]"
        clock = "[*]"

    border_line = border_char * 64

    completion_message = f"""

    {top_left}{border_line}{top_right}
    {vertical}{' ' * 64}{vertical}
    {vertical}        {checkmark}  RUNNING COMPLETED{' ' * (64 - 25 - len(checkmark))}{vertical}
    {vertical}        {clock}  Duration: {formatted_duration:<44}      {vertical}
    {vertical}{' ' * 64}{vertical}
    {bottom_left}{border_line}{bottom_right}
    """

    # Log the completion message with error handling for encoding issues
    try:
        logger.info(completion_message)
    except UnicodeEncodeError:
        # Fallback to ASCII if Unicode fails
        logger.info(completion_message.encode("ascii", errors="replace").decode("ascii"))


def log_box_message(message):
    import sys
    
    # Use ASCII characters on Windows to avoid encoding issues with console capture
    use_unicode = sys.platform != 'win32'
    
    # Add borders around the message
    if use_unicode:
        try:
            border = "    ╔" + "═" * (len(message) + 2) + "╗"
            footer = "    ╚" + "═" * (len(message) + 2) + "╝"
            centered_message = f"    ║ {message} ║"
            full_message = f"\n{border}\n{centered_message}\n{footer}"
        except (UnicodeEncodeError, UnicodeError):
            use_unicode = False
    
    if not use_unicode:
        # Use ASCII characters
        border = "    +" + "-" * (len(message) + 2) + "+"
        footer = "    +" + "-" * (len(message) + 2) + "+"
        centered_message = f"    | {message} |"
        full_message = f"\n{border}\n{centered_message}\n{footer}"

    # Log the centered completion message
    try:
        logger.info(full_message)
    except UnicodeEncodeError:
        # If logging still fails, use ASCII version
        border = "    +" + "-" * (len(message) + 2) + "+"
        footer = "    +" + "-" * (len(message) + 2) + "+"
        centered_message = f"    | {message} |"
        full_message = f"\n{border}\n{centered_message}\n{footer}"
        logger.info(full_message)

def format_time_difference(seconds, precision=0):
    """
    Format the time difference in seconds into a string with days, hours, minutes, and seconds.

    Parameters:
        seconds (int): The time difference in seconds.
        precision (int): The number of non-zero values to show (default is 0 which means show all).

    Returns:
        str: The formatted time difference.
    """
    periods = [
        ('days', 86400),  # 60 * 60 * 24
        ('hours', 3600),  # 60 * 60
        ('minutes', 60),
        ('seconds', 1)
    ]

    strings = []
    for name, count in periods:
        value = seconds // count
        if value:
            seconds -= value * count
            strings.append(f"{value} {name}")

    if precision > 0:
        strings = strings[:precision]

    return ' '.join(strings)


def log_boundary_points(boundary_points, n_pixel=50):
    """
    Convert boundary points into a heatmap of size n_pixel x n_pixel and log the result.

    Parameters:
    - logger: The logger object to log the heatmap message.
    - boundary_points: Array of shape (n, 2) where each row is (x, y) representing the boundary points.
    - n_pixel: The size of the heatmap (n_pixel x n_pixel//2).

    """
    # Normalize boundary points to the range [0, 1]
    if boundary_points.size == 0:
        return

    boundary_points = np.array(boundary_points)
    x_min, x_max = boundary_points[:, 0].min(), boundary_points[:, 0].max()
    y_min, y_max = boundary_points[:, 1].min(), boundary_points[:, 1].max()

    # Avoid division by zero if all points are the same
    if x_max == x_min:
        x_max = x_min + 1
    if y_max == y_min:
        y_max = y_min + 1

    normalized_points = (boundary_points - [x_min, y_min]) / [x_max - x_min, y_max - y_min]

    # Create an empty heatmap
    heatmap = np.zeros((n_pixel//4, n_pixel), dtype=int)

    # Populate the heatmap
    for x, y in normalized_points:
        x_idx = int(x * (n_pixel - 1-8)+4)
        y_idx = int((1-y) * (n_pixel//4 - 1-2)+1)
        heatmap[y_idx, x_idx] = 1

    heatmap = heatmap[::-1,:]

    # Convert the heatmap to a message with a border
    import sys
    # Use ASCII characters on Windows to avoid encoding issues with console capture
    use_unicode = sys.platform != 'win32'
    
    if use_unicode:
        try:
            border = "    ╔" + "═" * (n_pixel + 2) + "╗"
            footer = "    ╚" + "═" * (n_pixel + 2) + "╝"
            heatmap_message = "\n".join("    ║ " + "".join('*' if cell else ' ' for cell in row) + " ║" for row in heatmap)
            message = f"{border}\n{heatmap_message}\n{footer}"
        except (UnicodeEncodeError, UnicodeError):
            use_unicode = False
    
    if not use_unicode:
        # Use ASCII characters
        border = "    +" + "-" * (n_pixel + 2) + "+"
        footer = "    +" + "-" * (n_pixel + 2) + "+"
        heatmap_message = "\n".join("    | " + "".join('*' if cell else ' ' for cell in row) + " |" for row in heatmap)
        message = f"{border}\n{heatmap_message}\n{footer}"

    # Log the completion message
    try:
        logger.info("boundary:\n\n" + message + "\n")
    except UnicodeEncodeError:
        # If logging still fails, use ASCII version
        border = "    +" + "-" * (n_pixel + 2) + "+"
        footer = "    +" + "-" * (n_pixel + 2) + "+"
        heatmap_message = "\n".join("    | " + "".join('*' if cell else ' ' for cell in row) + " |" for row in heatmap)
        message = f"{border}\n{heatmap_message}\n{footer}"
        logger.info("boundary:\n\n" + message + "\n")





class UnicodeFilter(logging.Filter):
    def filter(self, record):
        # Replace special Unicode characters with a space or empty string
        record.msg = re.sub(r'[^\x00-\x7F]+', ' ', record.msg)  # Replace non-ASCII characters
        return True

