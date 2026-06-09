"""Local stage completion tracking using flag files."""

import os
from datetime import datetime
from utils.helpers import get_directory
from utils.logger import get_logger

logger = get_logger(__name__)

STAGE_FLAGS_DIR = ".stage_flags"


def get_stage_flag_path(config, stage_name, flag_type='completed'):
    """
    Get the path to a stage flag file.
    
    Args:
        config (dict): Configuration dictionary.
        stage_name (str): Name of the stage.
        flag_type (str): Type of flag ('completed' or 'failed').
        
    Returns:
        str: Path to the flag file.
    """
    output_dir = get_directory(config, 'output', long_path=False)
    flags_dir = os.path.join(output_dir, STAGE_FLAGS_DIR)
    os.makedirs(flags_dir, exist_ok=True)
    
    flag_filename = f".stage_{stage_name}_{flag_type}.flag"
    return os.path.join(flags_dir, flag_filename)


def check_stage_completed_locally(config, stage_name):
    """
    Check if a stage is marked as completed locally via flag file.
    
    Args:
        config (dict): Configuration dictionary.
        stage_name (str): Name of the stage.
        
    Returns:
        bool: True if stage is marked as completed, False otherwise.
    """
    flag_path = get_stage_flag_path(config, stage_name, flag_type='completed')
    return os.path.exists(flag_path)


def check_stage_failed_locally(config, stage_name):
    """
    Check if a stage is marked as failed locally via flag file.
    
    Args:
        config (dict): Configuration dictionary.
        stage_name (str): Name of the stage.
        
    Returns:
        bool: True if stage is marked as failed, False otherwise.
    """
    flag_path = get_stage_flag_path(config, stage_name, flag_type='failed')
    return os.path.exists(flag_path)


def mark_stage_completed_locally(config, stage_name):
    """
    Mark a stage as completed locally by creating a flag file.
    
    Args:
        config (dict): Configuration dictionary.
        stage_name (str): Name of the stage.
    """
    flag_path = get_stage_flag_path(config, stage_name, flag_type='completed')
    
    # Remove failed flag if it exists
    failed_flag_path = get_stage_flag_path(config, stage_name, flag_type='failed')
    if os.path.exists(failed_flag_path):
        os.remove(failed_flag_path)
    
    # Create completed flag file with timestamp
    with open(flag_path, 'w') as f:
        f.write(f"Stage '{stage_name}' completed at {datetime.now().isoformat()}\n")
    
    logger.debug(f"Marked stage '{stage_name}' as completed locally: {flag_path}")


def mark_stage_failed_locally(config, stage_name, error_message=None):
    """
    Mark a stage as failed locally by creating a flag file.
    
    Args:
        config (dict): Configuration dictionary.
        stage_name (str): Name of the stage.
        error_message (str, optional): Error message to include in the flag file.
    """
    flag_path = get_stage_flag_path(config, stage_name, flag_type='failed')
    
    # Remove completed flag if it exists
    completed_flag_path = get_stage_flag_path(config, stage_name, flag_type='completed')
    if os.path.exists(completed_flag_path):
        os.remove(completed_flag_path)
    
    # Create failed flag file with timestamp and error message
    with open(flag_path, 'w') as f:
        f.write(f"Stage '{stage_name}' failed at {datetime.now().isoformat()}\n")
        if error_message:
            f.write(f"Error: {error_message}\n")
    
    logger.debug(f"Marked stage '{stage_name}' as failed locally: {flag_path}")


def clear_stage_flag(config, stage_name, flag_type='completed'):
    """
    Clear a stage flag file (useful for force rerun).
    
    Args:
        config (dict): Configuration dictionary.
        stage_name (str): Name of the stage.
        flag_type (str): Type of flag to clear ('completed' or 'failed').
    """
    flag_path = get_stage_flag_path(config, stage_name, flag_type=flag_type)
    if os.path.exists(flag_path):
        os.remove(flag_path)
        logger.debug(f"Cleared {flag_type} flag for stage '{stage_name}': {flag_path}")

