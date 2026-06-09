"""Wrapper for config loading, seeding, and validation."""

import os
import random
import numpy as np
import torch
from utils.config import (
    load_config,
    add_params_and_hashes,
)
from utils.logger import get_logger
logger = get_logger(__name__)


def set_global_seed(seed):
    """
    Set global random seeds for reproducibility.
    
    Sets seeds for Python random, NumPy, and PyTorch (CPU and CUDA).
    Also sets PyTorch deterministic flags.
    
    Args:
        seed (int): Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Only set CUDA seed if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Set deterministic behavior for PyTorch (only relevant for CUDA)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        logger.warning("CUDA is not available. Using CPU only.")
    
    # Set environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # logger.info(f"Global random seed set to {seed}")


def validate_config(config):
    """
    Perform basic validation on configuration.
    
    Args:
        config (dict): Configuration dictionary.
        
    Raises:
        ValueError: If required config sections are missing.
    """
    required_sections = ['metadata', 'model', 'training']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Required config section '{section}' is missing.")
    
    metadata = config.get('metadata') or {}
    if 'session' not in metadata or not metadata.get('session'):
        raise ValueError("'metadata.session' is required.")
    
    # Validate model
    model_config = config.get('model', {})
    if 'name' not in model_config:
        raise ValueError("'model.name' is required.")
    
    logger.debug("Config validation passed.")


def prepare_config(config, force_rerun=False, grid_params=None):
    """
    Prepare a config dict (add params/hashes, seed, validate).
    Call when config was already loaded (e.g. from load_config or a merged permutation).

    Args:
        config (dict): The configuration dictionary to prepare.
        force_rerun (bool): If True, set run.rerun and clear completion flags as needed.
        grid_params (dict, optional): If provided, used for config['grid_params']
            (e.g. current permutation from pipeline). Otherwise derived from config.
    """
    add_params_and_hashes(config, overwrite=True, grid_params=grid_params)
    if 'seed' not in config:
        config['seed'] = 0
        logger.info("Seed not specified in config, using default: 0")
    set_global_seed(config['seed'])
    validate_config(config)
    config.setdefault('run', {})['rerun'] = force_rerun
    return config


def load_and_prepare_config(config_path, force_rerun=False):
    """
    Load, merge, validate, and prepare configuration.
    """
    config = load_config(config_path)
    config['_config_path'] = config_path
    return prepare_config(config, force_rerun)

