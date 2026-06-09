import os
import shutil
import traceback
import time
from datetime import datetime
import re
import yaml

from utils.helpers import get_directory, get_dataset_path, get_rooms_from_config, get_time_range_from_config, \
    format_room_name
from utils.logger import get_logger, log_welcome_message_from_config, setup_logger, is_logger_initialized
import copy
from itertools import product
import csv
import pandas as pd
import ast
import numpy as np
import hashlib
import json

# Initialize logger
logger = get_logger(__name__)

SKIP_LIST_FILE = "skipped_permutations.txt"

def load_config(config_path: str, load_default: bool = True, default_config_path: str = './config/common_config.yaml', load_local_config: bool = True) -> dict:
    """
    Load configuration file.

    This function loads the user-provided configuration file and optionally combines it with the default
    model configuration. The user configuration takes priority over the default configuration.

    Args:
        config_path (str): Path to the user-provided configuration file.
        load_default (bool): Flag indicating whether to load the default models config.
        default_config_path (str): Path to the default models configuration file.
                                   Defaults to './config/common_config.yaml'.
        load_local_config (bool): Flag indicating whether to load the local config.

    Returns:
        dict: Combined configuration dictionary with user values prioritized.

    Raises:
        FileNotFoundError: If the user-provided or default models configuration file is not found.
    """

    # combined_config = {}
    if load_default:
        # Load the models' default config from './config/common_config.yaml'
        # default_common_config_path = './config/common_config.yaml'

        # Check if common_config.yaml exists
        if not os.path.isfile(default_config_path):
            #logger.error(f"Default models config file not found at {default_config_path}")
            raise FileNotFoundError(f"Default models config file not found at {default_config_path}")

        with open(default_config_path, 'r') as file:
            default_config = yaml.safe_load(file)

        # combined_config.update(default_config)
    else:
        default_config = {}

    # Load the user-provided config
    if not os.path.isfile(config_path):
        #logger.error(f"Config file not found at {config_path}")
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, 'r') as file:
        user_config = yaml.safe_load(file)

    # Merge the two configs with user_config taking priority over default_config
    combined_config = update_config(default_config, user_config)

    # Load the local config if specified
    if load_local_config:
        local_config_path = get_dataset_path(combined_config, 'local_config')
        if os.path.isfile(local_config_path):
            with open(local_config_path, 'r') as file:
                local_config = yaml.safe_load(file)
            # Merge the local config with the combined config (combined_config taking priority)
            combined_config = update_config(local_config,combined_config)

    # Set animal and dates in metadata if not provided, example: 1130_18092024_2rooms > animal: 1130, date: 18092024
    metadata = combined_config.setdefault('metadata', {})
    if 'animal' not in metadata or 'date' not in metadata:
        session_id = metadata.get('session', '')
        animal_date = session_id.split('_')[0] if '_' in session_id else ''
        date = session_id.split('_')[1] if '_' in session_id else ''
        metadata.update({'animal': animal_date, 'date': date})

    # Set config path (absolute for reproducibility metadata and stable resolves)
    combined_config['_config_path'] = os.path.abspath(config_path)
    if load_default:
        combined_config['_common_config_path'] = os.path.abspath(default_config_path)

    # Initialize model from template if templates are present
    if 'model' in combined_config and 'templates' in combined_config.get('model', {}):
        try:
            combined_config = initialize_model_from_template(combined_config)
        except KeyError as e:
            # If template initialization fails, log warning but don't fail
            # This allows configs that don't use templates to still work
            logger.warning(f"Could not initialize model from template: {e}. Continuing without template initialization.")

    #logger.info("Configuration successfully loaded")

    return combined_config


def update_config(original_config: dict, new_config: dict) -> dict:
    """
    Recursively update the original configuration with values from the manual configuration.

    Args:
        original_config (dict): The original configuration dictionary.
        new_config (dict): The manual configuration dictionary with values to overwrite.

    Returns:
        dict: The updated configuration dictionary.
    """
    result = copy.deepcopy(original_config)

    for key, value in new_config.items():
        if isinstance(value, dict):
            orig_value = result.get(key)
            if isinstance(orig_value, dict):
                result[key] = update_config(orig_value, value)
            else:
                result[key] = copy.deepcopy(value)
        else:
            result[key] = value

    return result


def initialize_model_from_template(config, model_name = None, force=False):
    """
    Initialize model variables from a template and update the config.

    Args:
        config (dict): The configuration dictionary containing model settings.
        model_name (str, optional): The name of the model template to use. If None, it uses the model name from the config.
        force (bool): A flag to force the update of the model configuration with the selected template.

    Returns:
        dict: The updated configuration dictionary.
    """
    if model_name is None:
        model_name = config['model']['name']
    
    # Check if templates exist - if not, model is already initialized, return as-is
    if 'model' not in config or 'templates' not in config.get('model', {}):
        # Templates already removed, model already initialized from template
        return config
    
    copy_config = copy.deepcopy(config)
    try:
        # Update the model configuration with the selected template
        if not force:
            copy_config['model'] = update_config(copy_config['model']['templates'][model_name], copy_config['model'])
        else:
            copy_config['model'] = update_config(copy_config['model'],copy_config['model']['templates'][model_name])
    except KeyError as e:
        raise KeyError(f"Template for model '{model_name}' not found or invalid config structure: {e}")

    # remove the templates from the model config
    copy_config['model'].pop('templates', None)

    # generate cell indices if cell_filter is specified
    if 'cell_filter' in copy_config['model'] and isinstance(copy_config['model']['cell_filter'], dict):
        generate_cell_indices_from_filter(copy_config)
        # Update input_dim to match the number of active cells
        if 'cell_indices' in copy_config['model'] and isinstance(copy_config['model']['cell_indices'], list):
            copy_config['model']['input_dim'] = len(copy_config['model']['cell_indices'])
    elif 'neuron_filter' in copy_config['model'] and isinstance(copy_config['model']['neuron_filter'], dict):
        # Backward compatibility: convert neuron_filter to cell_filter
        copy_config['model']['cell_filter'] = copy_config['model'].pop('neuron_filter')
        generate_cell_indices_from_filter(copy_config)
        # Update input_dim to match the number of active cells
        if 'cell_indices' in copy_config['model'] and isinstance(copy_config['model']['cell_indices'], list):
            copy_config['model']['input_dim'] = len(copy_config['model']['cell_indices'])
    elif 'cell_indices' in copy_config['model'] and isinstance(copy_config['model']['cell_indices'], list):
        # If cell_indices is already present, count the filtered cells
        copy_config['model']['n_active_cells'] = len(copy_config['model']['cell_indices'])
        # Update input_dim to match the number of active cells
        copy_config['model']['input_dim'] = copy_config['model']['n_active_cells']
        # Validate cell_indices is not empty
        if copy_config['model']['n_active_cells'] == 0:
            raise ValueError("cell_indices is empty. At least one cell must be active after filtering.")
        # Set n_total_cells if not already set (should be set during preprocessing)
        if 'n_total_cells' not in copy_config['model']:
            # Try to get from input_dim if available (before filtering)
            # This is a fallback - ideally n_total_cells should be set during preprocessing
            copy_config['model']['n_total_cells'] = copy_config['model'].get('input_dim', None)
    elif 'neuron_indices' in copy_config['model'] and isinstance(copy_config['model']['neuron_indices'], list):
        # Backward compatibility: convert neuron_indices to cell_indices
        copy_config['model']['cell_indices'] = copy_config['model'].pop('neuron_indices')
        copy_config['model']['n_active_cells'] = len(copy_config['model']['cell_indices'])
        # Update input_dim to match the number of active cells
        copy_config['model']['input_dim'] = copy_config['model']['n_active_cells']
        if copy_config['model']['n_active_cells'] == 0:
            raise ValueError("cell_indices is empty. At least one cell must be active after filtering.")
        if 'n_total_cells' not in copy_config['model']:
            copy_config['model']['n_total_cells'] = copy_config['model'].get('input_dim', None)
    else:
        # If cell_indices is not specified, set it to None
        copy_config['model']['cell_indices'] = None
        # When no filtering, n_active_cells equals input_dim (all cells are used)
        input_dim = copy_config['model'].get('input_dim', 0)
        copy_config['model']['n_active_cells'] = input_dim
        copy_config['model']['n_total_cells'] = input_dim  # Total equals filtered when no filtering

    # add params and hashes to the model config
    add_params_and_hashes(copy_config, overwrite=True)

    # Return the updated configuration
    return copy_config

def generate_cell_indices_from_filter(config):
    """
    Generate cell indices based on the cell filter specified in the configuration.
    Args:
        config (dict): The configuration dictionary containing model settings.
    Returns:
        None: The function updates the config in place with the cell indices.
    Raises:
        ValueError: If no cells match the filter criteria (all cells filtered out).
    """
    if 'cell_filter' in config['model'] and isinstance(config['model']['cell_filter'], dict):
        cell_filter = config['model']['cell_filter']
        clusters_dataset_path = get_dataset_path(config, 'clusters')
        if os.path.exists(clusters_dataset_path):
            dfClusters = pd.read_csv(clusters_dataset_path)
            mask = pd.Series([True] * len(dfClusters))  # Start with all True mask
            for key, value in cell_filter.items():
                # Handle backward compatibility: Velocity_Cell <-> Speed_Cell
                # Note: Velocity0.6_Cell and SI1_Cell are no longer supported
                actual_key = key
                if key == 'Velocity_Cell' and 'Velocity_Cell' not in dfClusters.columns:
                    if 'Speed_Cell' in dfClusters.columns:
                        actual_key = 'Speed_Cell'
                elif key == 'Speed_Cell' and 'Speed_Cell' not in dfClusters.columns:
                    if 'Velocity_Cell' in dfClusters.columns:
                        actual_key = 'Velocity_Cell'
                
                if actual_key in dfClusters.columns:
                    # Create a mask based on the filter condition
                    key_mask = dfClusters[actual_key].isin(value) if isinstance(value, list) else dfClusters[actual_key] == value
                    # Update the mask
                    mask &= key_mask  # Combine masks using bitwise AND
                else:
                    logger.warning(f"Key '{key}' (or '{actual_key}') not found in clusters dataset. Skipping cell filter calculation for this key.")
            # Convert the mask to a numpy array
            mask = mask.to_numpy(dtype=bool)
            # Extract indices of active cells
            cell_indices = np.where(mask)[0].tolist()
            n_total_cells = len(dfClusters)
            
            # Edge case: Check if all cells were filtered
            if len(cell_indices) == 0:
                raise ValueError(
                    f"All cells were filtered out by the cell_filter. "
                    f"Filter criteria: {cell_filter}. "
                    f"At least one cell must remain active after filtering. "
                    f"Please check the cell_filter configuration and ensure it is correct."
                )
            
            # Add the cell indices to the config
            config['model']['cell_indices'] = cell_indices
            config['model']['n_active_cells'] = len(cell_indices)  # Count of active/selected cells
            config['model']['n_total_cells'] = n_total_cells  # Total number of cells before filtering
            # Update input_dim to match the number of active cells
            config['model']['input_dim'] = len(cell_indices)
            logger.info("Cell indices generated based on the cell filter.")
            logger.info(f"Active cells: {len(cell_indices)} out of {n_total_cells} total cells")

        del dfClusters

def add_params_and_hashes(config, overwrite=False, grid_params=None):
    """
    Adds params, grid_params, params_str, and hash to the config.

    Args:
        config (dict): The configuration dictionary to update.
        overwrite (bool): If True, overwrites existing values. Defaults to False.
        grid_params (dict, optional): If provided, used for config['grid_params']
            (e.g. current permutation from pipeline). Otherwise derived from config.
    """
    add_grid_params = config.get('training', {}).get('hyperparameter_search', {}).get('enabled', False)

    params = get_params_from_config(config, full_params=True)
    if grid_params is not None and isinstance(grid_params, dict):
        grid_params = dict(grid_params)
    else:
        grid_params = get_params_from_config(config, full_params=False)

    if overwrite or 'params' not in config:
        config['params'] = params

    if add_grid_params and (overwrite or 'grid_params' not in config):
        config['grid_params'] = grid_params

    if overwrite or 'params_str' not in config:
        config['params_str'] = format_params_string(grid_params, format_type='file_name')

    if overwrite or 'hash' not in config:
        params_hash = hash_params(params)
        grid_params_hash = hash_params(grid_params)
        config_hash = hash_config(config)

        config['hash'] = {
            'params': params_hash,
            'grid_params': grid_params_hash,
            'config': config_hash
        }


def export_config(config: dict, filename: str = "config.yaml") -> None:
    """
    Export the provided configuration to a YAML file.

    Args:
        config (dict): The configuration dictionary to export.
        filename (str): The name of the file to which the configuration will be exported.
                        Defaults to 'config_export.yaml'.

    Raises:
        Exception: If an error occurs while trying to export the configuration.
    """
    try:
        # Get the output directory using the provided function
        output_dir = get_directory(config, 'output', long_path=False)

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Define the full output path for the configuration file
        output_path = os.path.join(output_dir, filename)

        # Export the configuration to a YAML file
        with open(output_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)

        # Log the successful export
        logger.info(f"Configuration successfully exported to {output_path}")

    except Exception as e:
        # Log any exceptions that occur
        logger.error(f"Failed to export configuration: {e}")


def generate_param_grid(param_grid, order=None):
    """
    Generate all permutations of hyperparameters.
    If `order` is given, param_grid keys are reordered: first by `order`, then
    any remaining keys. Permutation order follows the (possibly reordered) key order.
    """
    if order:
        ordered = {k: param_grid[k] for k in order if k in param_grid}
        for k in param_grid:
            if k not in ordered:
                ordered[k] = param_grid[k]
        param_grid = ordered
    keys, values = zip(*param_grid.items())
    permutations = [dict(zip(keys, v)) for v in product(*values)]
    return permutations

def load_permutations_from_csv(csv_path):
    # """
    # Load hyperparameter permutations from a CSV file.
    # """
    # permutations = []
    # with open(csv_path, 'r') as csvfile:
    #     reader = csv.DictReader(csvfile)
    #     for row in reader:
    #         permutations.append({key: value for key, value in row.items()})
    # return permutations

    """
    Load hyperparameter permutations from a CSV file into a DataFrame,
    then extract the permutations as a list of dictionaries.
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Fix parameter values type
    if 'model.rnn.hidden_dim' in df.columns: df['model.rnn.hidden_dim'] = df.get('model.rnn.hidden_dim',pd.Series(dtype='int')).astype(int)
    if 'model.batch_size' in df.columns: df['model.batch_size'] = df.get('model.batch_size',pd.Series(dtype='int')).astype(int)
    if 'model.optimizer.learning_rate' in df.columns: df['model.optimizer.learning_rate'] = df.get('model.optimizer.learning_rate', pd.Series(dtype='float')).astype(float)
    # Extract permutations as a list of dictionaries
    permutations = df.to_dict(orient='records')
    for rec in permutations:
        for k in rec:
            if rec[k] == '[]':
                rec[k] = []

    return permutations


def save_permutations_to_csv(param_permutations, export_path):
    """
    Save hyperparameter permutations to a CSV file.

    Parameters:
        param_permutations (list of dict): List of hyperparameter permutations.
        export_path (str): Path to the CSV file for saving the permutations.

    Returns:
        None: Saves the permutations to a CSV file.
    """
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(param_permutations)

    # Save the DataFrame to a CSV file
    df.to_csv(export_path, index=False)

def convert_dict_flat_to_nested(flat_dict, sep='.'):
    """
    Convert a flat dictionary with dot notation into a nested dictionary.

    Args:
        flat_dict (dict): A dictionary where keys may contain dots representing nested structures.
        sep (str): Separator used to indicate nesting (default: '.').

    Returns:
        dict: A nested dictionary.
    """

    def smart_cast(value):
        """
        Try to convert the input string to int, float, bool, or list if possible.
        """
        if isinstance(value, str):
            val_lower = value.lower().strip()
            # Handle boolean strings from flat configs / CSV grids
            if val_lower in {"true", "false"}:
                return val_lower == "true"
            try:
                # Try int, float, list/dict or None using ast.literal_eval
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return value  # Return as-is (likely a string)

        elif isinstance(value, list):
            # If the value is a list, convert its elements to proper types
            return [smart_cast(item) for item in value]
        elif isinstance(value, (float, np.float64)) and np.isnan(value):
            return None
        elif isinstance(value, bool):
            return value

        return value

    nested_dict = {}
    for flat_key, value in flat_dict.items():
        keys = flat_key.split(sep)
        d = nested_dict
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        
        # Convert value using smart_cast
        cast_value = smart_cast(value)
        
        # Some numeric grids encode booleans as 0/1; normalize when the key looks boolean-ish.
        boolean_keywords = ['batch_norm', 'bidirectional', 'active', 'enabled', 'last_state', 'multi_room']
        if isinstance(cast_value, (int, float)) and cast_value in [0, 1]:
            if any(keyword in flat_key.lower() for keyword in boolean_keywords):
                cast_value = bool(cast_value)
        
        d[keys[-1]] = cast_value
    return nested_dict


def convert_dict_nested_to_flat(nested_dict, parent_key='', sep='.'):
    """
    Convert a nested dictionary into a flat dictionary with dot notation keys.

    Args:
        nested_dict (dict): The nested input dictionary.
        parent_key (str): The base key for recursion (used internally).
        sep (str): Separator for nested keys (default: '.').

    Returns:
        dict: A flat dictionary with dot notation keys.
    """
    items = {}
    for key, value in nested_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(convert_dict_nested_to_flat(value, new_key, sep=sep))
        else:
            items[new_key] = value
    return items



def extract_hyperparameters(config):
    """
    Extracts hyperparameters from the configuration.

    Prefers config['grid_params'] when present, so that hyperparameter extraction works when hyperparameter_grid is not available. Otherwise uses hyperparameter_grid to derive keys and
    extracts values from the nested config.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        dict: Dictionary of extracted hyperparameters.
    """
    # Prefer grid_params when already set (e.g. from load_permutations_from_csv or pipeline)
    grid_params = config.get('grid_params', {})
    if isinstance(grid_params, dict) and len(grid_params) > 0:
        return dict(grid_params)

    # Fall back to deriving from hyperparameter_grid
    params = {}
    hyperparameter_grid = config.get('training', {}).get('hyperparameter_search', {}).get('hyperparameter_grid', {})

    for key in hyperparameter_grid.keys():
        key_parts = key.split('.')
        current_value = config
        for part in key_parts:
            if part in current_value:
                current_value = current_value[part]
            else:
                current_value = None
                break
        params[key] = current_value

    return params

def get_params_from_config(config, simplify = False, ignore_keys = [], add_model_name=True, full_params=False):
    """
    Extract parameters from the configuration for creating a checkpoint name.

    Args:
    - config (dict): A dictionary containing configuration data for the model, training, and evaluation settings.
    - simplify (bool): A flag to simplify the parameter keys for easier display in formatted output.
    - ignore_keys (list): A list of keys to ignore when extracting parameters.
    - add_model_name (bool): A flag to include the model name in the parameters.

    Returns:
    - dict: A dictionary containing relevant parameters such as model name, room, learning rate, and batch size.
    """
    hyperparameter_search_enabled = config.get('training', {}).get('hyperparameter_search', {}).get('enabled', False)
    model_config = config['model']
    model_name = model_config.get('name', '')

    if hyperparameter_search_enabled and not full_params:
        # Extract hyperparameters if hyperparameter search is enabled
        params = extract_hyperparameters(config)
        params = covert_params_list_values_to_string(params)
    else:
        # Extract model and training parameters from the base configuration
        # train_room = config['training']['room']
        # batch_size = int(model_config['batch_size'])
        # learning_rate = float(model_config['optimizer']['learning_rate'])
        #
        # params = {
        #     'model.name': model_name,
        #     # 'training.room': train_room,
        #     'model.batch_size': batch_size,
        #     'model.optimizer.learning_rate': learning_rate
        # }

        params = convert_dict_nested_to_flat(model_config, parent_key='model', sep='.')
        params = covert_params_list_values_to_string(params)


    if add_model_name and 'model.name' not in params:
        params = update_config({'model.name': model_name}, params)

    # If order is specified, sort the params dictionary by the order of keys in hyperparameter_grid
    order = config.get('training', {}).get('hyperparameter_search', {}).get('order', [])
    if order:
        if add_model_name and 'model.name' not in order:
            order = ['model.name'] + order
        order_params = {}
        # Add params in the order specified, removing them from the original params
        for key in order:
            if key in params:
                order_params[key] = params.pop(key)
        # Add any remaining params that were not in the order
        for key, value in params.items():
            order_params[key] = value
        params = order_params  # Update params to the ordered version

    if simplify:
        params = simplify_params(params)

    if ignore_keys:
        for key in ignore_keys:
            params.pop(key, None)

    return params


def get_fold_offset_range(config):
    """
    Parse fold_offset_range from config and return offset list and count.
    
    Handles three cases:
    - None: Returns [0] with count 1 (default behavior, no offset variation)
    - int: Auto-generates range [0, 1, 2, ..., N-1] with count = N
    - list: Returns the list as-is with count = len(list)
    
    Examples:
        >>> config = {'training': {'fold_offset_range': None}}
        >>> get_fold_offset_range(config)
        ([0], 1)
        
        >>> config = {'training': {'fold_offset_range': 5}}
        >>> get_fold_offset_range(config)
        ([0, 1, 2, 3, 4], 5)
        
        >>> config = {'training': {'fold_offset_range': [0, 2, 4]}}
        >>> get_fold_offset_range(config)
        ([0, 2, 4], 3)
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (offset_list, offset_count) where:
            - offset_list (list): List of offset values (e.g., [0, 1, 2, ...])
            - offset_count (int): Number of offsets
    """
    fold_offset_range = config.get('training', {}).get('fold_offset_range', None)
    
    if fold_offset_range is None:
        # Default behavior: no offset
        offset_list = [0]
        offset_count = 1
    elif isinstance(fold_offset_range, int):
        # Auto-generate range [0, 1, 2, ..., N-1]
        offset_list = list(range(fold_offset_range))
        offset_count = fold_offset_range
    elif isinstance(fold_offset_range, list):
        # Use explicit list
        offset_list = fold_offset_range
        offset_count = len(fold_offset_range)
    else:
        # Fallback to no offset
        offset_list = [0]
        offset_count = 1
    
    return offset_list, offset_count


def get_target_columns_from_config(config):
    """
    Derive target columns from config based on model target type.
    
    Checks config in this order:
    1. config['target_columns'] (explicit)
    2. config['model']['target'] (derived from target type)
    3. Falls back to ['X', 'Y'] if unknown
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        list: List of target column names (e.g., ['X', 'Y'], ['HD'], ['V'])
    """
    # Check explicit target_columns first
    if 'target_columns' in config and config['target_columns']:
        return list(config['target_columns'])
    
    # Derive from model.target
    model_cfg = config.get('model', {}) or {}
    target = model_cfg.get('target')
    if target:
        target_lower = str(target).lower()
        if target_lower in ('xy_position', 'xy', 'position'):
            return ['X', 'Y']
        if target_lower in ('xyhd_position', 'xy_hd', 'xyhd'):
            return ['X', 'Y', 'HD']
        if target_lower in ('hd', 'head_direction'):
            return ['HD']
        if target_lower in ('velocity', 'v', 'speed'):
            return ['V']
    
    # Default fallback
    return ['X', 'Y']


def load_position_scaler_from_config(config):
    """
    Load the position scaler from the config's data directory.
    
    Loads `positions_scaler.pkl` from the session data directory when present.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        scaler: The position scaler object, or None if not found/error
    """
    import joblib
    import warnings
    try:
        data_dir = get_directory(config, 'data')
        scaler_path = os.path.join(data_dir, 'positions_scaler.pkl')
        if os.path.exists(scaler_path):
            # Suppress sklearn version mismatch warnings when loading scaler
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
            scaler = joblib.load(scaler_path)['scaler']
            return scaler
        else:
            logger.warning(f"Position scaler file not found at {scaler_path}")
            return None
    except Exception as e:
        logger.warning(f"Could not load position scaler: {e}")
        return None


def get_scale_factor_from_scaler(scaler):
    """
    Extract scale factor from a scaler object for converting normalized values to physical units.
    
    This is a general utility function that extracts the scale factor from a scaler,
    which can be used for converting any normalized values (e.g., RMSE, distances, etc.)
    back to physical units (e.g., cm).
    
    Assumes scaler normalizes jointly (single scale factor). Uses the first scale factor.
    
    Args:
        scaler: Scaler object (e.g., MinMaxScaler, StandardScaler) with 'scale_' attribute
        
    Returns:
        float: Scale factor for converting normalized values to physical units.
               Returns 1.0 if scaler is None or doesn't have 'scale_' attribute.
               
    Examples:
        >>> # Extract scale factor (assumes joint normalization)
        >>> scale_factor = get_scale_factor_from_scaler(scaler)
        >>> value_cm = value_normalized * scale_factor
    """
    if scaler is None:
        return 1.0
    
    if not hasattr(scaler, 'scale_'):
        logger.warning("Scaler does not have 'scale_' attribute, returning scale_factor=1.0")
        return 1.0
    
    # For joint normalization, use the scale from the first dimension
    if len(scaler.scale_) > 0:
        scale_factor = float(scaler.scale_[0])
    else:
        scale_factor = 1.0
    
    return scale_factor


def assign_room_column(df, config, room_column='room', timestamp_column='timestamp'):
    """
    Assign room labels to dataframe based on timestamp ranges in config.
    Applies format_room_name to the room_column.
    
    Args:
        df: DataFrame with timestamp column
        config: Configuration dictionary with map_rooms structure
        room_column: Name of room column to create/update (default: 'room')
        timestamp_column: Name of timestamp column (default: 'timestamp')
    
    Returns:
        DataFrame with room column assigned and formatted
    """
    df_copy = df.copy()
    
    if room_column in df_copy.columns:
        return df_copy
    
    map_rooms = config.get('preprocessing', {}).get('map_rooms', {}).get('rooms', {})
    if not map_rooms:
        raise ValueError(f"Room column '{room_column}' not found in dataframe and no map_rooms config to infer it.")

    if timestamp_column not in df_copy.columns:
        raise ValueError(f"Timestamp column '{timestamp_column}' not found in dataframe.")

    def infer_room(ts):
        for room_name, room_cfg in map_rooms.items():
            ts_range = room_cfg.get('range', [])
            if len(ts_range) == 2 and ts_range[0] <= ts < ts_range[1]:
                return room_name
        return None
    
    df_copy[room_column] = df_copy[timestamp_column].apply(infer_room)
    missing = df_copy[room_column].isna().sum()
    if missing > 0:
        logger.warning(f"Room inference left {missing} rows without room assignment.")
    
    # Apply format_room_name to room_column (now handles NaN/None internally)
    df_copy[room_column] = df_copy[room_column].apply(format_room_name)
    
    return df_copy


def hash_params(params, length=8):
    """
    Generate an order-invariant hash for the given parameters.

    Args:
        params (dict): A (potentially nested) dictionary of parameters to hash.
        length (int): Length of the returned hash string.

    Returns:
        str: An order-invariant hash string.
    """

    def make_order_invariant(obj):
        if isinstance(obj, dict):
            # Sort dictionary keys and recursively process values
            return {k: make_order_invariant(obj[k]) for k in sorted(obj)}
        elif isinstance(obj, list):
            # Process list items and sort them to ensure order invariance
            return sorted(make_order_invariant(x) for x in obj)
        elif isinstance(obj, (np.integer, np.floating)):
            # Convert NumPy types to native Python types
            return obj.item()
        else:
            return obj

    params = copy.deepcopy(params)  # Create a deep copy to avoid modifying the original
    # Remove specific keys that are not needed for hashing
    list_keys = ['model.input_dim', 'model.active_neurons', 'model.neuron_mask', 'model.neuron_filter', 'model.n_active_cells', 'model.cell_indices', 'model.neuron_indices']
    for key in list_keys:
        if key in params: params.pop(key, None)  # Remove the key if it exists

    ordered_params = make_order_invariant(params)
    # Use json.dumps with sort_keys=True and no whitespace for consistent string representation
    params_str = json.dumps(ordered_params, separators=(',', ':'), sort_keys=True)
    params_hash = hashlib.md5(params_str.encode()).hexdigest()[:length]
    return params_hash


def hash_config(config):
    """
    Generate a hash for the given configuration,
    including only dicts with simple types as values and ignoring 'hash' key.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        str: Short hash string.
    """
    copy_config = copy.deepcopy(config)

    def is_simple_type(v):
        return isinstance(v, (int, float, str, bool, type(None), list, tuple))

    def filter_simple_dict(d):
        """
        Recursively filter dicts to include only:
        - Keys != 'hash'
        - Values of simple types or nested dicts that pass the same filter
        """
        filtered = {}
        for k, v in d.items():
            if k == 'hash':
                continue  # skip 'hash' key
            if isinstance(v, dict):
                nested_filtered = filter_simple_dict(v)
                if nested_filtered:  # Only include non-empty dicts
                    filtered[k] = nested_filtered
            elif is_simple_type(v):
                filtered[k] = v
            # Otherwise ignore (lists, custom objects, models, runs, etc.)
        return filtered

    filtered_config = filter_simple_dict(copy_config)

    # Convert to order-invariant string representation
    config_str = yaml.dump(filtered_config, default_flow_style=False, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

    return config_hash


def get_params_from_string(params_str):
    """
    Parses a string of parameters in key=value format and returns a dictionary.

    Args:
        params_str (str): A string containing parameters formatted as "key=value_key2=value2...",
                          where each key-value pair is separated by an underscore ('_').

    Returns:
        dict: A dictionary where each key is a parameter name and each value is the corresponding parameter value.
    """
    params = {}
    while params_str:
        # Partition the string into key, '=', and the rest
        key, sep, remainder = params_str.partition('=')
        if not sep:  # No '=' found, stop the loop
            break
        # Replace underscores in the key with spaces
        #key = key.replace('_', ' ')

        # Partition the remainder to get the value and the rest after '_'
        value, sep, params_str = remainder.partition('_')

        # Store the key-value pair in the dictionary
        params[key] = value
    return params


def simplify_params(params):
    """
    Simplifies parameter keys for easier display in formatted output.

    Args:
        params (dict): Original parameter dictionary with nested keys.

    Returns:
        dict: Simplified parameter dictionary with shortened keys.
    """
    simplified_params = {}
    for key, value in params.items():
        # Handle specific cases like 'model.name' and 'training.room'
        if key == 'model.name':
            simplified_params['*model'] = value
        elif key == 'training.room':
            simplified_params['room'] = value
        elif key == 'model.optimizer.learning_rate':
            simplified_params['learning_rate'] = value
        # Remove 'model.' prefix for other 'model.' keys
        elif key.startswith('model.'):
            simplified_params[key.replace('model.', '', 1).replace('.', '_')] = value
        # Replace any remaining '.' with '_'
        else:
            simplified_params[key.replace('.', '_')] = value
    return simplified_params


def format_params_string(params, format_type="file_name", base_name=None):
    """
    Formats a dictionary of parameters into a string based on the specified format.

    Args:
        params (dict): Dictionary of parameters to format.
        format_type (str): Specifies the format type ('full', 'title_list', 'file_list', or 'values').
        base_name (str, optional): An optional base name to prepend for file_name format.

    Returns:
        str: Formatted string with parameters in the specified format.
    """
    if not params:
        return base_name or ""

    # For compact use concise _-separated key=value format
    if format_type == "full":
        #short_params = {k: params[k] for k in params.keys() if not k.startswith('*')}
        #param_str = '_'.join(f"{value}" if key.startswith('*') else f"{key}={value}" for key, value in params.items())
        param_str = '_'.join(f"{key}={value}" for key, value in params.items())
        return f"{param_str}"

    # For title_list, create a more readable, comma-separated list
    elif format_type == "title_list":
        params = simplify_params(params) # Simplify parameter keys
        if 'learning_rate' in params: params['learning_rate'] = f"{float(params['learning_rate']):.1e}"
        param_str = ', '.join(f"{key}: {value}" if not key.startswith('*') else f"{value}" for key, value in params.items())
        # Break into multiple lines for readability, 3 items per line
        return '\n'.join([', '.join(param_str.split(', ')[i:i + 3]) for i in range(0, len(param_str.split(', ')), 3)])

    # For file_name format, append the parameters to the base name and keep file extension
    elif format_type == "file_name":
        params = simplify_params(params)  # Simplify parameter keys
        if 'learning_rate' in params: params['learning_rate'] = f"{float(params['learning_rate']):.1e}"
        # Separate base name and extension if base_name is provided, otherwise use empty string
        name, ext = os.path.splitext(base_name) if base_name else ("", "")
        param_str = '_'.join(f"{key}={value}" if not key.startswith('*') else f"{value}" for key, value in params.items())
        return f"{name}_{param_str}{ext}".strip("_")  # Remove any leading/trailing underscores

    elif format_type == "values":
        param_values = [str(v) for v in params.values()]
        param_str = '_'.join(param_values)
        name, ext = os.path.splitext(base_name) if base_name else ("", "")
        # add the base name if provided
        param_str = f"{name}_{param_str}{ext}"
        return param_str

    else:
        raise ValueError(f"Unsupported format_type: {format_type}. Supported types are 'file_name', 'title_list', 'file_list'.")

def covert_params_list_values_to_string(params):
    """
    Convert all list/tuple values in a dictionary to strings, including nested dictionaries.
    This is useful for ensuring that parameters can be easily serialized or logged.
    Args:
        params (dict): A dictionary containing parameters, potentially with list or tuple values.
    Returns:
        dict: A new dictionary with all list/tuple values converted to strings.
    """

    # go over and dict (and sub dict) values convert all list/tuple values to string
    def convert_value(value):
        if isinstance(value, (list, tuple)):
            return str(value)
        elif isinstance(value, dict):
            return {k: convert_value(v) for k, v in value.items()}
        return value

    # Convert all values in the params dictionary
    converted_params = {k: convert_value(v) for k, v in params.items()}
    return converted_params


def format_file_name_with_params(config, format_type="file_name", base_name='', use_hash_params=False):
    """
    Format a file name with parameters from the configuration.

    Args:
        config (dict): Configuration dictionary containing model and training settings.
        format_type (str): Specifies the format type ('file_name', 'title_list', or 'values').
        base_name (str, optional): An optional base name to prepend for file_name format.
        use_hash_params (bool): If True, hashes the parameters instead of using their string representation.

    Returns:
        str: Formatted file name with parameters in the specified format.
    """
    params = config['grid_params'] if 'grid_params' in config else get_params_from_config(config)
    if use_hash_params:
        hash_params = config.get('hash', {}).get('params', {})
        hash = hash_params if hash_params else hash_params(params)
    params = simplify_params(params)

    if use_hash_params:
        name, ext = os.path.splitext(base_name) if base_name else ("", "")
        # add the base name if provided
        file_name = f"{name}_{hash}{ext}"
        return file_name
    else:
        return format_params_string(params=params, format_type=format_type, base_name=base_name)

def get_checkpoint_from_config(config, base_checkpoint_name='ckpt.pth', old_checkpoint_name=False):
    """
    Generate a checkpoint filename by appending relevant parameters from the configuration to the base checkpoint name.

    Args:
    - config (dict): A dictionary containing configuration data for the model, training, and evaluation settings.
    - base_checkpoint_name (str): The base name of the checkpoint file.
    - old_checkpoint_name (bool): If True, uses the old naming convention for the checkpoint file. If False, uses the new format.

    Returns:
    - str: The checkpoint filename with appended parameters.
    """

    if old_checkpoint_name:
        # Get parameters from the configuration
        params = get_params_from_config(config)

        # Generate the full checkpoint name with the appended parameters
        checkpoint_with_params = format_params_string(params=params, format_type="file_name", base_name=base_checkpoint_name)

        return checkpoint_with_params

    checkpoint_with_params = format_file_name_with_params(config, base_name=base_checkpoint_name, use_hash_params=True)

    return checkpoint_with_params


def get_checkpoint_paths(config, checkpoint_name=None, return_loss_csv=False, params_dict=None, **kwargs):
    """
    Get the paths for the checkpoint file and its associated loss CSV file.

    Parameters:
        config (dict): Configuration dictionary containing session metadata and paths information.
        checkpoint_name (str, optional): Name of the checkpoint file. If not provided,
                                         it will be loaded from config['model']['checkpoint_name'].
        return_loss_csv (bool, optional): If True, returns the loss CSV file path as well.
                                     Default is False.
        params_dict (dict, optional): Dictionary of parameters to format into checkpoint name.
                                     Parameters will be formatted using format_params_string.
        **kwargs: Additional parameters to format into checkpoint name. These will be merged
                 with params_dict if provided.

    Returns:
        tuple or str: A tuple containing the full path to the checkpoint file. If `return_loss_csv`
                      is True, it also includes the loss CSV file path. Otherwise, just the
                      checkpoint path is returned.

    Raises:
        KeyError: If neither `checkpoint_name` is provided nor `config['model']['checkpoint_name']` exists.
        FileNotFoundError: If the checkpoint file or the loss CSV file (if `return_csv` is True) does not exist.
    """
    # Check if checkpoint_name is None and load from config
    if checkpoint_name is None:
        try:
            checkpoint_name = config['model']['checkpoint_name']
        except KeyError:
            raise KeyError("Checkpoint name is not provided and does not exist in the config.")

    # Merge params_dict and kwargs into a single params dict, filtering out None values
    params = {}
    if params_dict:
        params.update({k: v for k, v in params_dict.items() if v is not None})
    if kwargs:
        params.update({k: v for k, v in kwargs.items() if v is not None})
    
    # Format checkpoint name with parameters if any are provided
    if params:
        checkpoint_name = format_params_string(params=params, format_type='file_name', base_name=checkpoint_name)

    # Get the checkpoint directory using the provided function
    checkpoint_dir = get_directory(config, 'checkpoint', long_path=False)

    # Define the full path for the checkpoint file
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    # # Check if the checkpoint file exists
    # if not os.path.isfile(checkpoint_path):
    #     raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' does not exist.")

    # Only define the csv_loss_file if return_loss_csv is True
    if return_loss_csv:
        # Ensure CSV filename includes fold and offset from checkpoint_name
        # Replace .pth extension with _loss.csv, or append _loss.csv if no extension
        if checkpoint_name.endswith('.pth'):
            csv_loss_file = os.path.join(checkpoint_dir, checkpoint_name.replace('.pth', '_loss.csv'))
        else:
            csv_loss_file = os.path.join(checkpoint_dir, f"{checkpoint_name}_loss.csv")
        
        # # Check if the CSV loss file exists
        # if not os.path.isfile(csv_loss_file):
        #     raise FileNotFoundError(f"CSV loss file '{csv_loss_file}' does not exist.")
        
        return checkpoint_path, csv_loss_file
    
    return checkpoint_path


def check_all_checkpoints_exist(config):
    """
    Check if all checkpoints for all fold-offset combinations exist.

    Args:
        config (dict): Configuration dictionary containing training settings.

    Returns:
        bool: True if all checkpoints exist, False otherwise.
    """
    k_folds = config['training'].get('cross_validation_folds', 0)
    checkpoint_name = get_checkpoint_from_config(config)
    
    # Get offset list from config
    offset_list, _ = get_fold_offset_range(config)

    for fold in range(1, k_folds + 1):
        for offset in offset_list:
            # Get the full checkpoint path with fold and offset parameters
            checkpoint_path = get_checkpoint_paths(config, checkpoint_name=checkpoint_name, fold=f'{fold}@{k_folds}', offset_B=offset)

            # Check if the checkpoint file exists
            if not os.path.isfile(checkpoint_path):
                return False  # As soon as one checkpoint is missing, return False

    return True  # All checkpoints exist




def save_config_to_file(config, output_dir=None, base_name='config.yaml', full_path=None, keys_to_ignore=None):
    """
    Save the provided configuration to a YAML file without special characters like &id001.

    Args:
        config (dict): Configuration dictionary to save.
        output_dir (str, optional): Directory where the configuration file will be saved.
        base_name (str): Base name for the configuration file. Defaults to 'config.yaml'.
        full_path (str, optional): Full path to the configuration file. If provided, output_dir and base_name are ignored.
        keys_to_ignore (list, optional): List of keys to ignore when saving the configuration.

    Returns:
        str: Path to the saved configuration file.
    """

    # Define a Dumper class that ignores aliases to avoid &id001 etc.
    class NoAliasDumper(yaml.SafeDumper):
        def ignore_aliases(self, data):
            return True

    if full_path is not None:
        # If full_path is provided, use it directly
        output_dir = os.path.dirname(full_path)
        config_file_name = os.path.basename(full_path)
    else:
        config_file_name = format_file_name_with_params(config, base_name=base_name, use_hash_params=True)

    if output_dir is None:
        output_dir = get_directory(config, 'output', long_path=False)
        output_dir = os.path.join(output_dir, 'configs')

    os.makedirs(output_dir, exist_ok=True)

    config_file_path = os.path.join(output_dir, config_file_name)

    temp_config = copy.deepcopy(config)

    # Preprocess the config to ensure all values are serializable
    def make_serializable(data):
        if isinstance(data, dict):
            return {key: make_serializable(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [make_serializable(item) for item in data]
        elif isinstance(data, (int, float, str, bool, type(None))):
            return data
        else:
            return str(data)  # Convert unsupported types to strings

    temp_config = make_serializable(temp_config)

    if keys_to_ignore:
        for flat_key in keys_to_ignore:
            keys = flat_key.split(".")
            d = temp_config
            for key in keys[:-1]:
                d = d.setdefault(key, {})
            d.pop(keys[-1], None)

    with open(config_file_path, 'w') as file:
        yaml.dump(temp_config, file, default_flow_style=False, Dumper=NoAliasDumper)

    return config_file_path

def modify_and_save_config(config_path: str, user_config: dict):
    """
    Loads a config file, updates it with the user_config dictionary, and saves it back to the same location.

    Args:
        config_path (str): Path to the existing config YAML file.
        user_config (dict): Dictionary with values to update in the config.
    """
    # Load and update
    config = load_config(config_path, load_default=False, load_local_config = False)
    updated_config = update_config(config, user_config)

    # Extract original directory and overwrite the same file
    output_dir = os.path.dirname(config_path)
    base_name = os.path.basename(config_path)

    save_config_to_file(updated_config, output_dir=output_dir, base_name=base_name)

    return config_path

def check_skip_permutation(params, dir):
    """
    Check if a hyperparameter permutation should be skipped based on prior runs.

    Args:
        params (dict): Dictionary of hyperparameters.

    Returns:
        bool: True if the permutation is already in the skip list, False otherwise.
    """
    param_str = format_params_string(params=params, format_type="full")
    # logger.info(f"Checking if permutation {param_str} should be skipped...")
    # file path
    skip_list_file = os.path.join(dir, SKIP_LIST_FILE)

    # Ensure the file exists
    if not os.path.exists(skip_list_file):
        return False  # No skipped permutations yet

    # Check if the permutation exists in the skip list
    with open(skip_list_file, "r") as f:
        skipped_permutations = set([line.split('|')[0].strip() for line in f.readlines()])

    return param_str in skipped_permutations

def add_to_skip_list(params, dir, r2=-1):
    """
    Add a hyperparameter permutation to the skip list.

    Args:
        params (dict): Dictionary of hyperparameters.
    """
    param_str = format_params_string(params=params, format_type="full")
    line = f"{param_str}|{r2:0.2%}|{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    # file path
    skip_list_file = os.path.join(dir, SKIP_LIST_FILE)

    with open(skip_list_file, "a") as f:
        f.write(line + "\n")




