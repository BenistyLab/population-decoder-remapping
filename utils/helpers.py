import argparse
import platform

import pandas as pd
import numpy as np
from itertools import product

import torch
from torch import optim
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
import os

from .logger import get_logger
import re
import os
from pathlib import Path

import glob
import traceback
from tqdm import tqdm

# Initialize logger
logger = get_logger(__name__)

def get_directory(config: dict, dir_type: str, long_path=False) -> str:
    """
    Get the directory path based on the provided configuration and directory type.

    Args:
        config (dict): Configuration dictionary containing session metadata and paths information.
        dir_type (str): Type of directory to return ('logs', 'checkpoint', 'output', or 'data').
        long_path (bool): Whether to return the full path or just the directory name. Defaults to False

    Returns:
        str: The path to the requested directory.

    Raises:
        KeyError: If the directory type is not found in the configuration.
    """

    # Extract directory path from config
    try:
        if 'directories' in config and isinstance(config['directories'], dict) and dir_type in config['directories']:
            base_dir = config['directories'][dir_type]
        else:
            raise KeyError(f"Directory type '{dir_type}' not found or 'directories' is not properly defined in configuration.")
    except KeyError:
        raise KeyError(f"Directory type '{dir_type}' not found in configuration.")

    metadata = config.get('metadata', {}) or {}
    session_id = metadata.get('session', '')
    session_group = metadata.get('group', '')  # Cell type: All_Cells, Allo_Cell, etc.
    session_version = metadata.get('version', '')  # General version run
    session_subdir = metadata.get('subdir', '')  # Subdirectory

    # Placeholder-aware directory templates:
    # <session>, <group>, <version>, <subdir>
    placeholders = {
        '<session>': session_id,
        '<group>': session_group,
        '<version>': session_version,
        '<subdir>': session_subdir,
    }

    dir_template = str(base_dir)
    has_placeholder = any(token in dir_template for token in placeholders)
    if has_placeholder:
        dir_path = dir_template
        for token, value in placeholders.items():
            dir_path = dir_path.replace(token, str(value or '').strip())
        # Normalize any empty-segment artifacts from optional placeholders.
        dir_path = os.path.normpath(dir_path)
    else:
        # Backward-compatible legacy behavior when no placeholders are used.
        dir_parts = [base_dir]
        if dir_type == 'data':
            dir_parts.append(session_id)
        elif dir_type == 'checkpoint':
            dir_parts.append(session_id)
            dir_parts.append('checkpoints')
        elif dir_type == 'output':
            dir_parts.append(session_id)
        elif dir_type == 'log':
            dir_parts.append(session_id)
        else:
            if session_version:
                dir_parts.append(session_version)
            dir_parts.append(session_id)
            if session_group:
                dir_parts.append(session_group)
            if session_subdir:
                dir_parts.append(session_subdir)
        dir_path = os.path.join(*dir_parts)

    # Ensure the directory exists
    os.makedirs(dir_path, exist_ok=True)

    # Handle long path for Windows
    if platform.system() == "Windows":
        if long_path:
            dir_path = r'\\?\\' + os.path.abspath(dir_path).replace('/', '\\')
        else:
            dir_path = os.path.abspath(dir_path).replace('/', '\\')
    else:
        # Normalize the path for Linux/macOS
        dir_path = os.path.abspath(dir_path)

    # return Path(dir_path)
    return dir_path

def get_dataset_path(config: dict, dataset_name: str = 'main') -> str:
    """
    Get the dataset path based on the provided configuration and dataset name.

    Args:
        config (dict): Configuration dictionary containing session metadata and paths information.
        dataset_name (str): Name of the dataset to return ('clusters', 'spike_rates', 'positions', or 'main').
                            Defaults to 'main'.

    Returns:
        str: The full path to the requested dataset.

    Raises:
        KeyError: If the dataset name is not found in the configuration.
    """

    # Extract dataset paths from config
    try:
        dataset_file = config['datasets'][dataset_name]
    except KeyError:
        raise KeyError(f"Dataset name '{dataset_name}' not found in configuration.")

    # Construct the full path to the dataset
    data_dir = get_directory(config,'data')
    dataset_path = os.path.join(data_dir, dataset_file)

    return dataset_path


def get_full_path_to_save_file(config, export_path):
    """
    Get the full path to save a file in the output directory, ensuring the directory exists.

    Parameters:
        config (dict): Configuration dictionary containing session metadata and paths information.
        export_path (str): Relative path to the file to be saved within the output directory.

    Returns:
        str: The full path to save the file.
    """
    output_dir = get_directory(config, 'output', long_path=False)

    full_path = os.path.abspath(os.path.join(output_dir, export_path))

    # Ensure the directory for the full path exists
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    return full_path

def generate_param_grid(param_grid):
    """
    Generate all permutations of hyperparameters.
    """
    keys, values = zip(*param_grid.items())
    permutations = [dict(zip(keys, v)) for v in product(*values)]
    return permutations


def custom_k_fold_split(data, num_folds, ):
    """
    Custom split the data into num_folds for cross-validation with train-validation-test setup.
    """
    data_blocks = np.array_split(data, num_folds+2)
    splits = []

    for fold in range(1, num_folds + 1):
        train_val_blocks = data_blocks[:fold+1]
        test_block = data_blocks[fold+1:]
        if len(test_block) == 0:
            test_block = data_blocks[-1:]
            train_val_blocks = data_blocks[:-1]

        train_val_indices = np.concatenate([block.index for block in train_val_blocks])
        test_indices = test_block[0].index

        sub_folds = []
        sub_kf = KFold(n_splits=len(train_val_blocks), shuffle=False) #, random_state=42)
        for sub_train_val_indices, sub_val_indices in sub_kf.split(train_val_indices):
            sub_train_indices = train_val_indices[sub_train_val_indices]
            sub_val_indices = train_val_indices[sub_val_indices]
            sub_folds.append((sub_train_indices, sub_val_indices, test_indices))

        splits.append(sub_folds)

    return splits



def k_fold_split(data, k, scenario='train_val_test'):
    """
    Custom split the data into num_folds for cross-validation based on the scenario.

    Scenarios:
    - 'train_val_test': train, validation, and test sets.
    - 'train_val': only train and validation sets.
    - 'train_test': only train and test sets.
    """
    data_blocks = np.array_split(data, k)
    splits = []

    for fold in range(k):
        if scenario == 'train_val_test':
            if fold + 2 >= k: break
            train_val_blocks = data_blocks[:fold + 2]
            test_block = data_blocks[fold + 2:]
        elif scenario == 'train_val':
            if fold + 1 >= k: break
            train_val_blocks = data_blocks[:fold + 2]
            test_block = []
        elif scenario == 'train_test':
            if fold + 1 >= k: break
            train_val_blocks = data_blocks[:fold+1]
            test_block = data_blocks[fold+1:]

        train_val_indices = np.concatenate([block.index for block in train_val_blocks])
        test_indices = test_block[0].index if scenario != 'train_val' else None

        sub_folds = []
        if scenario == 'train_val_test' or scenario == 'train_val':
            sub_kf = KFold(n_splits=len(train_val_blocks), shuffle=False)
            for sub_train_val_indices_train, sub_train_val_indices_val in sub_kf.split(train_val_indices):
                sub_train_indices = train_val_indices[sub_train_val_indices_train]
                sub_val_indices = train_val_indices[sub_train_val_indices_val]
                sub_folds.append((sub_train_indices, sub_val_indices, test_indices))
        elif scenario == 'train_test':
            sub_train_indices = train_val_indices
            sub_val_indices = None
            sub_folds.append((sub_train_indices, sub_val_indices, test_indices))

        splits.append(sub_folds)

    return splits



def get_data_by_timestamp_range(df, timestamp_range):
    start, end = timestamp_range
    return df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]



# def get_data_from_room(df, rooms_to_indices, room, event):
#     # return get_data_by_timestamp_range(full_data, room_split[room][incident]).copy()
#     timestamp_range = rooms_to_indices[room][event]
#     start, end = timestamp_range
#     return df[(df['timestamp'] >= start) & (df['timestamp'] <= end)].copy()

def flatten_dict_to_filename(d, separator='_'):
    """
    Flatten dictionary key and value pairs into a string separated by a specified character.

    :param d: Dictionary to flatten
    :param separator: Character to use as separator between key-value pairs
    :return: Flattened string
    """
    parts = []
    for key, value in d.items():
        parts.append(f"{key}{separator}{value}")
    return separator.join(parts)


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

def get_prediction_columns(columns, suffix='pred'):
    return [f"{col}_{suffix}" for col in columns]


def get_prediction_columns_for_room(target_columns, room_label, available_columns):
    """
    Get prediction columns for a specific room, preferring room-specific columns over standard ones.
    
    Args:
        target_columns: List of target column names (e.g., ['X', 'Y'])
        room_label: Room label (e.g., 'A', 'B', 'a')
        available_columns: Set or list of available column names in the DataFrame
    
    Returns:
        List of prediction column names for the specified room.
        Prefers room-specific format (e.g., 'X_pred_A') over standard format (e.g., 'X_pred').
    """
    pred_cols = []
    available_set = set(available_columns) if not isinstance(available_columns, set) else available_columns
    
    for tcol in target_columns:
        room_specific_col = f'{tcol}_pred_{room_label}'
        standard_col = f'{tcol}_pred'
        
        if room_specific_col in available_set:
            pred_cols.append(room_specific_col)
        elif standard_col in available_set:
            pred_cols.append(standard_col)
        # If neither exists, skip (caller can handle warning if needed)
    
    return pred_cols


def build_ensemble_data(df_data, target_columns, pred_columns_list, logger=None):
    """
    Build ensemble DataFrame (one row per timestamp) by aggregating offsets when present.

    When 'offset' exists and multiple offsets exist, aggregates by timestamp (mean for
    target/pred columns, first for others). Otherwise sets index to timestamp.

    Args:
        df_data: DataFrame with optional 'offset' and 'timestamp' columns
        target_columns: List of target column names
        pred_columns_list: List of prediction column names
        logger: Optional logger for aggregation messages

    Returns:
        pd.DataFrame: Indexed by 'timestamp'
    """
    if 'offset' in df_data.columns:
        n_offsets = df_data['offset'].nunique()
        if n_offsets > 1:
            if logger is not None:
                logger.info(f'Aggregating predictions across {n_offsets} offsets using mean...')
            agg_dict = {}
            for col in target_columns + pred_columns_list:
                if col in df_data.columns:
                    agg_dict[col] = 'mean'
            for col in df_data.columns:
                if col not in agg_dict and col not in ['timestamp', 'offset']:
                    agg_dict[col] = 'first'
            out = df_data.groupby('timestamp').agg(agg_dict).reset_index()
            out = out.set_index('timestamp')
            if logger is not None:
                logger.info(f'Aggregated data from {n_offsets} offsets per timestamp to single row per timestamp.')
            return out
        else:
            if logger is not None:
                logger.info(f'Only one offset found, skipping aggregation.')
            return df_data.set_index('timestamp')
    return df_data.set_index('timestamp') if 'timestamp' in df_data.columns else df_data


def save_data_to_csv(config, df_stats, output_file='stats.csv', overwrite=False):
    output_dir = get_directory(config, 'output')
    output_file = os.path.join(output_dir, output_file) #output_file = os.path.join(output_dir, '../', 'stats.csv')
    # create folder if not exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Save the DataFrame to a CSV file
    if not os.path.isfile(output_file) or overwrite:
        df_stats.to_csv(output_file, mode='w', header=True, index=False, encoding='utf-8')
    else:
        df_stats.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8')
    
    logger.info(f'Saved data to {output_file}')


def rename_files_in_directory(root_dir):
    # Traverse through the directory and its subdirectories
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # Check for specific file extensions
            if (filename.endswith('.pth') or filename.endswith('.csv')) and False:  # Removed STG_MLP check# and filename[len('ckpt_FR2XY2XY_room=')+1]=='-':
                #new_filename = filename.replace("lr=", "learning_rate=").replace("rooms=", "room=") # need to be change
                #new_filename = filename.replace("ckpt_SR2XY", "ckpt_FR2XY").replace("learning_rate=0.1", "learning_rate=1.0e-01").replace("learning_rate=0.01", "learning_rate=1.0e-02").replace("learning_rate=0.001", "learning_rate=1.0e-03").replace("learning_rate=0.0001", "learning_rate=1.0e-04").replace("learning_rate=0.00001", "learning_rate=1.0e-05").replace("learning_rate=0.000001", "learning_rate=1.0e-06")
                #new_filename = filename.replace("_room=prev_room=", "_room=")
                # pattern = r'(_FR2XY2XY)(_room=)([^-]+)-([^-]+)(_.*)'
                # new_filename = re.sub(pattern, r'\1_prev_room=\3\2\4\5', filename)
                #new_filename = filename.replace("ckpt_FR2XY2XY_room=A", "ckpt_FR2XY2XY_prev_room=B_room=A").replace("ckpt_FR2XY2XY_room=B", "ckpt_FR2XY2XY_prev_room=A_room=B")
                new_filename = filename.replace("_stg_lam=1.0_", "_stg_lam=1_")

                # Only rename if the new filename is different
                if new_filename != filename:

                    original_file_path = os.path.join(dirpath, filename)
                    new_file_path = os.path.join(dirpath, new_filename)

                    if os.path.isfile(new_file_path):
                        print(f'Cannot rename a file when that file name already exists: {new_filename}')
                        continue

                    # Rename the file
                    os.rename(original_file_path, new_file_path)
                    print(f'Renamed: {original_file_path} -> {new_file_path}')



def get_device():
    """
    Returns the device (GPU or CPU) based on availability.

    Returns:
        torch.device: The available device ('cuda:0' if GPU is available, otherwise 'cpu').
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def log_device_info(device=None):
    """
    Log diagnostic information about the device.
    
    This function provides detailed information about CUDA availability,
    device name, CUDA version, and memory information.

    Args:
        device (torch.device, optional): The device to log information about.
                                        If None, will get the device using get_device().
    """
    logger = get_logger(__name__)
    
    if device is None:
        device = get_device()
    
    logger.info(f"Using device: {device} (CUDA available: {torch.cuda.is_available()})")
    
    if torch.cuda.is_available():
        try:
            device_count = torch.cuda.device_count()
            if device_count > 0:
                device_name = torch.cuda.get_device_name(0)
                cuda_version = torch.version.cuda
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"GPU: {device_name}")
                logger.info(f"CUDA version: {cuda_version}")
                logger.info(f"GPU Memory: {memory_total:.2f} GB")
            else:
                logger.warning("CUDA is available but no devices found. Using CPU.")
        except Exception as e:
            logger.warning(f"Error getting CUDA device information: {e}")
    else:
        logger.warning("CUDA is not available. Using CPU device.")


def get_optimizer(optimizer, model, *args, **kwargs):
    """Return an optimizer instance, initializing it if necessary."""
    
    if isinstance(optimizer, optim.Optimizer):
        return optimizer
    elif isinstance(optimizer, str):
        try:
            optimizer_class = getattr(optim, optimizer)
        except AttributeError:
            raise ValueError(f"Unknown optimizer type: {optimizer}.")
        return optimizer_class(filter(lambda p: p.requires_grad, model.parameters()), **kwargs)
    else:
        raise ValueError(f"Optimizer must be a string or optim.Optimizer instance, got {type(optimizer)}.")




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


def extract_hyperparameters_from_string(hyperparameters_str):
    params = {}

    while hyperparameters_str:
        # Partition the string into key, '=', and the rest
        key, sep, remainder = hyperparameters_str.partition('=')
        if not sep:  # No '=' found, stop the loop
            break
        # Replace underscores in the key with spaces
        # key = key.replace('_', ' ')

        # Partition the remainder to get the value and the rest after '_'
        value, sep, hyperparameters_str = remainder.partition('_')

        # Store the key-value pair in the dictionary
        params[key] = value

    return params


def find_stats_files(base_folder, file_name='stats.csv', folder_name_filter=[]):
    """
    Finds all stats.csv files in the specified folder and its subdirectories.

    Parameters:
        base_folder (str): The folder to search for stats.csv files.

    Returns:
        list: List of file paths for all stats.csv files found.
    """
    stats_files = []
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file == file_name:
                stats_files.append(os.path.join(root, file))

    if folder_name_filter:
        # Filter the stats_files based on the last folder name
        stats_files = [file for file in stats_files if os.path.basename(os.path.dirname(file)) in folder_name_filter]

    return stats_files


def load_and_combine_stats_files(stats_files):
    """
    Loads and combines multiple stats.csv files into a single DataFrame.

    Parameters:
        stats_files (list): List of file paths for the stats.csv files.

    Returns:
        pd.DataFrame: A combined DataFrame containing data from all the input files.
    """
    combined_df = pd.DataFrame()  # Initialize an empty DataFrame

    for file_path in stats_files:
        try:
            # Load each file into a DataFrame
            df = pd.read_csv(file_path)
            # Append to the combined DataFrame
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")

    return combined_df


def mean_metrics(df, metric_cols=None):
    """
    Computes column-wise mean for numeric columns, element-wise mean for list columns,
    and recursively computes key-wise mean for dictionary columns in a DataFrame.

    Parameters:
        df (DataFrame): The input DataFrame, where some columns may contain lists or dictionaries.
        metric_cols (list, optional): List of columns to compute the mean for. If None, computes for all columns.

    Returns:
        Series: A Series containing the mean of each column.
    """

    def compute_mean(column):

        # check if the column is a string representation of a list
        if column.apply(lambda x: isinstance(x, str)).all() and column.apply(lambda x: '[' in x and ']' in x).any():
            # first replace all ' ' with ',' between brackets
            column = column.apply(lambda x: re.sub(r'(\[.*?])', lambda m: m.group(0).replace(' ', ','), x))
            # make sure to remove all duplicate commas
            column = column.apply(lambda x: re.sub(r',+', ',', x))
            # convert string representation of list to actual list
            column = column.apply(lambda x: eval(x) if isinstance(x, str) else x)

        if pd.api.types.is_numeric_dtype(column):
            # For numeric columns, return the mean
            return column.mean()
        elif column.apply(lambda x: isinstance(x, (list, np.ndarray))).all():
            # For list columns, compute element-wise mean
            return [np.mean(values) for values in zip(*column)]
        elif column.apply(lambda x: isinstance(x, dict)).all():
            # For dictionary columns, recursively compute means for each key
            all_keys = set().union(*column)  # Collect all keys
            key_means = {}
            for key in all_keys:
                values = [d[key] for d in column if key in d]
                key_means[key] = compute_mean(pd.Series(values))
            return key_means
        # elif column.apply(lambda x: isinstance(x, list)).all():
        #     # For list columns, compute element-wise mean
        #     return [np.mean(values) for values in zip(*column)]

        else:
            # For unsupported column types, return NaN
            return np.nan

    if metric_cols is not None:
        # Filter the DataFrame to include only the specified metric columns
        df = df[metric_cols]

    # Apply the compute_mean function to each column
    return df.apply(compute_mean)


def groupby_mean_mixed_values(df, group_cols, value_col = 'value', na_placeholder='unknown'):
    """
    General replacement for groupby(...).mean() that handles both scalar and tuple/list values of any length.

    Parameters:
    - df: Input DataFrame
    - group_cols: Columns to group by
    - value_col: Name of the value column to average (default: 'value')
    - na_placeholder: Value to replace NaNs in group columns (default: 'unknown')

    Returns:
    - DataFrame with group_cols + [value_col], where value_col contains mean scalar or mean tuple/list
    """

    def is_scalar(x):
        return isinstance(x, (int, float, np.number)) and not pd.isna(x)

    def is_sequence(x):
        return isinstance(x, (tuple, list, np.ndarray)) and all(isinstance(i, (int, float, np.number)) for i in x)

    # Replace NaNs in group_cols
    df_clean = df.copy()
    for col in group_cols:
        df_clean[col] = df_clean[col].fillna(na_placeholder)

    scalar_mask = df_clean[value_col].apply(is_scalar)
    sequence_mask = df_clean[value_col].apply(is_sequence)

    scalar_df = df_clean[scalar_mask].copy()
    sequence_df = df_clean[sequence_mask].copy()

    sequence_df[value_col] = sequence_df[value_col].apply(lambda x: tuple(x) if isinstance(x, list) or isinstance(x, np.ndarray) else x)

    # Scalar mean
    scalar_df[value_col] = scalar_df[value_col].astype(float)
    scalar_mean = scalar_df.groupby(group_cols, as_index=False)[value_col].mean()

    # Sequence mean
    if not sequence_df.empty:
        # Expand sequence into columns
        max_len = sequence_df[value_col].apply(len).max()
        expanded_cols = pd.DataFrame(sequence_df[value_col].tolist(), index=sequence_df.index)
        expanded_cols.columns = [f'__val_{i}' for i in range(max_len)]
        sequence_df = pd.concat([sequence_df[group_cols], expanded_cols], axis=1)

        # Group by and compute mean per component
        mean_cols = sequence_df.groupby(group_cols, as_index=False).mean()

        # Reassemble into tuple/list
        value_arrays = mean_cols[[col for col in mean_cols.columns if col.startswith('__val_')]].values.tolist()
        mean_cols[value_col] = [tuple(row) for row in value_arrays]

        # Keep only group_cols + value_col
        sequence_mean = mean_cols[group_cols + [value_col]]
    else:
        sequence_mean = pd.DataFrame(columns=group_cols + [value_col])

    # Combine scalar and sequence means
    combined = pd.concat([scalar_mean, sequence_mean], ignore_index=True)

    return combined


def convert_numpy_to_python(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, tuple):  # convert tuples to lists
        return [convert_numpy_to_python(i) for i in obj]
    else:
        return obj


def format_room_name(room):
    """
    Format room name(s) by standardizing aliases.
    
    Handles:
    - Strings: Standardizes and returns
    - Lists/tuples: Standardizes each element and returns formatted string
    - NaN/None: Returns None
    - Numeric types: Converts to string first
    - Other types: Returns None (graceful handling)
    """
    import pandas as pd
    import numpy as np
    
    def standardize_aliases(r):
        if not isinstance(r, str):
            return str(r) if r is not None else None
        if "'" in r.capitalize() == r:
            r = r.replace("'", "")
            r = r.lower()
        return r

    # Handle NaN/None values
    if room is None or (isinstance(room, float) and (np.isnan(room) or pd.isna(room))):
        return None
    
    # Handle strings
    if isinstance(room, str):
        return standardize_aliases(room)
    
    # Handle lists/tuples
    elif isinstance(room, (list, tuple)):
        cleaned = []
        for r in room:
            if r is None or (isinstance(r, float) and (np.isnan(r) or pd.isna(r))):
                continue
            cleaned.append(standardize_aliases(r))
        if cleaned:
            return "[" + ",".join(cleaned) + "]"
        else:
            return None
    
    # Handle numeric types - convert to string
    elif isinstance(room, (int, float)):
        return standardize_aliases(str(room))
    
    # For any other type, try to convert to string or return None
    else:
        try:
            return standardize_aliases(str(room))
        except (ValueError, TypeError):
            return None

def format_room_name_display(room):
    """Format a room name for display in plots, titles, and legends.

    Lowercase single-letter rooms get an uppercase + prime notation;
    uppercase and multi-character names are returned as-is.

    Mapping:
        'a'  → "A'"
        'b'  → "B'"
        'A'  → 'A'
        'B'  → 'B'
        'All' → 'All'
        None / NaN → None
    """
    import numpy as np
    import pandas as pd

    if room is None:
        return None
    if isinstance(room, float) and (np.isnan(room) or pd.isna(room)):
        return None
    if not isinstance(room, str):
        room = str(room)

    if len(room) == 1 and room.islower():
        return f"{room.upper()}'"
    return room


import re

def format_room_for_saving(room):
    """
    Format the room name for saving in a file:
    - If the room is a single lowercase letter, convert to 'p' + uppercase (e.g., 'a' → 'pA').
    - Otherwise, replace spaces with underscores and remove special characters.

    Args:
        room (str or list): Room name or list of room names to format.

    Returns:
        str: Formatted room name suitable for file naming.
    """

    def encode_string(s):
        if isinstance(s, str) and len(s) == 1 and s.islower():
            return f"{s.upper()}p"
        cleaned = re.sub(r'\W+', '_', s).replace(' ', '_')
        return cleaned

    if isinstance(room, str):
        return encode_string(room)
    elif isinstance(room, list):
        return '_'.join([encode_string(r) for r in room])
    else:
        raise ValueError("Input must be a string or a list of strings.")




def get_rooms_from_config(config):
    """
    Return the list of rooms to operate on. Uses data-derived rooms from preprocessing
    (rooms_list, room_indices, or rooms_index). If run.rooms is set, returns only
    rooms that are both in the data and in run.rooms (intersection), so the data
    config remains the source of truth and run.rooms is a filter.
    """
    if config.get('preprocessing', {}).get('map_rooms', {}).get('rooms_list', None) is not None:
        data_rooms = config['preprocessing']['map_rooms']['rooms_list']
    elif config.get('preprocessing', {}).get('room_indices', None) is not None:
        data_rooms = [key for key in config['preprocessing']['room_indices'].keys()]
    elif config.get('preprocessing', {}).get('rooms_index', None) is not None:
        data_rooms = [key for key in config['preprocessing']['rooms_index'].keys()]
    else:
        raise ValueError("No rooms defined in the configuration. Please check 'preprocessing.map_rooms', 'preprocessing.room_indices', or 'preprocessing.rooms_index'.")

    run_rooms = config.get('run', {}).get('rooms', None)
    if run_rooms and len(run_rooms) > 0:
        data_rooms = [r for r in data_rooms if r in run_rooms]
    return data_rooms

def get_time_range_from_config(config, room=None):
    if room is not None:
        rooms_info = config.get('preprocessing', {}).get('map_rooms', {}).get('rooms', None)
        room_indices = config.get('preprocessing', {}).get('room_indices', None)
        if rooms_info is not None:
            if room in rooms_info:
                room_range = rooms_info[room].get('range', [])
                return room_range
            else:
                logger.warning(f"Room '{room}' not found in 'preprocessing.map_rooms.rooms'.")
                return None
        elif room_indices is not None:
            if room in room_indices:
                return room_indices[room]
            else:
                logger.warning(f"Room '{room}' not found in 'preprocessing.room_indices'.")
                return None
        else:
            logger.warning("No rooms defined in the configuration. Please check 'preprocessing.map_rooms' or 'preprocessing.room_indices'.")
            return None
    else:
        # If no specific room is provided, return the overall time range
        rooms = get_rooms_from_config(config)
        overall_start = float('inf')
        overall_end = float('-inf')
        for r in rooms:
            r_range = get_time_range_from_config(config, r)
            if r_range and len(r_range) == 2:
                overall_start = min(overall_start, r_range[0])
                overall_end = max(overall_end, r_range[1])
        if overall_start == float('inf') or overall_end == float('-inf'):
            logger.warning("Could not determine overall time range from the configuration.")
            return None
        return [overall_start, overall_end]


def matrix_to_string(array: np.ndarray) -> str:
    """
    Serialize a NumPy array of any dimension into a compact string.
    Uses nested brackets: [1,2,3], [[1,2],[3,4]], etc.

    Args:
        array (np.ndarray): The NumPy array.

    Returns:
        str: Text representation.
    """
    return np.array2string(array, separator=',', max_line_width=np.inf).replace('\n', '')


def string_to_matrix(text: str) -> np.ndarray:
    """
    Deserialize a compact string into a NumPy array of any dimension.

    Args:
        text (str): The string, e.g. '[1,2,3]' or '[[1,2],[3,4]]'

    Returns:
        np.ndarray: Parsed NumPy array.
    """
    # Eval is safe here because the string was originally generated from array2string
    return np.array(eval(text))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def process_config_path(config_path, processor_func, use_tqdm=False):
    """
    Processes a configuration path, which can either be a single YAML file or a directory containing YAML files.

    Parameters:
        config_path (str): Path to a YAML config file or a directory containing YAML config files.
        processor_func (callable): A function to process each YAML file. It should accept a single argument (file path).
        use_tqdm (bool): Whether to use tqdm for progress indication when processing multiple files.
    Raises:
        Exception: If no valid YAML files are found or the path is invalid.
    """
    if os.path.isfile(config_path):
        print(f"Processing single config file: {config_path}...")
        try:
            processor_func(config_path)
        except Exception as e:
            print(f"Error processing config file {config_path}: {e}")
            print(f"{traceback.format_exc()}")
            print("Skipping this config file.")

    elif os.path.isdir(config_path):
        print(f"Processing all YAML files in folder: {config_path}...")
        config_files = glob.glob(os.path.join(config_path, '*.yaml'))
        total_configs = len(config_files)

        if total_configs == 0:
            raise Exception(f"No config files found in this directory: {config_path}")

        if use_tqdm:
            try:
                iterable = tqdm(config_files, desc="Processing configs", unit="file")
                writer = tqdm.write
                progress_wrapped = True
            except Exception:
                # Fall back to plain iterable/print if tqdm import fails
                iterable = config_files
                writer = print
                progress_wrapped = False
        else:
            iterable = config_files
            writer = print
            progress_wrapped = False

        for idx, config_file in enumerate(iterable, start=1):
            # When using tqdm wrapper, avoid printing per-iteration messages to prevent breaking the progress bar
            if not progress_wrapped:
                print(f"Processing {idx}/{total_configs}: {config_file}...")
            try:
                processor_func(config_file)
            except Exception as e:
                writer(f"Error processing config file {config_file}: {e}")
                writer(traceback.format_exc())
                writer("Skipping this config file.")


    else:
        raise Exception(f"{config_path} is neither a file nor a directory.")


def get_dt_from_config(config):
    resample_cfg = config.get('preprocessing', {}).get('resample', {})
    if resample_cfg.get('enabled', False):
        rate_hz = resample_cfg.get('rate_hz')
        return 1 / rate_hz if rate_hz else None
    return None


def collect_csv_files_by_pattern(base_dir, file_pattern, export_path, overwrite=False):
    """
    Recursively search for CSV files matching a regex pattern, aggregate them into a single DataFrame,
    and save to export_path. If export_path exists and overwrite=False, return cached path.
    
    Parameters:
    -----------
    base_dir : str
        Root directory to search recursively for CSV files
    file_pattern : str
        Regex pattern to match filenames (e.g., r'mapping_stats.*\.csv')
    export_path : str
        Path where aggregated CSV should be saved
    overwrite : bool, optional
        If True, recompute even if export_path exists (default: False)
    
    Returns:
    --------
    str
        Path to the aggregated CSV file (export_path)
    
    Raises:
    -------
    ValueError
        If no files matching the pattern are found
    """
    import re
    
    # Check if export_path exists and overwrite is False
    if os.path.exists(export_path) and not overwrite:
        logger.info(f"Using cached aggregated CSV: {export_path}")
        return export_path
    
    # Compile regex pattern
    pattern = re.compile(file_pattern)
    
    # Find all matching files
    matching_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if pattern.match(file):
                file_path = os.path.join(root, file)
                matching_files.append(file_path)
    
    if not matching_files:
        raise ValueError(f"No files matching pattern '{file_pattern}' found in {base_dir}")
    
    logger.info(f"Found {len(matching_files)} files matching pattern '{file_pattern}'")
    
    # Load and combine all CSV files
    dataframes = []
    failed_files = []
    
    for file_path in tqdm(matching_files, desc="Loading CSV files"):
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            # Add source file path as metadata column
            df['_source_file'] = file_path
            dataframes.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            failed_files.append(file_path)
    
    if not dataframes:
        raise ValueError(f"Failed to load any CSV files. {len(failed_files)} files failed to load.")
    
    if failed_files:
        logger.warning(f"{len(failed_files)} files failed to load out of {len(matching_files)} total files")
    
    # Concatenate all DataFrames
    logger.info(f"Combining {len(dataframes)} DataFrames...")
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    
    # Save aggregated CSV
    combined_df.to_csv(export_path, index=False, encoding='utf-8')
    logger.info(f"Saved aggregated CSV with {len(combined_df)} rows to {export_path}")
    
    return export_path


def apply_scaler_transform(data, scaler, scale_only=False, reverse=False):
    """
    Apply scaler transformation (or inverse transformation) to data.
    
    Assumes scaler normalizes jointly (single scale factor). Always reshapes data as column vector
    for scaler operations.
    
    Args:
        data (np.ndarray): Input data to transform. Can be 1D (single point/vector) or 2D (array of points).
        scaler: Scaler object (e.g., MinMaxScaler, StandardScaler, etc.) with 'transform'/'inverse_transform' methods.
                If None, returns data unchanged.
        scale_only (bool): If True, only apply scaling (from scale_ attribute) without translation.
                          If reverse=True, applies inverse scaling (divide by scale_).
                          Default False.
        reverse (bool): If True, apply inverse transformation (normalized -> physical).
                       If False, apply forward transformation (physical -> normalized).
                       Default False.
    
    Returns:
        np.ndarray: Transformed data with same shape as input.
    
    Examples:
        >>> # Convert normalized coordinates to cm (inverse transform)
        >>> data_cm = apply_scaler_transform(data_norm, scaler, reverse=True)
        
        >>> # Convert cm to normalized (forward transform)
        >>> data_norm = apply_scaler_transform(data_cm, scaler, reverse=False)
        
        >>> # Apply only forward scaling (no translation)
        >>> data_scaled = apply_scaler_transform(data, scaler, scale_only=True, reverse=False)
        
        >>> # Apply only inverse scaling (no translation)
        >>> data_unscaled = apply_scaler_transform(data, scaler, scale_only=True, reverse=True)
    """
    if scaler is None:
        return data
    
    # Convert scalar to 1D array for processing
    is_scalar = not isinstance(data, np.ndarray)
    if is_scalar:
        data = np.array([data])
    elif not isinstance(data, np.ndarray):
        return data
    
    # Get scale factor if available (assumes joint normalization)
    scale_factor = None
    if hasattr(scaler, 'scale_') and scaler.scale_.size > 0:
        scale_factor = float(scaler.scale_[0])
    
    # Handle NaN values - unified for all dimensions
    if data.ndim == 1:
        # For 1D, check NaNs directly
        nan_mask = np.isnan(data)
    else:
        # For 2D+, check NaNs along all axes except the first (rows)
        nan_mask = np.isnan(data).any(axis=tuple(range(1, data.ndim)))
    
    # Return unchanged if all NaNs
    if nan_mask.all():
        return data
    
    # Extract valid data and store original shape
    has_nans = nan_mask.any()
    if has_nans:
        result = data.copy()
        valid_data = data[~nan_mask]
    else:
        result = None
        valid_data = data
    
    if len(valid_data) == 0:
        return data if result is None else result
    
    # Store original shape for reshaping back
    original_shape = valid_data.shape
    
    # Apply transformation
    if scale_only:
        if scale_factor is not None:
            if reverse:
                # Inverse scaling: divide by scale factor
                transformed = valid_data / scale_factor
            else:
                # Forward scaling: multiply by scale factor
                transformed = valid_data * scale_factor
        else:
            transformed = valid_data
    else:
        # Full transformation: reshape to column vector and apply
        valid_reshaped = valid_data.reshape(-1, 1)
        transform_method = scaler.inverse_transform if reverse else scaler.transform
        transformed = transform_method(valid_reshaped)
        # Reshape back to original shape
        transformed = transformed.reshape(original_shape)
    
    # Update result
    if has_nans:
        result[~nan_mask] = transformed
        output = result
    else:
        output = transformed
    
    # Convert back to scalar if input was scalar
    if is_scalar:
        return output.item() if output.size == 1 else output
    else:
        return output