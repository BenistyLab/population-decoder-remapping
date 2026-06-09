import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.helpers import get_dataset_path, format_room_for_saving
from sklearn.model_selection import KFold
import warnings
from utils.logger import get_logger
from utils.config import get_target_columns_from_config



class TimeSeriesDataset(Dataset):
    def __init__(self, data, feature_columns, target_columns, features, targets, seq_length=1, timestamps=None, rooms=None, set=None):
        """
        Initialize the dataset.

        Args:
            data (pd.DataFrame): The original data used to create the dataset.
            feature_columns (list): The columns used as input features.
            target_columns (list): The columns used as target values.
            features (np.ndarray): The input features for the dataset.
            targets (np.ndarray): The target values for the dataset.
            seq_length (int): The sequence length used for each sample.
            timestamps (np.ndarray): The timestamps corresponding to each sample.
            rooms (np.ndarray): The room labels for each sample.
            set (str): The type of dataset (train, validation, test).
        """
        self.data = data
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.features = features
        self.targets = targets
        self.timestamps = timestamps if timestamps is not None else self.data['timestamp'].values
        self.rooms = rooms
        self.rooms_flag = rooms is not None
        self.seq_length = seq_length
        self.set = set
        if 'timestamp' in self.data.columns:
            self.data = self.data.set_index('timestamp')

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        """Return a dynamic dictionary of tensors based on available fields."""
        data_dict = {
            "inputs": torch.tensor(self.features[index], dtype=torch.float32),
            "targets": torch.tensor(self.targets[index], dtype=torch.float32),
            "timestamps": self.timestamps[index],
        }
        if self.rooms_flag:
            room_label = self.rooms[index]
            data_dict["room_labels"] = room_label
        return data_dict

    def reset_room_labels(self, new_room_labels):
        """
        Reset room labels with new values.
        
        This allows forcing the model to use a different room decoder by changing
        the room labels that are returned in __getitem__.
        
        Args:
            new_room_labels: numpy array or list of room label indices/values.
                            Must have the same length as the dataset.
        """
        if not self.rooms_flag:
            raise ValueError("Cannot reset room labels: dataset was not initialized with room labels")
        
        if len(new_room_labels) != len(self):
            raise ValueError(
                f"new_room_labels length ({len(new_room_labels)}) must match dataset length ({len(self)})"
            )
        
        # Convert to numpy array if needed
        if not isinstance(new_room_labels, np.ndarray):
            new_room_labels = np.array(new_room_labels)
        
        self.rooms = new_room_labels

    def get_data(self, timestamps=None):
        if timestamps is None: timestamps = self.timestamps
        return self.data.loc[timestamps].reset_index()

    def get_absolute_positions(self, timestamps=None):
        if timestamps is None: timestamps = self.timestamps
        positions = self.data.loc[timestamps][['X', 'Y']].values
        return positions

    def get_velocity(self, timestamps=None):#, normalized=False):
        if timestamps is None: timestamps = self.timestamps
        v = self.data.loc[timestamps]['V'].values
        return v

    def get_room(self, timestamps=None):
        if timestamps is None: timestamps = self.timestamps
        if 'room' not in self.data.columns:
            return None
        rooms = self.data.loc[timestamps]['room'].values
        return rooms

    def get_timestamps(self):
        return self.timestamps

    def get_target_columns(self):
        return self.target_columns

    def get_target_pred_columns(self):
        return [f'{column}_pred' for column in self.target_columns]

    def get_target(self, timestamps=None):
        if timestamps is None: timestamps = self.timestamps
        targets = self.data.loc[timestamps][self.target_columns].values
        return targets

    def get_target_pred(self, timestamps=None):
        if timestamps is None: timestamps = self.timestamps
        predictions = self.data.loc[timestamps][self.get_target_pred_columns()].values
        return predictions

    def update_predictions(self, predictions, timestamps=None):
        """
        Update the dataset with predicted target values.

        Args:
            predictions (np.ndarray or torch.Tensor): The predicted target values.
            timestamps (np.ndarray): The timestamps corresponding to the predictions.
        """
        target_pred_columns = self.get_target_pred_columns()

        # Check if the predicted columns already exist; if not, create them all at once
        missing_pred_columns = [col for col in target_pred_columns if col not in self.data.columns]

        if missing_pred_columns:
            # find the index of the last target column
            column_index = self.data.columns.get_loc(self.target_columns[-1]) + 1
            # Create empty columns for missing predictions
            for i,col in enumerate(missing_pred_columns):
                self.data.insert(column_index+i, col, np.nan)

        # if missing_pred_columns:
        #     column_index = self.data.columns.get_loc(self.target_columns[-1]) + 1
        #     # Create empty columns for missing predictions
        #     pred_df = pd.DataFrame(
        #         np.nan,
        #         index=self.data.index,
        #         columns=missing_pred_columns
        #     )
        #     # Insert new columns at the correct locations by combining the original data with the new columns
        #     self.data = pd.concat([self.data, pred_df], axis=1)

        # Update the predictions in the DataFrame using iloc to avoid size mismatch
        if timestamps is None: timestamps = self.timestamps
        if predictions.ndim == 3: predictions = predictions[:, -1, :]  # # If predictions are in sequence format, take the last state (last time step)
        self.data.loc[timestamps,target_pred_columns] = predictions

    def update_targets(self, targets, timestamps=None):
        """
        Update the dataset with predicted target values.

        Args:
            predictions (np.ndarray or torch.Tensor): The predicted target values.
            timestamps (np.ndarray): The timestamps corresponding to the predictions.
        """
        target_columns = self.get_target_columns()

        # Check if the predicted columns already exist; if not, create them all at once
        missing_target_columns = [col for col in target_columns if col not in self.data.columns]

        if missing_target_columns:
            # find the index of the last target column
            column_index = self.data.columns.get_loc(self.target_columns[-1]) + 1
            # Create empty columns for missing predictions
            for i,col in enumerate(missing_target_columns):
                self.data.insert(column_index+i, col, np.nan)

        # if missing_pred_columns:
        #     # Create empty columns for missing predictions
        #     pred_df = pd.DataFrame(
        #         np.nan,
        #         index=self.data.index,
        #         columns=missing_pred_columns
        #     )
        #     # Insert new columns at the correct locations by combining the original data with the new columns
        #     self.data = pd.concat([self.data, pred_df], axis=1)


        # Update the predictions in the DataFrame using iloc to avoid size mismatch
        if timestamps is not None: timestamps = self.timestamps
        self.data.loc[timestamps, target_columns] = targets



def extract_features_targets(data, feature_columns, target_columns):
    features = data[feature_columns].values
    targets = data[target_columns].values
    timestamps = data['timestamp'].values
    folds = data['fold'].values  # Include fold information to maintain fold-based sequences
    rooms = data['room'].values
    return features, targets, timestamps, rooms, folds

def create_sequences_samples(features, targets, timestamps, rooms, folds, seq_length, last_state, log_nan_stats=False):
    """
    Create sequence samples from features and targets, respecting fold boundaries.
    Drops sequences containing NaN values (e.g., from long interpolations).
    
    Args:
        features: Feature array
        targets: Target array
        timestamps: Timestamp array
        rooms: Room labels array
        folds: Fold labels array (for cross-validation)
        seq_length: Length of sequences to create
        last_state: If True, use only the last state of target; otherwise use full sequence
        log_nan_stats: If True, log NaN statistics (default: False)
    
    Returns:
        Tuple of (features, targets, timestamps, rooms) arrays with NaN sequences removed
    """
    logger = get_logger(__name__) if log_nan_stats else None
    
    if seq_length == 1:
        # For seq_length=1, filter out any samples with NaN values
        total_samples = len(features)
        valid_mask = ~(np.isnan(features).any(axis=1) | np.isnan(targets).any(axis=1))
        valid_samples = valid_mask.sum()
        nan_samples = total_samples - valid_samples
        if log_nan_stats and logger:
            nan_percentage = (nan_samples / total_samples * 100) if total_samples > 0 else 0.0
            logger.info(f"Dropped {nan_samples}/{total_samples} samples ({nan_percentage:.2f}%) due to NaN values (seq_length=1)")
        return (np.array(features[valid_mask]), np.array(targets[valid_mask]), 
                np.array(timestamps[valid_mask]), np.array(rooms[valid_mask]))
    
    # Prepare lists to store fold-based sequences
    all_seq_features, all_seq_targets, all_seq_timestamps, all_seq_rooms = [], [], [], []
    
    # Track total possible sequences and dropped sequences for logging
    total_sequences = 0
    dropped_sequences = 0
    
    # Process each fold individually to create sequences
    # Filter out NaN folds (which occur when k_folds=0)
    if folds.dtype == float or folds.dtype == object:
        # Handle NaN folds
        mask = ~np.isnan(folds) if folds.dtype == float else np.ones(len(folds), dtype=bool)
        unique_folds = np.unique(folds[mask])
    else:
        unique_folds = np.unique(folds)
    
    for fold in unique_folds:
        # Filter data by current fold
        if folds.dtype == float and (np.isnan(fold) if isinstance(fold, (float, np.floating)) else False):
            fold_indices = np.where(np.isnan(folds))[0]
        else:
            fold_indices = np.where(folds == fold)[0]
        
        if len(fold_indices) < seq_length + 1:
            # Skip folds that are too small to create sequences
            continue
            
        fold_features = features[fold_indices]
        fold_targets = targets[fold_indices]
        fold_timestamps = timestamps[fold_indices]
        fold_rooms = rooms[fold_indices]

        # Count total possible sequences for this fold
        fold_total_sequences = len(fold_features) - seq_length
        total_sequences += fold_total_sequences

        # Generate sequences within the fold and filter out NaN sequences
        for i in range(fold_total_sequences):
            seq_features = fold_features[i:i + seq_length]
            seq_targets = fold_targets[i + seq_length] if last_state else fold_targets[i:i + seq_length]
            
            # Check for NaN values in the sequence
            # Check features sequence for NaN (handles both 1D and 2D arrays)
            has_nan_features = np.isnan(seq_features).any()
            # Check targets for NaN (handles scalars, 1D, and 2D arrays)
            if seq_targets.ndim == 0:
                # Scalar target
                has_nan_targets = np.isnan(seq_targets)
            else:
                # Array target
                has_nan_targets = np.isnan(seq_targets).any()
            
            # Skip sequences with NaN values (from long interpolations that weren't allowed)
            if has_nan_features or has_nan_targets:
                dropped_sequences += 1
                continue
            
            all_seq_features.append(seq_features)
            all_seq_targets.append(seq_targets)
            all_seq_timestamps.append(fold_timestamps[i + seq_length])
            all_seq_rooms.append(fold_rooms[i + seq_length])

    # Log NaN percentage only if requested
    if log_nan_stats and logger:
        nan_percentage = (dropped_sequences / total_sequences * 100) if total_sequences > 0 else 0.0
        logger.info(f"Dropped {dropped_sequences}/{total_sequences} sequences ({nan_percentage:.2f}%) due to NaN values (seq_length={seq_length})")

    # Convert lists to arrays
    return (np.array(all_seq_features), np.array(all_seq_targets), np.array(all_seq_timestamps), np.array(all_seq_rooms))

def preprocess_data(data, feature_columns, target_columns, seq_length, last_state, log_nan_stats=False):
    features, targets, timestamps, rooms, folds = extract_features_targets(data, feature_columns, target_columns)
    return create_sequences_samples(features, targets, timestamps, rooms, folds, seq_length, last_state, log_nan_stats=log_nan_stats)

def validate_data_integrity(data, config, feature_columns=None, target_columns=None):
    """
    Validate data integrity by checking for missing columns, NaN values, and other issues.
    
    Args:
        data: DataFrame to validate
        config: Configuration dictionary
        feature_columns: List of feature column names (optional, for additional validation)
        target_columns: List of target column names (optional, for additional validation)
    
    Raises:
        ValueError: If critical validation checks fail
    """
    logger = get_logger(__name__)
    
    # Check required columns
    required_columns = ['timestamp', 'X', 'Y', 'HD', 'V']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for NaN values in critical columns (only timestamp is critical, others are expected to have NaNs)
    critical_columns = ['timestamp']
    for col in critical_columns:
        if col in data.columns:
            nan_count = data[col].isna().sum()
            if nan_count > 0:
                logger.warning(f"Column '{col}' has {nan_count} NaN values ({100*nan_count/len(data):.2f}%)")
    
    # Check timestamp uniqueness and sorting
    if data['timestamp'].duplicated().any():
        dup_count = data['timestamp'].duplicated().sum()
        logger.warning(f"Found {dup_count} duplicate timestamps")
    
    if not data['timestamp'].is_monotonic_increasing:
        logger.warning("Timestamps are not monotonically increasing")
    
    # Check feature/target columns if provided
    if feature_columns:
        missing_features = [col for col in feature_columns if col not in data.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")
    
    if target_columns:
        missing_targets = [col for col in target_columns if col not in data.columns]
        if missing_targets:
            raise ValueError(f"Missing target columns: {missing_targets}")
    
    # Check for cell columns if using cells as features
    cells_columns = [col for col in data.columns if col.startswith("Cell")]
    if len(cells_columns) == 0:
        logger.warning("No Cell_* columns found in dataset")
    else:
        # Check for cells with all NaN values
        cells_all_nan = [col for col in cells_columns if data[col].isna().all()]
        if cells_all_nan:
            logger.warning(f"Found {len(cells_all_nan)} cells with all NaN values: {cells_all_nan[:5]}...")
    
    logger.debug(f"Data validation passed. Dataset shape: {data.shape}, Cells: {len(cells_columns)}")


def get_cv_splits(config, data, k_folds, offset=0):
    """
    Get cross-validation splits for all folds.
    Uses create_fold_dataloaders for each fold to ensure consistency.
    
    Args:
        config: Configuration dictionary
        data: DataFrame with the dataset (must have 'room' column and be validated)
        k_folds: Number of cross-validation folds. Must be >= 2.
        offset: Offset to apply to room B's folds (default: 0). Only room B uses this offset.
               If None, defaults to 0.
    
    Returns:
        splits: List of (train_indices, val_indices, test_indices) tuples for each fold
        data: DataFrame with fold assignments
    """
    logger = get_logger(__name__)
    
    # Require k_folds >= 2 for cross-validation (consistent with create_fold_dataloaders)
    # Note: k_folds=0 is allowed in create_fold_dataloaders to disable CV, but get_cv_splits
    # is specifically for getting CV splits, so we require k_folds >= 2
    if k_folds < 2:
        raise ValueError(f"k_folds must be >= 2 for getting cross-validation splits, got {k_folds}.")
    
    # Make a copy of data to avoid modifying the original
    data = data.copy()
    
    # Use offset parameter (default 0) - always applies to room B
    if offset is None:
        offset = 0
    
    # Create fold_offsets dict with offset for room B
    fold_offsets = {'B': offset}
    
    # Room column should always exist in dataframe
    if 'room' not in data.columns:
        raise ValueError("Room column not found in data. Rooms should be added during preprocessing.")
    
    # Validate room column has data (consistent with create_fold_dataloaders)
    unique_rooms = data['room'].dropna().unique()
    if len(unique_rooms) == 0:
        raise ValueError("Room column exists but contains no valid room labels.")
    
    # Check for empty rooms (consistent with create_fold_dataloaders)
    for room_name in unique_rooms:
        room_count = (data['room'] == room_name).sum()
        if room_count == 0:
            logger.warning(f"Room '{room_name}' has no data points")
    
    # Remove fold column if it exists (to recompute with current offset)
    if 'fold' in data.columns:
        data = data.drop(columns=['fold'])
    
    # Add fold column with current offset
    # Note: create_fold_dataloaders will reassign this, but we do it here for consistency
    # and to ensure the returned data has the fold column
    data = assign_fold_column(data, k_folds=k_folds, fold_offsets=fold_offsets)
    
    # Use create_fold_dataloaders for each fold to get the splits
    splits = []
    for fold_index in range(k_folds):
        # Log only for the first fold
        log_nan_stats = (fold_index == 0)
        train_loader, val_loader, test_loader = create_fold_dataloaders(config, data, k_folds=k_folds, fold_index=fold_index, offset=offset, log_nan_stats=log_nan_stats)
        
        # Extract indices from datasets using timestamps
        # Get all timestamps from each dataset and map back to original data indices
        train_indices = np.array([])
        val_indices = np.array([])
        test_indices = np.array([])
        
        if train_loader is not None and len(train_loader.dataset) > 0:
            train_timestamps = train_loader.dataset.timestamps
            train_indices = data[data['timestamp'].isin(train_timestamps)].index.values
        
        if val_loader is not None and len(val_loader.dataset) > 0:
            val_timestamps = val_loader.dataset.timestamps
            val_indices = data[data['timestamp'].isin(val_timestamps)].index.values
        
        if test_loader is not None and len(test_loader.dataset) > 0:
            test_timestamps = test_loader.dataset.timestamps
            test_indices = data[data['timestamp'].isin(test_timestamps)].index.values
        
        splits.append((train_indices, val_indices, test_indices))

    return splits, data


def get_cv_splits_fast(config, data, k_folds, offset=0):
    """
    Get cross-validation splits for all folds without creating dataloaders.
    This is a lightweight version that only computes indices, skipping expensive preprocessing.
    Use this for visualization or when you only need split indices.
    
    Args:
        config: Configuration dictionary
        data: DataFrame with the dataset (must have 'room' column and be validated)
        k_folds: Number of cross-validation folds. Must be >= 2.
        offset: Offset to apply to room B's folds (default: 0). Only room B uses this offset.
               If None, defaults to 0.
    
    Returns:
        splits: List of (train_indices, val_indices, test_indices) tuples for each fold
        data: DataFrame with fold assignments
    """
    logger = get_logger(__name__)
    
    # Require k_folds >= 2 for cross-validation
    if k_folds < 2:
        raise ValueError(f"k_folds must be >= 2 for getting cross-validation splits, got {k_folds}.")
    
    # Make a copy of data to avoid modifying the original
    data = data.copy()
    
    # Use offset parameter (default 0) - always applies to room B
    if offset is None:
        offset = 0
    
    # Create fold_offsets dict with offset for room B
    fold_offsets = {'B': offset}
    
    # Room column should always exist in dataframe
    if 'room' not in data.columns:
        raise ValueError("Room column not found in data. Rooms should be added during preprocessing.")
    
    # Validate room column has data
    unique_rooms = data['room'].dropna().unique()
    if len(unique_rooms) == 0:
        raise ValueError("Room column exists but contains no valid room labels.")
    
    # Check for empty rooms
    for room_name in unique_rooms:
        room_count = (data['room'] == room_name).sum()
        if room_count == 0:
            logger.warning(f"Room '{room_name}' has no data points")
    
    # Remove fold column if it exists (to recompute with current offset)
    if 'fold' in data.columns:
        data = data.drop(columns=['fold'])
    
    # Add fold column with current offset
    data = assign_fold_column(data, k_folds=k_folds, fold_offsets=fold_offsets)
    
    # Set up cross-validation
    train_room = config['training']['room']
    test_room = config['evaluating']['room']
    
    # Get room indices using room column directly
    if isinstance(train_room, list):
        train_room_mask = data['room'].isin(train_room)
        train_room_indices = np.array(data[train_room_mask].index)
    else:
        train_room_mask = data['room'] == train_room
        train_room_indices = np.array(data[train_room_mask].index)
    
    if isinstance(test_room, list):
        test_room_mask = data['room'].isin(test_room)
        test_room_indices = np.array(data[test_room_mask].index)
    else:
        test_room_mask = data['room'] == test_room
        test_room_indices = np.array(data[test_room_mask].index)
    
    # Get fold assignments for train and test rooms
    train_room_folds = data.loc[train_room_indices, 'fold'].values
    train_room_rooms = data.loc[train_room_indices, 'room'].values
    test_room_folds = data.loc[test_room_indices, 'fold'].values
    
    # Compute splits for each fold
    splits = []
    train_room_list = train_room if isinstance(train_room, list) else [train_room]
    
    for fold_index in range(k_folds):
        fold_num = fold_index + 1  # Convert 0-indexed to 1-indexed
        
        # Test set: indices where test_room_folds == fold_num
        test_idx = np.where(test_room_folds == fold_num)[0]
        test_indices = test_room_indices[test_idx]
        
        # Train set: indices where train_room_folds != fold_num
        train_idx = np.where(train_room_folds != fold_num)[0]
        train_indices = train_room_indices[train_idx]
        
        # Validation set: take the last 1/(k_folds-1) of each fold as validation set (excluding test fold)
        val_indices = []
        if k_folds > 1:  # Avoid division by zero when k_folds is 1
            for curr_train_room in train_room_list:
                # Use 1-indexed folds (1 to k_folds)
                for curr_fold in range(1, k_folds + 1):
                    if curr_fold != fold_num:
                        # Use boolean indexing with room column directly
                        curr_fold_mask = (train_room_folds == curr_fold) & (train_room_rooms == curr_train_room)
                        curr_train_indices = train_room_indices[curr_fold_mask]
                        if len(curr_train_indices) > 0:
                            val_size = max(1, int(len(curr_train_indices) / (k_folds - 1)))
                            curr_val_indices = curr_train_indices[-val_size:]
                            val_indices.extend(curr_val_indices)
        val_indices = np.array(val_indices)
        
        # Ensure non-overlapping sets by verifying indices
        train_indices = np.setdiff1d(train_indices, np.concatenate([val_indices, test_indices]))
        
        splits.append((train_indices, val_indices, test_indices))
    
    return splits, data


def create_fold_dataloaders(config, data, k_folds=0, fold_index=0, offset=0, log_nan_stats=False):
    """
    Create dataloaders for a specific fold by splitting data into train/validation/test sets.
    
    Args:
        config: Configuration dictionary
        data: DataFrame with the dataset (must have 'room' column and be validated)
        k_folds: Number of cross-validation folds. Must be >= 2 for cross-validation, or 0 to disable.
        fold_index: Index of the fold to use (0-indexed)
        offset: Offset to apply to room B's folds (default: 0). Only room B uses this offset.
        log_nan_stats: If True, log NaN statistics during preprocessing (default: False)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    model_config = config['model']
    batch_size = model_config['batch_size']
    seq_length = model_config.get('seq_length', 1)
    last_state =  model_config.get('last_state', True)
    perspective = model_config.get('perspective', 'Allo')
    feature = model_config.get('feature', 'cells')
    target = model_config.get('target', 'xy_position')
    multi_room = model_config.get('multi_room', False)

    logger = get_logger(__name__)
    
    # Make a copy of data to avoid modifying the original (needed for different offsets)
    data = data.copy()
    
    # Require k_folds >= 2 for cross-validation
    if k_folds > 0 and k_folds < 2:
        raise ValueError(f"k_folds must be >= 2 for cross-validation, got {k_folds}. Use k_folds=0 to disable cross-validation.")
    
    # Use offset parameter (default 0) - always applies to room B
    if offset is None:
        offset = 0
    
    # Create fold_offsets dict with offset for room B
    fold_offsets = {'B': offset}
    
    # Room column should always exist in dataframe
    if 'room' not in data.columns:
        raise ValueError("Room column not found in data. Rooms should be added during preprocessing.")
    
    # Validate room column has data
    unique_rooms = data['room'].dropna().unique()
    if len(unique_rooms) == 0:
        raise ValueError("Room column exists but contains no valid room labels.")
    
    # Check for empty rooms
    for room_name in unique_rooms:
        room_count = (data['room'] == room_name).sum()
        if room_count == 0:
            logger.warning(f"Room '{room_name}' has no data points")
    
    # Remove fold column if it exists (to recompute with current offset)
    if 'fold' in data.columns:
        data = data.drop(columns=['fold'])
    
    # Add fold column with current offset
    data = assign_fold_column(data, k_folds=k_folds, fold_offsets=fold_offsets)
    
    # Set up cross-validation
    train_room = config['training']['room']
    test_room = config['evaluating']['room']

    # Get room indices using room column directly
    if isinstance(train_room, list):
        train_room_mask = data['room'].isin(train_room)
        train_room_indices = np.array(data[train_room_mask].index)
    else:
        train_room_mask = data['room'] == train_room
        train_room_indices = np.array(data[train_room_mask].index)

    if isinstance(test_room, list):
        test_room_mask = data['room'].isin(test_room)
        test_room_indices = np.array(data[test_room_mask].index)
    else:
        test_room_mask = data['room'] == test_room
        test_room_indices = np.array(data[test_room_mask].index)

    train_room_folds = data.loc[train_room_indices, 'fold'].values
    train_room_rooms = data.loc[train_room_indices, 'room'].values
    test_room_folds = data.loc[test_room_indices, 'fold'].values

    # Compute only the specific fold split needed (optimization)
    fold_num = fold_index + 1  # Convert 0-indexed to 1-indexed
    test_idx = np.where(test_room_folds == fold_num)[0]
    train_idx = np.where(train_room_folds != fold_num)[0]

    train_indices = train_room_indices[train_idx]
    test_indices = test_room_indices[test_idx]

    # Validation set: take the last 1/(k_folds-1) of each fold as validation set (excluding test fold) 
    val_indices = []
    train_room_list = train_room if isinstance(train_room, list) else [train_room]
    if k_folds > 1:  # Avoid division by zero when k_folds is 1
        for curr_train_room in train_room_list:
            # Use 1-indexed folds (1 to k_folds)
            for curr_fold in range(1, k_folds + 1):
                if curr_fold != fold_num:
                    # Use boolean indexing with room column directly
                    curr_fold_mask = (train_room_folds == curr_fold) & (train_room_rooms == curr_train_room)
                    curr_train_indices = train_room_indices[curr_fold_mask]
                    if len(curr_train_indices) > 0:
                        val_size = max(1, int(len(curr_train_indices) / (k_folds - 1)))
                        curr_val_indices = curr_train_indices[-val_size:]
                        val_indices.extend(curr_val_indices)
    val_indices = np.array(val_indices)

    # Ensure non-overlapping sets by verifying indices
    train_indices = np.setdiff1d(train_indices, np.concatenate([val_indices, test_indices]))

    # Convert head direction to radians
    data['HD'] = np.deg2rad(data['HD'])

    train_data = data.iloc[train_indices].copy()
    val_data = data.iloc[val_indices].copy() if val_indices is not None else pd.DataFrame()
    test_data = data.iloc[test_indices].copy() if test_indices is not None else pd.DataFrame()

    # Separate features and target
    cells_columns = [col for col in data.columns if col.startswith("Cell")]
    allo_columns = [col for col in data.columns if col.startswith('Allo')]
    ego_columns = [col for col in data.columns if col.startswith('Ego')]
    position_columns = ['X','Y']
    head_direction_column = 'HD'
    velucity_column = 'V'

    # Filter cells based on cell_indices if specified in config
    cell_indices = model_config.get('cell_indices', None)
    if cell_indices is None:
        # Backward compatibility: check for neuron_indices
        cell_indices = model_config.get('neuron_indices', None)
    
    if cell_indices is not None and feature == 'cells':
        # Edge case: Validate cell_indices is not empty
        if len(cell_indices) == 0:
            raise ValueError(
                "cell_indices is empty. At least one cell must be active. "
                "Check your cell_filter configuration."
            )
        # Validate indices are within range
        if max(cell_indices) >= len(cells_columns):
            raise ValueError(
                f"cell_indices contains index {max(cell_indices)} which is out of range. "
                f"Total cells: {len(cells_columns)}"
            )
        # Filter cells_columns to only include active cells
        cells_columns = [cells_columns[i] for i in cell_indices]
        logger.debug(f"Filtered to {len(cells_columns)} active cells based on cell_indices")

    if feature == 'cells':
        feature_columns = cells_columns
    elif feature == 'lidar_position':
        if perspective == 'Allo':
            feature_columns = allo_columns
        elif perspective == 'Ego':
            feature_columns = ego_columns
        else:
            raise ValueError('Invalid perspective')
    elif feature == 'xy_position':
        feature_columns = position_columns
    # elif feature == 'xy_homogeneous_position':
    #     feature_columns = position_columns + ['Z']
    elif feature == 'head_direction':
        feature_columns = [head_direction_column]
    elif feature == 'velucity':
        feature_columns = [velucity_column]
    else:
        raise ValueError('feature columns not found')

    if target == 'cells':
        # Use the same filtered cells_columns if cell_indices was applied
        target_columns = cells_columns
    elif target == 'lidar_position':
        if perspective == 'Allo':
            target_columns = allo_columns
        elif perspective == 'Ego':
            target_columns = ego_columns
        else:
            raise ValueError('Invalid perspective')
    elif target == 'xy_position':
        target_columns = position_columns
    elif target == 'head_direction':
        target_columns = [head_direction_column]
    elif target == 'velucity':
        target_columns = [velucity_column]
    elif target == 'xyh':
        target_columns = position_columns + [head_direction_column]
    else:
        raise ValueError('target columns not found')

    # Edge case: Validate feature_columns is not empty
    if len(feature_columns) == 0:
        raise ValueError(
            f"feature_columns is empty after filtering. "
            f"Feature type: {feature}, cell_indices: {cell_indices if cell_indices is not None else 'None'}"
        )

    model_config['input_dim'] = len(feature_columns)
    model_config['output_dim'] = len(target_columns)
    
    # Set n_total_cells if not already set (from preprocessing)
    if 'n_total_cells' not in model_config:
        # Get total number of cells before filtering
        all_cells_columns = [col for col in data.columns if col.startswith("Cell")]
        model_config['n_total_cells'] = len(all_cells_columns)
    
    # Set n_active_cells to match input_dim (number of cells actually used)
    model_config['n_active_cells'] = len(feature_columns) if feature == 'cells' else model_config.get('n_active_cells', model_config['input_dim'])
    
    # Preprocess data
    if log_nan_stats:
        logger.info("Preprocessing train set...")
    else:
        logger.debug("Preprocessing train set...")
    train_features, train_targets, train_timestamps, train_rooms = preprocess_data(train_data, feature_columns, target_columns, seq_length, last_state, log_nan_stats=log_nan_stats)
    if not val_data.empty:
        if log_nan_stats:
            logger.info("Preprocessing validation set...")
        else:
            logger.debug("Preprocessing validation set...")
        val_features, val_targets, val_timestamps, val_rooms = preprocess_data(val_data, feature_columns, target_columns, seq_length, last_state=last_state, log_nan_stats=log_nan_stats)
    else:
        val_features, val_targets, val_timestamps, val_rooms = (None, None, None, None)
    if not test_data.empty:
        if log_nan_stats:
            logger.info("Preprocessing test set...")
        else:
            logger.debug("Preprocessing test set...")
        test_features, test_targets, test_timestamps, test_rooms = preprocess_data(test_data, feature_columns, target_columns, seq_length, last_state=last_state, log_nan_stats=log_nan_stats)
    else:
        test_features, test_targets, test_timestamps, test_rooms = (None, None, None, None)

    # Create datasets and dataloaders
    logger.debug("Creating datasets and dataloaders...")
    train_dataset = TimeSeriesDataset(train_data, feature_columns, target_columns, train_features, train_targets, seq_length=seq_length, timestamps=train_timestamps, rooms=train_rooms if multi_room else None, set='train')
    val_dataset = TimeSeriesDataset(val_data, feature_columns, target_columns, val_features, val_targets, seq_length=seq_length, timestamps=val_timestamps, rooms=val_rooms if multi_room else None, set='validation') if val_features is not None else None
    test_dataset = TimeSeriesDataset(test_data, feature_columns, target_columns, test_features, test_targets, seq_length=seq_length, timestamps=test_timestamps, rooms=test_rooms if multi_room else None, set='test') if test_features is not None else None
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False) if val_dataset is not None else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False) if test_dataset is not None else None

    return train_loader, val_loader, test_loader


def get_fold_offsets_dict_from_offset_value(offset_val):
    """
    Create fold_offsets dict from offset value.
    
    Args:
        offset_val: Offset value to use for room B
    
    Returns:
        dict: fold_offsets dictionary with format {'B': offset_val}
    """
    return {'B': offset_val}


def assign_fold_column(data, k_folds=0, fold_offsets=None, use_offset_column=False):
    """
    Adds 'fold' column to the dataframe by splitting each room into k_folds folds.
    Assumes 'room' column already exists. Folds are 1-indexed (1 to k_folds).
    
    Supports fold offsets to create circular shifts between rooms for more train/test split variations.

    Parameters:
        data (DataFrame): Input dataframe with 'room' column.
        k_folds (int): Number of folds for fold splitting.
        fold_offsets (dict): Optional dict mapping room names to offset integers for circular shift.
                           Example: {'B': 1} shifts room B's folds by 1 (circular: 1→2, 2→3, ..., k_folds→1).
                           Rooms not in dict use offset=0 (no shift). Default: None (treated as {}).
        use_offset_column (bool): If True and 'offset' column exists, use it to create fold_offsets.
                                 If fold_offsets is also provided, it takes precedence.

    Returns:
        data (DataFrame): Updated dataframe with 'fold' column (1-indexed: 1 to k_folds).
    """
    if fold_offsets is None:
        fold_offsets = {}
    
    # If use_offset_column is True and offset column exists, create fold_offsets from it
    if use_offset_column and 'offset' in data.columns and not fold_offsets:
        # Group by offset and process each group separately
        offset_groups = data.groupby('offset')
        if len(offset_groups) > 1:
            # Multiple offsets - process separately
            result_dfs = []
            for offset_val, offset_group in offset_groups:
                group_fold_offsets = get_fold_offsets_dict_from_offset_value(offset_val)
                group_with_folds = assign_fold_column(offset_group.copy(), k_folds=k_folds, fold_offsets=group_fold_offsets, use_offset_column=False)
                result_dfs.append(group_with_folds)
            return pd.concat(result_dfs, ignore_index=True)
        elif len(offset_groups) == 1:
            # Single offset - use it directly
            offset_val = data['offset'].iloc[0]
            fold_offsets = get_fold_offsets_dict_from_offset_value(offset_val)
    
    if 'fold' not in data.columns:
        data['fold'] = np.nan
        if k_folds > 0:
            for room in data['room'].dropna().unique():
                room_mask = data['room'] == room
                room_indices = np.array(data[room_mask].index)
                if len(room_indices) > 0:
                    min_index, max_index = room_indices.min(), room_indices.max()
                    fold_interval = (max_index - min_index) / k_folds
                    # Calculate base fold (1-indexed: 1 to k_folds)
                    base_folds = np.floor((room_indices - min_index) / fold_interval).astype(int) + 1
                    # Ensure base_folds are in range [1, k_folds]
                    base_folds[base_folds > k_folds] = k_folds
                    
                    # Apply circular shift if offset specified for this room
                    offset = fold_offsets.get(room, 0)
                    if offset != 0:
                        # Circular shift: (base_fold - 1 + offset) % k_folds + 1
                        room_folds = ((base_folds - 1 + offset) % k_folds) + 1
                    else:
                        room_folds = base_folds
                    
                    data.loc[room_indices, 'fold'] = room_folds

    return data


def map_rooms_to_indices(config, full_data):
    # TODO: Remove this function later - rooms should always be in the dataframe
    # room_separation_time = config['preprocessing']['room_times']
    # room_separation_index = config['preprocessing']['room_indices']
    map_rooms = config.get('preprocessing', {}).get('map_rooms', {})

    if not map_rooms:
        raise ValueError("No room mapping found in the configuration. Please provide 'map_rooms' in 'preprocessing' section.")


    def get_data_by_timestamp_range(df, timestamp_range):
        return df[(df['timestamp'] >= timestamp_range[0]) & (df['timestamp'] < timestamp_range[1])]


    map_index_to_indices = {}
    map_room_to_indices = {}
    map_room_to_timestamps = {}

    for room_name, room in map_rooms['rooms'].items():
        timestamp_range = room['range']
        if 'room' in full_data.columns:
            room_indices = full_data[full_data['room'] == room_name].index
        else:
            room_indices = np.array(get_data_by_timestamp_range(full_data, timestamp_range).index)  # .tolist()

        room_timestamps = np.array(full_data.loc[room_indices, 'timestamp'])

        # Add a warning if no data is found for the given room
        if room_indices.size == 0:
            warnings.warn(f"No data found for room '{room_name}' in the specified timestamp range: {timestamp_range}")

        room_index = room['index']
        if room_index not in map_index_to_indices:
            map_index_to_indices[room_index] = []
        map_index_to_indices[room_index].extend(room_indices)

        if room_name not in map_room_to_indices:
            map_room_to_indices[room_name] = []
        map_room_to_indices[room_name].extend(room_indices)

        if room_name not in map_room_to_timestamps:
            map_room_to_timestamps[room_name] = []
        map_room_to_timestamps[room_name].extend(room_timestamps)

    return {'index': map_index_to_indices, 'name': map_room_to_indices, 'room_timestamps': map_room_to_timestamps}