"""
Centralized metrics calculation module.

This module provides unified metric calculation functions for model evaluation,
supporting both 2D coordinate metrics (X, Y) and general multi-output metrics.
"""

import numpy as np
import torch
import os
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error

# Import utils modules (use relative imports to avoid circular dependencies)
try:
    from .helpers import get_directory
    from .logger import get_logger
except ImportError:
    # Fallback for direct execution
    from utils.helpers import get_directory
    from utils.logger import get_logger

logger = get_logger(__name__)


def calculate_metrics(targets, predictions, target_columns=['x', 'y']):
    """
    Unified metrics calculation function.
    
    Computes evaluation metrics (MSE and R²) for predictions. For 2D coordinate 
    predictions, returns detailed per-axis metrics. Supports both tensors and numpy arrays.
    
    Args:
        targets: numpy array or torch.Tensor of shape (N, D) or (N, T, D) - true values
        predictions: numpy array or torch.Tensor of shape (N, D) or (N, T, D) - predicted values
        target_columns: list of column names (default: ['x', 'y']) for documentation/logging purposes
    
    Returns:
        dict
    """
    if targets.shape != predictions.shape:
        raise ValueError("Targets and predictions must have the same shape")
    if targets.shape[-1] != len(target_columns):
        raise ValueError(f"Targets must have {len(target_columns)} columns")

    # Take the last timestamp for each sequence (last index in second dimension) - legacy support
    if len(targets.shape) == 3:
        targets = targets[:, -1, :]
    if len(predictions.shape) == 3:
        predictions = predictions[:, -1, :]
    
    # Check if inputs are GPU tensors for optimized computation
    is_gpu_tensor = isinstance(predictions, torch.Tensor) and predictions.is_cuda
    is_torch_tensor = isinstance(predictions, torch.Tensor) or isinstance(targets, torch.Tensor)
    
    # Convert to numpy if torch tensor (defensive programming) - but keep on GPU if possible
    if is_gpu_tensor:
        # Keep on GPU for computation, convert to numpy only at the end
        if not isinstance(predictions, torch.Tensor):
            predictions = torch.tensor(predictions, device=targets.device, dtype=torch.float32)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, device=predictions.device, dtype=torch.float32)
        # Ensure same device
        if predictions.device != targets.device:
            targets = targets.to(predictions.device)
    else:
        # Convert to numpy for CPU computation
        predictions = np.array(predictions.detach().cpu().numpy(), dtype=np.float32) if isinstance(predictions, torch.Tensor) else np.array(predictions, dtype=np.float32)
        targets = np.array(targets.detach().cpu().numpy(), dtype=np.float32) if isinstance(targets, torch.Tensor) else np.array(targets, dtype=np.float32)
    
    # Handle NaN values
    if is_gpu_tensor:
        # GPU tensor NaN handling
        if predictions.numel() > 0 and torch.isnan(predictions).any():
            nan_indices = torch.isnan(predictions).any(dim=1)
            if nan_indices.any():
                targets = targets[~nan_indices]
                predictions = predictions[~nan_indices]
        is_empty = targets.shape[0] == 0 or predictions.shape[0] == 0
    else:
        # Numpy NaN handling
        if predictions.size > 0 and np.isnan(predictions).any():
            nan_indices = np.isnan(predictions).any(axis=1)
            if nan_indices.any():
                targets = targets[~nan_indices]
                predictions = predictions[~nan_indices]
        is_empty = len(targets) == 0 or len(predictions) == 0
    
    if is_empty:
        # Get n_outputs from shape if available, otherwise use target_columns length
        if is_gpu_tensor:
            n_outputs = targets.shape[1] if len(targets.shape) > 1 and targets.shape[0] > 0 else len(target_columns)
        else:
            n_outputs = targets.shape[1] if len(targets.shape) > 1 and len(targets) > 0 else len(target_columns)
        base_metrics = {
            'mse': 0.0,
            'r2_pooled': 0.0
        }
        for col in target_columns:
            base_metrics[f'sse_{col}'] = 0.0
            base_metrics[f'sst_{col}'] = 0.0
            base_metrics[f'r2_{col}'] = 0.0
        base_metrics['n_samples'] = 0

        return base_metrics
    
    # Compute MSE and R² (GPU-optimized if possible)
    # Use mean squared Euclidean distance for 2D spatial data
    is_2d_spatial = targets.shape[-1] == 2 and len(target_columns) == 2 and target_columns == ['x', 'y']
    
    if is_gpu_tensor:
        # GPU computation
        n_samples = targets.shape[0]
        if is_2d_spatial:
            mse = float(torch.mean(torch.sum((targets - predictions) ** 2, dim=1)).item())
        else:
            mse = float(torch.mean((targets - predictions) ** 2).item())
        
        # Compute SSE and SST for all dimensions
        sse_total = torch.sum((targets - predictions) ** 2).item()
        mean_targets = torch.mean(targets, dim=0)
        sst_total = torch.sum((targets - mean_targets) ** 2).item()
        
        # Compute pooled R² (clamp to be at least 0)
        r2_pooled = float(max(0.0, 1 - (sse_total / sst_total))) if sst_total > 0 else 0.0
    else:
        # CPU computation (numpy)
        if is_2d_spatial:
            mse = float(np.round(np.mean(np.sum((targets - predictions) ** 2, axis=1)), 5))
        else:
            mse = float(np.round(mean_squared_error(targets, predictions, multioutput='uniform_average'), 5))
        
        # Compute R² using SSE/SST approach (variance-weighted)
        n_samples = len(targets)
        
        # Compute SSE and SST for all dimensions
        sse_total = np.sum((targets - predictions) ** 2)
        mean_targets = np.mean(targets, axis=0)
        sst_total = np.sum((targets - mean_targets) ** 2)
        
        # Compute pooled R² (clamp to be at least 0)
        r2_pooled = float(max(0.0, 1 - (sse_total / sst_total))) if sst_total > 0 else 0.0
    
    metrics = {
        'mse': mse,
        'r2_pooled': r2_pooled
    }
    
    # Add detailed per-axis metrics for each target column
    n_outputs = targets.shape[-1] if len(targets.shape) > 1 else 1
    for idx, col in enumerate(target_columns):
        if idx >= n_outputs:
            break
        
        target_col = targets[:, idx]
        pred_col = predictions[:, idx]
        
        if is_gpu_tensor:
            # GPU computation
            # Compute SSE (Sum of Squared Errors) for this column
            sse_col = torch.sum((target_col - pred_col) ** 2).item()
            
            # Compute SST (Total Sum of Squares / variance) for this column
            mean_col = torch.mean(target_col).item()
            sst_col = torch.sum((target_col - mean_col) ** 2).item()
        else:
            # CPU computation (numpy)
            # Compute SSE (Sum of Squared Errors) for this column
            sse_col = np.sum((target_col - pred_col) ** 2)
            
            # Compute SST (Total Sum of Squares / variance) for this column
            mean_col = np.mean(target_col)
            sst_col = np.sum((target_col - mean_col) ** 2)
        
        # Compute R² for this column (clamp to be at least 0)
        r2_col = float(max(0.0, 1 - (sse_col / sst_col))) if sst_col > 0 else 0.0
        
        metrics[f'sse_{col}'] = float(sse_col)
        metrics[f'sst_{col}'] = float(sst_col)
        metrics[f'r2_{col}'] = float(r2_col)
    
    metrics['n_samples'] = n_samples
    
    return metrics


def compute_r2_metrics(targets, predictions):
    """
    Compute R² metrics including SSE, SST, and R² for X and Y coordinates.
    
    This function is a wrapper for backward compatibility. It calls calculate_metrics
    with default target_columns=['x', 'y'].
    
    Args:
        targets: numpy array of shape (N, 2) - true X and Y coordinates
        predictions: numpy array of shape (N, 2) - predicted X and Y coordinates
    
    Returns:
        dict with keys: sse_x, sse_y, sst_x, sst_y, r2_x, r2_y, r2_pooled, n_samples
    """
    return calculate_metrics(targets, predictions, target_columns=['x', 'y'])


def compute_r2_from_sse_sst(sse_dict, sst_dict, target_columns):
    """
    Compute R² metrics from SSE and SST dictionaries (for pooling across folds/offsets).
    
    Args:
        sse_dict: dict mapping target column names to SSE values (e.g., {'X': 100.0, 'Y': 200.0})
        sst_dict: dict mapping target column names to SST values (e.g., {'X': 150.0, 'Y': 250.0})
        target_columns: list of target column names
    
    Returns:
        dict with keys:
            - r2_{col} for each target column
            - r2_pooled (pooled across all dimensions)
    """
    metrics = {}
    
    # Compute per-column R²
    for col in target_columns:
        sse = sse_dict.get(col, 0.0)
        sst = sst_dict.get(col, 0.0)
        r2_col = max(0.0, 1 - (sse / sst)) if sst > 0 else 0.0
        metrics[f'r2_{col}'] = float(r2_col)
    
    # Compute pooled R² across all dimensions
    total_sse = sum(sse_dict.values())
    total_sst = sum(sst_dict.values())
    r2_pooled = max(0.0, 1 - (total_sse / total_sst)) if total_sst > 0 else 0.0
    metrics['r2_pooled'] = float(r2_pooled)
    
    return metrics


def compute_pooled_r2_from_sse_sst(sse_dict, sst_dict):
    """
    Compute pooled R² from SSE and SST dictionaries.
    
    Args:
        sse_dict: dict mapping target column names to SSE values
        sst_dict: dict mapping target column names to SST values
    
    Returns:
        float: pooled R² value (clamped to be non-negative)
    """
    total_sse = sum(sse_dict.values())
    total_sst = sum(sst_dict.values())
    return max(0.0, 1 - (total_sse / total_sst)) if total_sst > 0 else 0.0


def compute_statistics(values):
    """
    Compute mean, std, and sem from a list of values.
    
    Args:
        values: list or array of numeric values
    
    Returns:
        dict with keys: 'mean', 'std', 'sem'
    """
    if len(values) == 0:
        return {'mean': 0.0, 'std': 0.0, 'sem': 0.0}
    
    values_array = np.array(values)
    mean = float(np.mean(values_array))
    std = float(np.std(values_array))
    sem = float(std / np.sqrt(len(values))) if len(values) > 0 else 0.0
    
    return {
        'mean': mean,
        'std': std,
        'sem': sem
    }


def pool_metrics_across_folds(metrics_df, target_columns):
    """
    Pool metrics across folds by summing SSE/SST and computing R² from pooled values.
    
    Args:
        metrics_df: DataFrame with columns: sse_{col}, sst_{col}, r2_{col} for each target column
        target_columns: list of target column names
    
    Returns:
        dict with keys:
            - sse_{col}: pooled SSE for each column
            - sst_{col}: pooled SST for each column
            - r2_{col}: R² computed from pooled SSE/SST for each column
            - r2_pooled: pooled R² across all dimensions
    """
    pooled_metrics = {}
    sse_dict = {}
    sst_dict = {}
    
    # Pool SSE/SST across folds for each target column
    for col in target_columns:
        sse_col = f'sse_{col}'
        sst_col = f'sst_{col}'
        r2_col = f'r2_{col}'
        
        if sse_col in metrics_df.columns and sst_col in metrics_df.columns:
            sse_dict[col] = float(metrics_df[sse_col].sum())
            sst_dict[col] = float(metrics_df[sst_col].sum())
            pooled_metrics[sse_col] = sse_dict[col]
            pooled_metrics[sst_col] = sst_dict[col]
        else:
            sse_dict[col] = 0.0
            sst_dict[col] = 0.0
            pooled_metrics[sse_col] = 0.0
            pooled_metrics[sst_col] = 0.0
    
    # Compute R² from pooled SSE/SST
    r2_metrics = compute_r2_from_sse_sst(sse_dict, sst_dict, target_columns)
    pooled_metrics.update(r2_metrics)
    
    return pooled_metrics


def load_position_scaler_from_config(config):
    """
    Load position scaler from config's data directory.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        scaler: Scaler object, or None if not found
    """
    import warnings
    try:
        scaler_path = os.path.join(get_directory(config, "data"), "positions_scaler.pkl")
        if os.path.exists(scaler_path):
            # Load scaler directly to avoid circular import
            # Suppress sklearn version mismatch warnings when loading scaler
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
            data = joblib.load(scaler_path)
            scaler = data['scaler']
            return scaler
        else:
            logger.warning(f"Position scaler file not found at {scaler_path}")
            return None
    except Exception as e:
        logger.warning(f"Could not load position scaler: {e}")
        return None


def get_rmse_scale_factor(scaler):
    """
    Calculate scale factor to convert RMSE from normalized units to cm.
    
    RMSE_normalized * scale_factor = RMSE_cm
    
    This function is a wrapper around get_scale_factor_from_scaler for backward compatibility.
    Consider using get_scale_factor_from_scaler directly for new code.
    
    Args:
        scaler: Scaler object (e.g., StandardScaler)
        
    Returns:
        float: Scale factor for RMSE conversion
    """
    from utils.config import get_scale_factor_from_scaler
    return get_scale_factor_from_scaler(scaler)


def convert_rmse_to_cm(df, scaler=None, config=None):
    """
    Convert RMSE values in DataFrame from normalized units to cm.
    
    Args:
        df: DataFrame containing RMSE columns (columns with 'rmse' in name)
        scaler: Optional scaler object. If None and config is provided, will load from config
        config: Optional configuration dictionary to load scaler from
        
    Returns:
        DataFrame: DataFrame with RMSE columns converted to cm
    """
    # Load scaler from config if not provided
    if scaler is None and config is not None:
        scaler = load_position_scaler_from_config(config)
    
    if scaler is None:
        logger.warning("No scaler available, RMSE will remain in normalized units")
        return df
    
    # Calculate scale factor
    scale_factor = get_rmse_scale_factor(scaler)
    
    if scale_factor == 1.0:
        logger.warning("Scale factor is 1.0, RMSE values unchanged")
        return df
    
    # Convert RMSE columns to cm
    df_converted = df.copy()
    rmse_cols = [col for col in df_converted.columns if 'rmse' in col.lower()]
    
    if len(rmse_cols) == 0:
        logger.debug("No RMSE columns found in DataFrame")
        return df_converted
    
    for col in rmse_cols:
        if col in df_converted.columns:
            df_converted[col] = df_converted[col] * scale_factor
    
    logger.info(f"Converted {len(rmse_cols)} RMSE column(s) from normalized units to cm (scale_factor={scale_factor:.4f})")
    
    return df_converted


# ============================================================================
# Decoder metrics table building (offset-level, room-level, ensemble)
# ============================================================================

def build_decoder_offset_metrics_table(results_df, target_columns, pred_columns):
    """
    Build decoder_offset_metrics table (offset-level, detailed metrics with fold information).
    
    Creates:
    1. One row per (project, group, set, room, fold, offset) - fold-level rows with n_folds=1
    2. One row per (project, group, set, room, offset) with fold=None - all-fold-level rows
       that pool out-of-fold predictions across folds, with n_folds set to the count of folds.
    
    Args:
        results_df: DataFrame with one row per timestamp, columns:
            - project, group, version, set, timestamp, offset, fold, room
            - target columns (e.g., 'x', 'y')
            - prediction columns (e.g., 'x_pred', 'y_pred')
        target_columns: list of target column names
        pred_columns: list of prediction column names
    
    Returns:
        DataFrame with columns: project, group, version, room, offset, set, fold, n_folds, n_timestamps,
        r2_x, r2_y, r2_pooled, sse_x, sse_y, sst_x, sst_y
        Note: fold is set to None for all-fold-level rows, and n_folds indicates the count
    """
    if len(results_df) == 0:
        return pd.DataFrame()
    
    # Filter out "All" room for now - we'll handle it separately if needed
    results_df_filtered = results_df[results_df['room'] != 'All'].copy()
    
    rows = []
    
    # 1. Create fold-level rows (one per fold, offset, room combination)
    group_cols = ['project', 'group', 'version', 'set', 'offset', 'fold', 'room']
    for (project, group, version, set_name, offset, fold, room), group_df in results_df_filtered.groupby(group_cols):
        targets = group_df[target_columns].values
        predictions = group_df[pred_columns].values
        
        metrics = calculate_metrics(targets, predictions, target_columns=target_columns)
        
        row = {
            'project': project,
            'group': group,
            'version': version,
            'room': room,
            'offset': offset,
            'set': set_name,
            'fold': fold,
            'n_folds': 1,  # Single fold
            'n_timestamps': len(group_df),
            **metrics  # Unpack all metrics
        }
        
        rows.append(row)
    
    # 2. Create all-fold-level rows (pooled across folds per room, offset)
    # Get fold counts before pooling
    fold_counts = results_df_filtered.groupby(['project', 'group', 'version', 'set', 'room', 'offset']).agg({
        'fold': 'nunique' if 'fold' in results_df_filtered.columns else lambda x: 0,
    }).reset_index()
    fold_counts.columns = ['project', 'group', 'version', 'set', 'room', 'offset', 'n_folds']
    
    # Group by (project, group, version, set, room, offset) and pool predictions across folds
    all_fold_group_cols = ['project', 'group', 'version', 'set', 'room', 'offset']
    for (project, group, version, set_name, room, offset), group_df in results_df_filtered.groupby(all_fold_group_cols):
        # Pool all predictions and targets across folds into continuous vectors
        targets = group_df[target_columns].values
        predictions = group_df[pred_columns].values
        
        # Calculate metrics directly from pooled predictions vs targets
        metrics = calculate_metrics(targets, predictions, target_columns=target_columns)
        
        # Get fold count from pre-computed data
        counts = fold_counts[
            (fold_counts['project'] == project) &
            (fold_counts['group'] == group) &
            (fold_counts['version'] == version) &
            (fold_counts['set'] == set_name) &
            (fold_counts['room'] == room) &
            (fold_counts['offset'] == offset)
        ]
        n_folds = int(counts['n_folds'].iloc[0]) if len(counts) > 0 else 0
        
        row = {
            'project': project,
            'group': group,
            'version': version,
            'room': room,
            'offset': offset,
            'set': set_name,
            'fold': None,  # Empty fold field for all-fold rows
            'n_folds': n_folds,
            'n_timestamps': len(group_df),
            **metrics  # Unpack all metrics
        }
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def build_decoder_room_metrics_table(decoder_offset_metrics_df, target_columns, pred_columns):
    """
    Build decoder_room_metrics table (room-level summary statistics).
    
    Uses all-fold-level rows from decoder_offset_metrics_df (where fold is None) which already contain
    pooled metrics across folds. Then computes statistics across offsets.
    
    Args:
        decoder_offset_metrics_df: DataFrame from build_decoder_offset_metrics_table with columns:
            - project, group, version, room, offset, set, fold, n_folds, n_timestamps
            - r2_pooled, mse, r2_{col}, sse_{col}, sst_{col} for each target column
        target_columns: list of target column names (e.g., ['x', 'y'] or ['X', 'Y'])
        pred_columns: list of prediction column names (e.g., ['x_pred', 'y_pred'])
        Note: pred_columns is kept for API compatibility but not used
    
    Returns:
        DataFrame with columns: project, group, version, room, set, n_folds, n_offsets, n_timestamps_total,
        r2_pooled_mean, r2_pooled_std, r2_pooled_sem, r2_pooled_across_offsets,
        rmse_mean, rmse_std, rmse_sem,
        r2_{col}_mean, r2_{col}_std, r2_{col}_sem for each target column,
        sse_{col}_total, sst_{col}_total for each target column
        Note: Includes individual rooms and "All" room row, grouped by set
        Note: RMSE is computed as sqrt(MSE) and is preferred for logging since it has 
              the same units as the normalized arena coordinates
    """
    if len(decoder_offset_metrics_df) == 0:
        return pd.DataFrame()
    
    # Filter for all-fold-level rows (where fold is None) and exclude "All" room
    all_folds_df = decoder_offset_metrics_df[
        (decoder_offset_metrics_df['fold'].isna()) & 
        (decoder_offset_metrics_df['room'] != 'All')
    ].copy()
    
    if len(all_folds_df) == 0:
        return pd.DataFrame()
    
    # Use the all-fold rows directly as intermediate data
    intermediate_df = all_folds_df.copy()

    rows = []
    
    # Now group by (project, group, version, room, set) to aggregate across offsets
    for (project, group, version, room, set_name), group_df in intermediate_df.groupby(['project', 'group', 'version', 'room', 'set']):
        # Compute counts
        n_offsets = len(group_df['offset'].unique())
        n_folds = group_df['n_folds'].max()
        n_timestamps_total = group_df['n_timestamps'].sum()
        
        # Compute mean/std/sem across offsets for pooled R² using centralized function
        r2_pooled_stats = compute_statistics(group_df['r2_pooled'].values)
        r2_pooled_mean = r2_pooled_stats['mean']
        r2_pooled_std = r2_pooled_stats['std']
        r2_pooled_sem = r2_pooled_stats['sem']
        
        # Compute RMSE from MSE (sqrt of MSE for each offset, then compute stats)
        rmse_values = np.sqrt(group_df['mse'].values)
        rmse_stats = compute_statistics(rmse_values)
        rmse_mean = rmse_stats['mean']
        rmse_std = rmse_stats['std']
        rmse_sem = rmse_stats['sem']
        
        # Compute per-column metrics (columns exist as created by calculate_metrics)
        r2_metrics = {col: compute_statistics(group_df[f'r2_{col}'].values) for col in target_columns}
        sse_totals = {col: group_df[f'sse_{col}'].sum() for col in target_columns}
        sst_totals = {col: group_df[f'sst_{col}'].sum() for col in target_columns}
        
        # Compute pooled R² across offsets using centralized function
        r2_pooled_across_offsets = compute_pooled_r2_from_sse_sst(sse_totals, sst_totals)
        
        # Build row
        row = {
            'project': project,
            'group': group,
            'version': version,
            'room': room,
            'set': set_name,
            'n_folds': n_folds,
            'n_offsets': n_offsets,
            'n_timestamps_total': n_timestamps_total,
            'r2_pooled_mean': r2_pooled_mean,
            'r2_pooled_std': r2_pooled_std,
            'r2_pooled_sem': r2_pooled_sem,
            'r2_pooled_across_offsets': r2_pooled_across_offsets,
            'rmse_mean': rmse_mean,
            'rmse_std': rmse_std,
            'rmse_sem': rmse_sem,
        }
        
        # Add per-column metrics
        for col in target_columns:
            row.update({
                f'r2_{col}_mean': r2_metrics[col]['mean'],
                f'r2_{col}_std': r2_metrics[col]['std'],
                f'r2_{col}_sem': r2_metrics[col]['sem'],
                f'sse_{col}_total': sse_totals[col],
                f'sst_{col}_total': sst_totals[col]
            })
        
        rows.append(row)
    
    # Compute "All" room by pooling SSE/SST across all individual rooms, grouped by set
    if len(rows) > 0:
        # Group by (project, group, version, set) to compute "All" room for each set separately
        rows_df = pd.DataFrame(rows)
        for (project, group, version, set_name), set_df in rows_df.groupby(['project', 'group', 'version', 'set']):
            # Filter out "All" room if it exists
            individual_rooms = set_df[set_df['room'] != 'All'].to_dict('records')
            if len(individual_rooms) > 0:
                # Pool SSE/SST across all individual rooms for each column (columns exist as created above)
                sse_all_dict = {col: sum(r[f'sse_{col}_total'] for r in individual_rooms) for col in target_columns}
                sst_all_dict = {col: sum(r[f'sst_{col}_total'] for r in individual_rooms) for col in target_columns}
                r2_values_dict = {col: [r[f'r2_{col}_mean'] for r in individual_rooms] for col in target_columns}
                
                n_folds_all = max(r['n_folds'] for r in individual_rooms) if individual_rooms else 1
                n_offsets_all = max(r['n_offsets'] for r in individual_rooms) if individual_rooms else 1
                n_timestamps_all = sum(r['n_timestamps_total'] for r in individual_rooms)
                
                # Compute pooled R² across offsets using centralized function
                r2_pooled_across_offsets_all = compute_pooled_r2_from_sse_sst(sse_all_dict, sst_all_dict)
                
                # Compute mean/std/sem across individual rooms using centralized function
                r2_pooled_values = [r['r2_pooled_mean'] for r in individual_rooms]
                r2_pooled_stats_all = compute_statistics(r2_pooled_values)
                
                # Compute mean/std/sem across individual rooms for RMSE
                rmse_values = [r['rmse_mean'] for r in individual_rooms]
                rmse_stats_all = compute_statistics(rmse_values)
                
                # Build "All" row
                all_row = {
                    'project': individual_rooms[0]['project'],
                    'group': individual_rooms[0]['group'],
                    'version': individual_rooms[0]['version'],
                    'room': 'All',
                    'set': set_name,
                    'n_offsets': n_offsets_all,
                    'n_folds': n_folds_all,
                    'n_timestamps_total': n_timestamps_all,
                    'r2_pooled_mean': r2_pooled_stats_all['mean'],
                    'r2_pooled_std': r2_pooled_stats_all['std'],
                    'r2_pooled_sem': r2_pooled_stats_all['sem'],
                    'r2_pooled_across_offsets': r2_pooled_across_offsets_all,
                    'rmse_mean': rmse_stats_all['mean'],
                    'rmse_std': rmse_stats_all['std'],
                    'rmse_sem': rmse_stats_all['sem'],
                }
                
                # Add per-column metrics for "All" room 
                for col in target_columns:
                    r2_col_stats = compute_statistics(r2_values_dict[col])
                    all_row.update({
                        f'r2_{col}_mean': r2_col_stats['mean'],
                        f'r2_{col}_std': r2_col_stats['std'],
                        f'r2_{col}_sem': r2_col_stats['sem'],
                        f'sse_{col}_total': sse_all_dict[col],
                        f'sst_{col}_total': sst_all_dict[col]
                    })
                
                rows.append(all_row)
    
    return pd.DataFrame(rows)


def build_decoder_room_ensemble_metrics_table(results_df, target_columns, pred_columns):
    """
    Build decoder_room_ensemble_metrics table from aggregated predictions (e.g. mean or ensemble).
    
    For each (project, group, room), first aggregates predictions across offsets (for each timestamp),
    e.g. by mean or ensemble, then calculates metrics from aggregated predictions vs targets.
    Finally adds 'All' room as mean over rooms' metrics.
    
    Args:
        results_df: DataFrame with one row per timestamp, columns:
            - project, group, version, set, timestamp, offset, fold, room
            - target columns (e.g., 'x', 'y')
            - prediction columns (e.g., 'x_pred', 'y_pred')
        target_columns: list of target column names
        pred_columns: list of prediction column names
    
    Returns:
        DataFrame with columns: project, group, version, room, n_timestamps, n_folds, n_offsets,
        r2_x, r2_y, r2_pooled, mse, rmse, sse_x, sse_y, sst_x, sst_y
        Note: Includes individual rooms and "All" room row (mean over rooms)

    """
    if len(results_df) == 0:
        return pd.DataFrame()
    
    # Filter out "All" room for now - we'll compute it separately
    results_df_filtered = results_df[results_df['room'] != 'All'].copy()
    
    # Get fold and offset counts from original data before aggregation
    fold_offset_counts = results_df_filtered.groupby(['project', 'group', 'version', 'set', 'room']).agg({
        'fold': 'nunique' if 'fold' in results_df_filtered.columns else lambda x: 0,
        'offset': 'nunique' if 'offset' in results_df_filtered.columns else lambda x: 0
    }).reset_index()
    fold_offset_counts.columns = ['project', 'group', 'version', 'set', 'room', 'n_folds', 'n_offsets']
    
    # Group by (project, group, version, set, room, timestamp) and aggregate predictions across offsets
    # Targets should be the same for all offsets, so we take the first
    grouped = results_df_filtered.groupby(['project', 'group', 'version', 'set', 'room', 'timestamp'])
    
    # Aggregate: mean for predictions, first for targets (they should be identical)
    agg_dict = {col: 'first' for col in target_columns}
    agg_dict.update({col: 'mean' for col in pred_columns})
    
    mean_df = grouped.agg(agg_dict).reset_index()
    
    # Now group by (project, group, version, set, room) to compute metrics
    rows = []
    for (project, group, version, set_name, room), group_df in mean_df.groupby(['project', 'group', 'version', 'set', 'room']):
        targets = group_df[target_columns].values
        predictions = group_df[pred_columns].values
        
        # Calculate metrics from mean predictions
        metrics = calculate_metrics(targets, predictions, target_columns=target_columns)
        
        # add rmse to metrics
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # Get fold and offset counts from pre-computed data
        counts = fold_offset_counts[
            (fold_offset_counts['project'] == project) &
            (fold_offset_counts['group'] == group) &
            (fold_offset_counts['version'] == version) &
            (fold_offset_counts['set'] == set_name) &
            (fold_offset_counts['room'] == room)
        ]
        n_folds = int(counts['n_folds'].iloc[0]) if len(counts) > 0 else 0
        n_offsets = int(counts['n_offsets'].iloc[0]) if len(counts) > 0 else 0
        
        # Add row for this room
        rows.append({
            'project': project,
            'group': group,
            'version': version,
            'set': set_name,
            'room': room,
            'n_timestamps': len(group_df),
            'n_folds': n_folds,
            'n_offsets': n_offsets,
            **metrics  # Unpack all metrics
        })
    
    # Compute "All" room as mean over rooms' metrics (not pooled SSE/SST), grouped by set
    if rows:
        # Group by (project, group, version, set) to compute "All" room for each set separately
        rows_df = pd.DataFrame(rows)
        for (project, group, version, set_name), set_df in rows_df.groupby(['project', 'group', 'version', 'set']):
            # Calculate mean metrics across rooms for this set
            first_row = set_df.iloc[0]
            
            # Build all_metrics dynamically based on target_columns
            all_metrics = {
                'project': first_row['project'],
                'group': first_row['group'],
                'version': first_row['version'],
                'set': set_name,
                'room': 'All',
                'n_timestamps': int(set_df['n_timestamps'].sum()),
                'n_samples': int(set_df['n_samples'].sum()) if 'n_samples' in set_df.columns else int(set_df['n_timestamps'].sum()),
                'n_folds': int(set_df['n_folds'].max()),
                'n_offsets': int(set_df['n_offsets'].max()),
                'r2_pooled': set_df['r2_pooled'].mean(),
                'mse': set_df['mse'].mean(),
                'rmse': set_df['rmse'].mean()
            }
            
            # Add per-column metrics dynamically based on target_columns (columns exist from calculate_metrics)
            for col in target_columns:
                all_metrics.update({
                    f'r2_{col}': set_df[f'r2_{col}'].mean(),
                    f'sse_{col}': set_df[f'sse_{col}'].mean(),
                    f'sst_{col}': set_df[f'sst_{col}'].mean()
                })
            
            rows.append(all_metrics)
    
    return pd.DataFrame(rows)
