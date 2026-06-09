"""Cross-room affine remapping: metrics for ``mapping_stats.csv`` and plot inputs for ``nn_mapping.png``."""

import os
import time
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.prepared import prep

from utils.affine_transforms import find_linear_transformation, extract_affine_components
from utils.analysis import get_boundary_points_from_csv, interpolate_boundary_points
from utils.config import assign_room_column, get_rooms_from_config, get_target_columns_from_config
from utils.data_loader import map_rooms_to_indices
from utils.helpers import (
    apply_scaler_transform,
    format_room_name,
    get_directory,
    get_prediction_columns,
    save_data_to_csv,
)
from utils.logger import get_logger, log_boundary_points, log_completion_message
from utils.metrics import calculate_metrics, load_position_scaler_from_config


# ============================================================================
# Configuration Extraction
# ============================================================================
def extract_xy2xy_config(config):
    """
    Extract and validate XY2XY configuration from config dict.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        dict: Extracted XY2XY configuration with defaults
    """
    xy2xy_config = config.get('xy2xy', {})

    return {
        'allowed_pairs': xy2xy_config.get('allowed_pairs', [('A', 'a'), ('A', 'B'), ('B', 'a')]),
        'min_points_required': xy2xy_config.get('min_points_required', 10),
    }
def filter_points_in_polygons(source_xy, target_xy, polygon_source, polygon_target):
    """
    Vectorized polygon containment check using prepared geometries.
    Returns boolean mask for points that are inside both polygons.
    
    Args:
        source_xy: Array of source points (n, 2)
        target_xy: Array of target points (n, 2)
        polygon_source: Shapely Polygon for source room
        polygon_target: Shapely Polygon for target room
        
    Returns:
        tuple: (filter_source, filter_target, filter_overlap) - all boolean arrays
    """
    # Use prepared geometries for faster repeated queries
    prep_polygon_source = prep(polygon_source)
    prep_polygon_target = prep(polygon_target)
    
    # Vectorized containment check - still need to iterate but prepared geometries are faster
    filter_source = np.array([prep_polygon_source.contains(Point(pxy)) for pxy in source_xy], dtype=bool)
    filter_target = np.array([prep_polygon_target.contains(Point(nxy)) for nxy in target_xy], dtype=bool)
    
    # Points must be in both polygons
    filter_overlap = filter_source & filter_target
    
    return filter_source, filter_target, filter_overlap


def extract_room_pair_data(df_data, source_room, target_room, config, xy2xy_config, data_room_filter='both'):
    """
    Extract and validate data for a single room pair.
    Uses a room filter and subset so row alignment is preserved regardless of index (e.g. duplicate timestamps).

    Args:
        df_data: DataFrame with predictions (already filtered by offset if needed)
        source_room: Source room identifier
        target_room: Target room identifier
        config: Full configuration dictionary
        xy2xy_config: Extracted XY2XY configuration
        data_room_filter: 'both' | 'source' | 'target' — which rows to include by room.

    Returns:
        dict: 'source_xy', 'target_xy', 'room_values', 'total_points', boundary/pos_range keys; None if insufficient.
    """
    logger = get_logger(__name__)
    rooms = get_rooms_from_config(config)
    target_columns = get_target_columns_from_config(config)
    pred_target_columns = {room: get_prediction_columns(target_columns, suffix=f'pred_{room}') for room in rooms}
    pred_columns_list = [col for _, cols in pred_target_columns.items() for col in cols]

    df = df_data.copy()
    existing_pred = [c for c in pred_columns_list if c in df.columns]
    if existing_pred:
        df = df.dropna(subset=existing_pred)

    if data_room_filter == 'source':
        room_mask = (df['room'] == source_room)
    elif data_room_filter == 'target':
        room_mask = (df['room'] == target_room)
    else:
        room_mask = df['room'].isin([source_room, target_room])

    subset = df.loc[room_mask]
    if len(subset) == 0:
        logger.warning(f'No rows for room filter {data_room_filter} (source={source_room}, target={target_room}). Skipping.')
        return None

    source_pred_cols = [c for c in pred_target_columns.get(source_room, []) if c in df.columns]
    target_pred_cols = [c for c in pred_target_columns.get(target_room, []) if c in df.columns]
    if not source_pred_cols or not target_pred_cols:
        logger.warning(f'Missing prediction columns for {source_room} or {target_room}. Skipping this room pair.')
        return None

    source_xy = subset[source_pred_cols].values
    target_xy = subset[target_pred_cols].values
    room_values = subset['room'].values
    total_points = len(source_xy)

    if total_points < xy2xy_config['min_points_required']:
        logger.warning(f'Insufficient points ({total_points} < {xy2xy_config["min_points_required"]}) for {source_room}->{target_room}. Skipping.')
        return None

    boundary_points = get_boundary_points_from_csv(config)
    boundary_points_source = get_boundary_points_from_csv(config, room=source_room)
    boundary_points_target = get_boundary_points_from_csv(config, room=target_room)
    pos_range = (np.min(boundary_points[:, :2]), np.max(boundary_points[:, :2]))

    return {
        'source_xy': source_xy,
        'target_xy': target_xy,
        'room_values': room_values,
        'total_points': total_points,
        'boundary_points_source': boundary_points_source,
        'boundary_points_target': boundary_points_target,
        'pos_range': pos_range,
    }
def compute_affine_transformation(source_xy, target_xy, boundary_points_source, boundary_points_target):
    """
    Compute affine transformation from source to target in canonical frame.
    
    Args:
        source_xy: Source coordinates (already in canonical frame)
        target_xy: Target coordinates (already in canonical frame)
        boundary_points_source: Source boundary points (not yet canonical)
        boundary_points_target: Target boundary points (not yet canonical)
        
    Returns:
        dict: Contains 'A_canon', 'trans_properties', 'source_centroid', 'target_centroid', 
              'boundary_points_source_canon', 'boundary_points_target_canon'
    """
    # Create polygons for centroid computation
    polygon_source_room = Polygon(boundary_points_source[:,:2])
    polygon_target_room = Polygon(boundary_points_target[:,:2])
    
    # Compute polygon centroids
    c_src = np.array(polygon_source_room.centroid.coords[0])  # Shape: (2,)
    c_tgt = np.array(polygon_target_room.centroid.coords[0])  # Shape: (2,)
    
    # Center boundary points (for later use in polygon operations)
    boundary_points_source_canon = boundary_points_source[:, :2] - c_src
    boundary_points_target_canon = boundary_points_target[:, :2] - c_tgt
    
    # Fit affine using canonical (centered) points in homogeneous coordinates
    A_canon, transform, inverse_transform = find_linear_transformation(source_xy, target_xy)
    
    # Calculate physical transform if needed (for reconstruction/visualization)
    T_c_tgt = np.eye(3)
    T_c_tgt[:2, 2] = c_tgt
    T_minus_c_src = np.eye(3)
    T_minus_c_src[:2, 2] = -c_src
    A_phys = T_c_tgt @ A_canon @ T_minus_c_src
    
    # Extract 2x2 linear part (L_canon) from 3x3 affine matrix (A_canon) for decomposition
    L_canon = A_canon[:2, :2]
    
    # Calculate the transformation properties using canonical transform
    trans_properties = extract_affine_components(A_canon, center=np.array([0.0, 0.0]))
    
    # Verify that translation matrices are trivial (identity) in canonical frame
    to_origin = trans_properties['to_origin_translation_matrix']
    back_trans = trans_properties['back_translation_matrix']
    identity_3x3 = np.eye(3)
    
    if not np.allclose(to_origin, identity_3x3, atol=1e-10):
        raise ValueError(
            f"to_origin_translation_matrix is not identity in canonical frame! "
            f"Expected identity, got:\n{to_origin}\n"
            f"This indicates a bug in extract_affine_components or incorrect center parameter."
        )
    
    if not np.allclose(back_trans, identity_3x3, atol=1e-10):
        raise ValueError(
            f"back_translation_matrix is not identity in canonical frame! "
            f"Expected identity, got:\n{back_trans}\n"
            f"This indicates a bug in extract_affine_components or incorrect center parameter."
        )
    
    # Remove trivial matrices from storage
    del trans_properties['to_origin_translation_matrix']
    del trans_properties['back_translation_matrix']
    
    # Store centroids (centers of mass) for later interpretations and plotting
    trans_properties['com_source'] = c_src
    trans_properties['com_target'] = c_tgt
    
    trans_properties['affine_matrix'] = A_canon

    return {
        'A_canon': A_canon,
        'trans_properties': trans_properties,
        'source_centroid': c_src,
        'target_centroid': c_tgt,
        'boundary_points_source_canon': boundary_points_source_canon,
        'boundary_points_target_canon': boundary_points_target_canon,
        'transform': transform,
        'inverse_transform': inverse_transform
    }


# ============================================================================
# Metrics Computation
# ============================================================================
def compute_fit_quality_metrics(source_xy, target_xy, A_canon, config):
    """Affinity R² and RMSE for the affine fit in canonical coordinates."""
    source_xy_homog = np.hstack([source_xy, np.ones((source_xy.shape[0], 1))])
    target_xy_pred = (A_canon @ source_xy_homog.T).T[:, :2]
    residual_norms = np.linalg.norm(target_xy - target_xy_pred, axis=1)
    mse = np.mean(residual_norms ** 2)
    metrics = calculate_metrics(target_xy, target_xy_pred, target_columns=['x', 'y'])
    scaler = load_position_scaler_from_config(config)
    rmse_cm = (
        apply_scaler_transform(np.sqrt(mse), scaler, scale_only=True, reverse=True)
        if scaler is not None
        else np.nan
    )
    return {
        'affinity_r2': metrics['r2_pooled'],
        'rmse_cm': rmse_cm,
        'residual_median': float(np.median(residual_norms)),
    }


PUBLICATION_STATS_COLUMNS = [
    'source_room',
    'target_room',
    'affinity_r2',
    'rmse_cm',
    'angle_deg',
    'reflection',
    'max_eigenvalue',
    'min_eigenvalue',
]


def _scalar_for_csv(value):
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        if np.isnan(value) or pd.isna(value):
            return None
        return float(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        if value.ndim == 0:
            return _scalar_for_csv(value.item())
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_scalar_for_csv(v) for v in value]
    return value


def _build_publication_metrics_row(source_room, target_room, trans_props):
    row = {
        'source_room': source_room,
        'target_room': target_room,
        'affinity_r2': trans_props.get('affinity_r2'),
        'rmse_cm': trans_props.get('rmse_cm'),
        'angle_deg': trans_props.get('angle_deg'),
        'reflection': trans_props.get('reflection'),
        'max_eigenvalue': trans_props.get('max_eigenvalue'),
        'min_eigenvalue': trans_props.get('min_eigenvalue'),
    }
    return {k: _scalar_for_csv(row[k]) for k in PUBLICATION_STATS_COLUMNS}


def _analyze_room_pair(
    df_data,
    source_room,
    target_room,
    config,
    data_room_filter='both',
):
    """Analyze one room pair; return metrics row dict and mapping_data for nn_mapping.png."""
    logger = get_logger(__name__)
    logger.info('=' * 80)
    title = f'RUN XY2XY MODEL: {format_room_name(source_room)} -> {format_room_name(target_room)}'
    logger.info(f'{title:^80}')
    logger.info('=' * 80)
    
    xy2xy_config = extract_xy2xy_config(config)
    keys = (source_room, target_room)
    
    logger.info('Step 0: Data preparation - extracting room pair data and filtering points...')
    data = extract_room_pair_data(
        df_data, source_room, target_room, config, xy2xy_config, data_room_filter=data_room_filter
    )
    if data is None:
        return None
    
    source_xy = data['source_xy']
    target_xy = data['target_xy']
    total_points = data['total_points']
    boundary_points_source = data['boundary_points_source']
    boundary_points_target = data['boundary_points_target']

    polygon_source_room = Polygon(boundary_points_source[:, :2])
    polygon_target_room = Polygon(boundary_points_target[:, :2])

    filter_source_points_indices, filter_target_points_indices, filter_overlap_points_indices = (
        filter_points_in_polygons(source_xy, target_xy, polygon_source_room, polygon_target_room)
    )
    total_overlap_points = int(np.sum(filter_overlap_points_indices))
    total_points_percentage = (
        total_overlap_points / total_points if total_points > 0 else 0.0
    )

    logger.info(f'Total Points: {total_points}')
    logger.info(
        f'Points in {format_room_name(source_room)}: {np.sum(filter_source_points_indices)} '
        f'({np.sum(filter_source_points_indices) / total_points:.2%})'
    )
    logger.info(
        f'Points in {format_room_name(target_room)}: {np.sum(filter_target_points_indices)} '
        f'({np.sum(filter_target_points_indices) / total_points:.2%})'
    )
    logger.info(f'Total Overlap Points: {total_overlap_points} ({total_points_percentage:.2%})')

    source_xy = source_xy[filter_overlap_points_indices]
    target_xy = target_xy[filter_overlap_points_indices]

    logger.info('Step 1: Canonicalization - computing polygon centroids and centering points...')
    source_centroid = np.array(polygon_source_room.centroid.coords[0])
    target_centroid = np.array(polygon_target_room.centroid.coords[0])
    c_src = source_centroid
    c_tgt = target_centroid
    source_xy = source_xy - c_src
    target_xy = target_xy - c_tgt

    arrow_points = np.array([[0, 0], [0, -1.0], [0, 0], [1.0, 0]]) * 0.15 + source_centroid

    if len(source_xy) <= 1 or len(target_xy) <= 1 or len(source_xy) != len(target_xy):
        return None

    logger.info(
        f'Step 2: Finding linear transformation in canonical frame between '
        f'{format_room_name(source_room)} and {format_room_name(target_room)}...'
    )
    transformation = compute_affine_transformation(
        source_xy, target_xy, boundary_points_source, boundary_points_target
    )
    A_canon = transformation['A_canon']
    trans_properties = transformation['trans_properties']
    transform = transformation['transform']

    logger.info('Step 3: Computing affine fit quality metrics...')
    fit_quality_metrics = compute_fit_quality_metrics(source_xy, target_xy, A_canon, config)
    trans_properties.update(fit_quality_metrics)
    r2 = fit_quality_metrics['affinity_r2']
    rmse_cm = fit_quality_metrics['rmse_cm']
    max_eigenvalue = trans_properties['max_eigenvalue']
    min_eigenvalue = trans_properties['min_eigenvalue']
    logger.info(
        f'Fit Quality - Affinity R²: {r2:.4f}, RMSE: {rmse_cm:.2f} cm, '
        f'Residual median: {fit_quality_metrics["residual_median"]:.4f}'
    )

    transformed_arrow_points = transform(arrow_points)

    mapping_data_dict = {
        'arrows': {'prev': arrow_points, 'new': transformed_arrow_points},
        'source_xy': source_xy,
        'target_xy': target_xy,
        'com_source': source_centroid,
        'com_target': target_centroid,
    }

    angle = round(trans_properties['angle_deg'], 1)
    reflection = trans_properties['reflection']
    logger.info(
        f'Summary - Affinity R²: {r2:.4f} ({r2:.2%}), RMSE: {rmse_cm:.2f} cm, '
        f'Angle: {angle}°, Reflection: {reflection}, '
        f'Eigenvalues: [{max_eigenvalue:.4f}, {min_eigenvalue:.4f}]'
    )

    metrics_log_dict = _build_publication_metrics_row(
        source_room, target_room, trans_properties
    )
    return {
        'metrics': metrics_log_dict,
        'mapping_data': {keys: mapping_data_dict},
    }


def main(config, data_room_filter='both', offset_aggregation='all_offsets'):
    """
    Publication xy2xy analysis: flat ``mapping_stats.csv`` and in-memory mapping_data
    for ``nn_mapping.png`` (rendered in runner/stages/xy2xy.py).
    """
    logger = get_logger(__name__)

    output_folder = get_directory(config, 'output')
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, 'data_pred.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f'{file_path} not found. Run predict (or train:predict) before xy2xy.'
        )

    run_metadata = config.get('metadata') or {}
    project_name = run_metadata.get('session', 'UnknownProject')
    start_time = time.time()

    df_data = pd.read_csv(file_path)
    rooms = get_rooms_from_config(config)
    if 'room' not in df_data.columns:
        df_data = assign_room_column(df_data, config, room_column='room')
    map_rooms_to_indices(config, df_data)
    boundary_points = get_boundary_points_from_csv(config)
    log_boundary_points(interpolate_boundary_points(boundary_points)[:, :2])
    pos_range = (np.min(boundary_points[:, :2]), np.max(boundary_points[:, :2]))

    xy2xy_config = extract_xy2xy_config(config)
    allowed_pairs = xy2xy_config['allowed_pairs']
    pairs_of_rooms = [tuple(pair) for pair in allowed_pairs if pair[0] in rooms and pair[1] in rooms]

    logger.info('=' * 80)
    logger.info('Running XY2XY stage (publication)...')
    logger.info('Data room filter: %s', data_room_filter)
    logger.info('Offset aggregation: %s', offset_aggregation)
    logger.info('Allowed pairs of rooms: %s', pairs_of_rooms)
    logger.info('Rooms: %s', rooms)
    logger.info('=' * 80)

    if offset_aggregation != 'all_offsets':
        logger.warning(
            "Publication recipe uses offset_aggregation='all_offsets'; got %r.",
            offset_aggregation,
        )

    df_all = df_data.copy()
    if 'timestamp' in df_all.columns:
        if 'offset' in df_all.columns:
            df_all = df_all.set_index(['timestamp', 'offset'])
        else:
            df_all = df_all.set_index('timestamp')

    all_results = []
    for keys in pairs_of_rooms:
        source_room, target_room = keys
        result = _analyze_room_pair(
            df_data=df_all,
            source_room=source_room,
            target_room=target_room,
            config=config,
            data_room_filter=data_room_filter,
        )
        if result is not None:
            all_results.append(result)

    if all_results:
        metric_rows = [r['metrics'] for r in all_results if r is not None and 'metrics' in r]
        if metric_rows:
            df_mapping_stats = pd.DataFrame(metric_rows)[PUBLICATION_STATS_COLUMNS]
            save_data_to_csv(
                config,
                df_mapping_stats,
                output_file='mapping_stats.csv',
                overwrite=True,
            )
            logger.info(
                'Wrote mapping_stats.csv (%d room-pair rows) to session output root.',
                len(df_mapping_stats),
            )

    log_completion_message(start_time)

    mapping_data_all = {}
    for result in all_results:
        if result and 'mapping_data' in result:
            mapping_data_all.update(result['mapping_data'])
    return {
        'mapping_data': mapping_data_all,
        'boundary_points': boundary_points,
        'pos_range': pos_range,
        'project_name': project_name,
    }
