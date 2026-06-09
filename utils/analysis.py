import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
import os

from sklearn.metrics import r2_score
from tqdm import tqdm

from utils.helpers import get_directory, get_dataset_path
import joblib  # or import pickle
from shapely.geometry import Point, LineString, Polygon
from shapely.prepared import prep

from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from collections import defaultdict
import numpy as np

from scipy.ndimage import distance_transform_edt, label


def get_boundary_points_from_csv(config, room=None, use_normalize_units=True):
    """
    Load boundary points from a CSV file, optionally for a specific room, and convert them into a 2D array.

    Parameters:
        config (dict): The configuration dictionary containing paths to necessary files and settings.
        room (str, optional): The room key to filter boundary points. If None, all boundary points are returned.
        use_normalize_units (bool): 
            - True: Normalize boundary points TO 0-1 units (uses scaler.transform, cm → normalized)
            - False: Keep boundary points in cm (original units)
            Default is True.
            Note: Boundary points in config are stored in cm (original units). See docs/data_format.md for details.

    Returns:
        np.ndarray: A 2D array of shape (n_points, 3) where each row contains
                    [X, Y, Room Key].
    """
    # # Get the path to the boundary dataset from the config
    # boundary_dataset_path = get_dataset_path(config, 'boundary')
    #
    # # Load the boundary dataset CSV into a DataFrame
    # boundary_df = pd.read_csv(boundary_dataset_path)

    # Get boundary dataset from config
    boundary = config.get('preprocessing', {}).get('boundary', None)
    if boundary is None: raise ValueError("Boundary dataset not found in config['preprocessing']['boundary']")
    boundary_df = pd.DataFrame(boundary)

    # Ensure the CSV has the expected columns: 'Room', 'X', 'Y'
    if not {'Room', 'X', 'Y'}.issubset(boundary_df.columns):
        raise ValueError("CSV must contain 'Room', 'X', and 'Y' columns")

    # Convert the DataFrame to a NumPy array
    # Note: Boundary points in CSV are stored in cm (original units)
    boundary_array = boundary_df[['X', 'Y', 'Room']].to_numpy()

    # Load scaler only when normalization is requested.
    # This avoids unnecessary unpickle/version warnings in raw-units workflows.
    scaler = None
    if use_normalize_units:
        try:
            scaler_path = os.path.join(get_directory(config, 'data'), 'positions_scaler.pkl')
            scaler_pkl = joblib.load(scaler_path)  # Load the scaler using joblib
            scaler = scaler_pkl['scaler']
        except Exception:
            pass  # Scaler loading is optional

    if scaler is not None:
        if use_normalize_units:
            # use_normalize_units=True: Normalize the X and Y coordinates TO 0-1 units
            from utils.helpers import apply_scaler_transform
            normalized_points = apply_scaler_transform(boundary_array[:, :2], scaler, reverse=False)
            boundary_array[:, :2] = normalized_points
        else:
            # use_normalize_units=False: Keep boundary points in cm (original units)
            pass

    # If a specific room is provided
    if room:
        maps_rooms = config.get('preprocessing', {}).get('map_rooms', {}).get('rooms', {})
        if room not in maps_rooms:
            raise ValueError(f"Room '{room}' not found in config['preprocessing']['map_rooms']['rooms']")

        # Get the specific room index
        room_labels = maps_rooms[room]['index']

        # Return the boundary points for the specific room
        return boundary_array[boundary_array[:, 2] == room_labels]

    return boundary_array


def get_boundary_polygon_from_config(config, room, use_normalize_units=True):
    """Build a Shapely Polygon for *room*, optionally with holes from ``exclude_overlap``.

    Reads the room's boundary points as the exterior ring.  If the room's
    entry in ``config['preprocessing']['map_rooms']['rooms']`` contains an
    ``exclude_overlap`` list (e.g. ``['B']``), each listed room's boundary
    is used as a hole, producing a polygon with interior cutouts.

    Parameters:
        config (dict): Project configuration (must contain preprocessing.boundary).
        room (str): Room label (e.g. ``'A'``).
        use_normalize_units (bool): Passed through to ``get_boundary_points_from_csv``.

    Returns:
        shapely.geometry.Polygon: Valid polygon (possibly with holes).

    Raises:
        ValueError: If the room has fewer than 3 boundary points.
    """
    pts = get_boundary_points_from_csv(config, room=room, use_normalize_units=use_normalize_units)
    if pts is None or len(pts) < 3:
        raise ValueError(f"Room '{room}' has fewer than 3 boundary points.")
    exterior = pts[:, :2]

    rooms_cfg = config.get('preprocessing', {}).get('map_rooms', {}).get('rooms', {})
    exclude = rooms_cfg.get(room, {}).get('exclude_overlap', []) or []

    holes = []
    for hole_room in exclude:
        hole_pts = get_boundary_points_from_csv(config, room=hole_room, use_normalize_units=use_normalize_units)
        if hole_pts is not None and len(hole_pts) >= 3:
            holes.append(hole_pts[:, :2])

    poly = Polygon(exterior, holes) if holes else Polygon(exterior)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def interpolate_boundary_points(boundary_points, num_points=100):
    """
    Interpolate boundary points within each group.

    Parameters:
        boundary_points (numpy.ndarray): Array of boundary points with shape (n, 3),
                                          where columns are [x, y, group].
        num_points (int): Number of interpolation points between each pair of neighboring points.

    Returns:
        numpy.ndarray: Interpolated boundary points with shape (m, 3), where m depends on the number of groups.
    """
    groups = np.unique(boundary_points[:, 2])  # Get unique group IDs
    interpolated_points = []

    for group in groups:
        # Filter points for the current group
        group_points = boundary_points[boundary_points[:, 2] == group]

        # Close the loop: include the first point again at the end
        group_points = np.vstack([group_points, group_points[0]])

        # Interpolate between consecutive points
        for i in range(len(group_points) - 1):
            p1, p2 = group_points[i], group_points[i + 1]
            x_interp = np.linspace(p1[0], p2[0], num_points, endpoint=False)
            y_interp = np.linspace(p1[1], p2[1], num_points, endpoint=False)
            group_interp = np.full_like(x_interp, group)  # Keep group ID constant

            # Combine x, y, and group
            segment = np.column_stack([x_interp, y_interp, group_interp])
            interpolated_points.append(segment)

    # Concatenate all interpolated points
    return np.vstack(interpolated_points)

def get_boundary_points_from_trajectory(df, perspective='allo', delta=0):
    """
    Calculate and return boundary points from the given dataset, using either
    allocentric or egocentric distances depending on the coordinate system.

    Args:
        df (pd.DataFrame): DataFrame with 'X', 'Y' for positions and distances (allo or ego).
        coordinate_system (str): 'allo' for allocentric distances, 'ego' for egocentric. Default is 'allo'.

    Returns:
        np.ndarray: Array of boundary points.
    """

    def calculate_boundary_points(distances, positions):
        """
        Calculate boundary points by adding allocentric coordinates to positions.

        Args:
            distances (np.ndarray): Numpy array of shape (timestamps, 2) containing allocentric (x, y) coordinates.
            positions (np.ndarray): Numpy array of shape (timestamps, 2) containing original (x, y) positions.

        Returns:
            np.ndarray: Array of calculated boundary points (timestamps, 2).
        """
        # Add allocentric coordinates to positions
        boundary_points = positions + distances
        return boundary_points.reshape((-1,2))

    def calculate_distances(polar_distances, angles, head_direction=None):
        """
        Convert egocentric distances to allocentric (x, y) points using the head direction.

        Args:
            ego_distances (np.ndarray): Array of ego distances (timestamps, angles).
            head_direction (np.ndarray): Array of head direction angles (in degrees) (timestamps).

        Returns:
            np.ndarray: Array of allocentric (x, y) points (timestamps, 2).
        """
        num_timestamps, num_angles = polar_distances.shape
        x_coords = np.zeros((num_timestamps, num_angles))
        y_coords = np.zeros((num_timestamps, num_angles))

        for i in range(num_timestamps):
            hd = head_direction[i] if head_direction is not None else 0
            allo_angles = angles - hd  # Adjust ego angles by head direction

            # Compute the x, y positions based on allocentric angles and ego distances
            x_coords[i,:] = polar_distances[i] * np.cos(np.deg2rad(allo_angles))
            y_coords[i,:] = polar_distances[i] * np.sin(np.deg2rad(allo_angles))

        # Return an array with shape (angels,timestamps, 2)
        return np.array([x_coords, y_coords]).transpose((2,1,0))

    positions = df[['X', 'Y']].values
    if perspective == "allo":
        allo_columns = [col for col in df.columns if col.startswith('Allo')]
        allo_distances = df[allo_columns].values
        angles = np.array([int(col.split('_')[1]) for col in allo_columns])
        if delta: angles = angles + delta
        distances = calculate_distances(allo_distances, angles)
    elif perspective == "ego":
        # Convert egocentric distances to allocentric distances
        ego_columns = [col for col in df.columns if col.startswith('Ego')]
        ego_distances = df[ego_columns].values
        angles = np.array([int(col.split('_')[1]) for col in ego_columns])
        if delta: angles = angles+delta
        head_direction = df['HD'].values  # Head direction in degrees
        distances = calculate_distances(ego_distances, angles, head_direction)

    boundary_points = calculate_boundary_points(distances,positions)

    return boundary_points


def create_mse_heatmap(df_data, target_columns, prediction_columns, smoothed=False, sigma = 2, kernel_size = 8, n_pixel=34, pos_range=None, fill_value=np.nan):
    """
    Create a heatmap of Mean Squared Errors (MSE) based on spatial positions.

    Args:
        df_data (pandas.DataFrame): DataFrame containing columns for spatial positions, target values, and predictions.
        target_columns (list of str): Names of the columns containing the actual target values.
        prediction_columns (list of str): Names of the columns containing the predicted values.
        smoothed (bool, optional): If True, apply Gaussian smoothing to the heatmap. Default is False.
        sigma (float, optional): Standard deviation of the Gaussian kernel used for smoothing. Default is 2.
        kernel_size (int, optional): Size of the Gaussian kernel used for smoothing (width and height in pixels). Default is 8.
        n_pixel (int, optional): Number of bins for the x and y grids of the heatmap. Default is 34.
        pos_range (tuple, optional): Range of x and y positions for the grid. If None, it is determined from the data. Default is None.
        fill_value (float, optional): Value to initialize heatmaps with. Default is np.nan.

    Returns:
        np.ndarray: Heatmap of the MSE values (smoothed if `smoothed` is True).
        np.ndarray: x-coordinates grid for the heatmap.
        np.ndarray: y-coordinates grid for the heatmap.
    """

    # Extract target and prediction values
    actual_targets = df_data[target_columns].values
    predicted_targets = df_data[prediction_columns].values

    # Extract spatial positions and calculate normalized MSE for each position
    positions = df_data[['X', 'Y']].values
    normalized_mse  = np.mean((actual_targets - predicted_targets) ** 2, axis=1) / actual_targets.shape[1]

    # Define the range for the position grid
    if pos_range is None:
        pos_range = (np.min(positions), np.max(positions))

    # Generate grids for x and y positions
    x_grid = np.linspace(pos_range[0], pos_range[1], n_pixel)
    y_grid = np.linspace(pos_range[0], pos_range[1], n_pixel)

    # Initialize heatmaps with fill_value
    mse_heatmap = np.full((len(y_grid), len(x_grid)),fill_value)
    normalized_heatmap = np.full((len(y_grid), len(x_grid)),fill_value)

    # Populate heatmaps with MSE values for each grid cell
    for j in range(len(x_grid) - 1):
        for k in range(len(y_grid) - 1):
            # Find instances within the current grid cell
            instance_in_cell = (positions[:, 0] >= x_grid[j]) & (positions[:, 0] < x_grid[j + 1]) & \
                               (positions[:, 1] >= y_grid[k]) & (positions[:, 1] < y_grid[k + 1])
            sum_instances_in_cell = np.sum(instance_in_cell)

            mse_in_cell = normalized_mse[instance_in_cell]

            if sum_instances_in_cell > 0:
                mse_heatmap[k, j] = np.sum(mse_in_cell)
                normalized_heatmap[k, j] = mse_heatmap[k, j] / sum_instances_in_cell

    if smoothed:
        # Define Gaussian kernel for smoothing
        X_kernel, Y_kernel = np.meshgrid(np.arange(-kernel_size / 2, kernel_size / 2 + 1),
                                         np.arange(-kernel_size / 2, kernel_size / 2 + 1))
        kernel = np.exp(-(X_kernel ** 2 + Y_kernel ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
        kernel = kernel / np.sum(kernel)
        # Apply convolution to smooth the heatmap
        smoothed_heatmap = convolve2d(normalized_heatmap, kernel, mode='same')
        return smoothed_heatmap, x_grid, y_grid
    else:
        return normalized_heatmap, x_grid, y_grid


def create_rate_map(XY_positions, spike_count, dt, min_fr_bin_treshold_s=None, min_spike_treshold=None, smoothed=False, sigma = 2, kernel_size = 8, n_pixel=34, pos_range=None, room_boundary_points=None, fill_value=np.nan):
    """
    Calculate rate map of spike rates based on positions and firing rates.

    Parameters:
        XY_positions (np.ndarray): Array of shape (N, 2) with X and Y positions.
        firing_rates (np.ndarray): Array of shape (N,) with firing rates corresponding to the positions.
        dt (float): Time duration for each position sample.
        min_fr_bin_treshold_s (float): Minimum firing rate threshold per bin in seconds. Default is 0.2.
        min_spike_treshold (int): Minimum spike count threshold per bin. Default is 2.
        smoothed (bool, optional): If True, apply Gaussian smoothing to the heatmap. Default is False.
        sigma (float, optional): Standard deviation of the Gaussian kernel used for smoothing. Default is 2.
        kernel_size (int, optional): Size of the Gaussian kernel used for smoothing (width and height in pixels). Default is 8.
        n_pixel (int, optional): Number of pixels in the grid for the heatmap. Default is 34.
        pos_range (tuple, optional): Tuple (min, max) for the xy axis range. Default is None.
        room_boundary_points (np.ndarray, optional): Array of shape (M, 2) with room boundary points. Default is None.
        fill_value (float, optional): Value to fill NaN values in the heatmap. Default is np.nan.

    Returns:
        np.ndarray: Normalized heatmap of spike rates.
        np.ndarray: X grid for the heatmap.
        np.ndarray: Y grid for the heatmap
    """
    if pos_range is None:
        pos_range = (np.min(XY_positions), np.max(XY_positions))

    x_grid = np.linspace(pos_range[0], pos_range[1], n_pixel, dtype=np.float64)
    y_grid = np.linspace(pos_range[0], pos_range[1], n_pixel, dtype=np.float64)

    # Initialize heatmaps
    spike_count_heatmap = np.full((len(y_grid), len(x_grid)), fill_value, dtype=np.float64)
    normalized_heatmap = np.full((len(y_grid), len(x_grid)), fill_value, dtype=np.float64)
    occupancy_heatmap = np.full((len(y_grid), len(x_grid)), fill_value, dtype=np.float64)

    # Compute heatmaps
    for j in range(len(x_grid) - 1):
        for k in range(len(y_grid) - 1):
            in_bin = (XY_positions[:, 0] >= x_grid[j]) & (XY_positions[:, 0] < x_grid[j + 1]) & \
                     (XY_positions[:, 1] >= y_grid[k]) & (XY_positions[:, 1] < y_grid[k + 1])
            count = np.sum(in_bin)

            if (count > 0
                    and ((min_fr_bin_treshold_s is None) or (count * dt >= min_fr_bin_treshold_s))
                    and ((min_spike_treshold is None) or (np.sum(spike_count[in_bin]) >= min_spike_treshold))):
                spike_count_heatmap[k, j] = np.sum(spike_count[in_bin])
                occupancy_heatmap[k, j] = count * dt
                normalized_heatmap[k, j] = spike_count_heatmap[k, j] / occupancy_heatmap[k, j]

    spike_rate_heatmap = normalized_heatmap

    if smoothed and sigma > 0 and kernel_size > 0:
        # Define Gaussian kernel for smoothing
        X_kernel, Y_kernel = np.meshgrid(np.arange(-kernel_size / 2, kernel_size / 2 + 1, dtype=np.float64),
                                         np.arange(-kernel_size / 2, kernel_size / 2 + 1, dtype=np.float64))
        kernel = np.exp(-(X_kernel ** 2 + Y_kernel ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
        kernel = kernel / np.sum(kernel)
        # Apply convolution to smooth the heatmap
        if np.isnan(normalized_heatmap).all():
            smoothed_heatmap = np.zeros_like(spike_rate_heatmap)
        else:
            smoothed_heatmap = convolve2d(np.nan_to_num(normalized_heatmap), kernel, mode='same')

        heatmap = smoothed_heatmap
    else:
        heatmap = spike_rate_heatmap

    if room_boundary_points is not None:
        # Create grid of (x, y) points
        xv, yv = np.meshgrid(x_grid, y_grid)
        grid_points = np.c_[xv.ravel(), yv.ravel()]
        # Create the polygon of the rooms
        polygon_room = Polygon(room_boundary_points)
        # Check which points fall inside the polygon
        mask_inside = np.array([polygon_room.covers(Point(p)) for p in grid_points])
        mask_inside_2d = mask_inside.reshape(heatmap.shape)
        # Zero out values outside the polygon
        heatmap[~mask_inside_2d] = 0

    return heatmap, x_grid, y_grid


def get_threshold_value(rate_map, threshold_type='percentile', threshold_value=0.9):
    """
    Compute the threshold value based on type and value.
    """
    if np.all(rate_map <= 0):
        return np.inf  # Return unreachable threshold

    data = rate_map[rate_map > 0]
    typ = threshold_type
    val = threshold_value

    if typ == "absolute":
        return val
    elif typ == "percent":
        return np.nanmax(data) * val
    elif typ == "percentile":
        return np.percentile(data, val * 100)
    elif typ == "mean":
        return np.mean(data)
    else:
        raise ValueError(f"Unknown threshold_marker type: {typ}")


import numpy as np
from scipy.signal import convolve2d, medfilt2d


def create_polar_rate_map(distance_matrix, firing_rates, dt=None, smoothed=False, sigma=[2, 1], kernel_size=[9, 5],
                            distance_range=None, bin_size=0.024, angles=None, fill_value=np.nan, smoothing_method="gaussian", return_count=False, wrap_angles=False,
                            weighting_method = "exp",
                            decay_length = 0.1,
                            inv_power = 1.5,
                            softmax_tau = 0.1,
                            threshold_dist=0.25):

    """
    Create a polar tuning map of firing rate vs. wall distance and direction.

    By default the map is a "special bin rate": mean value per (distance, angle) bin. When dt is
    provided (for future use), the second argument is interpreted as spike_count_bin and the
    output is rate in Hz = sum(weighted counts) / sum(weighted occupancy) per bin.

    Parameters:
        distance_matrix (np.ndarray): shape (T, n_angles), distance to wall in each direction at each time.
        firing_rates (np.ndarray): shape (T,). When dt is None: value per timestamp (e.g. spike_count_bin → special bin rate). When dt is not None: spike count per bin → output in Hz.
        dt (float, optional): Time bin duration (s). If provided, output is rate in Hz; if None, output is special bin rate (mean per spatial bin). Kept for future use.
        smoothed (bool): Whether to apply spatial smoothing.
        sigma (list of float): Std. dev. for Gaussian smoothing [distance, angle].
        kernel_size (list of int): Kernel size [distance, angle]. Must be odd.
        distance_range (tuple): (min_distance, max_distance). If None, inferred from data.
        bin_size (float): Bin width for distance binning.
        angles (list or np.ndarray): Angle values in degrees. If None, assumed to be 36 angles from 0 to 350.
        fill_value (float): Value for unvisited bins.
        smoothing_method (str): 'gaussian' or 'median' smoothing method.
        return_count (bool): Whether to return count map.
        wrap_angles (bool): Whether to wrap the first angle column at the end for cyclic continuity.
        weighting_method (str): 'exp', 'inv', 'inv_pow', 'softmax', or 'uniform'.
        decay_length (float): Characteristic length for exponential decay.
        inv_power (float): Power for inverse distance method.
        softmax_tau (float): Temperature for softmax.
        threshold_dist (float): Not used (legacy compatibility).

    Returns:
        rate_map (np.ndarray): 2D map (distance_bin, angle_bin). Special bin rate when dt is None; Hz when dt is provided.
        distance_bins (np.ndarray): Bin center positions for distance.
        angles (np.ndarray): Wrapped or original angles (deg).
        (Optional) count_map (np.ndarray): Number of contributing samples per bin.
    """


    if angles is None:
        if distance_matrix.ndim == 2 and distance_matrix.shape[1] == 36:
            angles = np.arange(0, 360, 10)
        else:
            raise ValueError("Distance matrix must have 36 angles or angles must be provided.")

    if distance_range is None:
        distance_range = (0,np.nanmax(distance_matrix))
    min_distance,max_distance = distance_range

    n_angles = len(angles)
    # n_bins = int((max_distance - min_distance) / bin_size)
    distance_bin_edges = np.arange(min_distance, max_distance + bin_size / 2, bin_size)
    distance_bins = distance_bin_edges[:-1] + bin_size / 2
    n_bins = len(distance_bins)

    rate_map = np.zeros((n_bins, n_angles), dtype=np.float64)
    count_map = np.zeros((n_bins, n_angles), dtype=np.float64)
    if dt is not None:
        occupancy_map = np.zeros((n_bins, n_angles), dtype=np.float64)  # for future use: rate in Hz

    valid_mask = ~np.isnan(distance_matrix) & ~np.isnan(firing_rates[:, None])

    if weighting_method == "exp":
        weights_matrix = np.exp(-distance_matrix / decay_length)
    elif weighting_method == "inv":
        weights_matrix = 1 / (distance_matrix + 1e-5)
    elif weighting_method == "inv_pow":
        weights_matrix = (1 / np.maximum(distance_matrix + 1e-5, 1e-8)) ** inv_power
    elif weighting_method == "softmax":
        weights_matrix = np.exp(-distance_matrix / softmax_tau)
    elif weighting_method == "uniform":
        weights_matrix = np.ones_like(distance_matrix)
    elif weighting_method == "uniform_short":
        weights_matrix = np.where(distance_matrix < threshold_dist, 1.0, 0.0)
    else:
        raise ValueError(f"Unsupported weighting_method: {weighting_method}")

    weights_matrix[~valid_mask] = 0
    # Normalize weights to sum to 1
    weights_matrix_sum = weights_matrix.sum(axis=1, keepdims=True)
    weights_matrix = np.divide(weights_matrix, weights_matrix_sum, where=weights_matrix_sum != 0)

    # # Normalize each row (timestamp) to max = 1
    # max_per_row = np.max(weights_matrix, axis=1, keepdims=True)
    # weights_matrix = np.divide(weights_matrix, max_per_row, where=max_per_row != 0)

    for angle_index in range(n_angles):
        distances = distance_matrix[:, angle_index]
        weights = weights_matrix[:, angle_index]
        valid = ~np.isnan(distances) & ~np.isnan(firing_rates)

        # bin_indices = np.floor((distances - min_distance) / bin_size)
        # bin_indices = np.where(np.isfinite(bin_indices), bin_indices, -1).astype(int)
        bin_indices = np.digitize(distances, distance_bin_edges, right=False) - 1  # [-1..n_bins-1]
        in_range = (bin_indices >= 0) & (bin_indices < n_bins)
        valid = valid & in_range

        binned = bin_indices[valid]
        fr_valid = firing_rates[valid]
        w_valid = weights[valid]

        for b_idx, w, f in zip(binned, w_valid, fr_valid):
            if dt is not None:
                rate_map[b_idx, angle_index] += w * f
                occupancy_map[b_idx, angle_index] += w * dt
            elif weighting_method in ['uniform', 'uniform_short']:
                rate_map[b_idx, angle_index] += f
            else:
                rate_map[b_idx, angle_index] += w * f
            count_map[b_idx, angle_index] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        if dt is not None:
            mean_map = np.divide(rate_map, occupancy_map)
            mean_map[occupancy_map == 0] = fill_value
        else:
            mean_map = np.divide(rate_map, count_map)
            mean_map[count_map == 0] = fill_value

    # Skip smoothing if angle axis is empty (np.pad with mode='wrap' fails on size-0 axis)
    if smoothed and all(s > 0 for s in sigma) and all(k > 0 for k in kernel_size) and mean_map.shape[1] > 0:
        k_d, k_a = kernel_size
        s_d, s_a = sigma

        assert k_d % 2 == 1 and k_a % 2 == 1, "Kernel sizes should be odd for correct smoothing alignment."

        if smoothing_method == "gaussian":
            x = np.arange(-k_a // 2 + 1, k_a // 2 + 1)
            y = np.arange(-k_d // 2 + 1, k_d // 2 + 1)
            X, Y = np.meshgrid(x, y)
            kernel = np.exp(-(X**2 / (2 * s_a**2) + Y**2 / (2 * s_d**2)))
            kernel /= np.sum(kernel)

            pad_top_bottom = ((k_d, k_d), (0, 0))
            pad_left_right = ((0, 0), (k_a, k_a))

            padded_temp = np.pad(mean_map, pad_top_bottom, mode='edge')
            padded_wrap = np.pad(padded_temp, pad_left_right, mode='wrap')

            smoothed_map = convolve2d(padded_wrap, kernel, mode='same')
            smoothed_map = smoothed_map[k_d:-k_d, k_a:-k_a]

        elif smoothing_method == "median":
            smoothed_map = medfilt2d(mean_map, kernel_size=(k_d, k_a))

        else:
            raise ValueError("Unsupported smoothing method. Use 'gaussian' or 'median'.")


        rate_map = smoothed_map
    else:
        rate_map = mean_map

    if wrap_angles:
        rate_map = np.hstack([rate_map, rate_map[:, [0]]])  # duplicate first column at end
        angles = np.append(angles, angles[0] + 360 if angles[0] == 0 else angles[0])
        count_map = np.hstack([count_map, count_map[:, [0]]])

    if return_count:
        return rate_map, distance_bins, angles, count_map
    return rate_map, distance_bins, angles

def _wrap_polar_rate_map(rate_map, angle_bins_deg, tol_deg=2.0):
    """
    Ensure the angle axis closes the circle consistently, for 1D or 2D inputs.

    Parameters
    ----------
    rate_map : array-like
        1D: shape (T,) or 2D: shape (R, T) where T is the angle dimension (columns).
        For 2D, columns correspond to angle bins.
    angle_bins_deg : array-like, shape (T,)
        Angle bin centers in degrees.
    tol_deg : float
        Tolerance for deciding if a final step will close to ~360°.

    Returns
    -------
    rm_out : ndarray
        Same dimensionality as input (1D in → 1D out; 2D in → 2D out).
        If wrapping was added, an extra angle column (and value) is appended.
    ang_out : ndarray, shape (T or T+1,)
        Angles, possibly with a final (first+360) appended.
    """
    rm  = np.asarray(rate_map)
    ang = np.asarray(angle_bins_deg, float)

    if ang.ndim != 1 or rm.ndim not in (1, 2) or ang.size < 2:
        return rm, ang

    was_1d = (rm.ndim == 1)
    if was_1d:
        rm2 = rm[np.newaxis, :]          # (1, T)
    else:
        rm2 = rm                          # (R, T)

    steps = np.diff(ang)
    step  = float(np.median(steps)) if steps.size else 360.0
    span  = float(ang[-1] - ang[0])

    # Already closed
    if np.isclose(span, 360.0, atol=tol_deg):
        out_rm, out_ang = rm2, ang

    # One more typical step would close ~360°
    elif np.isclose(span + step, 360.0, atol=tol_deg):
        out_rm  = np.c_[rm2, rm2[:, [0]]]          # append first column
        out_ang = np.r_[ang, ang[0] + 360.0]       # append final angle

    else:
        out_rm, out_ang = rm2, ang

    if was_1d:
        out_rm = out_rm[0]  # back to 1D

    return out_rm, out_ang


def create_hd_rate_map(
    hd_angles_deg,
    firing_rates,
    angles=None,
    smoothed=False,
    sigma_ang=1.0,
    kernel_size_ang=5,
    smoothing_method="gaussian",   # 'gaussian' | 'median'
    fill_value=np.nan,
    wrap_angles=True,
    return_count=False,
):
    """
    Create a head-direction (HD) tuning curve: mean firing rate per angular bin.

    Parameters:
        hd_angles_deg (np.ndarray): shape (T,), head direction (deg) at each timestamp.
        firing_rates (np.ndarray):  shape (T,), firing rate per timestamp.
        angles (array-like | None): angular bin centers in degrees. If None, use 36 bins (0..350 step 10).
        smoothed (bool):            whether to apply circular smoothing along angle.
        sigma_ang (float):          Gaussian std (in bins) for smoothing (used if smoothing_method='gaussian').
        kernel_size_ang (int):      odd kernel length in bins for smoothing.
        smoothing_method (str):     'gaussian' or 'median'.
        fill_value (float):         value for bins with no visits.
        wrap_angles (bool):         duplicate the first angle bin at the end for cyclic continuity in output.
        return_count (bool):        whether to return the count map (samples per bin).

    Returns:
        rate_map (np.ndarray):  shape (n_angles,) or (n_angles+1,) if wrap_angles=True
        angles_out (np.ndarray): corresponding angle centers (deg), wrapped if requested
        (Optional) count_map (np.ndarray): integer counts per bin (wrapped if requested)
    """
    hd_angles_deg = np.asarray(hd_angles_deg, dtype=float).ravel()
    firing_rates  = np.asarray(firing_rates,  dtype=float).ravel()

    if angles is None:
        angles = np.arange(0, 360, 10.0)  # 36 bins by default
    angles = np.asarray(angles, dtype=float).ravel()
    n_angles = angles.size
    if n_angles < 2:
        raise ValueError("`angles` must contain at least 2 bin centers.")

    # Valid samples
    valid = (~np.isnan(hd_angles_deg)) & (~np.isnan(firing_rates))
    if not np.any(valid):
        rate_map = np.full(n_angles, fill_value, dtype=float)
        counts   = np.zeros(n_angles, dtype=float)
        if wrap_angles:
            rate_map = np.r_[rate_map, rate_map[0]]
            angles_out = np.r_[angles, (angles[0] + 360 if angles[0] == 0 else angles[0])]
            counts = np.r_[counts, counts[0]]
        else:
            angles_out = angles
        return (rate_map, angles_out, counts) if return_count else (rate_map, angles_out)

    hd = hd_angles_deg[valid] % 360.0
    fr = firing_rates[valid]

    # Decide mapping method: fast path for (almost) uniform bins, general fallback otherwise
    diffs = np.diff(np.r_[angles, angles[0] + 360.0])
    uniform = np.allclose(diffs, diffs[0], rtol=1e-6, atol=1e-6)

    bin_idx = None
    if uniform:
        step = diffs[0]
        # map to nearest bin center relative to angles[0]
        # offset = hd - angles[0], then round to nearest step
        rel = (hd - angles[0]) % 360.0
        bin_idx = np.floor((rel / step) + 0.5).astype(int) % n_angles
    else:
        # General nearest-center mapping (vectorized, but O(T * n_angles))
        # Compute minimal circular difference to each center and take argmin
        # diff in [-180, 180] via (x+180)%360 - 180 trick
        diffs_mat = (hd[:, None] - angles[None, :] + 180.0) % 360.0 - 180.0
        bin_idx = np.argmin(np.abs(diffs_mat), axis=1)

    # Accumulate sum and counts per bin
    sums   = np.zeros(n_angles, dtype=float)
    counts = np.zeros(n_angles, dtype=float)

    # Using np.add.at for scatter-add
    np.add.at(sums,   bin_idx, fr)
    np.add.at(counts, bin_idx, 1.0)

    with np.errstate(invalid='ignore', divide='ignore'):
        mean_map = sums / counts
        mean_map[counts == 0] = fill_value

    # Optional circular smoothing
    if smoothed and kernel_size_ang > 1 and kernel_size_ang % 2 == 1:
        k = kernel_size_ang
        pad = k // 2

        # pad circularly for wrap-around smoothing
        padded = np.pad(mean_map, (pad, pad), mode='wrap')

        if smoothing_method.lower() == "gaussian":
            # kernel in bins (index domain), normalized
            x = np.arange(-pad, pad + 1, dtype=float)
            kernel = np.exp(-(x**2) / (2.0 * (float(sigma_ang) ** 2)))
            kernel /= kernel.sum()
            smoothed_map = np.convolve(padded, kernel, mode='same')[pad:-pad]
        elif smoothing_method.lower() == "median":
            # Simple rolling median
            # For performance, a naive implementation; can replace with scipy.signal.medfilt if available.
            window = k
            # Build rolling windows via stride trick fallback
            W = np.lib.stride_tricks.sliding_window_view(padded, window_shape=window)
            # Center windows corresponding to original positions
            smoothed_map = np.median(W, axis=-1)[pad:-pad]
        else:
            raise ValueError("Unsupported smoothing method. Use 'gaussian' or 'median'.")
        rate_map = smoothed_map
    else:
        rate_map = mean_map

    # Optionally duplicate first angle at end (for plotting cyclic continuity)
    if wrap_angles:
        rate_map_out = np.r_[rate_map, rate_map[0]]
        angles_out   = np.r_[angles, (angles[0] + 360.0 if angles[0] == 0 else angles[0])]
        counts_out   = np.r_[counts, counts[0]]
    else:
        rate_map_out = rate_map
        angles_out   = angles
        counts_out   = counts

    return (rate_map_out, angles_out, counts_out) if return_count else (rate_map_out, angles_out)



from ot import emd2
from scipy.spatial.distance import cdist

def calculate_emd_2d(real_rate_map, pred_rate_map):
    """
    Calculate the Earth Mover's Distance (EMD) while preserving spatial information.

    Parameters:
        real_rate_map (np.ndarray): Normalized heatmap for real data.
        pred_rate_map (np.ndarray): Normalized heatmap for predicted data.

    Returns:
        float: Earth Mover's Distance between the two distributions.
    """
    try:
        # Get the shape of the grid
        h, w = real_rate_map.shape

        # Create the 2D grid of coordinates
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        grid_coords = np.stack([x.ravel(), y.ravel()], axis=1)

        # Compute the ground distance matrix (Euclidean distance in 2D space)
        ground_distance = cdist(grid_coords, grid_coords, metric='euclidean')

        # Flatten and normalize rate maps
        real_flat = real_rate_map.flatten()
        pred_flat = pred_rate_map.flatten()

        real_sum = np.nansum(real_flat)
        pred_sum = np.nansum(pred_flat)

        if real_sum == 0 or pred_sum == 0:
            raise ValueError("Sum of rate maps is zero, cannot compute EMD.")

        real_flat = real_flat.astype(np.float64) / real_sum
        pred_flat = pred_flat.astype(np.float64) / pred_sum

        # Replace NaNs with zeros
        real_flat = np.nan_to_num(real_flat)
        pred_flat = np.nan_to_num(pred_flat)

        # Compute 2D EMD using Optimal Transport
        emd_distance = emd2(real_flat, pred_flat, ground_distance)
    except ValueError:
        emd_distance = np.nan
        print("Error calculating EMD. Returning NaN.")
        print(f"error:{ValueError}")

    return emd_distance


from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr


def calculate_kl_divergence_2d(real_rate_map, pred_rate_map):
    """
    Calculate the Kullback-Leibler (KL) Divergence for 2D heatmaps.

    Parameters:
        real_rate_map (np.ndarray): Smoothed and normalized heatmap for real data.
        pred_rate_map (np.ndarray): Smoothed and normalized heatmap for predicted data.

    Returns:
        float: KL Divergence score between the two distributions.
    """
    try:
        # Flatten and normalize rate maps
        real_flat = real_rate_map.flatten()
        pred_flat = pred_rate_map.flatten()

        # Normalize distributions to sum to 1
        real_flat = real_flat.astype(np.float64) / np.nansum(real_flat)
        pred_flat = pred_flat.astype(np.float64) / np.nansum(pred_flat)

        # Replace NaNs and zeros to avoid log(0)
        real_flat = np.nan_to_num(real_flat, nan=1e-10)
        pred_flat = np.nan_to_num(pred_flat, nan=1e-10)
        pred_flat[pred_flat == 0] = 1e-10  # Avoid division by zero

        # Compute KL Divergence
        kl_div = np.sum(rel_entr(real_flat, pred_flat))

    except ValueError:
        kl_div = np.nan
        print("Error calculating KL Divergence. Returning NaN.")

    return kl_div


def calculate_js_divergence_2d(real_rate_map, pred_rate_map):
    """
    Calculate the Jensen-Shannon (JS) Divergence for 2D heatmaps.

    Parameters:
        real_rate_map (np.ndarray): Smoothed and normalized heatmap for real data.
        pred_rate_map (np.ndarray): Smoothed and normalized heatmap for predicted data.

    Returns:
        float: JS Divergence score between the two distributions.
    """
    try:
        # Flatten and normalize rate maps
        real_flat = real_rate_map.flatten()
        pred_flat = pred_rate_map.flatten()

        # Normalize distributions to sum to 1
        real_flat = real_flat.astype(np.float64) / np.nansum(real_flat)
        pred_flat = pred_flat.astype(np.float64) / np.nansum(pred_flat)

        # Replace NaNs and zeros to avoid log(0)
        real_flat = np.nan_to_num(real_flat, nan=1e-10)
        pred_flat = np.nan_to_num(pred_flat, nan=1e-10)

        # Compute JS Divergence
        js_div = jensenshannon(real_flat, pred_flat)

    except ValueError:
        js_div = np.nan
        print("Error calculating JS Divergence. Returning NaN.")

    return js_div


def calculate_correlation_2d(real_rate_map, pred_rate_map):
    """
    Calculate the correlation between two 2D rate maps.

    Parameters:
        real_rate_map (np.ndarray): Smoothed and normalized heatmap for real data.
        pred_rate_map (np.ndarray): Smoothed and normalized heatmap for predicted data.

    Returns:
        float: Correlation coefficient between the two distributions.
    """
    real_flat = real_rate_map.flatten()
    pred_flat = pred_rate_map.flatten()

    # Remove NaNs
    mask = ~np.isnan(real_flat) & ~np.isnan(pred_flat)
    return np.corrcoef(real_flat[mask], pred_flat[mask])[0, 1]

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import norm

import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import norm as sparse_norm
from scipy.spatial.distance import pdist, squareform


def compute_laplacian_smoothness(Xs, Ys, Xt, Yt, use_sparse=True, k=10, epsilon=1e-8):
    """
    Compute normalized Laplacian smoothness difference between source and target coordinates.

    Parameters:
        Xs, Ys: np.ndarray of shape (n,) - Source coordinates
        Xt, Yt: np.ndarray of shape (n,) - Target coordinates
        use_sparse (bool): Whether to use k-NN sparse graph
        k (int): Number of neighbors in sparse graph
        epsilon (float): Small constant for stability

    Returns:
        tuple: (lap_X, lap_Y, lap_total) normalized smoothness differences
    """

    Xs, Ys, Xt, Yt = map(lambda a: np.asarray(a).reshape(-1), (Xs, Ys, Xt, Yt))
    assert Xs.shape == Ys.shape == Xt.shape == Yt.shape, "All inputs must be shape (n,)"
    n = Xs.shape[0]
    if n < 2:
        # Nothing meaningful to compute
        return 0.0, 0.0, 0.0

    coords = np.column_stack([Xs, Ys])

    if use_sparse:
        # k-NN graph using KDTree
        k_eff = int(max(1, min(k, n - 1)))
        tree = KDTree(coords)
        dists, indices = tree.query(coords, k=k_eff + 1)  # k+1 includes self

        # Skip the first column (self distances)
        neighbors = indices[:, 1:k_eff + 1]
        neighbor_dists = dists[:, 1:k_eff + 1]

        # Robust sigma
        finite_pos = neighbor_dists[np.isfinite(neighbor_dists) & (neighbor_dists > 0)]
        if finite_pos.size:
            sigma = float(np.median(finite_pos))
        else:
            # fall back to a scale from pairwise distances or 1.0
            pd = squareform(pdist(coords))
            pd_finite = pd[np.isfinite(pd) & (pd > 0)]
            sigma = float(np.median(pd_finite)) if pd_finite.size else 1.0
        sigma = max(sigma, epsilon)

        # Compute weights for all edges at once
        weights_array = np.exp(-(neighbor_dists ** 2) / (2.0 * sigma ** 2))
        weights_array = np.nan_to_num(weights_array, nan=0.0, posinf=0.0, neginf=0.0)

        # Repeat row indices for each neighbor
        rows = np.repeat(np.arange(n, dtype=np.int64), k_eff)
        cols = neighbors.reshape(-1)
        data = weights_array.reshape(-1)

        # Symmetrize
        rows_sym = np.concatenate([rows, cols])
        cols_sym = np.concatenate([cols, rows])
        data_sym = np.concatenate([data, data])

        # Hard mask to avoid "axis index exceeds dimension"
        inb = (
                (rows_sym >= 0) & (rows_sym < n) &
                (cols_sym >= 0) & (cols_sym < n) &
                np.isfinite(data_sym)
        )
        if not inb.all():
            # optional: print/log the number dropped
            # print(f"Dropped {(~inb).sum()} invalid edges")
            rows_sym, cols_sym, data_sym = rows_sym[inb], cols_sym[inb], data_sym[inb]

        W = csr_matrix((data_sym, (rows_sym, cols_sym)), shape=(n, n))

        # Laplacian (normalized)
        D_vec = np.asarray(W.sum(axis=1)).ravel()
        L = diags(D_vec) - W
        D_inv_sqrt = diags(1.0 / (np.sqrt(D_vec) + epsilon))
        L_norm = D_inv_sqrt @ L @ D_inv_sqrt
        L_fro = float(sparse_norm(L_norm, 'fro'))

    else:
        # ---- Dense (Gaussian kernel on all pairs) ----
        W = squareform(pdist(coords))
        sigma = np.median(W[W > 0]) if np.any(W > 0) else 1.0
        sigma = max(float(sigma), epsilon)
        W = np.exp(-(W ** 2) / (2.0 * sigma ** 2))

        D_vec = W.sum(axis=1)
        L = np.diag(D_vec) - W
        D_inv_sqrt = np.diag(1.0 / (np.sqrt(D_vec) + epsilon))
        L_norm = D_inv_sqrt @ L @ D_inv_sqrt
        L_fro = float(np.linalg.norm(L_norm, 'fro'))

    # Guard against degenerate zero-Frobenius (e.g., isolated points)
    if not np.isfinite(L_fro) or L_fro <= 0:
        L_fro = 1.0

    # Signals as column vectors
    Xt, Yt, Xs, Ys = map(lambda x: x.reshape(-1, 1), (Xt, Yt, Xs, Ys))

    # Smoothness scores (normalized)
    lap_X = float((Xt.T @ L_norm @ Xt - Xs.T @ L_norm @ Xs)[0, 0]) / (L_fro * n)
    lap_Y = float((Yt.T @ L_norm @ Yt - Ys.T @ L_norm @ Ys)[0, 0]) / (L_fro * n)
    lap_total = 0.5 * (lap_X + lap_Y)

    return lap_X, lap_Y, lap_total


def compute_grid_laplacian_smoothness(X_src, Y_src, X_tgt, Y_tgt, neighbors=8, weighted=False, epsilon=1e-8):
    """
    Compute normalized Laplacian smoothness (lap) on a 4- or 8-neighbor grid over source bin indices.

    The graph is built on source coordinates: nodes are rows; two nodes are adjacent if their
    (X_src, Y_src) indices differ by ±1 on one axis (4-neighbor) or also diagonally (8-neighbor).
    lap = (E_tgt - E_src) / (2 n ||L||_F) with E = x' L x + y' L y (Dirichlet energy).
    n is the number of vertices (rows) in the induced graph.

    Parameters
    ----------
    X_src, Y_src : array-like, shape (n,)
        Source bin coordinates (integer indices, e.g. 1..34).
    X_tgt, Y_tgt : array-like, shape (n,)
        Target bin coordinates for each source bin.
    neighbors : int, 4 or 8
        Number of neighbors: 4 = axis-aligned only, 8 = axis + diagonal.
    weighted : bool
        If True and neighbors==8, diagonal edges get weight 1/sqrt(2); axis edges 1.
        If False, all edges weight 1.
    epsilon : float
        Small constant for numerical stability (degree matrix).

    Returns
    -------
    dict
        Keys: lap, lap_x, lap_y, E_src, E_tgt, dE, dE_over_Esrc, lap_exp, lap_ratio, lap_inv.
        dE = E_tgt - E_src (unnormalized Dirichlet energies); dE_over_Esrc = dE / (E_src + eps).
        lap_exp = exp(-lap), lap_ratio = E_src / max(E_tgt, eps), lap_inv = 1/(1+lap).
    """
    _empty = {
        "lap": 0.0, "lap_x": 0.0, "lap_y": 0.0,
        "E_src": 0.0, "E_tgt": 0.0,
        "dE": 0.0, "dE_over_Esrc": 0.0,
        "lap_exp": 1.0, "lap_ratio": 1.0, "lap_inv": 1.0,
    }
    X_src = np.asarray(X_src, dtype=float).reshape(-1)
    Y_src = np.asarray(Y_src, dtype=float).reshape(-1)
    X_tgt = np.asarray(X_tgt, dtype=float).reshape(-1)
    Y_tgt = np.asarray(Y_tgt, dtype=float).reshape(-1)
    n = X_src.shape[0]
    if n < 2:
        return _empty.copy()

    # Map (x, y) -> single row index (first occurrence)
    xy_to_idx = {}
    for i in range(n):
        key = (int(round(X_src[i])), int(round(Y_src[i])))
        if key not in xy_to_idx:
            xy_to_idx[key] = i

    # Neighbor deltas: 4-neighbor axis, 8-neighbor axis + diagonal
    if neighbors == 4:
        deltas = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        weights = [1.0] * 4
    else:
        deltas = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        if weighted:
            w_diag = 1.0 / np.sqrt(2.0)
            weights = [1.0, 1.0, 1.0, 1.0, w_diag, w_diag, w_diag, w_diag]
        else:
            weights = [1.0] * 8

    # Build sparse weight matrix W
    row_inds = []
    col_inds = []
    data = []
    for i in range(n):
        xi, yi = int(round(X_src[i])), int(round(Y_src[i]))
        for (dx, dy), w in zip(deltas, weights):
            j = xy_to_idx.get((xi + dx, yi + dy))
            if j is not None and j != i:
                row_inds.append(i)
                col_inds.append(j)
                data.append(w)

    if not row_inds:
        return _empty.copy()

    W = csr_matrix((data, (row_inds, col_inds)), shape=(n, n))
    # Symmetrize (in case of duplicates we may have asymmetric)
    W = (W + W.T) / 2.0
    W.data = np.minimum(W.data, 1.0)  # cap at 1 if doubled

    D_vec = np.asarray(W.sum(axis=1)).ravel()
    D_vec = np.maximum(D_vec, epsilon)
    L = diags(D_vec) - W
    D_inv_sqrt = diags(1.0 / (np.sqrt(D_vec) + epsilon))
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt
    L_fro = float(sparse_norm(L_norm, 'fro'))
    if not np.isfinite(L_fro) or L_fro <= 0:
        L_fro = 1.0

    # Quadratic forms: E = x' L x + y' L y (column vectors)
    x_src = X_src.reshape(-1, 1)
    y_src = Y_src.reshape(-1, 1)
    x_tgt = X_tgt.reshape(-1, 1)
    y_tgt = Y_tgt.reshape(-1, 1)
    E_src_x = float((x_src.T @ L_norm @ x_src)[0, 0])
    E_src_y = float((y_src.T @ L_norm @ y_src)[0, 0])
    E_tgt_x = float((x_tgt.T @ L_norm @ x_tgt)[0, 0])
    E_tgt_y = float((y_tgt.T @ L_norm @ y_tgt)[0, 0])
    E_src = E_src_x + E_src_y
    E_tgt = E_tgt_x + E_tgt_y
    lap_x = (E_tgt_x - E_src_x) / (L_fro * n)
    lap_y = (E_tgt_y - E_src_y) / (L_fro * n)
    lap = 0.5 * (lap_x + lap_y)
    # Derived metrics (0–1 style and ratio)
    eps_ratio = 1e-12
    E_tgt_safe = max(E_tgt, eps_ratio)
    lap_exp = np.exp(-lap)
    lap_ratio = E_src / E_tgt_safe
    lap_inv = 1.0 / (1.0 + lap) if np.isfinite(lap) and lap > -1 else np.nan
    dE = E_tgt - E_src
    dE_over_Esrc = float(dE / (E_src + eps_ratio)) if np.isfinite(E_src) else np.nan
    return {
        "lap": lap, "lap_x": lap_x, "lap_y": lap_y,
        "E_src": E_src, "E_tgt": E_tgt,
        "dE": float(dE),
        "dE_over_Esrc": float(dE_over_Esrc) if np.isfinite(dE_over_Esrc) else np.nan,
        "lap_exp": lap_exp, "lap_ratio": lap_ratio, "lap_inv": lap_inv,
    }


def compute_residual_laplacian_energy(canonical_source_points, residuals, k=100):
    """
    Compute Laplacian energy of residual field after affine removal.
    
    This measures how much structured non-affine distortion remains after removing
    the global affine transform. Low energy indicates little structured non-affine
    distortion; high energy indicates systematic nonlinear/topological distortion.
    
    Parameters:
        canonical_source_points (np.ndarray): Array of shape (n, 2) with canonical (centered) source points
        residuals (np.ndarray): Array of shape (n, 2) with residual vectors (r_x, r_y)
        k (int): Number of neighbors for kNN graph (default: 100)
    
    Returns:
        float: Residual Laplacian energy E_res = tr(R^T L R) = r_x^T L r_x + r_y^T L r_y
    """
    from scipy.spatial import KDTree
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import norm as sparse_norm
    
    canonical_source_points = np.asarray(canonical_source_points)
    residuals = np.asarray(residuals)
    
    if canonical_source_points.shape[0] != residuals.shape[0]:
        raise ValueError("canonical_source_points and residuals must have same number of rows")
    if canonical_source_points.shape[1] != 2 or residuals.shape[1] != 2:
        raise ValueError("Both inputs must have shape (n, 2)")
    
    n = canonical_source_points.shape[0]
    if n < 2:
        return 0.0
    
    # Build kNN graph on canonical source points
    k_eff = int(max(1, min(k, n - 1)))
    tree = KDTree(canonical_source_points)
    dists, indices = tree.query(canonical_source_points, k=k_eff + 1)  # k+1 includes self
    
    # Skip the first column (self distances)
    neighbors = indices[:, 1:k_eff + 1]
    neighbor_dists = dists[:, 1:k_eff + 1]
    
    # Robust sigma: median distance to kth neighbor
    finite_pos = neighbor_dists[np.isfinite(neighbor_dists) & (neighbor_dists > 0)]
    if finite_pos.size:
        sigma = float(np.median(finite_pos))
    else:
        # Fallback: use median of all pairwise distances
        from scipy.spatial.distance import pdist, squareform
        pd = squareform(pdist(canonical_source_points))
        pd_finite = pd[np.isfinite(pd) & (pd > 0)]
        sigma = float(np.median(pd_finite)) if pd_finite.size else 1.0
    sigma = max(sigma, 1e-8)
    
    # Compute weights: w_ij = exp(-d_ij^2 / (2 sigma^2))
    weights_array = np.exp(-(neighbor_dists ** 2) / (2.0 * sigma ** 2))
    weights_array = np.nan_to_num(weights_array, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Build sparse weight matrix
    rows = np.repeat(np.arange(n, dtype=np.int64), k_eff)
    cols = neighbors.reshape(-1)
    data = weights_array.reshape(-1)
    
    # Symmetrize
    rows_sym = np.concatenate([rows, cols])
    cols_sym = np.concatenate([cols, rows])
    data_sym = np.concatenate([data, data])
    
    # Filter invalid entries
    inb = (
        (rows_sym >= 0) & (rows_sym < n) &
        (cols_sym >= 0) & (cols_sym < n) &
        np.isfinite(data_sym)
    )
    if not inb.all():
        rows_sym, cols_sym, data_sym = rows_sym[inb], cols_sym[inb], data_sym[inb]
    
    W = csr_matrix((data_sym, (rows_sym, cols_sym)), shape=(n, n))
    
    # Build unnormalized Laplacian: L = D - W
    D_vec = np.asarray(W.sum(axis=1)).ravel()
    L = diags(D_vec) - W
    
    # Compute Laplacian energy: E_res = tr(R^T L R) = r_x^T L r_x + r_y^T L r_y
    # Where R is n x 2 matrix with columns r_x and r_y
    r_x = residuals[:, 0]
    r_y = residuals[:, 1]
    
    E_res = float(r_x.T @ L @ r_x + r_y.T @ L @ r_y)
    
    return E_res


def calculate_laplacian_smoothness_2d(real_rate_map, pred_rate_map):
    """
    Compute the normalized Laplacian smoothness difference between two 2D rate maps.

    Parameters:
        real_rate_map (np.ndarray): 2D array of shape (H, W)
        pred_rate_map (np.ndarray): 2D array of shape (H, W)

    Returns:
        float: Laplacian smoothness difference (pred - real)
    """
    assert real_rate_map.shape == pred_rate_map.shape, "Rate maps must have the same shape"

    H, W = real_rate_map.shape
    num_points = H * W

    # Create coordinate grid
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    coords = np.stack([x.ravel(), y.ravel()], axis=1)  # shape: (num_points, 2)

    # Compute pairwise distances
    W = squareform(pdist(coords))
    sigma = np.median(W)
    W = np.exp(-W ** 2 / (2 * sigma ** 2))

    # Degree and Laplacian
    D = np.diag(W.sum(axis=1))
    L = D - W
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt

    # Flatten rate maps into signal vectors
    real_vec = real_rate_map.flatten()
    pred_vec = pred_rate_map.flatten()

    # Compute smoothness scores
    smooth_real = real_vec.T @ L_norm @ real_vec
    smooth_pred = pred_vec.T @ L_norm @ pred_vec

    # Normalize and return difference
    fro_norm = norm(L_norm, 'fro')
    smoothness_score = (smooth_pred - smooth_real) / (fro_norm * num_points)

    return smoothness_score


def calculate_rate_map_stats(real_rate_map, pred_rate_map):
    """
    Calculate the EMD, KL Divergence, JS Divergence, correlation, and Laplacian smoothness between two 2D rate maps.

    Parameters:
        real_rate_map (np.ndarray): Smoothed and normalized heatmap for real data.
        pred_rate_map (np.ndarray): Smoothed and normalized heatmap for predicted data.

    Returns:
        dict: Dictionary containing EMD, KL Divergence, JS Divergence, correlation, and Laplacian smoothness.
    """
    emd = calculate_emd_2d(real_rate_map, pred_rate_map)
    kl_div = calculate_kl_divergence_2d(real_rate_map, pred_rate_map)
    js_div = calculate_js_divergence_2d(real_rate_map, pred_rate_map)
    correlation = calculate_correlation_2d(real_rate_map, pred_rate_map)
    lap_smoothness = calculate_laplacian_smoothness_2d(real_rate_map, pred_rate_map)

    return {
        'EMD': emd,
        'KL': kl_div,
        'JS': js_div,
        'Correlation': correlation,
        'Laplacian_Smoothness': lap_smoothness
    }

# def calculate_mapping_and_colors(XY2XY, pos_range, initial_xy=None, n_pixel=100, gradient_type="horizontal",
#                                  sample_fraction=1, distance_threshold=0.0005):
#     """
#     Calculates the necessary components for plotting the mapping of positions and gradient colors between two rooms.
#
#     Args:
#         XY2XY (function): Model to map XY positions from room A to room B.
#         pos_range (list/tuple): Positional range of the arena as [min, max].
#         initial_xy (np.ndarray, optional): Initial XY positions in room A for filtering. Defaults to None.
#         n_pixel (int, optional): Number of pixels for mesh grid resolution. Defaults to 100.
#         gradient_type (str, optional): Type of gradient ("horizontal", "vertical", "radial"). Defaults to "horizontal".
#         sample_fraction (float, optional): Fraction of points to sample from the grid. Defaults to 1.
#         distance_threshold (float, optional): Distance threshold for filtering points. Defaults to 0.0005.
#
#     Returns:
#         initial_colors (np.ndarray): Original gradient colors in room A.
#         new_colors (np.ndarray): Mapped colors in room B.
#         xv_new (np.ndarray): X grid for room B.
#         yv_new (np.ndarray): Y grid for room B.
#     """
#
#     def create_mesh_grid(pos_range, n_pixel=34):
#         """Generate mesh grid for the arena using the specified positional range and grid resolution."""
#         x_grid = np.linspace(pos_range[0], pos_range[1], n_pixel)
#         y_grid = np.linspace(pos_range[0], pos_range[1], n_pixel)
#         return np.meshgrid(x_grid, y_grid)
#
#     def apply_gradient_pattern(xv, yv, gradient_type="horizontal"):
#         """Apply gradient to mesh grid. Supports 'horizontal', 'vertical', and 'radial' gradients."""
#         if gradient_type == "horizontal":
#             gradient = xv
#         elif gradient_type == "vertical":
#             gradient = yv
#         elif gradient_type == "radial":
#             gradient = np.sqrt(xv ** 2 + yv ** 2)
#         else:
#             raise ValueError("Unknown gradient type.")
#         return (gradient - gradient.min()) / (gradient.max() - gradient.min())
#
#     def map_positions(xy_positions, XY2XY):
#         """Map XY positions from room A to room B using the given transformation model."""
#         xy_tensor = torch.tensor(xy_positions, dtype=torch.float32)
#         mapped_xy = XY2XY(xy_tensor).detach().numpy()  # Apply mapping model
#         return mapped_xy
#
#     def filter_mapped_positions(mapped_xy, pos_range):
#         """Filter mapped XY positions to retain those within the defined positional range."""
#         x_min, x_max = pos_range[0], pos_range[1]
#         return (mapped_xy[:, 0] >= x_min) & (mapped_xy[:, 0] <= x_max) & (mapped_xy[:, 1] >= x_min) & (
#                     mapped_xy[:, 1] <= x_max)
#
#     def sample_subset_points(xv, yv, sample_fraction):
#         """Randomly sample a subset of mesh grid points."""
#         total_points = xv.size
#         sampled_indices = np.random.choice(total_points, size=int(total_points * sample_fraction), replace=False)
#         return np.stack([xv.flatten()[sampled_indices], yv.flatten()[sampled_indices]], axis=1), sampled_indices
#
#     def filter_close_points(sampled_xy, random_indices, initial_xy, threshold):
#         """Filter points in sampled_xy that are close to any point in initial_xy based on a distance threshold."""
#         distances = np.linalg.norm(sampled_xy[:, np.newaxis] - initial_xy, axis=2)
#         mask = np.any(distances < threshold, axis=1)
#         return sampled_xy[mask], random_indices[mask]
#
#     def map_pixels_to_new_grid(sampled_xy, mapped_xy, initial_colors, sampled_indices, pos_range, n_pixel=34):
#         """Map pixel colors from room A to room B based on transformed positions."""
#         valid_mask = filter_mapped_positions(mapped_xy, pos_range)
#         valid_mapped_xy, valid_colors = mapped_xy[valid_mask], initial_colors.flatten()[sampled_indices][valid_mask]
#
#         # Initialize new_colors with NaN to represent empty spots
#         new_colors = np.full((n_pixel, n_pixel), np.nan)
#         # Create a counter array to track the number of points mapped to each pixel
#         pixel_count = np.zeros((n_pixel, n_pixel))
#
#         x_grid, y_grid = np.linspace(pos_range[0], pos_range[1], n_pixel), np.linspace(pos_range[0], pos_range[1],n_pixel)
#
#         for i, pos in enumerate(valid_mapped_xy):
#             x_idx, y_idx = np.argmin(np.abs(x_grid - pos[0])), np.argmin(np.abs(y_grid - pos[1]))
#
#             # If pixel already has color, average the new color with the current one
#             if not np.isnan(new_colors[y_idx, x_idx]):
#                 new_colors[y_idx, x_idx] = (new_colors[y_idx, x_idx] * pixel_count[y_idx, x_idx] + valid_colors[i]) / (
#                             pixel_count[y_idx, x_idx] + 1)
#             else:
#                 new_colors[y_idx, x_idx] = valid_colors[i]
#
#             pixel_count[y_idx, x_idx] += 1
#
#         return new_colors, *np.meshgrid(x_grid, y_grid), pixel_count
#
#
#
#     # Step 1: Create mesh grids for room A
#     xv, yv = create_mesh_grid(pos_range, n_pixel)
#
#     # Step 2: Apply gradient pattern to room A
#     initial_colors = apply_gradient_pattern(xv, yv, gradient_type=gradient_type)
#
#     # Step 3: Sample a subset of points from the grid
#     sampled_xy, sampled_indices = sample_subset_points(xv, yv, sample_fraction)
#
#     # Step 4: Filter sampled points based on initial positions (if provided)
#     if initial_xy is not None:
#         sampled_xy, sampled_indices = filter_close_points(sampled_xy, sampled_indices, initial_xy, distance_threshold)
#
#     # Step 5: Move sampled colors into a separate array without flattening the entire grid
#     initial_colors_flat = initial_colors.flatten()
#     temp_colors = initial_colors_flat[sampled_indices].copy()
#     initial_colors_flat [:] = np.nan # Clear all colors
#     initial_colors_flat[sampled_indices] = temp_colors # Set only sampled colors back
#     initial_colors=initial_colors_flat.reshape(initial_colors.shape)
#
#     # Step 6: Map positions from room A to room B using the XY2XY model
#     mapped_xy = map_positions(sampled_xy, XY2XY)
#
#     # Step 7: Map the pixel colors from room A to their new locations in room B
#     new_colors, xv_new, yv_new, pixel_count = map_pixels_to_new_grid(sampled_xy, mapped_xy, initial_colors, sampled_indices,
#                                                         pos_range, n_pixel)
#
#
#
#     return initial_colors, new_colors, xv_new, yv_new, pixel_count
#

def calculate_mapping_and_colors(XY2XY, pos_range, initial_xy=None, n_pixel=100, gradient_type="horizontal",
                                 sample_fraction=1, distance_threshold=0.0005, use_initial_only=False, homogeneous_position=True):
    """
    Calculates the necessary components for plotting the mapping of positions and gradient colors between two rooms.

    Args:
        XY2XY (function): Model to map XY positions from room A to room B.
        pos_range (list/tuple): Positional range of the arena as [min, max].
        initial_xy (np.ndarray, optional): Initial XY positions in room A for filtering. Defaults to None.
        n_pixel (int, optional): Number of pixels for mesh grid resolution. Defaults to 100.
        gradient_type (str, optional): Type of gradient ("horizontal", "vertical", "radial"). Defaults to "horizontal".
        sample_fraction (float, optional): Fraction of points to sample from the grid. Defaults to 1.
        distance_threshold (float, optional): Distance threshold for filtering points. Defaults to 0.0005.
        use_initial_only (bool, optional): If True, use `initial_xy` directly. If False, sample points. Defaults to False.

    Returns:
        initial_colors (np.ndarray): Original gradient colors in room A (for sampled points).
        mapped_xy (np.ndarray): Mapped XY positions in room B.
        sampled_xy (np.ndarray): Sampled or initial XY positions in room A.
    """

    def create_mesh_grid(pos_range, n_pixel=34):
        """Generate mesh grid for the arena using the specified positional range and grid resolution."""
        x_grid = np.linspace(pos_range[0], pos_range[1], n_pixel)
        y_grid = np.linspace(pos_range[0], pos_range[1], n_pixel)
        return np.meshgrid(x_grid, y_grid)

    def apply_gradient_pattern(xv, yv, gradient_type="horizontal"):
        """Apply gradient to mesh grid. Supports 'horizontal', 'vertical', and 'radial' gradients."""
        if gradient_type == "horizontal":
            gradient = xv
        elif gradient_type == "vertical":
            gradient = yv
        elif gradient_type == "radial":
            x_midpoint = (np.max(xv) - np.min(xv))/2
            y_midpoint = (np.max(yv) - np.min(yv))/2
            gradient = np.sqrt((xv-x_midpoint) ** 2 + (yv-y_midpoint) ** 2)
        else:
            raise ValueError("Unknown gradient type.")
        return (gradient - gradient.min()) / (gradient.max() - gradient.min())

    def map_positions(xy_positions, XY2XY):
        """Map XY positions from room A to room B using the given transformation model."""
        xy_tensor = torch.tensor(xy_positions, dtype=torch.float32)
        mapped_xy = XY2XY(xy_tensor).detach().numpy()  # Apply mapping model
        return mapped_xy

    def filter_close_points(sampled_xy, initial_xy, threshold):
        """Filter points in sampled_xy that are close to any point in initial_xy based on a distance threshold."""
        distances = np.linalg.norm(sampled_xy[:, np.newaxis] - initial_xy, axis=2)
        mask = np.any(distances < threshold, axis=1)
        return sampled_xy[mask]

    # Step 1: Use initial_xy directly if use_initial_only is True, otherwise sample points from a grid.
    if use_initial_only:
        sampled_xy = initial_xy
    else:
        xv, yv = create_mesh_grid(pos_range, n_pixel)
        sampled_xy = np.stack([xv.flatten(), yv.flatten()], axis=1)

        # Step 2: Filter points based on initial_xy if provided
        if initial_xy is not None:
            sampled_xy = filter_close_points(sampled_xy, initial_xy, distance_threshold)

        # Step 3: Randomly sample points
        total_points = sampled_xy.shape[0]
        sampled_indices = np.random.choice(total_points, size=int(total_points * sample_fraction), replace=False)
        sampled_xy = sampled_xy[sampled_indices]

    # Step 4: Apply gradient pattern based on sampled points (initial colors)
    xv_sampled, yv_sampled = sampled_xy[:, 0], sampled_xy[:, 1]
    colors = apply_gradient_pattern(xv_sampled, yv_sampled, gradient_type)

    # Step 5: Map positions from room A to room B using the XY2XY model
    epsilon = 1e-9
    if homogeneous_position:
        # Add an extra coordinate of 1 to sampled_xy (still in NumPy)
        sampled_xy_with_ones = np.hstack((sampled_xy, np.ones((sampled_xy.shape[0], 1))))
        # Map positions (using the extended sampled_xy)
        mapped_xy_with_ones = map_positions(sampled_xy_with_ones, XY2XY)
        # Normalize mapped_xy by dividing by the third coordinate
        mapped_xy = mapped_xy_with_ones[:, :-1] / (mapped_xy_with_ones[:, -1, np.newaxis] + epsilon)
    else:
        mapped_xy = map_positions(sampled_xy, XY2XY)

    return sampled_xy, mapped_xy, colors


def calculate_rayleigh_vector(radii, angles):
    """
    Calculate the Rayleigh vector based on angles and radii.

    Args:
        angles (pd.Series): Series with angles in degrees.
        radii (np.ndarray): Array of radii (e.g., spike rates).

    Returns:
        tuple: A tuple containing the radius and angle of the Rayleigh vector.
    """

    n = len(radii)

    # Adjust angles and radii for negative values
    angles = (radii < 0) * 180 + angles
    radii = np.abs(radii) / np.sum(np.abs(radii))

    # Convert angles to radians
    angles_radians = np.deg2rad(angles)

    # Compute the Rayleigh vector components
    x_components = radii * np.cos(angles_radians)
    y_components = radii * np.sin(angles_radians)
    R_x = np.sum(x_components)  # / n
    R_y = np.sum(y_components)  # / n

    # Calculate the length of the Rayleigh vector
    rayleigh_radius = np.sqrt(R_x ** 2 + R_y ** 2)

    # Calculate the angle of the Rayleigh vector
    rayleigh_angle = np.rad2deg(np.arctan2(R_y, R_x))

    return rayleigh_radius, rayleigh_angle


def rayleigh_stats(rate_map, angle_bins, distance_bins):
    """
    Compute circular stats from polar rate map.

    rate_map: (n_r, n_theta) depending on your create_polar_rate_map convention.
    """
    # Ensure R x T
    if rate_map.shape[0] == len(angle_bins) and rate_map.shape[1] == len(distance_bins):
        # user map might be T x R; try to detect/transpose
        rate_map = rate_map.T  # -> R x T

    R, T = rate_map.shape
    thetas_deg = np.asarray(angle_bins)
    thetas_rad = np.deg2rad(thetas_deg)

    if not np.isfinite(rate_map).any():
        # Empty/invalid map
        return {
            "peak_radius_index": None,
            "peak_angle_index": None,
            "peak_value": np.nan,
            "peak_radius": np.nan,
            "peak_angle_deg": np.nan,
            "rayleigh_R_at_peak_radius": np.nan,
            "rayleigh_T_at_peak_radius": np.nan,
            "rayleigh_R_collapsed_over_radius": np.nan,
            "rayleigh_T_collapsed_over_radius": np.nan,
        }

    # Global maximum
    flat_index_of_peak = int(np.nanargmax(rate_map))
    peak_radius_index, peak_angle_index = np.unravel_index(flat_index_of_peak, rate_map.shape)
    peak_value = float(rate_map[peak_radius_index, peak_angle_index])
    peak_angle_deg = int(thetas_deg[peak_angle_index])
    peak_radius_value = float(distance_bins[peak_radius_index])

    # Rayleigh at peak radius
    angle_tuning_at_peak_radius = rate_map[peak_radius_index, :].astype(float)
    rv_peak = compute_rayleigh(angle_tuning_at_peak_radius, thetas_deg)
    rayleigh_R_at_peak_radius = float(rv_peak["rayleigh_R"])
    rayleigh_T_at_peak_radius = float(rv_peak["rayleigh_T"])

    # Collapsed over radius Rayleigh
    angle_tuning_collapsed = rate_map.sum(axis=0)  # shape: (T,)
    peak_collapsed_value = float(angle_tuning_collapsed.max())
    rv_collapsed = compute_rayleigh(angle_tuning_collapsed, thetas_deg)
    rayleigh_R_collapsed_over_radius = float(rv_collapsed["rayleigh_R"])
    rayleigh_T_collapsed_over_radius = float(rv_collapsed["rayleigh_T"])

    # Build result with readable keys
    result = {
        "peak_radius_index": int(peak_radius_index),
        "peak_angle_index": int(peak_angle_index),
        "peak_value": peak_value,
        "peak_radius": peak_radius_value,
        "peak_angle_deg": peak_angle_deg,
        "rayleigh_R_at_peak_radius": rayleigh_R_at_peak_radius,
        "rayleigh_T_at_peak_radius": rayleigh_T_at_peak_radius,
        'peak_collapsed_value': peak_collapsed_value,
        "rayleigh_R_collapsed_over_radius": rayleigh_R_collapsed_over_radius,
        "rayleigh_T_collapsed_over_radius": rayleigh_T_collapsed_over_radius,
    }
    return result


def compute_rayleigh(angle_tuning, angles_deg):
    tuning = np.clip(np.asarray(angle_tuning, dtype=float), 0.0, None)
    mass = float(tuning.sum())
    if mass <= 0 or not np.isfinite(mass):
        return {"rayleigh_R": np.nan, "rayleigh_T": np.nan, "X": 0.0, "Y": 0.0, "mass": 0.0}
    theta_rad = np.deg2rad(np.asarray(angles_deg, dtype=float))
    X = float(np.sum(tuning * np.cos(theta_rad)))
    Y = float(np.sum(tuning * np.sin(theta_rad)))
    R = float(np.hypot(X, Y) / mass)
    angle_deg = float((np.degrees(np.arctan2(Y, X)) + 360.0) % 360.0)
    return {"rayleigh_R": R, "rayleigh_T": angle_deg, "X": X, "Y": Y, "mass": mass}


def calculate_correlations_by_timestamp(df_spike_rate, blocks_timestamps):
    """
    Calculate pairwise Pearson correlations between cells for specific time blocks.

    This function computes correlation matrices between the firing rates of cells
    for each specified time block. The input DataFrame should contain spike rate
    data with associated timestamps and cell identifiers.

    Parameters:
        df_spike_rate (pd.DataFrame): A DataFrame with columns ['timestamp', 'cell', 'spike_rate'],
            where each row contains the spike rate of a single cell at a given timestamp.
        blocks_timestamps (list or array-like): A list where each element is an index or list of indices
            (relative to df_spike_rate sorted by timestamp) representing a temporal block of timestamps
            to compute the correlation over. Each element will be used to slice the pivoted spike rate matrix.

    Returns:
        np.ndarray: A 3D numpy array of shape (n_blocks, n_cells, n_cells), where each [i] entry contains the
            correlation matrix (n_cells × n_cells) of firing rates across all cells for the i-th block of timestamps.

    Notes:
        - Any missing values in the correlation computation are filled with 0.
        - The cells are ordered according to the columns in the pivoted DataFrame.
        - Assumes that the input DataFrame contains at least one value per (timestamp, cell) pair within each block.
    """
    pivot_df = df_spike_rate.pivot(index='timestamp', columns='cell', values='spike_rate')
    correlations = []
    timestamps = pivot_df.index
    cells = pivot_df.columns
    n_cells = len(cells)

    n_blocks = len(blocks_timestamps)

    corr_tensor = np.zeros((n_blocks, n_cells, n_cells))

    for i in tqdm(range(n_blocks), desc="Calculating Temporally Correlations"):
        timestamp = blocks_timestamps[i]
        data_window = pivot_df.iloc[timestamp]
        corr_tensor[i, :, :] = data_window.corr().fillna(0)

    return corr_tensor


def calculate_correlations_with_phase_shifts(pivot_df, blocks_timestamps, max_shift=3):
    cells = pivot_df.columns
    n_cells = len(cells)
    n_blocks = len(blocks_timestamps)

    corr_tensor = np.zeros((n_blocks, n_cells, n_cells))

    for i in tqdm(range(n_blocks), desc="Calculating Temporally Correlations with Phase Shifts"):

        # Extract the data for the current block
        block_indices = blocks_timestamps[i]
        block_data = pivot_df.iloc[block_indices].values
        n_timepoints = block_data.shape[0]

        # Pad the data with zeros
        padded_block_data = np.pad(block_data, ((max_shift, max_shift), (0, 0)), mode='constant')

        correlation_matrices = []

        # Calculate correlation matrices for each phase shift
        for shift in range(-max_shift, max_shift + 1):
            # Select data window for current shift
            if shift < 0:
                shifted_data = padded_block_data[max_shift + shift:max_shift + shift + n_timepoints]
                original_data = padded_block_data[max_shift:max_shift + n_timepoints]
            else:
                shifted_data = padded_block_data[max_shift:max_shift + n_timepoints]
                original_data = padded_block_data[max_shift + shift:max_shift + shift + n_timepoints]

            # Calculate correlation matrix between shifted and original data
            corr_matrix = np.corrcoef(shifted_data.T, original_data.T)[:n_cells, n_cells:]
            corr_matrix[np.isnan(corr_matrix)] = 0
            correlation_matrices.append(corr_matrix)

        # Average the correlation matrices for the entire block
        block_corr_matrix = np.max(np.stack(correlation_matrices), axis=0)

        corr_tensor[i, :, :] = block_corr_matrix

    return corr_tensor


def calculate_position_correlation_matrix(df_dataset, n_pixel=34):
    # Extract cell indices
    cells_columns = [col for col in df_dataset.columns if col.startswith('Cell')]
    # cells = [int(col.split('_')[1]) for col in cells_columns]
    n_cells = len(cells_columns)

    # Define the grid for x and y positions
    pos_range = (df_dataset[['X', 'Y']].values.min(), df_dataset[['X', 'Y']].values.max())
    x_grid = np.linspace(pos_range[0], pos_range[1], n_pixel)
    y_grid = np.linspace(pos_range[0], pos_range[1], n_pixel)

    sigma = 2
    kernel_size = 8
    X_kernel, Y_kernel = np.meshgrid(np.arange(-kernel_size / 2, kernel_size / 2 + 1),
                                     np.arange(-kernel_size / 2, kernel_size / 2 + 1))
    kernel = np.exp(-(X_kernel ** 2 + Y_kernel ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    kernel = kernel / np.sum(kernel)

    # Initialize a matrix to store normalized spike rates for each cell
    normalized_spike_rates = np.zeros((n_cells, n_pixel, n_pixel))

    for i, cells_column in tqdm(enumerate(cells_columns), desc="Processing cells", total=n_cells):
        spike_rate_heatmap = np.zeros((n_pixel, n_pixel))
        normalized_heatmap = np.zeros((n_pixel, n_pixel))

        for j in range(len(x_grid) - 1):
            for k in range(len(y_grid) - 1):
                instance_in_cell = (df_dataset['X'] >= x_grid[j]) & (df_dataset['X'] < x_grid[j + 1]) & \
                                   (df_dataset['Y'] >= y_grid[k]) & (df_dataset['Y'] < y_grid[k + 1])
                sum_instances_in_cell = np.sum(instance_in_cell)

                spike_rates_in_cell = df_dataset.loc[instance_in_cell, cells_column]

                if sum_instances_in_cell > 0:
                    spike_rate_heatmap[k, j] = np.sum(spike_rates_in_cell)
                    normalized_heatmap[k, j] = spike_rate_heatmap[k, j] / sum_instances_in_cell

        smoothed_heatmap = convolve2d(normalized_heatmap, kernel, mode='same')

        normalized_spike_rates[i, :, :] = smoothed_heatmap

    # Reshape the matrix to (n_cells, n_pixel * n_pixel)
    reshaped_spike_rates = normalized_spike_rates.reshape(n_cells, -1)

    # Calculate the correlation matrix
    correlation_matrix = np.corrcoef(reshaped_spike_rates)
    correlation_matrix[np.isnan(correlation_matrix)] = 0

    return correlation_matrix


def calculate_correlations_by_positions(df_dataset, blocks_indices, n_pixel=34):
    cells_columns = [col for col in df_dataset.columns if col.startswith('Cell')]
    n_cells = len(cells_columns)
    n_blocks = len(blocks_indices)

    corr_tensor = np.zeros((n_blocks, n_cells, n_cells))

    for i in range(n_blocks):  # tqdm(range(n_blocks), desc="Calculating Temporally Correlations"):
        print(f'Calculating Temporally Correlations {i + 1}/{n_blocks}')
        block_indices = blocks_indices[i]
        data_window = df_dataset.iloc[block_indices]

        correlation_matrix = calculate_position_correlation_matrix(data_window, n_pixel)
        corr_tensor[i, :, :] = correlation_matrix

    return corr_tensor


def calculate_hd_correlation_matrix(df_dataset, n_angles=36):
    # Extract cell indices
    cells_columns = [col for col in df_dataset.columns if col.startswith('Cell')]
    n_cells = len(cells_columns)

    # Define the angles for head direction
    angles = np.linspace(0, 360, n_angles, endpoint=False)
    diff_angles = angles[1] - angles[0]

    # Initialize a matrix to store normalized spike rates for each cell
    normalized_spike_rates = np.zeros((n_cells, n_angles))

    for i, cell_column in tqdm(enumerate(cells_columns), desc="Processing cells", total=n_cells):
        tuning_curve = np.zeros(len(angles))
        normalized_tuning_curve = np.zeros(len(angles))

        for j, angle in enumerate(angles):
            if (angle - diff_angles / 2 < 0) | (angle + diff_angles / 2 > 360):
                instance_in_angle = (df_dataset['HD'] >= (angle - diff_angles / 2) % 360) | (
                            df_dataset['HD'] < (angle + diff_angles / 2) % 360)
            else:
                instance_in_angle = (df_dataset['HD'] >= (angle - diff_angles / 2) % 360) & (
                            df_dataset['HD'] < (angle + diff_angles / 2) % 360)
            sum_instances_in_angle = np.sum(instance_in_angle)
            spike_rates_in_angle = df_dataset.loc[instance_in_angle, cell_column]

            if sum_instances_in_angle > 0:
                tuning_curve[j] = np.sum(spike_rates_in_angle)
                normalized_tuning_curve[j] = tuning_curve[j] / sum_instances_in_angle

        # Smooth the tuning curve using Gaussian filter
        smoothed_tuning_curve = gaussian_filter1d(normalized_tuning_curve, sigma=1.5, mode='wrap')

        normalized_spike_rates[i, :] = normalized_tuning_curve

    # Calculate the correlation matrix
    correlation_matrix = np.corrcoef(normalized_spike_rates)
    correlation_matrix[np.isnan(correlation_matrix)] = 0

    return correlation_matrix


def calculate_correlations_by_hd(df_dataset, blocks_indices, n_angles=36):
    cells_columns = [col for col in df_dataset.columns if col.startswith('Cell')]
    n_cells = len(cells_columns)
    n_blocks = len(blocks_indices)

    corr_tensor = np.zeros((n_blocks, n_cells, n_cells))

    for i in range(n_blocks):  # tqdm(range(n_blocks), desc="Calculating Temporally Correlations"):
        print(f'Calculating Temporally Correlations {i + 1}/{n_blocks}')
        block_indices = blocks_indices[i]
        data_window = df_dataset.iloc[block_indices]

        correlation_matrix = calculate_hd_correlation_matrix(data_window, n_angles)
        corr_tensor[i, :, :] = correlation_matrix

    return corr_tensor


def cluster_cells(embedding, max_clusters=100, min_cells_per_cluster=5, threshold_significant_eigencector=0.1):
    """
    Cluster cells based on their embeddings.

    Parameters:
    - embedding (np.ndarray): An array of shape (n_samples, n_features) containing the cell embeddings.
    - max_clusters (int): The maximum number of clusters to form.
    - min_cells_per_cluster (int): The minimum number of cells required to form a cluster.

    Returns:
    - labels (np.ndarray): An array of shape (n_samples,) containing the cluster labels for each cell.
      Cells not assigned to any cluster have a label of 0.
    """
    # Initialize the labels for each cell to 0 (unclustered)
    labels = np.zeros(embedding.shape[0])

    # Calculate the norm of each embedding and sort cells by norm in descending order
    norms = np.linalg.norm(embedding, axis=1)
    sorted_indices_by_norm = np.argsort(norms)[::-1].tolist()

    cluster_id = 1
    while cluster_id <= max_clusters and len(sorted_indices_by_norm) > 0:
        # Select the cell with the maximum norm
        max_norm_cell_idx = sorted_indices_by_norm[0]

        # Identify significant eigenvectors based on the embedding of the max norm cell
        squared_embedding = embedding[max_norm_cell_idx, :] ** 2
        significant_eigenvectors = (squared_embedding / np.max(squared_embedding)) > threshold_significant_eigencector
        significant_embedding = embedding[:, significant_eigenvectors]

        # Calculate the normal vector for the significant embedding of the max norm cell
        normal_vector = significant_embedding[max_norm_cell_idx, :] / np.linalg.norm(
            significant_embedding[max_norm_cell_idx, :])

        # Project all significant embeddings onto the normal vector
        projection = significant_embedding @ normal_vector.T

        # Calculate distances from the max norm cell and from the origin
        max_norm_embedding = significant_embedding[max_norm_cell_idx, :].reshape(1, -1)
        distance_to_max_cell = cdist(significant_embedding, max_norm_embedding)
        distance_to_origin = cdist(significant_embedding, np.zeros_like(max_norm_embedding))

        # Identify cells closer to the max norm cell than to the origin
        closer_to_max_cell = (distance_to_max_cell < distance_to_origin).flatten()

        # Ensure cells are not already assigned to a cluster
        unclustered_cells = (labels == 0)
        potential_cluster_members = closer_to_max_cell & unclustered_cells

        # Assign cluster labels if there are enough cells to form a cluster
        if np.sum(potential_cluster_members) >= min_cells_per_cluster:
            # print(f'Number of significant eigenvectors: {np.sum(significant_eigenvectors)}')
            # print(f'Number of potential cluster members: {np.sum(potential_cluster_members)}')
            labels[potential_cluster_members] = cluster_id
            cluster_id += 1

            # Remove the assigned cells from the list of cells to consider
            sorted_indices_by_norm = [idx for idx in sorted_indices_by_norm if
                                      idx not in np.nonzero(potential_cluster_members)[0]]
        else:
            # If the potential cluster is too small, remove the current max norm cell from consideration
            sorted_indices_by_norm.pop(0)
    # print(f'Number of clusters: {cluster_id+1}')
    return labels


def merge_clusters(corr_matrix, df_clusters, threshold=0.5):
    """
    Merge clusters based on correlations between them.

    Parameters:
    - corr_matrix (np.ndarray): A symmetric matrix representing correlations between cells.
    - df_clusters (pd.DataFrame): A DataFrame containing cell indices and their assigned cluster labels.

    Returns:
    - df_clusters (pd.DataFrame): Updated DataFrame with merged clusters.
    """

    def get_cluster_correlations(corr_matrix, df_clusters):
        """Calculate the mean intra-cluster and inter-cluster correlations."""
        cluster_ids = df_clusters['cluster'].unique()
        cluster_means = {}
        inter_cluster_means = []

        for i, cluster_id in enumerate(cluster_ids):
            current_cluster_cells = df_clusters[df_clusters['cluster'] == cluster_id].index
            intra_cluster_corr = corr_matrix[np.ix_(current_cluster_cells, current_cluster_cells)]
            upper_triangle = np.triu_indices(len(current_cluster_cells), k=1)
            intra_corr_mean = np.mean(intra_cluster_corr[upper_triangle])
            cluster_means[cluster_id] = intra_corr_mean

            for j in range(i + 1, len(cluster_ids)):
                next_cluster_id = cluster_ids[j]
                next_cluster_cells = df_clusters[df_clusters['cluster'] == next_cluster_id].index
                inter_cluster_corr = corr_matrix[np.ix_(current_cluster_cells, next_cluster_cells)]
                inter_corr_mean = np.mean(inter_cluster_corr)
                inter_cluster_means.append((cluster_id, next_cluster_id, inter_corr_mean))

        return cluster_means, sorted(inter_cluster_means, key=lambda x: x[2], reverse=True)

    # Initialize the merged flag
    merged = True

    # Iterate until no further merging can be done
    while merged:
        merged = False
        cluster_means, inter_cluster_means = get_cluster_correlations(corr_matrix, df_clusters)

        for cluster_id1, cluster_id2, inter_corr_mean in inter_cluster_means:
            if cluster_id1 not in df_clusters['cluster'].values or cluster_id2 not in df_clusters['cluster'].values:
                continue

            # Calculate intra-cluster means for both clusters
            intra_corr_mean1 = cluster_means[cluster_id1]
            intra_corr_mean2 = cluster_means[cluster_id2]

            # Check if the mean intra-cluster correlation is lower than twice the mean inter-cluster correlation
            if ((inter_corr_mean / max(intra_corr_mean1, intra_corr_mean2) > threshold) & (
                    min(intra_corr_mean1, intra_corr_mean2) / max(intra_corr_mean1, intra_corr_mean2) < threshold)) | (
                    max(intra_corr_mean1, intra_corr_mean2) < 0.1):
                # Merge clusters
                df_clusters.loc[df_clusters['cluster'] == cluster_id2, 'cluster'] = cluster_id1
                merged = True
                break

        # plot_cluster_correlation_heatmap(corr_matrix, df_clusters)

    return df_clusters


import numpy as np


def calculate_occupancy(config, df_data, pos_range=None, n_pixel=None):
    """
    Calculates the occupancy percentage for each bin in the grid and updates the DataFrame with weights based on bin occupancy.

    Args:
        config (dict): Configuration dictionary containing preprocessing parameters.
        df_data (pd.DataFrame): DataFrame with columns 'X' and 'Y' representing positions.
        pos_range (tuple, optional): Range of positions as (min, max). If None, it is derived from df_data.
        n_pixel (int, optional): Number of pixels (bins) for the grid. If None, it is derived from config.

    Returns:
        occupancy_percentage (np.ndarray): Normalized occupancy percentage for each bin.
        grid
    """

    if pos_range is None:
        pos_range = config['preprocessing'].get('pos_range',(np.min(df_data[['X','Y']].values), np.max(df_data[['X','Y']].values)))

    if n_pixel is None:
        n_pixel = config['preprocessing'].get('n_pixel', 34)

    # Create grid for the arena
    x_bins = np.linspace(pos_range[0], pos_range[1], n_pixel + 1)
    y_bins = np.linspace(pos_range[0], pos_range[1], n_pixel + 1)

    # Assign each timestamp to a bin
    df_data['x_bin'] = np.digitize(df_data['X'], x_bins) - 1  # Bins are 1-indexed
    df_data['y_bin'] = np.digitize(df_data['Y'], y_bins) - 1

    # Filter out-of-bounds points
    df_data = df_data[(df_data['x_bin'] >= 0) & (df_data['x_bin'] < n_pixel) &
                      (df_data['y_bin'] >= 0) & (df_data['y_bin'] < n_pixel)]

    # Calculate occupancy for each bin
    occupancy, _, _ = np.histogram2d(df_data['X'], df_data['Y'], bins=[x_bins, y_bins])

    # Normalize occupancy to get weights (percentage)
    total_points = df_data.shape[0]
    occupancy_percentage = occupancy / total_points

    # Assign weights to each timestamp based on the occupancy of the bin it's in
    x_bin_indices = df_data['x_bin'].to_numpy()
    y_bin_indices = df_data['y_bin'].to_numpy()
    df_data = df_data.assign(weight=occupancy_percentage[y_bin_indices, x_bin_indices])

    return occupancy_percentage, (x_bins,y_bins)





def mapping_kde_smoothing(prev_xy, new_xy, pos_range=None, n_pixel=500, sigma=0.01, min_weight=1e-3, min_points_threshold=5,
                            query_points=None, method='fast', debug_plot=False, boundary_points=None):
    """
    KDE-based smoothing of matched coordinates with optional fast mode using KDTree.

    Parameters:
        prev_xy (np.ndarray): Source coordinates (N, 2)
        new_xy (np.ndarray): Target coordinates (N, 2)
        pos_range (tuple, optional): (min, max) range for both axes. Used if query_points is None.
        n_pixel (int): Number of grid points per axis if query_points is not provided.
        sigma (float): Bandwidth of the Gaussian kernel.
        min_weight (float): Minimum total kernel weight to include a smoothed point.
        min_points_threshold (int): Minimum number of neighboring points required to compute smoothed estimate.
        query_points (np.ndarray, optional): Grid of shape (M, 2). If None, generate grid from pos_range and n_pixel.
        method (str): 'exact', or 'fast' — determines which method to use.
        debug_plot (bool): If True, plot debugging information for each query point.

    Returns:
        result_prev_xy (np.ndarray): Filtered query points.
        result_new_xy (np.ndarray): KDE-smoothed estimates of new_xy corresponding to result_prev_xy.
    """

    assert prev_xy.shape == new_xy.shape

    if query_points is None:
        if pos_range is None:
            combined = np.vstack([prev_xy, new_xy])
            pos_range = (combined.min(), combined.max())
        x_grid = np.linspace(pos_range[0], pos_range[1], n_pixel)
        y_grid = np.linspace(pos_range[0], pos_range[1], n_pixel)
        xv, yv = np.meshgrid(x_grid, y_grid)
        query_points = np.column_stack([xv.ravel(), yv.ravel()])
        # query_points = prev_xy

    # # Automatically switch to fast method if large M×N
    # if method == 'auto':
    #     M, N = len(query_points), len(prev_xy)
    #     method = 'fast' if M * N > 1e6 else 'exact'

    if method == 'exact':
        distances = cdist(query_points, prev_xy)
        weights = np.exp(-0.5 * (distances / sigma)**2)
        weights_sum = weights.sum(axis=1, keepdims=True)
        weights_normalized = np.where(weights_sum > min_weight, weights / weights_sum, 0)
        smoothed_new = weights_normalized @ new_xy
        # smoothed_prev = weights_normalized @ prev_xy
        return query_points, smoothed_new

    elif method == 'fast':
        tree = cKDTree(prev_xy)
        radius = 3*sigma
        result_prev_xy = []
        result_new_xy = []
        # debugging parameters
        max_plots = 5
        j = 0
        for i,q in enumerate(query_points):
            idx = tree.query_ball_point(q, r=radius)
            # if not idx:
            #     smoothed_new.append([np.nan, np.nan])
            #     continue

            if not idx or len(idx) < min_points_threshold:
                continue  # Skip if not enough points

            dists = np.linalg.norm(prev_xy[idx] - q, axis=1)
            weights = np.exp(-0.5 * (dists / sigma) ** 2)
            total_weight = weights.sum()
            if total_weight < min_weight:
                continue  # Skip if total weight is too low

            weighted_new = (weights[:, None] * new_xy[idx]).sum(axis=0) / total_weight
            weighted_prev = (weights[:, None] * prev_xy[idx]).sum(axis=0) / total_weight
            # Use weighted_prev instead of query point q (as indicated by comment)
            # This ensures result_prev_xy and result_new_xy have matching lengths
            result_prev_xy.append(weighted_prev) # instead of q
            result_new_xy.append(weighted_new)

            if debug_plot:
                if i>j*5000 and j < max_plots:
                    # === Visualization for debugging ===
                    plt.figure(figsize=(6, 6))
                    plt.scatter(prev_xy[idx][:, 0], prev_xy[idx][:, 1], color='blue', label='Prev points')
                    plt.scatter(new_xy[idx][:, 0], new_xy[idx][:, 1], color='green', label='New points')
                    plt.scatter(q[0], q[1], color='red', marker='x', label='Grid Point')
                    plt.scatter(weighted_new[0], weighted_new[1], color='purple', marker='*', s=100, label='Weighted New')
                    if boundary_points is not None:
                        room_keys = np.unique(boundary_points[:, -1])
                        for room_key in room_keys:
                            room_boundary_points = boundary_points[boundary_points[:, -1] == room_key]
                            room_boundary_points = np.vstack((room_boundary_points, room_boundary_points[0]))
                            plt.plot(room_boundary_points[:, 0] , room_boundary_points[:, 1], '-', color='black', alpha=1.0,markersize=3)
                    plt.legend()
                    plt.title(f'Debugging Point {i}')
                    plt.xlabel('X')
                    plt.ylabel('Y')
                    plt.xlim(pos_range)
                    plt.ylim(pos_range)
                    plt.grid(True)
                    plt.show()

                    j=j+1

        result_prev_xy = np.array(result_prev_xy)
        result_new_xy = np.array(result_new_xy)
        if result_prev_xy.size == 0 or result_new_xy.size == 0:
            return np.empty((0, 2)), np.empty((0, 2))
        return result_prev_xy, result_new_xy

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'exact', 'fast', or 'auto'.")

def mapping_spatial_binning_smoothing(prev_xy, new_xy, pos_range=None, n_pixel=34, min_points=1):
    """
    Snap prev_xy points to nearest grid location, group by bin, and average both prev and new points in each bin.

    Parameters:
        prev_xy (np.ndarray): Source coordinates (N, 2)
        new_xy (np.ndarray): Target coordinates (N, 2)
        pos_range (tuple, optional): Tuple (min, max) for both axes. Default is inferred from data.
        n_pixel (int): Number of grid points along each axis (for square grid).
        min_points (int): Minimum number of points required in a bin to include it.

    Returns:
        smoothed_prev (np.ndarray): Averaged prev_xy per bin.
        smoothed_new (np.ndarray): Averaged new_xy per bin.
    """
    assert prev_xy.shape == new_xy.shape

    # Infer pos_range from data if not provided
    if pos_range is None:
        combined = np.vstack([prev_xy, new_xy])
        pos_range = (combined.min(), combined.max())

    # Create grid
    x_grid = np.linspace(pos_range[0], pos_range[1], n_pixel)
    y_grid = np.linspace(pos_range[0], pos_range[1], n_pixel)

    # Snap prev_xy points to nearest grid index
    x_idx = np.abs(prev_xy[:, 0, np.newaxis] - x_grid).argmin(axis=1)
    y_idx = np.abs(prev_xy[:, 1, np.newaxis] - y_grid).argmin(axis=1)
    bin_keys = list(zip(x_idx, y_idx))

    # Group matched points by grid bin
    bins = defaultdict(lambda: {'prev': [], 'new': []})
    for i, key in enumerate(bin_keys):
        bins[key]['prev'].append(prev_xy[i])
        bins[key]['new'].append(new_xy[i])

    # Average within each bin
    smoothed_prev = []
    smoothed_new = []
    for group in bins.values():
        if len(group['prev']) >= min_points:
            smoothed_prev.append(np.mean(group['prev'], axis=0))
            smoothed_new.append(np.mean(group['new'], axis=0))

    return np.array(smoothed_prev), np.array(smoothed_new)

def mapping_spatial_binning_snap(prev_xy, new_xy, pos_range=None, n_pixel=42):
    """
    Snap both prev_xy and new_xy to the nearest grid points, keeping all matched pairs.

    Parameters:
        prev_xy (np.ndarray): Source coordinates (N, 2)
        new_xy (np.ndarray): Target coordinates (N, 2)
        pos_range (tuple, optional): Tuple (min, max) for both axes. If None, inferred from combined data.
        n_pixel (int): Number of grid points along each axis.

    Returns:
        snapped_prev (np.ndarray): Snapped prev_xy coordinates (N, 2)
        snapped_new (np.ndarray): Snapped new_xy coordinates (N, 2)
    """
    assert prev_xy.shape == new_xy.shape

    # Infer range from all coordinates if not provided
    if pos_range is None:
        combined = np.vstack([prev_xy, new_xy])
        pos_range = (combined.min(), combined.max())

    # Create grid
    x_grid = np.linspace(pos_range[0], pos_range[1], n_pixel)
    y_grid = np.linspace(pos_range[0], pos_range[1], n_pixel)

    def snap(points):
        x_snapped = x_grid[np.abs(points[:, 0, np.newaxis] - x_grid).argmin(axis=1)]
        y_snapped = y_grid[np.abs(points[:, 1, np.newaxis] - y_grid).argmin(axis=1)]
        return np.stack([x_snapped, y_snapped], axis=1)

    snapped_prev = snap(prev_xy)
    snapped_new = snap(new_xy)

    return snapped_prev, snapped_new


def get_projection_stats_along_axis(pos_range, n_pixel, room_boundary_points, direction_vector):
    """
    Computes the variance and max distance of projected room shape along a single direction vector.

    Parameters:
        pos_range (tuple): (min, max) spatial range for both X and Y axes.
        n_pixel (int): Number of pixels (resolution) per axis for grid sampling.
        room_boundary_points (np.ndarray): Array of boundary points for the room (shape: Nx2).
        direction_vector (np.ndarray): 2D vector specifying the direction to project onto (shape: (2,)).

    Returns:
        tuple: (mean, variance) of projections along the given direction.
    """
    # Create a grid of 2D points
    x_vec = np.linspace(pos_range[0], pos_range[1], n_pixel, dtype=np.float64)
    y_vec = np.linspace(pos_range[0], pos_range[1], n_pixel, dtype=np.float64)
    X, Y = np.meshgrid(x_vec, y_vec)
    points = np.vstack([X.ravel(), Y.ravel()]).T

    # Filter points inside the room polygon
    polygon = Polygon(room_boundary_points[:, :2])
    mask = np.array([polygon.covers(Point(p)) for p in points])
    valid_points = points[mask]

    # center the valid points around the origin
    valid_points = valid_points - np.mean(valid_points, axis=0)

    # debug plot
    # plt.figure(figsize=(6, 6))
    # plt.scatter(valid_points[:, 0], valid_points[:, 1])
    # plt.gca().set_aspect('equal')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.grid(True, linestyle=':', linewidth=0.5)
    # plt.show()

    if valid_points.shape[0] == 0:
        return np.nan, np.nan

    # Normalize direction vector for meaningful projection
    direction_unit = direction_vector / np.linalg.norm(direction_vector)

    # Project points onto the direction vector
    projections = valid_points @ direction_unit  # shape (N,)

    # # debug Plot histogram
    # plt.figure(figsize=(6, 4))
    # plt.hist(projections, bins=50, color='purple', edgecolor='black', alpha=0.7)
    # plt.title('Projection Histogram')
    # plt.xlabel('Projection Value')
    # plt.ylabel('Frequency')
    # plt.grid(True, linestyle=':', linewidth=0.5)
    # plt.tight_layout()
    # plt.show()

    # Compute mean and variance of projections
    minmax = np.max(projections)- np.min(projections)
    variance = np.var(projections)

    return variance, minmax


def get_room_grid_points(room_boundary_points,pos_range,n_pixel):
    x_vec = np.linspace(pos_range[0], pos_range[1], n_pixel)
    y_vec = np.linspace(pos_range[0], pos_range[1], n_pixel)
    X, Y = np.meshgrid(x_vec, y_vec)
    grid_points = np.vstack([X.ravel(), Y.ravel()]).T
    polygon = Polygon(room_boundary_points[:, :2])
    mask = np.array([polygon.covers(Point(p)) for p in grid_points])
    return grid_points[mask]

def rotation_matrix(angle_deg, reflect=False):
    theta = np.radians(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    if reflect:
        R[0, :] *= -1  # flip across Y-axis
    return R

def variance_mismatch(X1, X2):
    cov1 = np.cov(X1.T)
    cov2 = np.cov(X2.T)
    return np.linalg.norm(cov1 - cov2, ord='fro')

def grid_occupancy(points, pos_range, n_pixel):
    min_pos, max_pos = pos_range
    hist, _, _ = np.histogram2d(points[:, 0], points[:, 1],
                                 bins=n_pixel,
                                 range=[[min_pos, max_pos],
                                        [min_pos, max_pos]])
    return (hist > 0).astype(int)

def jaccard_index(grid1, grid2):
    intersection = np.sum((grid1 & grid2))
    union = np.sum((grid1 | grid2))
    return intersection / union if union > 0 else 0





def compute_border_score_and_length_for_cell(
    rate_map_room,
    x_grid, y_grid,
    room_boundary_points,
    threshold_percentile=0.7,
    end_dist=6,
    max_band_width=6,
    min_band_width=2,
    max_strip_ratio=0.5,
    debug_plot=False
):
    """
    Computes the border score and border length for a given cell's rate map in each room.

    Args:
        df_data (pd.DataFrame): DataFrame containing the data for the cell.
        cell_column (str): Column name in df_data representing the cell's spike rates.
        map_rooms (dict): Dictionary mapping room names to their properties.
        boundary_points (np.ndarray): Array of boundary points for each room.
        target_columns (list): Columns to use for rate map creation.
        smoothed_ratemap (bool): Whether to apply smoothing to rate map.
        pos_range (tuple): Range of positions for map binning. If None, inferred.
        sigma (float): Std of Gaussian for smoothing.
        kernel_size (int): Kernel size for smoothing.
        n_pixel (int): Number of pixels in each dimension.
        threshold_percentile (float): Percentile threshold (between 0 and 1).
        end_dist (int): Maximum distance from wall for band.
        max_band_width (int or None): Optional max band width.
        min_band_width (int): Minimum band width.
        max_strip_ratio (float): Maximum ratio of strip width to room size.
        debug_plot (bool): Whether to plot debug information.

    Returns:
        (dict, dict) or (dict, dict, dict): border_score, border_length, [rate_maps]
    """

    def _get_mask_inside_polygon(room_boundary_points, x_grid, y_grid):
        X, Y = np.meshgrid(x_grid, y_grid)
        coords = np.stack([X.ravel(), Y.ravel()], axis=1)
        height, width = len(y_grid), len(x_grid)

        room_boundary = Polygon(room_boundary_points[:, :2])
        mask_inside = np.array([room_boundary.covers(Point(p)) for p in coords]).reshape((height, width))
        return mask_inside

    def _threshold_rate_map(rate_map, mask_inside, threshold_percentile):
        rate_map_masked = np.where(mask_inside, rate_map, 0)
        nonzero_vals = rate_map_masked[rate_map_masked > 0]
        if len(nonzero_vals) == 0:
            return None
        threshold = np.quantile(nonzero_vals, threshold_percentile)
        return np.where(rate_map_masked >= threshold, rate_map_masked, 0)

    def _compute_best_border_strip(rate_map_thresh, mask_inside, distance_map, end_dist, max_band_width,min_band_width,max_strip_ratio):
        best_score = -np.inf
        best_strip_mask = None

        for start in range(end_dist + 1):
            for stop in range(start+1, end_dist + 1):
                if max_band_width is not None and (stop - start) > max_band_width:
                    continue
                if (stop - start) < min_band_width:
                    continue

                in_mask = (distance_map > (start)) & (distance_map <= stop) & mask_inside
                out_mask = mask_inside & ~in_mask


                S_in, S_out = in_mask.sum(), out_mask.sum()
                if S_in == 0 or S_out == 0:
                    continue
                if S_in / (S_in + S_out) > max_strip_ratio:
                    continue

                P_in = rate_map_thresh[in_mask].sum()
                P_out = rate_map_thresh[out_mask].sum()
                denom = P_in * S_out + P_out * S_in
                if denom == 0:
                    continue

                score = (P_in * S_out - P_out * S_in) / denom
                if score > best_score:
                    best_score = score
                    best_strip_mask = in_mask

                    if debug_plot:
                        _debug_plot(rate_map_thresh, f"Best Strip (Score: {score:.2f}) - S_in: {S_in}, S_out: {S_out}, P_in: {P_in:.2f}, P_out: {P_out:.2f}", overlay=in_mask)

        return best_score, best_strip_mask

    def _calculate_border_length(rate_map, best_strip_mask ,debug_plot=False):
        blob_map = rate_map * best_strip_mask
        peak_val = blob_map.max()
        if peak_val == 0:
            return 0.0

        field_mask = blob_map >= 0.3 * peak_val
        peak_idx = np.unravel_index(blob_map.argmax(), blob_map.shape)
        labeled_blobs, _ = label(field_mask, structure=np.ones((3, 3)))
        blob_id = labeled_blobs[peak_idx]
        blob_mask = (labeled_blobs == blob_id)

        blob_within_strip = blob_mask & best_strip_mask

        length_score = blob_within_strip.sum() / best_strip_mask.sum()

        if debug_plot:
            blob_overlay = blob_mask.astype(float) if best_strip_mask is not None else None
            _debug_plot(blob_within_strip, f"Final Blob in Strip (Length Score: {length_score:.2f})",overlay=blob_overlay)

        return length_score

    def _debug_plot(data, title="", cmap="viridis", overlay=None, alpha=0.5):
        plt.figure(figsize=(6, 6))
        plt.imshow(data, origin="lower", cmap=cmap)
        if overlay is not None:
            plt.imshow(overlay, origin="lower", cmap="Reds", alpha=alpha)
        plt.colorbar()
        plt.title(title)
        plt.tight_layout()
        plt.show()

    # --- Main loop over rooms ---

    if debug_plot: _debug_plot(rate_map_room, f"Rate Map")

    mask_inside = _get_mask_inside_polygon(room_boundary_points, x_grid, y_grid)
    if debug_plot: _debug_plot(mask_inside, f"Inside Polygon Mask", cmap="Greys")
    rate_map_thresh = _threshold_rate_map(rate_map_room, mask_inside, threshold_percentile)
    if debug_plot: _debug_plot(rate_map_thresh, f"Thresholded Rate Map")

    if rate_map_thresh is None or np.sum(rate_map_thresh) == 0:
        border_score = np.nan
        border_length = 0.0
        return border_score, border_length

    distance_map = distance_transform_edt(mask_inside)
    score, best_strip_mask = _compute_best_border_strip(rate_map_thresh, mask_inside, distance_map, end_dist, max_band_width,min_band_width, max_strip_ratio)
    if best_strip_mask is not None and debug_plot: _debug_plot(rate_map_thresh, f"Best Strip Overlay (Score: {score:.2f})", overlay=best_strip_mask)

    border_score = score

    if best_strip_mask is None or np.sum(best_strip_mask) == 0:
        border_length = 0.0
    else:
        border_length = _calculate_border_length(rate_map_room, best_strip_mask, debug_plot)


    return border_score, border_length





def _ray_hit_distance(p_xy, theta_deg, polygon: Polygon, ray_length=None):
    """
    Distance from point p_xy along direction theta_deg (deg) until intersecting the polygon boundary.
    Returns np.nan if no forward intersection is found.
    """
    x, y = float(p_xy[0]), float(p_xy[1])
    if ray_length is None:
        minx, miny, maxx, maxy = polygon.bounds
        # generous length: a few-times the diagonal of the bbox
        ray_length = 4.0 * np.hypot(maxx - minx, maxy - miny)

    # build the forward ray
    th = np.deg2rad(theta_deg)
    dx, dy = np.cos(th), np.sin(th)
    ray = LineString([(x, y), (x + ray_length * dx, y + ray_length * dy)])

    inter = polygon.exterior.intersection(ray)

    # Normalize intersections to a list of candidate points
    candidates = []
    if inter.is_empty:
        return np.nan
    if inter.geom_type == "Point":
        candidates = [inter]
    elif inter.geom_type in ("MultiPoint", "GeometryCollection"):
        for g in inter.geoms:
            if g.geom_type == "Point":
                candidates.append(g)
            elif g.geom_type == "LineString":  # rare: overlapping segment with the ray
                # sample both segment endpoints
                coords = list(g.coords)
                if len(coords) >= 2:
                    candidates.append(Point(coords[0]))
                    candidates.append(Point(coords[-1]))
    elif inter.geom_type == "LineString":
        coords = list(inter.coords)
        if len(coords) >= 2:
            candidates = [Point(coords[0]), Point(coords[-1])]

    if not candidates:
        return np.nan

    # Keep only forward hits (non-negative projection on the ray direction)
    best = np.inf
    for pt in candidates:
        vx, vy = pt.x - x, pt.y - y
        proj = vx * dx + vy * dy  # signed distance along the ray direction
        if proj >= 0:
            # euclidean distance equals proj only if perfectly colinear; use hypot for safety
            d = np.hypot(vx, vy)
            if d < best:
                best = d
    return best if np.isfinite(best) else np.nan


import numpy as np
from shapely.geometry import Point, Polygon
from shapely.prepared import prep
from shapely.ops import nearest_points

def _compute_allocentric_distance_matrix(
    positions_xy: np.ndarray,
    room_boundary_points: np.ndarray,
    angles_deg: list[float],
    *,
    # sentinels / behavior
    fill_nohit: float = -1.0,          # write when a ray doesn't intersect
    outside_fill: float = -1.0,        # write for entire row if far outside
    # fewer-outsides controls
    nearby_eps_norm: float = 0.05,     # ~5 cm in normalized units
    nudge_if_nearby: bool = True,      # snap+push small-outsides back inside
    buffer_for_contains: bool = False,  # use polygon.buffer(eps) for contains test
    # ray casting
    dynamic_ray: bool = False           # extend ray based on distance to boundary
) -> np.ndarray:
    """
    Returns distances (N x n_angles). Points slightly outside (<= nearby_eps_norm) are
    either accepted via buffered 'inside' test and/or nudged inward before ray casting.
    Distances are always computed vs the ORIGINAL polygon boundary.
    """
    polygon = Polygon(np.asarray(room_boundary_points)[:, :2])
    # prepared = prep(polygon)

    # buffered polygon for 'inside' tolerance (won't be used for intersections)
    poly_for_contains = polygon.buffer(nearby_eps_norm, join_style=2) if buffer_for_contains else polygon
    prepared_contains = prep(poly_for_contains)

    # base ray length from bbox
    minx, miny, maxx, maxy = polygon.bounds
    base_ray_len = 4.0 * float(np.hypot(maxx - minx, maxy - miny))

    N = positions_xy.shape[0]
    n_angles = len(angles_deg)
    D = np.empty((N, n_angles), dtype=float)

    # Pre-pick an interior target for nudge direction
    interior_pt = polygon.representative_point()
    cx, cy = interior_pt.x, interior_pt.y

    for i in range(N):
        px, py = float(positions_xy[i, 0]), float(positions_xy[i, 1])
        pt = Point(px, py)

        if pt.is_empty or polygon.exterior.is_empty:
            D[i, :] = np.nan
            continue

        # Inside (or on edge) with tolerance?
        if prepared_contains.covers(pt):
            use_px, use_py = px, py
        else:
            # Far outside?
            dist_to_poly = pt.distance(polygon.exterior)
            if dist_to_poly > nearby_eps_norm or not nudge_if_nearby:
                D[i, :] = outside_fill
                continue

            # Near-miss: snap to boundary and push epsilon inside
            # nearest on boundary:
            _, n_on = nearest_points(pt, polygon.exterior)
            nx, ny = n_on.x, n_on.y
            # push direction: toward polygon interior
            vx, vy = cx - nx, cy - ny
            norm = np.hypot(vx, vy)
            if norm < 1e-9:
                # fallback: tiny inward push along +y
                vx, vy, norm = 0.0, 1.0, 1.0
            ux, uy = vx / norm, vy / norm
            use_px, use_py = nx + nearby_eps_norm * ux, ny + nearby_eps_norm * uy

        # choose ray length
        if dynamic_ray:
            # + a couple of boundary distances to be safe
            extra = 2.0 * float(Point(use_px, use_py).distance(polygon.exterior))
            ray_len = max(base_ray_len, base_ray_len + extra)
        else:
            ray_len = base_ray_len

        # cast rays from (use_px, use_py) against ORIGINAL polygon
        for j, a in enumerate(angles_deg):
            d = _ray_hit_distance((use_px, use_py), a, polygon, ray_length=ray_len)
            D[i, j] = fill_nohit if (d is None or np.isnan(d)) else float(d)

    return D




def polar_rate_map_from_positions(df_room_data,
                                  position_columns=('X', 'Y'),
                                  room_boundary_points=None,
                                  distance_range=None,
                                  bin_size=None,
                                  angles=None,
                                  softmax_tau=None,
                                  **kwargs):
    """
    Convenience wrapper:
      1) compute allocentric distance matrix from positions+boundary using shapely
      2) call create_polar_rate_map with ones-weights
      3) return (rate_map, distance_bins, angles_bins, count_map)

    Parameters
    ----------
    df_room_data : pandas.DataFrame
        Data for one room (already filtered). Must have position columns.
    position_columns : tuple[str, str]
        Name of X/Y columns.
    room_boundary_points : np.ndarray (M,2 or more)
        Boundary vertices in order (units must match positions).
    distance_range, bin_size, angles, softmax_tau : as you currently use them
    create_polar_rate_map : function
        Pass in the existing create_polar_rate_map reference from utils.analysis
        if you’re wiring this from elsewhere. If you drop this into utils/analysis.py,
        you can import and call it directly instead of passing it.
    **kwargs
        Any extra keyword args forwarded to create_polar_rate_map, e.g.:
        smoothed=True, sigma=[2,1], kernel_size=[7,3], weighting_method='softmax',
        return_count=True, wrap_angles=True, fill_value=0

    Returns
    -------
    rate_map, distance_bins, angles_bins, count_map
    """
    if angles is None or room_boundary_points is None or distance_range is None or bin_size is None or softmax_tau is None:
        raise ValueError("angles, room_boundary_points, distance_range, bin_size, and softmax_tau are required.")

    positions_xy = df_room_data[list(position_columns)].to_numpy(dtype=float)

    distance_matrix = _compute_allocentric_distance_matrix(
        positions_xy=positions_xy,
        room_boundary_points=np.asarray(room_boundary_points),
        angles=list(angles)
    )

    # weights: one per sample
    sample_weights = np.ones(len(df_room_data), dtype=float)

    rate_map, distance_bins, angles_bins, count_map = create_polar_rate_map(
        distance_matrix,
        sample_weights,
        distance_range=distance_range,
        bin_size=bin_size,
        angles_deg=angles,
        weighting_method='softmax',
        softmax_tau=softmax_tau,
        **kwargs
    )

    return rate_map, distance_bins, angles_bins, count_map


def circular_std(angles, period=360.0, deg=True):
    """
    Compute circular standard deviation of angles.
    
    Uses formula: sqrt(-2 * log(R)) where R is the mean resultant length.
    
    Parameters:
        angles (np.ndarray): Array of angles
        period (float): Period of the circular variable (360 for rotations, 180 for reflections)
        deg (bool): If True, angles are in degrees; if False, in radians
    
    Returns:
        float: Circular standard deviation (in same units as input)
    """
    angles = np.asarray(angles, dtype=float)
    if angles.size == 0:
        return np.nan
    
    # Convert to radians if needed
    if deg:
        angles_rad = np.deg2rad(angles)
        period_rad = np.deg2rad(period)
    else:
        angles_rad = angles
        period_rad = period
    
    # Normalize to [0, period)
    angles_rad = angles_rad % period_rad
    
    # Convert to unit circle: exp(i * theta)
    complex_vectors = np.exp(1j * 2 * np.pi * angles_rad / period_rad)
    
    # Mean resultant length R
    R = np.abs(np.mean(complex_vectors))
    
    # Circular standard deviation: sqrt(-2 * log(R))
    # Handle edge cases
    if R <= 0:
        return np.inf
    if R >= 1.0:
        return 0.0
    
    circ_std_rad = np.sqrt(-2 * np.log(R))
    
    # Convert back to original units
    if deg:
        # Convert from radians to degrees, accounting for period
        circ_std = circ_std_rad * period_rad / (2 * np.pi) * 360.0 / period_rad
    else:
        circ_std = circ_std_rad * period_rad / (2 * np.pi)
    
    return circ_std


def circular_variance(angles, period=360.0, deg=True):
    """
    Compute circular variance of angles.
    
    Circular variance = 1 - R, where R is the mean resultant length.
    Range: [0, 1], where 0 = no variance (all angles identical), 1 = uniform distribution.
    
    Parameters:
        angles (np.ndarray): Array of angles
        period (float): Period of the circular variable (360 for rotations, 180 for reflections)
        deg (bool): If True, angles are in degrees; if False, in radians
    
    Returns:
        float: Circular variance (dimensionless, range [0, 1])
    """
    angles = np.asarray(angles, dtype=float)
    if angles.size == 0:
        return np.nan
    
    # Convert to radians if needed
    if deg:
        angles_rad = np.deg2rad(angles)
        period_rad = np.deg2rad(period)
    else:
        angles_rad = angles
        period_rad = period
    
    # Normalize to [0, period)
    angles_rad = angles_rad % period_rad
    
    # Convert to unit circle: exp(i * theta)
    complex_vectors = np.exp(1j * 2 * np.pi * angles_rad / period_rad)
    
    # Mean resultant length R
    R = np.abs(np.mean(complex_vectors))
    
    # Circular variance = 1 - R
    return 1.0 - R


def wrap_angle(angle, period=None, symmetric=True, deg=True):
    """
    Wrap angles into a chosen interval of length 'period'.

    Args:
        angle: scalar or array of angles
        period: float, the period (360 for degrees, 2*np.pi for radians, 180 for axial)
        symmetric:
            - True  -> (-period/2, period/2]
            - False -> [0, period)
        deg: if True, interpret inputs/outputs in degrees; if False, in radians

    Returns:
        wrapped angles, same shape as input
    """
    period = period or (360.0 if deg else 2 * np.pi)
    angle = np.asarray(angle, dtype=float)
    if symmetric:
        return (angle + period / 2) % period - period / 2
    else:
        return angle % period


def angle_difference(a, b, warp=True, **warp_kwargs):
    """
    Minimal difference a - b on the circle.

    Args:
        a, b: scalar or array-like of angles
        warp: if True, wrap output angle to (-period/2, period/2]
        warp_kwargs: additional args passed to wrap_angle if warp=True
    """
    diff = np.asarray(a) - np.asarray(b)
    return wrap_angle(diff, **warp_kwargs) if warp else diff


def angle_average(angles, weights=None, deg=True, warp=True, **warp_kwargs):
    """
    Circular mean of one or more angles.

    Args:
        angles: array-like of angles
        weights: optional array-like of weights, same shape as angles
        deg: if True, interpret inputs/outputs in degrees; if False, in radians
        warp: if True, wrap output angle to (-period/2, period/2]
        warp_kwargs: additional args passed to wrap_angle if warp=True

    Returns:
        circular mean angle, wrapped to (-period/2, period/2]
    """
    angles = np.asarray(angles, dtype=float)
    if weights is None:
        weights = np.ones_like(angles, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    # convert to radians
    ar = np.deg2rad(angles) if deg else angles

    # weighted vector sum
    x = np.sum(weights * np.cos(ar))
    y = np.sum(weights * np.sin(ar))

    if np.isclose(x, 0) and np.isclose(y, 0):
        return np.nan  # undefined mean (vectors cancel)

    mean = np.arctan2(y, x)
    out = np.rad2deg(mean) if deg else mean
    return wrap_angle(out, deg=deg, **warp_kwargs) if warp else out




import os
import numpy as np
import pandas as pd

def export_tracking_stats(
    dfPos,
    streams,                 # list of stream bases like "PC_HeadPos" (None -> auto-discover)
    out_dir,                 # output folder
    config,                  # reads thresholds from config["preprocessing"]
    save_name_prefix="interp_stats",
):
    """
    TS-based interpolation + velocity exporter with FPS statistics.

    Reads from config['preprocessing']:
      - position_label
      - bin_width_s (default 60)
      - long_time_threshold_s (default 3)
      - long_dist_threshold_cm (default 15)
      - v_local_window_s (default 2.0)
      - speed_max_cm_s (default None)
      - z_speed_thresh (default 3.5)
      - lowP_thresh (default 0.3)

    Exports:
      - {prefix}_summary.csv       : per (source, stream) headline metrics
      - {prefix}_gap_runs.csv      : one row per interpolated run with distance_cm & flags
      - {prefix}_per_timebin.csv   : per-bin stats (bin_width_s)
      - {prefix}_fps_stats.csv     : recording-level FPS & Δt stats derived from TS
    """
    # ------------------ read params ------------------
    pp = (config or {}).get("preprocessing", {}) or {}
    bin_width_s            = float(pp.get("bin_width_s", 60.0))
    long_time_threshold_s  = float(pp.get("long_time_threshold_s", 3.0))
    long_dist_threshold_cm = float(pp.get("long_dist_threshold_cm", 15.0))
    v_local_window_s       = float(pp.get("v_local_window_s", 2.0))
    speed_max_cm_s_raw     = pp.get("speed_max_cm_s", None)
    speed_max_cm_s         = (float(speed_max_cm_s_raw) if speed_max_cm_s_raw is not None else None)
    z_speed_thresh         = float(pp.get("z_speed_thresh", 3.5))
    lowP_thresh            = float(pp.get("lowP_thresh", 0.3))
    primary_stream         = str(pp.get("position_label", "PC_HeadPos"))

    os.makedirs(out_dir, exist_ok=True)

    # ------------------ helpers ------------------
    def _get_ts(df):
        if "TS" in df.columns:
            return df["TS"].to_numpy(dtype=float)
        elif "timestamps" in df.columns:
            return df["timestamps"].to_numpy(dtype=float)
        else:
            raise ValueError("dfPos must contain a 'TS' or 'timestamps' column (seconds).")

    def _instant_speed_from_xy_ts(ts, x, y):
        dt = np.diff(ts)
        dxy = np.hypot(np.diff(x), np.diff(y))
        v = np.divide(dxy, dt, out=np.full_like(dxy, np.nan), where=dt > 0)
        return np.r_[np.nan, v]  # align length

    def _frames_for_seconds(ts, window_s):
        dt_med = np.nanmedian(np.diff(ts))
        if not np.isfinite(dt_med) or dt_med <= 0:
            return 1
        return max(1, int(round(window_s / dt_med)))

    def _robust_mad_stats(x):
        x = np.asarray(x, dtype=float)
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
        return med, (mad if mad > 0 else np.nan)

    def _zscore(val, med, mad):
        if not np.isfinite(med) or not np.isfinite(mad) or mad == 0:
            return np.nan
        return (val - med) / mad

    def _find_true_runs(mask):
        if mask.size == 0:
            return []
        diff = np.diff(mask.astype(int), prepend=0, append=0)
        starts = np.flatnonzero(diff == 1)
        ends   = np.flatnonzero(diff == -1) - 1
        return [(int(s), int(e), int(e - s + 1)) for s, e in zip(starts, ends)]

    def _source_of(stream_key):
        return stream_key.split("_", 1)[0] if "_" in stream_key else stream_key.split(".", 1)[0]

    # Auto-discover streams if not provided
    if streams is None:
        bases = []
        for c in dfPos.columns:
            if c.endswith(".X"):
                base = c[:-2]
                if base.startswith(("TC_", "PC_", "EC_")) and f"{base}.Y" in dfPos.columns:
                    bases.append(base)
        streams = sorted(set(bases))

    # ------------------ global TS/FPS stats ------------------
    ts = _get_ts(dfPos)
    n = len(ts)
    if n < 2:
        raise ValueError("Not enough timestamps to compute FPS statistics.")

    dt = np.diff(ts)
    # instantaneous FPS from dt
    fps_inst = np.divide(1.0, dt, out=np.full_like(dt, np.nan), where=dt > 0)

    # frames-per-second counts (integer seconds buckets)
    sec = (ts // 1).astype(int)
    # drop last element of sec to align with dt/fps_inst edges if needed
    # but for counts per second we count frames by exact timestamp second
    fps_per_sec = pd.Series(1, index=sec).groupby(level=0).sum().astype(float)
    # normalize: some seconds may be missing; we’ll still summarize existing ones

    def _stat(arr, fn, default=np.nan):
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0 or not np.any(np.isfinite(arr)):
            return default
        return float(fn(arr[np.isfinite(arr)]))

    fps_stats = {
        "recording_duration_sec": float(ts[-1] - ts[0]) if np.isfinite(ts[-1] - ts[0]) else np.nan,
        "frames_total": int(n),
        # dt stats
        "dt_median_s": _stat(dt, np.nanmedian),
        "dt_mean_s":   _stat(dt, np.nanmean),
        "dt_p05_s":    _stat(dt, lambda a: np.nanpercentile(a, 5)),
        "dt_p95_s":    _stat(dt, lambda a: np.nanpercentile(a, 95)),
        "dt_min_s":    _stat(dt, np.nanmin),
        "dt_max_s":    _stat(dt, np.nanmax),
        # instantaneous FPS stats
        "fps_inst_median": _stat(fps_inst, np.nanmedian),
        "fps_inst_mean":   _stat(fps_inst, np.nanmean),
        "fps_inst_p05":    _stat(fps_inst, lambda a: np.nanpercentile(a, 5)),
        "fps_inst_p95":    _stat(fps_inst, lambda a: np.nanpercentile(a, 95)),
        "fps_inst_min":    _stat(fps_inst, np.nanmin),
        "fps_inst_max":    _stat(fps_inst, np.nanmax),
        # frames per second (counted)
        "fps_count_median": _stat(fps_per_sec.values, np.nanmedian),
        "fps_count_mean":   _stat(fps_per_sec.values, np.nanmean),
        "fps_count_p05":    _stat(fps_per_sec.values, lambda a: np.nanpercentile(a, 5)),
        "fps_count_p95":    _stat(fps_per_sec.values, lambda a: np.nanpercentile(a, 95)),
        "fps_count_min":    _stat(fps_per_sec.values, np.nanmin),
        "fps_count_max":    _stat(fps_per_sec.values, np.nanmax),
    }

    # ------------------ per-stream computations ------------------
    bin_ids = (ts // bin_width_s).astype(int)
    provided_speed_cols = {"TC": "TC_V", "PC": "PC_V", "EC": "EC_V"}

    summary_rows, timebin_rows, runs_rows = [], [], []

    for key in streams:
        if key == "":
            x_col, y_col = "X", "Y"
        else:
            x_col, y_col = f"{key}.X", f"{key}.Y"
        if x_col not in dfPos.columns or y_col not in dfPos.columns:
            continue

        x = dfPos[x_col].to_numpy(dtype=float)
        y = dfPos[y_col].to_numpy(dtype=float)
        P = dfPos.get(f"{key}.P", pd.Series(np.nan, index=dfPos.index)).to_numpy(dtype=float)
        Int = dfPos.get(f"{key}.Int", pd.Series(0, index=dfPos.index)).to_numpy(dtype=float)

        missing = np.isnan(x) | np.isnan(y)
        interpolated = (Int > 0) & (~missing)
        valid = (~missing) & (Int == 0)

        runs = _find_true_runs(interpolated)

        v_manual = _instant_speed_from_xy_ts(ts, x, y)
        src = _source_of(key)
        v_provided = dfPos[provided_speed_cols[src]].to_numpy(dtype=float) if (src in provided_speed_cols and provided_speed_cols[src] in dfPos.columns) else None

        # state per frame: 0=missing,1=valid,2=short,3=long_time,4=long_distance
        state = np.zeros(n, dtype=int)
        state[valid] = 1

        # collections
        run_lengths, run_durations, run_distances = [], [], []
        run_is_long_time, run_is_long_distance, run_is_long = [], [], []
        run_is_too_fast, run_is_too_fast_abs, run_is_too_fast_rel = [], [], []
        run_gap_avg_speed, run_gap_peak_speed, run_v_local_median = [], [], []
        run_z_avg, run_z_peak, run_p_mean = [], [], []

        for (s_idx, e_idx, L) in runs:
            t_start, t_end = ts[s_idx], ts[e_idx]
            duration = float(t_end - t_start) if np.isfinite(t_end - t_start) else np.nan

            dist = float(np.hypot(x[e_idx] - x[s_idx], y[e_idx] - y[s_idx])) if (np.isfinite(x[s_idx]) and np.isfinite(y[s_idx]) and np.isfinite(x[e_idx]) and np.isfinite(y[e_idx])) else np.nan

            is_long_t = bool(np.isfinite(duration) and (duration >= long_time_threshold_s))
            is_long_d = bool(np.isfinite(dist) and (dist >= long_dist_threshold_cm))
            is_long_u = bool(is_long_t or is_long_d)

            gap_avg_v = float(dist / duration) if (np.isfinite(dist) and duration > 0) else np.nan

            lo = max(0, s_idx - 1)
            hi = min(n - 1, e_idx + 1)
            gap_peak_v = float(np.nanmax(v_manual[lo:hi+1])) if np.any(np.isfinite(v_manual[lo:hi+1])) else np.nan

            win = _frames_for_seconds(ts, v_local_window_s)
            pre_lo = max(0, s_idx - win); pre_hi = s_idx
            post_lo = e_idx + 1; post_hi = min(n, e_idx + 1 + win)
            local_samples = np.r_[v_manual[pre_lo:pre_hi], v_manual[post_lo:post_hi]]
            v_loc_med, v_loc_mad = _robust_mad_stats(local_samples)

            z_avg  = _zscore(gap_avg_v,  v_loc_med, v_loc_mad)
            z_peak = _zscore(gap_peak_v, v_loc_med, v_loc_mad)

            is_fast_abs = False
            if speed_max_cm_s is not None and np.isfinite(speed_max_cm_s):
                is_fast_abs = (np.isfinite(gap_avg_v) and gap_avg_v >= speed_max_cm_s) or \
                              (np.isfinite(gap_peak_v) and gap_peak_v >= speed_max_cm_s)

            is_fast_rel = ((np.isfinite(z_avg)  and z_avg  >= z_speed_thresh) or
                           (np.isfinite(z_peak) and z_peak >= z_speed_thresh))

            is_fast = bool(is_fast_abs or is_fast_rel)

            run_lengths.append(L)
            run_durations.append(duration)
            run_distances.append(dist)
            run_is_long_time.append(is_long_t)
            run_is_long_distance.append(is_long_d)
            run_is_long.append(is_long_u)
            run_is_too_fast.append(is_fast)
            run_is_too_fast_abs.append(is_fast_abs)
            run_is_too_fast_rel.append(is_fast_rel)
            run_gap_avg_speed.append(gap_avg_v)
            run_gap_peak_speed.append(gap_peak_v)
            run_v_local_median.append(v_loc_med)
            run_z_avg.append(z_avg)
            run_z_peak.append(z_peak)
            run_p_mean.append(float(np.nanmean(P[s_idx:e_idx+1])) if np.any(np.isfinite(P[s_idx:e_idx+1])) else np.nan)

            if is_long_d:
                state[s_idx:e_idx+1] = 4
            elif is_long_t:
                state[s_idx:e_idx+1] = 3
            else:
                state[s_idx:e_idx+1] = 2

            runs_rows.append({
                "source": src,
                "stream": key,
                "is_primary": bool(key == primary_stream),
                "start_idx": s_idx,
                "end_idx": e_idx,
                "length_frames": L,
                "t_start": float(t_start),
                "t_end": float(t_end),
                "duration_sec": float(duration) if np.isfinite(duration) else np.nan,
                "distance_cm": float(dist) if np.isfinite(dist) else np.nan,
                "is_long_time": bool(is_long_t),
                "is_long_distance": bool(is_long_d),
                "is_long": bool(is_long_u),
                "gap_avg_speed_cm_s": float(gap_avg_v) if np.isfinite(gap_avg_v) else np.nan,
                "gap_peak_speed_cm_s": float(gap_peak_v) if np.isfinite(gap_peak_v) else np.nan,
                "v_local_median_cm_s": float(v_loc_med) if np.isfinite(v_loc_med) else np.nan,
                "z_gap_avg_speed": float(z_avg) if np.isfinite(z_avg) else np.nan,
                "z_gap_peak_speed": float(z_peak) if np.isfinite(z_peak) else np.nan,
                "is_gap_too_fast_absolute": bool(is_fast_abs),
                "is_gap_too_fast_relative": bool(is_fast_rel),
                "is_gap_too_fast": bool(is_fast),
                "p_mean_run": run_p_mean[-1],
                # provenance for this row
                "bin_width_s": float(bin_width_s),
                "long_time_threshold_s": float(long_time_threshold_s),
                "long_dist_threshold_cm": float(long_dist_threshold_cm),
                "speed_max_cm_s": (float(speed_max_cm_s) if speed_max_cm_s is not None else np.nan),
                "z_speed_thresh": float(z_speed_thresh),
                "v_local_window_s": float(v_local_window_s),
                "lowP_thresh": float(lowP_thresh),
            })

        # ---- per-stream summary ----
        frames_total = n
        n_missing = int((state == 0).sum())
        n_valid   = int((state == 1).sum())
        n_short   = int((state == 2).sum())
        n_ltime   = int((state == 3).sum())
        n_ldist   = int((state == 4).sum())

        pct_missing = 100.0 * n_missing / max(frames_total, 1)
        pct_valid   = 100.0 * n_valid   / max(frames_total, 1)
        pct_short   = 100.0 * n_short   / max(frames_total, 1)
        pct_ltime   = 100.0 * n_ltime   / max(frames_total, 1)
        pct_ldist   = 100.0 * n_ldist   / max(frames_total, 1)
        pct_long_union = 100.0 * (n_ltime + n_ldist) / max(frames_total, 1)

        def _arr(a): return np.array(a, dtype=float) if a else np.array([], dtype=float)
        lengths_arr, durations_arr, distances_arr = map(_arr, (run_lengths, run_durations, run_distances))

        is_long_time_arr     = np.array(run_is_long_time, dtype=bool) if run_is_long_time else np.array([], dtype=bool)
        is_long_distance_arr = np.array(run_is_long_distance, dtype=bool) if run_is_long_distance else np.array([], dtype=bool)
        is_long_arr          = np.array(run_is_long, dtype=bool) if run_is_long else np.array([], dtype=bool)
        is_too_fast_arr      = np.array(run_is_too_fast, dtype=bool) if run_is_too_fast else np.array([], dtype=bool)

        vman = _instant_speed_from_xy_ts(ts, x, y)
        vman = vman[np.isfinite(vman)]
        vman_stats = {
            "v_manual_mean": float(np.nanmean(vman)) if vman.size else np.nan,
            "v_manual_median": float(np.nanmedian(vman)) if vman.size else np.nan,
            "v_manual_p05": float(np.nanpercentile(vman, 5)) if vman.size else np.nan,
            "v_manual_p95": float(np.nanpercentile(vman, 95)) if vman.size else np.nan,
            "v_manual_max": float(np.nanmax(vman)) if vman.size else np.nan,
        }

        v_provided = dfPos[provided_speed_cols[_source_of(key)]].to_numpy(dtype=float) \
            if (_source_of(key) in provided_speed_cols and provided_speed_cols[_source_of(key)] in dfPos.columns) else None

        vprov_stats = {"v_provided_mean": np.nan, "v_provided_median": np.nan,
                       "v_provided_p05": np.nan, "v_provided_p95": np.nan, "v_provided_max": np.nan,
                       "speed_pearson_r": np.nan, "speed_bias_cm_s": np.nan, "speed_rmse_cm_s": np.nan}
        if v_provided is not None:
            vp = v_provided[np.isfinite(v_provided)]
            if vp.size:
                vprov_stats.update({
                    "v_provided_mean": float(np.nanmean(vp)),
                    "v_provided_median": float(np.nanmedian(vp)),
                    "v_provided_p05": float(np.nanpercentile(vp, 5)),
                    "v_provided_p95": float(np.nanpercentile(vp, 95)),
                    "v_provided_max": float(np.nanmax(vp)),
                })
            mask_both = np.isfinite(v_manual) & np.isfinite(v_provided)
            if np.count_nonzero(mask_both) >= 3:
                a, b = v_manual[mask_both], v_provided[mask_both]
                r = np.corrcoef(a, b)[0,1]
                diff = b - a
                vprov_stats.update({
                    "speed_pearson_r": float(r),
                    "speed_bias_cm_s": float(np.nanmean(diff)),
                    "speed_rmse_cm_s": float(np.sqrt(np.nanmean(diff**2))),
                })

        P_valid  = P[valid]
        P_interp = P[interpolated]
        mean_P_valid  = float(np.nanmean(P_valid))  if np.any(np.isfinite(P_valid))  else np.nan
        mean_P_interp = float(np.nanmean(P_interp)) if np.any(np.isfinite(P_interp)) else np.nan
        frac_lowP = float(np.mean(P < lowP_thresh)) if np.any(np.isfinite(P)) else np.nan

        n_gaps = len(runs)
        pct_runs_long_time     = (100.0 * is_long_time_arr.mean())     if n_gaps else 0.0
        pct_runs_long_distance = (100.0 * is_long_distance_arr.mean()) if n_gaps else 0.0
        pct_runs_long          = (100.0 * is_long_arr.mean())          if n_gaps else 0.0
        pct_runs_too_fast      = (100.0 * is_too_fast_arr.mean())      if n_gaps else 0.0

        def _p(arr, q):
            arr = arr[np.isfinite(arr)]
            return float(np.nanpercentile(arr, q)) if arr.size else np.nan

        summary_row = {
            "source": _source_of(key),
            "stream": key,
            "is_primary": bool(key == primary_stream),
            "frames_total": int(n),
            "frames_missing": int((state == 0).sum()),
            "frames_valid": int((state == 1).sum()),
            "frames_interp_short": int((state == 2).sum()),
            "frames_interp_long_time": int((state == 3).sum()),
            "frames_interp_long_distance": int((state == 4).sum()),
            "percent_missing": 100.0 * (state == 0).mean(),
            "percent_valid": 100.0 * (state == 1).mean(),
            "percent_interp_short": 100.0 * (state == 2).mean(),
            "percent_interp_long_time": 100.0 * (state == 3).mean(),
            "percent_interp_long_distance": 100.0 * (state == 4).mean(),
            "percent_interp_long_union": 100.0 * ((state == 3).mean() + (state == 4).mean()),
            "num_gaps": int(n_gaps),
            "gap_len_frames_median": float(_p(lengths_arr, 50)),
            "gap_len_frames_p95": float(_p(lengths_arr, 95)),
            "gap_len_frames_max": float(np.nanmax(lengths_arr)) if lengths_arr.size else np.nan,
            "gap_duration_sec_median": float(_p(durations_arr, 50)),
            "gap_duration_sec_p95": float(_p(durations_arr, 95)),
            "gap_duration_sec_max": float(np.nanmax(durations_arr)) if durations_arr.size else np.nan,
            "distance_cm_median": float(_p(distances_arr, 50)),
            "distance_cm_p95": float(_p(distances_arr, 95)),
            "distance_cm_max": float(np.nanmax(distances_arr)) if distances_arr.size else np.nan,
            "percent_runs_long_time": (100.0 * is_long_time_arr.mean()) if n_gaps else 0.0,
            "percent_runs_long_distance": (100.0 * is_long_distance_arr.mean()) if n_gaps else 0.0,
            "percent_runs_long": (100.0 * is_long_arr.mean()) if n_gaps else 0.0,
            "percent_runs_too_fast": (100.0 * is_too_fast_arr.mean()) if n_gaps else 0.0,
            "mean_P_valid": mean_P_valid,
            "mean_P_interp": mean_P_interp,
            "frac_lowP": frac_lowP,
            # provenance
            "bin_width_s": float(bin_width_s),
            "long_time_threshold_s": float(long_time_threshold_s),
            "long_dist_threshold_cm": float(long_dist_threshold_cm),
            "speed_max_cm_s": (float(speed_max_cm_s) if speed_max_cm_s is not None else np.nan),
            "z_speed_thresh": float(z_speed_thresh),
            "v_local_window_s": float(v_local_window_s),
            "lowP_thresh": float(lowP_thresh),
        }
        summary_row.update(vman_stats)
        summary_row.update(vprov_stats)
        summary_rows.append(summary_row)

        # ---------- per-time-bin (per stream) ----------
        df_tmp = pd.DataFrame({
            "bin": bin_ids,
            "state": state,
            "v_manual": v_manual,
            "v_provided": v_provided if v_provided is not None else np.nan,
            "P": P,
        })

        # base aggregates
        g = df_tmp.groupby("bin")
        tb = g.agg(
            count=("state", "size"),
            mean_speed_manual=("v_manual", "mean"),
            mean_speed_provided=("v_provided", "mean"),
            mean_P=("P", "mean"),
        ).reset_index()

        # counts by state (0..4) -> wide
        state_counts = (
            df_tmp.groupby(["bin", "state"])
            .size()
            .unstack(fill_value=0)  # columns become state codes present
        )

        # ensure columns for the 3 we plot exist even if absent
        for code in (2, 3, 4):
            if code not in state_counts.columns:
                state_counts[code] = 0
        state_counts = state_counts[[2, 3, 4]]  # order

        state_counts = state_counts.rename(columns={
            2: "count_short",
            3: "count_long_time",
            4: "count_long_distance",
        })

        # join counts
        tb = tb.merge(state_counts, left_on="bin", right_index=True, how="left")

        # percentages
        tb["pct_short"] = 100.0 * tb["count_short"] / tb["count"].clip(lower=1)
        tb["pct_long_time"] = 100.0 * tb["count_long_time"] / tb["count"].clip(lower=1)
        tb["pct_long_distance"] = 100.0 * tb["count_long_distance"] / tb["count"].clip(lower=1)
        tb["count_long"] = tb["count_long_time"] + tb["count_long_distance"]
        tb["pct_long"] = 100.0 * tb["count_long"] / tb["count"].clip(lower=1)

        # frac_lowP (robust to all-NaN bins)
        def _frac_lowP_bin(s):
            arr = s.to_numpy()
            m = np.isfinite(arr)
            return float(np.mean(arr[m] < lowP_thresh)) if np.any(m) else np.nan

        tb["frac_lowP"] = g["P"].apply(_frac_lowP_bin).to_numpy()

        # time bounds
        tb["bin_start_s"] = tb["bin"] * bin_width_s
        tb["bin_end_s"] = (tb["bin"] + 1) * bin_width_s

        # annotate and collect
        tb.insert(0, "source", _source_of(key))
        tb.insert(1, "stream", key)
        tb.insert(2, "is_primary", bool(key == primary_stream))
        timebin_rows.append(tb)

    # ------------------ save ------------------
    df_summary  = pd.DataFrame(summary_rows)
    df_gaps     = pd.DataFrame(runs_rows)
    df_timebin  = pd.concat(timebin_rows, ignore_index=True) if timebin_rows else pd.DataFrame(
        columns=["source","stream","is_primary","bin","count","count_short","pct_short","count_long_time",
                 "pct_long_time","count_long_distance","pct_long_distance","count_long","pct_long",
                 "mean_speed_manual","mean_speed_provided","mean_P","frac_lowP","bin_start_s","bin_end_s"]
    )

    df_fps = pd.DataFrame([fps_stats])

    path_summary = os.path.join(out_dir, f"{save_name_prefix}_summary.csv")
    path_gaps    = os.path.join(out_dir, f"{save_name_prefix}_gap_runs.csv")
    path_bins    = os.path.join(out_dir, f"{save_name_prefix}_per_timebin.csv")
    path_fps     = os.path.join(out_dir, f"{save_name_prefix}_fps_stats.csv")

    df_summary.to_csv(path_summary, index=False)
    df_gaps.to_csv(path_gaps, index=False)
    df_timebin.to_csv(path_bins, index=False)
    df_fps.to_csv(path_fps, index=False)

    return {
        "summary_csv": path_summary,
        "gap_runs_csv": path_gaps,
        "per_timebin_csv": path_bins,
        "fps_stats_csv": path_fps,
        "out_dir": out_dir,
    }




def compute_mean_firing_rate(
    df,
    cell_col,
    room=None,
    data_kind="counts",   # "counts" or "rate"
):
    """
    Compute mean firing rate (Hz) from df, optionally within a room.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `cell_col` and optionally timing columns.
    cell_col : str
        Column with spike counts per sample ("counts") or instantaneous rate in Hz ("rate").
    room : int or list of int, optional
        If provided, filter df by df["room"] in room.
    data_kind : {"counts","rate"}, default="counts"
        - "counts": values are spike counts per sample; mean = total_spikes / total_time.
        - "rate"  : values are instantaneous Hz; computes time-weighted mean if timebase known.

    Returns
    -------
    float
        Mean firing rate in Hz (NaN if df empty after filtering).

    """
    if room is not None:
        if not isinstance(room, (list, tuple, set)):
            room = [room]
        df = df[df.get("room").isin(room)] if "room" in df.columns else df
    if cell_col not in df.columns:
        raise ValueError(f"'{cell_col}' column not found in df.")

    if df.empty:
        return float("nan")

    x = pd.to_numeric(df[cell_col], errors="coerce").to_numpy()

    # --- resolve total time (seconds) ---
    total_time = None
    ts = pd.to_numeric(df['timestamp'], errors="coerce").to_numpy()
    ts = ts[np.isfinite(ts)]
    if ts.size >= 2:
        total_time = float(ts[-1] - ts[0])

    if total_time is None or total_time <= 0:
        raise ValueError(
            "Cannot determine total time. Ensure 'timestamp' column exists and has at least two valid entries."
        )

    # --- compute mean rate ---
    if data_kind == "counts":
        total_spikes = float(np.nansum(x))
        rate = total_spikes / total_time
    elif data_kind == "rate":
        rate = float(np.nanmean(x))
    else:
        raise ValueError("data_kind must be 'counts' or 'rate'.")

    return rate


import numpy as np
import pandas as pd

def compute_max_firing_rate_by_window(
    df: pd.DataFrame,
    cell_col: str,
    w: float,                               # window size in seconds (>0)
    room=None,
    data_kind: str = "counts",              # "counts" or "rate"
):
    """
    Add int(timestamp / w) as `window_col`, compute mean firing rate per window
    by reusing `compute_mean_firing_rate`, then return the max.

    Returns
    -------
    rates_by_window : pd.Series
        Index = window id (int), values = mean firing rate (Hz) for that window.
    max_rate : float
        Maximum firing rate across windows (Hz).
    """
    if w is None or w <= 0:
        raise ValueError("w must be a positive number of seconds.")
    if cell_col not in df.columns:
        raise ValueError(f"'{cell_col}' column not found in df.")

    # Work on a copy so we don't mutate the caller's df
    d = df.copy()

    # Optional room filtering (do it once, then pass room=None to the inner call)
    if room is not None and "room" in d.columns:
        if not isinstance(room, (list, tuple, set)):
            room = [room]
        d = d[d["room"].isin(room)]

    # Clean timestamps and compute window ids
    ts = pd.to_numeric(d['timestamp'], errors="coerce")
    d = d.loc[ts.notna()].copy()
    if d.empty:
        return pd.Series(dtype=float), float("nan")

    # int(timestamp / w)
    d['win'] = np.floor(d['timestamp'].astype(float) / float(w)).astype("int64")

    # Group and compute mean firing rate per window using your function
    def _rate_for_group(g: pd.DataFrame) -> float:
        # Need at least 2 timestamps so compute_mean_firing_rate can infer duration
        if g['timestamp'].nunique() < 2:
            return np.nan
        try:
            return compute_mean_firing_rate(
                df=g,
                cell_col=cell_col,
                room=None,                  # already filtered above
                data_kind=data_kind,
            )
        except Exception:
            return np.nan


    rates_by_window = d.groupby('win', sort=True, as_index=True, group_keys=False).apply(_rate_for_group, include_groups=False)
    max_rate = float(np.nanmax(rates_by_window.values)) if len(rates_by_window) else float("nan")
    return max_rate



