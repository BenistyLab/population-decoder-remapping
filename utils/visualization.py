import math
import json
import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.cm import ScalarMappable
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from utils.helpers import get_full_path_to_save_file, get_prediction_columns, apply_scaler_transform
from utils.config import format_params_string, assign_room_column, load_position_scaler_from_config, get_rooms_from_config
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score
import os
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
from itertools import combinations
import functools

from .analysis import (
    create_mse_heatmap,
    create_rate_map,
    calculate_rate_map_stats,
    get_threshold_value,
    _wrap_polar_rate_map,
    get_boundary_points_from_csv,
    create_hd_rate_map,
    calculate_rayleigh_vector,
)
from .logger import get_logger
from matplotlib.ticker import PercentFormatter
import matplotlib.cm as cm
import plotly.graph_objects as go
from matplotlib.collections import LineCollection
from shapely.geometry import Point, Polygon

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from numpy.linalg import eigh

# from adjustText import adjust_text
from collections import defaultdict, Counter
from operator import itemgetter

from .metrics import calculate_metrics, load_position_scaler_from_config
from matplotlib_venn import venn2
from adjustText import adjust_text
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure as MPLFigure
import matplotlib.colors as mcolors



try:
    from upsetplot import UpSet, from_indicators, from_memberships
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "upsetplot"])
    from upsetplot import UpSet, from_indicators, from_memberships



# Initialize logger
logger = get_logger(__name__)


#%%  MODEL

def save_plot(func):
    @functools.wraps(func)
    def wrapper_save_plot(*args, **kwargs):

        save_params = kwargs.get('save_params', {})  # Get 'config' from kwargs and set default value
        config      = save_params.get('config', {})  # Get 'config' from kwargs and set default value
        export_path = save_params.get('path', '')  # Get export_path from kwargs
        dpi         = save_params.get("dpi", 160)
        bbox_inches = save_params.get("bbox_inches", "tight")
        pad_inches  = save_params.get("pad_inches", 0.1)
        show        = save_params.get('showFig', False)  # Get show from kwargs
        close       = save_params.get('closeFig', True)  # Get close from kwargs
        icon        = save_params.get('icon', False)  # Get icon from kwargs

        if export_path == '' and not config:
            show = True  # If no export_path and no config, show the plot
            close = False  # Do not close the plot after showing

        ax_in = kwargs.get('ax_in', None)
        if ax_in is not None:
            close = False
            show = False


        fig = func(*args, **kwargs)  # Call the original function

        if fig is None:
            return None, None

        if export_path:
            if not config:
                # If config is empty and export_path is provided, treat export_path as full path
                full_path = export_path
                dir_path = os.path.dirname(export_path)
                os.makedirs(dir_path, exist_ok=True)
            else:
                full_path = get_full_path_to_save_file(config, export_path)

            # Save the figure
            file_ext = os.path.splitext(full_path)[-1].lower()[1:]

            # ---- Detect Plotly ----
            if isinstance(fig, go.Figure):
                if file_ext == 'html':
                    fig.write_html(full_path, auto_play=False, include_plotlyjs=True)
                else:
                    try:
                        fig.write_image(full_path)  # requires kaleido
                    except (RuntimeError, Exception) as e:
                        # Re-raise the error so caller can handle it (e.g., log warning and continue)
                        # Common errors: ChromeNotFoundError, kaleido not available
                            raise
            # ---- Otherwise assume Matplotlib ----
            elif isinstance(fig, MPLFigure):
                if icon:
                    # Use transparent setting from save_params if provided, otherwise default to True for icons
                    icon_transparent = save_params.get("transparent", True)
                    if icon_transparent:
                        fig.patch.set_alpha(0)
                    # Use pad_inches from save_params if provided, otherwise default to 0 for icons
                    icon_pad_inches = save_params.get("pad_inches", 0)
                    icon_dpi = save_params.get("dpi", 10)
                    fig.savefig(
                        full_path,
                        dpi=icon_dpi,
                        bbox_inches='tight',
                        pad_inches=icon_pad_inches,
                        transparent=icon_transparent
                    )
                else:
                    # fig.savefig(full_path)
                    fig.savefig(full_path, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)

            else:
                raise TypeError(f"Unsupported figure type: {type(fig)}")

            logger.info(f"Figure saved at: {full_path}")
        else:
            full_path=None

        # Show the plot
        if show: plt.show()

        # Close the figure to free memory
        if close and isinstance(fig, plt.Figure):
            plt.close(fig)

        return fig, full_path

    return wrapper_save_plot

@save_plot
def plot_distances(real_distances, predicted_distances, positions, V, angles, save_params={}):
    """
    Plots real and predicted distances for each timestamp on a single 2D graph.

    Parameters:
        real_distances (numpy.ndarray): Numpy array of shape (timestamps, angles), representing the real distances.
        predicted_distances (numpy.ndarray): Numpy array of shape (timestamps, angles), representing the predicted distances.
        positions (numpy.ndarray): Numpy array of shape (timestamps, 2) containing (x, y) positions.
        V (numpy.ndarray): velocity.
        angles (numpy.ndarray): Numpy array of angles (in degrees).

        save_params (dict): Dictionary containing parameters for saving the plot.

    Returns:
        fig, full_path: The figure and the path where it was saved (if applicable).
    """
    num_timestamps = real_distances.shape[0]
    num_angles = real_distances.shape[1]

    fig, ax = plt.subplots(figsize=(6, 6), sharex=True, sharey=True)

    # Prepare data for plotting
    all_real_x, all_real_y = [], []
    all_predicted_x_static, all_predicted_y_static = [], []
    all_predicted_x_active, all_predicted_y_active = [], []
    central_positions_x, central_positions_y = [], []

    for i in range(num_timestamps):
        x, y = positions[i]
        v = V[i]

        real_x = x + real_distances[i] * np.cos(np.deg2rad(angles))
        real_y = y + real_distances[i] * np.sin(np.deg2rad(angles))
        all_real_x.extend(real_x)
        all_real_y.extend(real_y)

        predicted_x = x + predicted_distances[i] * np.cos(np.deg2rad(angles))
        predicted_y = y + predicted_distances[i] * np.sin(np.deg2rad(angles))

        # Create a mask to filter out elements where real_distances[j][i] < 30
        #mask = real_distances[i] < 30

        if v < 2:
            all_predicted_x_static.extend(predicted_x)
            all_predicted_y_static.extend(predicted_y)
        else:
            all_predicted_x_active.extend(predicted_x)
            all_predicted_y_active.extend(predicted_y)

        central_positions_x.append(x)
        central_positions_y.append(y)

    # Plot predicted distances
    # print(active_indices)
    ax.plot([0], [0], 'o', label='Predicted Distances (static)', color='red', alpha=0.5, markersize=2)
    ax.plot([0], [0], 'o', label='Predicted Distances (active)', color='orange', alpha=0.5, markersize=2)

    ax.plot(all_predicted_x_static, all_predicted_y_static, 'o', color='red', alpha=0.002, markersize=2)
    ax.plot(all_predicted_x_active, all_predicted_y_active, 'o', color='orange', alpha=0.002, markersize=2)

    # Plot real distances
    ax.plot(all_real_x, all_real_y, 'o', label='Real Distances', color='blue', alpha=0.5, markersize=1)
    ax.scatter(central_positions_x, central_positions_y, color='black', label='Central Positions', alpha=0.2,
               s=1)  # Central positions

    # Set titles and labels
    ax.set_title('Real and Predicted Distances')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid()
    ax.set_aspect('equal')  # Ensure the aspect ratio is 1:1
    ax.set_ylim([-50, 200])
    ax.set_xlim([-50, 200])
    # plt.show()

    return plt.gcf()


def calc_points_from_distances(distances, positions, angles):
    """
    Plots real and predicted distances for each timestamp on a single 2D graph.

    :param distances: Numpy array of shape (timestamps, angles)
    :param positions: Numpy array of shape (timestamps, 2) containing (x, y) positions
    :param angles: Numpy array of angles (in degrees)
    """
    num_timestamps = distances.shape[0]
    num_angles = distances.shape[1]

    # Prepare data for plotting
    all_x, all_y = [], []

    for i in range(num_timestamps):
        x, y = positions[i]

        real_x = x + distances[i] * np.cos(np.deg2rad(angles))
        real_y = y + distances[i] * np.sin(np.deg2rad(angles))
        all_x.extend(real_x)
        all_y.extend(real_y)

    points = np.array([all_x, all_y]).T
    return points

@save_plot
def plot_trajectory(positions, rooms_to_indices=None, map_rooms=None, rooms=None, folds=None, scaler=None, occupancy=None, pos_range=None, boundary_points=None, bins=None, only_in_boundary_points=False, scatter_size=1,boundary_size=3, title='Trajectory and Boundary', minimized_layout=False, show_out_of_rooms=False, ax_in=None, unit_name=None, reverse_y=True, transparent_background=None, save_params={}): #,positions_B=None
    """
    Plots the trajectory of positions with optional occupancy heatmap and boundary points.

    Parameters:
        positions (numpy.ndarray): Numpy array of shape (timestamps, 2) containing (x, y) positions.
        rooms_to_indices (dict, optional): Dictionary mapping room indices or names to lists of positions.
        map_rooms (dict, optional): Dictionary mapping room indices or names to lists of positions.
        rooms (list, optional): List of room names to plot. If None, all rooms will be plotted.
        folds (numpy.ndarray, optional): Numpy array of fold indices for each position.
        scaler (object, optional): Scaler object with an inverse_transform method for rescaling (x, y) positions.
        occupancy (numpy.ndarray): Occupancy grid for the heatmap.
        boundary_points (numpy.ndarray): Numpy array of boundary points.
        bins (tuple): Bins for the occupancy grid.
        only_in_boundary_points (bool): If True, only plot points within the boundary points.
        pos_range (tuple): Range for the x and y axes.
        scatter_size (int): Size of the scatter points.
        boundary_size (int): Size of the boundary points.
        title (str): Title of the plot.
        ax_in (matplotlib.axes.Axes, optional): Axis to plot on (if None, create a new figure).
        transparent_background (bool, optional): Whether to use transparent background. If None, uses value from save_params['transparent'] or defaults based on icon setting.
        save_params (dict): Dictionary containing parameters for saving the plot. Can include 'transparent' (bool) to control background transparency.

    Returns:
        ax or fig: The axis if an existing axis was used, or the figure if a new one was created.
    """
    figsize = save_params.get('figsize', (6,6))
    transparent_background = save_params.get('transparent', None)    
    scatter_rasterized = bool(save_params.get('scatter_rasterized', False))
    # Create a new figure if no axis is provided
    if not ax_in:
        fig, ax = plt.subplots(figsize=figsize, sharex=True, sharey=True)
        if transparent_background:
            fig.patch.set_alpha(0) # Set figure background to transparent
            ax.patch.set_alpha(0)  # Set axis background to transparent
        else:
            fig.patch.set_facecolor('white')  # Set figure background
            ax.set_facecolor('white')  # Set axis background
        # set transparent background
        if minimized_layout:
            for spine in ax.spines.values():
                spine.set_linewidth(0)  # or: spine.set_alpha(0)
    else:
        ax = ax_in

    positions = positions.copy()  # Create a copy of positions to avoid modifying the original data
    boundary_points = boundary_points.copy() if boundary_points is not None else None

    # Inverse transform the positions using the scaler if provided
    if scaler is not None:
        positions = apply_scaler_transform(positions, scaler, reverse=True)
        if boundary_points is not None:
            boundary_points[:,:2] = apply_scaler_transform(boundary_points[:,:2], scaler, reverse=True)
        if pos_range is not None:
            pos_range = tuple(apply_scaler_transform(value, scaler, reverse=True) for value in pos_range)

    if occupancy is not None:
        x_bins, y_bins = bins

        # Calculate the midpoints for x and y grid (to represent bin centers)
        x_mid = (x_bins[:-1] + x_bins[1:]) / 2  # Midpoints of x bins
        y_mid = (y_bins[:-1] + y_bins[1:]) / 2  # Midpoints of y bins
        # Create the grid from midpoints for heatmap plotting
        x_grid, y_grid = np.meshgrid(x_mid, y_mid)
        # Plot the heatmap
        cax = ax.contourf(x_grid, y_grid, occupancy.T, alpha=0.75, levels=25, cmap="Reds")
        # Create a divider for the existing axis (ax_in)
        divider = make_axes_locatable(ax)
        # Append a new axis for the colorbar on the right with specific size and padding
        cbar_ax = divider.append_axes("right", size="4%", pad=0.05)

        cb = plt.colorbar(cax, cax=cbar_ax)
        cbar_ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2))

    if show_out_of_rooms:
        ignore_indices = []
        for i, (room_name, room_indices) in enumerate(rooms_to_indices['name'].items()):
            ignore_indices.extend(room_indices)
        ignore_indices = np.array(ignore_indices)
        # Filter positions to only include those not in any room
        filter_positions = positions[~np.isin(np.arange(len(positions)), ignore_indices)]
        # Plot all positions in a light gray color
        ax.scatter(
            filter_positions[:, 0],
            filter_positions[:, 1],
            color='black',
            alpha=1.0,
            s=scatter_size,
            label='Out of Rooms',
            rasterized=scatter_rasterized,
        )
    # Plot each room's trajectory in a different pastel color if rooms_to_indices is provided
    elif rooms_to_indices is not None:
        color_palette = cm.get_cmap("tab20c", len(rooms_to_indices['name'])+2)  # Generate a color palette
        for i, (room_name, room_indices) in enumerate(rooms_to_indices['name'].items()):
            if rooms is not None and room_name not in rooms:
                continue
            if len(room_indices) == 0:
                continue
            room_positions = positions[room_indices]  # Use room index to slice positions
            # Filter points to only include those within the boundary points
            if only_in_boundary_points and boundary_points is not None and (map_rooms is not None and map_rooms):
                room_index = map_rooms['rooms'][room_name]['index']
                # Create the polygon of the rooms
                polygon_room = Polygon(boundary_points[boundary_points[:, 2] == room_index])
                # Create a mask for points within the polygon
                filter_points_indices = [polygon_room.covers(Point(xy)) for xy in room_positions]
                # filter points
                room_positions = room_positions[filter_points_indices]
            # Plot the room positions with proper label
            ax.scatter(
                room_positions[:, 0],
                room_positions[:, 1],
                color=color_palette(i),
                alpha=0.5,
                s=scatter_size,
                label=str(room_name),
                rasterized=scatter_rasterized,
            )
    elif folds is not None:
        unique_folds = np.unique(folds)
        if len(unique_folds) == 1:
            # Create a color gradient for points
            color_values = np.linspace(0, 1, len(positions))  # Values for the gradient
            # chose gradient colormap
            cmap = cm.get_cmap("viridis")
            scatter = ax.scatter(
                positions[:, 0],
                positions[:, 1],
                c=color_values,
                cmap=cmap,
                s=10,
                alpha=0.8,
                rasterized=scatter_rasterized,
            )
            # cb = plt.colorbar(scatter, ax=ax, orientation='vertical', pad=0.05)
            #cb.set_label("Position Progression")
        else:
            color_palette = cm.get_cmap("tab20c", len(unique_folds)+2)  # Generate a color palette
            for i, fold in enumerate(unique_folds):
                fold_positions = positions[folds==fold]
                label = f"Fold {fold}" if not minimized_layout else None
                ax.scatter(
                    fold_positions[:, 0],
                    fold_positions[:, 1],
                    color=color_palette(i),
                    alpha=0.5,
                    s=scatter_size,
                    label=label,
                    rasterized=scatter_rasterized,
                )
    else:
        # Plot the entire trajectory in blue if no rooms_to_indices is provided
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            color='blue',
            alpha=0.2,
            s=scatter_size,
            rasterized=scatter_rasterized,
        )

    # Plot boundary points
    if boundary_points is not None:
        plot_boundaries(boundary_points,ax, linewidth=boundary_size)

    # Set titles and labels
    ax.set_title(title)
    if not ax_in:
        xlabel = 'X'
        ylabel = 'Y'
        if unit_name:
            xlabel = f'X ({unit_name})'
            ylabel = f'Y ({unit_name})'
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
    if pos_range is not None:
        ax.set_xlim(pos_range[0], pos_range[1])
        if reverse_y:
            ax.set_ylim(pos_range[1], pos_range[0])  # Reversed Y-axis
        else:
            ax.set_ylim(pos_range[0], pos_range[1])  # Normal Y-axis
    ax.grid()
    ax.set_aspect('equal')  # Ensure the aspect ratio is 1:1

    if minimized_layout:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        ax.legend().set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    # Return the figure if a new axis was created, otherwise return the existing axis
    if not ax_in:
        return plt.gcf()
    return ax


# DEPRECATED: This function is no longer used. Use plot_trajectory instead.
"""
@save_plot
def plot_trajectory_v2(config, df, rooms=None, position_columns=['X','Y'], scaler=None, pos_range=None, boundary_points=None, only_in_boundary_points=False, color=None, scatter_size=1,boundary_size=10, title='Trajectory and Boundary', minimized_layout=False, temporal_fade=False, alpha=0.5, ax_in=None, save_params={}):
    \"\"\"
    Plots the trajectory of positions with optional occupancy heatmap and boundary points.

    Parameters:
        positions (numpy.ndarray): Numpy array of shape (timestamps, 2) containing (x, y) positions.
        map_rooms (dict, optional): Dictionary mapping room indices or names to lists of positions.
        folds (numpy.ndarray, optional): Numpy array of fold indices for each position.
        scaler (object, optional): Scaler object with an inverse_transform method for rescaling (x, y) positions.
        occupancy (numpy.ndarray): Occupancy grid for the heatmap.
        boundary_points (numpy.ndarray): Numpy array of boundary points.
        bins (tuple): Bins for the occupancy grid.
        only_in_boundary_points (bool): If True, only plot points within the boundary points.
        pos_range (tuple): Range for the x and y axes.
        color (str): Color for the trajectory points.
        scatter_size (int): Size of the scatter points.
        boundary_size (int): Size of the boundary points.
        title (str): Title of the plot.
        ax_in (matplotlib.axes.Axes, optional): Axis to plot on (if None, create a new figure).
        save_params (dict): Dictionary containing parameters for saving the plot.

    Returns:
        ax or fig: The axis if an existing axis was used, or the figure if a new one was created.
    \"\"\"

    # Create a new figure if no axis is provided
    if not ax_in:
        fig, ax = plt.subplots(figsize=(6, 6), sharex=True, sharey=True)
        fig.patch.set_facecolor('white')  # Set figure background
    else:
        ax = ax_in
    ax.set_facecolor('white')  # Set axis background
    df = df.copy()  # Create a copy of df to avoid modifying the original data
    boundary_points = boundary_points.copy() if boundary_points is not None else None
    map_rooms = config.get('preprocessing', {}).get('map_rooms', {})

    # Inverse transform the positions using the scaler if provided
    if scaler is not None:
        df[position_columns] = apply_scaler_transform(df[position_columns].values, scaler, reverse=True)
        if boundary_points is not None:
            boundary_points[:,:2] = apply_scaler_transform(boundary_points[:,:2], scaler, reverse=True)
        if pos_range is not None:
            pos_range = tuple(apply_scaler_transform(value, scaler, reverse=True) for value in pos_range)

    all_rooms = config['preprocessing']['room_indices'].keys()
    n_rooms = len(all_rooms)
    if rooms is None:
        rooms = all_rooms # Use all rooms if none are specified
    elif isinstance(rooms, str):
        rooms = [rooms]  # Convert to list if a single room name is provided

    color_palette = cm.get_cmap("tab20c",n_rooms+2)  # Generate a color palette

    # Plot each room's trajectory in a different pastel color if map_rooms is provided
    if 'room' in df.columns:
        for i, room in enumerate(all_rooms):
            if room not in rooms:
                continue

            room_color = color_palette(i) if color is None else color  # Use the provided color or default to the palette
            room_positions = df.loc[df['room']==room,position_columns].values
            # Filter points to only include those within the boundary points
            if only_in_boundary_points and boundary_points is not None:
                room_index = map_rooms['rooms'][room]['index']
                # Create the polygon of the rooms
                polygon_room = Polygon(boundary_points[boundary_points[:, 2] == room_index])
                # Create a mask for points within the polygon
                valid_indices = [polygon_room.covers(Point(xy)) for xy in room_positions]
                # filter points
                room_positions = room_positions[valid_indices]
            # Plot the room positions
            # Compute alpha gradient if animation effect is enabled
            if temporal_fade:
                n_points = len(room_positions)
                if n_points == 0:
                    continue
                alphas = np.linspace(0.05, 1.0, n_points)  # alpha from low to high
                alphas = np.exp(-np.linspace(3, 0, n_points))  # Decay from ~0.05 to 1
                for j, (x, y) in enumerate(room_positions):
                    ax.scatter(x, y, color=room_color, alpha=alphas[j], s=scatter_size)
            else:
                ax.scatter(room_positions[:, 0], room_positions[:, 1], color=room_color, alpha=alpha, s=scatter_size)
            if not minimized_layout:
                ax.scatter(0, 0, color=room_color, s=15, label=f"Trajectory {room}") # Plot labeled point at (0, 0) for each room color
    else:
        # Plot the entire trajectory in blue if no map_rooms is provided
        positions = df[position_columns].values
        ax.scatter(positions[:, 0], positions[:, 1], color=color if color is not None else color_palette(0), alpha=alpha, s=scatter_size)

    # Plot boundary points
    if boundary_points is not None:
        plot_boundaries(boundary_points,ax, linewidth=boundary_size)

    # Set titles and labels
    ax.set_title(title)
    if not ax_in:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
    if pos_range is not None:
        low, high = pos_range
        pad = (high - low) * 0.05
        ax.set_xlim(low - pad, high + pad)
        ax.set_ylim(high + pad, low - pad)  # Reversed Y-axis
    ax.grid()
    ax.set_aspect('equal')  # Ensure the aspect ratio is 1:1

    if minimized_layout:
        ax.set_xlabel('')
        ax.set_ylabel('')
        # ax.set_title('')
        ax.legend().set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid('off')

    # Return the figure if a new axis was created, otherwise return the existing axis
    if not ax_in:
        return plt.gcf()
    return ax
"""

@save_plot
def plot_distances_animated(real_distances, predicted_distances, positions, angles, save_params={}):
    """
    Creates an animated plot of real and predicted distances for each timestamp and optionally saves it.

    Parameters:
        real_distances (numpy.ndarray): Numpy array of shape (timestamps, angles) representing the real distances.
        predicted_distances (numpy.ndarray): Numpy array of shape (timestamps, angles) representing the predicted distances.
        positions (numpy.ndarray): Numpy array of shape (timestamps, 2) containing (x, y) positions.
        angles (numpy.ndarray): Numpy array of angles (in degrees).

        save_params (dict): Dictionary containing parameters for saving the animation.

    Returns:
        fig: The figure containing the animated plot.
    """
    num_timestamps = real_distances.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True)

    # Initialize plots for real and predicted distances
    real_scatter, = axes[0].plot([], [], 'o', label='Real Distances', color='blue', alpha=0.6)
    predicted_scatter, = axes[1].plot([], [], 'o', label='Predicted Distances', color='red', alpha=0.6)
    central_scatter_real = axes[0].scatter([], [], color='black', s=50, zorder=5)
    central_scatter_predicted = axes[1].scatter([], [], color='black', s=50, zorder=5)

    def init():
        axes[0].set_title('Real Distances')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        axes[0].legend()
        axes[0].grid()

        axes[1].set_title('Predicted Distances')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        axes[1].legend()
        axes[1].grid()

        return real_scatter, predicted_scatter, central_scatter_real, central_scatter_predicted

    def update(i):
        x, y = positions[i]

        real_x = x + real_distances[i] * np.cos(np.deg2rad(angles))
        real_y = y + real_distances[i] * np.sin(np.deg2rad(angles))

        predicted_x = x + predicted_distances[i] * np.cos(np.deg2rad(angles))
        predicted_y = y + predicted_distances[i] * np.sin(np.deg2rad(angles))

        real_scatter.set_data(real_x, real_y)
        predicted_scatter.set_data(predicted_x, predicted_y)
        central_scatter_real.set_offsets(np.c_[x, y])
        central_scatter_predicted.set_offsets(np.c_[x, y])
        return real_scatter, predicted_scatter, central_scatter_real, central_scatter_predicted

    ani = FuncAnimation(fig, update, frames=num_timestamps, init_func=init, blit=True, repeat=True)

    # if save_path:
    #     writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
    #     ani.save(save_path, writer=writer)
    # else:
    #     plt.show()

    return plt.gcf()



@save_plot
def plot_polar_ev(df_stats, xticks, title, save_params={}):
    """
    Generate a polar plot of the 'ev' column with error bars.

    Parameters:
        df_stats (DataFrame): DataFrame containing evaluation metrics.
        xticks (list): List of x-axis tick labels.
        title (str): Title for the plot.
        save_params (dict): Dictionary containing parameters for saving the plot.

    Returns:
        fig: The figure containing the polar plot.
    """

    # Filter DataFrame for 'test' set and target columns in columns
    # df_filtered = df_stats[(df_stats['set'] == 'test') & df_stats['target'].isin(columns)]

    # Calculate mean and standard deviation of 'ev' for each target variable
    mean_ev = df_stats.groupby(['perspective','angle'])['ev_uni'].mean()
    std_ev = df_stats.groupby(['perspective','angle'])['ev_uni'].std() / 2

    # Plot polar projection of mean EV for Egocentrism
    angles = np.deg2rad(mean_ev.reset_index()['angle'].values)#np.linspace(0, 2 * np.pi, len(columns), endpoint=False)
    old_xticks = np.linspace(0, 2 * np.pi, len(xticks), endpoint=False)

    mean_ev = mean_ev.values
    std_ev = std_ev.values

    # Generate plot
    plt.figure(figsize=(4, 4))
    ax = plt.subplot(111, polar=True)

    # Plot mean values with error bars
    ax.errorbar(angles, mean_ev, yerr=std_ev, fmt='o', markersize=4, capsize=0.5, capthick=0, alpha=1)

    # Set labels and title
    ax.fill(angles, mean_ev, alpha=0.25)
    # ax.fill_between(angles, mean_ev - std_ev, mean_ev + std_ev, alpha=0.25)

    ax.set_xticks(old_xticks)
    ax.set_xticklabels(xticks, fontsize=8)
    # ax.set_ylabel('Explained Variance')
    ax.set_title(f'Explained Variance - {title}')
    ax.set_ylim(0, 1)  # Adjust y-axis limits if necessary

    plt.tight_layout()

    return plt.gcf()

@save_plot
def plot_k_fold_splits(splits, scenario='train_val_test', save_params={}):
    """
    Plot the indices of each fold and sub-fold.

    Parameters:
        splits (list): List of splits from the k_fold_split function.
        scenario (str): Scenario used for splitting ('train_val_test', 'train_val', 'train_test').
        save_params (dict): Dictionary containing parameters for saving the plot.

    Returns:
        fig: The figure containing the k-fold split plot.
    """
    plot_data = []
    for fold, sub_folds in enumerate(splits):
        for sub_fold, (train_indices, val_indices, test_indices) in enumerate(sub_folds):
            if scenario=='train_test':
                fold_name = f"{fold + 1}"
            else:
                fold_name = f"{fold + 1}.{sub_fold + 1}"
            for idx in train_indices:
                plot_data.append((idx, fold_name, 'Train'))
            if val_indices is not None:
                for idx in val_indices:
                    plot_data.append((idx, fold_name, 'Validation'))
            if test_indices is not None:
                for idx in test_indices:
                    plot_data.append((idx, fold_name, 'Test'))

    plot_df = pd.DataFrame(plot_data, columns=['Indices', 'Fold_SubFold', 'Set'])
    fig = plt.figure(figsize=(10, 6))
    sns.scatterplot(data=plot_df, x='Indices', y='Fold_SubFold', hue='Set',
                    palette={'Train': 'blue', 'Validation': 'orange', 'Test': 'green'}, marker="|", s=100,
                    linewidths=1)
    plt.xlabel('Indices')
    plt.ylabel('Fold')
    plt.title(f'Indices of Each Fold ({scenario})')
    plt.legend(title='Set')
    plt.grid(True)
    plt.tight_layout()

    return plt.gcf()

@save_plot
def plot_folds_mse_ev(df, ncols=1, save_params={}):
    """
    Visualize the Mean Squared Error (MSE) and Explained Variance (EV) for different folds
    with error bars versus epoch number.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data to plot, which should include
                           columns for 'fold', 'epoch', 'ev', 'mse', and 'params'.
        ncols (int): Number of columns to use for subplots.
        save_params (dict): Dictionary containing parameters for saving the plot.

    Returns:
        fig: The figure containing the MSE and EV plots across different folds.
    """
    title = 'Mean Squared Error and Explained Variance Across Folds'
    gb_fold = df.groupby('fold')
    nrows = -(-len(gb_fold) // ncols)  # Calculate the number of rows needed
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols + 4, 2 * nrows))#, sharex=True, sharey=True)
    axes = axes.flatten()  # Flatten axes array to easily iterate

    colors = sns.color_palette("tab20", n_colors=len(df['params'].unique()))
    param_color_map = {param: colors[i % len(colors)] for i, param in enumerate(df['params'].unique())}

    for i, (fold, data) in enumerate(gb_fold):
        ax1 = axes[i]
        ax2 = ax1.twinx()

        all_handles = []
        all_labels = []

        for j, (params, pdata) in enumerate(data.groupby('params')):
            param_name = f'Param {j}'
            p_mean = pdata.groupby('epoch')[['ev_uni', 'mse_uni']].mean().reset_index()
            p_std = pdata.groupby('epoch')[['ev_uni', 'mse_uni']].std().reset_index()
            color = param_color_map[params]

            #mse_line, = ax1.plot(p_mean['epoch'].values, p_mean['mse'].values, label=f'{param_name}', linestyle='--', color=color)
            #ax1.fill_between(p_mean['epoch'].values, p_mean['mse'].values - p_std['mse'].values / 2, p_mean['mse'].values + p_std['mse'].values / 2,
            #                 alpha=0.1, color=color)

            ev_line, = ax2.plot(p_mean['epoch'].values, p_mean['ev_uni'].values, label=f'{param_name}', color=color)
            ax2.fill_between(p_mean['epoch'].values, p_mean['ev_uni'].values - p_std['ev_uni'].values / 2, p_mean['ev_uni'].values + p_std['ev_uni'].values / 2,
                             alpha=0.1, color=color)

            all_handles.append(ev_line)
            all_labels.append(params)

        ax1.set_ylabel('MSE')
        ax2.set_ylabel('EV')
        ax1.set_ylim(0, 0.5)
        ax2.set_ylim(0, 1)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}%'.format(y * 100)))
        ax1.set_title(f'Fold {fold}')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])  # Remove unused subplots

    fig.legend(handles=all_handles, labels=all_labels, loc='center left', bbox_to_anchor=(1.0, 0.5), title='Params')
    plt.xlabel('Epoch')
    # Set sparse x-ticks
    # unique_epochs = df['epoch'].sort_values().unique()
    # xticks = unique_epochs[::max(1, len(unique_epochs) // 5)]
    # plt.xticks(xticks, xticks.astype(int), rotation=45)
    plt.xlim([1, max(p_mean['epoch'])])

    # Add the main title
    plt.suptitle(title, x=0.5, y=1.02, ha='center', fontsize=16)

    plt.tight_layout()

    return plt.gcf()


@save_plot
def plot_boundaries_and_positions(real_distances, predicted_distances, positions, V, angles, k=1, ncols=1, save_params={}):
    """
    Generate a figure with subplots showing boundaries and tracking positions for each fold.

    Parameters:
        real_distances (list of numpy arrays): List containing real distances for each fold.
        predicted_distances (list of numpy arrays): List containing predicted distances for each fold.
        positions (list of numpy arrays): List containing (x, y) positions for each timestamp.
        V (numpy array): Array containing velocity values for each fold.
        angles (numpy array): Array of angles (in degrees) for calculating positions.
        k (int): Number of blocks into which indices are divided.
        ncols (int): Number of columns to use for subplots.
        save_params (dict): Dictionary containing parameters for saving the plot.

    Returns:
        fig: Generated figure object containing the plots for each fold.
    """
    num_folds = len(real_distances)
    num_timestamps = real_distances[0].shape[0]

    # Divide indices into k equal blocks
    block_size = num_folds // k
    fold_blocks = [list(range(i * block_size, (i + 1) * block_size)) for i in range(k)]

    nrows = -(-k // ncols)  # Calculate the number of rows needed
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3 * ncols + 3, 3 * nrows), sharex=True, sharey=True)
    axes = axes.flatten()  # Flatten axes array to easily iterate

    for i in range(k):
        ax = axes[i]

        # Prepare data for plotting
        all_real_x, all_real_y = [], []
        all_predicted_x_static, all_predicted_y_static = [], []
        all_predicted_x_active, all_predicted_y_active = [], []
        central_positions_x, central_positions_y = [], []
        fold_real_distances, fold_predicted_distances = [], []

        # Get indices for the current block
        fold_indices = fold_blocks[i]

        for j in fold_indices:
            x, y = positions[j]
            v = V[j]

            real_x = x + real_distances[j] * np.cos(np.deg2rad(angles))
            real_y = y + real_distances[j] * np.sin(np.deg2rad(angles))
            all_real_x.extend(real_x)
            all_real_y.extend(real_y)

            predicted_x = x + predicted_distances[j] * np.cos(np.deg2rad(angles))
            predicted_y = y + predicted_distances[j] * np.sin(np.deg2rad(angles))

            # Create a mask to filter out elements where real_distances[j][i] < 30
            #mask = real_distances[i] < 30

            if v < 2:
                all_predicted_x_static.extend(predicted_x)
                all_predicted_y_static.extend(predicted_y)
            else:
                all_predicted_x_active.extend(predicted_x)
                all_predicted_y_active.extend(predicted_y)

            central_positions_x.append(x)
            central_positions_y.append(y)

            # Collect distances for explained variance score
            fold_real_distances.extend(real_distances[j])
            fold_predicted_distances.extend(predicted_distances[j])

        # Plot predicted distances
        ax.plot([0], [0], 'o', label='Predicted Distances (static)', color='red', alpha=0.5, markersize=2)
        ax.plot([0], [0], 'o', label='Predicted Distances (active)', color='orange', alpha=0.5, markersize=2)
        ax.plot(all_predicted_x_active, all_predicted_y_active, 'o', color='orange', alpha=0.01, markersize=2)
        ax.plot(all_predicted_x_static, all_predicted_y_static, 'o', color='red', alpha=0.01, markersize=2)

        # Plot real distances
        ax.plot(all_real_x, all_real_y, 'o', label='Real Distances', color='blue', alpha=0.8, markersize=1)
        ax.scatter(central_positions_x, central_positions_y, color='black', label='Central Positions', alpha=0.2,
                   s=1)  # Central positions

        # Calculate explained variance score
        ev_score_percent = max(0, explained_variance_score(fold_real_distances, fold_predicted_distances)) * 100

        # Set titles and labels
        ax.set_title(f'Fold {i}', loc='center', pad=23)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        # ax.legend()
        ax.grid()
        ax.set_aspect('equal')  # Ensure the aspect ratio is 1:1
        ax.set_ylim([-50, 200])
        ax.set_xlim([-50, 200])
        # Add explained variance score as subtitle
        ax.text(0.5, 1.05, f'EV: {ev_score_percent:.0f}%', ha='center', transform=ax.transAxes, fontsize=10)

    for j in range(k, len(axes)):
        fig.delaxes(axes[j])  # Remove unused subplots

    # Add the main title
    title = 'Boundaries and Tracking Positions Across Blocks'
    plt.suptitle(title, x=0.5, y=1.02, ha='center', fontsize=16)

    plt.tight_layout()

    return plt.gcf()


@save_plot
def plot_loss_curve(csv_file,ckpt_file, title='Loss Curves', params=None, save_params={}):
    """
    Read the CSV file and plot the loss curves for training and validation losses.

    Parameters:
        csv_file (str): Path to the CSV file containing loss values.
        ckpt_file (str): Path to the checkpoint file to retrieve early stopping epoch.
        title (str): Title of the plot. Default is 'Loss Curves'.
        params (dict, optional): Additional parameters to be appended to the title.
        save_params (dict): Dictionary containing parameters for saving the plot.

    Returns:
        fig: Generated figure object containing the loss curves.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Find the epoch with the minimum validation loss
    if os.path.isfile(ckpt_file):
        state = torch.load(ckpt_file)
        early_stopping_epoch = state['best_epoch']
    else:
        early_stopping_epoch = 0

    # Set up the plot style
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))

    _, idx = np.unique(df['set'].values, return_index=True)
    sets = df['set'].values[np.sort(idx)]


    # Plot validation losses with pastel colors
    colors = sns.color_palette("pastel", n_colors=len(sets))
    for i, set in enumerate(sets):
        if set.startswith('validation'):
            epochs = df[df['set'] == set]['epoch'].values
            loss = df[df['set'] == set]['mse'].values
            if len(sets) < 5:
                label = f'loss {set}'
            else:
                if (i-1)%((len(sets)-1)//6)==0:
                    label = f'loss {set}'
                else:
                    label = ''
            plt.plot(epochs, loss, color=colors[i], label=label, linestyle='--', marker=' ',
                     markersize=6)

    # Plot train losses
    for set in sets:
        if set.startswith('train'):
            epochs = df[df['set'] == set]['epoch'].values
            loss = df[df['set'] == set]['mse'].values
            plt.plot(epochs, loss, color='red', label=f'loss {set}', linestyle='-', marker=' ', markersize=6)

    # Highlight the early stopping epoch
    if early_stopping_epoch>0:
        plt.axvline(x=early_stopping_epoch, color='red', linestyle='--',
                    label=f'Early Stopping at Epoch {early_stopping_epoch}')

    # Set the y-axis to symlog scale with linthresh around 0.002 and y-limits between 0 and 10^3
    plt.yscale('symlog', linthresh=0.002)
    plt.ylim(0, 1)#1e3

    # Determine the x-tick positions and labels to show only 10 x-ticks
    num_ticks = 10
    #tick_labels = np.linspace(1, np.max(df['epoch'].values), num_ticks, dtype=int)
    #tick_positions = tick_labels-1
    # Add labels and title
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    if params is not None:
        title = f"{format_params_string(params=params, format_type='title_list')}\n\n{title}"
    plt.title(title, fontsize=14, pad=(title.count('\n') + 1) * 10)
    #plt.xticks(tick_positions, tick_labels.astype(int))  # Show limited x-ticks with integer values
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show the plot
    plt.tight_layout()
    #plt.show()
    return plt.gcf()


from scipy.signal import convolve2d


# def plot_position_mse(positions, mse, boundary_points=None, show_trajectory=True, base_heatmap=None, n_pixel=34, title='', save_params={},
#                       ax_in=None, mse_range=None):
#     """
#     Plot a heatmap of Mean Squared Errors (MSE) by x, y positions, with optional smoothing and boundary overlays.
#
#     Parameters:
#         df_data (DataFrame): DataFrame containing columns for 'X', 'Y' positions and 'mse' for the mean squared error.
#                              Columns should include 'X', 'Y' for the spatial positions and 'mse' for the corresponding errors.
#         boundary_points (np.ndarray, optional): Array of boundary points to overlay on the heatmap.
#                                                 Shape should be (n, 2), where each row contains the x, y coordinates of a boundary point. Default is None.
#         n_pixel (int, optional): Number of pixels (bins) for the x and y grids of the heatmap. Default is 34.
#         title (str, optional): Title for the plot. If not provided, 'MSE' will be used as a default. Default is ''.
#         export_path (str, optional): Path to export the generated plot as a file. If not provided, the plot is not saved. Default is ''.
#         config (dict, optional): Additional configuration options, reserved for future use. Default is {}.
#         ax_in (Axes, optional): Pre-existing Matplotlib Axes object to plot the heatmap on. If not provided, a new figure and axis will be created. Default is None.
#
#     Returns:
#         None
#     """
#
#     if boundary_points is None:
#         pos_range = (np.min(positions), np.max(positions))
#     else:
#         pos_range = (np.min(boundary_points[:,:2]), np.max(boundary_points[:,:2]))
#
#     # Generate grids for x and y positions
#     x_grid = np.linspace(pos_range[0], pos_range[1], n_pixel)
#     y_grid = np.linspace(pos_range[0], pos_range[1], n_pixel)
#
#     # Define Gaussian kernel for smoothing
#     sigma = 2
#     kernel_size = 8
#     X_kernel, Y_kernel = np.meshgrid(np.arange(-kernel_size / 2, kernel_size / 2 + 1),
#                                      np.arange(-kernel_size / 2, kernel_size / 2 + 1))
#     kernel = np.exp(-(X_kernel ** 2 + Y_kernel ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
#     kernel = kernel / np.sum(kernel)
#
#     # Initialize heatmaps
#     mse_heatmap = np.zeros((len(y_grid), len(x_grid)))
#     normalized_heatmap = np.zeros((len(y_grid), len(x_grid)))
#
#     # Populate heatmaps with MSE values for each grid cell
#     for j in range(len(x_grid) - 1):
#         for k in range(len(y_grid) - 1):
#             # Find instances within the current grid cell
#             instance_in_cell = (positions[:, 0] >= x_grid[j]) & (positions[:, 0] < x_grid[j + 1]) & \
#                                (positions[:, 1] >= y_grid[k]) & (positions[:, 1] < y_grid[k + 1])
#             sum_instances_in_cell = np.sum(instance_in_cell)
#
#             mse_in_cell = mse[instance_in_cell]
#
#             if sum_instances_in_cell > 0:
#                 mse_heatmap[k, j] = np.sum(mse_in_cell)
#                 normalized_heatmap[k, j] = mse_heatmap[k, j] / sum_instances_in_cell
#
#             if base_heatmap is not None:
#                 if base_heatmap[k, j]:
#                     normalized_heatmap[k, j] = normalized_heatmap[k, j] / base_heatmap[k, j]
#                 else:
#                     normalized_heatmap[k, j] = 0
#
#             if normalized_heatmap[k, j] == 0:
#                 normalized_heatmap[k, j] = np.nan
#
#     # Apply convolution to smooth the heatmap
#     smoothed_heatmap = convolve2d(normalized_heatmap, kernel, mode='same')
#
#     # Create a new figure if no axis is provided
#     if not ax_in:
#         fig, ax = plt.subplots(figsize=(4, 4))
#     else:
#         ax = ax_in
#
#     # If MSE range is provided, use it to normalize the color scale
#     vmin, vmax = mse_range if mse_range else (np.nanmin(normalized_heatmap), np.nanmax(normalized_heatmap))
#
#     # Plot the smoothed MSE heatmap
#     cax = ax.contourf(x_grid, y_grid, normalized_heatmap, alpha=0.75, levels=25, cmap='jet', vmin=vmin, vmax=vmax)
#
#     # Overlay boundary points if provided
#     if boundary_points is not None:
#         ax.plot(boundary_points[:, 0], boundary_points[:, 1], 'o', label='Boundaries', color='black', alpha=0.5,
#                 markersize=1)
#     if show_trajectory:
#         ax.scatter(positions[:, 0], positions[:, 1], color='black', label='Central Positions', alpha=0.2,
#                    s=1)  # Central positions
#
#     # Set title and axis properties
#     ax.set_title(title if title else 'MSE\n', fontdict={'fontsize': 14, 'fontweight': 'bold'})
#     ax.grid(True)
#     ax.set_xlim(x_grid.min(), x_grid.max())
#     ax.set_ylim(y_grid.min(), y_grid.max())
#     ax.set_aspect('equal')
#
#     # Display the plot
#     if not ax_in:
#         cb = plt.colorbar(cax)
#         plt.show()
#


@save_plot
def plot_position_mse(heatmap, x_grid=None, y_grid=None, boundary_points=None, positions=None, show_trajectory=False,
                      fill_zero=False, proportion=False, n_pixel=34, title='',
                      ax_in=None, pos_range=None, mse_range=None, show_colormap=True, save_params={}):
    """
    Plot a heatmap of Mean Squared Errors (MSE) by x, y positions with optional smoothing and boundary overlays.

    Parameters:
        heatmap (np.ndarray): Heatmap data to be visualized.
        x_grid (np.ndarray, optional): x-coordinates grid. If None, it will be generated based on `pos_range`. Default is None.
        y_grid (np.ndarray, optional): y-coordinates grid. If None, it will be generated based on `pos_range`. Default is None.
        boundary_points (np.ndarray, optional): Array of boundary points to overlay on the heatmap. Shape should be (n, 2). Default is None.
        show_trajectory (bool, optional): Whether to overlay trajectory points on the plot. Default is False.
        fill_zero (bool, optional): If True, zero values in `heatmap` will be filled with a specified `fill_value`. Default is False.
        proportion (bool, optional): If True, normalize the heatmap and use a diverging colormap or log scale. Default is False.
        n_pixel (int, optional): Number of pixels (bins) for the x and y grids of the heatmap. Default is 34.
        title (str, optional): Title for the plot. Default is ''.
        ax_in (Axes, optional): Matplotlib Axes object to plot on. If None, a new figure and axis will be created. Default is None.
        pos_range (tuple, optional): Range for x and y positions. Used to generate `x_grid` and `y_grid` if not provided. Default is None.
        mse_range (tuple, optional): Range for MSE values to normalize the color scale. Default is None.
        show_colormap (bool, optional): Whether to display the colorbar. Default is True.
        save_params (dict): Dictionary containing parameters for saving the plot, including 'config', 'path', 'showFig', and 'closeFig'.

    Returns:
        fig: The created figure if `ax_in` is provided. Otherwise, displays the plot and returns None.
    """
    # Generate grids if not provided
    if (x_grid is None or y_grid is None) and (pos_range is not None):
        # Generate grids for x and y positions
        x_grid = np.linspace(pos_range[0], pos_range[1], n_pixel)
        y_grid = np.linspace(pos_range[0], pos_range[1], n_pixel)
    else:
        raise ValueError("x_grid and y_grid must be provided")

    # Set position range if not provided
    if pos_range is None:
        if boundary_points is None:
            pos_range = (np.min(positions), np.max(positions))
        else:
            pos_range = (np.min(boundary_points[:,:2]), np.max(boundary_points[:,:2]))

    # Create a new figure if no axis is provided
    if not ax_in:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        ax = ax_in

    # Define color normalization and colormap based on `mse_range` and `proportion`
    vmin, vmax = mse_range if mse_range else (np.nanmin(heatmap), np.nanmax(heatmap))

    if proportion:
        epsilon = 1e-6
        vmax = max(vmax, 0 + epsilon)
        vmin = min(vmin, 0 - epsilon)
        cmap = "coolwarm"
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)
    else:
        cmap = "jet"
        norm = None

    # Plot the heatmap
    cax = ax.contourf(x_grid, y_grid, heatmap, alpha=0.75, levels=25, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)

    # Overlay boundary and trajectory points if provided
    if boundary_points is not None:
        plot_boundaries(boundary_points,ax)
        #ax.plot(boundary_points[:, 0], boundary_points[:, 1], 'o', label='Boundaries', color='black', alpha=0.5, markersize=1)
    if show_trajectory:
        ax.scatter(positions[:, 0], positions[:, 1], color='black', label='Central Positions', alpha=0.2, s=1)  # Central positions

    # Set plot properties
    ax.set_title(title if title else 'MSE\n', fontdict={'fontsize': 14, 'fontweight': 'bold'})
    ax.grid(True)
    ax.set_xlim(x_grid.min(), x_grid.max())
    ax.set_ylim(y_grid.max(), y_grid.min())  # Reversed Y-axis
    ax.set_aspect('equal')

    # Add colorbar if required
    if show_colormap:
        # Create a divider for the existing axis (ax_in)
        divider = make_axes_locatable(ax)
        # Append a new axis for the colorbar on the right with specific size and padding
        cbar_ax = divider.append_axes("right", size="4%", pad=0.05)

        cb = plt.colorbar(cax, cax=cbar_ax)

        # Customize colorbar ticks
        if proportion:
            tick_values = np.linspace(vmin, vmax, num=2)
            tick_values = np.append(tick_values, 0)
            tick_values = np.unique(tick_values)
            # Set ticks on colorbar
            cb.set_ticks(tick_values)
            cb.set_ticklabels([f'{tick:.2f}' for tick in tick_values])

    # Show or return the plot
    if not ax_in:
        #cb = plt.colorbar(cax)
        plt.show()
    else:
        return plt.gcf()

@save_plot
def plot_advanced_comparison_heatmaps(df_results,df_base_results,metrics,base_metrics,pos_range,rooms,position_columns=["X","Y"],boundary_points=None,n_pixel=34,main_title='MSE Comparison of FR2XY and FR2XY2XY Models', fill_value=np.nan,save_params={}):
    """
    Plot a detailed comparison of FR2XY and FR2XY2XY models using heatmaps and trajectories.

    Parameters:
        df_results (pd.DataFrame): Results DataFrame for the FR2XY2XY model, containing actual and predicted positions.
        df_base_results (pd.DataFrame): Results DataFrame for the FR2XY model, containing actual and predicted positions.
        metrics (dict): Metrics for the FR2XY2XY model (e.g., 'mse', 'ev', etc.).
        base_metrics (dict): Metrics for the FR2XY model (e.g., 'mse', 'ev', etc.).
        pos_range (tuple): Range of positions for the plots (e.g., spatial extent of the arena), in the format (min, max).
        rooms (list): List of room names used in the titles, in the order [previous room, train room, test room].
        position_columns (list): List of column names representing the x and y coordinates (default is ["X", "Y"]).
        boundary_points (np.ndarray, optional): Array of points defining the boundaries of the arena to overlay on the plots.
        n_pixel (int): Number of bins for the x and y grids of the heatmaps (default is 34).
        main_title (str): Main title for the plot (default is 'MSE Comparison of FR2XY and FR2XY2XY Models').
        fill_value (float): Value to fill when computing the MSE proportion (e.g., `np.nan` for undefined regions).
        save_params (dict): Parameters for saving the plot, such as file path or format (default is an empty dictionary).

    """

    # Extract position and timestamp data
    real_FR2XY_positions = df_base_results[position_columns].values
    FR2XY_positions = df_base_results[get_prediction_columns(position_columns)].values
    real_FR2XY2XY_positions = df_results[position_columns].values
    FR2XY2XY_positions = df_results[get_prediction_columns(position_columns)].values
    FR2XY_folds = df_base_results['fold'].values
    FR2XY2XY_folds = df_results['fold'].values

    # Generate heatmaps for FR2XY and FR2XY2XY
    old_heatmap, _, _ = create_mse_heatmap(df_base_results, position_columns, get_prediction_columns(position_columns), smoothed=False, pos_range=pos_range)
    new_heatmap, _, _ = create_mse_heatmap(df_results, position_columns, get_prediction_columns(position_columns), smoothed=False, pos_range=pos_range)

    # Create titles for the subplots
    prev_room, train_room, test_room = rooms
    titles = [
        f"FR2XY\n{prev_room}:{test_room}\nMSE: {base_metrics['mse']:.4}, R2: {base_metrics['r2_pooled']:.0%}",
        f"FR2XY2XY\n{train_room}:{test_room}\nMSE: {metrics['mse']:.4}, R2: {metrics['r2_pooled']:.0%}",
        "\nMSE Log Proportion\n",
        "Predicted Positions",
        "Predicted Positions",
        "Real Positions",
    ]

    # Initialize figure and axes
    show_trajectory = True
    if show_trajectory:
        fig, axes = plt.subplots(3, 3, figsize=(12, 9))  # 2 rows, 3 columns
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))  # 1 row, 3 columns

    axes = axes.flatten()

    # Define the MSE range
    mse_range = (0, np.nanmax([new_heatmap, old_heatmap]))

    # Plot base FR2XY heatmap
    ax = axes[0]
    plot_position_mse(old_heatmap, boundary_points=boundary_points, n_pixel=n_pixel, show_trajectory=False, ax_in=ax,
                      pos_range=pos_range, mse_range=mse_range, title=titles[0])

    # Plot FR2XY2XY heatmap
    ax = axes[1]
    plot_position_mse(new_heatmap, boundary_points=boundary_points, n_pixel=n_pixel, show_trajectory=False, ax_in=ax,
                      pos_range=pos_range, mse_range=mse_range, title=titles[1])

    # Plot the MSE proportion between new and old heatmaps
    ax = axes[2]
    mse_proportion = np.where(old_heatmap != 0, np.log(new_heatmap / old_heatmap), fill_value)
    plot_position_mse(mse_proportion, proportion=True, boundary_points=boundary_points, n_pixel=n_pixel,
                      show_trajectory=False, ax_in=ax, pos_range=pos_range, title=titles[2])

    if show_trajectory:

        # Plot real positions trajectory
        ax = axes[3]
        plot_trajectory(real_FR2XY_positions, folds=FR2XY_folds, pos_range=pos_range, boundary_points=boundary_points, title=titles[5], ax_in=ax)

        # Plot real positions trajectory
        ax = axes[4]
        plot_trajectory(real_FR2XY2XY_positions, folds=FR2XY2XY_folds, pos_range=pos_range, boundary_points=boundary_points, title=titles[5], ax_in=ax)

        # Plot FR2XY predicted positions trajectory
        ax = axes[6]
        plot_trajectory(FR2XY_positions, folds=FR2XY_folds, pos_range=pos_range, boundary_points=boundary_points, title=titles[3], ax_in=ax)

        # Plot FR2XY2XY predicted positions trajectory
        ax = axes[7]
        plot_trajectory(FR2XY2XY_positions, folds=FR2XY2XY_folds, pos_range=pos_range, boundary_points=boundary_points, title=titles[4], ax_in=ax)

    axes[5].axis('off')  # Hide the empty subplot
    axes[8].axis('off')  # Hide the empty subplot

    plt.tight_layout()

    if show_trajectory:
        fig.subplots_adjust(hspace=0.5, right=0.95)  # Adjust space for colorbar + titles
    else:
        fig.subplots_adjust(right=0.95)  # Adjust space for colorbar

    # Add the main title
    plt.suptitle(main_title, x=0.5, y=0.98, ha='center', fontsize=14)
    # Add space for the title
    plt.subplots_adjust(top=0.75)  # Adjust 'top' to add more space (higher value creates more space)

    # Show or return the plot
    # plt.show()
    return plt.gcf()


@save_plot
def plot_comparison_heatmaps(new_heatmap, old_heatmap, pos_range, n_pixel=34, titles=None, main_title='MSE Comparison of FR2XY and FR2XY2XY Models', positions=[None,None,None], folds=[None,None,None], boundary_points=None, show_trajectory=False, fill_value=np.nan, save_params={}):
    """
    Plot a comparison between two heatmaps in a 1x3 grid of subplots, showing the new model (FR2XY2XY),
    the baseline model (FR2XY), and the improvement using the proportion of the MSE.

    Parameters:
        new_heatmap (np.ndarray): Heatmap data for the new comparison model (FR2XY2XY).
        old_heatmap (np.ndarray): Heatmap data for the baseline comparison model (FR2XY).
        pos_range (tuple): Range of positions for the plots (e.g., spatial extent of the arena).
        n_pixel (int, optional): Number of pixels (bins) for the x and y grids of the heatmaps. Default is 34.
        titles (list, optional): List of titles for the subplots. It should contain:
            - titles[0]: Title for the baseline heatmap (FR2XY).
            - titles[1]: Title for the new heatmap (FR2XY2XY).
            - titles[2]: Title for the proportion heatmap (FR2XY2XY/FR2XY).
            - titles[3-5]: Titles for the trajectory plots (optional).
        main_title (str, optional): Main title for the entire figure. Default is 'MSE Comparison of FR2XY and FR2XY2XY Models'.
        positions (list, optional): List of position data for the trajectory plots. Default is [None, None, None].
        boundary_points (np.ndarray, optional): Optional array of boundary points to overlay on the plots.
        show_trajectory (bool, optional): Boolean flag to show the trajectory on the plots. Default is False.
        fill_value (float, optional): Value to use where `old_heatmap` is zero. Default is `np.nan`.
        save_params (dict, optional): Parameters for saving the figure (not implemented in the current version).

    Returns:
        matplotlib.figure.Figure: The figure containing the heatmaps.
    """
    # Check if trajectory plots should be added
    if show_trajectory and all(pos is not None for pos in positions):
        fig, axes = plt.subplots(2, 3, figsize=(9 + 2.5, 6 + 1.5))  # 2 rows, 3 columns
    else:
        fig, axes = plt.subplots(1, 3, figsize=(9 + 2.5, 3 + 1.5))  # 1 row, 3 columns

    axes = axes.flatten()  # Flatten axes array to easily iterate

    if titles is None:
        titles = ["MSE FR2XY",
                  "MSE FR2XY2XY",
                  "MSE Log Proportion",
                  "Predicted Positions\nFR2XY",
                  "Predicted Positions\nFR2XY2XY",
                  "Real Positions\n"]


    # Define the MSE range
    mse_range = (0, np.nanmax([new_heatmap, old_heatmap]))

    # Plot base FR2XY heatmap
    ax = axes[0]
    plot_position_mse(old_heatmap, boundary_points=boundary_points, n_pixel=n_pixel, show_trajectory=False, ax_in=ax,pos_range=pos_range, mse_range=mse_range,title=titles[0])

    # Plot FR2XY2XY heatmap
    ax = axes[1]
    plot_position_mse(new_heatmap, boundary_points=boundary_points, n_pixel=n_pixel, show_trajectory=False, ax_in=ax, pos_range=pos_range,mse_range=mse_range, title=titles[1])


    # Plot the MSE proportion between new and old heatmaps
    ax = axes[2]
    mse_proportion = np.where(old_heatmap != 0, np.log(new_heatmap / old_heatmap), fill_value)
    plot_position_mse(mse_proportion, proportion=True, boundary_points=boundary_points, n_pixel=n_pixel, show_trajectory=False, ax_in=ax, pos_range=pos_range,title=titles[2])

    if show_trajectory and all(pos is not None for pos in positions):
        # Plot FR2XY predicted positions trajectory
        ax = axes[3]
        plot_trajectory(positions[1], pos_range=pos_range, boundary_points=boundary_points, title=titles[3], ax_in=ax)

        # Plot FR2XY2XY predicted positions trajectory
        ax = axes[4]
        plot_trajectory(positions[2], pos_range=pos_range, boundary_points=boundary_points, title=titles[4], ax_in=ax)

        # Plot real positions trajectory
        ax = axes[5]
        plot_trajectory(positions[0], pos_range=pos_range, boundary_points=boundary_points, title=titles[5], ax_in=ax)

    plt.tight_layout()

    if show_trajectory and all(pos is not None for pos in positions):
        fig.subplots_adjust(hspace=0.5, right=0.95)  # Adjust space for colorbar + titles
    else:
        fig.subplots_adjust(right=0.95)  # Adjust space for colorbar

    # Add the main title
    plt.suptitle(main_title, x=0.5, y=0.98, ha='center', fontsize=14)
    # Add space for the title
    plt.subplots_adjust(top=0.75)  # Adjust 'top' to add more space (higher value creates more space)

    # Show or return the plot
    # plt.show()
    return plt.gcf()


@save_plot
def plot_heatmaps_per_room(heatmaps, rooms, pos_range, titles=None, main_title=None, boundary_points=None, positions=None, show_trajectory=False, share_mse_range=True,n_pixel=34, save_params={}):
    """
    Plot heatmaps for different rooms in a 1xN grid of subplots, sharing a colorbar.

    Parameters:
        heatmaps (numpy.ndarray): 3D array where each slice along the first dimension is a heatmap for a room.
        rooms (list): List of room names or identifiers corresponding to each heatmap.
        pos_range (tuple): Range of positions for the plots (e.g., spatial extent of the arena).
        boundary_points (optional): Optional boundary points to overlay on the plots.
        show_trajectory (bool): Boolean flag to indicate whether to show trajectory on the plots.
        share_mse_range (bool): Whether to share the MSE range across heatmaps. Default is True.
        n_pixel (int): Number of pixels for the heatmap resolution. Default is 34.
        save_params (dict): Dictionary containing parameters for saving the figure.

    Returns:
        fig: The figure containing the heatmaps.
    """
    num_rooms = len(rooms)

    # Define the MSE range
    #mse_range = (int(np.nanmin(heatmaps)), int(np.nanmin(np.nanmax(heatmaps, axis=(1, 2)))))
    mse_range = (np.nanmin(heatmaps), np.nanmax(heatmaps)) if share_mse_range else None
    if mse_range[1]-mse_range[0] > 10: mse_range = (int(mse_range[0]), int(mse_range[1]))

    # Create a unified subplot for all rooms
    fig, axes = plt.subplots(1, num_rooms, figsize=(3 * num_rooms, 3))  # 1 row, N columns
    #fig.subplots_adjust(right=0.85)  # Adjust space for colorbar

    # Create a colorbar
    #cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])  # Colorbar placement
    #all_levels = np.linspace(mse_range[0], mse_range[1], 25)
    #norm = mcolors.BoundaryNorm(boundaries=all_levels, ncolors=256)
    #sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
    #fig.colorbar(sm, cax=cbar_ax, boundaries=all_levels, ticks=all_levels[::6])

    for i, test_room in enumerate(rooms):
        title = titles[i] if titles is not None else f"Room {test_room}"
        # Plot each heatmap
        positions_room = positions[i] if positions is not None else None
        plot_position_mse(heatmaps[i, :, :], boundary_points=boundary_points, positions=positions_room, show_trajectory=show_trajectory,n_pixel=n_pixel,
                          ax_in=axes.flatten()[i], pos_range=pos_range, mse_range=mse_range, show_colormap=True, title=title)

    # Add main title for the entire figure
    if main_title:
        plt.suptitle(main_title, fontsize=16, y=0.99)  # Adjust y-position as needed

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show or return the plot
    #     plt.show()
    return plt.gcf()


@save_plot
def plot_advanced_heatmaps_per_room(df_results, metrics, rooms, pos_range, main_title, boundary_points, n_pixel, train_room=None, save_params={}):
    """
    Plot advanced heatmaps for different rooms in a grid of subplots, with additional subplots for MSE-X, MSE-Y, and Mapping Trajectory.

    Parameters:
        df_results (DataFrame): DataFrame containing the results with 'room' column and predictions.
        metrics (list): List of metrics
        rooms (list): List of room names or identifiers.
        pos_range (tuple): Range of positions for the plots (e.g., spatial extent of the arena).
        main_title (str): Main title for the entire figure.
        boundary_points (optional): Optional boundary points to overlay on the plots.
        n_pixel (int): Number of pixels for the heatmap resolution.
        save_params (dict): Dictionary containing parameters for saving the figure.

    Returns:
        fig: The figure containing the heatmaps and additional subplots.
    """

    # Set position range if not provided
    if pos_range is None:
        if boundary_points is not None:
            pos_range = (np.min(boundary_points[:,:2]), np.max(boundary_points[:,:2]))

    num_rooms = len(rooms)
    fig, axes = plt.subplots(5, num_rooms, figsize=(num_rooms * 3, 5 * 3))

    for i, test_room in enumerate(rooms):
        # Filter the DataFrame for the current room
        df_room = df_results[df_results['room'] == test_room]

        # Create the main heatmap for MSE
        heatmap, _, _ = create_mse_heatmap(df_room, ['X','Y'], get_prediction_columns(['X','Y']), n_pixel=n_pixel, smoothed=False, pos_range=pos_range)
        # heatmap subplot
        ax_mse = axes[0, i]

        title = f'{train_room}-{test_room}' if train_room else f'{test_room}'
        title += f"\nMSE: {metrics[i]['mse']:.3f}\nR2: {metrics[i]['r2_pooled']:.0%}"
        # Plot each heatmap
        plot_position_mse(heatmap, boundary_points=boundary_points, n_pixel=n_pixel,
                          ax_in=ax_mse, pos_range=pos_range, mse_range=None, show_colormap=True,title=title)

        # Create subplots for MSE-X, MSE-Y, and Mapping Trajectory
        heatmaps_x, _, _ = create_mse_heatmap(df_room, ['X'], get_prediction_columns(['X']), n_pixel=n_pixel, smoothed=False, pos_range=pos_range)
        heatmaps_y, _, _ = create_mse_heatmap(df_room, ['Y'], get_prediction_columns(['Y']), n_pixel=n_pixel, smoothed=False, pos_range=pos_range)

        # Subplots for MSE-X, MSE-Y, and Trajectory on the 4th row
        ax_mse_x = axes[1, i]       # Subplot for MSE-X
        ax_mse_y = axes[2, i]   # Subplot for MSE-Y
        ax_target = axes[3, i]  # Subplot for target
        ax_pred = axes[4, i]  # Subplot for predictions

        # MSE-X subplot
        r2_x_val = metrics[i].get('r2_x', metrics[i]['r2_pooled']) if 'r2_x' in metrics[i] else metrics[i]['r2_pooled']
        plot_position_mse(heatmaps_x, boundary_points=boundary_points, n_pixel=n_pixel,
                          ax_in=ax_mse_x, pos_range=pos_range, mse_range=None, show_colormap=True, title=f"MSE: {metrics[i]['mse']:.3f}\nR2: {r2_x_val:.0%}")

        # MSE-Y subplot
        r2_y_val = metrics[i].get('r2_y', metrics[i]['r2_pooled']) if 'r2_y' in metrics[i] else metrics[i]['r2_pooled']
        plot_position_mse(heatmaps_y, boundary_points=boundary_points, n_pixel=n_pixel,
                          ax_in=ax_mse_y, pos_range=pos_range, mse_range=None, show_colormap=True, title=f"MSE: {metrics[i]['mse']:.3f}\nR2: {r2_y_val:.0%}")

        # Target subplot
        plot_trajectory(df_room[['X', 'Y']].values, folds=df_room['fold'].values, pos_range=pos_range, boundary_points=boundary_points, title=' ', ax_in=ax_target)
        # if boundary_points is not None:
        #     plot_boundaries(boundary_points, ax_target)
        # ax_target.scatter(df_room['X'], df_room['Y'],c=df_room['fold'], cmap='viridis', alpha=0.6, s=1)
        # # ax_traj.set_title("Mapping Trajectory", fontsize=10)
        # ax_target.grid(True)
        # ax_target.set_xlim(pos_range[0], pos_range[1])
        # ax_target.set_ylim(pos_range[1], pos_range[0])  # Reversed Y-axis
        # ax_target.set_aspect('equal')

        # Predictions subplot
        plot_trajectory(df_room[get_prediction_columns(['X', 'Y'])].values, folds=df_room['fold'].values, pos_range=pos_range, boundary_points=boundary_points, title=' ', ax_in=ax_pred)
        # if boundary_points is not None:
        #     plot_boundaries(boundary_points, ax_pred)
        # ax_pred.scatter(df_room[get_prediction_columns(['X'])], df_room[get_prediction_columns(['Y'])],c=df_room['fold'], cmap='viridis', alpha=0.6, s=1)
        # # ax_traj.set_title("Mapping Trajectory", fontsize=10)
        # ax_pred.grid(True)
        # ax_pred.set_xlim(pos_range[0], pos_range[1])
        # ax_pred.set_ylim(pos_range[1], pos_range[0])  # Reversed Y-axis
        # ax_pred.set_aspect('equal')

    # Add y-axis labels for the leftmost subplots
    axes[0, 0].set_ylabel("MSE", fontsize=14, rotation=90, labelpad=30, ha='center', va='center', fontweight='bold')
    axes[1, 0].set_ylabel("MSE - X", fontsize=14, rotation=90, labelpad=30, ha='center', va='center', fontweight='bold')
    axes[2, 0].set_ylabel("MSE - Y", fontsize=14, rotation=90, labelpad=30, ha='center', va='center', fontweight='bold')
    axes[3, 0].set_ylabel("Target", fontsize=14, rotation=90, labelpad=30, ha='center', va='center', fontweight='bold')
    axes[4, 0].set_ylabel("Predictions", fontsize=14, rotation=90, labelpad=30, ha='center', va='center', fontweight='bold')

    # Add main title
    plt.suptitle(main_title, fontsize=16)
    plt.subplots_adjust(top=1 - ((main_title.count('\n') + 1) * 0.05))  # Decrease the top margin


    # Adjust layout
    plt.tight_layout()

    return plt.gcf()





# @save_plot
# def plot_colors_mapping(initial_colors, new_colors, xv_new, yv_new, pos_range, boundary_points=None, save_params={}):
#     """
#     Visualize the mapping of colors from room A to room B.
#
#     Args:
#         initial_colors: Gradient colors for room A.
#         new_colors: Mapped colors in room B.
#         xv_new, yv_new: Meshgrid for room B.
#         pos_range: The original positional range of the arena.
#         boundary_points: Optional, points representing boundaries to be plotted.
#     """
#     plt.figure(figsize=(10, 5))
#
#     # Original gradient pattern in room A
#     ax = plt.subplot(1, 2, 1)
#     cax = plt.imshow(initial_colors, origin='lower', extent=(pos_range[0], pos_range[1], pos_range[0], pos_range[1]), cmap='jet', vmin=0, vmax=1)
#     plt.title('Original Gradient (Room A)')
#
#     # Create a divider for the existing axis (ax_in)
#     divider = make_axes_locatable(ax)
#     # Append a new axis for the colorbar on the right with specific size and padding
#     cbar_ax = divider.append_axes("right", size="4%", pad=0.05)
#
#     cb = plt.colorbar(cax, cax=cbar_ax)
#     cb.set_ticks([])  # Remove ticks from the color bar
#
#     # Plot boundary points if provided
#     if boundary_points is not None:
#         ax.plot(boundary_points[:, 0], boundary_points[:, 1], 'o', label='Boundaries', color='black', alpha=0.5,
#                 markersize=1)
#
#     # Mapped gradient pattern in room B
#     ax = plt.subplot(1, 2, 2)
#     cax = plt.imshow(new_colors, origin='lower', extent=(pos_range[0], pos_range[1], pos_range[0], pos_range[1]), cmap='jet', vmin=0, vmax=1)
#     plt.title('Mapped Gradient (Room B)')
#
#     # Create a divider for the existing axis (ax_in)
#     divider = make_axes_locatable(ax)
#     # Append a new axis for the colorbar on the right with specific size and padding
#     cbar_ax = divider.append_axes("right", size="4%", pad=0.05)
#
#     cb = plt.colorbar(cax, cax=cbar_ax)
#     cb.set_ticks([])  # Remove ticks from the color bar
#
#
#     # Plot boundary points if provided
#     if boundary_points is not None:
#         ax.plot(boundary_points[:, 0], boundary_points[:, 1], 'o', label='Boundaries', color='black', alpha=0.5,
#                 markersize=1)
#
#     return plt.gcf()

#@save_plot
def plot_trajectory_animation(
    real_positions,
    predicted_positions,
    boundary_points=None,
    pos_range=None,
    skip_frames=10,
    points_per_frame=60,
    export_path='',
    config={},
    showFig=False,
    closeFig=True,
    frame_duration_ms=None,
):
    """
    Creates an animated plot comparing real and predicted xy positions over time.

    :param real_positions: Numpy array of shape (timestamps, 2) containing (x, y) real positions.
    :param predicted_positions: Numpy array of shape (timestamps, 2) containing (x, y) predicted positions.
    :param boundary_points: Optional Numpy array of boundary points to plot (e.g., walls or arena boundaries).
    :param pos_range: Tuple (min, max) for the xy axis range (optional).
    :param skip_frames: Number of frames to skip for faster animation (default is 2).
    :param points_per_frame: Number of points to display per frame (default is 60).
    :param export_path: Path to save the animation video or GIF (optional).
    :param config: Configuration for saving the animation (optional).
    :param showFig: Whether to show the figure during execution (optional).
    :param closeFig: Whether to close the figure after saving (optional).
    :param frame_duration_ms: Milliseconds per frame for GIF (optional); if set, fps = 1000 / frame_duration_ms.
    """

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')

    if pos_range:
        ax.set_xlim(pos_range[0], pos_range[1])
        ax.set_ylim(pos_range[1], pos_range[0])

    ax.set_title('Real vs Predicted Trajectory')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')

    num_timestamps = real_positions.shape[0]
    max_trace_length = 30  # Show only the last 30 points

    if boundary_points is not None:
        plot_boundaries(boundary_points,ax)
        #ax.plot(boundary_points[:, 0], boundary_points[:, 1], 'o', label='Boundaries', color='black', alpha=0.5, markersize=1)

    # Set color maps for gradients
    real_cmap = cm.get_cmap('Blues')
    predicted_cmap = cm.get_cmap('Reds')

    # Initialize the scatter plots
    real_scatter, = ax.plot([], [], 'bo-', lw=2, label='Real', alpha=0.8)
    predicted_scatter, = ax.plot([], [], 'ro-', lw=2, label='Predicted', alpha=0.8)

    def init():
        real_scatter.set_data([], [])
        predicted_scatter.set_data([], [])
        return real_scatter, predicted_scatter

    def update(frame):
        # Calculate the current frame index to skip frames
        current_frame = frame * skip_frames
        # Ensure we start from max_trace_length
        start_idx = max(0, current_frame - points_per_frame + 1)
        end_idx = min(num_timestamps, current_frame + 1)

        # Slice the positions for real and predicted
        real_trace = real_positions[start_idx:end_idx]
        predicted_trace = predicted_positions[start_idx:end_idx]

        # Update positions for both real and predicted
        real_scatter.set_data(real_trace[:, 0], real_trace[:, 1])
        predicted_scatter.set_data(predicted_trace[:, 0], predicted_trace[:, 1])

        # Plot with color gradient for Real
        for i in range(real_trace.shape[0] - 1):
            color = real_cmap(i / (real_trace.shape[0] - 1))
            ax.plot(real_trace[i:i + 2, 0], real_trace[i:i + 2, 1], color=color, lw=2)

        # Plot with color gradient for Predicted
        for i in range(predicted_trace.shape[0] - 1):
            color = predicted_cmap(i / (predicted_trace.shape[0] - 1))
            ax.plot(predicted_trace[i:i + 2, 0], predicted_trace[i:i + 2, 1], color=color, lw=2)

        return real_scatter, predicted_scatter

    ax.legend()
    ax.grid()

    # Create animation
    ani = FuncAnimation(fig, update, frames=num_timestamps // skip_frames, init_func=init, blit=True, repeat=True)

    #Save animation if a path is given
    if export_path:
        if not config:
            full_path = export_path
            dir_path = os.path.dirname(export_path)
            os.makedirs(dir_path, exist_ok=True)
        else:
            full_path = get_full_path_to_save_file(config, export_path)

        if export_path.endswith('.gif'):
            fps = (1000.0 / frame_duration_ms) if frame_duration_ms and frame_duration_ms > 0 else 30
            ani.save(full_path, writer='pillow', fps=fps)
        elif export_path.endswith('.mp4'):
            writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
            ani.save(full_path, writer=writer)

        logger.info(f"Animation saved at: {full_path}")
    else:
        if showFig:
            plt.show()

    if closeFig:
        plt.close(fig)

    return ani

    # return plt.gcf()


@save_plot
def plot_line_plot_over_time(real_positions, predicted_positions, timestamp, groups=None, pos_range=None, title="Real vs Predicted Positions over Time", save_params={}):
    """
    Plot real vs predicted positions over time for both x and y components, including error bands.

    Parameters:
        real_positions (numpy.ndarray): Array of real positions of shape (N, 2) where N is the number of timestamps.
        predicted_positions (numpy.ndarray): Array of predicted positions of shape (N, 2) where N is the number of timestamps.
        timestamp (iterable): Iterable containing timestamps for the x-axis.
        groups (iterable, optional): Iterable containing group identifiers for each timestamp.
        pos_range (tuple, optional): Tuple (min, max) for the xy axis range. Default is None.
        title (str, optional): Custom title for the plot. Default is "Real vs Predicted Positions over Time".
        save_params (dict): Dictionary containing parameters for saving the plot.

    Returns:
        fig: The figure containing the line plots of real and predicted positions over time.
    """

    if timestamp is None:
        timestamp = range(len(real_positions))  # Assuming positions are indexed by time

    # Create a DataFrame for easy grouping and plotting
    data = pd.DataFrame({
        'timestamp': timestamp,
        'real_x': real_positions[:, 0],
        'real_y': real_positions[:, 1],
        'pred_x': predicted_positions[:, 0],
        'pred_y': predicted_positions[:, 1],
        'group': groups if groups is not None else "All"
    })

    # Group by 'timestamp' and calculate the mean and std for each position
    data_stats = data.groupby(['timestamp','group']).agg({
        'real_x': ['mean', 'std'],
        'real_y': ['mean', 'std'],
        'pred_x': ['mean', 'std'],
        'pred_y': ['mean', 'std']
    }).reset_index()

    # Flatten the MultiIndex columns after aggregation
    data_stats.columns = ['_'.join(col).strip() for col in data_stats.columns.values]
    # Remove the extra '_' from column names like 'timestamp_' and 'group_'
    data_stats.columns = [col[:-1] if col.endswith('_') else col for col in data_stats.columns]

    # Sort the DataFrame by the 'timestamp' column
    data_stats = data_stats.sort_values(by='timestamp')

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    # Set axis limits if pos_range is provided
    if pos_range:
        axs[0].set_ylim(pos_range)
        axs[1].set_ylim(pos_range)

    # Track if the legend has already been added
    legend_added = False

    # Plot X and Y positions, grouped if groups are provided
    for (name, data_grouped) in data_stats.groupby('group'):
        # Plot X and Y positions, grouped if groups are provided
        axs[0].plot(data_grouped['timestamp'].values, data_grouped['real_x_mean'].values, label=f"Real X" if not legend_added else "",
                    color="blue", alpha=0.6)
        axs[0].plot(data_grouped['timestamp'].values, data_grouped['pred_x_mean'].values,
                    label=f"Predicted X" if not legend_added else "", color="red", alpha=0.6)
        axs[0].fill_between(data_grouped['timestamp'].values,
                            data_grouped['real_x_mean'].values - data_grouped['real_x_std'].values,
                            data_grouped['real_x_mean'].values + data_grouped['real_x_std'].values,
                            color="blue", alpha=0.3)
        axs[0].fill_between(data_grouped['timestamp'].values,
                            data_grouped['pred_x_mean'].values - data_grouped['pred_x_std'].values,
                            data_grouped['pred_x_mean'].values + data_grouped['pred_x_std'].values,
                            color="red", alpha=0.3)

        axs[1].plot(data_grouped['timestamp'].values, data_grouped['real_y_mean'].values, label=f"Real Y" if not legend_added else "",
                    color="blue", alpha=0.6)
        axs[1].plot(data_grouped['timestamp'].values, data_grouped['pred_y_mean'].values,
                    label=f"Predicted Y" if not legend_added else "", color="red", alpha=0.6)
        axs[1].fill_between(data_grouped['timestamp'].values,
                            data_grouped['real_y_mean'].values - data_grouped['real_y_std'].values,
                            data_grouped['real_y_mean'].values + data_grouped['real_y_std'].values,
                            color="blue", alpha=0.3)
        axs[1].fill_between(data_grouped['timestamp'].values,
                            data_grouped['pred_y_mean'].values - data_grouped['pred_y_std'].values,
                            data_grouped['pred_y_mean'].values + data_grouped['pred_y_std'].values,
                            color="red", alpha=0.3)

        legend_added = True  # Set legend flag to True after first plot

    # Configure plot aesthetics

    #axs[0].set_title(title, fontsize=14, ha='center', pad=(title.count('\n') + 0) * 10)
    axs[0].set_ylabel("X Position")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Y Position")

    # Show legends
    axs[0].legend()
    axs[1].legend()


    # Adjust title space based on the number of line breaks
    fig.suptitle(title, fontsize=14, ha='center')

    # Adjust the space for the main title
    plt.subplots_adjust(top=1 - ((title.count('\n') + 1) * 0.05))  # Decrease the top margin


    return plt.gcf()



@save_plot
def plot_position_trajectory_animation_html(real_positions, predicted_positions, timestamp=None, boundary_points=None,
                                            pos_range=None, title = "Real vs Predicted Trajectory", skip_frames=20, points_per_frame=40,
                                            save_params={}):
    """
    Creates an animated plot comparing real and predicted xy positions over time using Plotly.

    Parameters:
        real_positions (numpy.ndarray): Array of shape (timestamps, 2) containing real (x, y) positions.
        predicted_positions (numpy.ndarray): Array of shape (timestamps, 2) containing predicted (x, y) positions.
        timestamp (numpy.ndarray, optional): Array of timestamps for the slider. If None, timestamps will be auto-generated.
        boundary_points (numpy.ndarray, optional): Array of boundary points (e.g., walls or arena boundaries).
        pos_range (tuple, optional): Tuple (min, max) defining the xy axis range.
        title (str, optional): Title for the plot. Defaults to "Real vs Predicted Trajectory".
        skip_frames (int, optional): Number of frames to skip when generating the animation. Defaults to 20.
        points_per_frame (int, optional): Number of points to display per frame in the animation. Defaults to 40.
        save_params (dict, optional): Dictionary containing parameters for saving the plot, such as file path, format,
                                      and display options.

    Returns:
        fig: The matplotlib figure object containing the animated plot.
    """

    num_timestamps = real_positions.shape[0]
    initial_duration = 50  # Initial duration (ms per frame)

    # Create figure and add scatter traces for both real and predicted positions
    fig = go.Figure()

    # Initialize scatter plots for real and predicted positions with all points invisible
    fig.add_trace(go.Scatter(x=real_positions[:, 0], y=real_positions[:, 1], mode='lines+markers',
                             line=dict(color='blue'), name='Real', visible=False))
    fig.add_trace(go.Scatter(x=predicted_positions[:, 0], y=predicted_positions[:, 1], mode='lines+markers',
                             line=dict(color='red'), name='Predicted', visible=False))

    # Add boundary lines for each room if boundary_points are provided
    if boundary_points is not None:
        room_keys = np.unique(boundary_points[:, -1])
        for room_key in room_keys:
            # Get points for the current room
            room_boundary_points = boundary_points[boundary_points[:, -1] == room_key]

            # Ensure the first and last points are connected to close the boundary
            room_boundary_points = np.vstack((room_boundary_points, room_boundary_points[0]))

            # Plot the boundary for the current room
            fig.add_trace(go.Scatter(
                x=room_boundary_points[:, 0],
                y=room_boundary_points[:, 1],
                mode='lines',
                line=dict(color='black', width=2),
                name=f'Boundaries',
                hoverinfo='none',  # Disable hover information to save memory
                visible=True  # Keep the boundaries visible for all frames
            ))

    # Customize the layout
    fig.update_layout(
        title=title,
        xaxis_title="X Position",
        yaxis_title="Y Position",
        showlegend=True,
        xaxis=dict(range=pos_range) if pos_range else None,
        yaxis=dict(range=pos_range[::-1]) if pos_range else None,
        width=800,
        height=800,
        updatemenus=[{
            'buttons': [
                {'label': '⏵', 'method': 'animate',
                 'args': [None, {'frame': {'duration': initial_duration, 'redraw': True},
                                 'fromcurrent': True, 'mode': 'immediate'}]},
                {'label': '⏸', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                                                          'mode': 'immediate'}]}
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 70},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }]
    )

    # Create frames for animation
    frames = []
    for i in range(0, num_timestamps, skip_frames):
        frame_data = [
            go.Scatter(x=real_positions[max(0, i - points_per_frame):i, 0],
                       y=real_positions[max(0, i - points_per_frame):i, 1],
                       name='Real', visible=True),  # Set visibility for current frame
            go.Scatter(x=predicted_positions[max(0, i - points_per_frame):i, 0],
                       y=predicted_positions[max(0, i - points_per_frame):i, 1],
                       name='Predicted', visible=True),
        ]

        # Add the boundary lines to each frame
        if boundary_points is not None:
            for room_key in room_keys:
                room_boundary_points = boundary_points[boundary_points[:, -1] == room_key]
                room_boundary_points = np.vstack((room_boundary_points, room_boundary_points[0]))
                frame_data.append(
                    go.Scatter(
                        x=room_boundary_points[:, 0],
                        y=room_boundary_points[:, 1],
                        mode='lines',
                        line=dict(color='black', width=2),
                        name=f'Boundaries',
                        hoverinfo='none',
                        visible=True
                    )
                )

        frames.append(go.Frame(data=frame_data, name=str(i)))

    # Add slider to control the animation
    if timestamp is None:
        timestamp_labels = [str(i) for i in range(0, num_timestamps, skip_frames)]
    else:
        timestamp_labels = [f"{ts:.2f}" for ts in timestamp[::skip_frames]]

    sliders = [{
        'steps': [{'args': [[str(i)], {'frame': {'duration': 0, 'redraw': True},
                                       'mode': 'immediate'}],
                   'label': timestamp_labels[i // skip_frames],
                   'method': 'animate'} for i in range(0, num_timestamps, skip_frames)],
        'currentvalue': {'visible': True, 'prefix': 'Time: ', 'xanchor': 'right'},
        'pad': {'b': 10, 't': 50},
        'x': 0.1,
        'len': 0.9
    }]

    # ## Speed slider to dynamically adjust the frame duration (animation speed)
    # speed_slider = {
    #     'active': 0,
    #     'yanchor': 'top',
    #     'xanchor': 'left',
    #     'currentvalue': {
    #         'prefix': 'Speed (ms/frame): ',
    #         'font': {'size': 10}
    #     },
    #     'pad': {'b': 10, 't': 70},
    #     'len': 0.1,
    #     'x': 0,  # Position the speed slider to the right of the main slider
    #     'y': -0.15,  # Position the speed slider below the main slider
    #     'steps': [
    #         {'args': [None, {'frame': {'duration': speed, 'redraw': True},
    #                          'mode': 'immediate'}],
    #          'label': str(speed),
    #          'method': 'animate'} for speed in [5, 10, 50, 100, 200, 500]  # Custom speed values
    #     ]
    # }

    # Update layout with frames and slider
    fig.update(frames=frames)
    fig.update_layout(sliders=sliders)
    # fig.update_layout({'sliders': [sliders[0], speed_slider]})  # Add both sliders to the layout


    return  fig


@save_plot
def plot_score_heatmaps(pivot_table, main_title=None, save_params={}):
    """
    Plots heatmaps representing explained variance (EV) for each project and room combination with numeric and percentage annotations.

    Parameters:
        pivot_table (DataFrame): The pivot table to visualize.
        main_title (str, optional): Main title for the entire figure.

        save_params (dict): Dictionary containing parameters for saving the heatmaps.

    Returns:
        None: Displays or saves the heatmaps.
    """

    # check if index is a MultiIndex
    if not isinstance(pivot_table.index, pd.MultiIndex):
        unique_projects = pivot_table.index.unique()  # Get unique project names
        num_projects = len(unique_projects)  # Number of unique projects
    else:
        # Get unique projects
        unique_projects = pivot_table.index.levels[0]  # Get unique project names
        num_projects = len(unique_projects)  # Number of unique projects

    # Determine the number of rows and columns (max 5 columns per row)
    cols = min(5, len(unique_projects))
    rows = (num_projects + cols - 1) // cols  # Calculate the required number of rows

    # Set the size of the figure and create subplots
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))  # Adjust size for rows and columns
    axes = axes.flatten() if num_projects > 1 else [axes]  # Flatten axes for easier indexing

    for i, project in enumerate(unique_projects):
        # Filter pivot_table for the specific project and copy the data
        project_data = pivot_table.xs(project, level=0).copy()  # Extract and copy data for the current project

        # Drop rows and columns with all NaN values
        project_data = project_data.dropna(how='all', axis=0)  # Drop empty rows
        project_data = project_data.dropna(how='all', axis=1)  # Drop empty columns

        # Get union of all x and y values (columns and index) across projects
        all_x_values = project_data.columns
        all_y_values = project_data.index  # Assuming rooms are in level 1 of the MultiIndex

        # Reindex project_data to ensure consistent x and y values
        # project_data = project_data.reindex(index=all_y_values, columns=all_x_values)

        # Create the heatmap using the project data
        sns.heatmap(
            project_data.astype(float),  # Convert the pivot table values to float
            annot=True,  # Show the values on the heatmap
            fmt=".0%",  # Format for numeric values
            cmap='RdYlGn',  # Red-Green colormap
            cbar=False,  # Disable the color bar
            vmin=0,  # Set minimum color range
            vmax=1,  # Set maximum color range
            linewidths=0.1,  # Add grid lines
            linecolor='gray',  # Color of the grid lines
            square=True,  # Make cells square
            ax=axes[i]  # Specify the subplot axes
        )

        # Set the x and y labels using the capitalized index and column names
        x_label = project_data.columns.name.title()  # Capitalize each word of the column names
        y_label = project_data.index.name.title()  # Capitalize each word of the index names
        axes[i].set_xlabel(x_label, fontsize=10)
        axes[i].set_ylabel(y_label, fontsize=10)

        # Set consistent x and y ticks
        axes[i].set_xticks([x + 0.5 for x in range(len(all_x_values))])
        axes[i].set_xticklabels(all_x_values, fontsize=10)
        # Move x-axis labels to the top
        axes[i].xaxis.set_label_position('top')  # Move x-labels to the top
        axes[i].xaxis.tick_top()  # Move ticks to the top

        axes[i].set_yticks([x + 0.5 for x in range(len(all_y_values))])  # Shift y-ticks by 0.5
        axes[i].set_yticklabels(all_y_values, fontsize=10)

        # Remove small ticks from x and y axes
        axes[i].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

        # Set the title for the current subplot
        axes[i].set_title(project, fontsize=12, pad=15)

    # Remove unused subplots, if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])  # Delete extra subplots that are not used

    # Add main title if provided
    if main_title:
        fig.suptitle(main_title, fontsize=14, weight='bold', y=0.98)  # Adjust 'y' and reduce font size

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make sure there's room for the title

    return plt.gcf()


@save_plot
def plot_score_heatmap_minimal(pivot_table, main_title=None, save_params={}):
    """
    Minimal heatmap plot for R² scores.

    Parameters:
        pivot_table (DataFrame): Must be a single project × test room format.
        main_title (str): Title for the entire figure (project name or metric name).
        save_params (dict): Saving options.

    Returns:
        matplotlib Figure
    """

    fig, ax = plt.subplots(figsize=(2,1.8))

    # Remove unused rows/cols
    pivot_table = pivot_table.dropna(how='all', axis=0).dropna(how='all', axis=1)

    # Plot the heatmap
    sns.heatmap(
        pivot_table.astype(float),
        annot=True,  # Show the values on the heatmap
        fmt=".0%",  # Format for numeric values
        cmap='RdYlGn',  # Red-Green colormap
        cbar=False,  # Disable the color bar
        vmin=0,  # Set minimum color range
        vmax=1,  # Set maximum color range
        linewidths=0.3,  # Add grid lines
        linecolor='white',  # Color of the grid lines
        square=True,  # Make cells square
        ax=ax
    )

    # Keep only top x-axis (Test Room) and project name title
    ax.set_xticks([x + 0.5 for x in range(len(pivot_table.columns))])
    ax.set_xticklabels(pivot_table.columns, fontsize=10)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    # ax.set_xlabel("Test Room", fontsize=10)

    # Remove y-axis labels and ticks
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticklabels([])

    # Clean tick marks
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

    # Add title
    if main_title:
        fig.suptitle(main_title, fontsize=10, y=0.9)

    # Adjust layout without clipping the title
    plt.subplots_adjust(top=0.65, bottom=0.05)
    plt.tight_layout()
    return fig


def plot_boundaries(boundary_points, ax, color='black', alpha=1.0, linewidth=3, shift=0):
    """
    Plot the boundaries of different rooms.

    Parameters:
        boundary_points (np.ndarray): 2D array of shape (n_points, 3) where each row contains
                                       [X, Y, Room Key].
        ax (matplotlib.axes.Axes): The axis on which to plot the boundaries.
        color (str, optional): The color of the boundary lines. Default is 'black'.
        alpha (float, optional): The transparency of the boundary lines. Default is 0.5.
        linewidth (int, optional): The width of the boundary lines. Default is 3.
        shift (int, optional): Amount to shift the boundary points. Default is 0.
    """
    # Extract unique room keys
    room_keys = np.unique(boundary_points[:, -1])
    first_plot = True
    for room_key in room_keys:
        # Get the points for the current room
        room_boundary_points = boundary_points[boundary_points[:, -1] == room_key]

        # Ensure the first and last points are connected
        room_boundary_points = np.vstack((room_boundary_points, room_boundary_points[0]))

        # Plot the room boundary
        ax.plot(room_boundary_points[:, 0] + shift, room_boundary_points[:, 1],
                '-',  label='Boundaries' if first_plot else "", color=color, alpha=alpha, linewidth=linewidth)
        first_plot = False




@save_plot
def plot_cells_Pos_TC(df_clusters, df_dataset, max_cells, boundary_points=None, cell_info=None, save_params={}):
    """
    Generates position tuning curves for cells grouped by clusters.

    Parameters:
        df_clusters (DataFrame): DataFrame containing cluster information with columns for 'cell', 'cluster', 'depth', 'fr', 'amp', and 'KSlabel'.
        df_dataset (DataFrame): DataFrame containing the dataset used for plotting.
        max_cells (int): Maximum number of cells to display per cluster.
        boundary_points (optional): Optional boundary points to overlay on the plots.
        cell_info (dict): Dictionary where keys are column names to display, and values are short titles for each column.
        save_params (dict): Dictionary containing parameters for saving the plots, such as 'path' and 'config'.

    Returns:
        fig: The figure containing the position tuning curves for the specified cells in clusters.
    """

    clusters = np.sort(df_clusters.cluster.unique().tolist())
    n_clusters = len(clusters)

    fig, axs = plt.subplots(n_clusters, max_cells, figsize=(2 * max_cells, 2 * n_clusters), sharey=True)

    # Flatten the array of axes for easier iteration
    if n_clusters == 1:
        axs = axs.reshape(1, -1)  # Reshape if only one cluster to avoid issues with subplot indexing

    # Flatten the array of axes and apply settings to all
    for ax in axs.flatten():
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

    # Use tqdm to monitor progress
    pbar = tqdm(total=len(df_clusters), desc='Generating Position Tuning Curves')

    for c, cluster_number in enumerate(clusters):
        cluster_cells = df_clusters[df_clusters['cluster'] == cluster_number]['cell'].tolist()
        n_cells = len(cluster_cells)

        for i, cell in enumerate(cluster_cells):
            if i >= max_cells:
                for j in range(i, n_cells):
                    pbar.update(1)  # Update progress bar
                break

            df_cell = df_clusters[df_clusters['cell'] == cell]
            cell_details = {title: df_cell[col].values[0] for col, title in (cell_info or {}).items()}
            title = f'Cell #{cell}'

            # Generate cell details string
            details = ' | '.join([f'{title}: {value}' for title, value in cell_details.items()])

            title = title + '\n' + details

            # depth = df_cell['depth'].values[0]
            # fr = df_cell['fr'].values[0]
            # amp = df_cell['amp'].values[0]
            # KSlabel = df_cell['KSlabel'].values[0]
            # title = f'Cell #{cell}'  # \n{KSlabel:.1}|{depth:^ 5d}|{fr:^ 5.1f}|{amp:^ 5.1f}' #{"KSlabel":^7}|{"depth":^5}|{"fr":^5}|{"amp":^5}\n
            # details = f'{KSlabel:.1}|{depth:^ 5d}|{fr:^ 5.1f}|{amp:^ 5.1f}'

            ax = axs[c, i]
            cell_column = f'Cell_{cell}'
            plot_position_spike_rates(df_dataset, cell_column, title=title, boundary_points=boundary_points, n_columns=1,save_params={'showFig':False}, ax_in=ax)

            # Add details text above the plot
            ax.text(0.5, 1.1, details, fontsize=7, ha='center', va='bottom', transform=ax.transAxes,
                    fontdict={'color': 'black', 'family': 'monospace'})

            # Set cluster title for the first subplot in each row
            if i == 0:
                cluster_label = f'Cluster {cluster_number}'
                headers = ' | '.join([f'{title}' for title in cell_info.values()])
                ax.text(-0.15, 1.1, cluster_label, fontsize=10, fontweight='bold', ha='center', va='bottom', transform=ax.transAxes)
                ax.text(0.5, 1.25, headers, fontsize=8, fontweight='bold', ha='center', va='bottom', transform=ax.transAxes)



            # ax.text(-0.01, 0.5, details, fontsize=7, ha='right', va='center', rotation='vertical',
            #         transform=ax.transAxes,
            #         fontdict={'color': 'black', 'family': 'monospace'})  # , fontdict={'family': 'monospace'})
            #
            #     # Set title only for the first subplot in each row
            # if i == 0:
            #     ax.text(-0.3, 0.5, f'Cluster {cluster_number}', fontsize=10, fontweight='bold', va='center', ha='right',
            #             rotation='vertical', transform=ax.transAxes)
            #     ax.text(-0.15, 0.5, f'{"KS":^3}|{"depth":^7}|{"fr":^7}|{"amp":^7}', fontsize=8, fontweight='bold',
            #             va='center', ha='right', rotation='vertical', transform=ax.transAxes)

            pbar.update(1)  # Update progress bar

    pbar.close()  # Close tqdm progress bar
    # plt.tight_layout()
    return fig



@save_plot
def plot_position_spike_rates(
    df_data,
    cells_columns,
    target_columns=['X', 'Y'],
    boundary_points=None,
    map_rooms=None,
    dt=None,
    min_fr_bin_treshold_s=None,
    min_spike_treshold=None,
    n_pixel=34,
    sigma=2,
    kernel_size=8,
    n_columns=8,
    pos_range=None,
    smoothed=True,
    threshold_marker=None,
    show_colorbar=False,
    color_range=None,
    main_title='',
    title='',
    title_color='black',
    fill_value=0,
    room_normalization=False,
    minimized_layout=False,
    pad_in_pixels=1,
    ax_in=None,
    save_params={},
):
    """
    Perform heatmap analysis of spike rates by positions and plot the results.

    Parameters:
        df_data (DataFrame): DataFrame containing the dataset used for plotting.
        cells_columns (list): List of columns containing spike rates for each cell.
        target_columns (list, optional): List of columns to use as target columns for the heatmap. Default is ['X', 'Y'].
        boundary_points (np.ndarray, optional): Array of boundary points to overlay on the heatmaps. Default is None.
        map_rooms (dict, optional): Dictionary mapping room names to indices. Default is None.
        dt (float, optional): Time duration for each position sample.
        min_fr_bin_treshold_s (float, optional): Minimum time in a bin to consider for firing rate calculation.
        min_spike_treshold (int, optional): Minimum spikes in a bin to consider for calculation.
        n_pixel (int, optional): Number of pixels in the grid for heatmap. Default is 34.
        sigma (float, optional): Standard deviation for Gaussian smoothing kernel. Default is 2.
        kernel_size (int, optional): Size of smoothing kernel. Default is 8.
        n_columns (int, optional): Number of columns for the subplot grid. Default is 8.
        pos_range (tuple, optional): Tuple (min, max) for the xy axis range. Default is None.
        smoothed (bool, optional): Whether to apply smoothing to the heatmap. Default is True.
        threshold_marker (dict, optional): Dictionary of threshold_type (percentile/percent/absolute) and threshold_value for markers. Default is None.
        show_colorbar (bool, optional): Whether to display the colorbar. Default is False.
        color_range (tuple, optional): Tuple (min, max) for the colorbar. Default is None.
        main_title (str, optional): Main title for the figure. Default is ''.
        title (str, optional): Title for individual subplots. Default is ''.
        title_color (str, optional): Color of the title text. Default is 'black'.
        fill_value (float, optional): Value to fill NaN values in the heatmap. Default is 0.
        room_normalization (bool, optional): Whether to normalize the heatmap by room. Default is False.
        minimized_layout (bool, optional): If True, produce minimal layout.
        pad_in_pixels (int, optional): Padding to apply (in pixels). Default is 1.
        ax_in (Axes, optional): Matplotlib Axes object to plot on. If None, a new figure and axes are created.
        save_params (dict): Dictionary containing parameters for saving the plot.

    Returns:
        None
    """
    

    # Get the range of positions if not provided
    if pos_range is None:
        if boundary_points is not None:
            pos_range = (np.min(boundary_points[:,:2]), np.max(boundary_points[:,:2]))
        else:
            pos_range = (np.min(df_data[target_columns].values), np.max(df_data[target_columns].values))

    # # Heatmap of Spike Rate Sum by positions
    # x_grid = np.linspace(pos_range[0], pos_range[1], n_pixel)
    # y_grid = np.linspace(pos_range[0], pos_range[1], n_pixel)

    # sigma = 2
    # kernel_size = 8
    # X_kernel, Y_kernel = np.meshgrid(np.arange(-kernel_size / 2, kernel_size / 2 + 1),
    #                                  np.arange(-kernel_size / 2, kernel_size / 2 + 1))
    # kernel = np.exp(-(X_kernel ** 2 + Y_kernel ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    # kernel = kernel / np.sum(kernel)
    if isinstance(cells_columns, int):
        cells_columns = [f"Cell_{cells_columns}"]
    elif isinstance(cells_columns, list) and all(isinstance(cell,int) for cell in cells_columns):
        cells_columns = [f'Cell_{cell}' for cell in cells_columns]
    elif not isinstance(cells_columns, list):
        cells_columns = [cells_columns]

    if dt is None:
        if 'timestamp' in df_data.columns:
            timestamps = df_data['timestamp'].values
            dt = np.median(np.diff(timestamps))
        else:
            raise ValueError("dt is not provided and 'timestamp' column is not in df_data.")

    n_cells = len(cells_columns)
    room_max_vals = {}
    thresholds = {}
    marker_maps = {}
    # cell_columns = [f'Cell_{cell}' for cell in cells]
    if n_cells < n_columns:
        n_columns = n_cells

    n_rows = n_cells // n_columns + int(n_cells % n_columns > 0)  # Calculate the number of rows needed for subplots
    if not ax_in:
        fig, axs = plt.subplots(n_rows, n_columns, figsize=(2.5 * n_columns, 3 * n_rows), sharex=True, sharey=True)  # Adjust figsize according to the number of cells
        if main_title:
            # Add the main title
            plt.suptitle(main_title, x=0.5, y=0.98, ha='center', fontsize=14)
            # Add space for the title
            plt.subplots_adjust(top=0.85)  # Adjust 'top' to add more space (higher value creates more space)

    for i, cell_column in enumerate(cells_columns):
        # cell_df = df_data[df_data['cell'] == cell]
        # calculate the rate map
        # todo: room_normalization is not implemented right, need to be fixed
        if 'room' in df_data.columns and room_normalization:
            rate_maps = []
            x_grid, y_grid = None, None  # will store these from the first map
            # Group by room and calculate the rate map for each room separately
            for room, group in df_data.groupby('room'):
                cell_rate_map_room, x_grid, y_grid = create_rate_map(group[target_columns].values,
                                                                     group[cell_column].values,
                                                                     dt=dt,
                                                                     min_fr_bin_treshold_s=min_fr_bin_treshold_s,
                                                                     min_spike_treshold=min_spike_treshold,
                                                                     smoothed=smoothed,
                                                                     pos_range=pos_range, fill_value=fill_value,
                                                                     n_pixel=n_pixel, sigma=sigma, kernel_size=kernel_size)

                if map_rooms is not None and boundary_points is not None:
                    room_key = map_rooms['rooms'][room]['index']
                    room_boundary_points = boundary_points[boundary_points[:, -1] == room_key]
                    # Create grid of (x, y) points
                    xv, yv = np.meshgrid(x_grid, y_grid)
                    grid_points = np.c_[xv.ravel(), yv.ravel()]
                    # Create the polygon of the rooms
                    polygon_room = Polygon(room_boundary_points)
                    # Check which points fall inside the polygon
                    mask_inside = np.array([polygon_room.covers(Point(p)) for p in grid_points])
                    mask_inside_2d = mask_inside.reshape(cell_rate_map_room.shape)
                    # Zero out values outside the polygon
                    cell_rate_map_room[~mask_inside_2d] = np.nan


                # Set marker by threshold
                if threshold_marker is not None:
                    threshold = get_threshold_value(cell_rate_map_room, **threshold_marker)
                    thresholds[room] = threshold
                    marker_maps[room] = (cell_rate_map_room >= threshold).astype(int)

                abs_min,abs_max = (np.nanmin(cell_rate_map_room), np.nanmax(cell_rate_map_room))
                # if abs_max == abs_min: show_colorbar=False

                if room_normalization:
                    # Normalize, avoiding divide-by-zero
                    min_pos, max_pos = (0,1) # = ((abs_min,abs_max) - abs_min) / (abs_max-abs_min)
                    vmin,vmax = color_range if color_range else (min_pos, max_pos)
                    room_max_vals[room] = {"color_range": (vmin, vmax), "abs_range":(abs_min, abs_max),"pos_range":(min_pos, max_pos)}
                    normalized_rate_map = (cell_rate_map_room - abs_min) / (abs_max - abs_min) if (abs_max - abs_min) > 0 else cell_rate_map_room
                    rate_maps.append(normalized_rate_map)
                else:
                    vmin,vmax = color_range if color_range else (abs_min,abs_max)
                    room_max_vals[room] = {"color_range": (vmin, vmax), "abs_range":(abs_min, abs_max),"pos_range":(abs_min,abs_max)}
                    rate_maps.append(cell_rate_map_room)


            # Stack into shape (n_maps, height, width)
            if len(rate_maps) > 0:
                stacked_maps = np.stack(rate_maps, axis=0)
            else:
                return fig
            # Sum across all maps, ignoring NaNs
            cell_rate_map = np.nansum(stacked_maps, axis=0)
            # Optionally restore the fill_value where all values were nan
            if np.isnan(fill_value):
                all_nan = np.isnan(stacked_maps).all(axis=0)
            else:
                all_nan = (stacked_maps == fill_value).all(axis=0)
            cell_rate_map[all_nan] = fill_value
        else:

            cell_rate_map, x_grid, y_grid = create_rate_map(df_data[target_columns].values, df_data[cell_column].values,
                                                            dt=dt,
                                                            min_fr_bin_treshold_s=min_fr_bin_treshold_s,
                                                            min_spike_treshold=min_spike_treshold,
                                                            smoothed=smoothed, pos_range=pos_range,
                                                            fill_value=fill_value, n_pixel=n_pixel)

            if map_rooms is not None and boundary_points is not None:
                room_keys = np.unique(boundary_points[:, -1])
                polygon_rooms = []
                for room_key in room_keys:
                    room_boundary_points = boundary_points[boundary_points[:, -1] == room_key]
                    polygon_room = Polygon(room_boundary_points)
                    # Create the polygon of the rooms
                    polygon_rooms.append(polygon_room)

                # Create grid of (x, y) points
                xv, yv = np.meshgrid(x_grid, y_grid)
                grid_points = np.c_[xv.ravel(), yv.ravel()]
                # Check which points fall inside the polygon
                mask_inside = np.zeros(len(grid_points), dtype=bool)
                for polygon in polygon_rooms:
                    mask_inside |= np.array([polygon.covers(Point(p)) for p in grid_points])
                mask_inside_2d = mask_inside.reshape(cell_rate_map.shape)
                # Zero out values outside the polygon
                cell_rate_map[~mask_inside_2d] = 0

            # Set marker by threshold
            if threshold_marker is not None:
                threshold = get_threshold_value(cell_rate_map, **threshold_marker)
                thresholds["All"] = threshold
                marker_maps["All"] = (cell_rate_map >= threshold).astype(int)

            abs_min,abs_max = np.nanmin(cell_rate_map), np.nanmax(cell_rate_map)
            vmin,vmax = color_range if color_range else (abs_min,abs_max)
            room_max_vals["All"] = {"color_range": (vmin, vmax), "abs_range":(abs_min, abs_max),"pos_range":(abs_min,abs_max)}
            if abs_max==0:show_colorbar=False


        if not ax_in:
            row = i // n_columns
            col = i % n_columns
            ax = axs[row, col] if n_rows > 1 else axs[col] if n_columns > 1 else axs
        else:
            ax = ax_in


        vmin, vmax = color_range if color_range else (np.nanmin(cell_rate_map), np.nanmax(cell_rate_map))
        cax = ax.contourf(x_grid, y_grid, cell_rate_map, levels=25, cmap='jet', alpha=0.75, vmin=vmin, vmax=vmax)


        if show_colorbar: #and not ax_in:
            #add color bar
            room_max_vals = {room: room_max_vals[room] for room in sorted(room_max_vals.keys(), reverse=True)}
            add_stacked_colorbars(ax, room_max_vals, list(room_max_vals.keys()), thresholds=thresholds)

        if boundary_points is not None:
            # plot_boundaries(boundary_points*(n_pixel+1)/n_pixel, ax)
            plot_boundaries(boundary_points, ax, shift=(pos_range[1] - pos_range[0])/(2*n_pixel))

        if threshold_marker is not None:
            for room, marker_map in marker_maps.items():
                if np.any(marker_map):
                    ax.contour(x_grid, y_grid, marker_map, levels=[0.5], colors='black', linewidths=1.5)

            # if ax_in: cb = plt.colorbar(cax)
        if not title:
            ax.set_title(f'{cell_column.replace("_", " ")}', fontdict={'fontsize': 14, 'fontweight': 'bold'})
            # ax.set_title(f'Cell #{cells[i]}', fontdict={'fontsize': 14, 'fontweight': 'bold'})
        else:
            if isinstance(title, list):
                ax.set_title(title[i], fontdict={'fontsize': 10, 'fontweight': 'bold', 'color': title_color})
            elif isinstance(title, str):
                if '\n' in title:
                    title_part, subtitle_part = title.split('\n', 1)
                    ax.set_title(f'{title_part}\n', fontdict={'fontsize': 10, 'fontweight': 'bold', 'color': title_color})
                    ax.text(0.5, 1.03, subtitle_part, fontsize=7, ha='center', transform=ax.transAxes,
                            fontdict={'color': 'black', 'family': 'monospace'})  # , fontdict={'family': 'monospace'})
                else:
                    ax.set_title(title, fontdict={'fontsize': 10, 'fontweight': 'bold', 'color': title_color})


        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True)
        if minimized_layout:
            ax.set_axis_off()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
        pad = pad_in_pixels * (pos_range[1] - pos_range[0]) / n_pixel
        ax.set_xlim(pos_range[0] - pad, pos_range[1] + pad)
        ax.set_ylim(pos_range[1] + pad, pos_range[0] - pad)
        ax.set_aspect('equal')

    # hide empty subplots
    for j in range(i + 1, n_rows * n_columns):
        if n_rows > 1:
            axs[j // n_columns, j % n_columns].axis('off')
        else:
            axs[j].axis('off')

    # if not ax_in:
    #     plt.show()
    if ax_in: return ax_in
    else: return fig


def _get_boundary_mask(points, boundary_points):
    """
    Helper function to create a mask for points inside boundary polygon(s).
    
    Parameters:
        points (np.ndarray): Array of points of shape (n_points, 2).
        boundary_points (np.ndarray): Array of boundary points of shape (n_points, 2) or (n_points, 3).
                                      If shape is (n_points, 3), the third column contains room indices.
    
    Returns:
        np.ndarray: 1D boolean mask of shape (n_points,) where True indicates points inside boundary.
    """
    # Check if boundary_points has room indices (3rd column)
    if boundary_points.shape[1] >= 3:
        # Multiple rooms - check if points are in any of the rooms
        unique_rooms = np.unique(boundary_points[:, 2])
        if len(unique_rooms) > 1:
            # Multiple rooms: check if points are inside any room polygon
            mask_inside = np.zeros(len(points), dtype=bool)
            for room_idx in unique_rooms:
                room_boundary = boundary_points[boundary_points[:, 2] == room_idx]
                room_boundary_xy = room_boundary[:, :2]
                polygon = Polygon(room_boundary_xy)
                room_mask = np.array([polygon.covers(Point(p)) for p in points])
                mask_inside |= room_mask
            return mask_inside
        else:
            # Single room, use all boundary points
            boundary_xy = boundary_points[:, :2]
            polygon = Polygon(boundary_xy)
            mask_inside = np.array([polygon.covers(Point(p)) for p in points])
            return mask_inside
    else:
        # No room column, use all boundary points as single polygon
        boundary_xy = boundary_points[:, :2]
        polygon = Polygon(boundary_xy)
        mask_inside = np.array([polygon.covers(Point(p)) for p in points])
        return mask_inside


def _get_boundary_mask_grid(x_grid, y_grid, boundary_points):
    """
    Helper function to create a mask for grid points inside boundary polygon.
    
    Parameters:
        x_grid (np.ndarray): Grid edges for x dimension.
        y_grid (np.ndarray): Grid edges for y dimension.
        boundary_points (np.ndarray): Array of boundary points of shape (n_points, 2) or (n_points, 3).
    
    Returns:
        np.ndarray: 2D boolean mask of shape (len(y_grid)-1, len(x_grid)-1) where True indicates points inside boundary.
    """
    # Create grid of (x, y) points
    x_centers = (x_grid[:-1] + x_grid[1:]) / 2
    y_centers = (y_grid[:-1] + y_grid[1:]) / 2
    xv, yv = np.meshgrid(x_centers, y_centers)
    grid_points = np.c_[xv.ravel(), yv.ravel()]
    
    # Use _get_boundary_mask to check which points are inside
    mask_inside = _get_boundary_mask(grid_points, boundary_points)
    
    # Reshape to 2D
    mask_inside_2d = mask_inside.reshape((len(y_centers), len(x_centers)))
    
    return mask_inside_2d


@save_plot
def plot_rate_map(
    df_data,
    cell_column,
    target_columns=['X', 'Y'],
    config=None,
    n_pixel=34,
    smooth={'enabled': True, 'sigma': 2, 'kernel_size': 8},
    show_colorbar=False,
    title='',
    fill_value=0,
    room_normalization=False,
    per_room=True,
    minimized_layout=False,
    use_normalize_units=True,
    ax_in=None,
    save_params={}
):
    """
    Create and plot a single rate map for a specific cell.
    
    This is a simpler alternative to plot_position_spike_rates for single cell rate maps.
    It handles room merging with normalization options and boundary filtering.
    
    Parameters:
        df_data (DataFrame): DataFrame containing the dataset with positions and cell firing rates.
            Note: Position columns (target_columns) are expected to be in normalized units (0-1).
        cell_column (str): Column name containing firing rates for the cell to plot.
        target_columns (list, optional): List of columns to use as position columns. Default is ['X', 'Y'].
        config (dict, optional): Configuration dictionary. Required for boundary filtering and room handling.
        pos_range (tuple, optional): Tuple (min, max) for the xy axis range. Default is None (inferred from data).
        threshold (dict, optional): Dictionary with 'dt', 'min_fr_bin_treshold_s', 'min_spike_treshold'. Default is all None.
        n_pixel (int, optional): Number of pixels in the grid for rate map. Default is 34.
        smooth (dict, optional): Dictionary with 'enabled' (bool), 'sigma' (float), 'kernel_size' (int) for smoothing. Default is {'enabled': True, 'sigma': 2, 'kernel_size': 8}.
        show_colorbar (bool, optional): Whether to display the colorbar. Default is False.
        title (str, optional): Title for the plot. Default is ''.
        fill_value (float, optional): Value to fill NaN values in the rate map. Default is 0.
        room_normalization (bool, optional): Whether to normalize the rate map by room. Default is False.
        minimized_layout (bool, optional): If True, produce minimal layout. Default is False.
        use_normalize_units (bool, optional): 
            - True: Work in normalized units (0-1). Data stays normalized, boundary points get normalized.
            - False: Work in cm units. Data gets inverse transformed to cm, boundary points stay in cm.
            Default is True. See docs/data_format.md for details.
        ax_in (Axes, optional): Matplotlib Axes object to plot on. If None, a new figure and axes are created. Default is None.
        save_params (dict): Dictionary containing parameters for saving the plot.
    
    Returns:
        matplotlib.figure.Figure: The figure containing the plot.
    """
    # Get threshold parameters from config
    if config is not None:
        preprocessing = config.get('preprocessing', {})
        min_fr_bin_treshold_s = preprocessing.get('min_fr_bin_treshold_s', None)
        min_spike_treshold = preprocessing.get('min_spike_treshold', None)
    else:
        min_fr_bin_treshold_s = None
        min_spike_treshold = None
    
    # Extract smooth parameters
    smoothed = smooth.get('enabled', True)
    sigma = smooth.get('sigma', 2)
    kernel_size = smooth.get('kernel_size', 8)
    
    # Assign room column if missing and config provided
    if 'room' not in df_data.columns:
        if config is not None:
            df_data = assign_room_column(df_data, config, room_column='room')
        else:
            raise ValueError("'room' column not found in df_data and config not provided to assign it.")
    
    # Transform df_data target_columns based on use_normalize_units flag (use a copy)
    # Convention: use_normalize_units=True means work in normalized units, use_normalize_units=False means work in cm
    df_data = df_data.copy()
    if config is not None:
        try:
            scaler = load_position_scaler_from_config(config)
            if scaler is not None and not use_normalize_units:
                # use_normalize_units=False: Inverse transform data from normalized (0-1) to cm
                from utils.helpers import apply_scaler_transform
                xy_data = df_data[target_columns].values.copy()
                xy_data = apply_scaler_transform(xy_data, scaler, reverse=True)
                df_data[target_columns[0]] = xy_data[:, 0]
                df_data[target_columns[1]] = xy_data[:, 1]
        except Exception:
            pass  # Transform is optional
    
    # Get boundary points if config provided
    # Convention: use_normalize_units=True means normalize boundary points, use_normalize_units=False means keep in cm
    boundary_points = None
    if config is not None:
        try:
            boundary_points = get_boundary_points_from_csv(config, use_normalize_units=use_normalize_units)
        except Exception:
            pass  # Boundary points are optional
    
    # Get the range of positions from boundary_points if available, otherwise from data
    pos_range = None
    if pos_range is None:
        if boundary_points is not None:
            pos_range = (np.min(boundary_points[:,:2]), np.max(boundary_points[:,:2]))
        else:
            pos_range = (np.min(df_data[target_columns].values), np.max(df_data[target_columns].values))
    
    # Get dt from timestamps
    if 'timestamp' in df_data.columns:
        timestamps = df_data['timestamp'].values
        dt = np.median(np.diff(timestamps))
    else:
        raise ValueError("dt is not provided and 'timestamp' column is not in df_data.")
    
    # Handle room merging scenarios
    if per_room and 'room' in df_data.columns:
        # Process per room
        if room_normalization:
            # Scenario 1: With room_normalization=True - normalize each room, then merge
            rate_maps = []
            x_grid, y_grid = None, None
            
            # Group by room and calculate the rate map for each room separately
            for room, group in df_data.groupby('room'):
                cell_rate_map_room, x_grid, y_grid = create_rate_map(
                    group[target_columns].values,
                    group[cell_column].values,
                    dt=dt,
                    min_fr_bin_treshold_s=min_fr_bin_treshold_s,
                    min_spike_treshold=min_spike_treshold,
                    smoothed=smoothed,
                    pos_range=pos_range,
                    fill_value=fill_value,
                    n_pixel=n_pixel,
                    sigma=sigma,
                    kernel_size=kernel_size
                )
                
                # Apply boundary filtering if config provided
                if config is not None:
                    try:
                        room_boundary_points = get_boundary_points_from_csv(config, room=room, use_normalize_units=use_normalize_units)
                        mask_inside_2d = _get_boundary_mask_grid(x_grid, y_grid, room_boundary_points)
                        # Zero out values outside the polygon
                        cell_rate_map_room[~mask_inside_2d] = np.nan
                    except Exception:
                        pass  # Boundary filtering is optional
                
                # Normalize per room
                abs_min, abs_max = (np.nanmin(cell_rate_map_room), np.nanmax(cell_rate_map_room))
                normalized_rate_map = (cell_rate_map_room - abs_min) / (abs_max - abs_min) if (abs_max - abs_min) > 0 else cell_rate_map_room
                rate_maps.append(normalized_rate_map)
        
        else:
            # Scenario 2: With room_normalization=False - no normalization, just merge
            rate_maps = []
            x_grid, y_grid = None, None
            
            # Group by room and calculate the rate map for each room separately
            for room, group in df_data.groupby('room'):
                cell_rate_map_room, x_grid, y_grid = create_rate_map(
                    group[target_columns].values,
                    group[cell_column].values,
                    dt=dt,
                    min_fr_bin_treshold_s=min_fr_bin_treshold_s,
                    min_spike_treshold=min_spike_treshold,
                    smoothed=smoothed,
                    pos_range=pos_range,
                    fill_value=fill_value,
                    n_pixel=n_pixel,
                    sigma=sigma,
                    kernel_size=kernel_size
                )
                
                # Apply boundary filtering if config provided
                if config is not None:
                    try:
                        room_boundary_points = get_boundary_points_from_csv(config, room=room, use_normalize_units=use_normalize_units)
                        mask_inside_2d = _get_boundary_mask_grid(x_grid, y_grid, room_boundary_points)
                        cell_rate_map_room[~mask_inside_2d] = np.nan
                    except Exception:
                        pass  # Boundary filtering is optional
                
                rate_maps.append(cell_rate_map_room)
        
        # Stack and merge rate maps
        if len(rate_maps) > 0:
            stacked_maps = np.stack(rate_maps, axis=0)
            cell_rate_map = np.nansum(stacked_maps, axis=0)
            # Optionally restore the fill_value where all values were nan
            if np.isnan(fill_value):
                all_nan = np.isnan(stacked_maps).all(axis=0)
            else:
                all_nan = (stacked_maps == fill_value).all(axis=0)
            cell_rate_map[all_nan] = fill_value
        else:
            # Fallback if no rooms processed
            cell_rate_map, x_grid, y_grid = create_rate_map(
                df_data[target_columns].values,
                df_data[cell_column].values,
                dt=dt,
                min_fr_bin_treshold_s=min_fr_bin_treshold_s,
                min_spike_treshold=min_spike_treshold,
                smoothed=smoothed,
                pos_range=pos_range,
                fill_value=fill_value,
                n_pixel=n_pixel
            )
    else:
        # Scenario 3: per_room=False - process all data without room handling
        cell_rate_map, x_grid, y_grid = create_rate_map(
            df_data[target_columns].values,
            df_data[cell_column].values,
            dt=dt,
            min_fr_bin_treshold_s=min_fr_bin_treshold_s,
            min_spike_treshold=min_spike_treshold,
            smoothed=smoothed,
            pos_range=pos_range,
            fill_value=fill_value,
            n_pixel=n_pixel,
            sigma=sigma,
            kernel_size=kernel_size
        )
        
        # Apply boundary filtering if config provided
        if config is not None:
            try:
                boundary_points = get_boundary_points_from_csv(config, use_normalize_units=use_normalize_units)
                mask_inside_2d = _get_boundary_mask_grid(x_grid, y_grid, boundary_points)
                cell_rate_map[~mask_inside_2d] = np.nan
            except Exception:
                pass  # Boundary filtering is optional
    
    
    # Create axes if needed
    if ax_in is None:
        fig, ax = plt.subplots(figsize=(3, 3))
    else:
        ax = ax_in
        fig = ax.figure
    
    # Plot rate map
    vmin, vmax = np.nanmin(cell_rate_map), np.nanmax(cell_rate_map)
    cax = ax.contourf(x_grid, y_grid, cell_rate_map, levels=25, cmap='jet', alpha=0.75, vmin=vmin, vmax=vmax)
    
    # Add colorbar if requested
    if show_colorbar:
        plt.colorbar(cax, ax=ax)
    
    # Plot boundary points if provided
    if boundary_points is not None:
        plot_boundaries(boundary_points, ax, shift=(pos_range[1] - pos_range[0])/(2*n_pixel))
    
    # Set title
    if title:
        ax.set_title(title, fontdict={'fontsize': 14, 'fontweight': 'bold'})
    else:
        ax.set_title(f'{cell_column.replace("_", " ")}', fontdict={'fontsize': 14, 'fontweight': 'bold'})
    
    # Set labels and layout
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    
    if minimized_layout:
        ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
    
    # Set axis limits
    pad = (pos_range[1] - pos_range[0]) / n_pixel
    ax.set_xlim(pos_range[0] - pad, pos_range[1] + pad)
    ax.set_ylim(pos_range[1] + pad, pos_range[0] - pad)
    ax.set_aspect('equal')
    
    return fig


def create_vector_field(
    positions,  # (N, 2) array
    hd_angles,   # (N,) array
    firing_rates,  # (N,) array
    x_grid,      # Grid edges for x
    y_grid,      # Grid edges for y
    angle_bins,  # Angle bins for HD tuning curve
    dt,
    min_fr_bin_treshold_s=None,
    min_spike_treshold=None,
    smooth_vector_field=False,
    smoothing_sigma=1.0,
    smoothing_kernel_size=5,
    arrow_scale=1.0
):
    """
    Create a vector field for a single room/dataset.
    
    For each spatial bin, computes a head direction tuning curve and uses the Rayleigh Vector
    to determine the preferred direction. The magnitude represents the mean firing rate at that
    preferred direction, making it comparable across spatial bins.
    
    Parameters:
        positions (np.ndarray): Array of shape (N, 2) containing position data.
        hd_angles (np.ndarray): Array of shape (N,) containing head direction angles in degrees.
        firing_rates (np.ndarray): Array of shape (N,) containing firing rates (spike counts).
        x_grid (np.ndarray): Grid edges for x dimension.
        y_grid (np.ndarray): Grid edges for y dimension.
        angle_bins (np.ndarray): Angle bins for head direction tuning curve.
        dt (float): Time duration for each position sample.
        min_fr_bin_treshold_s (float, optional): Minimum time (s) in a bin to consider. Default is None.
        min_spike_treshold (int, optional): Minimum spikes in a bin to consider. Default is None.
        smooth_vector_field (bool, optional): Whether to smooth the vector field. Default is False.
        smoothing_sigma (float, optional): Standard deviation for Gaussian smoothing kernel. Default is 1.0.
        smoothing_kernel_size (int, optional): Size of smoothing kernel (must be odd). Default is 5.
        arrow_scale (float, optional): Scale factor for arrow lengths. Default is 1.0.
    
    Returns:
        tuple: (U, V, magnitudes, x_grid, y_grid) where:
            - U, V: arrays of shape (n_bins_y, n_bins_x) containing unit direction vectors
            - magnitudes: array of shape (n_bins_y, n_bins_x) containing mean firing rates at preferred direction
            - x_grid, y_grid: grid edges (same as input)
    """
    n_bins_x = len(x_grid) - 1
    n_bins_y = len(y_grid) - 1
    
    # Initialize arrays
    U = np.full((n_bins_y, n_bins_x), np.nan, dtype=np.float64)
    V = np.full((n_bins_y, n_bins_x), np.nan, dtype=np.float64)
    magnitudes = np.full((n_bins_y, n_bins_x), np.nan, dtype=np.float64)
    
    # For each spatial bin, compute preferred direction
    for j in range(n_bins_x):
        for k in range(n_bins_y):
            # Filter data points within the bin boundaries
            in_bin = (positions[:, 0] >= x_grid[j]) & (positions[:, 0] < x_grid[j + 1]) & \
                     (positions[:, 1] >= y_grid[k]) & (positions[:, 1] < y_grid[k + 1])
            
            bin_hd = hd_angles[in_bin]
            bin_fr = firing_rates[in_bin]
            
            # Check thresholds
            valid = (~np.isnan(bin_hd)) & (~np.isnan(bin_fr))
            n_valid = np.sum(valid)
            count = n_valid
            total_spikes = np.sum(bin_fr[valid]) if n_valid > 0 else 0
            
            if not (count > 0
                    and ((min_fr_bin_treshold_s is None) or (count * dt >= min_fr_bin_treshold_s))
                    and ((min_spike_treshold is None) or (total_spikes >= min_spike_treshold))):
                magnitudes[k, j] = 0.0
                U[k, j] = 0.0
                V[k, j] = 0.0
                continue
            
            bin_hd_valid = bin_hd[valid]
            bin_fr_valid = bin_fr[valid]
            
            # Create head direction tuning curve
            try:
                tuning_curve, angles_out = create_hd_rate_map(
                    bin_hd_valid,
                    bin_fr_valid,
                    angles=angle_bins,
                    smoothed=True,
                    sigma_ang=1.0,
                    kernel_size_ang=5,
                    wrap_angles=False,
                    return_count=False
                )
                
                # Check if tuning curve has valid data
                valid_tc_mask = ~np.isnan(tuning_curve)
                tuning_curve_clean = tuning_curve[valid_tc_mask]
                angles_out_clean = angles_out[valid_tc_mask]
                
                if len(tuning_curve_clean) == 0 or np.sum(np.abs(tuning_curve_clean)) == 0:
                    magnitudes[k, j] = 0.0
                    U[k, j] = 0.0
                    V[k, j] = 0.0
                    continue
                
                # Calculate Rayleigh vector to get preferred direction
                rayleigh_radius, rayleigh_angle = calculate_rayleigh_vector(tuning_curve_clean, angles_out_clean)
                
                if np.isnan(rayleigh_radius) or np.isnan(rayleigh_angle):
                    magnitudes[k, j] = 0.0
                    U[k, j] = 0.0
                    V[k, j] = 0.0
                    continue
                
                # Find the mean firing rate at the preferred direction
                # Find nearest angle bin to preferred direction (handle circular wrap-around)
                diff = ((angles_out_clean - rayleigh_angle + 180) % 360) - 180   # in [-180, 180)
                nearest_idx = np.argmin(np.abs(diff))
                mean_firing_rate_at_preferred = tuning_curve_clean[nearest_idx]
                
                # Use mean firing rate as magnitude (instead of rayleigh_radius)
                # This makes magnitudes comparable across spatial bins
                if np.isnan(mean_firing_rate_at_preferred) or mean_firing_rate_at_preferred < 0:
                    magnitudes[k, j] = 0.0
                else:
                    magnitudes[k, j] = mean_firing_rate_at_preferred
                
                # Convert angle to radians and compute U, V components
                angle_rad = np.deg2rad(rayleigh_angle)
                unit_length = 1.0 * arrow_scale
                U[k, j] = unit_length * np.cos(angle_rad)
                V[k, j] = unit_length * np.sin(angle_rad)
                
            except Exception:
                magnitudes[k, j] = 0.0
                U[k, j] = 0.0
                V[k, j] = 0.0
                continue
    
    # Apply smoothing if requested
    if smooth_vector_field and smoothing_kernel_size > 0 and smoothing_sigma > 0:
        # Ensure kernel size is odd
        if smoothing_kernel_size % 2 == 0:
            smoothing_kernel_size += 1
        
        # Create Gaussian kernel
        pad = smoothing_kernel_size // 2
        x = np.arange(-pad, pad + 1, dtype=float)
        y = np.arange(-pad, pad + 1, dtype=float)
        X_kernel, Y_kernel = np.meshgrid(x, y)
        kernel = np.exp(-(X_kernel**2 + Y_kernel**2) / (2 * smoothing_sigma**2))
        kernel /= np.sum(kernel)
        
        # Handle NaN values: create masks and replace with 0 for smoothing, then restore
        U_nan_mask = np.isnan(U)
        V_nan_mask = np.isnan(V)
        magnitudes_nan_mask = np.isnan(magnitudes)
        
        # Replace NaN with 0 for smoothing
        U_smooth = np.nan_to_num(U, nan=0.0)
        V_smooth = np.nan_to_num(V, nan=0.0)
        magnitudes_smooth = np.nan_to_num(magnitudes, nan=0.0)
        
        # Pad arrays with edge values to handle boundaries
        U_padded = np.pad(U_smooth, pad, mode='edge')
        V_padded = np.pad(V_smooth, pad, mode='edge')
        magnitudes_padded = np.pad(magnitudes_smooth, pad, mode='edge')
        
        # Apply convolution
        from scipy.signal import convolve2d
        U_smoothed = convolve2d(U_padded, kernel, mode='same')
        V_smoothed = convolve2d(V_padded, kernel, mode='same')
        magnitudes_smoothed = convolve2d(magnitudes_padded, kernel, mode='same')
        
        # Remove padding
        U_smoothed = U_smoothed[pad:-pad, pad:-pad] if pad > 0 else U_smoothed
        V_smoothed = V_smoothed[pad:-pad, pad:-pad] if pad > 0 else V_smoothed
        magnitudes_smoothed = magnitudes_smoothed[pad:-pad, pad:-pad] if pad > 0 else magnitudes_smoothed
        
        # Restore NaN values where they were originally
        U_smoothed[U_nan_mask] = np.nan
        V_smoothed[V_nan_mask] = np.nan
        magnitudes_smoothed[magnitudes_nan_mask] = np.nan
        
        # For vector fields, we need to normalize directions while preserving magnitudes
        # Compute magnitude of smoothed direction vectors
        smoothed_dir_magnitude = np.sqrt(U_smoothed**2 + V_smoothed**2)
        # Normalize direction vectors to unit length, but preserve zero vectors
        non_zero_mask = smoothed_dir_magnitude > 1e-10
        U_smoothed[non_zero_mask] = U_smoothed[non_zero_mask] / smoothed_dir_magnitude[non_zero_mask]
        V_smoothed[non_zero_mask] = V_smoothed[non_zero_mask] / smoothed_dir_magnitude[non_zero_mask]
        
        # Use smoothed values
        U = U_smoothed
        V = V_smoothed
        magnitudes = magnitudes_smoothed
    
    return U, V, magnitudes, x_grid, y_grid


@save_plot
def plot_hd_vector_field(
    df_data, 
    cell_column, 
    target_columns=['X', 'Y'],
    config=None,
    rooms=None,
    n_pixel=10,
    n_angles=36,
    show_rate_map_background=True,
    smooth={'enabled': False, 'sigma': 1.0, 'kernel_size': 5},
    arrow_scale=1.0,
    arrow_color='black',
    title='',
    room_normalization=True,
    per_room=True,
    use_normalize_units=True,
    minimized_layout=False,
    ax_in=None,
    save_params={}
):
    """
    Plot a vector field representation showing the preferred firing direction at each spatial position for a specific cell.
    
    For each spatial bin, computes a head direction tuning curve and uses the Rayleigh Vector to determine
    the preferred direction. The arrow length represents the mean firing rate at that preferred direction,
    making firing rates comparable across spatial bins. The vectors are displayed as arrows on a spatial grid,
    optionally overlaid on a rate map background.
    
    Parameters:
        df_data (DataFrame): DataFrame containing the dataset with positions, head directions, and cell firing rates.
            Note: Position columns (target_columns) are expected to be in normalized units (0-1).
        cell_column (str): Column name containing firing rates for the cell to plot.
        target_columns (list, optional): List of columns to use as position columns. Default is ['X', 'Y'].
        config (dict, optional): Configuration dictionary. Required for boundary filtering and room handling.
        rooms (list, optional): List of room names to filter data (e.g., ['A', 'B']). If None, uses rooms from df_data. Default is None.
        n_pixel (int, optional): Number of pixels in the grid for spatial binning. Default is 10.
        n_angles (int, optional): Number of angular bins for head direction tuning curve. Default is 36.
        show_rate_map_background (bool, optional): Whether to show rate map as background. Default is True.
        smooth (dict, optional): Dictionary with 'enabled' (bool), 'sigma' (float), 'kernel_size' (int) for smoothing. Default is {'enabled': False, 'sigma': 1.0, 'kernel_size': 5}.
        arrow_scale (float, optional): Scale factor for arrow lengths. Default is 1.0.
        arrow_color (str, optional): Color of the arrows. Default is 'black'.
        title (str, optional): Title for the plot. If empty, extracts cell number from cell_column (e.g., "Cell_4" -> "Cell 4"). Default is ''.
        room_normalization (bool, optional): Whether to normalize magnitudes per room independently. Default is True.
        per_room (bool, optional): Whether to process each room separately. Default is True.
        use_normalize_units (bool, optional): 
            - True: Work in normalized units (0-1). Data stays normalized, boundary points get normalized.
            - False: Work in cm units. Data gets inverse transformed to cm, boundary points stay in cm.
            Default is True. See docs/data_format.md for details.
        minimized_layout (bool, optional): Whether to use minimized layout (smaller figure, no labels/grid). Default is False.
        ax_in (Axes, optional): Matplotlib Axes object to plot on. If None, a new figure and axes are created. Default is None.
        save_params (dict): Dictionary containing parameters for saving the plot.
    
    Returns:
        matplotlib.figure.Figure: The figure containing the plot.
    """
    # Get threshold parameters from config
    if config is not None:
        preprocessing = config.get('preprocessing', {})
        min_fr_bin_treshold_s = preprocessing.get('min_fr_bin_treshold_s', None)
        min_spike_treshold = preprocessing.get('min_spike_treshold', None)
        rate_map_n_pixel = preprocessing.get('n_pixel', 34)
    else:
        min_fr_bin_treshold_s = None
        min_spike_treshold = None
        rate_map_n_pixel = 34
    
    # Extract smooth parameters
    smooth_vector_field = smooth.get('enabled', False)
    smoothing_sigma = smooth.get('sigma', 1.0)
    smoothing_kernel_size = smooth.get('kernel_size', 5)
    
    # Assign room column if missing and config provided
    if 'room' not in df_data.columns:
        if config is not None:
            df_data = assign_room_column(df_data, config, room_column='room')
        else:
            raise ValueError("'room' column not found in df_data and config not provided to assign it.")
    
    # Load scaler for transformation if needed
    scaler = None
    if config is not None:
        try:
            from utils.metrics import load_position_scaler_from_config as load_scaler
            scaler = load_scaler(config)
        except Exception:
            pass  # Scaler loading is optional
    
    # Save original df_data before transformation (for plot_rate_map which handles its own transformation)
    df_data_original = df_data.copy()
    
    # Transform df_data positions based on use_normalize_units flag (use a copy)
    # Convention: use_normalize_units=True means work in normalized units, use_normalize_units=False means work in cm
    df_data = df_data.copy()
    if scaler is not None:
        try:
            xy_data = df_data[target_columns].values.copy()
            nan_mask = np.isnan(xy_data).any(axis=1)
            if not nan_mask.all():
                valid_xy = xy_data[~nan_mask]
                if len(valid_xy) > 0:
                    if not use_normalize_units:
                        # use_normalize_units=False: Inverse transform data from normalized (0-1) to cm
                        from utils.helpers import apply_scaler_transform
                        valid_transformed = apply_scaler_transform(valid_xy, scaler, reverse=True)
                        xy_data[~nan_mask] = valid_transformed
                        df_data[target_columns[0]] = xy_data[:, 0]
                        df_data[target_columns[1]] = xy_data[:, 1]
                        
                        # Debug: log transformation
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.debug(f"Inverse transformed positions to cm: range [{np.min(valid_transformed):.3f}, {np.max(valid_transformed):.3f}]")
                    else:
                        # use_normalize_units=True: Keep data as-is (already normalized)
                        pass
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to transform positions: {e}")
            pass  # Transform is optional
    
    # Get boundary points if config provided
    # Convention: use_normalize_units=True means normalize boundary points, use_normalize_units=False means keep in cm
    boundary_points = None
    if config is not None:
        try:
            boundary_points = get_boundary_points_from_csv(config, use_normalize_units=use_normalize_units)
        except Exception:
            pass  # Boundary points are optional
    
    # Get the range of positions from boundary_points if available, otherwise from data
    pos_range = None
    if boundary_points is not None:
        pos_range = (np.min(boundary_points[:,:2]), np.max(boundary_points[:,:2]))
    else:
        pos_range = (np.min(df_data[target_columns].values), np.max(df_data[target_columns].values))
    
    # Setup spatial grid
    x_grid = np.linspace(pos_range[0], pos_range[1], n_pixel, dtype=np.float64)
    y_grid = np.linspace(pos_range[0], pos_range[1], n_pixel, dtype=np.float64)
    
    # Get dt if not provided
    if 'timestamp' in df_data.columns:
        timestamps = df_data['timestamp'].values
        dt = np.median(np.diff(timestamps))
    else:
        raise ValueError("dt is not provided and 'timestamp' column is not in df_data.")
    
    # Check for required columns
    if 'HD' not in df_data.columns:
        raise ValueError("'HD' column (head direction) is required in df_data.")
    if cell_column not in df_data.columns:
        raise ValueError(f"Cell column '{cell_column}' not found in df_data.")
    
    # Create angle bins for head direction tuning curve
    angle_bins = np.linspace(0, 360, n_angles, endpoint=False)
    
    # Initialize arrays for vector field
    # We'll compute for bins, so shape is (n_pixel-1, n_pixel-1)
    n_bins_x = len(x_grid) - 1
    n_bins_y = len(y_grid) - 1
    
    # Process vector field per room or all data together
    if per_room and 'room' in df_data.columns:
        # Determine rooms to process - get from room column unique values
        if rooms is None:
            rooms = df_data['room'].unique().tolist()
        elif isinstance(rooms, str):
            rooms = [rooms]
        
        if len(rooms) == 0:
            raise ValueError("No rooms found in data or rooms list is empty.")
        # Process each room separately
        room_vector_fields = {}
        
        for room in rooms:
            # Get room-specific data
            df_room = df_data[df_data['room'] == room].copy()
            
            if len(df_room) == 0:
                continue
            
            # Get room-specific data
            positions_room = df_room[target_columns].values
            hd_angles_room = df_room['HD'].values
            firing_rates_room = df_room[cell_column].values
            
            # Use create_vector_field helper to compute vector field for this room
            U_room, V_room, magnitudes_room, _, _ = create_vector_field(
                positions_room,
                hd_angles_room,
                firing_rates_room,
                x_grid,
                y_grid,
                angle_bins,
                dt,
                min_fr_bin_treshold_s=min_fr_bin_treshold_s,
                min_spike_treshold=min_spike_treshold,
                smooth_vector_field=smooth_vector_field,
                smoothing_sigma=smoothing_sigma,
                smoothing_kernel_size=smoothing_kernel_size,
                arrow_scale=arrow_scale
            )
            
            # Filter vectors to only include those whose centers are within the room boundary
            if config is not None:
                try:
                    room_boundary_points = get_boundary_points_from_csv(config, room=room, use_normalize_units=use_normalize_units)
                    room_mask_2d = _get_boundary_mask_grid(x_grid, y_grid, room_boundary_points)
                    # Set vectors outside room boundary to NaN
                    U_room[~room_mask_2d] = np.nan
                    V_room[~room_mask_2d] = np.nan
                    magnitudes_room[~room_mask_2d] = np.nan
                except Exception:
                    pass  # Boundary filtering is optional
            
            # Normalize magnitudes per room if room_normalization is True
            if room_normalization:
                abs_min = np.nanmin(magnitudes_room)
                abs_max = np.nanmax(magnitudes_room)
                
                if abs_max > abs_min:
                    # Normalize to [0, 1] range
                    magnitudes_room_normalized = (magnitudes_room - abs_min) / (abs_max - abs_min)
                else:
                    magnitudes_room_normalized = magnitudes_room
                
                # Set NaN values in normalized magnitudes to NaN (preserve room boundary filtering)
                magnitudes_room_normalized[np.isnan(magnitudes_room)] = np.nan
            else:
                # Keep raw magnitudes (will be normalized globally after merging)
                magnitudes_room_normalized = magnitudes_room
            
            room_vector_fields[room] = {
                'U': U_room,
                'V': V_room,
                'magnitudes': magnitudes_room,
                'magnitudes_normalized': magnitudes_room_normalized
            }
        
        # Merge vector fields from all rooms
        # Initialize merged arrays with fill_value (np.nan)
        fill_value = np.nan
        U = np.full((n_bins_y, n_bins_x), fill_value, dtype=np.float64)
        V = np.full((n_bins_y, n_bins_x), fill_value, dtype=np.float64)
        magnitudes = np.full((n_bins_y, n_bins_x), fill_value, dtype=np.float64)
        
        # Combine data from all rooms (use normalized magnitudes)
        for room, room_data in room_vector_fields.items():
            room_mask = ~np.isnan(room_data['magnitudes_normalized'])
            # Where we have data from this room, use it (overwrites previous room data if overlap)
            U[room_mask] = room_data['U'][room_mask]
            V[room_mask] = room_data['V'][room_mask]
            magnitudes[room_mask] = room_data['magnitudes_normalized'][room_mask]
        
        # When room_normalization=True, magnitudes are already normalized per room
        # When room_normalization=False, we need to normalize globally across all rooms
        if not room_normalization:
            # Normalize merged magnitudes globally (across all rooms)
            abs_min_merged = np.nanmin(magnitudes)
            abs_max_merged = np.nanmax(magnitudes)
            if abs_max_merged > abs_min_merged:
                valid_mask = ~np.isnan(magnitudes)
                magnitudes[valid_mask] = (magnitudes[valid_mask] - abs_min_merged) / (abs_max_merged - abs_min_merged)
        
        # Set counters for logging (approximate)
        bins_with_sufficient_samples = np.sum(~np.isnan(magnitudes))
        bins_with_insufficient_samples = np.sum(np.isnan(magnitudes))
        bins_with_errors = 0
        total_bins = n_bins_x * n_bins_y
    
    # Create meshgrid for quiver plot (bin centers)
    # Create centers for the bins - these are the exact centers of each spatial bin
    x_centers = (x_grid[:-1] + x_grid[1:]) / 2
    y_centers = (y_grid[:-1] + y_grid[1:]) / 2
    
    # Create meshgrid matching U, V shape: (n_bins_y, n_bins_x)
    # For U[k, j] and V[k, j] (row k, column j), we want:
    # X[k, j] = x_centers[j] (x position of column j)
    # Y[k, j] = y_centers[k] (y position of row k)
    # Default indexing='xy' gives: X has shape (len(y_centers), len(x_centers))
    # where X[k, j] = x_centers[j] and Y[k, j] = y_centers[k] - this is correct!
    X, Y = np.meshgrid(x_centers, y_centers, indexing='xy')
    
    # U, V, and magnitudes have shape (n_bins_y, n_bins_x)
    # X, Y now have matching shape where X[k,j] = x_centers[j], Y[k,j] = y_centers[k]
    
    # Create axes if needed
    if ax_in is None:
        figsize = (5, 5) if not minimized_layout else (3, 3)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = ax_in
        fig = ax.figure
    
    # Optionally show rate map background using plot_rate_map
    if show_rate_map_background:
        # Create a temporary single-cell dataframe for plot_rate_map
        # Use original df_data (before transformation) so plot_rate_map can handle transformation itself
        # Include timestamp column if it exists
        required_cols = target_columns + [cell_column]
        if 'timestamp' in df_data_original.columns:
            required_cols.append('timestamp')
        if 'room' in df_data_original.columns:
            required_cols.append('room')
        df_single_cell = df_data_original[required_cols].copy()
        
        # Call plot_rate_map to plot the background (it will plot on ax if provided)
        # plot_rate_map handles its own transformation based on use_normalize_units
        plot_rate_map(
            df_single_cell,
            cell_column,
            target_columns=target_columns,
            config=config,
            n_pixel=rate_map_n_pixel,
            smooth={
                'enabled': True,
                'sigma': 2,
                'kernel_size': 8
            },
            show_colorbar=False,
            title='',  # No title for background
            fill_value=np.nan,
            room_normalization=room_normalization,
            per_room=per_room,
            minimized_layout=False,
            use_normalize_units=use_normalize_units,
            ax_in=ax,  # Plot on the same axes
            save_params={}  # Don't save separately
        )
    
    # Plot boundary points if provided
    if boundary_points is not None:
        plot_boundaries(boundary_points, ax, shift=(pos_range[1] - pos_range[0])/(2*n_pixel))
    
    # Log summary information
    logger = logging.getLogger(__name__)
    threshold_str = f"count*dt>={min_fr_bin_treshold_s}s" if min_fr_bin_treshold_s is not None else ""
    threshold_str += " and " if threshold_str and min_spike_treshold is not None else ""
    threshold_str += f"spikes>={min_spike_treshold}" if min_spike_treshold is not None else ""
    threshold_str = threshold_str or "any data"
    logger.info(f"Vector field computation: {bins_with_sufficient_samples}/{total_bins} bins meet thresholds ({threshold_str}) (grid: {n_pixel}x{n_pixel})")
    
    # Plot vector field using quiver with length representing magnitude
    # Replace NaN with 0 for plotting (bins with no data)
    U_plot = np.nan_to_num(U, nan=0.0)
    V_plot = np.nan_to_num(V, nan=0.0)
    magnitudes_plot = np.nan_to_num(magnitudes, nan=0.0)
    
    # Calculate bin size for scaling
    bin_size = (pos_range[1] - pos_range[0]) / n_pixel
    
    # Normalize magnitudes so max arrow length equals bin size
    # Scale U and V components by normalized magnitude
    if np.max(magnitudes_plot) > 0:
        # Normalize magnitudes to [0, 1] range
        magnitudes_normalized = magnitudes_plot / np.max(magnitudes_plot)
        # Scale arrow length: max length = bin_size
        # U and V are unit vectors (from cos/sin), so multiply by magnitude * bin_size
        U_scaled = U_plot * magnitudes_normalized * bin_size
        V_scaled = V_plot * magnitudes_normalized * bin_size
    else:
        # All magnitudes are zero
        U_scaled = U_plot
        V_scaled = V_plot
    
    # Quiver draws arrows starting at (X, Y). To center arrows at bin centers,
    # we need to offset the start position by half the arrow length in the opposite direction
    # Offset positions so arrows are centered at bin centers
    X_centered = X - U_scaled / 2
    Y_centered = Y - V_scaled / 2
    
    quiver = ax.quiver(
        X_centered, Y_centered,
        U_scaled, V_scaled,
        color=arrow_color,  # Use single color (black)
        angles='xy',
        scale_units='xy',
        scale=1.0,  # Scale of 1 means arrows are drawn at their actual size
        width=0.003,
        headwidth=3,
        headlength=3,
        headaxislength=2.5,
        minlength=0.0  # Allow zero-length vectors
    )
    
    # Set title - extract cell number from cell_column (e.g., "Cell_4" -> "Cell 4")
    if not title:
        # Extract number from cell_column (handles "Cell_4", "Cell_42", etc.)
        import re
        cell_match = re.search(r'(\d+)', cell_column)
        if cell_match:
            cell_num = cell_match.group(1)
            title = f'Cell {cell_num}'
        else:
            # Fallback if no number found
            title = cell_column.replace("_", " ")
    ax.set_title(title, fontdict={'fontsize': 14, 'fontweight': 'bold'})
    
    # Set axis labels and limits
    if not minimized_layout:
        # Add 'cm' unit if we're NOT using normalized units (coordinates are in cm)
        # use_normalize_units=True means normalized (0-1), use_normalize_units=False means cm
        unit_label = ' (cm)' if not use_normalize_units else ''
        ax.set_xlabel(f'X{unit_label}')
        ax.set_ylabel(f'Y{unit_label}')
    pad = (pos_range[1] - pos_range[0]) / (2 * n_pixel)
    ax.set_xlim(pos_range[0] - pad, pos_range[1] + pad)
    ax.set_ylim(pos_range[1] + pad, pos_range[0] - pad)
    ax.set_aspect('equal')
    if not minimized_layout:
        ax.grid(True, alpha=0.3)
    else:
        ax.grid(False)
    
    return fig


def add_stacked_colorbars(ax, norm_data, rooms, cmap='jet', spacing=0.15,thresholds=None):
    """
    Add stacked colorbars aligned to a reference Axes object, with manual control over height and vertical alignment.

    Parameters:
        ax (Axes): Reference axes to anchor colorbars to.
        norm_data (dict):
        rooms (list): Ordered list of room names.
        cmap (str): Colormap name.
        width (float): Width of colorbar as a fraction of figure width.
        x_offset (float): Horizontal offset from the right edge of the axis (figure fraction).
        spacing (float): Vertical spacing between bars (figure fraction).
        height_scale (float): Fraction of the axis height to use (e.g., 0.9 for 90% height).
        bottom_offset (float): Offset from the bottom of the axis (figure fraction).
    """
    ax_pos = ax.get_position()
    n_rooms = len(rooms)

    # Step 1: Create dummy colorbar to get ideal position
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    dummy_cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.01)

    # Step 2: Get position of that dummy bar
    cbar_pos = dummy_cbar.ax.get_position()
    dummy_cbar.ax.remove()

    # Step 3: Split it vertically for stacked bars
    total_bar_height = cbar_pos.height
    spacing = total_bar_height * spacing / n_rooms
    total_spacing = (n_rooms - 1) * spacing
    bar_height = (total_bar_height - total_spacing) / n_rooms
    left = cbar_pos.x0
    width = cbar_pos.width


    for i, room in enumerate(rooms):
        color_range, abs_range, pos_range = norm_data[room].values()
        vmin, vmax = color_range
        abs_min, abs_max = abs_range
        pos_min, pos_max = pos_range

        bottom = cbar_pos.y0 + i * (bar_height + spacing)
        cbar_ax = ax.figure.add_axes([left, bottom, width, bar_height])

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        cb = plt.colorbar(sm, cax=cbar_ax)

        abs_thresh = None
        pos_thresh = None
        if thresholds is not None and room in thresholds:
            thresh_val = thresholds[room]
            abs_thresh = thresh_val

            pos_thresh = (thresh_val - abs_min) / (abs_max - abs_min) * (pos_max - pos_min) + pos_min if (pos_max - pos_min) > 0 else 0

            # vmin_, vmax_ = sm.get_clim()

            # if np.isfinite(thresh_val) and abs_min <= thresh_val <= abs_max:

            if np.isfinite(pos_thresh) and vmin <= pos_thresh <= vmax:
                cb.ax.hlines(y=pos_thresh,xmin=0.05,xmax=0.95,color='black',linewidth=0.5,linestyle='-',transform=cb.ax.transAxes,zorder=5)
            else:
                pos_thresh = abs_thresh = None

        if np.isfinite(pos_min) and vmin <= pos_min <= vmax:
            cb.ax.hlines(y=pos_min,xmin=0.05,xmax=0.95,color='blue',linewidth=0.5,linestyle='-',transform=cb.ax.transAxes,zorder=5)
        if np.isfinite(pos_max) and vmin <= pos_min <= vmax:
            cb.ax.hlines(y=pos_max,xmin=0.05,xmax=0.95,color='red',linewidth=0.5,linestyle='-',transform=cb.ax.transAxes,zorder=5)

        def format_tick(value):
            if math.isinf(value):
                return '∞' if value > 0 else '−∞'
            return f'{value:^3.1f}' if value < 10 else f'{round(value):^3.0f}'

        # Start with base tick values and labels as lists
        tick_values = [vmin, vmax, pos_min, pos_max]
        tick_labels = [format_tick(vmin), format_tick(vmax), format_tick(abs_min), format_tick(abs_max)]

        # Optionally add threshold
        if pos_thresh is not None and abs_thresh is not None:
            tick_values.append(pos_thresh)
            tick_labels.append(format_tick(abs_thresh))

        # Optionally add room name as text label (instead of tick)
        if room != "All":
            center_y = (vmax + vmin) / 2
            cb.ax.text(0.5, (center_y - vmin) / (vmax - vmin), room,
                       ha='center', va='center', transform=cb.ax.transAxes,
                       fontsize=6, fontfamily='Arial', color='black')

        # Convert to NumPy arrays for processing
        tick_values = np.array(tick_values)
        tick_labels = np.array(tick_labels)

        # Remove duplicates (keep last occurrence)
        _, unique_last_indices = np.unique(tick_values[::-1], return_index=True)
        keep_indices = len(tick_values) - 1 - unique_last_indices

        # Sort by tick value
        sorted_indices = np.argsort(tick_values[keep_indices])

        # Apply filtering and sorting
        tick_values = tick_values[keep_indices][sorted_indices]
        tick_labels = tick_labels[keep_indices][sorted_indices]

        # Set to colorbar
        cb.set_ticks(tick_values)
        cb.set_ticklabels(tick_labels)


        # cb.ax.tick_params(
        #     labelsize=8,
        #     length=0,
        #     pad=-5,
        #     direction='in'
        # )
        cb.ax.tick_params(
            labelsize=6,
            length=3,
            width=0.5,
            pad=1,
            direction='out'
        )

        for collection in cb.ax.collections:
            collection.set_zorder(0)

        for label in cb.ax.get_yticklabels():
            label.set_zorder(10)
            label.set_clip_on(False)
            label.set_fontsize(6)
            label.set_fontfamily('Arial')




@save_plot
def plot_rate_map_comparison(df_data_cell=None, df_data_cell_pred=None, cell_column=None, target_columns=None, pred_target_columns=None, threshold_marker=None, title='Rate Map Comparison', boundary_points=None, pos_range=None, smoothed=True, save_params={}):
    """
    Plot and save the rate map comparison between real and predicted data.

    Parameters:
        df_data_cell (DataFrame): DataFrame containing the real data.
        df_data_cell_pred (DataFrame): DataFrame containing the predicted data.
        cell_column (str): Column name for the cell data.
        target_columns (list): List of columns to use as target columns for the heatmap.
        pred_target_columns (list, optional): List of columns to use as target columns for the predicted data. Default is None.
        title (str): Title for the plot. Default is 'Rate Map Comparison'.
        boundary_points (np.ndarray, optional): Array of boundary points to overlay on the heatmaps. Default is None.
        pos_range (tuple, optional): Tuple (min, max) for the xy axis range. Default is None.
        smoothed (bool, optional): Whether to apply smoothing to the heatmap. Default is True.
        save_params (dict): Dictionary containing parameters for saving the plot.
    """

    if df_data_cell is None:
        raise ValueError("df_data_cell cannot be None")
    elif target_columns is None:
        raise ValueError("target_columns cannot be None")
    elif df_data_cell_pred is None and pred_target_columns is None:
        raise ValueError("Either df_data_cell_pred or pred_target_columns must be provided")

    if df_data_cell_pred is None:
        df_data_cell_pred = df_data_cell

    if pred_target_columns is None:
        pred_target_columns = target_columns


    fig, (ax_real, ax_pred) = plt.subplots(1, 2, figsize=(8, 7))
    fig.suptitle(title, fontsize=14, ha='center', y=0.98)

    # Plot the rate maps
    plot_position_spike_rates(df_data_cell, cell_column, target_columns=target_columns, boundary_points=boundary_points,
                              pos_range=pos_range, smoothed=smoothed,threshold_marker=threshold_marker, ax_in=ax_real, title='Real Rate Map')
    plot_position_spike_rates(df_data_cell_pred, cell_column, target_columns=pred_target_columns,
                              boundary_points=boundary_points, smoothed=smoothed, pos_range=pos_range, threshold_marker=threshold_marker, ax_in=ax_pred, title='Pred Rate Map')




    return fig


@save_plot
def plot_full_cell_rate_map_comparison(df_data_cell, cell_column, target_columns, pred_target_columns,
                                       threshold_marker=None, title='Rate Map Comparison', boundary_points=None,
                                       pos_range=None, smoothed=True, green_threshold=5, red_threshold=10,
                                       rooms_to_indices=None,
                                       room_normalization=True,
                                       show_colorbar=False,
                                       save_params={}):
    """
    Plot and save the rate map comparison between real and predicted data for all rooms together.

    Parameters:
        df_data_cell (DataFrame): DataFrame containing the real data.
        cell_column (str): Column name for the cell data.
        target_columns (list): List of columns to use as target columns for the heatmap.
        pred_target_columns (dict): Dictionary mapping room names to predicted target columns.
        threshold_marker (float, optional): Threshold marker for the rate map. Default is None.
        title (str): Title for the plot. Default is 'Rate Map Comparison'.
        boundary_points (np.ndarray, optional): Array of boundary points to overlay on the heatmaps. Default is None.
        pos_range (tuple, optional): Tuple (min, max) for the xy axis range. Default is None.
        smoothed (bool, optional): Whether to apply smoothing to the heatmap. Default is True.
        green_threshold (float, optional): Threshold for green color coding. Default is 5.
        red_threshold (float, optional): Threshold for red color coding. Default is 10.
        save_params (dict): Dictionary containing parameters for saving the plot.

    """

    rooms = pred_target_columns.keys()
    n_rooms = len(rooms)
    cell = cell_column.split('_')[1]

    # create grid of n_rooms by n_rooms
    fig, axs = plt.subplots(n_rooms, n_rooms + 1, figsize=(3 * (n_rooms + 1), 3 * n_rooms), sharex=True, sharey=True)
    fig.suptitle(title, fontsize=14, ha='center', y=0.98)

    for i, row_room in enumerate(rooms):

        ax = axs[i, 0] if n_rooms > 1 else axs[0]
        df_row_data_cell = df_data_cell[df_data_cell['room'] == row_room]

        # Set Title
        ax_title = f"Rate Map Room {row_room}"

        # Plot the rate maps
        plot_position_spike_rates(df_row_data_cell, cell_column, target_columns=target_columns,
                                  boundary_points=boundary_points,
                                  pos_range=pos_range, smoothed=smoothed,
                                  rooms_to_indices=rooms_to_indices,
                                  room_normalization=room_normalization,
                                  show_colorbar=show_colorbar,
                                  threshold_marker=threshold_marker,
                                  ax_in=ax, title=ax_title)

        real_rate_map, _, _ = create_rate_map(df_row_data_cell[target_columns].values,
                                              df_row_data_cell[cell_column].values,
                                              smoothed=smoothed, fill_value=0.0, pos_range=pos_range)

        for j, col_room in enumerate(rooms):

            ax = axs[i, j + 1] if n_rooms > 1 else axs[j + 1]
            df_col_data_cell = df_data_cell[df_data_cell['room'] == col_room]

            # if i != j:
            pred_rate_map, _, _ = create_rate_map(df_col_data_cell[pred_target_columns[row_room]].values,
                                                  df_col_data_cell[cell_column].values,
                                                  smoothed=smoothed, fill_value=0.0, pos_range=pos_range)

            if not np.any(real_rate_map > 0) or not np.any(pred_rate_map > 0):
                continue

            # filter the rate maps by threshold in precent
            if threshold_marker is not None:
                for rate_map in [real_rate_map, pred_rate_map]:
                    if np.any(rate_map > 0):
                        # get trheshold value
                        threshold_value = get_threshold_value(rate_map, **threshold_marker)
                        rate_map[rate_map < threshold_value] = 0

            # Calculate metrics
            metrics = calculate_rate_map_stats(real_rate_map, pred_rate_map)
            rooms_to_indices_temp = rooms_to_indices.copy()

            rooms_to_indices_temp['room_indices'] = {room:rooms_to_indices_temp['room_indices'][row_room] for room, room_key in rooms_to_indices['room_indices'].items()}

            # Plot the rate maps
            plot_position_spike_rates(df_col_data_cell, cell_column, target_columns=pred_target_columns[row_room],
                                      boundary_points=boundary_points,
                                      pos_range=pos_range, smoothed=smoothed,
                                      rooms_to_indices=rooms_to_indices_temp,
                                      room_normalization=room_normalization,
                                      show_colorbar=show_colorbar,
                                      threshold_marker=threshold_marker,
                                      ax_in=ax, title="")

            # Set Title
            ax_title = f"EMD: {metrics['EMD']:.2f} - Correlation: {metrics['Correlation']:.2f}\nKL: {metrics['KL']:.2f} - JS: {metrics['JS']:.2f}"
            if metrics['EMD'] < green_threshold:
                ax.set_title(ax_title, fontsize=7, color='green')
            elif metrics['EMD'] > red_threshold:
                ax.set_title(ax_title, fontsize=7, color='red')
            else:
                ax.set_title(ax_title, fontsize=7, color='black')

            # else:
            #     # Set Title
            #     ax_title = f"Rate Map Room {row_room}"
            #
            #     # Plot the rate maps
            #     plot_position_spike_rates(df_row_data_cell, cell_column, target_columns=target_columns,
            #                               boundary_points=boundary_points,
            #                               pos_range=pos_range, smoothed=smoothed, threshold_marker=threshold_marker,
            #                               ax_in=ax, title=ax_title)

    # remove the x and y ticks
    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

    # Set the x and y labels as rooms
    for i, row_room in enumerate(rooms):
        ax = axs[i, 0] if n_rooms > 1 else axs[0]
        ax.set_ylabel(f"Position Room {row_room}")

        ax = axs[i, 1] if n_rooms > 1 else axs[1]
        ax.set_ylabel(f"Predicted Position Room {row_room}")

    for i, col_room in enumerate(rooms):
        ax = axs[n_rooms - 1, i + 1] if n_rooms > 1 else axs[i + 1]
        ax.set_xlabel(f"Firing Rate Room {col_room}")

    ax = axs[n_rooms - 1, 0] if n_rooms > 1 else axs[0]
    ax.set_xlabel(f"Firing Rate")

    return fig



@save_plot
def plot_cell_rate_maps_per_room(df_data_cell, cell_column, target_columns, rooms,
                                threshold_marker=None, title='Rate Maps', sub_titles=[], boundary_points=None,
                                sub_titles_colors=None,  # list of colors for each subplot title
                                pos_range=None, smoothed=True,
                                map_rooms=None, room_normalization=True,
                                show_colorbar=False, save_params={}):
    """
    Plot only the real rate maps for each room in a single horizontal row.

    Parameters:
        df_data_cell (DataFrame): DataFrame containing the real data.
        cell_column (str): Column name for the cell data.
        target_columns (list): Columns to use as target columns for the heatmap.
        rooms (list): List of room names/IDs to include in the plot.
        threshold_marker (float, optional): Threshold marker for rate map filtering.
        title (str): Title for the plot.
        sub_titles (list, optional): List of subtitles for each room subplot.
        boundary_points (np.ndarray, optional): Boundary overlay for rate maps.
        pos_range (tuple, optional): XY-axis range.
        smoothed (bool): Whether to smooth the rate maps.
        map_rooms (dict): Index mapping for room-specific positions.
        room_normalization (bool): Whether to normalize maps room-wise.
        show_colorbar (bool): Whether to display colorbars.
        save_params (dict): Parameters for saving the figure.
    """

    n_rooms = len(rooms)
    fig, axs = plt.subplots(1, n_rooms + 1, figsize=(4 * n_rooms, 4), sharex=True, sharey=True)
    fig.suptitle(title, fontsize=14, ha='center', y=0.95)

    ax = axs[0]

    # Title for each subplot
    if sub_titles and 0 < len(sub_titles):
        ax_title = sub_titles[0]
        ax_title_color = sub_titles_colors[0] if sub_titles_colors else 'black'
    else:
        ax_title = 'All Rooms'
        ax_title_color = 'black'
    plot_position_spike_rates(df_data_cell, cell_column, target_columns=target_columns,
                                boundary_points=boundary_points,
                                pos_range=pos_range, smoothed=smoothed,
                                map_rooms=map_rooms,
                                room_normalization=False,
                                show_colorbar=False,
                                threshold_marker=None,
                                ax_in=ax, title=ax_title, title_color=ax_title_color)

    for j, room in enumerate(rooms):
        i=j+1
        ax = axs[i]
        df_room = df_data_cell[df_data_cell['room'] == room]

        # Title for each subplot
        if sub_titles and i < len(sub_titles):
            ax_title = sub_titles[i]
            ax_title_color = sub_titles_colors[i] if sub_titles_colors else 'black'
        else:
            ax_title = f"Room {room}"
            ax_title_color = 'black'

        plot_position_spike_rates(df_room, cell_column, target_columns=target_columns,
                                  boundary_points=boundary_points,
                                  pos_range=pos_range, smoothed=smoothed,
                                  map_rooms=map_rooms,
                                  room_normalization=room_normalization,
                                  show_colorbar=show_colorbar,
                                  threshold_marker=threshold_marker,
                                  ax_in=ax, title=ax_title, title_color=ax_title_color)


    # Remove x/y ticks
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

    return fig





def plot_real_rate_maps_row(df_data_cell, cell_column, target_columns, rooms,
                             threshold_marker=None, title='Rate Map Comparison', boundary_points=None,
                             pos_range=None, smoothed=True, green_threshold=5, red_threshold=10,
                             rooms_to_indices=None,
                             room_normalization=True,
                             show_colorbar=False,
                             save_params={}):
    """
    Plot all real (non-predicted) rate maps for a single cell in one row.
    Order: all rooms, room1, room2, ...
    """
    # Prepare figure
    num_maps = 1 + len(rooms)  # one for 'all', one per room
    fig, axes = plt.subplots(1, num_maps, figsize=(4 * num_maps, 4), squeeze=False)
    axes = axes[0]  # flatten row

    # First: all rooms
    ax = axes[0]
    cell_data = df_data_cell[cell_column]
    rate_map_all = cell_data.get('rate_map_all_smoothed' if smoothed else 'rate_map_all_raw', None)
    if rate_map_all is not None:
        im = ax.imshow(rate_map_all, origin='lower')
        ax.set_title('All Rooms')
        if show_colorbar:
            plt.colorbar(im, ax=ax)
    else:
        ax.set_title('All Rooms\n(no data)')
        ax.axis('off')

    # Then each room
    for i, room in enumerate(rooms):
        ax = axes[i + 1]
        key = f'rate_map_{room}_smoothed' if smoothed else f'rate_map_{room}_raw'
        rate_map = cell_data.get(key, None)
        if rate_map is not None:
            if room_normalization:
                vmax = np.nanmax(rate_map)
            else:
                vmax = np.nanmax(rate_map_all) if rate_map_all is not None else np.nan
            im = ax.imshow(rate_map, origin='lower', vmax=vmax)
            ax.set_title(f'Room {room}')
            if show_colorbar:
                plt.colorbar(im, ax=ax)
        else:
            ax.set_title(f'Room {room}\n(no data)')
            ax.axis('off')

    # Overall title and layout
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return fig


@save_plot
def plot_cv_indices(timestamps, rooms, groups, splits, n_splits, lh=10, title="Cross-Validation Splits Visualization", save_params={}, offsets=None):
    """
    Create a sample plot for indices of a cross-validation object.
    Args:
        timestamps (array-like): Timestamps for the data samples.
        rooms (array-like): Room labels for each sample.
        groups (array-like): Group labels for each sample.
        splits (iterable or dict): A collection of (train_idx, val_idx, test_idx) indices for each split.
                                  If offsets is provided, can be a dict mapping offset -> list of splits.
        n_splits (int): Number of splits in the cross-validation object.
        lh (int): Length of the horizontal lines.
        title (str): Title for the plot.
        save_params (dict): Parameters for saving the plot.
        offsets (list, optional): List of offsets. If provided, splits should be organized by offset.
    Returns:
        Plot figure.
    """
    cmap_cv = cm.coolwarm  # Colormap for CV splits
    cmap_data = cm.Paired  # Colormap for data classes
    lw = 2

    # Handle multiple offsets
    if offsets is not None and len(offsets) > 1:
        # Multiple offsets: organize splits by offset
        if isinstance(splits, dict):
            splits_by_offset = splits
        else:
            # If splits is a flat list, assume it's organized as [offset0_fold0, offset0_fold1, ..., offset1_fold0, ...]
            splits_by_offset = {}
            splits_per_offset = n_splits
            for i, offset in enumerate(offsets):
                start_idx = i * splits_per_offset
                end_idx = start_idx + splits_per_offset
                if end_idx <= len(splits):
                    splits_by_offset[offset] = splits[start_idx:end_idx]
                else:
                    splits_by_offset[offset] = []
        
        # Calculate total rows: (n_splits * n_offsets) + 1 for room row
        n_offsets = len(offsets)
        total_rows = n_splits * n_offsets + 1
        
        # Make rows more compact (squeeze everything)
        row_height = 0.25  # More compact rows
        fig_height = max(6, total_rows * row_height + 1.5)  # Add margin for labels
        
        # Create figure with space for offset/fold columns on the left
        fig = plt.figure(figsize=(12, fig_height))
        gs = GridSpec(1, 2, figure=fig, width_ratios=[0.08, 0.92], hspace=0, wspace=0.05)
        
        # Left axis for offset/fold labels
        ax_labels = fig.add_subplot(gs[0])
        ax_labels.set_xlim(0, 1)
        ax_labels.set_ylim(total_rows, -0.5)
        ax_labels.axis('off')
        
        # Right axis for data visualization
        ax = fig.add_subplot(gs[1])
        
        # Plot each offset-fold combination
        row_idx = 0
        offset_labels = []
        fold_labels = []
        
        for offset in offsets:
            offset_splits = splits_by_offset.get(offset, [])
            for fold_idx, (train_idx, val_idx, test_idx) in enumerate(offset_splits):
                # Fill in indices with the training/test groups
                indices = np.array([np.nan] * len(timestamps))
                indices[train_idx] = 0  # Training set
                indices[val_idx] = 1    # Validation set
                indices[test_idx] = 2   # Test set

                # Visualize the results
                ax.scatter(
                    range(len(indices)),
                    [row_idx + 0.5] * len(indices),
                    c=indices,
                    marker="|",
                    lw=lw,
                    s=lh * 0.8,  # Slightly smaller markers for compactness
                    cmap=cmap_cv,
                    vmin=0,
                    vmax=2,  # Limits to map each set to a unique color
                )
                
                # Store separate labels for offset and fold
                offset_labels.append(str(offset))
                fold_labels.append(str(fold_idx + 1))
                row_idx += 1
        
        # Plot the data classes (room labels)
        ax.scatter(
            range(len(timestamps)), [row_idx + 0.5] * len(timestamps), c=rooms.values, marker="|", lw=lw, s=lh * 0.8, cmap=cmap_data
        )
        offset_labels.append("")
        fold_labels.append("Room")
        
        # Add offset and fold columns as text
        # Offset column (left)
        for i, label in enumerate(offset_labels):
            ax_labels.text(0.3, i + 0.5, label, ha='center', va='center', fontsize=9, fontweight='bold' if label == "" else 'normal')
        
        # Fold column (right of offset column)
        for i, label in enumerate(fold_labels):
            ax_labels.text(0.95, i + 0.5, label, ha='center', va='center', fontsize=9)
        
        # Add column headers
        ax_labels.text(0.3, -0.5, 'Offset', ha='center', va='top', fontsize=10, fontweight='bold')
        ax_labels.text(0.95, -0.5, 'Fold', ha='center', va='top', fontsize=10, fontweight='bold')
        
        # Formatting and labels for main plot
        ax.set(
            yticks=np.arange(total_rows) + 0.5,
            yticklabels=[],  # No y-axis labels since we have separate columns
            xlabel="Index",
            ylabel="",
            ylim=[total_rows, -0.5],
            xlim=[0, len(timestamps)],
        )
    else:
        # Single offset (original behavior)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Generate the training/testing visualizations for each CV split
        for ii, (train_idx, val_idx, test_idx) in enumerate(splits):
            # Fill in indices with the training/test groups
            indices = np.array([np.nan] * len(timestamps))
            indices[train_idx] = 0  # Training set
            indices[val_idx] = 1    # Validation set
            indices[test_idx] = 2   # Test set

            # Visualize the results
            ax.scatter(
                range(len(indices)),
                [ii + 0.5] * len(indices),
                c=indices,
                marker="|",
                lw=lw,
                s=lh,
                cmap=cmap_cv,
                vmin=0,
                vmax=2,  # Limits to map each set to a unique color
            )

        # Plot the data classes (room labels)
        ax.scatter(
            range(len(timestamps)), [n_splits + 0.5] * len(timestamps), c=rooms.values, marker="|", lw=lw, s=lh, cmap=cmap_data
        )

        # Formatting and labels
        yticklabels = [f'Fold {fold+1}' for fold in range(n_splits)] + ["Room"]
        ax.set(
            yticks=np.arange(n_splits + 1) + 0.5,
            yticklabels=yticklabels,
            xlabel="Index",
            ylabel="CV Iteration",
            ylim=[n_splits + 1, -0.5],
            xlim=[0, len(timestamps)],
        )
    
    if title: ax.set_title(title, fontsize=15)
    # Legend with filled boxes instead of "|" markers
    legend_labels = ["Training", "Validation", "Test"]
    from matplotlib.patches import Rectangle
    legend_patches = [Rectangle((0, 0), 1, 1, facecolor=cmap_cv(i / 2), edgecolor='black') for i in range(3)]
    ax.legend(legend_patches, legend_labels, loc="upper right", title="Set Type")

    return plt.gcf()


@save_plot
def plot_cv_room_metrics_panel(decoder_room_metrics_df, decoder_room_ensemble_metrics_df=None, decoder_offset_metrics_df=None, title=None, save_params={}):
    """
    Plot a two-panel summary (R2 and RMSE) using decoder room-level metrics,
    with error bars showing variability across offsets and optional ensemble-predictor markers.
    Includes scatter points for individual offset-level aggregated points if decoder_offset_metrics_df is provided.
    """
    if decoder_room_metrics_df is None or len(decoder_room_metrics_df) == 0:
        logger.warning("decoder_room_metrics_df is empty; skipping metrics panel plot.")
        return None

    stats_df = decoder_room_metrics_df.copy()
    mean_df = decoder_room_ensemble_metrics_df.copy() if decoder_room_ensemble_metrics_df is not None else pd.DataFrame()
    metrics_df = decoder_offset_metrics_df.copy() if decoder_offset_metrics_df is not None and len(decoder_offset_metrics_df) > 0 else pd.DataFrame()

    # Focus on test set if available
    if 'set' in stats_df.columns and 'test' in stats_df['set'].unique():
        stats_df = stats_df[stats_df['set'] == 'test']
    if len(mean_df) > 0 and 'set' in mean_df.columns and 'test' in mean_df['set'].unique():
        mean_df = mean_df[mean_df['set'] == 'test']
    if len(metrics_df) > 0 and 'set' in metrics_df.columns and 'test' in metrics_df['set'].unique():
        metrics_df = metrics_df[metrics_df['set'] == 'test']

    # Extract n_seeds once (assume same for all rooms)
    n_seeds = int(stats_df['n_offsets'].iloc[0]) if 'n_offsets' in stats_df.columns else 1

    # Extract group from config if available
    config = save_params.get('config', {})
    group_name = config.get('project_info', {}).get('group', '')
    if group_name and title:
        title = f"{title} - {group_name}"
    elif group_name:
        title = group_name

    all_project_rooms = get_rooms_from_config(config)
    
    # Get rooms from stats_df and combine with all project rooms
    rooms_from_stats = stats_df['room'].tolist()
    # Combine: all project rooms (excluding 'All'), then 'All' if present
    rooms = [r for r in all_project_rooms if r != 'All'] if all_project_rooms else [r for r in rooms_from_stats if r != 'All']
    # Add 'All' at the end if it exists in stats
    if 'All' in stats_df['room'].values:
        rooms.append('All')
    
    # Reindex to include all rooms (missing rooms will have NaN values)
    stats_df = stats_df.set_index('room').reindex(rooms)
    mean_df = mean_df.set_index('room').reindex(rooms) if len(mean_df) > 0 else mean_df

    # Metric configuration: (stats_col, std_col, ensemble_predictor_col, metric_col_in_decoder_offset_metrics, ylabel, title_suffix)
    # Note: For RMSE, fallback logic applies: if 'rmse_cm' is missing, try 'mse' (with sqrt, no unit conversion)
    #       No unit conversion is performed in this function - values are used as-is
    metrics_config = [
        ('r2_pooled_mean', 'r2_pooled_std', 'r2_pooled', 'r2_pooled', 'R² (pooled)', 'R²'),
        ('rmse_mean', 'rmse_std', 'rmse_cm', 'rmse_cm', 'RMSE (cm)', 'RMSE')  # Fallback: rmse_cm -> mse (sqrt)
    ]

    # Styling constants
    MODEL_COLOR = '#2E86AB'
    BASELINE_COLOR = '#6C757D'
    SCATTER_COLOR = '#1B4F72'  # Darker blue for scatter points
    BAR_WIDTH = 0.35
    BAR_OFFSET = 0.2
    ERROR_CAP_SIZE = 5
    SCATTER_ALPHA = 0.6
    SCATTER_SIZE = 30
    SCATTER_JITTER = BAR_WIDTH * 0.25  # Jitter width proportional to bar width

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('white')
    x = np.arange(len(rooms))

    # Plot each metric panel
    for ax_idx, (mean_col, std_col, mean_predictor_col, metric_col, ylabel, title_suffix) in enumerate(metrics_config):
        ax = axes[ax_idx]

        # Model bars with error bars
        model_values = stats_df[mean_col]
        model_errors = stats_df[std_col] if std_col in stats_df.columns else None
        
        bars_model = ax.bar(
            x - BAR_OFFSET,
            model_values,
            width=BAR_WIDTH,
            yerr=model_errors,
            capsize=ERROR_CAP_SIZE,
            label='Individual Models',
            color=MODEL_COLOR,
            alpha=0.85,
            edgecolor='white',
            linewidth=1.2,
            zorder=2
        )

        # Add scatter points for individual offset-level aggregated points (all-folds rows only)
        if len(metrics_df) > 0 and 'room' in metrics_df.columns and metric_col in metrics_df.columns:
            scatter_x_list = []
            scatter_y_list = []
            
            # Filter for all-folds rows only (where fold is None/NaN)
            all_folds_metrics = metrics_df[metrics_df['fold'].isna()].copy()
            
            for room_idx, room in enumerate(rooms):
                # Filter all_folds_metrics for this room
                room_metrics = all_folds_metrics[all_folds_metrics['room'] == room].copy()
                
                # For 'All' room, if no data exists, calculate mean across all other rooms for each offset
                if len(room_metrics) == 0 and room == 'All' and len(all_folds_metrics) > 0:
                    # Get all individual rooms (excluding 'All')
                    individual_rooms = [r for r in rooms if r != 'All']
                    if individual_rooms:
                        # Group by offset and calculate mean across rooms for each offset
                        if 'offset' in all_folds_metrics.columns:
                            room_metrics_list = []
                            for offset in all_folds_metrics['offset'].unique():
                                offset_data = all_folds_metrics[
                                    (all_folds_metrics['offset'] == offset) & 
                                    (all_folds_metrics['room'].isin(individual_rooms))
                                ]
                                if len(offset_data) > 0:
                                    # Calculate mean for this metric across rooms for this offset
                                    if metric_col in offset_data.columns:
                                        mean_val = offset_data[metric_col].mean()
                                    elif metric_col == 'rmse_cm' and 'mse' in offset_data.columns:
                                        mean_val = np.sqrt(offset_data['mse'].mean())
                                    else:
                                        continue
                                    
                                    # Create a row for 'All' room with this offset
                                    row = offset_data.iloc[0].copy()
                                    row['room'] = 'All'
                                    if metric_col in row.index:
                                        row[metric_col] = mean_val
                                    elif metric_col == 'rmse_cm' and 'mse' in row.index:
                                        row['mse'] = offset_data['mse'].mean()
                                    room_metrics_list.append(row)
                            
                            if room_metrics_list:
                                room_metrics = pd.DataFrame(room_metrics_list)
                
                if len(room_metrics) > 0:
                    # Extract metric values with fallback logic for RMSE
                    # For RMSE: try 'rmse_cm' -> 'mse' (calculate sqrt, no unit conversion)
                    # For R2: use 'r2_pooled' directly
                    if metric_col in room_metrics.columns:
                        metric_values = room_metrics[metric_col].values
                    elif metric_col == 'rmse_cm':
                        # Fallback chain for RMSE: rmse_cm -> mse (calculate sqrt)
                        if 'mse' in room_metrics.columns:
                            metric_values = np.sqrt(room_metrics['mse'].values)
                        else:
                            metric_values = np.array([])
                    else:
                        # Fallback: skip if column not found
                        metric_values = np.array([])
                    
                    # Add jitter to x positions
                    n_points = len(metric_values)
                    if n_points > 0:
                        jitter = np.random.uniform(-SCATTER_JITTER, SCATTER_JITTER, size=n_points)
                        scatter_x = (room_idx - BAR_OFFSET) + jitter
                        scatter_x_list.extend(scatter_x)
                        scatter_y_list.extend(metric_values)
            
            if len(scatter_x_list) > 0:
                ax.scatter(
                    scatter_x_list,
                    scatter_y_list,
                    s=SCATTER_SIZE,
                    color=SCATTER_COLOR,
                    alpha=SCATTER_ALPHA,
                    edgecolors='white',
                    linewidths=0.5,
                    zorder=3,
                    label='Offset-level points'
                )

        # Mean predictor bars (if available)
        # mean_predictor_col should be 'r2_pooled' or 'rmse_cm' (already in correct units)
        # For RMSE: fallback chain: rmse_cm -> mse (calculate sqrt, no unit conversion)
        if len(mean_df) > 0:
            if mean_predictor_col in mean_df.columns:
                baseline_values = mean_df[mean_predictor_col]
            elif mean_predictor_col == 'rmse_cm':
                # Fallback chain for RMSE: rmse_cm -> mse (calculate sqrt)
                if 'mse' in mean_df.columns:
                    baseline_values = np.sqrt(mean_df['mse'].values)
                else:
                    baseline_values = None
            else:
                baseline_values = None
            
            if baseline_values is not None:
                bars_baseline = ax.bar(
                    x + BAR_OFFSET,
                    baseline_values,
                    width=BAR_WIDTH,
                    label='Ensemble (mean)',
                    color=BASELINE_COLOR,
                    alpha=0.85,
                    edgecolor='white',
                    linewidth=1.2,
                    hatch='///',
                    zorder=2
                )

        # Fix R2 axis to 0-1 for the first subplot (R2)
        if ax_idx == 0:  # R2 subplot
            ax.set_ylim(0, 1)
        
        # Styling
        ax.set_xticks(x)
        ax.set_xticklabels(rooms, fontsize=11, fontweight='medium')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='medium')
        ax.set_title(f'{title_suffix} across rooms (n={n_seeds})', fontsize=12, fontweight='bold', pad=10)
        
        # Grid and spines
        ax.grid(True, axis='y', alpha=0.25, linestyle='--', linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')
        
        # Legend
        ax.legend(
            loc='lower right',
            frameon=True,
            fancybox=True,
            shadow=False,
            framealpha=0.9,
            fontsize=10,
            edgecolor='#CCCCCC'
        )

    # Main title
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


@save_plot
def plot_tracking_overview(df_filtered,df, map_rooms, title='Tracking Overview', save_params={}):
    """
    Generate tracking overview plots based on the data in the given DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing the tracking data with columns:
            - PC_HeadPos.X, PC_HeadPos.Y, PC_HeadPos.Int
            - TC_HeadPos.X, TC_HeadPos.Y, TC_HeadPos.Int
            - EC_TailPos.X, EC_TailPos.Y, EC_TailPos.Int
            - TS
        save_params: Parameters for saving the plot.
    """
    rw, cl = 13, 6
    max_dist = 2

    # Calculate distances
    dx = df['PC_HeadPos.X'] - df['TC_HeadPos.X']
    dy = df['PC_HeadPos.Y'] - df['TC_HeadPos.Y']
    d = np.sqrt((dx - dx.mean())**2 + (dy - dy.mean())**2)

    # Create a grid layout
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(rw, cl, figure=fig)


    # PC HeadPos points with <5 interpolations
    ax = fig.add_subplot(gs[0:2, 0:1])
    if 'PC_HeadPos.Int' in df.columns:
        ax.scatter(df['PC_HeadPos.X'][df['PC_HeadPos.Int'] < 5],
                   df['PC_HeadPos.Y'][df['PC_HeadPos.Int'] < 5],
                   s=6, c='k', label='<5 interpolations')
        ax.scatter(df['PC_HeadPos.X'][df['PC_HeadPos.Int'] >= 5],
                   df['PC_HeadPos.Y'][df['PC_HeadPos.Int'] >= 5],
                   s=2, c='y', label='>=5 interpolations')
        ax.set_title(f'PC HeadPos (<5 interpolations)\n{((df["PC_HeadPos.Int"] < 5).sum() / len(df)) * 100:.0f}% points', pad=10)
    else:
        ax.scatter(df['PC_HeadPos.X'][df['PC_HeadPos.P'] > 0.5],
                   df['PC_HeadPos.Y'][df['PC_HeadPos.P'] > 0.5],
                   s=6, c='k', label='P>0.5')
        ax.scatter(df['PC_HeadPos.X'][df['PC_HeadPos.P'] <= 0.5],
                   df['PC_HeadPos.Y'][df['PC_HeadPos.P'] <= 0.5],
                   s=2, c='y', label='P<=0.5')
        ax.set_title(f'PC HeadPos (P>0.5)\n{((df["PC_HeadPos.P"] > 0.5).sum() / len(df)) * 100:.0f}% points', pad=10)
    ax.set_aspect('equal')
    ax.set_xlim([-10, 130])
    ax.set_ylim([-10, 130])


    # Points with TC HeadPos to PC Headpos dist < MaxDist
    ax = fig.add_subplot(gs[0:2, 1:2])
    ax.scatter(df['TC_HeadPos.X'][d < max_dist],
               df['TC_HeadPos.Y'][d < max_dist],
               s=6, c='g', label=f'distance < {max_dist}')
    ax.set_aspect('equal')
    ax.set_xlim([-10, 130])
    ax.set_ylim([-10, 130])
    ax.set_title(f'TC-HeadPos dist<{max_dist:.1f}cm\n{(d < max_dist).sum() / len(d) * 100:.0f}% points', pad=10)

    # TC HeadPos points with <5 interpolations
    ax = fig.add_subplot(gs[0:2, 2:3])
    if 'TC_HeadPos.Int' in df.columns:
        ax.scatter(df['TC_HeadPos.X'][df['TC_HeadPos.Int'] < 5],
                   df['TC_HeadPos.Y'][df['TC_HeadPos.Int'] < 5],
                   s=6, c='k', label='<5 interpolations')
        ax.scatter(df['TC_HeadPos.X'][df['TC_HeadPos.Int'] >= 5],
                   df['TC_HeadPos.Y'][df['TC_HeadPos.Int'] >= 5],
                   s=2, c='y', label='>=5 interpolations')
        ax.set_title(f'TC HeadPos (<5 interpolations)\n{((df["TC_HeadPos.Int"] < 5).sum() / len(df)) * 100:.0f}% points', pad=10)
    else:
        ax.scatter(df['TC_HeadPos.X'][df['TC_HeadPos.P'] > 0.5],
                   df['TC_HeadPos.Y'][df['TC_HeadPos.P'] > 0.5],
                   s=6, c='k', label='P>0.5')
        ax.scatter(df['TC_HeadPos.X'][df['TC_HeadPos.P'] <= 0.5],
                   df['TC_HeadPos.Y'][df['TC_HeadPos.P'] <= 0.5],
                   s=2, c='y', label='P<=0.5')
        ax.set_title(f'TC HeadPos (P>0.5)\n{((df["TC_HeadPos.P"] > 0.5).sum() / len(df)) * 100:.0f}% points', pad=10)
    ax.set_aspect('equal')
    ax.set_xlim([-10, 140])
    ax.set_ylim([-10, 140])

    # EC TailPos points with <5 interpolations
    ax = fig.add_subplot(gs[0:2, 3:4])
    if 'EC_TailPos.X' in df.columns:
        if 'EC_TailPos.Int' in df.columns:
            ax.scatter(df['EC_TailPos.X'][df['EC_TailPos.Int'] < 5],
                       df['EC_TailPos.Y'][df['EC_TailPos.Int'] < 5],
                       s=6, c='k', label='<5 interpolations')
            ax.scatter(df['EC_TailPos.X'][df['EC_TailPos.Int'] >= 5],
                       df['EC_TailPos.Y'][df['EC_TailPos.Int'] >= 5],
                       s=2, c='y', label='>=5 interpolations')
            ax.set_title(f'EC Tail (<5 interpolations)\n{((df["EC_TailPos.Int"] < 5).sum() / len(df)) * 100:.0f}% points', pad=10)
        else:
            ax.scatter(df['EC_TailPos.X'][df['EC_TailPos.P'] > 0.5],
                       df['EC_TailPos.Y'][df['EC_TailPos.P'] > 0.5],
                       s=6, c='k', label='P>0.5')
            ax.scatter(df['EC_TailPos.X'][df['EC_TailPos.P'] <= 0.5],
                       df['EC_TailPos.Y'][df['EC_TailPos.P'] <= 0.5],
                       s=2, c='y', label='P<=0.5')
            ax.set_title(f'PC TailPos (P>0.5)\n{((df["EC_TailPos.P"] > 0.5).sum() / len(df)) * 100:.0f}% points', pad=10)

    ax.set_aspect('equal')
    ax.set_xlim([-10, 140])
    ax.set_ylim([-10, 140])


    # PC X HeadPos
    ax = fig.add_subplot(gs[2, :])
    if 'PC_HeadPos.Int' in df.columns:
        color = (df['PC_HeadPos.Int'] > 5).astype(int)
        sub_title = 'PC HeadPos (green/yellow: >5 interpolations)'
    elif 'PC_HeadPos.P' in df.columns:
        color = (df['PC_HeadPos.P'] < 0.5).astype(int)
        sub_title = 'PC HeadPos (green/yellow: P < 0.5)'
    else:
        color = np.zeros(len(df))
        sub_title = 'PC HeadPos'
    color[0] = 0
    ax.scatter(df['TS'], df['PC_HeadPos.X'], s=2, c=color, cmap='winter',label='X')
    ax.scatter(df['TS'], df['PC_HeadPos.Y'], s=2, c=color, cmap='autumn',label='Y')
    ax.set_ylim([-10, 140])
    ax.set_xlim([0, 4000])
    ax.legend()
    ax.set_title(sub_title,  pad=10)

    # TC X HeadPos
    ax = fig.add_subplot(gs[3, :])
    if 'TC_HeadPos.Int' in df.columns:
        color = (df['TC_HeadPos.Int'] > 5).astype(int)
        sub_title = 'TC HeadPos (green/yellow: >5 interpolations)'
    elif 'TC_HeadPos.P' in df.columns:
        color = (df['TC_HeadPos.P'] < 0.5).astype(int)
        sub_title = 'TC HeadPos (green/yellow: P < 0.5)'
    else:
        color = np.zeros(len(df))
        sub_title = 'TC HeadPos'
    color[0] = 0
    ax.scatter(df['TS'], df['TC_HeadPos.X'], s=2, c=color, cmap='winter',label='X')
    ax.scatter(df['TS'], df['TC_HeadPos.Y'], s=2, c=color, cmap='autumn',label='Y')
    ax.set_ylim([-10, 140])
    ax.set_xlim([0, 4000])
    ax.legend()
    ax.set_title(sub_title, pad=10)

    # EC TailPos X
    if 'EC_TailPos.X' in df.columns:
        ax = fig.add_subplot(gs[4, :])
        if 'EC_TailPos.Int' in df.columns:
            color = (df['EC_TailPos.Int'] > 5).astype(int)
            sub_title = 'EC TailPos (green/yellow: >5 interpolations)'
        elif 'EC_TailPos.P' in df.columns:
            color = (df['EC_TailPos.P'] < 0.5).astype(int)
            sub_title = 'EC TailPos (green/yellow: P < 0.5)'
        else:
            color = np.zeros(len(df))
            sub_title = 'EC TailPos'
        color[0] = 0
        ax.scatter(df['TS'], df['EC_TailPos.X'], s=2, c=color, cmap='winter', label='X')
        ax.scatter(df['TS'], df['EC_TailPos.Y'], s=2, c=color, cmap='autumn', label='Y')
        ax.set_ylim([-10, 140])
        ax.set_xlim([0, 4000])
        ax.legend()
        ax.set_title(sub_title, pad=10)

    # # ECTailPos X
    # if 'EC_TailPos.X' in df.columns:
    #     ax = fig.add_subplot(gs[5, :])
    #     color = df['EC_TailPos.P'] < 0.5 if 'EC_TailPos.P' in df.columns else 0
    #     ax.scatter(df['TS'], df['EC_TailPos.X'], s=2, c=color, cmap='winter')
    #     ax.set_ylim([-10, 140])
    #     ax.set_xlim([0, 4000])
    #     ax.set_title('EC X TailPos (green: P < 0.5)', pad=10)

    # X
    ax = fig.add_subplot(gs[6, :])
    color = np.zeros(len(df_filtered))
    ax.scatter(df_filtered['timestamp'], df_filtered['X'], s=2, c=color, cmap='winter', label='X')
    ax.scatter(df_filtered['timestamp'], df_filtered['Y'], s=2, c=color, cmap='autumn', label='Y')
    # Add vertical lines for room times
    for room_name, room in map_rooms['rooms'].items():
        times = room['range']
        for time in times:
            plt.axvline(x=time, color='black', linestyle='--', linewidth=1, label=f'Rooms Segments'  if time == times[0] and room_name == map_rooms['rooms_list'][0] else "")

    ax.set_ylim([-10, 140])
    ax.set_xlim([0, 4000])
    ax.legend()
    ax.set_title('Position', pad=10)

    # PC HD
    if 'PC_HD' in df.columns or 'PC_HD.Angle' in df.columns:
        if 'PC_HD' in df.columns:
            HD_column = 'PC_HD'
        else:
            HD_column = 'PC_HD.Angle'

        # # Normalize HD column (optional but safer)
        # normed_hd = df[HD_column] % 360  # wrap around just in case
        #
        # ax = fig.add_subplot(gs[7, :])
        # ax.scatter(df['TS'], df[HD_column], s=2, c=normed_hd, cmap='twilight')
        # ax.set_ylim([-10, 370])
        # ax.set_xlim([0, 4000])
        # ax.set_title('PC HD', pad=10)

        # Convert HD to radians (with wrap-around)
        hd_rad = np.deg2rad(df[HD_column] % 360)

        # Plot sine and cosine of PC HD over time
        ax = fig.add_subplot(gs[7, :])
        ax.plot(df['TS'], np.sin(hd_rad), label='sin(HD)', alpha=0.7, color='blue')
        # ax.plot(df['TS'], np.cos(hd_rad), label='cos(HD)', alpha=0.7)

        ax.set_ylim([-1.4, 1.4])
        ax.set_xlim([0, 4000])
        ax.set_title('PC Head Direction (Sine View)', pad=10)
        ax.legend()

    # TC HD
    if 'TC_HD' in df.columns or 'TC_HD.Angle' in df.columns:
        if 'TC_HD' in df.columns:
            HD_column = 'TC_HD'
        else:
            HD_column = 'TC_HD.Angle'

        # # Normalize HD column (optional but safer)
        # normed_hd = df[HD_column] % 360  # wrap around just in case
        #
        # ax = fig.add_subplot(gs[8, :])
        # ax.scatter(df['TS'], df[HD_column], s=2, c=normed_hd, cmap='twilight')
        # ax.set_ylim([-10, 370])
        # ax.set_xlim([0, 4000])
        # ax.set_title('TC HD', pad=10)

        # Convert HD to radians (ensure wrap-around is clean)
        hd_rad = np.deg2rad(df[HD_column] % 360)

        # Plot sine and cosine of HD over time
        ax = fig.add_subplot(gs[8, :])
        ax.plot(df['TS'], np.sin(hd_rad), label='sin(HD)', alpha=0.7, color='blue')
        # ax.plot(df['TS'], np.cos(hd_rad), label='cos(HD)', alpha=0.7)

        ax.set_ylim([-1.4, 1.4])
        ax.set_xlim([0, 4000])
        ax.set_title('TC Head Direction (Sine View)', pad=10)
        ax.legend()

    # Trajectory over time colored by room
    ax = fig.add_subplot(gs[9, :])
    room_series = df_filtered['room'] if 'room' in df_filtered.columns else pd.Series([None] * len(df_filtered))
    cmap = plt.get_cmap('tab10')
    room_color_map = {}
    color_idx = 0

    for room in room_series.unique():
        if pd.isna(room) or room is None:
            room_color_map[room] = 'grey'
        else:
            room_color_map[room] = cmap(color_idx % 10)
            color_idx += 1

    for room, group in df_filtered.groupby(room_series):
        color = room_color_map[room]
        ax.plot(group['timestamp'], group['X'], '.', markersize=2, color=color, label=f'{room}')
        ax.plot(group['timestamp'], group['Y'], '.', markersize=2, color=color, label=None)

    ax.set_title('Trajectory over time colored by room')
    ax.set_xlim([0, 4000])
    ax.set_ylim([-10, 140])
    ax.legend(markerscale=5, fontsize='x-small', loc='best')




    # Histograms
    ax = fig.add_subplot(gs[-2, 0:2])
    ax.hist(d, bins=50, color='blue', alpha=0.7)
    ax.set_xlim([0, 10])
    ax.set_title(f'TC-PC distance, Std={d.std():.1f}', pad=10)

    ax = fig.add_subplot(gs[-2, 2:4])
    ax.hist(dx, bins=50, color='blue', alpha=0.7)
    ax.set_xlim([-10, 10])
    ax.set_title(f'TC-PC X distance, Std={dx.std():.1f}', pad=10)

    ax = fig.add_subplot(gs[-2, 4:6])
    ax.hist(dy, bins=50, color='blue', alpha=0.7)
    ax.set_xlim([-10, 10])
    ax.set_title(f'TC-PC Y distance, Std={dy.std():.1f}', pad=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
    plt.suptitle(title, fontsize=16, y=0.98)

    return plt.gcf()


@save_plot
def plot_advanced_line_plot_over_time(
        df_results, pos_range=None, boundary_points=None, scaler=None,
        target_columns=['X', 'Y'], pred_target_columns=['X_pred', 'Y_pred'],
        title="Real vs Predicted Positions over Time", save_params={}, n_seeds=None, unit_name='cm'):
    """
    Plots real vs predicted positions over time for X and Y components with
    statistical overlays and 2D trajectories, grouped by folds.

    This function generates the following plots:
    1. Real vs predicted X positions over time (line plot with standard deviation bands for predictions) + position distribution
    2. Real vs predicted Y positions over time (line plot with standard deviation bands for predictions) + position distribution
    3. Error over time (total and per-coordinate) + error distribution
    4. Real 2D trajectories for each fold.
    5. Predicted 2D trajectories for each fold.

    Parameters:
        df_results (DataFrame): A pandas DataFrame containing the results with columns:
            - 'timestamp': Timestamps for the data points (unique per row).
            - target_columns: Real positions (e.g., 'X', 'Y').
            - pred_target_columns: Predicted positions (e.g., 'X_pred', 'Y_pred').
            - '{pred_col}_std': Standard deviation columns for predictions (e.g., 'X_pred_std', 'Y_pred_std').
            - 'fold': Fold identifier for grouping.
            - 'room': Room identifier (single room expected).
        target_columns (list, optional): List of columns to use as target columns for the real data.
        pred_target_columns (list, optional): List of columns to use as target columns for the predicted data.
        pos_range (tuple, optional): Tuple of (min, max) for the x and y axis range.
        boundary_points (np.ndarray, optional): Array of boundary points to overlay on the plots.
        scaler (sklearn.preprocessing.MinMaxScaler, optional): Scaler used to transform the positions.
        title (str): Title for the plot.
        save_params (dict): Parameters for saving the plot.
        n_seeds (int, optional): Number of seeds/offsets for uncertainty band legend.
        unit_name (str, optional): Unit name for axes labels (default: 'cm').

    Returns:
        matplotlib.figure.Figure: The generated figure containing all subplots.
    """

    df_results = df_results.copy()
    boundary_points = boundary_points.copy() if boundary_points is not None else None
    # Ensure required columns exist
    if 'fold' not in df_results.columns:
        df_results['fold'] = 0
    if 'room' not in df_results.columns:
        df_results['room'] = 'unknown'
    if len(df_results['room'].unique()) > 1:
        raise ValueError("plot_advanced_line_plot_over_time only supports one room at a time")

    # Handle scaler if provided: reverse transform positions and scale std values
    # Note: We'll create valid_mask after scaler transformation, so transform all values first
    # The scaler expects data reshaped to (-1, 1) - it was fitted on flattened values
    # sklearn scalers don't handle NaNs in inverse_transform, so we need to handle them specially
    if scaler is not None:
        # Inverse transform target columns - only transform valid (non-NaN) values
        # Filling NaNs with 0 causes incorrect transformation (0 maps to min value, not 0)
        positions = df_results[target_columns].values.copy()
        nan_mask_targets = np.isnan(positions)
        # Only transform valid values, keep NaNs as NaNs
        if not nan_mask_targets.all():
            valid_positions = positions[~nan_mask_targets]
            if len(valid_positions) > 0:
                # Transform valid values only
                valid_transformed = apply_scaler_transform(valid_positions, scaler, reverse=True)
                positions[~nan_mask_targets] = valid_transformed
        df_results[target_columns] = positions
        
        # Inverse transform prediction columns - only transform valid (non-NaN) values
        # Filling NaNs with 0 causes incorrect transformation (0 maps to min value, not 0)
        predictions = df_results[pred_target_columns].values.copy()
        nan_mask_preds = np.isnan(predictions)
        # Only transform valid values, keep NaNs as NaNs
        if not nan_mask_preds.all():
            valid_predictions = predictions[~nan_mask_preds]
            if len(valid_predictions) > 0:
                # Transform valid values only
                valid_transformed = apply_scaler_transform(valid_predictions, scaler, reverse=True)
                predictions[~nan_mask_preds] = valid_transformed
        df_results[pred_target_columns] = predictions
        
        # Inverse transform boundary points (from normalized to cm)
        if boundary_points is not None:
            boundary_points[:, :2] = apply_scaler_transform(boundary_points[:, :2], scaler, reverse=True)
        
        # Inverse transform pos_range (from normalized to cm) to match transformed positions
        if pos_range is not None:
            pos_range = tuple(apply_scaler_transform(value, scaler, reverse=True) for value in pos_range)
        
        # Scale std values back to original units
        # For MinMaxScaler: X_scaled = scale_ * X + offset  =>  SD_original = SD_scaled / scale_
        # Use get_scale_factor_from_scaler for consistency
        from utils.config import get_scale_factor_from_scaler
        scale_factor = get_scale_factor_from_scaler(scaler)
        
        if scale_factor != 1.0:
            for i, p_col in enumerate(pred_target_columns):
                std_col = f'{p_col}_std'
                if std_col in df_results.columns:
                    df_results[std_col] = df_results[std_col] / scale_factor
    
    # Keep NaNs for plotting (will be skipped, creating gaps in the line)
    # Don't fill NaNs - we want to skip them when plotting

    # Create a single valid mask for rows where both targets and predictions are valid (not NaN)
    # This mask will be used throughout the function
    valid_mask = (
        df_results[target_columns].notna().all(axis=1) & 
        df_results[pred_target_columns].notna().all(axis=1)
    )
    
    # If no valid rows, create an empty figure and return
    if not valid_mask.any():
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No valid data to plot\n(All predictions are NaN)', 
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title(title)
        return fig

    # Calculate metrics using calculate_metrics for the full time range (excluding NaNs)
    df_for_metrics = df_results[valid_mask].copy()
    all_targets = df_for_metrics[target_columns].values
    all_predictions = df_for_metrics[pred_target_columns].values
    metrics = calculate_metrics(all_targets, all_predictions, target_columns=target_columns)

    # Compute errors (in scaler units) - handle NaNs
    # Use the same valid_mask created at the start
    target_vals = df_results[target_columns].values
    pred_vals = df_results[pred_target_columns].values
    df_results['error'] = np.nan
    df_results.loc[valid_mask, 'error'] = np.linalg.norm(
        target_vals[valid_mask] - pred_vals[valid_mask],
        axis=1
    )
    for t_col, p_col in zip(target_columns, pred_target_columns):
        df_results[f'{t_col}_error'] = np.abs(df_results[t_col] - df_results[p_col])
        # Set to NaN where either column is NaN
        df_results.loc[~np.isfinite(df_results[t_col]) | ~np.isfinite(df_results[p_col]), f'{t_col}_error'] = np.nan

    # Sort by timestamp for plotting
    df_results = df_results.sort_values(by='timestamp')

    # Get unique folds, excluding NaN (only from valid rows)
    unique_folds = df_results.loc[valid_mask, 'fold'].dropna().unique()
    n_folds = len(unique_folds) if len(unique_folds) > 0 else 1

    # Initialize figure
    # Calculate number of rows dynamically:
    # - len(target_columns) rows for time/distribution plots
    # - 1 row for error plots
    # - 1 row for real trajectories (with colorbar in last column)
    # - 1 row for predicted trajectories
    n_rows = len(target_columns) + 3
    fig = plt.figure(figsize=(4 * n_folds, 20))
    gs = fig.add_gridspec(n_rows, n_folds + 1)

    # Axes for each target dimension + error
    axes_time = []
    axes_dist = []
    for i, t_col in enumerate(target_columns):
        ax_time = fig.add_subplot(gs[i, :-1])
        ax_dist = fig.add_subplot(gs[i, -1])
        axes_time.append(ax_time)
        axes_dist.append(ax_dist)

    # Error plot axes
    ax_error_time = fig.add_subplot(gs[len(target_columns), :-1])
    ax_error_dist = fig.add_subplot(gs[len(target_columns), -1])

    # Define colors
    # Error colors: red for X (index 0), blue for Y (index 1) to match minimal mapping plot
    colors = {
        'real': '#1f77b4',  # Blue
        'predicted': '#d62728',  # Red
        'error_total': '#9467bd',  # Purple
        'error_dim': ['#d62728', '#1f77b4', '#17becf', '#bcbd22', '#e377c2'],  # Red for X, Blue for Y, then other colors
    }

    legend_added = False

    # Plot trajectory over time for each dimension (Plot SD bands for pred columns using std)
    for i, (t_col, p_col) in enumerate(zip(target_columns, pred_target_columns)):
        ax = axes_time[i]
        
        # Plot real values (NaN values are automatically skipped using masked arrays)
        # For very large datasets, we might need to downsample for better visualization
        ts = df_results['timestamp'].values
        real_vals = np.ma.masked_invalid(df_results[t_col].values)
        
        ax.plot(ts, real_vals, color=colors['real'], 
                label=f"Real {t_col}" if not legend_added else "",
                alpha=0.8, linewidth=0.5 if len(ts) > 50000 else 1.0)
        
        # Plot predicted values (NaN values are automatically skipped using masked arrays)
        pred_vals = np.ma.masked_invalid(df_results[p_col].values)
        ax.plot(ts, pred_vals, color=colors['predicted'], 
                label=f"Predicted {t_col}" if not legend_added else "",
                alpha=0.8, linewidth=0.5 if len(ts) > 50000 else 1.0)
        
        # Plot SD bands for pred columns using std (only for valid values)
        std_col = f'{p_col}_std'
        if std_col in df_results.columns:
            p_std_vals = np.abs(df_results[std_col].values)  # Ensure non-negative
            # Create condition for where to fill (only where predictions are valid)
            valid_pred_mask = np.isfinite(df_results[p_col].values) & np.isfinite(p_std_vals)
            ax.fill_between(ts,
                            df_results[p_col].values - p_std_vals,
                            df_results[p_col].values + p_std_vals,
                            where=valid_pred_mask,
                            color=colors['predicted'], alpha=0.3)
    
    # Set legend_added after trajectory plots
    legend_added = True

    # Calculate timestamp range and fold boundaries for x-axis limits and ticks
    timestamp_min = df_results['timestamp'].min()
    timestamp_max = df_results['timestamp'].max()
    
    # Find fold boundaries (where fold changes) - use first timestamp of each new fold
    # Only consider rows with valid data and valid (non-NaN) fold values
    df_sorted = df_results[valid_mask & df_results['fold'].notna()].sort_values('timestamp').reset_index(drop=True)
    # Vectorized approach: find where fold changes using diff()
    if len(df_sorted) > 0:
        fold_diff = df_sorted['fold'].diff()
        fold_changes = (fold_diff != 0) & fold_diff.notna()
        fold_boundaries = df_sorted.loc[fold_changes, 'timestamp'].tolist()
    else:
        fold_boundaries = []
    
    # Add min and max timestamps to boundaries for complete coverage
    x_ticks = [timestamp_min] + fold_boundaries + [timestamp_max]
    x_ticks = sorted(list(set(x_ticks)))  # Remove duplicates and sort

    # Distributions for each coordinate (only use valid data)
    for i, (t_col, p_col) in enumerate(zip(target_columns, pred_target_columns)):
        ax = axes_dist[i]
        real_vals = df_results.loc[valid_mask, t_col].values
        pred_vals = df_results.loc[valid_mask, p_col].values
        ax.hist(real_vals, bins=50, color=colors['real'], edgecolor='black', alpha=0.5, label=f"Real {t_col}")
        ax.hist(pred_vals, bins=50, color=colors['predicted'], edgecolor='black', alpha=0.5,
                label=f"Predicted {t_col}")

    # Calculate error statistics for mean/median lines (only use valid data)
    error_values = df_results.loc[valid_mask, 'error'].values
    mean_error = np.mean(error_values) if len(error_values) > 0 else 0.0
    median_error = np.median(error_values) if len(error_values) > 0 else 0.0
    mean_unit = f" {unit_name}" if scaler is not None else ""

    # Plot error over time for total error and each coordinate error
    # Reset legend_added for error plots so they get labels
    legend_added = False
    ts = df_results['timestamp'].values
    error_vals = np.ma.masked_invalid(df_results['error'].values)
    ax_error_time.plot(ts, error_vals, color=colors['error_total'], 
                       label="Error (Total)" if not legend_added else "",
                       alpha=0.8)

    for idx, t_col in enumerate(target_columns):
        error_color = colors['error_dim'][idx % len(colors['error_dim'])]
        error_col_vals = np.ma.masked_invalid(df_results[f'{t_col}_error'].values)
        ax_error_time.plot(ts, error_col_vals,
                           label=f"Error ({t_col})" if not legend_added else "",
                           color=error_color, alpha=0.3)
    
    # Add horizontal mean and median lines for total error (always show labels)
    ax_error_time.axhline(mean_error, color='black', linestyle='-', linewidth=2,
                          label=f"Mean = {mean_error:.2f}{mean_unit}")
    ax_error_time.axhline(median_error, color='black', linestyle='--', linewidth=2,
                          label=f"Median = {median_error:.2f}{mean_unit}")


    # Plot Error distribution for total error and each coordinate error (only use valid data)
    if len(error_values) > 0:
        ax_error_dist.hist(error_values, bins=50, color=colors['error_total'], edgecolor='black', alpha=0.7,
                           label="Error (Total)")
        ax_error_dist.axvline(mean_error, color='black', linestyle='-', linewidth=2,
                              label=f"Mean = {mean_error:.2f}{mean_unit}")
        ax_error_dist.axvline(median_error, color='black', linestyle='--', linewidth=2,
                              label=f"Median = {median_error:.2f}{mean_unit}")
    
    # Add error distributions for each coordinate (only use valid data)
    for idx, t_col in enumerate(target_columns):
        error_col_values = df_results.loc[valid_mask, f'{t_col}_error'].values
        error_color = colors['error_dim'][idx % len(colors['error_dim'])]
        ax_error_dist.hist(error_col_values, bins=50, color=error_color, 
                          edgecolor='black', alpha=0.2, label=f"Error ({t_col})")

    # Configure aesthetics
    unit_label = f" ({unit_name})" if scaler is not None else ""
    for i, (ax, t_col) in enumerate(zip(axes_time, target_columns)):
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel(f"{t_col}{unit_label}")
        # Set x limits to min/max timestamp (no padding)
        ax.set_xlim(timestamp_min, timestamp_max)
        # Set x ticks at fold boundaries if available
        if len(x_ticks) > 0:
            ax.set_xticks(x_ticks)
        # Get existing legend handles and labels
        handles, labels = ax.get_legend_handles_labels()
        # Add uncertainty band description if n_seeds is provided
        if n_seeds is not None and n_seeds > 1:
            from matplotlib.patches import Patch
            uncertainty_patch = Patch(facecolor='gray', alpha=0.3, edgecolor='none')
            handles.append(uncertainty_patch)
            labels.append(f"Shaded region = ±1 SD across {n_seeds} seeds")
        ax.legend(handles, labels)
        r2_value = metrics.get(f'r2_{t_col}', metrics.get('r2_pooled', 0.0))
        ax.set_title(f"{t_col} over Time (R²={r2_value:.2f})")
        ax.grid(True)

    for ax, t_col in zip(axes_dist, target_columns):
        ax.set_xlabel(f"{t_col}{unit_label}")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.set_title(f"{t_col} Distribution")
        ax.grid(True)

    ax_error_time.set_xlabel("Time (seconds)")
    ax_error_time.set_ylabel(f"Error{unit_label}")
    # Set x limits to min/max timestamp (no padding)
    ax_error_time.set_xlim(timestamp_min, timestamp_max)
    # Set x ticks at fold boundaries if available
    if len(x_ticks) > 0:
        ax_error_time.set_xticks(x_ticks)
    ax_error_time.legend()
    ax_error_time.set_title("Error over Time")
    ax_error_time.grid(True)

    ax_error_dist.set_xlabel(f"Error{unit_label}")
    ax_error_dist.set_ylabel("Frequency")
    ax_error_dist.legend()
    ax_error_dist.set_title("Error Distribution")
    ax_error_dist.grid(True)

    # Plot 2D trajectory plots per fold
    # Prepare colormap for time-based coloring
    cmap = cm.get_cmap("viridis")
    # Only group by non-NaN folds from valid rows
    # CRITICAL: Filter to only rows where predictions exist (not NaN) to avoid plotting all timestamps for each fold
    # After merge, we may have rows with fold values but NaN predictions - these should be excluded
    df_valid = df_results[valid_mask & df_results['fold'].notna()].copy()
    
    for i, (fold, df_fold) in enumerate(df_valid.groupby('fold')):
        # Ensure data is sorted by timestamp within each fold to match time-series plots
        df_fold = df_fold.sort_values('timestamp').reset_index(drop=True)
        
        # Filter out any rows where predictions are still NaN (shouldn't happen after valid_mask, but double-check)
        # This ensures each fold only plots timestamps where it actually has predictions
        pred_notna = df_fold[pred_target_columns].notna().all(axis=1)
        df_fold = df_fold[pred_notna].reset_index(drop=True)
        
        if len(df_fold) == 0:
            logger.warning(f"No valid prediction data for fold {fold}. Skipping trajectory plot.")
            continue
        
        # Use the valid_mask subset for this fold (all rows in df_fold are already valid)
        fold_targets = df_fold[target_columns].values
        fold_predictions = df_fold[pred_target_columns].values
        fold_metrics = calculate_metrics(fold_targets, fold_predictions, target_columns=target_columns)
        fold_r2 = fold_metrics['r2_pooled']
        fold_rmse = np.sqrt(fold_metrics['mse'])

        ax_2d_real = fig.add_subplot(gs[len(target_columns) + 1, i])  # Row after error plots
        ax_2d_pred = fig.add_subplot(gs[len(target_columns) + 2, i])  # Row after real trajectory

        # Real trajectory - no metrics in title
        # All rows in df_fold are already valid (from valid_mask)
        real_trajectory = df_fold[target_columns].values
        real_folds = df_fold['fold'].values
        
        # For very large datasets, reduce scatter size to avoid overplotting
        scatter_size = 0.5 if len(real_trajectory) > 10000 else 1.0
        
        # Both real_trajectory and boundary_points are already in cm space (inverse transformed above in plot_advanced_line_plot_over_time)
        # Don't pass scaler to plot_trajectory to avoid double transformation
        plot_trajectory(
            real_trajectory, folds=real_folds, pos_range=pos_range,
            boundary_points=boundary_points,
            scaler=None,  # Don't transform again - already in cm space
            scatter_size=scatter_size,
            title=f'Fold {int(fold)}\nReal Trajectory',
            ax_in=ax_2d_real,
            reverse_y=False
        )

        # Predicted trajectory - show R² and RMSE in 2-line title
        # All rows in df_fold are already valid (from valid_mask)
        pred_trajectory = df_fold[pred_target_columns].values
        pred_folds = df_fold['fold'].values
        
        rmse_unit = f" {unit_name}" if scaler is not None else ""
        # Both pred_trajectory and boundary_points are already in cm space (inverse transformed above)
        # Don't pass scaler to plot_trajectory to avoid double transformation
        plot_trajectory(
            pred_trajectory, folds=pred_folds, pos_range=pos_range,
            boundary_points=boundary_points,
            scaler=None,  # Don't transform again - already in cm space
            scatter_size=scatter_size,
            title=f'R²={fold_r2:.2f}, RMSE={fold_rmse:.2f}{rmse_unit}\nPredicted Trajectory',
            ax_in=ax_2d_pred,
            reverse_y=False
        )

    # Add colorbar in the same row as real trajectories, in the last column (n_folds)
    ax_colorbar = fig.add_subplot(gs[len(target_columns) + 1, n_folds])  # Same row as real trajectories, last column
    ax_colorbar.set_axis_off()  # Hide axes
    # Create a ScalarMappable for the colorbar
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_colorbar, orientation='horizontal', location='top', pad=0.05)
    # Set only two ticks: 'start' and 'end'
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['start', 'end'])
    cbar.set_label('Time', rotation=0, labelpad=5)

    # Add title with metrics (use main metrics, no group_metrics)
    rmse_unit_str = f" {unit_name}" if scaler is not None else ""
    main_rmse = np.sqrt(metrics.get('mse', 0))
    r2_pooled = metrics.get('r2_pooled', 0.0)
    

    enhanced_title = f"{title}\nR²={r2_pooled:.3f}, RMSE={main_rmse:.3f}{rmse_unit_str}"
    fig.suptitle(enhanced_title, fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return plt.gcf()


@save_plot
def visualize_model_parameters(model_handler, title='Box Plot of Model Parameters', range=None, grad_flag=False, save_params={}):
    """
    Visualizes the distribution of model parameters from a model handler using Seaborn box plots.

    Args:
        model_handler: An instance of ModelHandler with a `get_model_state_dict` method.
        title: Title for the plot.
        range: Optional range for the y-axis.
        grad_flag: Flag to indicate if parameters requiring gradients should be shown in a different color.
        save_params: Optional dictionary of parameters for saving the plot.
    """
    # Get the model's state dictionary
    param_dict = model_handler.get_model_state_dict()
    # Get named parameters
    named_params = model_handler.get_named_parameters()

    # Prepare data for visualization
    param_data = []
    param_names = []
    param_grad = []

    # for named parameters check if grad is required
    for name, param in named_params:
        flattened_values = param.flatten().detach().cpu().numpy()
        param_data.extend(flattened_values)
        param_names.extend([name] * len(flattened_values))
        param_grad.extend([param.requires_grad] * len(flattened_values))

    # for name, param in param_dict.items():
    #     if param.ndimension() > 0:  # Only process tensors with dimensions
    #         flattened_values = param.flatten().cpu().numpy()
    #         param_data.extend(flattened_values)
    #         param_names.extend([name] * len(flattened_values))  # Repeat name for each value
    #         param_grad_flags.extend([param.requires_grad] * len(flattened_values))  # Repeat grad flag for each value

    # Create a DataFrame for Seaborn
    data = pd.DataFrame({'Parameter': param_names, 'Value': param_data, 'RequiresGrad': param_grad})

    # Plot using Seaborn
    plt.figure(figsize=(16, 8))
    if grad_flag:
        sns.boxplot(data=data, x='Parameter', y='Value', hue='RequiresGrad', showmeans=True,
                    meanprops={'marker': 'o', 'markerfacecolor': 'green', 'markeredgecolor': 'black'},
                    medianprops={'color': 'red'},
                    palette={True: 'lightcoral', False: 'lightblue'})
    else:
        sns.boxplot(data=data, x='Parameter', y='Value', showmeans=True,
                    meanprops={'marker': 'o', 'markerfacecolor': 'green', 'markeredgecolor': 'black'},
                    medianprops={'color': 'red'},
                    palette='lightblue')

    # Customize plot
    plt.xticks(rotation=90, ha='right', fontsize=9)
    plt.xlabel('Parameter Names')
    plt.ylabel('Parameter Values')
    plt.title(title)
    if range: plt.ylim(range)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    return plt.gcf()


@save_plot
def plot_hyperparameter_violin_plots(df, hyperparameters=None, target_column='r2_var', target_label='R2 (%)', percentage=True, max_columns=8, title='Hyperparameter Violin Plots', save_params={}):
    """
    Plots violin plots for each hyperparameter against the target performance metric.
    Marks the min and max values for each unique hyperparameter value.

    Parameters:
        df (DataFrame): The DataFrame containing hyperparameter information.
        hyperparameters (list): List of hyperparameters to plot. Defaults to ['model.batch_size', 'model.optimizer.learning_rate'].
        target_column (str): The target column to plot against.
        percentage (bool): Whether to convert the target column to percentage.
        title (str): The title of the plot.
        save_params (dict): Parameters for saving the plot.
        max_columns (int): Maximum number of columns per row in subplots.

    Returns:
        fig: The matplotlib figure object.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create a list of hyperparameter columns
    if hyperparameters is None:
        hyperparameters = ['model.batch_size', 'model.optimizer.learning_rate']
    df = df.copy()

    # Ensure hyperparameters are categorical
    for hyperparam in hyperparameters:
        df[hyperparam] = df[hyperparam].astype('category')

    # Convert target_column to percentage if needed
    if percentage:
        df[target_column] = df[target_column] * 100

    sns.set(style='whitegrid')

    num_hyperparameters = len(hyperparameters)

    # Determine number of rows and columns based on max_columns
    ncols = min(num_hyperparameters, max_columns)
    nrows = int(np.ceil(num_hyperparameters / max_columns))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)  # Always 2D array of axes

    pastel_colors = sns.color_palette("pastel", num_hyperparameters)

    for idx, (hyperparam, color) in enumerate(zip(hyperparameters, pastel_colors)):
        row_idx = idx // max_columns
        col_idx = idx % max_columns
        ax = axes[row_idx, col_idx]

        sns.violinplot(x=hyperparam, y=target_column, data=df, ax=ax, inner='quartile', color=color, legend=False)

        ax.set_xlabel(hyperparam, fontsize=10)
        ax.set_ylabel(target_label, fontsize=10)
        ax.set_ylim(0, df[target_column].max() * 1.1)

        # Group by hyperparameter and annotate min/max
        min_max_values = df.groupby(hyperparam, observed=True)[target_column].agg(['min', 'max']).reset_index()

        for idx2, row in min_max_values.iterrows():
            value = row[hyperparam]
            min_val = row['min']
            max_val = row['max']

            x_pos = list(min_max_values[hyperparam]).index(value)

            ax.annotate(f'{min_val:.2f}%', xy=(x_pos, min_val),
                        xytext=(x_pos + 0.1, min_val + 2),
                        arrowprops=dict(arrowstyle='->', color='black'),
                        fontsize=9, color='black')

            ax.annotate(f'{max_val:.2f}%', xy=(x_pos, max_val),
                        xytext=(x_pos + 0.1, max_val - 2),
                        arrowprops=dict(arrowstyle='->', color='black'),
                        fontsize=9, color='black')

    # Hide unused subplots
    total_subplots = nrows * ncols
    if num_hyperparameters < total_subplots:
        for i in range(num_hyperparameters, total_subplots):
            row_idx = i // max_columns
            col_idx = i % max_columns
            fig.delaxes(axes[row_idx][col_idx])

    fig.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()

    return fig


@save_plot
def plot_ranked_score_pairplot(cells_ranks_df, vars=['order', 'cell', 'R2', 'R2_modified', 'z_score', 'z_score_modified'], title="Pairplot of Ranked Scores for Each Cell in Each Room", save_params={}):
    """
    Generate a seaborn pairplot showing the relationship between ranked cell scores across rooms.

    Parameters:
        cells_ranks_df (pd.DataFrame): DataFrame with per-cell scores.
        vars (list): List of columns to include in the pairplot.
        title (str): Title of the plot. Default is a descriptive string.
        save_params (dict): Dictionary of save parameters passed to the @save_plot decorator.

    Returns:
        matplotlib.figure.Figure: The resulting figure object.
    """

    # sns.set(style="whitegrid")

    df = cells_ranks_df.copy()
    if 'order' in df.columns:
        df['order'] = df.groupby('room')['z_score_modified'].rank(ascending=False).astype(int)

    # Ensure all columns in `vars` are numeric
    df[vars] = df[vars].apply(pd.to_numeric, errors='coerce')

    pairplot = sns.pairplot(
        df,
        hue='room',
        vars=vars,
        markers='o',
        palette='tab10',
        diag_kind='kde',
        plot_kws={'alpha': 0.5, 's': 10, 'edgecolor': 'w'},
        diag_kws={'fill': True, 'alpha': 0.5}
    )
    # Add title and adjust layout to prevent cutting
    pairplot.fig.suptitle(title, fontsize=14)
    pairplot.fig.subplots_adjust(top=0.95)  # leave room for suptitle
    plt.tight_layout()

    return pairplot.fig

@save_plot
def plot_subset_r2_curve_by_room(ranked_scores_df, baseline_r2, rooms, n_cells,
                                  custom_subset_length=None,
                                  title="Minimal Subset of Cells to Recover Performance",
                                  save_params={}):
    """
    Plot R² score as a function of subset size per room, including full-model baselines and optionally highlight a custom subset.

    Parameters:
        ranked_scores_df (pd.DataFrame): DataFrame with columns ['room', 'subset_size', 'R2'].
        baseline_r2 (dict): Dictionary mapping room name to baseline R² score (full model).
        rooms (list): List of room names to include in the plot.
        n_cells (int): Maximum number of neurons per room.
        custom_subset_length (int or None): If provided, highlights subset performance at this size.
        title (str): Plot title.
        save_params (dict): Parameters used by @save_plot (e.g., {'config':..., 'path':...}).

    Returns:
        matplotlib.figure.Figure: The resulting figure.
    """

    colors = sns.color_palette("tab10", n_colors=len(rooms))
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, room in enumerate(rooms):
        room_df = ranked_scores_df[ranked_scores_df["room"] == room]
        color = colors[i]

        # Plot R² curve
        sns.lineplot(data=room_df, x="subset_size", y="R2", ax=ax,
                     color=color, marker='o', markersize=4, linewidth=1.5, label=f"Ranked Subsets ({room})")

        # Plot baseline line
        baseline = baseline_r2.get(room)
        if baseline is not None:
            ax.plot([-10, n_cells], [baseline, baseline], linestyle='--',
                    color=color, linewidth=1.5, marker='o', markersize=8, label=f"All Cells ({room})")

        # Highlight custom subset point if requested
        if custom_subset_length is not None:
            custom_row = room_df[room_df["subset_size"] == custom_subset_length]
            if not custom_row.empty:
                r2_val = custom_row["R2"].values[0]

                # Vertical and horizontal lines
                ax.axvline(custom_subset_length, color=color, linestyle='-', alpha=0.3)
                ax.axhline(r2_val, color=color, linestyle='-', alpha=0.3)

                # Highlighted point
                ax.plot(custom_subset_length, r2_val, marker='o', markersize=8,
                        color=color, markeredgecolor='black',
                        label=f"Top {custom_subset_length} Cells ({room})")

    ax.set_xlim(0, n_cells + 1)
    ax.set_xlabel("Number of Cells Used")
    ax.set_ylabel("R² Score")
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1), title="Rooms")
    fig.tight_layout()
    return fig


@save_plot
def plot_subset_r2_histogram(subset_scores_df,
                             ranked_scores_df=None,
                             custom_subset_length=None,
                             title="Distribution of R² Scores by Subset Size",
                             save_params={}):
    """
    Plot histograms of R² scores by subset size and room, with optional lines for top-ranked and custom subset scores.

    Parameters:
        subset_scores_df (pd.DataFrame): Must include ['room', 'size', 'R2', 'is_baseline'].
        ranked_scores_df (pd.DataFrame or None): Must include ['room', 'subset_size', 'R2'] if provided.
        custom_subset_length (int or None): Highlight a fixed subset size across all rooms.
        title (str): Plot title.
        save_params (dict): For @save_plot decorator (e.g., {'config':..., 'path':...}).

    Returns:
        matplotlib.figure.Figure: The resulting figure.
    """

    df = subset_scores_df[subset_scores_df['is_baseline'] == False].copy()
    df = df[df["R2"].notna()].copy()
    df["R2"] = df["R2"].clip(lower=0.0)

    subset_sizes = sorted(df["size"].unique())
    rooms = sorted(df["room"].unique())

    n_rows = len(subset_sizes)
    n_cols = len(rooms)
    palette = sns.color_palette("husl", n_colors=n_rows)

    bins = np.linspace(0, 1, 21)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), squeeze=False)

    for i, size in enumerate(subset_sizes):
        for j, room in enumerate(rooms):
            ax = axes[i][j]
            df_subset = df[(df["size"] == size) & (df["room"] == room)]

            sns.histplot(df_subset["R2"], bins=bins, kde=False, color=palette[i], ax=ax, edgecolor='black')

            ax.set_xlim(0, 1)
            ax.set_title(f"Size = {size}, Room = {room}")
            ax.set_xlabel("R² Score")
            ax.set_ylabel("Frequency")
            ax.grid(True)

            # Top-ranked subset (red dashed line)
            if ranked_scores_df is not None:
                top_row = ranked_scores_df[(ranked_scores_df["subset_size"] == size) & (ranked_scores_df["room"] == room)]
                if not top_row.empty:
                    top_r2 = top_row["R2"].values[0]
                    ax.axvline(top_r2, color='red', linestyle='--', linewidth=1.5)

            # Custom subset marker (green dashed line and dot)
            if ranked_scores_df is not None and custom_subset_length is not None:
                custom_row = ranked_scores_df[(ranked_scores_df["subset_size"] == custom_subset_length) & (ranked_scores_df["room"] == room)]
                if not custom_row.empty:
                    custom_r2 = custom_row["R2"].values[0]
                    ax.axvline(custom_r2, color='green', linestyle='--', linewidth=1.5)
                    ax.plot(custom_r2, 0, 'o', color='green', markersize=6)

    # Create a shared legend at the bottom
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linestyle='--', label='Top-ranked subset'),
        Line2D([0], [0], color='green', linestyle='--', marker='o', label=f'Selected size = {custom_subset_length}')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])  # Leave space for legend
    return fig




@save_plot
def plot_score_distribution_by_subset_size(subset_scores_df,
                                           score_column="z_score",
                                           title="Distribution of Scores by Subset Size",
                                           show_median_line=True,
                                           save_params={}):
    """
    Plot a violin plot showing the distribution of a specified score (e.g., Z-score or R²)
    across different subset sizes, with optional median trend line.

    Parameters:
        subset_scores_df (pd.DataFrame): DataFrame with 'size' and the selected score column.
        score_column (str): Name of the column to plot on the y-axis. Default is 'z_score'.
        title (str): Plot title.
        show_median_line (bool): Whether to overlay a line connecting medians across subset sizes.
        save_params (dict): Parameters for @save_plot (e.g., {'config':..., 'path':...}).

    Returns:
        matplotlib.figure.Figure: The resulting figure.
    """
    if score_column not in subset_scores_df.columns:
        raise ValueError(f"'{score_column}' not found in subset_scores_df.")

    plt.figure(figsize=(7, 4))
    ax = sns.violinplot(data=subset_scores_df,
                        x="size",
                        y=score_column,
                        palette="pastel",
                        inner="box",
                        cut=0)

    if show_median_line:
        # Ensure correct alignment by mapping subset_size to categorical positions
        sizes_sorted = sorted(subset_scores_df["size"].unique())
        medians = subset_scores_df.groupby("size")[score_column].median()
        x_positions = [sizes_sorted.index(s) for s in medians.index]
        ax.plot(x_positions, medians.values, color='black', linestyle='--', marker='o', label="Median")

    ax.set_title(title)
    ax.set_xlabel("Subset Size (Number of Cells)")
    ax.set_ylabel(score_column.replace("_", " ").title())
    ax.grid(True, axis='y')

    if show_median_line:
        ax.legend()

    plt.tight_layout()
    return plt.gcf()



@save_plot
def plot_participation_histograms_by_subset_size(subsets_df,
                                                  n_cols=2,
                                                  title="Participation Frequency by Subset Size",
                                                  save_params={}):
    """
    Plot histograms of cell participation frequency for each subset size.

    Parameters:
        subsets_df (pd.DataFrame): Includes 'cells' (list of cell indices) and 'size' (subset size).
        n_cols (int): Number of subplot columns. Default is 2.
        title (str): Figure title.
        save_params (dict): For @save_plot decorator (e.g., {'config':..., 'path':...}).

    Returns:
        matplotlib.figure.Figure: The resulting figure.
    """

    # Explode 'cells' list into long format
    df_exploded = subsets_df[subsets_df["is_baseline"] == False][["cells", "size"]].explode("cells").rename(
        columns={"cells": "cell"}
    )

    # Count participation of each cell per subset size
    participation_counts = (
        df_exploded.groupby(["size", "cell"])
        .size()
        .reset_index(name="count")
    )

    subset_sizes = sorted(participation_counts["size"].unique())
    n_plots = len(subset_sizes)
    n_rows = (n_plots + n_cols - 1) // n_cols
    palette = sns.color_palette("husl", n_colors=n_plots)

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), squeeze=False)
    axes = axes.flatten()

    for i, subset_size in enumerate(subset_sizes):
        ax = axes[i]
        df_subset = participation_counts[participation_counts["size"] == subset_size]
        sns.histplot(df_subset["count"], bins=10, kde=False, color=palette[i], ax=ax)

        ax.set_title(f"Subset Size = {subset_size}")
        ax.set_xlabel("Subset Participation Count")
        ax.set_ylabel("Number of Cells")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.grid(True)

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig



@save_plot
def plot_ranked_cell_score_histogram(cells_ranks_df,
                                     top_n=25,
                                     score_col="z_score_modified",
                                     title="Histogram of Ranked Cell Scores",
                                     save_params={}):
    """
    Plot histogram of cell scores, highlighting top-N ranked cells and marking their threshold.

    Parameters:
        cells_ranks_df (pd.DataFrame): Must include ['cell', 'room', score_col, 'order'].
        top_n (int): Number of top cells to highlight. Default is 25.
        score_col (str): Score column to histogram. Default is 'z_score_modified'.
        title (str): Plot title.
        save_params (dict): For @save_plot decorator (e.g., {'config':..., 'path':...}).

    Returns:
        matplotlib.figure.Figure: The resulting figure.
    """

    # Sort cells by ranking order
    df_sorted = cells_ranks_df.sort_values(by=score_col, ascending=False).reset_index(drop=True)
    # top_cells = df_sorted.iloc[:top_n]
    # rest_cells = df_sorted.iloc[top_n:]

    # # Determine bin range and edges
    # all_scores = df_sorted[score_col].dropna()
    # min_val, max_val = np.floor(all_scores.min()), np.ceil(all_scores.max())
    # bins = np.linspace(min_val, max_val, num=21)

    # Use fixed range and bins
    # bins = np.linspace(0, 1, 21)

    fig, ax = plt.subplots(figsize=(6, 4))

    # # Plot histograms using consistent binning and range
    # sns.histplot(rest_cells[score_col], color='skyblue', alpha=0.7, #, bins=bins
    #              edgecolor='black', label='Other Cells', ax=ax)
    #
    # if top_n > 0:
    #     sns.histplot(top_cells[score_col], color='crimson', alpha=0.8, #, bins=bins
    #                  edgecolor='black', label=f'Top {top_n} Cells', ax=ax)

    # Plot a single histogram with both top and rest cells using stacked bars
    df_sorted['is_top'] = (df_sorted.index < top_n).astype(int)
    sns.histplot(df_sorted, x=score_col, hue='is_top', multiple='stack', palette=['skyblue', 'crimson'],
                 edgecolor='black', stat='count', ax=ax)

    # Add a custom legend for the colors
    # ax.legend(title="Cell Ranking", labels=["Other Cells", f"Top {top_n} Cells"])

    if top_n > 0:
        threshold_score = df_sorted.loc[top_n - 1:top_n, score_col].mean()
        ax.axvline(threshold_score, color='red', linestyle='--', linewidth=1,
                   label=f"Top {top_n} Cells Threshold")

    # ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.set_xlabel(score_col.replace("_", " ").title())
    ax.set_ylabel("Number of Cells")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


@save_plot
def plot_ranked_cell_scores(cells_ranks_df,
                             top_n=25,
                             score_col="z_score_modified",
                             rooms_to_compare=["A", "B"],
                             title="Ranked Cell Scores by Room",
                             save_params={}):
    """
    Enhanced plot of ranked cell scores across rooms, including:
    - Venn diagram of shared top-N cells (top-left)
    - Rank correlation scatter of score_col values in two rooms (top-right)
    - Labeled top-N cell scores for each room (bottom-left)
    - Full score curves for all cells (bottom-right)

    Parameters:
        cells_ranks_df (pd.DataFrame): Must include ['cell', 'room', and score_col].
        top_n (int): Number of top cells to highlight per room. Default is 25.
        score_col (str): Which score to plot on Y-axis. Default is 'z_score_modified'.
        rooms_to_compare (list): List of two room names to compare in scatter and venn. Default is ["A", "B"].
        title (str): Plot title.
        save_params (dict): For @save_plot decorator (e.g., {'config':..., 'path':...}).

    Returns:
        matplotlib.figure.Figure: The resulting figure.
    """

    if score_col not in cells_ranks_df.columns:
        raise ValueError(f"'{score_col}' not found in DataFrame.")
    if len(rooms_to_compare) != 2:
        raise ValueError("rooms_to_compare must contain exactly two room names.")

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), gridspec_kw={"height_ratios": [2, 1], "width_ratios": [2, 2]})
    ax_venn, ax_scatter, ax_top, ax_all = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    palette = sns.color_palette("tab10")
    rooms = sorted(cells_ranks_df["room"].unique())
    texts = []
    cell_points = []

    for i, room in enumerate(rooms):
        df_room = cells_ranks_df[cells_ranks_df["room"] == room].copy()
        df_room = df_room.sort_values(by=score_col, ascending=False).reset_index(drop=True)
        x = np.arange(len(df_room))
        y = df_room[score_col].values
        cell_labels = df_room["cell"].values

        # BOTTOM LEFT: plot top-N cells with labels
        ax_top.plot(x[:top_n], y[:top_n], color=palette[i], linewidth=2, label=room)
        ax_top.scatter(x[:top_n], y[:top_n], color=palette[i], s=30, edgecolor='black', zorder=5)

        for xi, yi, label in zip(x[:top_n], y[:top_n], cell_labels[:top_n]):
            texts.append(ax_top.text(xi, yi, str(label), fontsize=8, color=palette[i]))
            cell_points.append({"cell": label, "x": xi, "y": yi, "room": room})

        # BOTTOM RIGHT: full curve
        ax_all.plot(x, y, color=palette[i], linewidth=1.5, label=room)
        ax_all.scatter(x[:top_n], y[:top_n], color=palette[i], s=20, edgecolor='black', zorder=5)

    # Connect shared top-N cells
    grouped_cells = defaultdict(list)
    for pt in cell_points:
        grouped_cells[pt["cell"]].append(pt)

    for cell_id, pts in grouped_cells.items():
        if len(pts) > 1:
            pts_sorted = sorted(pts, key=itemgetter("y"), reverse=True)
            x_vals = [pt["x"] for pt in pts_sorted]
            y_vals = [pt["y"] for pt in pts_sorted]
            ax_top.plot(x_vals, y_vals, linestyle="--", color="gray", linewidth=0.7, alpha=0.6, zorder=1)

    adjust_text(texts, ax=ax_top, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))

    # Final formatting
    ax_top.set_title("Top-Ranked Cells")
    ax_top.set_xlabel("Cell Rank")
    ax_top.set_ylabel(score_col.replace("_", " ").title())
    ax_top.grid(True)

    ax_all.set_title("All Cells")
    ax_all.set_xlabel("Cell Rank")
    ax_all.grid(True)
    ax_all.legend()

    # Top scatterplot of A vs B
    room_a, room_b = rooms_to_compare
    # df_a = cells_ranks_df[cells_ranks_df["room"] == room_a].copy()
    # df_b = cells_ranks_df[cells_ranks_df["room"] == room_b].copy()
    # merged = pd.merge(df_a[["cell", score_col]], df_b[["cell", score_col]], on="cell", suffixes=(f"_{room_a}", f"_{room_b}"))
    # top_a = df_a.sort_values(by=score_col, ascending=False).head(top_n)
    # top_b = df_b.sort_values(by=score_col, ascending=False).head(top_n)

    pivoted = cells_ranks_df.pivot_table(index="cell", columns="room", values=score_col)
    pivoted = pivoted[[room_a, room_b]].dropna().reset_index()
    pivoted.columns = [f"{score_col}_{room}" if room in [room_a, room_b] else "cell" for room in pivoted.columns]

    top_a = pivoted.sort_values(by=f"{score_col}_{room_a}", ascending=False).head(top_n)
    top_b = pivoted.sort_values(by=f"{score_col}_{room_b}", ascending=False).head(top_n)
    top_cells_a = set(top_a["cell"])
    top_cells_b = set(top_b["cell"])

    pivoted['color'] = 'lightgray'
    for idx, row in pivoted.iterrows():
        if row["cell"] in top_cells_a and row["cell"] in top_cells_b:
            pivoted.at[idx, 'color'] = 'black'  # shared top cells
        elif row["cell"] in top_cells_a:
            pivoted.at[idx, 'color'] = palette[0]  # top cells in room A
        elif row["cell"] in top_cells_b:
            pivoted.at[idx, 'color'] = palette[1]  # top cells in room B

    # for _, row in merged.iterrows():
    for _, row in pivoted.iterrows():
        ax_scatter.scatter(row[f"{score_col}_{room_a}"], row[f"{score_col}_{room_b}"],
                           color=row["color"], s=20, edgecolor='black' if row["color"] != "gray" else 'none', alpha=0.8)

    ax_scatter.set_title(f"{score_col.replace('_', ' ').title()} Comparison")
    ax_scatter.set_xlabel(f"{score_col.replace('_', ' ').title()} - Room {room_a}")
    ax_scatter.set_ylabel(f"{score_col.replace('_', ' ').title()} - Room {room_b}")
    ax_scatter.grid(True)

    venn2([top_cells_a, top_cells_b], set_labels=(f"{room_a}", f"{room_b}"), set_colors=(palette[0], palette[1]), ax=ax_venn)
    ax_venn.set_title("Top-N Cell Overlap")

    fig.suptitle(title, fontsize=14, y=0.95)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


@save_plot
def plot_filtering_sankey(filter_stages, filter_labels, title="Cell Filtering Process", save_params={}):
    import numpy as np
    import plotly.graph_objects as go

    n_cells = len(filter_stages[0])
    init_mask = filter_stages[0]
    final_mask = filter_stages[-1]
    n_layers = len(filter_stages)
    last_layer = n_layers - 1

    # Colors
    color_kept = "rgba(50, 200, 50, 0.8)"       # Green
    color_dropped = "rgba(220, 60, 60, 0.6)"    # Red
    color_node_pass = "rgba(150, 255, 150, 0.4)"
    color_node_fail = "rgba(255, 150, 150, 0.4)"
    color_node_all = "lightgray"

    node_labels = []
    node_colors = []
    node_ids = {}
    group_masks = {}
    node_idx = 0

    ### Add Pass/Fail nodes per layer (starting from layer 1)
    for i in range(0, n_layers-1):
        fmask = filter_stages[i]
        for status in ['fail','pass']:
            mask = fmask if status == 'pass' else ~fmask
            count = mask.sum()
            if count == 0:
                continue
            pct = 100 * count / n_cells
            if i == 0:
                label = f"All Cells<br>{count}<br>{pct:.1f}%"
            else:
                label = f"{status.capitalize()}<br>{count}<br>{pct:.1f}%"
            node_labels.append(label)
            node_colors.append(color_node_pass if status == 'pass' else color_node_fail)
            node_ids[(i, status)] = node_idx
            group_masks[(i, status)] = mask
            node_idx += 1

    ### Add Final Outcome nodes (Kept / Dropped)
    for kept in [False,True]:
        mask = final_mask if kept else ~final_mask
        count = mask.sum()
        if count == 0:
            continue
        pct = 100 * count / n_cells
        label = f"{'Kept' if kept else 'Dropped'}<br>{count}<br>{pct:.1f}%"
        node_labels.append(label)
        node_colors.append(color_kept if kept else color_dropped)
        node_ids[(last_layer, kept)] = node_idx
        group_masks[(last_layer, kept)] = mask
        node_idx += 1

    ### Build flows
    sources, targets, values, colors = [], [], [], []

    # Layer → Layer
    for i in range(0, n_layers - 1):
        for status1 in ['fail','pass']:
            for status2 in ['fail','pass']:
                src_key = (i, status1)
                tgt_key = (i + 1, status2)
                if src_key not in group_masks or tgt_key not in group_masks:
                    continue
                mask = group_masks[src_key] & group_masks[tgt_key]
                for kept in [False,True]:
                    flow_mask = mask & (final_mask if kept else ~final_mask)
                    count = flow_mask.sum()
                    if count == 0:
                        continue
                    sources.append(node_ids[src_key])
                    targets.append(node_ids[tgt_key])
                    values.append(count)
                    colors.append(color_kept if kept else color_dropped)

    # Last Layer → Final Outcome
    for status in ['fail','pass']:
        src_key = (last_layer-1, status)
        if src_key not in group_masks:
            continue
        mask = group_masks[src_key]
        for kept in [False,True]:
            flow_mask = mask & (final_mask if kept else ~final_mask)
            count = flow_mask.sum()
            if count == 0:
                continue
            sources.append(node_ids[src_key])
            targets.append(node_ids[(last_layer, kept)])
            values.append(count)
            colors.append(color_kept if kept else color_dropped)

    ### Plot
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=60,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=colors,
            hovertemplate="%{value} cells<extra></extra>"
        )
    ))

    # Add filter stage titles (centered between columns)
    x_positions = np.linspace(0.005 , 0.995, n_layers)
    annotations = []
    filter_labels = filter_labels
    for i, label in enumerate(filter_labels):
        annotations.append(dict(
            x=x_positions[i],
            y=1.07,
            text=f"<b>{label.replace(' ', '<br>')}</b>",
            showarrow=False,
            xanchor='center',
            yanchor='bottom',
            font=dict(size=14)
        ))

    fig.update_layout(
        title_text=title,
        font_size=11,
        height=400,
        margin=dict(t=140, r=40),
        annotations=annotations
    )

    return fig


# todo: temp until check with shai
@save_plot
def compare_polar_distances(
    df_data,
    df_data_polar,
    angles_deg,
    room_col='room',
    by_room=True,
    id_col='timestamp',
    make_plots=True,
    sample_scatter=20000,
    title_prefix='Polar distance check',
    save_params={}
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    assert len(angles_deg) > 0, "angles_deg must be non-empty"

    def _resolve_angle_columns(dfA, dfB, angles_deg, prefix):
        cols = []
        for a in angles_deg:
            c1 = f"{prefix}_{int(round(a))}"
            c2 = f"{prefix}_{int(round(a)):03d}"
            if c1 in dfA.columns and c1 in dfB.columns:
                cols.append(c1)
            elif c2 in dfA.columns and c2 in dfB.columns:
                cols.append(c2)
            else:
                cols.append(None)
        idx_keep = [i for i, c in enumerate(cols) if c is not None]
        cols_kept = [cols[i] for i in idx_keep]
        angles_kept = np.array([angles_deg[i] for i in idx_keep], dtype=float)
        return cols_kept, angles_kept

    def _align_frames(df_true, df_calc, id_col='timestamp'):
        if id_col and id_col in df_true.columns and id_col in df_calc.columns:
            merged = df_true[[id_col]].merge(df_calc[[id_col]], on=id_col, how='inner', suffixes=('_x', '_y'))
            df_true_aligned = df_true.loc[merged.index]
            df_calc_aligned = df_calc.loc[merged.index]
            df_true_aligned = df_true_aligned.set_index(id_col, drop=False)
            df_calc_aligned = df_calc_aligned.set_index(id_col, drop=False)
            return df_true_aligned, df_calc_aligned
        common_idx = df_true.index.intersection(df_calc.index)
        return df_true.loc[common_idx], df_calc.loc[common_idx]

    # 1) Align frames
    df_true, df_calc = _align_frames(df_data, df_data_polar, id_col=id_col)

    # 2) Resolve columns
    allo_cols, allo_angles = _resolve_angle_columns(df_true, df_calc, angles_deg, 'Allo')
    ego_cols,  ego_angles  = _resolve_angle_columns(df_true, df_calc, angles_deg, 'Ego')
    if len(allo_cols) == 0 and len(ego_cols) == 0:
        raise ValueError("No overlapping Allo_* or Ego_* columns found in both dataframes.")

    def _compute_err_stats(true_mat, calc_mat, angle_labels):
        true_mat = np.asarray(true_mat, dtype=float)
        calc_mat = np.asarray(calc_mat, dtype=float)
        valid = (true_mat >= 0) & (calc_mat >= 0)
        err = np.where(valid, calc_mat - true_mat, np.nan)
        mae_per  = np.nanmean(np.abs(err), axis=0)
        rmse_per = np.sqrt(np.nanmean(err**2, axis=0))
        overall_mae  = float(np.nanmean(np.abs(err)))
        overall_rmse = float(np.sqrt(np.nanmean(err**2)))
        return {
            'overall_mae': overall_mae,
            'overall_rmse': overall_rmse,
            'mae_per_angle': pd.Series(mae_per, index=angle_labels),
            'rmse_per_angle': pd.Series(rmse_per, index=angle_labels),
        }

    # 3) Global stats
    metrics = {}
    if len(allo_cols) > 0:
        metrics['Allo'] = _compute_err_stats(df_true[allo_cols].to_numpy(float),
                                             df_calc[allo_cols].to_numpy(float),
                                             allo_angles)
    if len(ego_cols) > 0:
        metrics['Ego']  = _compute_err_stats(df_true[ego_cols].to_numpy(float),
                                             df_calc[ego_cols].to_numpy(float),
                                             ego_angles)

    # 3b) Per-room -1 stats (counts + %)
    def _minus_one_stats(cols):
        stats = {}
        if by_room and room_col in df_true.columns:
            for room, idx in df_true.groupby(room_col).groups.items():
                t_mat = df_true.loc[idx, cols].to_numpy(float)
                c_mat = df_calc.loc[idx, cols].to_numpy(float)
                n_entries = t_mat.size
                t_cnt = int(np.sum(t_mat < 0))
                c_cnt = int(np.sum(c_mat < 0))
                stats[room] = {
                    'true_count': t_cnt,
                    'calc_count': c_cnt,
                    'true_pct': (100.0 * t_cnt / n_entries) if n_entries else np.nan,
                    'calc_pct': (100.0 * c_cnt / n_entries) if n_entries else np.nan,
                    'total_entries': int(n_entries),
                }
        else:
            # overall (single “All” bucket)
            t_mat = df_true[cols].to_numpy(float)
            c_mat = df_calc[cols].to_numpy(float)
            n_entries = t_mat.size
            t_cnt = int(np.sum(t_mat < 0))
            c_cnt = int(np.sum(c_mat < 0))
            stats['All'] = {
                'true_count': t_cnt, 'calc_count': c_cnt,
                'true_pct': (100.0 * t_cnt / n_entries) if n_entries else np.nan,
                'calc_pct': (100.0 * c_cnt / n_entries) if n_entries else np.nan,
                'total_entries': int(n_entries),
            }
        return stats

    minus_allo = _minus_one_stats(allo_cols) if len(allo_cols) else {}
    minus_ego  = _minus_one_stats(ego_cols)  if len(ego_cols)  else {}

    # 4) Per-room MAE/RMSE
    rooms_list = []
    if by_room and room_col in df_true.columns:
        metrics['by_room'] = {}
        rooms_list = list(pd.unique(df_true[room_col].dropna()))
        for room, idx in df_true.groupby(room_col).groups.items():
            r_true = df_true.loc[idx]; r_calc = df_calc.loc[idx]
            per_room = {}
            if len(allo_cols) > 0:
                per_room['Allo'] = _compute_err_stats(r_true[allo_cols].to_numpy(float),
                                                      r_calc[allo_cols].to_numpy(float),
                                                      allo_angles)
            if len(ego_cols) > 0:
                per_room['Ego']  = _compute_err_stats(r_true[ego_cols].to_numpy(float),
                                                      r_calc[ego_cols].to_numpy(float),
                                                      ego_angles)
            metrics['by_room'][room] = per_room

    if not make_plots:
        return metrics

    # 5) Plots (now 3 columns: bars | scatter | %-(-1) per room)
    n_panels = (1 if 'Allo' in metrics else 0) + (1 if 'Ego' in metrics else 0)
    fig, axes = plt.subplots(n_panels, 3, figsize=(18, 4*n_panels))
    if n_panels == 1:
        axes = np.atleast_2d(axes)

    # colors per room + red for negatives
    if rooms_list:
        cmap = plt.get_cmap('tab20' if len(rooms_list) > 10 else 'tab10')
        room_colors = {room: cmap(i % cmap.N) for i, room in enumerate(sorted(rooms_list))}
    else:
        room_colors = {}

    # --- grouped bars per room with dashed lines + ticks every 3 ---
    def _grouped_bars(ax, base_angles, metrics_total, metrics_by_room, title_text):
        x_idx = np.arange(len(base_angles))
        if metrics_by_room and len(metrics_by_room) > 0:
            rooms = sorted(metrics_by_room.keys())
            width = 0.8 / max(1, len(rooms))
            for i, room in enumerate(rooms):
                series = metrics_by_room[room]['mae_per_angle'].reindex(base_angles)
                ax.bar(
                    x_idx + (i - (len(rooms)-1)/2.0)*width,
                    series.values,
                    width=width,
                    label=str(room),
                    color=room_colors.get(room, None),
                    alpha=0.9
                )
            # per-room dashed MAE lines
            for room in rooms:
                mae_r = metrics_by_room[room]['overall_mae']
                if np.isfinite(mae_r):
                    ax.axhline(mae_r, ls='--', lw=1.2,
                               color=room_colors.get(room, None),
                               alpha=0.9, label=f"{room} MAE")
            ax.legend(loc='upper right', fontsize=8, ncols=2)
        else:
            series = metrics_total['mae_per_angle'].reindex(base_angles)
            ax.bar(x_idx, series.values)

        # global dashed MAE (black)
        if np.isfinite(metrics_total['overall_mae']):
            ax.axhline(metrics_total['overall_mae'], ls='--', lw=1.2, color='k', label='overall MAE')
            handles, labels = ax.get_legend_handles_labels()
            if 'overall MAE' not in labels:
                from matplotlib.lines import Line2D
                h = Line2D([0], [0], color='k', lw=1.2, ls='--', label='overall MAE')
                handles.append(h); labels.append('overall MAE')
                ax.legend(handles, labels, loc='upper right', fontsize=8, ncols=2)

        # ticks every 3 steps
        tick_idx = x_idx[::3]
        tick_lbl = [int(base_angles[i]) for i in tick_idx]
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_lbl)

        ax.set_title(title_text)
        ax.set_xlabel("Angle (deg)")
        ax.set_ylabel("MAE (distance units)")
        ax.grid(True, alpha=0.2)

    # --- scatter by room; negatives plotted as red AFTER clamping to zero ---
    def _scatter_by_room(ax, cols, title, overall_mae, overall_rmse, minus_stats):
        n_rooms = max(1, len(rooms_list))
        per_room_budget = (sample_scatter // n_rooms) if sample_scatter else None

        if rooms_list:
            for room in sorted(rooms_list):
                idx = df_true[df_true[room_col] == room].index
                t_sub = df_true.loc[idx, cols].to_numpy(float).ravel()
                c_sub = df_calc.loc[idx, cols].to_numpy(float).ravel()
                if per_room_budget and len(t_sub) > per_room_budget:
                    rng = np.random.default_rng(0)
                    sel = rng.choice(len(t_sub), size=per_room_budget, replace=False)
                    t_sub = t_sub[sel]; c_sub = c_sub[sel]

                neg_mask = (t_sub < 0) | (c_sub < 0)
                pos_mask = ~neg_mask

                # Replace negatives with 0 for plotting ONLY
                t_plot = t_sub.copy(); c_plot = c_sub.copy()
                t_plot[ t_plot < 0] = 0.0
                c_plot[ c_plot < 0] = 0.0

                # Label with -1% stats
                # if room in minus_stats:
                #     t_pct = minus_stats[room]['true_pct']
                #     c_pct = minus_stats[room]['calc_pct']
                #     label = f"{room} (T {t_pct:.1f}%, C {c_pct:.1f}%)"
                # else:
                label = str(room)

                if np.any(pos_mask):
                    ax.scatter(t_plot[pos_mask], c_plot[pos_mask], s=6, alpha=0.35,
                               label=label, color=room_colors.get(room, None))
                if np.any(neg_mask):
                    ax.scatter(t_plot[neg_mask], c_plot[neg_mask], s=8, alpha=0.6,
                               color='red', label=None)
        else:
            t = df_true[cols].to_numpy(float).ravel()
            c = df_calc[cols].to_numpy(float).ravel()
            if sample_scatter and len(t) > sample_scatter:
                rng = np.random.default_rng(0)
                sel = rng.choice(len(t), size=sample_scatter, replace=False)
                t = t[sel]; c = c[sel]

            neg_mask = (t < 0) | (c < 0)
            pos_mask = ~neg_mask

            t_plot = t.copy(); c_plot = c.copy()
            t_plot[t_plot < 0] = 0.0
            c_plot[c_plot < 0] = 0.0

            ax.scatter(t_plot[pos_mask], c_plot[pos_mask], s=6, alpha=0.35, label='valid')
            if np.any(neg_mask):
                ax.scatter(t_plot[neg_mask], c_plot[neg_mask], s=8, alpha=0.6, color='red', label='negative')

        # y=x using autoscaled limits (no explicit limits set)
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        lo = min(x0, y0); hi = max(x1, y1)
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1)

        # Legend (ensure 'negative' once)
        from matplotlib.lines import Line2D
        handles, labels = ax.get_legend_handles_labels()
        # test if any negatives exist overall
        has_neg = False
        # quick check using full columns (not subsampled)
        full_t = df_true[cols].to_numpy(float).ravel()
        full_c = df_calc[cols].to_numpy(float).ravel()
        if np.any((full_t < 0) | (full_c < 0)):
            has_neg = True
        if has_neg and 'negative' not in labels:
            neg_handle = Line2D([0], [0], marker='o', color='red', linestyle='', markersize=5, label='negative')
            handles.append(neg_handle); labels.append('negative')
        if handles:
            ax.legend(handles, labels, loc='upper left', fontsize=8, ncols=2)

        ax.set_title(f"{title}\n"
                     f"overall MAE: {overall_mae:.2f}, RMSE: {overall_rmse:.2f}")
        ax.set_xlabel("source (df_data)")
        ax.set_ylabel("calc (df_data_polar)")
        ax.grid(True, alpha=0.2)

    # --- third column: %-(-1) per room panel ---
    def _minus_one_panel(ax, minus_stats, title):
        rooms = sorted(minus_stats.keys())
        if not rooms:
            ax.axis('off')
            return
        x = np.arange(len(rooms))
        w = 0.38
        for i, room in enumerate(rooms):
            color = room_colors.get(room, None) if rooms_list else None
            t_pct = minus_stats[room]['true_pct']
            c_pct = minus_stats[room]['calc_pct']
            ax.bar(i - w/2, t_pct, width=w, color=color, alpha=0.45)
            ax.bar(i + w/2, c_pct, width=w, color=color, alpha=0.90)
            # annotate
            if np.isfinite(t_pct):
                ax.text(i - w/2, t_pct, f"{t_pct:.1f}%", ha='center', va='bottom', fontsize=7)
            if np.isfinite(c_pct):
                ax.text(i + w/2, c_pct, f"{c_pct:.1f}%", ha='center', va='bottom', fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([str(r) for r in rooms], rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('% (-1)', fontsize=9)
        ax.set_title(title, fontsize=10)
        from matplotlib.patches import Patch
        legend_handles = [Patch(facecolor='k', alpha=0.45, label='source (-1%)'),
                          Patch(facecolor='k', alpha=0.90, label='calc (-1%)')]
        ax.legend(handles=legend_handles, fontsize=8, frameon=False, loc='upper right')
        ax.grid(True, axis='y', alpha=0.2)

    row = 0
    if 'Allo' in metrics:
        # Bars
        ax = axes[row, 0]
        _grouped_bars(ax, allo_angles, metrics['Allo'],
                      {r: metrics['by_room'][r]['Allo'] for r in rooms_list} if rooms_list else None,
                      f"{title_prefix} — Allo: MAE per angle (negatives excluded)")
        # Scatter
        ax = axes[row, 1]
        _scatter_by_room(ax, allo_cols, "Allo: calc vs source (y=x)",
                         metrics['Allo']['overall_mae'], metrics['Allo']['overall_rmse'],
                         minus_allo)
        # %-(-1) per room
        ax = axes[row, 2]
        _minus_one_panel(ax, minus_allo, title='% (-1) by room — Allo')
        row += 1

    if 'Ego' in metrics:
        # Bars
        ax = axes[row, 0]
        _grouped_bars(ax, ego_angles, metrics['Ego'],
                      {r: metrics['by_room'][r]['Ego'] for r in rooms_list} if rooms_list else None,
                      f"{title_prefix} — Ego: MAE per angle (negatives excluded)")
        # Scatter
        ax = axes[row, 1]
        _scatter_by_room(ax, ego_cols, "Ego: calc vs source (y=x)",
                         metrics['Ego']['overall_mae'], metrics['Ego']['overall_rmse'],
                         minus_ego)
        # %-(-1) per room
        ax = axes[row, 2]
        _minus_one_panel(ax, minus_ego, title='% (-1) by room — Ego')

    plt.tight_layout()
    return fig



@save_plot
def plot_celltype_upset(
    dfClusters,
    cell_types_cols,
    min_count=1,
    sort_by="cardinality",   # "cardinality" or "degree"
    top_k=None,
    title=None,
    save_params={}
):
    """
    Simple, robust UpSet:
      - No rows (cells) are filtered out.
      - Adds No_Class (rows with no other labels) as a real set, shown first.
      - Computes intersection COUNTS explicitly, then applies min_count/top_k to intersections only.
    """
    def _placeholder(msg, sub=None):
        fig = plt.figure(figsize=(6, 2)); plt.axis("off")
        plt.text(0.02, 0.70, msg, fontsize=12)
        if sub: plt.text(0.02, 0.40, sub, fontsize=9)
        return fig

    # --- validate requested columns ---
    if not isinstance(cell_types_cols, (list, tuple)) or not cell_types_cols:
        return _placeholder("UpSet: No cell-type columns provided.")
    cols_present = [c for c in cell_types_cols if c in dfClusters.columns]
    if not cols_present:
        return _placeholder("UpSet: None of the requested columns exist.",
                            f"Checked: {', '.join(cell_types_cols)}")

    # --- boolean matrix over ALL rows (no row filtering) ---
    df_bool = dfClusters[cols_present].fillna(False).astype(bool)

    # Add No_Class as first set (rows with no True in any provided column)
    df_bool.insert(0, "No_Class", ~df_bool.any(axis=1))
    all_cols = ["No_Class"] + [c for c in cols_present if c != "No_Class"]

    # --- raw Series (one entry per row), MultiIndex of booleans ---
    series = from_indicators(df_bool)          # length = n_rows, values = 1.0
    # UpSet can aggregate itself, but we need sizes to filter/top_k:
    # sizes = series.groupby(level=series.index.names).size().astype(int)
    #
    # # --- min_count filter on INTERSECTIONS (not rows) ---
    # min_count = max(1, int(min_count))
    # keep_idx = sizes[sizes >= min_count].index
    # if len(keep_idx) == 0:
    #     return _placeholder("UpSet: No intersections after filtering.",
    #                         f"min_count={min_count}")
    # series = series[series.index.isin(keep_idx)]
    #
    # # --- top_k selection (by cardinality or degree), always keep No_Class ---
    # if top_k is not None and int(top_k) > 0 and len(keep_idx) > int(top_k):
    #     k = int(top_k)
    #
    #     # compute ordering keys on the aggregated sizes
    #     if sort_by == "degree":
    #         degree = sizes.index.to_frame().sum(axis=1).astype(int)
    #         order = (
    #             pd.DataFrame({"degree": degree, "count": sizes.values}, index=sizes.index)
    #             .sort_values(["degree", "count"], ascending=[False, False])
    #             .index
    #         )
    #     else:  # cardinality
    #         order = sizes.sort_values(ascending=False).index
    #
    #     # build keep list (pin No_Class first if present)
    #     ordered = list(order)
    #     keep_list = ordered[:k]
    #
    #     # filter the *series* to these intersections only
    #     series = series[series.index.isin(keep_list)]


    # --- plot ---
    fig = plt.figure(figsize=(14, 8))
    up = UpSet(
        series,
        subset_size="count",         # we pass counts directly
        sort_by=sort_by,
        show_counts=True,
        sort_categories_by="input",   # respects all_cols order -> No_Class row on top
        min_subset_size = int(max(1, min_count))  # <--- key change
    )
    try:
        up.plot(fig=fig)
    except Exception as e:
        plt.close(fig)
        return _placeholder("UpSet: Failed to plot.", str(e))

    if title is None:
        title = f"Cell-Type Intersections (Total cells: {len(dfClusters)})"
    plt.suptitle(title, y=0.98)
    return fig




@save_plot
def plot_fps_histograms(
    ts,
    title="FPS diagnostics",
    bins_dt=60,
    bins_fps=60,
    iqr_clip=True,
    smooth_window_sec=1,      # moving-average window for the per-second mean-FPS trace
    save_params={}
):
    """
    Figure layout:
      Row 1 (split):
        - Left (wide): temporal FPS per second (frames/sec & mean(1/Δt)/sec) + per-frame inst. FPS
        - Right (narrow): 5 s zoom (no legend)
      Row 2: histogram of Δt
      Row 3: histogram of instantaneous FPS (= 1/Δt)
    All panels show median, mean, p05, p95 with consistent styles/colors.
    """

    # unified styles
    mm_color = "black"     # for median & mean
    pp_color = "tab:purple"   # for p05 & p95
    ls_median = "-"
    ls_mean   = ":"
    ls_p05    = "--"
    ls_p95    = "-."

    def _ts_to_seconds(ts):
        ts = pd.Series(ts).dropna()
        if np.issubdtype(ts.dtype, np.datetime64):
            return ts.view('int64') / 1e9  # ns -> s
        return ts.astype(float).to_numpy()

    def _clean_deltas(t, iqr_clip=True):
        dt = np.diff(t)
        pos = dt > 0
        dt = dt[pos]
        t_right = t[1:][pos]
        if iqr_clip and dt.size >= 8:
            q1, q3 = np.percentile(dt, [25, 75])
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            keep = (dt >= max(lo, 1e-9)) & (dt <= hi)
            return dt[keep], t_right[keep], (1 - keep.sum() / max(len(dt), 1)) * 100.0
        return dt, t_right, 0.0

    t = _ts_to_seconds(ts)
    if t.size < 2:
        print("Not enough timestamps")
        return

    # cleaned deltas (and right-edge timestamps)
    dt_clean, t_right, pct_dropped = _clean_deltas(t, iqr_clip=iqr_clip)
    if dt_clean.size == 0:
        print("No positive deltas after cleaning")
        return

    # ---------- stats for Δt ----------
    dt_med  = float(np.median(dt_clean))
    dt_mean = float(np.mean(dt_clean))
    dt_p05  = float(np.percentile(dt_clean, 5))
    dt_p95  = float(np.percentile(dt_clean, 95))
    dt_std  = float(np.std(dt_clean))

    # ---------- stats for FPS (= 1/Δt) ----------
    fps_inst = 1.0 / dt_clean
    fps_med  = float(np.median(fps_inst))
    fps_mean = float(np.mean(fps_inst))
    fps_p05  = float(np.percentile(fps_inst, 5))
    fps_p95  = float(np.percentile(fps_inst, 95))

    # per-frame time for scatter
    t0        = float(t.min())
    time_inst = t_right - t0

    # ---------- per-second series (using the same cleaned set) ----------
    sec_idx = np.floor(time_inst).astype(int)
    max_sec = int(np.max(sec_idx)) if sec_idx.size else 0
    secs    = np.arange(max_sec + 1, dtype=float)

    frames_per_sec = np.bincount(sec_idx, minlength=max_sec + 1).astype(float)
    sum_fps = np.bincount(sec_idx, weights=fps_inst, minlength=max_sec + 1)
    cnt_fps = np.bincount(sec_idx, minlength=max_sec + 1)
    mean_fps_per_sec = np.divide(sum_fps, np.maximum(cnt_fps, 1), where=cnt_fps > 0)
    mean_fps_per_sec[cnt_fps == 0] = np.nan

    # optional smoothing
    if smooth_window_sec and smooth_window_sec > 1 and mean_fps_per_sec.size >= smooth_window_sec:
        mean_fps_per_sec = (
            pd.Series(mean_fps_per_sec)
            .rolling(window=int(smooth_window_sec), center=True, min_periods=1)
            .mean()
            .to_numpy()
        )

    # 5-second zoom around the middle
    total_sec = secs[-1] if secs.size else 0.0
    if total_sec <= 5:
        z_lo, z_hi = 0.0, total_sec
    else:
        mid = total_sec / 2.0
        z_lo, z_hi = max(0.0, mid - 1.0), min(total_sec, mid + 1.0)

    # ---------- figure ----------
    fig = plt.figure(figsize=(11, 8), constrained_layout=False)
    gs = GridSpec(
        nrows=3, ncols=2, figure=fig,
        height_ratios=[1.6, 1.0, 1.0],
        width_ratios=[4.0, 1.1],
        hspace=0.6, wspace=0.25
    )

    # Row 1, col 0: main temporal
    ax_time = fig.add_subplot(gs[0, 0])
    ax_time.plot(time_inst, fps_inst, lw=0.5, color="gray", alpha=0.35, label="Instantaneous FPS (per frame)", zorder=1)
    ax_time.plot(secs, frames_per_sec, lw=1.3, label="Frames/sec (count)")
    ax_time.plot(secs, mean_fps_per_sec, lw=1.8, label="Mean instantaneous FPS/sec")

    # FPS reference lines (same color scheme everywhere)
    ax_time.axhline(fps_med,  color=mm_color, linestyle=ls_median, linewidth=1.2, label=f"median FPS ≈ {fps_med:.3f}")
    ax_time.axhline(fps_mean, color=mm_color, linestyle=ls_mean,   linewidth=1.2, label=f"mean FPS ≈ {fps_mean:.3f}")
    ax_time.axhline(fps_p05,  color=pp_color, linestyle=ls_p05,    linewidth=1.0, label=f"p05 FPS ≈ {fps_p05:.3f}")
    ax_time.axhline(fps_p95,  color=pp_color, linestyle=ls_p95,    linewidth=1.0, label=f"p95 FPS ≈ {fps_p95:.3f}")

    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel("FPS")
    ax_time.set_title("Temporal FPS (per second)")
    ax_time.grid(alpha=0.3)
    ax_time.legend(frameon=False, ncol=3, loc="upper right", fontsize=8)

    # Row 1, col 1: zoom (no legend)
    ax_zoom = fig.add_subplot(gs[0, 1]) #, sharey=ax_time
    ax_zoom.plot(time_inst, fps_inst, lw=0.5, color="gray", alpha=0.35, zorder=1)
    ax_zoom.plot(secs, frames_per_sec, lw=1.3)
    ax_zoom.plot(secs, mean_fps_per_sec, lw=1.8)
    ax_zoom.axhline(fps_med,  color=mm_color, linestyle=ls_median, linewidth=1.2)
    ax_zoom.axhline(fps_mean, color=mm_color, linestyle=ls_mean,   linewidth=1.2)
    ax_zoom.axhline(fps_p05,  color=pp_color, linestyle=ls_p05,    linewidth=1.0)
    ax_zoom.axhline(fps_p95,  color=pp_color, linestyle=ls_p95,    linewidth=1.0)
    ax_zoom.set_xlim(z_lo, z_hi)
    in_range_mask = (time_inst >= z_lo) & (time_inst <= z_hi)
    if np.any(in_range_mask):
        max_in_range = np.nanmax(fps_inst[in_range_mask])
        min_in_range = np.nanmin(fps_inst[in_range_mask])
        ax_zoom.set_ylim(max(0.0, min_in_range * 0.9), max_in_range * 1.1)
    ax_zoom.set_xlabel("Time (s)")
    ax_zoom.set_title("Zoom: 3 seconds")
    ax_zoom.grid(alpha=0.3)

    # Row 2: Δt histogram (+ all four refs, same style/colors but vertical)
    ax_dt = fig.add_subplot(gs[1, :])
    ax_dt.hist(dt_clean, bins=bins_dt, alpha=0.9)
    ax_dt.axvline(dt_med,  color=mm_color, linestyle=ls_median, linewidth=1.5, label=f"median Δt={dt_med*1000:.2f} ms")
    ax_dt.axvline(dt_mean, color=mm_color, linestyle=ls_mean,   linewidth=1.5, label=f"mean Δt={dt_mean*1000:.2f} ms")
    ax_dt.axvline(dt_p05,  color=pp_color, linestyle=ls_p95,    linewidth=1.2, label=f"p05 Δt={dt_p05*1000:.2f} ms")
    ax_dt.axvline(dt_p95,  color=pp_color, linestyle=ls_p05,    linewidth=1.2, label=f"p95 Δt={dt_p95*1000:.2f} ms")
    ax_dt.set_xlabel("Inter-frame Δt (s)")
    ax_dt.set_ylabel("Count")
    ax_dt.set_title("Inter-frame time (Δt) histogram")
    ax_dt.grid(alpha=0.3)
    ax_dt.legend(frameon=False, ncol=2, loc="upper right", fontsize=8)

    # Row 3: FPS histogram (+ all four refs)
    ax_fps = fig.add_subplot(gs[2, :])
    lo, hi = np.percentile(fps_inst, [1, 99])
    ax_fps.hist(fps_inst, bins=bins_fps, alpha=0.9)
    ax_fps.axvline(fps_med,  color=mm_color, linestyle=ls_median, linewidth=1.5, label=f"median FPS={fps_med:.3f}")
    ax_fps.axvline(fps_mean, color=mm_color, linestyle=ls_mean,   linewidth=1.5, label=f"mean FPS={fps_mean:.3f}")
    ax_fps.axvline(fps_p05,  color=pp_color, linestyle=ls_p05,    linewidth=1.2, label=f"p05 FPS={fps_p05:.3f}")
    ax_fps.axvline(fps_p95,  color=pp_color, linestyle=ls_p95,    linewidth=1.2, label=f"p95 FPS={fps_p95:.3f}")
    ax_fps.set_xlim(lo, hi)
    ax_fps.set_xlabel("Instantaneous FPS (= 1 / Δt)")
    ax_fps.set_ylabel("Count")
    ax_fps.set_title("Instantaneous FPS histogram")
    ax_fps.grid(alpha=0.3)
    ax_fps.legend(frameon=False, ncol=2, loc="upper right", fontsize=8)

    # overall title + stat box
    fig.suptitle(title, y=0.99, fontsize=12)
    txt = (
        f"Frames used: {len(dt_clean)+1} (dropped {pct_dropped:.1f}% deltas)"
    )
    ax_dt.text(0.99, 0.05, txt, transform=ax_dt.transAxes, va="bottom", ha="right",
               fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))

    plt.subplots_adjust(top=0.90)
    return fig

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.cm import ScalarMappable
from math import ceil

@save_plot
def plot_tracking_quality_dashboard(
    dfPos,
    df_summary,
    df_gaps,
    df_timebin,
    df_fps,
    config,
    streams,
    state_colors,
    title="Tracking Quality Dashboard",
    save_params={},
):
    sp = save_params or {}
    figsize  = sp.get("figsize", (18, 22))
    gridsize = int(sp.get("gridsize", 50))
    mincnt   = int(sp.get("mincnt_all", 5))

    pp = (config or {}).get("preprocessing", {}) or {}
    bin_s   = float(pp.get("bin_width_s", 60.0))
    t_thr_s = float(pp.get("long_time_threshold_s", 3.0))
    d_thr   = float(pp.get("long_dist_threshold_cm", 15.0))
    v_cap   = pp.get("speed_max_cm_s", None)
    v_cap   = float(v_cap) if v_cap is not None else None
    primary = str(pp.get("position_label", streams[0] if streams else ""))

    if "TS" not in dfPos.columns:
        raise ValueError("dfPos must contain 'TS' (seconds).")
    ts = dfPos["TS"].to_numpy(float)
    n  = len(ts)
    S  = len(streams)

    def v_calc(ts, x, y):
        dt = np.diff(ts)
        dd = np.hypot(np.diff(x), np.diff(y))
        v  = np.divide(dd, dt, out=np.full_like(dd, np.nan), where=dt > 0)
        return np.r_[np.nan, v]

    def pct(a, d): d = max(int(d), 1); return 100.0 * (float(a)/d)

    def extent(x_all, y_all, pad=0.02):
        xs = np.concatenate([a[np.isfinite(a)] for a in x_all]) if x_all else np.array([0,1])
        ys = np.concatenate([a[np.isfinite(a)] for a in y_all]) if y_all else np.array([0,1])
        if xs.size==0 or ys.size==0: return (0,1,0,1)
        xmin,xmax = xs.min(), xs.max(); ymin,ymax = ys.min(), ys.max()
        dx,dy = xmax-xmin, ymax-ymin
        return (xmin-pad*dx, xmax+pad*dx, ymin-pad*dy, ymax+pad*dy)

    def hist(ax, vals, color, bins, fill=False, alpha=0.28):
        v = np.asarray(vals, float)
        v = v[np.isfinite(v)]
        if v.size==0: return
        if fill:
            ax.hist(v, bins=bins, histtype="bar", color=color, alpha=alpha, edgecolor="none")
        else:
            c,e = np.histogram(v, bins=bins)
            ctr = 0.5*(e[:-1]+e[1:])
            ax.step(ctr, c, where="mid", color=color)

    def hist_counts(vals, bins):
        v = np.asarray(vals, float); v = v[np.isfinite(v)]
        if v.size == 0:
            return np.array([]), np.array([])
        c, e = np.histogram(v, bins=bins)
        ctr  = 0.5*(e[:-1] + e[1:])
        return c, ctr

    # per-stream arrays
    tab10 = plt.cm.get_cmap("tab10")
    s_col = {k: tab10(i % 10) for i, k in enumerate(streams)}
    per, all_x, all_y = {}, [], []
    for key in streams:
        x = dfPos.get(f"{key}.X", pd.Series(np.nan, index=dfPos.index)).to_numpy(float)
        y = dfPos.get(f"{key}.Y", pd.Series(np.nan, index=dfPos.index)).to_numpy(float)
        P = dfPos.get(f"{key}.P", pd.Series(np.nan, index=dfPos.index)).to_numpy(float)
        It= dfPos.get(f"{key}.Int", pd.Series(0, index=dfPos.index)).to_numpy(float)

        miss = np.isnan(x)|np.isnan(y)
        interp = (It>0)&(~miss); valid=(~miss)&(It==0)
        st = np.zeros(n, int); st[valid]=1; st[interp]=2

        m_ld = np.zeros(n, bool); m_lt = np.zeros(n, bool); m_both = np.zeros(n, bool)
        gk = df_gaps[df_gaps["stream"]==key]
        for _, r in gk.iterrows():
            a = max(0, min(int(r["start_idx"]), n-1))
            b = max(0, min(int(r["end_idx"]),   n-1))
            ld = bool(r.get("is_long_distance", False))
            lt = bool(r.get("is_long_time", False))
            if ld:           st[a:b+1]=4; m_ld[a:b+1]=True
            if (not ld) and lt: st[a:b+1]=3; m_lt[a:b+1]=True
            if ld and lt:    m_both[a:b+1]=True

        per[key] = {"x":x,"y":y,"P":P,"st":st,"v":v_calc(ts,x,y),
                    "prim":(key==primary),"m_ld":m_ld,"m_lt":m_lt,"m_both":m_both}
        all_x.append(x); all_y.append(y)

    xy_ext = extent(all_x, all_y, pad=0.01)
    cmap = mcolors.ListedColormap([state_colors[i] for i in (0,1,2,3,4)])
    norm = mcolors.BoundaryNorm([-0.5,0.5,1.5,2.5,3.5,4.5], cmap.N)

    # dynamic row plan
    h_tl = 1.15
    h_lt = 1.7
    h_ld = 1.7
    h_sp = 1.9
    h_sc = 3.6
    rows_spatial = 1#max(1, ceil(S / 3))
    rows_xy = 1#max(1, ceil(S / 3))
    h_spat = 3.0
    h_xy = 6.0

    heights = [h_tl] * S + [h_lt, h_ld, h_sp, h_sc] + [h_spat] * rows_spatial + [h_xy] * rows_xy
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=len(heights), ncols=S, height_ratios=heights, hspace=0.75, wspace=0.30)

    def set_xt(ax):
        step = 5*60
        tsec = np.arange(0, max(ts[-1],0)+1, step)
        tf   = np.searchsorted(ts, tsec)
        ax.set_xticks(tf); ax.set_xticklabels([f"{int(t//60)}" for t in tsec])
        ax.grid(axis="x", linestyle=":", alpha=0.4)

    # timelines
    for i, key in enumerate(streams):
        st = per[key]["st"]; x=per[key]["x"]; y=per[key]["y"]
        ax = fig.add_subplot(gs[i, :])
        ax.imshow(st[np.newaxis,:], aspect="auto", extent=[0,n,0,1], cmap=cmap, norm=norm, interpolation="nearest", alpha=0.6)
        m = np.isfinite(x)&np.isfinite(y)
        if m.any():
            xn=(x[m]-np.nanmin(x))/max(np.nanmax(x)-np.nanmin(x),1e-9)
            yn=(y[m]-np.nanmin(y))/max(np.nanmax(y)-np.nanmin(y),1e-9)
            t  = np.flatnonzero(m)
            ax.plot(t, 0.22+0.30*xn, lw=0.8, alpha=0.95, color="#222")
            ax.plot(t, 0.58+0.30*yn, lw=0.8, alpha=0.95, color="#555")
        p_valid=pct((st==1).sum(),n); p_short=pct((st==2).sum(),n)
        p_lt=pct((st==3).sum(),n);   p_ld=pct((st==4).sum(),n); p_m=pct((st==0).sum(),n)
        lab=f"{key}{' (primary)' if per[key]['prim'] else ''}"
        ax.set_title(f"Timeline — {lab} (Valid:{p_valid:.1f}%, Short:{p_short:.1f}%, Long-time:{p_lt:.1f}%, Long-dist:{p_ld:.1f}%, Missing:{p_m:.1f}%)")
        ax.set_yticks([0.3,0.7]); ax.set_yticklabels(["x","y"]); set_xt(ax); ax.set_xlabel("Time (min)")
        if i==0:
            leg_items = [Patch(facecolor=state_colors[1],label="Valid"),
                         Patch(facecolor=state_colors[2],label="Short"),
                         Patch(facecolor=state_colors[3],label="Long-time"),
                         Patch(facecolor=state_colors[4],label="Long-dist"),
                         Patch(facecolor=state_colors[0],label="Missing")]
            ax.legend(leg_items, [h.get_label() for h in leg_items],
                      frameon=False, ncol=1, bbox_to_anchor=(1.02,1.0), loc="upper left", fontsize=9, borderaxespad=0.0)

    row0 = S  # next free row index

    # long-time row (2:1)
    split = min(S-1,ceil(2*S/3))
    ax_lt  = fig.add_subplot(gs[row0, :split]); ax_lth = fig.add_subplot(gs[row0, split])
    lt_lines=[]; lt_counts_all=[]; lt_centers_all=[]
    for k in streams:
        tb = df_timebin[df_timebin["stream"]==k]
        if tb.empty: continue
        c = s_col[k]
        lt_lines.append(ax_lt.plot(tb["bin"]*(bin_s/60.0), tb["pct_long_time"], lw=1.1, color=c, label=k)[0])
        gk = df_gaps[df_gaps["stream"]==k]
        if not gk.empty:
            dur = gk["duration_sec"].to_numpy(float)
            if np.any(np.isfinite(dur)):
                xmax = max(t_thr_s*2, np.nanpercentile(dur[np.isfinite(dur)], 99.5))
                bins = np.linspace(0, xmax, 60)
                cnt, ctr = hist_counts(dur, bins)
                lt_counts_all.append(cnt); lt_centers_all.append(ctr)
                ax_lth.step(ctr, cnt, where="mid", color=c)
    ax_lt.set_xlabel("Time (min)"); ax_lt.set_ylabel("% long-time"); ax_lt.set_title("Per-bin % long-time"); ax_lt.grid(True, alpha=0.35)
    ax_lth.axvline(t_thr_s, color="k", ls=":", lw=1.2)
    # new y-limit rule
    if lt_counts_all:
        cnt = np.vstack(lt_counts_all); ctr = lt_centers_all[0]
        after = cnt[:, ctr >= t_thr_s]
        y1 = np.max(after) if after.size else 0.0
        y2 = np.percentile(cnt, 95)
        ylim_ = y1*2#1.05*max(y1, y2)
        ax_lth.set_ylim(0, ylim_)
    ax_lth.set_xlabel("Gap duration (s)"); ax_lth.set_ylabel("Count"); ax_lth.set_title("Gap duration distribution"); ax_lth.grid(True, alpha=0.3)

    # long-distance row (2:1)
    row1 = row0+1
    split = min(S - 1, ceil(2 * S / 3))
    ax_ld  = fig.add_subplot(gs[row1, :split]); ax_ldh = fig.add_subplot(gs[row1, split])
    ld_lines=[]; ld_counts_all=[]; ld_centers_all=[]
    for k in streams:
        tb = df_timebin[df_timebin["stream"]==k]
        if tb.empty: continue
        c = s_col[k]
        ld_lines.append(ax_ld.plot(tb["bin"]*(bin_s/60.0), tb["pct_long_distance"], lw=1.1, color=c, label=k)[0])
        gk = df_gaps[df_gaps["stream"]==k]
        if not gk.empty:
            dist = gk["distance_cm"].to_numpy(float)
            if np.any(np.isfinite(dist)):
                xmax = max(d_thr*2, np.nanpercentile(dist[np.isfinite(dist)], 99.5))
                bins = np.linspace(0, xmax, 60)
                cnt, ctr = hist_counts(dist, bins)
                ld_counts_all.append(cnt); ld_centers_all.append(ctr)
                ax_ldh.step(ctr, cnt, where="mid", color=c)
    ax_ld.set_xlabel("Time (min)"); ax_ld.set_ylabel("% long-distance"); ax_ld.set_title("Per-bin % long-distance"); ax_ld.grid(True, alpha=0.35)
    ax_ldh.axvline(d_thr, color="k", ls=":", lw=1.2)
    if ld_counts_all:
        cnt = np.vstack(ld_counts_all); ctr = ld_centers_all[0]
        after = cnt[:, ctr >= d_thr]
        y1 = np.max(after) if after.size else 0.0
        y2 = np.percentile(cnt, 95)
        ylim_ = y1 * 2  # 1.05*max(y1, y2)
        ax_ldh.set_ylim(0, ylim_)
    ax_ldh.set_xlabel("Gap distance (cm)"); ax_ldh.set_ylabel("Count"); ax_ldh.set_title("Gap distance distribution"); ax_ldh.grid(True, alpha=0.3)

    # speed row (2:1)
    row2 = row1+1
    split = min(S - 1, ceil(2 * S / 3))
    ax_sp  = fig.add_subplot(gs[row2, :split]); ax_sph = fig.add_subplot(gs[row2, split])
    sp_lines=[]; sp_vals=[]
    for k in streams:
        tb = df_timebin[df_timebin["stream"]==k]
        if tb.empty: continue
        c=s_col[k]
        x=tb["bin"]*(bin_s/60.0); y=tb["mean_speed_manual"].to_numpy(float)
        sp_lines.append(ax_sp.plot(x, y, lw=1.1, alpha=0.95, label=k, color=c)[0]); sp_vals.append((k,y,c))
    if v_cap is not None: ax_sp.axhline(v_cap, color="k", ls=":", lw=1.1)
    ax_sp.set_xlabel("Time (min)"); ax_sp.set_ylabel("Mean speed (cm/s)"); ax_sp.set_title("Speed per bin (calculated)"); ax_sp.grid(True, alpha=0.3)
    xmax_sp=0.0
    for _, y, _ in sp_vals:
        v=y[np.isfinite(y)]
        if v.size: xmax_sp=max(xmax_sp, np.percentile(v,99.5))
    if xmax_sp<=0: xmax_sp=(v_cap or 1.0)*1.2
    bins_sp=np.linspace(0,xmax_sp,50)
    for _, y, c in sp_vals: hist(ax_sph, y, c, bins_sp, fill=True, alpha=0.28)
    if v_cap is not None: ax_sph.axvline(v_cap, color="k", ls=":", lw=1.1)
    ax_sph.set_xlabel("Per-bin mean speed (cm/s)"); ax_sph.set_ylabel("Count")
    ax_sph.set_title("Speed distribution (per-bin mean)"); ax_sph.grid(True, alpha=0.3)

    # shared legend for the three rows
    share = {}
    for h in (lt_lines + ld_lines + sp_lines):
        if h and h.get_label() and not h.get_label().startswith("_"): share[h.get_label()] = h
    # anchor to top-right of the long-time row
    bb = ax_lt.get_position()
    fig.legend(list(share.values()), list(share.keys()), frameon=False, ncol=1,
               bbox_to_anchor=(0.99, bb.y1), loc="upper right", fontsize=9)

    # scatter
    row3 = row2+1
    ax_sc = fig.add_subplot(gs[row3, :])
    xlim_sc = ax_lth.get_xlim(); ylim_sc = ax_ldh.get_xlim()
    for k in streams:
        gk = df_gaps[df_gaps["stream"]==k]
        if gk.empty: continue
        c  = s_col[k]
        xd = gk["duration_sec"].to_numpy(float)
        yd = gk["distance_cm"].to_numpy(float)
        m  = np.isfinite(xd) & np.isfinite(yd)
        ax_sc.plot(xd[m], yd[m], "o", ms=3.0, alpha=0.5,  markeredgecolor=c, markerfacecolor="none")
    # grey infeasible region: speed > 20 cm/s  => y > 20 * x
    v_inf = 20.0
    xg = np.linspace(0, xlim_sc[1], 200)
    yg = v_inf * xg
    ax_sc.fill_between(xg, yg, ylim_sc[1], color="#999999", alpha=0.15, label="> 20 cm/s")
    # guide lines (optional)
    for vref in [5,10,20,40,80,120,160]:
        if vref*xlim_sc[1] <= ylim_sc[1]*1.05:
            ax_sc.plot(xg, vref*xg, ls="--", lw=0.8, color="#888888")
    ax_sc.set_xlim(xlim_sc); ax_sc.set_ylim(0, ylim_sc[1])
    ax_sc.set_xlabel("Gap duration (s)"); ax_sc.set_ylabel("Gap distance (cm)")
    ax_sc.set_title("Gap duration vs distance")
    ax_sc.grid(True, alpha=0.3)
    # mini legend just for the grey region
    ax_sc.legend(frameon=False, loc="upper right", fontsize=8)


    # spatial reliability (rows_spatial rows, 3 cols)
    row_start_sp = row3+1
    ax_rowtitle  = fig.add_subplot(gs[row_start_sp, :]); ax_rowtitle.axis("off"); ax_rowtitle.set_title("Spatial reliability (mean P)\n", fontsize=12, pad=2)
    P_all = np.concatenate([per[k]["P"][np.isfinite(per[k]["P"])] for k in streams]) if streams else np.array([])
    vmin = np.nanmin(P_all) if P_all.size else 0.0; vmax = np.nanmax(P_all) if P_all.size else 1.0
    cmapP = plt.cm.viridis; normP = mcolors.Normalize(vmin=vmin, vmax=vmax)

    sp_axes = []
    for idx, k in enumerate(streams):
        r = row_start_sp + idx//S#3
        c = idx % S#3
        ax = fig.add_subplot(gs[r, c])
        x=per[k]["x"]; y=per[k]["y"]; P=per[k]["P"]; m=np.isfinite(x)&np.isfinite(y)&np.isfinite(P)
        if m.any():
            ax.hexbin(x[m], y[m], C=P[m], gridsize=gridsize, extent=xy_ext, reduce_C_function=np.nanmean,
                      mincnt=mincnt, cmap=cmapP, norm=normP)
        ax.set_title(k, fontsize=10); ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (cm)"); ax.set_ylabel("Y (cm)")
        sp_axes.append(ax)
    sm = ScalarMappable(norm=normP, cmap=cmapP); sm.set_array([])
    fig.colorbar(sm, ax=sp_axes, fraction=0.025, pad=0.04, location="right", label="Mean P")

    # XY trajectories (rows_xy rows, 3 cols)
    row_start_xy = row_start_sp + rows_spatial
    ax_xy_title  = fig.add_subplot(gs[row_start_xy, :]); ax_xy_title.axis("off"); ax_xy_title.set_title("XY trajectories with long-time / long-dist / both overlays\n", fontsize=12, pad=2)
    xy_handles = {}
    for idx, k in enumerate(streams):
        r = row_start_xy + idx//S#3
        c = idx % S#3
        ax = fig.add_subplot(gs[r, c])
        x=per[k]["x"]; y=per[k]["y"]
        m=np.isfinite(x)&np.isfinite(y)
        if m.any(): ax.scatter(x[m], y[m], s=1, alpha=0.08, c="#999")
        m_lt=per[k]["m_lt"]&m; m_ld=per[k]["m_ld"]&m; m_b=per[k]["m_both"]&m
        if m_lt.any(): xy_handles.setdefault("long-time", ax.scatter(x[m_lt], y[m_lt], s=5, alpha=0.9, c=state_colors[3], label="long-time"))
        if m_ld.any(): xy_handles.setdefault("long-dist", ax.scatter(x[m_ld], y[m_ld], s=5, alpha=0.9, c=state_colors[4], label="long-dist"))
        if m_b.any():  xy_handles.setdefault("both",      ax.scatter(x[m_b],  y[m_b],  s=7, alpha=0.95, c="#d62728",      label="both"))
        ax.set_title(k, fontsize=10); ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (cm)"); ax.set_ylabel("Y (cm)")
    if xy_handles:
        fig.legend(list(xy_handles.values()), list(xy_handles.keys()), frameon=False, ncol=1,
                   bbox_to_anchor=(0.99, 0.22), loc="upper right", fontsize=9, markerscale=4)

    # footer
    if isinstance(df_fps, pd.DataFrame) and not df_fps.empty:
        r = df_fps.iloc[0].to_dict()
        txt = (f"Recording: {r.get('recording_duration_sec', np.nan):.1f}s, "
               f"frames={int(r.get('frames_total', np.nan))} | "
               f"dt median={r.get('dt_median_s', np.nan):.4f}s, p95={r.get('dt_p95_s', np.nan):.4f}s | "
               f"fps_inst median={r.get('fps_inst_median', np.nan):.2f}, p95={r.get('fps_inst_p95', np.nan):.2f}")
        fig.text(0.02, 0.012, txt, fontsize=9, ha="left", va="bottom", wrap=True)

    fig.suptitle(title, fontsize=16, y=0.99)
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.985])
    return fig


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch

@save_plot
def plot_tracking_overview(
    dfPos,
    title="Tracking — Overview",
    speed_threshold=None,       # cm/s line on V panel (e.g., 120.0)
    unwrap_hd=True,             # unwrap HD to avoid jumps
    rooms_to_indices=None,
    map_rooms=None,
    pos_range=None,
    dfBoundary=None,            # DataFrame with boundary XY (optional)
    save_params=None,
):
    """
    Expects dfPos columns: ['timestamp','X','Y','V','HD','valid','X_smooth','Y_smooth','room']
    Shades gaps (valid==False or X/Y NaN) across ALL panels.
    Adds bottom row with one XY trajectory panel per room using plot_trajectory(...).
    """
    sp = save_params or {}
    figsize = sp.get("figsize", (12, 16))
    gap_alpha = sp.get("gap_alpha", 0.30)  # transparent gray background
    hspace = sp.get("hspace", 0.60)        # space between rows

    required = ["timestamp","X","Y","V","HD","valid","room"]
    missing_cols = [c for c in required if c not in dfPos.columns]
    if missing_cols:
        raise ValueError(f"dfPos missing columns: {missing_cols}")
    smooth_cols = ["X_smooth","Y_smooth"]
    is_smooth_col = all(c in dfPos.columns for c in smooth_cols)

    # --- arrays ---
    t  = dfPos["timestamp"].to_numpy(float)
    x  = dfPos["X"].to_numpy(float)
    y  = dfPos["Y"].to_numpy(float)
    if is_smooth_col:
        xs = dfPos["X_smooth"].to_numpy(float)
        ys = dfPos["Y_smooth"].to_numpy(float)
    v  = dfPos["V"].to_numpy(float)
    hd = dfPos["HD"].to_numpy(float)
    valid = dfPos["valid"].astype(bool).to_numpy()
    room  = dfPos["room"].astype(str).fillna("NA").replace("None", "NA").to_numpy()

    # minutes axis
    tmin = (t - t.min()) / 60.0

    # gaps
    gap_mask = (~valid) | ~np.isfinite(x) | ~np.isfinite(y)
    def runs(mask):
        if mask.size == 0: return []
        diff = np.diff(mask.astype(int), prepend=0, append=0)
        starts = np.flatnonzero(diff == 1)
        ends   = np.flatnonzero(diff == -1) - 1
        return [(int(s), int(e)) for s, e in zip(starts, ends)]
    gap_runs = runs(gap_mask)

    # unwrap HD (degrees)
    hd_plot = hd.copy()
    if unwrap_hd:
        hd_plot = np.rad2deg(np.unwrap(np.deg2rad(hd_plot)))

    # rooms & colors (exclude "NA" for the per-room columns)
    unique_rooms_all = pd.unique(room)
    unique_rooms = [r for r in unique_rooms_all if r != "NA"]
    n_rooms = max(1, len(unique_rooms))
    color_palette = cm.get_cmap("tab20c", len(unique_rooms) + 2)
    room_colors = {r: color_palette(i) for i, r in enumerate(unique_rooms)}
    room_colors["NA"] = "#cccccc"

    # --- layout: R columns; top 6 rows span all columns; bottom row has R panels ---
    fig = plt.figure(figsize=figsize)
    heights = [0.8, 1.0, 1.6, 1.6, 1.6, 1.6, 6.2]
    gs = fig.add_gridspec(
        nrows=len(heights),
        ncols=n_rooms,
        height_ratios=heights,
        hspace=hspace,
        wspace=0.25
    )

    # helpers
    def shade_gaps(ax, y0=None, y1=None, alpha=gap_alpha, color="#bdbdbd"):
        for a, b in gap_runs:
            ax.axvspan(
                tmin[a], tmin[b],
                ymin=0 if y0 is None else y0,
                ymax=1 if y1 is None else y1,
                color=color, alpha=alpha, linewidth=0
            )
    def set_time_axis(ax):
        span = tmin.max() - tmin.min()
        step = 5.0 if span > 15 else 2.0
        ticks = np.arange(np.floor(tmin.min()/step)*step, tmin.max()+step, step)
        ax.set_xticks(ticks)
        ax.set_xlabel("Time (min)")
        ax.grid(axis="x", which="both", linestyle=":", alpha=0.35)

    # --- Row 0: Room lane (span all columns)
    ax_room = fig.add_subplot(gs[0, :])
    # compute runs of room changes
    room_runs = []
    if room.size:
        start = 0
        for i in range(1, len(room)):
            if room[i] != room[i-1]:
                room_runs.append((start, i-1))
                start = i
        room_runs.append((start, len(room)-1))
    for a, b in room_runs:
        ax_room.axvspan(tmin[a], tmin[b], color=room_colors.get(room[a], "#cccccc"), alpha=0.65, linewidth=0)
    r_index = pd.Series(room).astype("category").cat.codes.to_numpy()
    ax_room.plot(tmin, r_index, lw=0.4, color="#222222", alpha=0.5)
    room_handles = [Patch(facecolor=room_colors[r], edgecolor="none", label=str(r)) for r in unique_rooms]
    ax_room.set_yticks([]); ax_room.set_ylabel("Room"); ax_room.set_title("Room")
    set_time_axis(ax_room)

    # --- Row 1: Validity lane (span all columns)
    ax_valid = fig.add_subplot(gs[1, :], sharex=ax_room)
    ax_valid.plot(tmin, valid.astype(int), lw=1.2, color="#1b9e77", label="valid (1/0)")
    shade_gaps(ax_valid)
    ax_valid.set_ylim(-0.1, 1.1)
    ax_valid.set_yticks([0, 1]); ax_valid.set_yticklabels(["0","1"])
    ax_valid.set_title("Validity")
    set_time_axis(ax_valid)

    # --- Row 2: X vs X_smooth (span all columns)
    ax_x = fig.add_subplot(gs[2, :], sharex=ax_room)
    shade_gaps(ax_x)
    ax_x.plot(tmin, x,  lw=1.0, color="#1f78b4", alpha=0.9, label="X")
    if is_smooth_col:
        ax_x.plot(tmin, xs, lw=1.0, color="#a6cee3", alpha=0.9, label="X_smooth")
        ax_x.set_title("X / X_smooth"); ax_x.set_ylabel("X (cm)")
    else:
        ax_x.set_title("X"); ax_x.set_ylabel("X (cm)")
    set_time_axis(ax_x)

    # --- Row 3: Y vs Y_smooth (span all columns)
    ax_y = fig.add_subplot(gs[3, :], sharex=ax_room)
    shade_gaps(ax_y)
    ax_y.plot(tmin, y,  lw=1.0, color="#33a02c", alpha=0.9, label="Y")
    if is_smooth_col:
        ax_y.plot(tmin, ys, lw=1.0, color="#b2df8a", alpha=0.9, label="Y_smooth")
        ax_y.set_title("Y / Y_smooth"); ax_y.set_ylabel("Y (cm)")
    else:
        ax_y.set_title("Y"); ax_y.set_ylabel("Y (cm)")
    set_time_axis(ax_y)

    # --- Row 4: Speed (span all columns)
    ax_v = fig.add_subplot(gs[4, :], sharex=ax_room)
    shade_gaps(ax_v)
    ax_v.plot(tmin, v, lw=1.0, color="#e31a1c", alpha=0.95, label="V (cm/s)")
    if speed_threshold is not None:
        ax_v.axhline(float(speed_threshold), color="#444444", ls="--", lw=1.0, label=f"speed ≥ {speed_threshold:g}")
    ax_v.set_title("Speed (V)"); ax_v.set_ylabel("cm/s")
    set_time_axis(ax_v)

    # --- Row 5: Head Direction (span all columns)
    ax_hd = fig.add_subplot(gs[5, :], sharex=ax_room)
    shade_gaps(ax_hd)
    ax_hd.plot(tmin, hd_plot, lw=1.0, color="#6a3d9a", alpha=0.95, label="HD")
    ax_hd.set_title("Head Direction (HD)")
    ax_hd.set_ylabel("deg" + (" (unwrapped)" if unwrap_hd else ""))
    set_time_axis(ax_hd)

    # --- Shared legend on right (rooms + line overlays)
    shared_handles = {}
    for ax in (ax_x, ax_y, ax_v, ax_hd):
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            shared_handles[li] = hi
    all_handles = room_handles + list(shared_handles.values())
    all_labels  = [h.get_label() for h in room_handles] + list(shared_handles.keys())
    if all_handles:
        top_bb = ax_room.get_position()
        fig.legend(all_handles, all_labels, frameon=False, ncol=1,
                   bbox_to_anchor=(0.995, top_bb.y1), loc="upper right", fontsize=9)

    # --- Row 6: Per-room trajectories (one column per room)
    # If there are no concrete rooms (only NA), still create a single empty axes.
    boundary_points = dfBoundary.values if dfBoundary is not None else None
    if len(unique_rooms) == 0:
        ax = fig.add_subplot(gs[6, 0])
        ax.axis("off")
        ax.set_title("No rooms to plot")
    else:
        for j, rname in enumerate(unique_rooms):
            ax = fig.add_subplot(gs[6, j])
            # select only rows within this room
            m = (room == rname) & np.isfinite(x) & np.isfinite(y)
            if np.any(m):
                plot_trajectory(
                    dfPos[["X", "Y"]].values,
                    rooms_to_indices=rooms_to_indices,
                    map_rooms=map_rooms,
                    pos_range=pos_range,
                    rooms = [rname],
                    boundary_points=boundary_points,
                    title=f"Room {rname}",
                    ax_in=ax
                )
            else:
                ax.set_title(f"Room {rname}")
            ax.set_aspect("equal", adjustable="box")

    fig.suptitle(title, fontsize=14, y=0.99)
    fig.tight_layout(rect=[0.02, 0.03, 0.96, 0.985])
    return fig


# @save_plot
# def plot_interpolation_overview_dashboard(
#     df_summary: pd.DataFrame,
#     df_gaps: pd.DataFrame,
#     df_per_minute: pd.DataFrame,
#     long_gap_frames: int = 6,
#     save_params: dict = None,
# ):
#     if save_params is None:
#         save_params = {}
#     figsize     = save_params.get("figsize", (16, 18))
#     max_projects_in_panel = int(save_params.get("topn_projects", 12))
#     heatmap_minutes_max   = int(save_params.get("heatmap_minutes_max", 90))
#     gap_length_max = int(save_params.get("gap_length_max", 0))  # frames
#     Long_gap_durations_max = int(save_params.get("long_gap_durations_max", 0))  # seconds
#
#     # ---------- defensive helpers ----------
#     def _empty_df(cols): return pd.DataFrame({c: [] for c in cols})
#
#
#
#     dfS = df_summary.copy() if df_summary is not None else _empty_df(
#         ["source_key","source_label","project_name","frames_total","percent_valid",
#          "percent_interp_short","percent_interp_long","percent_missing",
#          "num_gaps","gap_len_p95","gap_len_max"]
#     )
#     dfG = df_gaps.copy() if df_gaps is not None else _empty_df(
#         ["source_key","source_label","project_name","length_frames","duration_sec","type"]
#     )
#     dfM = df_per_minute.copy() if df_per_minute is not None else _empty_df(
#         ["source","minute","pct_short","pct_long","project_name"]
#     )
#
#     if "source" in dfS.columns and "source_key" not in dfS.columns:
#         dfS = dfS.rename(columns={"source":"source_key"})
#     if "label" in dfS.columns and "source_label" not in dfS.columns:
#         dfS = dfS.rename(columns={"label":"source_label"})
#     if "source" in dfM.columns and "source_key" not in dfM.columns:
#         dfM = dfM.rename(columns={"source":"source_key"})
#
#     for df in (dfS, dfG, dfM):
#         if "project_name" not in df.columns:
#             df["project_name"] = ""
#
#     src_labels = (
#         dfS.groupby("source_key", dropna=False)["source_label"]
#            .agg(lambda x: x.dropna().iloc[0] if len(x.dropna()) else (x.iloc[0] if len(x) else ""))
#            .to_dict()
#     )
#     if not src_labels and "source_key" in dfG.columns:
#         for sk in dfG["source_key"].unique(): src_labels[sk] = sk
#     sources_order = list(src_labels.keys())
#
#     # ---------- aggregations ----------
#     agg_overall = (
#         dfS.groupby("source_key", as_index=False)
#            .agg(valid=("percent_valid","mean"),
#                 short=("percent_interp_short","mean"),
#                 long=("percent_interp_long","mean"),
#                 missing=("percent_missing","mean"))
#     )
#     agg_overall["source_label"] = agg_overall["source_key"].map(src_labels)
#     if len(agg_overall):
#         agg_overall = agg_overall.set_index("source_label").loc[[src_labels[k] for k in sources_order]]
#
#     df_long = dfS.loc[:, ["project_name","source_key","source_label","percent_interp_long"]].copy()
#     df_long["source_label"] = df_long["source_key"].map(src_labels).fillna(df_long["source_key"])
#     top_rows = []
#     for sk, sub in df_long.groupby("source_key"):
#         top_rows.append(sub.sort_values("percent_interp_long", ascending=False).head(max_projects_in_panel))
#     df_top = pd.concat(top_rows, ignore_index=True) if top_rows else _empty_df(df_long.columns)
#
#     lengths = dfG["length_frames"].to_numpy(float) if not dfG.empty else np.array([])
#     counts = edges = ccdf_x = ccdf_y = None
#     if lengths.size:
#         bins = np.arange(1, max(int(np.nanmax(lengths)) + 2, long_gap_frames + 2))
#         counts, edges = np.histogram(lengths, bins=bins)
#         sorted_l = np.sort(lengths[~np.isnan(lengths)])
#         if sorted_l.size:
#             uniq, uniq_counts = np.unique(sorted_l, return_counts=True)
#             ccdf_x = uniq
#             ccdf_y = 1.0 - np.cumsum(uniq_counts) / sorted_l.size
#
#     G_long = dfG[dfG["type"].str.lower() == "long"] if "type" in dfG.columns else dfG[dfG["length_frames"] >= long_gap_frames]
#     dur = G_long["duration_sec"].to_numpy(float) if not G_long.empty else np.array([])
#     counts_per_src = (G_long.groupby("source_key")["duration_sec"].count() if not G_long.empty else pd.Series(dtype=int))
#     counts_per_src = counts_per_src.reindex(sources_order).fillna(0).astype(int)
#
#     # ---- Heatmap fix: force dense minute axis 0..heatmap_minutes_max ----
#     if not dfM.empty and "pct_long" in dfM.columns:
#         M = dfM.copy()
#         M["minute"] = M["minute"].astype(int)
#         full_minutes = pd.Index(np.arange(0, heatmap_minutes_max+1), name="minute")
#         heat = (M.groupby(["source_key","minute"])["pct_long"]
#                   .mean().unstack("minute")
#                   .reindex(index=sources_order, columns=full_minutes))
#         heat = heat.fillna(0.0)
#     else:
#         heat = pd.DataFrame(index=sources_order, columns=np.arange(0,1), data=0.0)
#
#     # ---------- figure ----------
#     fig = plt.figure(figsize=figsize, constrained_layout=False)
#     gs  = GridSpec(
#         nrows=6, ncols=3, figure=fig,
#         height_ratios=[1.2, 2.1, 1.5, 1.5, 1.7, 1.3],
#         hspace=0.95, wspace=0.35
#     )
#
#     # Panel 1
#     ax1 = fig.add_subplot(gs[0, :])
#     if not agg_overall.empty:
#         labels = agg_overall.index.tolist()
#         x = np.arange(len(labels))
#         width = 0.6
#         v = agg_overall["valid"].to_numpy()
#         s = agg_overall["short"].to_numpy()
#         l = agg_overall["long"].to_numpy()
#         m = agg_overall["missing"].to_numpy()
#         ax1.bar(x, v, width, label="Valid")
#         ax1.bar(x, s, width, bottom=v, label="Short")
#         ax1.bar(x, l, width, bottom=v+s, label=f"Long (≥{long_gap_frames}f)")
#         ax1.bar(x, m, width, bottom=v+s+l, label="Missing")
#         ax1.set_xticks(x); ax1.set_xticklabels(labels)
#         ax1.set_ylabel("% of frames"); ax1.set_title("Overall composition per source (mean across projects)")
#         ax1.set_ylim(0, 100)
#         # NEW: y-ticks every 20%
#         ax1.set_yticks(np.arange(0, 101, 20))
#         ax1.legend(frameon=False, bbox_to_anchor=(1.02,1.0), loc="upper left")
#         ax1.grid(axis="y", alpha=0.3, linestyle=":")
#     else:
#         ax1.text(0.5,0.5,"No summary data",ha="center",va="center"); ax1.axis("off")
#
#     # Panel 2 (Top-N projects) — unchanged from last version
#     ax2 = fig.add_subplot(gs[1, :])
#     if not df_top.empty:
#         left = 0.0; sep = 0.8
#         tick_pos, tick_lbl = [], []; group_patches = []
#         for i, sk in enumerate(sources_order):
#             sub = df_top[df_top["source_key"] == sk]
#             if sub.empty: continue
#             vals = sub["percent_interp_long"].to_numpy(); n = len(vals)
#             xs = left + np.arange(n); color = plt.cm.tab10(i % 10)
#             ax2.bar(xs, vals, width=0.8, color=color, alpha=0.9)
#             group_patches.append((src_labels.get(sk, sk), color))
#             tick_pos.extend(xs.tolist()); tick_lbl.extend(sub["project_name"].tolist())
#             left = xs[-1] + sep + 1
#         ax2.set_xticks(tick_pos); ax2.set_xticklabels(tick_lbl, rotation=30, ha="right")
#         ax2.set_ylabel("% long")
#         ax2.set_title(f"Top-{max_projects_in_panel} projects by % long interpolation per source")
#         ax2.grid(axis="y", alpha=0.3, linestyle=":")
#         handles = [plt.Line2D([0],[0], color=c, lw=8, label=lab) for lab, c in group_patches]
#         leg2 = ax2.legend(handles=handles, frameon=False, bbox_to_anchor=(1.02,1.0), loc="upper left")
#         ax2.add_artist(leg2)
#     else:
#         ax2.text(0.5,0.5,"No per-project data",ha="center",va="center"); ax2.axis("off")
#
#     # ---------- Panel 3: Gap length per source (step) + CCDF (all sources) ----------
#     ax3 = fig.add_subplot(gs[2, :2])
#     ax3r = fig.add_subplot(gs[2, 2])
#
#     if not dfG.empty and "length_frames" in dfG.columns:
#         # choose bin width and max range
#         step_f = 25  # frames per step
#         max_len = int(dfG["length_frames"].max()) if gap_length_max <= 0 else int(
#             min(gap_length_max, dfG["length_frames"].max()))
#         max_len = max(max_len, long_gap_frames + step_f)  # ensure we pass the threshold
#         edges = np.arange(0, max_len + step_f, step_f)
#         centers = 0.5 * (edges[:-1] + edges[1:])
#
#         # one color per source
#         tab10 = plt.cm.get_cmap("tab10")
#         ymax_after_all = 0
#
#         for i, sk in enumerate(sources_order):
#             vals = dfG.loc[dfG["source_key"] == sk, "length_frames"].to_numpy(dtype=float)
#             if vals.size == 0:
#                 continue
#             cnt, _ = np.histogram(vals, bins=edges)
#             ax3.step(centers, cnt, where="mid", lw=2, alpha=0.9, color=tab10(i % 10),
#                      label=src_labels.get(sk, sk))
#             # track max only after threshold for y-lim
#             mask_after = centers >= long_gap_frames
#             if np.any(mask_after):
#                 ymax_after_all = max(ymax_after_all, int(cnt[mask_after].max()) if cnt[mask_after].size else 0)
#
#         ax3.axvline(long_gap_frames, color="k", linestyle=":", linewidth=1.5,
#                     label=f"Threshold {long_gap_frames}f")
#         ax3.set_xlabel("Gap length (frames)")
#         ax3.set_ylabel("Count")
#         ax3.set_title("Gap length distribution — step per source")
#         ax3.grid(True, alpha=0.3, linestyle=":")
#         if ymax_after_all > 0:
#             ax3.set_ylim(0, max(1, int(ymax_after_all * 1.05)))
#         ax3.legend()
#
#         # CCDF (all sources pooled)
#         lengths_all = dfG["length_frames"].to_numpy(dtype=float)
#         if lengths_all.size:
#             s = np.sort(lengths_all[~np.isnan(lengths_all)])
#             uniq, counts_u = np.unique(s, return_counts=True)
#             ccdf_x = uniq
#             ccdf_y = 1.0 - np.cumsum(counts_u) / s.size
#             ax3r.plot(ccdf_x, ccdf_y, lw=2)
#             ax3r.axvline(long_gap_frames, color="k", linestyle=":", linewidth=1.5)
#             if gap_length_max > 0:
#                 ax3r.set_xlim(-10, gap_length_max)
#             ax3r.set_xlabel("Gap length (frames)")
#             ax3r.set_ylabel("CCDF")
#             ax3r.set_title("CCDF of gap length (all sources)")
#             ax3r.set_ylim(0, 1)
#             ax3r.grid(True, alpha=0.3, linestyle=":")
#         else:
#             ax3r.axis("off")
#     else:
#         ax3.text(0.5, 0.5, "No gap data", ha="center", va="center");
#         ax3.axis("off")
#         ax3r.axis("off")
#
#     # ---------- Panel 4: Long-gap durations per source (step) + counts per source ----------
#     ax4 = fig.add_subplot(gs[3, :2])
#     ax4b = fig.add_subplot(gs[3, 2])
#
#     # Left: per-source step distributions of long-gap durations (seconds)
#     if not G_long.empty and "duration_sec" in G_long.columns:
#         # cap x at requested limit (default handled by caller; you can still pass <=0 to skip)
#         max_sec_data = float(G_long["duration_sec"].max())
#         cap_sec = Long_gap_durations_max if Long_gap_durations_max > 0 else 600.0  # default ≤ 10 min
#         x_max = min(cap_sec, max_sec_data) if max_sec_data > 0 else cap_sec
#         x_max = max(10.0, x_max)  # keep some visible window
#
#         # choose bin width for steps (seconds)
#         step_s = 1.0
#         edges_s = np.arange(0.0, x_max + step_s, step_s)
#         centers_s = 0.5 * (edges_s[:-1] + edges_s[1:])
#
#         tab10 = plt.cm.get_cmap("tab10")
#         for i, sk in enumerate(sources_order):
#             vals = G_long.loc[G_long["source_key"] == sk, "duration_sec"].to_numpy(dtype=float)
#             if vals.size == 0:
#                 continue
#             vals = vals[vals <= x_max]  # respect cap
#             cnt, _ = np.histogram(vals, bins=edges_s)
#             ax4.step(centers_s, cnt, where="mid", lw=2, alpha=0.9, color=tab10(i % 10),
#                      label=src_labels.get(sk, sk))
#
#         ax4.set_xlim(right=x_max)
#         ax4.set_xlabel("Long-gap duration (s)")
#         ax4.set_ylabel("Count")
#         ax4.set_title("Long-gap duration — step per source")
#         ax4.grid(True, alpha=0.3, linestyle=":")
#         ax4.legend()
#     else:
#         ax4.text(0.5, 0.5, "No long-gap durations", ha="center", va="center");
#         ax4.axis("off")
#
#     # Right: counts per source (keep bar chart)
#     if counts_per_src.shape[0]:
#         ax4b.bar([src_labels.get(k, k) for k in counts_per_src.index], counts_per_src.values)
#         ax4b.set_title("Number of long gaps per source")
#         ax4b.set_ylabel("Count")
#         ax4b.set_xticklabels([src_labels.get(k, k) for k in counts_per_src.index])
#         ax4b.grid(axis="y", alpha=0.3, linestyle=":")
#     else:
#         ax4b.axis("off")
#
#     # Panel 5: Per-minute % long heatmap (dense minutes axis -> real heatmap)
#     ax5 = fig.add_subplot(gs[4, :])
#     if not heat.empty:
#         # use the full dense minute axis we created above
#         im = ax5.imshow(
#             heat.to_numpy(),
#             aspect="auto", interpolation="nearest",
#             extent=[0, heat.shape[1], -0.5, heat.shape[0]-0.5],
#             vmin=0, vmax=100
#         )
#         ax5.set_yticks(np.arange(heat.shape[0]))
#         ax5.set_yticklabels([src_labels.get(k,k) for k in heat.index])
#         # ticks every 5 minutes
#         xticks = np.arange(0, heat.shape[1]+1, 5)
#         ax5.set_xticks(xticks); ax5.set_xticklabels([str(m) for m in xticks])
#         ax5.set_xlabel("Minute"); ax5.set_ylabel("Source")
#         ax5.set_title("% long interpolation per minute (avg across projects)")
#         cb = fig.colorbar(im, ax=ax5, fraction=0.046, pad=0.02); cb.set_label("% long")
#     else:
#         ax5.text(0.5,0.5,"No per-minute data",ha="center",va="center"); ax5.axis("off")
#
#     # Panel 6: Top 20 worst
#     ax6 = fig.add_subplot(gs[5, :])
#     if not dfS.empty:
#         worst = (
#             dfS.loc[:, ["project_name","source_key","source_label","percent_interp_long","num_gaps","gap_len_p95","gap_len_max"]]
#                .sort_values(["percent_interp_long","gap_len_p95","gap_len_max"], ascending=False)
#                .head(20)
#         )
#         ax6.axis("off")
#         lines = ["Top 20 worst (by % long)"]
#         for _, r in worst.iterrows():
#             lines.append(
#                 f"• {r['project_name']} — {src_labels.get(r['source_key'], r['source_key'])}: "
#                 f"%long={r['percent_interp_long']:.2f} | gaps={int(r['num_gaps'])} | "
#                 f"p95={r['gap_len_p95']:.1f}f | max={int(r['gap_len_max'])}f"
#             )
#         ax6.text(0.01, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10)
#     else:
#         ax6.text(0.5,0.5,"No summary data",ha="center",va="center"); ax6.axis("off")
#
#     fig.suptitle("Interpolation Overview — All Experiments", y=0.995, fontsize=15)
#     fig.tight_layout()
#     return fig

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

@save_plot
def plot_united_overview(
    df_summary,
    df_gaps,
    df_timebin,   # unused (API compat)
    df_fps,
    streams=None,
    title="Tracking — United Overview",
    config=None,
    save_params=None,
    state_colors=None,
):
    sp = save_params or {}
    figsize   = sp.get("figsize", (18, 18))
    topN      = int(sp.get("topN_per_source", 10))

    pp      = (config or {}).get("preprocessing", {}) or {}
    t_thr_s = float(pp.get("long_time_threshold_s", 3.0))
    d_thr   = float(pp.get("long_dist_threshold_cm", 15.0))
    v_thr   = pp.get("speed_max_cm_s", 30.0)
    v_thr   = float(v_thr) if v_thr is not None else 30.0  # default for speed x-limit

    default_streams = ['TC_HeadPos', 'PC_HeadPos', 'EC_TailPos']
    streams = list(streams) if streams else default_streams
    if isinstance(df_summary, pd.DataFrame) and not df_summary.empty and "stream" in df_summary.columns:
        present = [s for s in streams if s in set(df_summary["stream"].astype(str))]
        streams = present if present else streams
    if len(streams) == 0:
        raise ValueError("No streams to plot.")

    tab10 = plt.cm.get_cmap("tab10")
    stream_colors = {k: tab10(i % 10) for i, k in enumerate(streams)}

    if state_colors is None:
        state_colors = {
            0: "#bdbdbd",
            1: "#1b9e77",
            2: "#d95f02",
            3: "#7570b3",
            4: "#e7298a",
        }

    def _aggregate_summary_by_stream(dfsum, streams):
        if dfsum.empty or "stream" not in dfsum.columns:
            return pd.DataFrame(index=streams)
        df = dfsum[dfsum["stream"].isin(streams)].copy()
        count_cols = [
            "frames_total","frames_valid","frames_interp_short",
            "frames_interp_long_time","frames_interp_long_distance","frames_missing"
        ]
        have_counts = set(count_cols).issubset(df.columns)
        if have_counts:
            agg = df.groupby("stream", as_index=True)[count_cols].sum(numeric_only=True)
            denom = agg["frames_total"].clip(lower=1)
            out = pd.DataFrame({
                "percent_valid":                 100.0 * agg["frames_valid"]                 / denom,
                "percent_interp_short":          100.0 * agg["frames_interp_short"]          / denom,
                "percent_interp_long_time":      100.0 * agg["frames_interp_long_time"]      / denom,
                "percent_interp_long_distance":  100.0 * agg["frames_interp_long_distance"]  / denom,
                "percent_missing":               100.0 * agg["frames_missing"]               / denom,
            })
            out["percent_interp_long_union"] = out["percent_interp_long_time"] + out["percent_interp_long_distance"]
            return out.reindex(streams)
        perc_cols = [
            "percent_valid","percent_interp_short",
            "percent_interp_long_time","percent_interp_long_distance","percent_missing",
            "percent_interp_long_union"
        ]
        present = [c for c in perc_cols if c in df.columns]
        out = df.groupby("stream", as_index=True)[present].mean(numeric_only=True)
        return out.reindex(streams)

    # layout:
    # 0) composition
    # 1) counts by type (sufficient conditions)
    # 2) distributions row (duration | distance | speed)
    # 3) Top-N projects (single vertical bar chart)
    # 4) Worst-20 table
    heights = [1.3, 1.4, 1.8, 2.3, 2.0]
    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(nrows=len(heights), ncols=3, height_ratios=heights, hspace=0.55, wspace=0.30)

    # 0) composition
    ax_comp = fig.add_subplot(gs[0, :])
    comp = _aggregate_summary_by_stream(df_summary, streams)
    order = [
        ("percent_valid",                 1, "valid"),
        ("percent_interp_short",          2, "short"),
        ("percent_interp_long_time",      3, "long-time"),
        ("percent_interp_long_distance",  4, "long-distance"),
        ("percent_missing",               0, "missing"),
    ]
    bottom = np.zeros(len(streams))
    handles = []; labels = []
    for col, code, lab in order:
        if col in comp.columns:
            vals = comp[col].to_numpy(float)
            h = ax_comp.bar(np.arange(len(streams)), vals, bottom=bottom, width=0.6,
                            color=state_colors.get(code, "#ccc"), label=lab)
            if np.nanmax(vals) > 0:
                handles.append(h[0]); labels.append(lab)
            bottom += np.nan_to_num(vals, nan=0.0)
    ax_comp.set_xticks(np.arange(len(streams))); ax_comp.set_xticklabels(streams, rotation=20)
    ax_comp.set_ylabel("%"); ax_comp.set_title("State composition by stream")
    if handles:
        bb = ax_comp.get_position()
        fig.legend(handles, labels, frameon=False, ncol=1,
                   bbox_to_anchor=(0.995, bb.y1), loc="upper right", fontsize=9)

    # 1) counts by type (sufficient conditions)
    ax_cnt = fig.add_subplot(gs[1, :])
    g = df_gaps[df_gaps["stream"].isin(streams)].copy() if ("stream" in df_gaps.columns) else df_gaps.copy()
    cnt_handles = []; cnt_labels=[]
    if (not g.empty) and {"is_long_time","is_long_distance","stream"}.issubset(g.columns):
        # ensure booleans, no NaNs
        g["is_long_time"] = g["is_long_time"].fillna(False).astype(bool)
        g["is_long_distance"] = g["is_long_distance"].fillna(False).astype(bool)
        grp = g.groupby("stream").agg(
            n_long_time_any=("is_long_time", "sum"),
            n_long_dist_any=("is_long_distance", "sum"),
            n_both=("stream", lambda s: int(np.sum(g.loc[s.index, "is_long_time"] & g.loc[s.index, "is_long_distance"])))
        ).reindex(streams).fillna(0)
        x = np.arange(len(streams)); w = 0.27
        h1 = ax_cnt.bar(x - w, grp["n_long_time_any"], width=w, color="#7570b3", label="long-time (any)")
        h2 = ax_cnt.bar(x + 0.0, grp["n_long_dist_any"], width=w, color="#e7298a", label="long-distance (any)")
        h3 = ax_cnt.bar(x + w, grp["n_both"],          width=w, color="#d62728", label="both")
        cnt_handles = [h1[0], h2[0], h3[0]]; cnt_labels=["long-time (any)","long-distance (any)","both"]
        ax_cnt.set_xticks(x); ax_cnt.set_xticklabels(streams, rotation=20)
        ax_cnt.set_ylabel("count"); ax_cnt.set_title("Gap counts (sufficient conditions)")
    else:
        ax_cnt.text(0.5, 0.5, "No gap-type flags", ha="center", va="center", transform=ax_cnt.transAxes)
        ax_cnt.set_axis_off()
    if cnt_handles:
        bb = ax_cnt.get_position()
        fig.legend(cnt_handles, cnt_labels, frameon=False, ncol=1,
                   bbox_to_anchor=(0.995, bb.y1), loc="upper right", fontsize=9)

    # 2) Distributions row
    # (a) duration
    ax_dur = fig.add_subplot(gs[2, 0])
    dur_handles = []; dur_labels = []
    dur_ctrs = None; dur_counts = []
    for k in streams:
        gk = g[g["stream"]==k] if "stream" in g.columns else g
        v = gk["duration_sec"].to_numpy(float) if "duration_sec" in gk.columns else np.array([])
        v = v[np.isfinite(v)]
        if v.size:
            xmax = max(t_thr_s*4, np.percentile(v, 99.5))
            bins = np.linspace(0, xmax, 60)
            c, e = np.histogram(v, bins=bins)
            ctr = 0.5*(e[:-1]+e[1:])
            dur_ctrs = ctr; dur_counts.append(c)
            h = ax_dur.step(ctr, c, where="mid", color=stream_colors[k], label=k)[0]
            dur_handles.append(h); dur_labels.append(k)
    ax_dur.axvline(t_thr_s, color="k", ls=":", lw=1.2)
    if dur_counts:
        counts = np.vstack(dur_counts)
        after  = counts[:, dur_ctrs >= t_thr_s]
        y1 = np.max(after) if after.size else 0.0
        y2 = np.percentile(counts, 95)
        ax_dur.set_ylim(0, 1.05 * max(y1, y2))
    ax_dur.set_xlabel("Gap duration (s)"); ax_dur.set_ylabel("count"); ax_dur.set_title("Gap duration")
    ax_dur.grid(True, alpha=0.3)
    if dur_handles:
        bb = ax_dur.get_position()
        fig.legend(dur_handles, dur_labels, frameon=False, ncol=1,
                   bbox_to_anchor=(0.995, bb.y1), loc="upper right", fontsize=9)

    # (b) distance
    ax_dst = fig.add_subplot(gs[2, 1])
    dst_handles = []; dst_labels = []
    dst_ctrs = None; dst_counts = []
    for k in streams:
        gk = g[g["stream"]==k] if "stream" in g.columns else g
        v = gk["distance_cm"].to_numpy(float) if "distance_cm" in gk.columns else np.array([])
        v = v[np.isfinite(v)]
        if v.size:
            xmax = max(d_thr*4, np.percentile(v, 99.5))
            bins = np.linspace(0, xmax, 60)
            c, e = np.histogram(v, bins=bins)
            ctr = 0.5*(e[:-1]+e[1:])
            dst_ctrs = ctr; dst_counts.append(c)
            h = ax_dst.step(ctr, c, where="mid", color=stream_colors[k], label=k)[0]
            dst_handles.append(h); dst_labels.append(k)
    ax_dst.axvline(d_thr, color="k", ls=":", lw=1.2)
    if dst_counts:
        counts = np.vstack(dst_counts)
        after  = counts[:, dst_ctrs >= d_thr]
        y1 = np.max(after) if after.size else 0.0
        y2 = np.percentile(counts, 95)
        ax_dst.set_ylim(0, 1.05 * max(y1, y2))
    ax_dst.set_xlabel("Gap distance (cm)"); ax_dst.set_ylabel("count"); ax_dst.set_title("Gap distance")
    ax_dst.grid(True, alpha=0.3)
    if dst_handles:
        bb = ax_dst.get_position()
        fig.legend(dst_handles, dst_labels, frameon=False, ncol=1,
                   bbox_to_anchor=(0.995, bb.y1), loc="upper right", fontsize=9)

    # (c) speed histogram (transparent) + dashed threshold
    ax_sph = fig.add_subplot(gs[2, 2])
    speed_vals = []
    if isinstance(df_timebin, pd.DataFrame) and not df_timebin.empty and {"stream","mean_speed_manual","bin"}.issubset(df_timebin.columns):
        for k in streams:
            tb = df_timebin[df_timebin["stream"]==k]
            if not tb.empty:
                v = tb["mean_speed_manual"].to_numpy(float)
                v = v[np.isfinite(v)]
                if v.size: speed_vals.append((k, v, stream_colors[k]))
    if (not speed_vals) and "v_manual_mean" in df_summary.columns:
        for k in streams:
            v = df_summary.loc[df_summary["stream"]==k, "v_manual_mean"].to_numpy(float)
            v = v[np.isfinite(v)]
            if v.size: speed_vals.append((k, v, stream_colors[k]))

    xmax_sp = 4.0 * v_thr
    bins_sp = np.linspace(0, xmax_sp, 50)
    counts_all = []
    for _, v, c in speed_vals:
        cts, _ = np.histogram(v, bins=bins_sp); counts_all.append(cts)
        ax_sph.hist(v, bins=bins_sp, histtype="bar", color=c, alpha=0.28, edgecolor="none")
    ax_sph.axvline(v_thr, color="k", ls=":", lw=1.2)  # dashed threshold line
    if counts_all:
        counts  = np.vstack(counts_all)
        centers = 0.5*(bins_sp[:-1] + bins_sp[1:])
        after   = counts[:, centers >= v_thr]
        y1 = np.max(after) if after.size else 0.0
        y2 = np.percentile(counts, 95)
        ax_sph.set_ylim(0, max(2.0*y1, 1.05*y2))  # requested rule
    ax_sph.set_xlim(0, xmax_sp)
    ax_sph.set_xlabel("Mean speed (cm/s)"); ax_sph.set_ylabel("count"); ax_sph.set_title("Speed (distribution)")
    ax_sph.grid(True, alpha=0.3)

    # 3) Top-N projects — single vertical bar chart (grouped by stream; ordered within stream)
    ax_top = fig.add_subplot(gs[3, :])
    dfS = df_summary.copy()
    if dfS.empty or not {"stream","project_name"}.issubset(dfS.columns):
        ax_top.axis("off"); ax_top.set_title("Top-N by long % — missing columns")
    else:
        have_counts = {"frames_total","frames_interp_long_time","frames_interp_long_distance"}.issubset(dfS.columns)
        if have_counts:
            grp = dfS.groupby(["project_name","stream"], as_index=False).agg(
                frames_total=("frames_total","sum"),
                frames_interp_long_time=("frames_interp_long_time","sum"),
                frames_interp_long_distance=("frames_interp_long_distance","sum"),
            )
            grp["pct_long_union"] = 100.0*(grp["frames_interp_long_time"]+grp["frames_interp_long_distance"]) / grp["frames_total"].clip(lower=1)
        else:
            cols = {"percent_interp_long_time","percent_interp_long_distance"}
            if cols.issubset(dfS.columns):
                grp = dfS.groupby(["project_name","stream"], as_index=False).agg(
                    pct_long_time=("percent_interp_long_time","mean"),
                    pct_long_distance=("percent_interp_long_distance","mean"),
                )
                grp["pct_long_union"] = grp["pct_long_time"] + grp["pct_long_distance"]
            else:
                grp = pd.DataFrame(columns=["project_name","stream","pct_long_union"])

        bars_x = []; bars_h = []; bars_c = []; tick_labels = []
        for k in streams:
            dk = grp[grp["stream"]==k].sort_values("pct_long_union", ascending=False).head(topN)
            for _, r in dk.iterrows():
                bars_x.append(len(tick_labels))
                bars_h.append(r["pct_long_union"])
                bars_c.append(stream_colors[k])
                tick_labels.append(f"{r['project_name']}")
        if bars_x:
            ax_top.bar(bars_x, bars_h, color=bars_c, alpha=0.9)
            ax_top.set_xticks(bars_x); ax_top.set_xticklabels(tick_labels, rotation=30, ha="right")
            ax_top.set_ylabel("% long"); ax_top.set_title(f"Top-{topN} projects by % long (union) — grouped by stream")
            # one legend with stream colors outside right
            handles = [plt.Line2D([0],[0], color=stream_colors[k], lw=6, label=k) for k in streams]
            bb = ax_top.get_position()
            fig.legend(handles, [h.get_label() for h in handles], frameon=False, ncol=1,
                       bbox_to_anchor=(0.995, bb.y1), loc="upper right", fontsize=9)

    # 4) Worst-20 table (primary stream = first in list) with requested stats
    # primary = streams[0]
    ax_tbl = fig.add_subplot(gs[4, :]); ax_tbl.axis("off")
    lines = []
    if (not df_summary.empty) and {"project_name","stream"}.issubset(df_summary.columns):
        # frames + percents from summary
        sums = df_summary[df_summary["is_primary"]].groupby("project_name", as_index=False).agg(
            stream=("stream","first"),
            frames_total=("frames_total","first"),
            frames_lt=("frames_interp_long_time","first"),
            frames_ld=("frames_interp_long_distance","first"),
            precent_lt=("percent_interp_long_time","first"),
            percent_ld=("percent_interp_long_distance","first"),
            percent_long_union=("percent_interp_long_union","first"),
        )
        # sums["pct_long_time"]      = 100.0 * sums["frames_lt"] / sums["frames_total"].clip(lower=1)
        sums["pct_long_time"]      = sums["precent_lt"]
        # sums["pct_long_distance"]  = 100.0 * sums["frames_ld"] / sums["frames_total"].clip(lower=1)
        sums["pct_long_distance"]  = sums["percent_ld"]
        # sums["pct_long_union"]     = sums["pct_long_time"] + sums["pct_long_distance"]
        sums["pct_long_union"]     = sums["percent_long_union"]

        # gaps + durations/distances from df_gaps
        gg = df_gaps[df_gaps["is_primary"]].copy() if "stream" in df_gaps.columns else df_gaps.copy()
        if not gg.empty and "project_name" in gg.columns:
            agg_g = gg.groupby("project_name").agg(
                num_gaps=("project_name","size"),
                dur_med=("duration_sec","median"),
                dur_p95=("duration_sec", lambda v: np.percentile(pd.to_numeric(v, errors="coerce").dropna(), 95) if pd.notna(v).any() else np.nan),
                dur_max=("duration_sec","max"),
                dist_med=("distance_cm","median"),
                dist_p95=("distance_cm", lambda v: np.percentile(pd.to_numeric(v, errors="coerce").dropna(), 95) if pd.notna(v).any() else np.nan),
                dist_max=("distance_cm","max"),
            ).reset_index()
        else:
            agg_g = pd.DataFrame(columns=["project_name","num_gaps","dur_med","dur_p95","dur_max","dist_med","dist_p95","dist_max"])

        dd = sums.merge(agg_g, on="project_name", how="left")
        dd = dd.sort_values("pct_long_union", ascending=False).head(20)

        header = f"{'Top 20 worst projects':36s}  {'stream':11s}  {'gaps':5s}  {'%long':7s}  {'%lt':7s}  {'%ld':8s}  {'dur (med/p95/max s)':20s}  {'dist (med/p95/max cm)':23s}  {'frames_total':13s}"
        lines.append(header)
        for _, r in dd.iterrows():
            pname = str(r["project_name"])[:36]
            lines.append(
                f"{pname:36s}  {str(r['stream']):10s}  "
                f"{int(r.get('num_gaps', 0)):5d}  "
                f"{r['pct_long_union']:6.2f}%  {r['pct_long_time']:6.2f}%  {r['pct_long_distance']:6.2f}%  "
                f"{r.get('dur_med',np.nan):6.2f}/{r.get('dur_p95',np.nan):6.2f}/{r.get('dur_max',np.nan):6.2f}  "
                f"{r.get('dist_med',np.nan):6.2f}/{r.get('dist_p95',np.nan):6.2f}/{r.get('dist_max',np.nan):6.2f}  "
                f"{int(r.get('frames_total',0)):10d}"
            )
    else:
        lines.append("Top 20 worst — data missing")

    ax_tbl.text(0.01, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9)

    if isinstance(df_fps, pd.DataFrame) and not df_fps.empty:
        r = df_fps.iloc[0].to_dict()
        txt = (f"Recording: {r.get('recording_duration_sec', np.nan):.1f}s, "
               f"frames={int(r.get('frames_total', np.nan))} | "
               f"dt median={r.get('dt_median_s', np.nan):.4f}s, p95={r.get('dt_p95_s', np.nan):.4f}s | "
               f"fps_inst median={r.get('fps_inst_median', np.nan):.2f}, p95={r.get('fps_inst_p95', np.nan):.2f}")
        fig.text(0.02, 0.012, txt, fontsize=9, ha="left", va="bottom", wrap=True)

    fig.suptitle(title, fontsize=16, y=0.99)
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.985])
    return fig

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

@save_plot
def plot_spike_rate_comparison(
    dfSpikeRate,
    rate_cols=("spike_rate_bin", "spike_rate_rect", "spike_rate_gauss", "spike_rate_kN"),
    n_cells=7,
    low_k=2,
    high_k=2,
    max_points_per_trace=5000,
    smooth_s=0.0,                 # plot-only Gaussian smoothing in seconds; 0 disables
    random_state=0,
    time_range=None,              # (tmin, tmax) to crop, or None
    title="Spike-rate per cell (low → mid → high by mean spike_rate)",
    save_params={},
):

    figsize = save_params.get("figsize", (12, 16))

    # Optional time crop BEFORE selection & plotting
    if time_range is not None:
        tmin, tmax = time_range
        dfSpikeRate = dfSpikeRate[(dfSpikeRate["timestamp"] >= tmin) & (dfSpikeRate["timestamp"] <= tmax)]
        if dfSpikeRate.empty:
            raise ValueError("No data left after applying time_range.")

    # keep only existing columns and exclude 'spike_rate' from plotting
    rate_cols = [c for c in rate_cols if c in dfSpikeRate.columns]
    if len(rate_cols) == 0:
        raise ValueError("No plottable rate columns found in dfSpikeRate.")
    if "spike_rate" not in dfSpikeRate.columns:
        raise ValueError("dfSpikeRate must contain 'spike_rate' for selection by mean rate.")

    # select cells: low, mid, high by mean spike_rate (computed on cropped data if time_range set)
    means = (dfSpikeRate.groupby("cell", as_index=False)["spike_rate"]
             .mean().rename(columns={"spike_rate": "mean_rate"}))
    total_cells = means.shape[0]
    if total_cells < n_cells:
        n_cells = total_cells
    mid_k = max(0, n_cells - low_k - high_k)
    low_k = min(low_k, n_cells)
    high_k = min(high_k, max(0, n_cells - low_k))

    means_sorted = means.sort_values("mean_rate", ascending=True).reset_index(drop=True)
    low_cells  = means_sorted.head(low_k)["cell"].to_numpy()
    high_cells = means_sorted.tail(high_k)["cell"].to_numpy()

    rem = means_sorted.iloc[low_k:len(means_sorted)-high_k].copy()
    if mid_k > 0 and not rem.empty:
        idx_vals = rem.index.values.astype(float)
        if idx_vals.size >= 2:
            mid_center = 0.5 * (idx_vals[0] + idx_vals[-1])
        else:
            mid_center = idx_vals[0]
        rem["center_dist"] = np.abs(idx_vals - mid_center)
        mid_cells = rem.sort_values(["center_dist", "mean_rate"]).head(mid_k)["cell"].to_numpy()
    else:
        mid_cells = np.array([], dtype=means_sorted["cell"].dtype)

    selected_cells = np.concatenate([low_cells, mid_cells, high_cells])
    seen = set(); ordered = []
    for c in selected_cells:
        if c not in seen:
            seen.add(c); ordered.append(c)
    selected_cells = np.array(ordered[:n_cells])

    # figure & colors
    fig, axes = plt.subplots(nrows=n_cells, ncols=1, figsize=figsize, sharex=True)
    if n_cells == 1:
        axes = [axes]
    cmap = cm.get_cmap('tab10')
    color_map = {col: cmap(i % 10) for i, col in enumerate(rate_cols)}
    color_map["spike_rate_bin"] = "#444444"
    alpha_map = {col: 0.8 if col != "spike_rate_bin" else 0.2 for col in rate_cols}

    rng = np.random.default_rng(int(random_state))
    legend_handles = []
    legend_labels = []

    for row_idx, (ax, cell) in enumerate(zip(axes, selected_cells)):
        d = dfSpikeRate[dfSpikeRate["cell"] == cell].sort_values("timestamp")
        t = d["timestamp"].to_numpy(dtype=float)
        dt_med = float(np.median(np.diff(t))) if t.size > 1 else 1.0

        # simple Gaussian smoothing kernel (single pass; edges attenuated)
        if smooth_s and smooth_s > 0 and dt_med > 0:
            sigma_samples = max(0.001, float(smooth_s) / dt_med)
            half = int(np.ceil(3.0 * sigma_samples))
            kx = np.arange(-half, half + 1, dtype=float)
            kern = np.exp(-0.5 * (kx / sigma_samples) ** 2)
            kern /= np.sum(kern)
        else:
            kern = None

        stride = int(np.ceil(t.size / max_points_per_trace)) if t.size > max_points_per_trace else 1

        for col in rate_cols:
            y = d[col].to_numpy(dtype=float)
            if kern is not None and np.isfinite(y).any():
                y = np.convolve(np.nan_to_num(y, nan=0.0), kern, mode="same")
                # restore NaNs so lines don’t bridge big gaps
                y[~np.isfinite(d[col].to_numpy(dtype=float))] = np.nan
            line, = ax.plot(t[::stride], y[::stride], lw=1.5, alpha=alpha_map[col], label=col, color=color_map[col])

            # collect handles/labels once (first row) for a single legend there
            if row_idx == 0:
                legend_handles.append(line)
                legend_labels.append(col)

        mean_sr = float(d["spike_rate"].mean())
        ax.set_ylabel(f"cell {cell}\nμ={mean_sr:.2f} Hz", rotation=0, ha="right", va="center")
        ax.grid(True, alpha=0.25, linewidth=0.6)

    axes[-1].set_xlabel("time (s)")

    # single legend on the TOP RIGHT of the FIRST row
    if legend_handles:
        # dedup while preserving order
        seen = set()
        uniq_h, uniq_l = [], []
        for h, l in zip(legend_handles, legend_labels):
            if l not in seen:
                seen.add(l); uniq_h.append(h); uniq_l.append(l)
        axes[0].legend(uniq_h, uniq_l, loc="upper right", frameon=False, title="rate variants")

    fig.suptitle(title, y=0.995)
    # fig.tight_layout(rect=[0, 0, 1, 0.98])  # full width since legend sits inside first axes

    return fig





@save_plot
def plot_polar_ratemap(
    rate_map,
    angle_bins,          # degrees
    radius_bins,         # radii
    *,
    cmap="jet",
    levels=25,
    title=None,
    figsize=(3, 3),
    minimized_layout=False,
    ax_in=None,
    overlay=None,        # dict: {'theta_deg','R_norm','peak_radius','median_deg','threshold','text'}
    tick_fontsize=8,
    theta_tick_pad=0,
    r_tick_pad=0,
    xticks_deg = [0, 90, 180, 270],
    xticklabels = ['0° E', '90° S', '180° W', '270° N'],
    save_params=None,
):
    """
    Draw a single polar rate map with optional Rayleigh overlay.

    Parameters
    ----------
    rate_map : array-like, shape (R, T)
        2D array of values to contour (e.g., firing rate across radius × angle).
    angle_bins : array-like, shape (T,)
        Angle bins in degrees corresponding to columns in `rate_map`.
    radius_bins : array-like, shape (R,)
        Radius values corresponding to rows in `rate_map` (already in desired units).
    cmap : str, default="jet"
        Matplotlib colormap for `contourf`.
    levels : int, default=25
        Number of contour levels.
    title : str, optional
        Figure title (used only if a new figure is created internally).
    figsize : tuple(int, int), default=(6, 5)
        Figure size (inches). Used only if `ax_in` is None.
    minimized_layout : bool, default=False
        If True, hide all ticks/labels for a clean tile.
    ax_in : matplotlib.axes._axes.Axes or PolarAxes, optional
        If provided, plot into this axes (must be a polar projection). If None,
        create a new figure and polar axes.
    overlay : dict, optional
        Optional overlay elements. Recognized keys:
        - `theta_deg` (float): Preferred direction in degrees.
        - `R_norm` (float in [0, 1]): Rayleigh vector length (used to scale arrow).
        - `peak_radius` (float): Draw a dashed ring at this radius (magenta).
        - `median_deg` (float): Draw a vertical (radial) line at this direction (orange).
        - `threshold` (float): If provided, text color turns green when ``R_norm > threshold``.
        - `text` (str): Small label rendered in the top-left of the axes.
    tick_fontsize : int, default=8
        Font size for tick labels (ignored if ``minimized_layout`` is True).
    theta_tick_pad : float, default=0
        Extra padding for theta (angle) tick labels.
    r_tick_pad : float, default=0
        Extra padding for radial tick labels.
    save_params : dict, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure that contains the axes.
    """
    rm, ang = _wrap_polar_rate_map(rate_map, angle_bins)
    th = np.deg2rad(ang)
    rr = np.asarray(radius_bins, float)
    radius_bin_size = radius_bins[1]-radius_bins[0]
    radius_bin_edges = np.concatenate(([radius_bins[0]-radius_bin_size/2], radius_bins + radius_bin_size/2))
    TH, RR = np.meshgrid(th, rr, indexing="xy")

    if ax_in is None:
        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(111, projection="polar")
    else:
        ax  = ax_in
        fig = ax.figure

    # --- contour ---
    ax.contourf(TH, RR, rm, levels=levels, cmap=cmap, alpha=0.85, antialiased=True, zorder=0)

    # --- styling ---
    ax.set_facecolor('white')
    # fig.patch.set_alpha(0)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.spines["polar"].set_visible(False)
    ax.grid(True, which="major", alpha=1.0)

    if minimized_layout:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        # theta_degs = np.arange(0, 360, 90)
        ax.set_xticks(np.deg2rad(xticks_deg))
        ax.set_xticklabels(xticklabels)
        if len(radius_bin_edges) >= 4:
            yticks = np.linspace(np.nanmin(radius_bin_edges), np.nanmax(radius_bin_edges), 4).astype(int)
        else:
            yticks = radius_bin_edges
        ax.set_yticks(yticks)
        ax.tick_params(axis="x", labelsize=tick_fontsize, pad=theta_tick_pad)
        ax.tick_params(axis="y", labelsize=tick_fontsize, pad=r_tick_pad)
        ax.set_thetagrids(xticks_deg)

    # “hole” look + headroom
    top = float(np.nanmax(rr)) if rr.size else 1.0
    margin = 0.2 * top
    ax.set_rlim(bottom=-margin, top=top)

    # --- overlays (optional) ---
    if overlay:
        theta_deg  = overlay.get("theta_deg", np.nan)
        R_norm     = overlay.get("R_norm", np.nan)          # [0,1]
        peak_r     = overlay.get("peak_radius", np.nan)
        median_deg = overlay.get("median_deg", np.nan)
        thr        = overlay.get("threshold", None)
        text       = overlay.get("text", "")

        # Ring at peak radius
        if np.isfinite(peak_r):
            ax.plot(np.linspace(0, 2*np.pi, 360), np.full(360, peak_r),
                    color="magenta", linestyle="--", linewidth=0.7)

        # Median direction
        if np.isfinite(median_deg):
            mrad = np.deg2rad((median_deg + 360.0) % 360.0)
            ax.plot([mrad, mrad], [0, top + 5], color="orange", linestyle="-", linewidth=1.0, alpha=0.9, zorder=20)

        # Rayleigh vector arrow (length scaled by max radius)
        if np.isfinite(theta_deg) and np.isfinite(R_norm):
            trad = np.deg2rad((theta_deg + 360.0) % 360.0)
            ray_len = rr.max() * float(np.clip(R_norm, 0.0, 1.0)) if rr.size else 1.0
            ax.annotate(
                "", xy=(trad, ray_len + 0.02 * (rr.max() if rr.size else 1.0)), xytext=(trad, -margin),
                arrowprops=dict(facecolor="magenta", edgecolor="magenta", arrowstyle="->", lw=1.0),
                annotation_clip=False
            )

        # Small text box (green if above threshold)
        if text:
            txt_color = "green" if (thr is not None and np.isfinite(R_norm) and R_norm > thr) else "black"
            pad = -0.28  # position text a bit more to the left
            ax.text(pad, 1 - pad, text,
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=max(tick_fontsize+2, 7), color=txt_color,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.35, boxstyle="round,pad=0.15"),
                    clip_on=False, zorder=50)

    if title:
        fig.suptitle(str(title), y=1.02, fontsize=10, fontweight="bold")
        # fig.tight_layout(rect=[0, 0, 1, 0.98])


    return fig


@save_plot
def plot_angle_radius_ratemap(
    rate_map,
    angle_bins,          # degrees
    radius_bins,         # radii (distance from wall)
    *,
    cmap="jet",
    levels=25,
    title=None,
    figsize=(3, 3),
    minimized_layout=False,
    ax_in=None,
    overlay=None,        # dict: {'theta_deg','R_norm','peak_radius','median_deg','threshold','text'}
    tick_fontsize=8,
    theta_tick_pad=0,
    r_tick_pad=0,
    xticks_deg = [0, 90, 180, 270],
    xticklabels = ['0° E', '90° S', '180° W', '270° N'],
    xlabel="Angle (deg)",
    ylabel="Distance from wall",
    save_params=None,
):
    """
    Draw a 2D (angle × distance) rate map on a standard Cartesian axes.

    Parameters
    ----------
    rate_map : array-like, shape (R, T)
        2D array of values to contour (e.g., firing rate across radius × angle).
    angle_bins : array-like, shape (T,)
        Angle bins in degrees corresponding to columns in `rate_map`.
    radius_bins : array-like, shape (R,)
        Distance-from-wall values corresponding to rows in `rate_map`.
    """
    # Ensure wrapped angle grid like your polar function
    rm, ang = _wrap_polar_rate_map(rate_map, angle_bins)  # rm shape (R, T_wrapped), ang in degrees
    rr = np.asarray(radius_bins, float)
    radius_bin_size = radius_bins[1]-radius_bins[0]
    radius_bin_edges = np.concatenate(([radius_bins[0]-radius_bin_size/2], radius_bins + radius_bin_size/2))
    TH, RR = np.meshgrid(ang, rr, indexing="xy")             # TH: degrees, RR: radii

    # Create axes if needed
    if ax_in is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax  = ax_in
        fig = ax.figure

    # --- contour ---
    # Use contourf in Cartesian coords: x=angle (deg), y=radius
    cf = ax.contourf(TH, RR, rm, levels=levels, cmap=cmap, alpha=0.85, antialiased=True, zorder=0)

    # --- styling ---
    ax.set_facecolor('white')
    ax.grid(True, which="major", alpha=1.0, linewidth=0.6)

    if minimized_layout:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
    else:
        ax.set_xlabel(xlabel, labelpad=max(theta_tick_pad, 0))
        ax.set_ylabel(ylabel, labelpad=max(r_tick_pad, 0))

        # X ticks at compass-like angles
        ax.set_xticks(xticks_deg)
        ax.set_xticklabels(xticklabels, fontsize=tick_fontsize)

        # Y ticks sampled ~quarterly across radii
        if rr.size:
            yt = np.linspace(np.nanmin(radius_bin_edges), np.nanmax(radius_bin_edges), 4)
            ax.set_yticks(yt)
        ax.tick_params(axis="y", labelsize=tick_fontsize)
        ax.tick_params(axis="x", labelsize=tick_fontsize, pad=theta_tick_pad)

        # Keep x-limits to angle span
        ax.set_xlim(float(np.min(ang)), float(np.max(ang)))

    # Keep same visual “flip y” as your polar version (larger distance downwards)
    if rr.size:
        top = float(np.nanmax(rr))
        bottom = float(np.nanmin(rr))
        ax.set_ylim(top, bottom)  # invert y

    # --- overlays (optional) ---
    if overlay:
        theta_deg  = overlay.get("theta_deg", np.nan)
        R_norm     = overlay.get("R_norm", np.nan)          # kept for parity; not used to scale here
        peak_r     = overlay.get("peak_radius", np.nan)
        median_deg = overlay.get("median_deg", np.nan)
        thr        = overlay.get("threshold", None)
        text       = overlay.get("text", "")

        # Horizontal line at peak radius
        if np.isfinite(peak_r):
            ax.axhline(peak_r, color="magenta", linestyle="--", linewidth=0.7, zorder=10)

        # Vertical line at median direction
        if np.isfinite(median_deg):
            ax.axvline(((median_deg + 360.0) % 360.0), color="orange", linestyle="-", linewidth=1.0, alpha=0.9, zorder=11)

        # Preferred direction marker (use vertical line; R_norm not used for length in Cartesian)
        if np.isfinite(theta_deg):
            ax.axvline(((theta_deg + 360.0) % 360.0), color="magenta", linestyle="-", linewidth=1.0, alpha=0.9, zorder=12)

        # Small text box (green if above threshold)
        if text:
            txt_color = "green" if (thr is not None and np.isfinite(R_norm) and R_norm > thr) else "black"
            ax.text(0.02, 0.98, text,
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=max(tick_fontsize+2, 7), color=txt_color,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.35, boxstyle="round,pad=0.15"),
                    clip_on=False, zorder=50)

    if title:
        fig.suptitle(str(title), y=1.02, fontsize=10, fontweight="bold")

    return fig


@save_plot
def plot_latent_space_3d(latent_pca_df, title='Latent Space (3D PCA)', 
                         scatter_size=10, alpha=0.6, separation_pca_per_model=None, save_params={}):
    """
    Create a grid of 3D scatter plots of latent space colored by room.
    Grid layout: rows = folds, columns = offsets.
    Each subplot shows one latent space (one fold/offset combination).
    
    Args:
        latent_pca_df: DataFrame with columns 'PC1', 'PC2', 'PC3', 'offset', 'fold', and 'room'
        title: Plot title (default: 'Latent Space (3D PCA)')
        scatter_size: Size of scatter points (default: 10)
        alpha: Transparency of points (default: 0.6)
        separation_pca_per_model: DataFrame with room separation results for PCA (optional)
            Expected columns: room_pair, misclassification_rate, fold, offset
            If provided, will display A-a and A-B separation misclassification rates in each subplot title
        save_params: Dictionary with 'config' and 'path' for saving
    
    Returns:
        matplotlib figure
    """
    # Check required columns
    required_cols = ['PC1', 'PC2', 'PC3', 'offset', 'fold']
    if not all(col in latent_pca_df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    

    # Get unique folds and offsets
    folds = sorted(latent_pca_df['fold'].unique())
    offsets = sorted(latent_pca_df['offset'].unique())
    n_folds = len(folds)
    n_offsets = len(offsets)
    
    # Get unique rooms and create color palette (same for all subplots)
    rooms = sorted(latent_pca_df['room'].unique())
    n_rooms = len(rooms)
    color_palette = cm.get_cmap("tab20c", n_rooms + 2)
    
    # Create figure with grid of subplots
    fig = plt.figure(figsize=(3 * n_offsets, 3 * n_folds))
    
    # Create grid: rows = folds, columns = offsets
    for row_idx, fold in enumerate(folds):
        for col_idx, offset in enumerate(offsets):
            subplot_idx = row_idx * n_offsets + col_idx + 1
            
            # Create 3D subplot
            ax = fig.add_subplot(n_folds, n_offsets, subplot_idx, projection='3d')
            
            # Filter data for this fold/offset combination
            mask = (latent_pca_df['fold'] == fold) & (latent_pca_df['offset'] == offset)
            subset_df = latent_pca_df[mask]
            
            if len(subset_df) == 0:
                ax.text(0.5, 0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Fold {fold}, Offset {offset}\n(No data)', fontsize=10)
                continue
            
            # Plot each room with different color
            for i, room in enumerate(rooms):
                room_mask = subset_df['room'] == room
                room_data = subset_df[room_mask]
                
                if len(room_data) == 0:
                    continue
                
                ax.scatter(
                    room_data['PC1'].values,
                    room_data['PC2'].values,
                    room_data['PC3'].values,
                    c=[color_palette(i)],
                    label=str(room),
                    s=scatter_size,
                    alpha=alpha
                )
            
            # Set labels (only on edges)
            if col_idx == 0:
                ax.set_ylabel('PC2', fontsize=10)
            if row_idx == n_folds - 1:
                ax.set_xlabel('PC1', fontsize=10)
            ax.set_zlabel('PC3', fontsize=10)
            
            # Set title for each subplot with optional room separation info
            subplot_title = f'Fold {fold}, Offset {offset}'
            
            # Add room separation info if provided
            if separation_pca_per_model is not None and len(separation_pca_per_model) > 0:
                try:
                    # Filter separation data for this fold/offset
                    sep_mask = (separation_pca_per_model['fold'] == fold) & (separation_pca_per_model['offset'] == offset)
                    sep_subset = separation_pca_per_model[sep_mask]
                    
                    if len(sep_subset) > 0 and 'misclassification_rate' in sep_subset.columns:
                        # Look for A-a and A-B pairs
                        sep_parts = []
                        
                        # Check for A_a or a_A
                        for pair_name in ['A_a', 'a_A']:
                            pair_mask = sep_subset['room_pair'] == pair_name
                            if pair_mask.any():
                                error_val = sep_subset[pair_mask]['misclassification_rate'].iloc[0]
                                if pd.notna(error_val):
                                    sep_parts.append(f"A-a: {error_val:.2f}")
                                break
                        
                        # Check for A_B or B_A
                        for pair_name in ['A_B', 'B_A']:
                            pair_mask = sep_subset['room_pair'] == pair_name
                            if pair_mask.any():
                                error_val = sep_subset[pair_mask]['misclassification_rate'].iloc[0]
                                if pd.notna(error_val):
                                    sep_parts.append(f"A-B: {error_val:.2f}")
                                break
                        
                        if sep_parts:
                            subplot_title += f"\n{', '.join(sep_parts)} (chance=0.5)"
                except Exception as e:
                    # Silently fail if separation info can't be extracted
                    pass
            
            ax.set_title(subplot_title, fontsize=10)
            
            # Add legend only to first subplot
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='upper left', fontsize=8, bbox_to_anchor=(0, 1))
            
            # Set grid
            ax.grid(True, alpha=0.3)
    
    # Set overall title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])  # Leave space for suptitle
    
    return fig


@save_plot
def plot_latent_space_2d(latent_pca_df, title='Latent Space (2D PCA)', 
                         scatter_size=10, alpha=0.6, separation_pca_per_model=None, save_params={}):
    """
    Create a grid of 2D scatter plots of latent space colored by room.
    Grid layout: rows = folds, columns = offsets.
    Each subplot shows one latent space (one fold/offset combination).
    
    Args:
        latent_pca_df: DataFrame with columns 'PC1', 'PC2', 'offset', 'fold', and 'room'
        title: Plot title (default: 'Latent Space (2D PCA)')
        scatter_size: Size of scatter points (default: 10)
        alpha: Transparency of points (default: 0.6)
        separation_pca_per_model: DataFrame with room separation results for PCA (optional)
            Expected columns: room_pair, misclassification_rate, fold, offset
            If provided, will display A-a and A-B separation misclassification rates in each subplot title
        save_params: Dictionary with 'config' and 'path' for saving
    
    Returns:
        matplotlib figure
    """
    # Check required columns
    required_cols = ['PC1', 'PC2', 'offset', 'fold']
    if not all(col in latent_pca_df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    

    # Get unique folds and offsets
    folds = sorted(latent_pca_df['fold'].unique())
    offsets = sorted(latent_pca_df['offset'].unique())
    n_folds = len(folds)
    n_offsets = len(offsets)
    
    # Get unique rooms and create color palette (same for all subplots)
    rooms = sorted(latent_pca_df['room'].unique())
    n_rooms = len(rooms)
    color_palette = cm.get_cmap("tab20c", n_rooms + 2)
    
    # Create figure with grid of subplots
    fig, axes = plt.subplots(n_folds, n_offsets, figsize=(3 * n_offsets, 3 * n_folds))
    
    # Handle case where there's only one subplot
    if n_folds == 1 and n_offsets == 1:
        axes = np.array([[axes]])
    elif n_folds == 1:
        axes = axes.reshape(1, -1)
    elif n_offsets == 1:
        axes = axes.reshape(-1, 1)
    
    # Create grid: rows = folds, columns = offsets
    for row_idx, fold in enumerate(folds):
        for col_idx, offset in enumerate(offsets):
            ax = axes[row_idx, col_idx]
            
            # Filter data for this fold/offset combination
            mask = (latent_pca_df['fold'] == fold) & (latent_pca_df['offset'] == offset)
            subset_df = latent_pca_df[mask]
            
            if len(subset_df) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Fold {fold}, Offset {offset}\n(No data)', fontsize=10)
                continue
            
            # Plot each room with different color
            for i, room in enumerate(rooms):
                room_mask = subset_df['room'] == room
                room_data = subset_df[room_mask]
                
                if len(room_data) == 0:
                    continue
                
                ax.scatter(
                    room_data['PC1'].values,
                    room_data['PC2'].values,
                    c=[color_palette(i)],
                    label=str(room),
                    s=scatter_size,
                    alpha=alpha
                )
            
            # Set labels (only on edges)
            if col_idx == 0:
                ax.set_ylabel('PC2', fontsize=10)
            if row_idx == n_folds - 1:
                ax.set_xlabel('PC1', fontsize=10)
            
            # Set title for each subplot with optional room separation info
            subplot_title = f'Fold {fold}, Offset {offset}'
            
            # Add room separation info if provided
            if separation_pca_per_model is not None and len(separation_pca_per_model) > 0:
                try:
                    # Filter separation data for this fold/offset
                    sep_mask = (separation_pca_per_model['fold'] == fold) & (separation_pca_per_model['offset'] == offset)
                    sep_subset = separation_pca_per_model[sep_mask]
                    
                    if len(sep_subset) > 0 and 'misclassification_rate' in sep_subset.columns:
                        # Look for A-a and A-B pairs
                        sep_parts = []
                        
                        # Check for A_a or a_A
                        for pair_name in ['A_a', 'a_A']:
                            pair_mask = sep_subset['room_pair'] == pair_name
                            if pair_mask.any():
                                error_val = sep_subset[pair_mask]['misclassification_rate'].iloc[0]
                                if pd.notna(error_val):
                                    sep_parts.append(f"A-a: {error_val:.2f}")
                                break
                        
                        # Check for A_B or B_A
                        for pair_name in ['A_B', 'B_A']:
                            pair_mask = sep_subset['room_pair'] == pair_name
                            if pair_mask.any():
                                error_val = sep_subset[pair_mask]['misclassification_rate'].iloc[0]
                                if pd.notna(error_val):
                                    sep_parts.append(f"A-B: {error_val:.2f}")
                                break
                        
                        if sep_parts:
                            subplot_title += f"\n{', '.join(sep_parts)} (chance=0.5)"
                except Exception as e:
                    # Silently fail if separation info can't be extracted
                    pass
            
            ax.set_title(subplot_title, fontsize=10)
            
            # Add legend only to first subplot
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='upper left', fontsize=8, bbox_to_anchor=(0, 1))
            
            # Set grid
            ax.grid(True, alpha=0.3)
    
    # Set overall title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])  # Leave space for suptitle
    
    return fig


@save_plot
def plot_room_separation_heatmap(separation_df, metric='misclassification_rate', title='Room Separation', save_params={}):
    """
    Create a heatmap showing room separation metrics for each room pair.
    
    Args:
        separation_df: DataFrame with columns: room_pair, misclassification_rate_mean (or per-model columns)
        metric: Which metric to plot ('misclassification_rate')
        title: Title for the plot
        save_params: Dictionary with 'config' and 'path' for saving
    
    Returns:
        matplotlib figure
    """
    # Determine if we have aggregated or per-model data
    if f'{metric}_mean' in separation_df.columns:
        metric_col = f'{metric}_mean'
        has_std = f'{metric}_std' in separation_df.columns
    elif metric in separation_df.columns:
        # Per-model data - aggregate it
        metric_col = metric
        separation_df = separation_df.groupby('room_pair').agg({
            metric: ['mean', 'std']
        })
        separation_df.columns = ['_'.join(col).strip() for col in separation_df.columns.values]
        separation_df = separation_df.reset_index()
        metric_col = f'{metric}_mean'
        has_std = f'{metric}_std' in separation_df.columns
    else:
        raise ValueError(f"Metric '{metric}' not found in DataFrame columns: {separation_df.columns.tolist()}")
    
    # Extract room pairs and create symmetric matrix
    room_pairs = separation_df['room_pair'].values
    rooms = set()
    for pair_str in room_pairs:
        rooms.update(pair_str.split('_'))
    rooms = sorted(list(rooms))
    
    # Create symmetric matrix
    n_rooms = len(rooms)
    matrix = np.full((n_rooms, n_rooms), np.nan)
    
    for _, row in separation_df.iterrows():
        pair_str = row['room_pair']
        room1, room2 = pair_str.split('_')
        idx1 = rooms.index(room1)
        idx2 = rooms.index(room2)
        value = row[metric_col]
        matrix[idx1, idx2] = value
        matrix[idx2, idx1] = value  # Make symmetric
    
    # Set diagonal to 1.0 (perfect separation with itself) or NaN
    np.fill_diagonal(matrix, np.nan)
    
    # Create DataFrame for heatmap
    heatmap_df = pd.DataFrame(matrix, index=rooms, columns=rooms)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(6, n_rooms * 0.8), max(5, n_rooms * 0.7)))
    
    # Plot heatmap (fix scale so chance level 0.5 is visible for misclassification_rate)
    vmin = np.nanmin(matrix)
    vmax = np.nanmax(matrix)
    if metric == 'misclassification_rate':
        vmin = min(vmin, 0.0)
        vmax = max(vmax, 1.0)
    
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': metric.replace('_', ' ').title()},
        square=True,
        linewidths=0.5,
        linecolor='gray',
        ax=ax
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Room', fontsize=12)
    ax.set_ylabel('Room', fontsize=12)
    
    plt.tight_layout()
    
    return fig


@save_plot
def plot_room_separation_bar_comparison(separation_df, accuracy_cols, title='Room Separation Accuracy Comparison', save_params={}):
    """
    Create a bar plot comparing room separation accuracy across different methods.
    
    Args:
        separation_df: DataFrame with columns: room_pair, and mean/std columns for each accuracy type
                      Expected columns: room_pair, accuracy_raw_latent_mean, accuracy_raw_latent_std, etc.
        accuracy_cols: List of accuracy column names (e.g., ['accuracy_raw_latent', 'accuracy_pca', 'accuracy_pca_3d', 'accuracy_pca_2d'])
        title: Title for the plot
        save_params: Dictionary with 'config' and 'path' for saving
    
    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Filter to available columns
    available_cols = [col for col in accuracy_cols if f'{col}_mean' in separation_df.columns]
    
    if len(available_cols) == 0:
        raise ValueError(f"No accuracy columns found in DataFrame. Available columns: {separation_df.columns.tolist()}")
    
    # Get room pairs
    room_pairs = separation_df['room_pair'].values
    n_pairs = len(room_pairs)
    n_methods = len(available_cols)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(max(8, n_pairs * 1.5), 6))
    
    # Bar width and positions
    bar_width = 0.8 / n_methods
    x = np.arange(n_pairs)
    
    # Color palette for different methods
    colors = plt.cm.Set2(np.linspace(0, 1, n_methods))
    
    # Method labels
    method_labels = {
        'accuracy_raw_latent': 'Raw Latents',
        'accuracy_pca': 'Full PCA',
        'accuracy_pca_3d': 'PCA (3D)',
        'accuracy_pca_2d': 'PCA (2D)'
    }
    
    # Plot bars for each method
    for i, acc_col in enumerate(available_cols):
        mean_col = f'{acc_col}_mean'
        std_col = f'{acc_col}_std'
        
        means = separation_df[mean_col].values
        stds = separation_df[std_col].values if std_col in separation_df.columns else np.zeros(n_pairs)
        
        # Calculate bar positions (centered around each room pair)
        offset = (i - (n_methods - 1) / 2) * bar_width
        positions = x + offset
        
        # Plot bars with error bars
        ax.bar(
            positions,
            means,
            width=bar_width,
            yerr=stds,
            capsize=3,
            label=method_labels.get(acc_col, acc_col),
            color=colors[i],
            alpha=0.8,
            edgecolor='white',
            linewidth=1
        )
    
    # Styling (plot shows misclassification rate: lower is better, chance = 0.5)
    ax.set_xlabel('Room Pair', fontsize=12, fontweight='medium')
    ax.set_ylabel('Misclassification rate', fontsize=12, fontweight='medium')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(room_pairs, fontsize=10)
    # Chance level: show 0.5 once (real chance is still computed and stored in data)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, label='Chance (0.5)')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_ylim([0, 1.1])
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    return fig


@save_plot
def _plot_single_trajectory_gradient_comparison(
    room_data,
    latent_subset,
    chosen_room,
    other_rooms,
    boundary_points,
    pos_range,
    scaler=None,
    fold=None,
    offset=None,
    save_params={}
):
    """
    Create a single gradient comparison figure for one fold/offset/room combination.
    This is a helper function that generates one figure with @save_plot decorator.
    
    Layout: 2 rows (horizontal/vertical gradient) × (n_rooms + 1) columns
    - Column 1: Chosen room trajectory
    - Column 2: Latent space
    - Columns 3+: Other rooms (one per column)
    
    Args:
        room_data: DataFrame with chosen room's predictions and all rooms' predictions
        latent_subset: DataFrame with latent PCA data for chosen room
        chosen_room: Chosen room identifier
        other_rooms: List of other room identifiers
        boundary_points: Boundary points for plotting
        pos_range: Position range tuple (min, max) for both X and Y
        scaler: Optional scaler for transforming positions (default: None)
        fold: Fold number (optional)
        offset: Offset value (optional)
        save_params: Dictionary with 'config' and 'path' for saving
    
    Returns:
        matplotlib figure
    """
    logger = get_logger(__name__)
    
    boundary_points = boundary_points.copy() if boundary_points is not None else None

    # Helper function to apply gradient pattern
    def apply_gradient_pattern(xv, yv, gradient_type="horizontal", range_tuple=None):
        if gradient_type == "horizontal":
            gradient = xv
            min_val, max_val = range_tuple[0], range_tuple[1]  # X range
        elif gradient_type == "vertical":
            gradient = yv
            min_val, max_val = range_tuple[0], range_tuple[1]  # Y range
        else:
            raise ValueError("gradient_type must be 'horizontal' or 'vertical'")
        
        if gradient.size == 0:
            return np.zeros_like(gradient)
        
        # Normalize to [0, 1] based on boundary range
        normalized = (gradient - min_val) / (max_val - min_val) if (max_val - min_val) > 0 else np.zeros_like(gradient)
        return np.clip(normalized, 0, 1)
    
    # Get chosen room's X and Y positions and timestamps
    chosen_room_x = room_data[f'X_pred_{chosen_room}'].values
    chosen_room_y = room_data[f'Y_pred_{chosen_room}'].values
    chosen_room_timestamps = room_data['timestamp'].values
    
    # Transform positions if scaler is provided (from normalized to cm)
    if scaler is not None:
        # Transform all prediction columns at once
        all_pred_cols = [col for col in room_data.columns if col.startswith('X_pred_') or col.startswith('Y_pred_')]

        # Get positions
        positions = room_data[all_pred_cols].values.copy()
        
        # Handle NaNs: flatten, transform valid values, restore shape
        nan_mask = np.isnan(positions).any(axis=1)
        if not nan_mask.all():
            valid_positions = positions[~nan_mask]
            if len(valid_positions) > 0:
                # Transform valid positions
                valid_transformed = apply_scaler_transform(valid_positions, scaler, reverse=True)
                positions[~nan_mask] = valid_transformed
        
        # Put transformed values back into room_data
        room_data[all_pred_cols] = positions
        
        # Update chosen_room_x and chosen_room_y after transformation
        chosen_room_x = room_data[f'X_pred_{chosen_room}'].values
        chosen_room_y = room_data[f'Y_pred_{chosen_room}'].values
        
        # Transform boundary_points if provided (from normalized to cm)
        if boundary_points is not None:
            boundary_xy = boundary_points[:, :2].copy()
            nan_mask = np.isnan(boundary_xy).any(axis=1)
            if not nan_mask.all():
                valid_boundary = boundary_xy[~nan_mask]
                if len(valid_boundary) > 0:
                    valid_transformed = apply_scaler_transform(valid_boundary, scaler, reverse=True)
                    boundary_xy[~nan_mask] = valid_transformed
            boundary_points[:, :2] = boundary_xy

        # Inverse transform pos_range
        if pos_range is not None:
            pos_range = tuple(apply_scaler_transform(value, scaler, reverse=True) for value in pos_range)
    
    # Use pos_range for both x_range and y_range for gradient normalization
    # This must be AFTER scaler transformation so ranges match the transformed positions (cm)
    x_range = y_range = pos_range
    
    # Compute chosen room's gradient values
    chosen_room_gradient_h = apply_gradient_pattern(chosen_room_x, chosen_room_y, "horizontal", x_range)
    chosen_room_gradient_v = apply_gradient_pattern(chosen_room_x, chosen_room_y, "vertical", y_range)
    
    # Create color mapping dictionaries
    color_map_h = {ts: grad for ts, grad in zip(chosen_room_timestamps, chosen_room_gradient_h)}
    color_map_v = {ts: grad for ts, grad in zip(chosen_room_timestamps, chosen_room_gradient_v)}
    
    # Colormap
    cmap = cm.get_cmap('jet')
    
    # Convert to RGB colors for chosen room
    colors_h = cmap(chosen_room_gradient_h)
    colors_v = cmap(chosen_room_gradient_v)
    
    # Create figure with 2 rows × (n_rooms + 1) columns
    n_cols = len(other_rooms) + 2  # chosen room + latent + other rooms
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))
    fig.suptitle(f'Trajectory Gradient Comparison - Fold {fold}, Offset {offset}, Room {chosen_room}', fontsize=14, fontweight='bold')
    
    # Add row titles on the left
    fig.text(0.02, 0.75, 'Horizontal Gradient', rotation=90, fontsize=12, fontweight='bold', 
            ha='center', va='center')
    fig.text(0.02, 0.25, 'Vertical Gradient', rotation=90, fontsize=12, fontweight='bold', 
            ha='center', va='center')
    
    # Helper function to set up axis with boundaries
    def setup_axis_with_boundaries(ax):
        plot_boundaries(boundary_points, ax, color='black', alpha=0.5, linewidth=2)
        ax.set_xlim(pos_range)
        ax.set_ylim(pos_range[::-1])  # Reverse y-axis
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
    
    # Row 1: Horizontal Gradient
    # Column 1: Chosen room trajectory with horizontal gradient
    ax = axes[0, 0]
    ax.scatter(chosen_room_x, chosen_room_y, c=colors_h, s=10, alpha=0.6, edgecolors='none')
    setup_axis_with_boundaries(ax)
    ax.set_title(f'Room {chosen_room}', fontsize=10)
    
    # Column 2: Latent space 2D projection colored by chosen room's gradient
    ax = axes[0, 1]
    latent_timestamps = latent_subset['timestamp'].values
    latent_colors_h = [color_map_h.get(ts, 0.5) for ts in latent_timestamps]
    latent_colors_h = np.array(latent_colors_h)
    latent_colors_h_rgb = cmap(latent_colors_h)
    ax.scatter(
        latent_subset['PC1'].values,
        latent_subset['PC2'].values,
        c=latent_colors_h_rgb,
        s=10,
        alpha=0.6,
        edgecolors='none'
    )
    ax.set_title('Latent Space (PC1 vs PC2)', fontsize=10)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.grid(True, alpha=0.3)
    
    # Columns 3+: Other rooms trajectories (one per column)
    for col_idx, other_room in enumerate(other_rooms, start=2):
        ax = axes[0, col_idx]
        x_col = f'X_pred_{other_room}'
        y_col = f'Y_pred_{other_room}'
        
        if x_col not in room_data.columns or y_col not in room_data.columns:
            ax.text(0.5, 0.5, f'Room {other_room}\nData not available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title(f'Room {other_room}', fontsize=10)
            continue
        
        other_colors_h = [color_map_h.get(ts, 0.5) for ts in chosen_room_timestamps]
        other_colors_h = np.array(other_colors_h)
        other_colors_h_rgb = cmap(other_colors_h)
        
        ax.scatter(
            room_data[x_col].values,
            room_data[y_col].values,
            c=other_colors_h_rgb,
            s=10,
            alpha=0.6,
            edgecolors='none'
        )
        setup_axis_with_boundaries(ax)
        ax.set_title(f'Room {other_room}', fontsize=10)
    
    # Row 2: Vertical Gradient
    # Column 1: Chosen room trajectory with vertical gradient
    ax = axes[1, 0]
    ax.scatter(chosen_room_x, chosen_room_y, c=colors_v, s=10, alpha=0.6, edgecolors='none')
    setup_axis_with_boundaries(ax)
    ax.set_title(f'Room {chosen_room}', fontsize=10)
    
    # Column 2: Latent space 2D projection colored by chosen room's Y gradient
    ax = axes[1, 1]
    latent_colors_v = [color_map_v.get(ts, 0.5) for ts in latent_timestamps]
    latent_colors_v = np.array(latent_colors_v)
    latent_colors_v_rgb = cmap(latent_colors_v)
    ax.scatter(
        latent_subset['PC1'].values,
        latent_subset['PC2'].values,
        c=latent_colors_v_rgb,
        s=10,
        alpha=0.6,
        edgecolors='none'
    )
    ax.set_title('Latent Space (PC1 vs PC2)', fontsize=10)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.grid(True, alpha=0.3)
    
    # Columns 3+: Other rooms trajectories (one per column)
    for col_idx, other_room in enumerate(other_rooms, start=2):
        ax = axes[1, col_idx]
        x_col = f'X_pred_{other_room}'
        y_col = f'Y_pred_{other_room}'
        
        if x_col not in room_data.columns or y_col not in room_data.columns:
            ax.text(0.5, 0.5, f'Room {other_room}\nData not available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title(f'Room {other_room}', fontsize=10)
            continue
        
        other_colors_v = [color_map_v.get(ts, 0.5) for ts in chosen_room_timestamps]
        other_colors_v = np.array(other_colors_v)
        other_colors_v_rgb = cmap(other_colors_v)
        
        ax.scatter(
            room_data[x_col].values,
            room_data[y_col].values,
            c=other_colors_v_rgb,
            s=10,
            alpha=0.6,
            edgecolors='none'
        )
        setup_axis_with_boundaries(ax)
        ax.set_title(f'Room {other_room}', fontsize=10)
    
    plt.tight_layout(rect=[0.03, 0, 1, 0.97])
    
    return fig


def plot_trajectory_with_latent_gradient_comparison(
    merged_df, 
    rooms,
    boundary_points,
    pos_range,
    scaler=None,
    k_folds=None,
    max_figures=10,
    save_params={}
):
    """
    Create gradient comparison plots for each fold/offset/room combination.
    Each figure shows predicted position trajectories with gradient colormaps 
    (horizontal and vertical) alongside latent space 2D projections.
    All colors are based on the chosen room's predicted X/Y positions.
    
    Args:
        merged_df: Unified DataFrame with predictions and latent PCA data merged
                   Expected columns: timestamp, offset, fold, room, X_pred_A, Y_pred_A, X_pred_B, Y_pred_B, X_pred_a, Y_pred_a, PC1, PC2
        rooms: List of room names (e.g., ['A', 'B', 'a'])
        boundary_points: Array of boundary points for plotting
        pos_range: Tuple (min, max) for position range (used for both X and Y)
        scaler: Optional scaler for transforming positions (default: None)
        k_folds: Number of folds for cross-validation (optional, not used currently)
        max_figures: Maximum number of figures to generate (default: 10)
        save_params: Dictionary with 'config' and 'path' for saving
    
    Returns:
        List of figure paths created
    """
    logger = get_logger(__name__)
    
    if len(rooms) < 2:
        logger.warning(f"Expected at least 2 rooms, found {len(rooms)}. Skipping.")
        return []
    
    # Verify required columns exist
    latent_cols = ['PC1', 'PC2', 'PC3'] if 'PC3' in merged_df.columns else ['PC1', 'PC2']
    if not all(col in merged_df.columns for col in latent_cols):
        logger.warning("PC columns not found in merged_df. Skipping.")
        return []
    
    # Get unique (fold, offset, room) combinations
    common_combinations = merged_df[['fold', 'offset', 'room']].drop_duplicates()
    
    if len(common_combinations) == 0:
        logger.warning("No (fold, offset, room) combinations found. Skipping.")
        return []
    
    # Limit to max_figures by grouping
    grouped = merged_df.groupby(['fold', 'offset', 'room'])
    
    figure_paths = []
    
    # Process each (fold, offset, room) combination using groupby
    for idx, ((fold, offset, chosen_room), group) in enumerate(grouped):
        if idx >= max_figures:
            break
        
        other_rooms = [r for r in rooms if r != chosen_room]
        
        logger.info(f"Generating trajectory gradient comparison figure {idx+1}: fold={fold}, offset={offset}, room={chosen_room}")
        
        # Use the group directly as room_data
        room_data = group.copy()
        
        if len(room_data) == 0:
            logger.warning(f"No data for fold={fold}, offset={offset}, room={chosen_room}. Skipping.")
            continue
        
        # Extract latent columns
        latent_subset = room_data[['timestamp'] + latent_cols].copy()
        
        # Verify that room_data has the prediction columns for all rooms
        expected_cols = [f'X_pred_{r}' for r in rooms] + [f'Y_pred_{r}' for r in rooms]
        missing_cols = [col for col in expected_cols if col not in room_data.columns]
        if missing_cols:
            logger.warning(f"Missing prediction columns in room_data for fold={fold}, offset={offset}, room={chosen_room}: {missing_cols}. "
                          f"Available columns: {[c for c in room_data.columns if 'pred' in c.lower()]}")
        
        # Prepare save_params for this specific figure
        save_config = save_params.get('config', {})
        base_path = save_params.get('path', '')
        if base_path:
            # Use base_path and add parameters to it
            base_name, base_ext = os.path.splitext(base_path)
            figure_path = f'{base_name}_fold{fold}_offset{offset}_room{chosen_room}{base_ext}'
        else:
            # Use config to get output directory
            from utils.config import get_directory
            output_dir = get_directory(save_config, 'output') if save_config else '.'
            filename = f'trajectory_gradient_comparison_fold{fold}_offset{offset}_room{chosen_room}.png'
            figure_path = os.path.join(output_dir, 'latent', filename)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(figure_path), exist_ok=True)
        
        # Call the helper function with @save_plot decorator
        figure_save_params = save_params.copy()
        figure_save_params['path'] = figure_path
        
        result = _plot_single_trajectory_gradient_comparison(
            room_data=room_data,
            latent_subset=latent_subset,
            chosen_room=chosen_room,
            other_rooms=other_rooms,
            boundary_points=boundary_points,
            pos_range=pos_range,
            scaler=scaler,
            fold=fold,
            offset=offset,
            save_params=figure_save_params
        )
        
        # Extract the saved path from the decorator's return value
        if result is not None:
            fig, saved_path = result if isinstance(result, tuple) else (result, figure_path)
            figure_paths.append(saved_path if saved_path else figure_path)
    
    return figure_paths
