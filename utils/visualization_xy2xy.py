"""
XY2XY mapping figures: minimal color-field panels for room-pair remapping.
"""
from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.helpers import format_room_name_display
from utils.logger import get_logger
from utils.visualization import plot_boundaries

logger = get_logger(__name__)


def apply_gradient_pattern(xv, yv, gradient_type="horizontal"):
    """Apply horizontal or vertical gradient to mesh grid (normalized 0-1)."""
    if gradient_type == "horizontal":
        gradient = xv
    elif gradient_type == "vertical":
        gradient = yv
    else:
        raise ValueError("Unknown gradient type.")
    if gradient.size == 0:
        return np.zeros_like(gradient)
    return (gradient - gradient.min()) / (gradient.max() - gradient.min())


def _draw_transformation_arrows(ax, arrows, arrow_type, gradient_type):
    """Draw transformation arrows with black border and colored fill."""
    if arrow_type not in arrows:
        return

    arrow_data = arrows[arrow_type]
    if len(arrow_data) < 2:
        return

    shapes = ["full", "left"] if gradient_type == "vertical" else ["right", "full"]

    for k in range(0, len(arrow_data), 2):
        if k + 1 >= len(arrow_data):
            break

        start = arrow_data[k]
        end = arrow_data[k + 1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        arrow_idx = k // 2
        colors = ["blue", "red"]

        ax.arrow(
            start[0], start[1], dx, dy,
            head_width=0.034, head_length=0.032,
            fc='none', ec='black', shape=shapes[arrow_idx],
            linestyle='-', lw=2.4, zorder=3,
        )
        ax.arrow(
            start[0], start[1], dx, dy,
            head_width=0.03, head_length=0.03,
            fc=colors[arrow_idx], ec=colors[arrow_idx],
            shape=shapes[arrow_idx], linestyle='-', lw=2.0, zorder=4,
        )


def _setup_axis(ax, pos_range):
    """Setup axis properties for color mapping plots."""
    ax.set_xlim(pos_range)
    ax.set_ylim(pos_range[::-1])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])


def _resolve_mapping_rooms(remapping_dict) -> Tuple[Any, List[Any]]:
    source_room = None
    target_rooms: List[Any] = []
    for key in remapping_dict.keys():
        if source_room is None:
            source_room = key[0]
        if key[0] == source_room and key[1] != source_room and key[1] not in target_rooms:
            target_rooms.append(key[1])

    if source_room is None or len(target_rooms) == 0:
        source_room = "A"
        target_rooms = ["a", "B"]

    return source_room, target_rooms[:2]


def _plot_mapping_on_axis(
    ax,
    remapping_dict,
    key,
    *,
    gradient_type,
    boundary_points,
    plotting_kwargs,
    identity_ax=None,
    unique_rooms=None,
    num_rooms=None,
):
    source_room, target_room = key
    same_room = source_room == target_room
    if same_room:
        if unique_rooms is None or num_rooms is None:
            unique_rooms = sorted(
                set([k[0] for k in remapping_dict.keys()] + [k[1] for k in remapping_dict.keys()])
            )
            num_rooms = len(unique_rooms)
        next_room = unique_rooms[(unique_rooms.index(source_room) + 1) % num_rooms]
        key = (source_room, next_room)

    if key not in remapping_dict:
        return

    point_size = plotting_kwargs.get('point_size', 1.0)
    point_alpha = plotting_kwargs.get('point_alpha', 1.0)
    boundary_color = plotting_kwargs.get('boundary_color', 'black')
    boundary_alpha = plotting_kwargs.get('boundary_alpha', 1.0)
    boundary_linewidth = plotting_kwargs.get('boundary_linewidth', 3)
    pos_range = plotting_kwargs.get('pos_range', None)

    source_xy_canon = remapping_dict[key]["source_xy"]
    target_xy_canon = remapping_dict[key]["target_xy"]
    com_source = remapping_dict[key].get('com_source')
    com_target = remapping_dict[key].get('com_target')
    source_xy = source_xy_canon + np.array(com_source) if com_source is not None else source_xy_canon
    target_xy = target_xy_canon + np.array(com_target) if com_target is not None else target_xy_canon

    cmap = cm.get_cmap('jet')
    colors = apply_gradient_pattern(source_xy[:, 0], source_xy[:, 1], gradient_type=gradient_type)
    colors_rgb = cmap(colors)

    if not same_room:
        sns.scatterplot(
            x=target_xy[:, 0], y=target_xy[:, 1], color=colors_rgb,
            s=point_size, alpha=point_alpha, edgecolor='none', ax=ax,
        )
        if identity_ax is not None:
            sns.scatterplot(
                x=source_xy[:, 0], y=source_xy[:, 1], color=colors_rgb,
                s=point_size, alpha=point_alpha, edgecolor='none', ax=identity_ax,
            )

    if boundary_points is not None:
        plot_boundaries(
            boundary_points, ax,
            color=boundary_color, alpha=boundary_alpha, linewidth=boundary_linewidth,
        )

    arrow_key = key if same_room else (source_room, target_room)
    if arrow_key in remapping_dict and "arrows" in remapping_dict[arrow_key]:
        arrows = remapping_dict[arrow_key]["arrows"]
        arrow_type = 'prev' if same_room else 'new'
        _draw_transformation_arrows(ax, arrows, arrow_type, gradient_type)

    _setup_axis(ax, pos_range)


def _draw_minimal_mapping_row(
    ax_left,
    ax_right,
    remapping_dict,
    gradient_type,
    boundary_points,
    plotting_kwargs,
    *,
    source_room,
    target_rooms,
):
    unique_rooms = sorted(
        set([key[0] for key in remapping_dict.keys()] + [key[1] for key in remapping_dict.keys()])
    )
    num_rooms = len(unique_rooms)

    identity_key = (source_room, source_room)
    _plot_mapping_on_axis(
        ax_left, remapping_dict, identity_key,
        gradient_type=gradient_type,
        boundary_points=boundary_points,
        plotting_kwargs=plotting_kwargs,
        unique_rooms=unique_rooms,
        num_rooms=num_rooms,
    )

    for target_room in target_rooms:
        mapping_key = (source_room, target_room)
        if mapping_key in remapping_dict:
            _plot_mapping_on_axis(
                ax_right, remapping_dict, mapping_key,
                gradient_type=gradient_type,
                boundary_points=boundary_points,
                plotting_kwargs=plotting_kwargs,
                identity_ax=ax_left,
                unique_rooms=unique_rooms,
                num_rooms=num_rooms,
            )

    ax_left.set_title("")
    ax_right.set_title("")

    for ax in (ax_left, ax_right):
        if ax.get_legend() is not None:
            ax.legend_.remove()
        for coll in ax.collections:
            coll.set_rasterized(True)


def plot_nn_mapping_figure(
    remapping_dict,
    *,
    boundary_points=None,
    pos_range=None,
    project: str = "",
    figure_title: str = "Cross-room mapping",
    stats_csv: Optional[str] = None,
):
    """Build the publication ``nn_mapping.png`` figure (caller saves the returned figure)."""
    if not remapping_dict:
        raise ValueError("remapping_dict is empty")

    stat_lines: List[str] = []
    if stats_csv and os.path.isfile(stats_csv):
        try:
            df = pd.read_csv(stats_csv)
            for _, row in df.iterrows():
                src_raw = row.get("source_room")
                tgt_raw = row.get("target_room")
                src_label = format_room_name_display(str(src_raw)) if pd.notna(src_raw) else "?"
                tgt_label = format_room_name_display(str(tgt_raw)) if pd.notna(tgt_raw) else "?"
                line = f"{src_label} → {tgt_label}"
                r2 = row.get("affinity_r2")
                if r2 is not None and pd.notna(r2) and np.isfinite(float(r2)):
                    line += f" | Affinity = {float(r2):03.0%}"
                angle_deg = row.get("angle_deg")
                refl = row.get("reflection")
                if pd.isna(refl):
                    refl = None
                elif isinstance(refl, str):
                    refl = refl.strip().lower() in ("true", "1", "yes")
                else:
                    refl = bool(refl)
                if angle_deg is not None and pd.notna(angle_deg) and np.isfinite(float(angle_deg)):
                    ang = float(angle_deg)
                    if refl is not None:
                        sub = "ref" if refl else "rot"
                        line += f" | θ$_{{{sub}}}$ = {ang:+06.1f}°"
                    else:
                        line += f" | θ = {ang:+06.1f}°"
                smin = row.get("min_eigenvalue")
                smax = row.get("max_eigenvalue")
                if (
                    smin is not None
                    and smax is not None
                    and pd.notna(smin)
                    and pd.notna(smax)
                    and np.isfinite(float(smin))
                    and np.isfinite(float(smax))
                ):
                    line += f" | λ = [{float(smin):.2f},{float(smax):.2f}]"
                stat_lines.append(line)
        except Exception as e:
            logger.warning("Could not read mapping stats for nn_mapping header: %s", e)

    plotting_kwargs = {
        "point_size": 0.5,
        "boundary_linewidth": 3,
        "figsize": (7, 2.5),
    }
    if pos_range is not None:
        plotting_kwargs["pos_range"] = pos_range

    source_room, target_rooms = _resolve_mapping_rooms(remapping_dict)
    gradient_types = ("horizontal", "vertical")
    row_w, row_h = plotting_kwargs["figsize"]
    fig, axes = plt.subplots(
        len(gradient_types), 2,
        figsize=(row_w, row_h * len(gradient_types)),
        squeeze=False,
    )

    for row_idx, gradient_type in enumerate(gradient_types):
        _draw_minimal_mapping_row(
            axes[row_idx, 0],
            axes[row_idx, 1],
            remapping_dict,
            gradient_type,
            boundary_points,
            plotting_kwargs,
            source_room=source_room,
            target_rooms=target_rooms,
        )

    project_line = (project or "").strip() or "NN mapping"
    fig_title = (figure_title or "Cross-room mapping").strip()
    suptitle = f"{project_line}\n{fig_title}"
    n_stat = len(stat_lines)
    top_reserved = 0.11 + (0.034 * n_stat if n_stat else 0.0) + 0.04

    fig.subplots_adjust(left=0.11, right=0.89, top=1.0 - top_reserved, bottom=0.02, wspace=0.12, hspace=0.12)
    fig.suptitle(suptitle, fontsize=12, y=0.995, va="top")
    if stat_lines:
        fig.text(
            0.5,
            1.0 - 0.11 - 0.01,
            "\n".join(stat_lines),
            fontsize=11,
            family="monospace",
            ha="center",
            va="top",
            linespacing=1.15,
            transform=fig.transFigure,
        )

    return fig
