"""Evaluation stage: decoder metrics, summary tables, and per-room figures."""

import os
import shutil

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from model.evaluation import evaluate_from_data_pred
from runner.stage_flags import check_stage_completed_locally
from utils.config import get_directory
from utils.helpers import format_room_name_display, save_data_to_csv
from utils.logger import get_logger

logger = get_logger(__name__)

# Default room colors when ``colors.room_colors`` is not set in config.
_DEFAULT_ROOM_COLORS_HEX = {
    "A": "#3284bd",
    "B": "#f68e42",
    "a": "#00a651",
}


def _hex_to_rgb01(hex_str: str) -> tuple:
    h = hex_str.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _room_color(room: str, config: dict) -> tuple:
    rs = str(room)
    cfg_colors = (config.get("colors") or {}).get("room_colors") or {}
    c = cfg_colors.get(rs)
    if c is None:
        c = _DEFAULT_ROOM_COLORS_HEX.get(rs, "#808080")
    if isinstance(c, (list, tuple)) and len(c) >= 3:
        return tuple(float(c[i]) for i in range(3))
    if isinstance(c, str) and c.startswith("#"):
        return _hex_to_rgb01(c)
    try:
        return mcolors.to_rgb(c)
    except ValueError:
        return (0.5, 0.5, 0.5)


def _ordered_rooms_for_decoder_plot(rooms: list[str], config: dict) -> list[str]:
    room_set = {str(r) for r in rooms}
    preferred = (config.get("xy2xy") or {}).get("room_order") or [
        "A",
        "B",
        "a",
        "b",
        "C",
        "c",
        "All",
    ]
    ordered = [r for r in preferred if r in room_set]
    for r in sorted(room_set, key=lambda x: (len(x), x)):
        if r not in ordered:
            ordered.append(r)
    return ordered


def _plot_decoder_by_room_bars(
    df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    out_path: str,
    config: dict,
    *,
    y_as_percent: bool,
) -> None:
    """Write a per-room decoder metric bar chart to ``out_path``."""

    plot_df = df.dropna(subset=["room", value_col]).copy()
    plot_df["room"] = plot_df["room"].astype(str)
    rooms_order = _ordered_rooms_for_decoder_plot(plot_df["room"].unique().tolist(), config)
    plot_df["room"] = pd.Categorical(plot_df["room"], categories=rooms_order, ordered=True)
    plot_df = plot_df.sort_values("room")

    rooms = [str(r) for r in plot_df["room"].tolist()]
    vals = pd.to_numeric(plot_df[value_col], errors="coerce").values.astype(float)
    if not rooms:
        logger.warning("No decoder-by-room rows to plot; skipping %s", out_path)
        return
    colors = [_room_color(r, config) for r in rooms]
    labels = [f"Room {format_room_name_display(r)}" for r in rooms]

    n = len(rooms)
    fig_w = max(4.0, 0.55 * max(n, 1))
    fig, ax = plt.subplots(figsize=(fig_w, 4.0))
    x = np.arange(n)
    ax.bar(
        x,
        vals,
        color=colors,
        edgecolor="black",
        linewidth=1.5,
        zorder=2,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    if y_as_percent:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(True, axis="y", color="0.8", linestyle="--", linewidth=1.0, alpha=1.0, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_evaluate_stage(config, force_rerun=False):
    """Compute decoder metrics from ``data_pred.csv`` and write summary CSVs and bar figures."""
    logger.info("Running evaluation stage...")

    if not check_stage_completed_locally(config, "predict"):
        logger.warning(
            "Prediction stage flag not found; ensure predict ran and produced data_pred.csv."
        )

    output_folder = get_directory(config, "output")
    data_pred_path = os.path.join(output_folder, "data_pred.csv")
    if not os.path.exists(data_pred_path):
        raise FileNotFoundError(
            "Evaluation requires data_pred.csv from the prediction stage. "
            f"Missing: {data_pred_path}"
        )

    config.setdefault("run", {})["rerun"] = force_rerun
    results = evaluate_from_data_pred(config, data_pred_path)
    cv_tables = results.get("cv_tables") if isinstance(results, dict) else None

    if not cv_tables or not isinstance(cv_tables, dict):
        raise RuntimeError("evaluate_from_data_pred returned no cv_tables")

    decoder_room_metrics_df = cv_tables.get("decoder_room_metrics_df")
    if decoder_room_metrics_df is None or len(decoder_room_metrics_df) == 0:
        raise RuntimeError("decoder_room_metrics_df is empty; cannot build decoder_by_room.csv")

    stats_df = decoder_room_metrics_df.copy()
    if "set" in stats_df.columns:
        stats_df = stats_df[stats_df["set"] == "test"].copy()
    if "room" in stats_df.columns:
        stats_df = stats_df[stats_df["room"].astype(str) != "All"].copy()

    rows = []
    for _, row in stats_df.iterrows():
        room = row.get("room")
        r2 = row.get("r2_pooled_mean", row.get("r2_pooled", np.nan))
        rmse = row.get("rmse_mean", row.get("rmse", np.nan))
        rec = {"room": room, "r2": r2, "rmse": rmse}
        for opt in ("fold", "offset", "n"):
            if opt in row.index:
                rec[opt] = row.get(opt)
        rows.append(rec)

    decoder_by_room = pd.DataFrame(rows)
    save_data_to_csv(
        config,
        decoder_by_room,
        output_file="decoder_by_room.csv",
        overwrite=True,
    )

    session_id = (config.get("metadata") or {}).get("session", "")
    r2_path = os.path.join(output_folder, "decoder_r2_by_room.png")
    rmse_path = os.path.join(output_folder, "decoder_rmse_by_room.png")
    _plot_decoder_by_room_bars(
        decoder_by_room,
        "r2",
        "Decoder Accuracy (R²)",
        f"{session_id}\nDecoder accuracy by room",
        r2_path,
        config,
        y_as_percent=True,
    )
    _plot_decoder_by_room_bars(
        decoder_by_room,
        "rmse",
        "Decoder RMSE (cm)",
        f"{session_id}\nDecoder RMSE by room",
        rmse_path,
        config,
        y_as_percent=False,
    )

    data_dir = get_directory(config, "data")
    icon_src = os.path.join(data_dir, "icon.png")
    icon_dst = os.path.join(output_folder, "icon.png")
    if os.path.isfile(icon_src):
        shutil.copy2(icon_src, icon_dst)
        logger.info("Copied icon.png to output folder.")
    else:
        logger.warning("No icon.png in data folder; skipping copy.")

    logger.info("Evaluation stage completed.")
    return {"success": True}