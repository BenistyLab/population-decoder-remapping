"""Cross-room (xy2xy) mapping stage: affine remapping metrics and ``nn_mapping.png``."""

from __future__ import annotations

import copy
import gc
import os

import matplotlib.pyplot as plt

from . import xy2xy_runner
from utils.config import get_directory
from utils.logger import get_logger, log_box_message
from utils.visualization_xy2xy import plot_nn_mapping_figure

logger = get_logger(__name__)


def run_xy2xy_stage(config, force_rerun=False):
    """Run xy2xy analysis, write ``mapping_stats.csv`` and ``nn_mapping.png``."""
    logger.info("Running xy2xy mapping stage...")
    config.setdefault("run", {})["rerun"] = force_rerun
    config_copy = copy.deepcopy(config)
    log_box_message("XY2XY Model")
    runner_result = xy2xy_runner.main(config_copy)

    output_folder = get_directory(config, "output")
    stats_csv = os.path.join(output_folder, "mapping_stats.csv")
    if not os.path.isfile(stats_csv):
        logger.warning("mapping_stats.csv missing after xy2xy analysis.")

    mapping_data = (runner_result or {}).get("mapping_data")
    if not mapping_data:
        logger.warning("No mapping_data for nn_mapping.png; skipping.")
    else:
        md = config.get("metadata") or {}
        session = (md.get("session") or "").strip() if md.get("session") is not None else ""
        project = (runner_result or {}).get("project_name") or session
        if not project:
            logger.warning(
                "metadata.session is missing or empty; nn_mapping.png title will use a placeholder."
            )
            project = "NN mapping"

        out_png = os.path.join(output_folder, "nn_mapping.png")
        try:
            fig = plot_nn_mapping_figure(
                mapping_data,
                boundary_points=(runner_result or {}).get("boundary_points"),
                pos_range=(runner_result or {}).get("pos_range"),
                stats_csv=stats_csv,
                project=str(project),
            )
            fig.savefig(out_png, dpi=150, bbox_inches="tight", pad_inches=0.08)
            plt.close(fig)
            logger.info("Wrote %s", out_png)
        except Exception as e:
            logger.warning("Could not build nn_mapping.png: %s", e, exc_info=True)

    try:
        plt.close("all")
        gc.collect()
    except Exception as e:
        logger.warning("Cleanup after xy2xy: %s", e)
    return {"success": True}
