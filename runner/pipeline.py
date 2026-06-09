"""Pipeline orchestrator (publication build: local filesystem artifacts + logs only)."""

from __future__ import annotations

import copy
import os
import time
from typing import Any, Dict, List, Optional

from runner.stages import STAGE_REGISTRY, STAGE_ORDER
from runner.wrapper import load_and_prepare_config, prepare_config
from utils.config import (
    load_config,
    save_config_to_file,
)
from utils.logger import get_logger, log_welcome_message_from_config, setup_logger
from runner.stage_flags import (
    check_stage_completed_locally,
    clear_stage_flag,
    mark_stage_completed_locally,
    mark_stage_failed_locally,
)

logger = get_logger(__name__)


def parse_stage_selection(
    stages_str: str, start: Optional[int] = None, end: Optional[int] = None
) -> List[str]:
    if stages_str:
        stages: List[str] = []
        for part in stages_str.split(","):
            part = part.strip()
            if ":" in part:
                start_stage, end_stage = part.split(":", 1)
                start_stage = start_stage.strip()
                end_stage = end_stage.strip()
                if start_stage not in STAGE_ORDER:
                    raise ValueError(f"Unknown start stage: {start_stage}")
                if end_stage not in STAGE_ORDER:
                    raise ValueError(f"Unknown end stage: {end_stage}")
                start_idx = STAGE_ORDER.index(start_stage)
                end_idx = STAGE_ORDER.index(end_stage)
                if start_idx > end_idx:
                    raise ValueError(
                        f"Start stage '{start_stage}' comes after end stage '{end_stage}'"
                    )
                stages.extend(STAGE_ORDER[start_idx : end_idx + 1])
            else:
                if part not in STAGE_ORDER and part not in STAGE_REGISTRY:
                    raise ValueError(f"Unknown stage: {part}")
                stages.append(part)
        seen = set()
        result: List[str] = []
        for stage in stages:
            if stage not in seen:
                seen.add(stage)
                result.append(stage)
        return result
    if start is not None or end is not None:
        start_idx = start if start is not None else 0
        end_idx = end if end is not None else len(STAGE_ORDER)
        return STAGE_ORDER[start_idx:end_idx]
    return STAGE_ORDER.copy()


def _run_pipeline_for_config(
    config: Dict[str, Any],
    stages: List[str],
    force_rerun: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    config_path = config.get("_output_config_path") or config.get("_config_path")
    params = config.get("grid_params", config.get("params", {}))
    log_welcome_message_from_config(config, logger, params, "Population decoder (publication)")

    results: Dict[str, Any] = {
        "config_path": config_path,
        "stages": stages,
        "results": {},
        "success": True,
    }
    for stage_name in stages:
        if stage_name not in STAGE_REGISTRY:
            logger.error("Unknown stage: %s", stage_name)
            results["success"] = False
            continue

        stage_func = STAGE_REGISTRY[stage_name]
        stage_start_time = time.time()
        logger.info("=== Starting stage: %s ===", stage_name)

        should_skip_completion_check = force_rerun
        if not should_skip_completion_check:
            if check_stage_completed_locally(config, stage_name):
                logger.info(
                    "Stage '%s' already completed locally (flag file). Skipping.",
                    stage_name,
                )
                results["results"][stage_name] = {
                    "skipped": True,
                    "reason": "local_completed",
                }
                continue

        if force_rerun:
            clear_stage_flag(config, stage_name, flag_type="completed")
            clear_stage_flag(config, stage_name, flag_type="failed")
            config.setdefault("run", {})["rerun"] = True

        if force_rerun and "run" in config:
            config["run"].pop("rerun", None)

        try:
            stage_config = copy.deepcopy(config)
            stage_config["_config_path"] = config_path
            stage_result = stage_func(stage_config, force_rerun=force_rerun)

            mark_stage_completed_locally(config, stage_name)
            stage_duration = time.time() - stage_start_time
            logger.info(
                "=== Stage '%s' completed in %.2fs ===", stage_name, stage_duration
            )
            results["results"][stage_name] = {
                "success": True,
                "duration": stage_duration,
                "result": stage_result,
            }
            if isinstance(stage_result, dict) and "config_path" in stage_result:
                config_path = stage_result["config_path"]
                logger.info("Updating config path to: %s", config_path)
                config = load_and_prepare_config(config_path, force_rerun=force_rerun)
        except Exception as e:
            logger.error("Error in stage '%s': %s", stage_name, e, exc_info=True)
            mark_stage_failed_locally(config, stage_name, error_message=str(e))
            results["results"][stage_name] = {"success": False, "error": str(e)}
            results["success"] = False
            raise
    return results


def run_pipeline(
    config_path: str,
    stages: Optional[List[str]] = None,
    stages_str: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    force_rerun: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    if stages is None:
        stages = parse_stage_selection(stages_str, start, end)
    logger.info("Pipeline stages to execute: %s", ", ".join(stages))
    if dry_run:
        logger.info("DRY RUN: Would execute:")
        for stage in stages:
            logger.info("  - %s", stage)
        return {"dry_run": True, "stages": stages}

    base_config = load_config(config_path)
    base_config["_config_path"] = config_path
    setup_logger(base_config, __name__)

    seed = base_config.setdefault("seed", 0)
    logger.info("Global random seed set to %s", seed)

    keys_to_ignore = [
        "model.templates",
        "training.room",
        "evaluating.room",
        "model.prev_room",
    ]

    logger.info("Preparing config for single run.")
    config = prepare_config(base_config, force_rerun)
    try:
        saved_path = save_config_to_file(config, keys_to_ignore=keys_to_ignore)
        logger.info("Configuration saved to %s", os.path.basename(saved_path))
        config["_output_config_path"] = saved_path
        return _run_pipeline_for_config(config, stages, force_rerun, dry_run)
    except Exception as e:
        logger.error("Error in pipeline run: %s", e, exc_info=True)
        return {"success": False, "error": str(e), "results": {}}
