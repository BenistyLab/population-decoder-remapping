"""Publication pipeline: train → predict → evaluate → xy2xy only."""

from .train import run_train_stage
from .predict import run_predict_stage
from .evaluate import run_evaluate_stage
from .xy2xy import run_xy2xy_stage

STAGE_REGISTRY = {
    "train": run_train_stage,
    "predict": run_predict_stage,
    "evaluate": run_evaluate_stage,
    "xy2xy": run_xy2xy_stage,
}

STAGE_ORDER = ["train", "predict", "evaluate", "xy2xy"]

__all__ = [
    "STAGE_REGISTRY",
    "STAGE_ORDER",
    "run_train_stage",
    "run_predict_stage",
    "run_evaluate_stage",
    "run_xy2xy_stage",
]
