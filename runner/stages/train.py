"""Training stage adapter."""

from . import training_runner
from utils.logger import get_logger

logger = get_logger(__name__)


def run_train_stage(config, force_rerun=False):
    """
    Run training stage.
    
    Args:
        config (dict): Configuration dictionary.
        force_rerun (bool): Force rerun even if training is done.
        
    Returns:
        dict: Stage result dictionary.
    """
    logger.info("Running training stage...")
    
    model_config = config.get('model', {})
    n_active_cells = model_config.get('n_active_cells', model_config.get('input_dim', 0))
    min_cells = model_config.get('min_cells_threshold', 10)
    
    if n_active_cells < min_cells:
        warning_msg = (
            f"Population has {n_active_cells} active cells, which is below the minimum threshold of {min_cells}. "
            f"Training will proceed, but results may be unreliable."
        )
        logger.warning(warning_msg)

    try:
        success = training_runner.run_config_from_file(config, force_rerun=force_rerun)
        
        if success:
            logger.info("Training completed successfully.")
            return {'success': True}
        else:
            logger.error("Training failed.")
            raise RuntimeError("Training stage returned failure.")
            
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise

