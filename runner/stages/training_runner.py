"""Training orchestration: checkpoints, two-step room schedule, and completion flags."""

import sys
import os
import time
import glob
import re
import logging

import torch
import yaml
import pandas as pd

from model.training import *
from model.evaluation import *
from utils.analysis import *

from utils.logger import *
from utils.config import *
from utils.model import *
from utils.data_loader import map_rooms_to_indices
from utils.analysis import get_boundary_points_from_csv
from utils.helpers import format_room_name, format_room_for_saving, get_directory, get_rooms_from_config, get_device
from model.ModelHandler import ModelHandler

config = {}  # Define an empty global config dictionary

# ============================================================================
# Training State Management
# ============================================================================

class TrainingState:
    """
    Represents the current state of training.
    
    Attributes:
        status: Current training status ('complete', 'first_step_in_progress', 
                'second_step_in_progress', 'not_started')
        checkpoints: Dictionary mapping (fold, offset) tuples to checkpoint paths
        last_room: Last room that was trained (from metadata)
        trained_rooms: List of rooms that have been trained (from metadata)
        checkpoint_name_base: Base checkpoint name
        missing_checkpoints: List of (fold, offset) tuples for missing checkpoints
    """
    def __init__(self, status, checkpoints, last_room, trained_rooms, checkpoint_name_base, missing_checkpoints):
        self.status = status
        self.checkpoints = checkpoints
        self.last_room = last_room
        self.trained_rooms = trained_rooms
        self.checkpoint_name_base = checkpoint_name_base
        self.missing_checkpoints = missing_checkpoints
    
    def is_complete(self):
        """Check if training is complete."""
        return self.status == 'complete'
    
    def is_step1_complete(self):
        """Check if Step 1 (A/B training) is complete."""
        normalized_trained = {format_room_name(r) for r in self.trained_rooms}
        first_step_rooms = {'A', 'B'}
        return normalized_trained.issubset(first_step_rooms) and len(self.missing_checkpoints) == 0
    
    def get_remaining_rooms(self, all_rooms):
        """Get list of rooms that still need to be trained."""
        normalized_all = {format_room_name(r) for r in all_rooms}
        normalized_trained = {format_room_name(r) for r in self.trained_rooms}
        remaining = normalized_all - normalized_trained
        # Return original room names (not normalized)
        return [r for r in all_rooms if format_room_name(r) in remaining]
    
    def should_skip_room(self, room, all_rooms):
        """Check if a room should be skipped (already trained)."""
        normalized_room = format_room_name(room)
        normalized_trained = {format_room_name(r) for r in self.trained_rooms}
        return normalized_room in normalized_trained
    
    def to_dict(self):
        """Convert to dictionary format (for backward compatibility)."""
        return {
            'status': self.status,
            'checkpoints': self.checkpoints,
            'last_room': self.last_room,
            'trained_rooms': self.trained_rooms,
            'checkpoint_name_base': self.checkpoint_name_base,
            'missing_checkpoints': self.missing_checkpoints
        }
    
    @classmethod
    def from_dict(cls, state_dict):
        """Create from dictionary format (for backward compatibility)."""
        return cls(
            status=state_dict.get('status', 'not_started'),
            checkpoints=state_dict.get('checkpoints', {}),
            last_room=state_dict.get('last_room', None),
            trained_rooms=state_dict.get('trained_rooms', []),
            checkpoint_name_base=state_dict.get('checkpoint_name_base', ''),
            missing_checkpoints=state_dict.get('missing_checkpoints', [])
        )


class TrainingEnvironment:
    """
    Contains all initialized components needed for training.
    
    Attributes:
        config: Configuration dictionary
        logger: Logger instance
        df_data: Loaded dataset DataFrame
        rooms: List of all rooms
        rooms_to_indices: Mapping of rooms to indices
        boundary_points: Room boundary points
        model_handler: ModelHandler instance
        model: Model instance
        device: Torch device
        start_time: Training start time
    """
    def __init__(self, config, logger, df_data, rooms, rooms_to_indices, boundary_points,
                 model_handler, model, device, start_time):
        self.config = config
        self.logger = logger
        self.df_data = df_data
        self.rooms = rooms
        self.rooms_to_indices = rooms_to_indices
        self.boundary_points = boundary_points
        self.model_handler = model_handler
        self.model = model
        self.device = device
        self.start_time = start_time


# ============================================================================
# Checkpoint Management Helper Functions
# ============================================================================

def get_training_completion_flag_path(config):
    """
    Get the path to the training completion flag file.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        str: Path to the completion flag file
    """
    checkpoint_dir = get_directory(config, 'checkpoint', long_path=False)
    checkpoint_name_base = config['model'].get('checkpoint_name', get_checkpoint_from_config(config))
    base_name = os.path.splitext(checkpoint_name_base)[0]  # Remove .pth extension
    return os.path.join(checkpoint_dir, f"{base_name}.training_complete")


def check_training_completion_flag(config):
    """
    Check if training completion flag file exists.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        bool: True if flag file exists, False otherwise
    """
    flag_path = get_training_completion_flag_path(config)
    return os.path.exists(flag_path)


def create_training_completion_flag(config):
    """
    Create training completion flag file.
    
    Args:
        config (dict): Configuration dictionary
    """
    flag_path = get_training_completion_flag_path(config)
    checkpoint_dir = os.path.dirname(flag_path)
    
    # Ensure directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create empty flag file
    with open(flag_path, 'w') as f:
        f.write("")  # Empty file, just existence matters
    
    logger = get_logger(__name__)
    logger.info(f"Created training completion flag: {os.path.basename(flag_path)}")


def delete_training_completion_flag(config):
    """
    Delete training completion flag file (used for rerun).
    
    Args:
        config (dict): Configuration dictionary
    """
    flag_path = get_training_completion_flag_path(config)
    if os.path.exists(flag_path):
        os.remove(flag_path)
        logger = get_logger(__name__)
        logger.info(f"Deleted training completion flag: {os.path.basename(flag_path)}")




def get_metadata_from_checkpoints(config, checkpoints_dict, model_handler=None):
    """
    Load metadata from a sample checkpoint (latest by fold/offset).
    
    Args:
        config: Configuration dictionary
        checkpoints_dict: Dictionary of {(fold, offset): path}, list of checkpoint paths, or list of checkpoint info dicts
        model_handler: Optional ModelHandler instance (will create one if not provided)
    
    Returns:
        dict: Metadata dictionary containing 'trained_rooms', 'last_room', 'room_complete', etc.
              Returns None if failed to load metadata
    """
    logger = get_logger(__name__)
    
    if not checkpoints_dict:
        return None
    
    # Handle different formats: dict, list of paths, or list of dicts
    checkpoint_paths = []
    if isinstance(checkpoints_dict, dict):
        checkpoint_paths = list(checkpoints_dict.values())
    elif isinstance(checkpoints_dict, list):
        for item in checkpoints_dict:
            if isinstance(item, dict):
                # Extract path from checkpoint info dict
                path = item.get('path') if isinstance(item, dict) else item
                if path:
                    checkpoint_paths.append(path)
            else:
                # Assume it's a path string
                checkpoint_paths.append(item)
    else:
        checkpoint_paths = [checkpoints_dict]
    
    # Try to find latest existing checkpoint (highest fold and offset)
    sample_checkpoint_path = None
    
    # First pass: find all existing checkpoints with their fold/offset
    checkpoint_info = []
    for checkpoint_path in checkpoint_paths:
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                # Try to extract fold and offset from filename
                filename = os.path.basename(checkpoint_path)
                fold_match = re.search(r'fold=(\d+)@(\d+)', filename)
                offset_match = re.search(r'offset_B=(\d+)', filename)
                
                fold = int(fold_match.group(1)) if fold_match else 0
                offset = int(offset_match.group(1)) if offset_match else 0
                
                checkpoint_info.append((fold, offset, checkpoint_path))
            except Exception:
                # If we can't extract, still include it but with fold=0, offset=0
                checkpoint_info.append((0, 0, checkpoint_path))
    
    if checkpoint_info:
        # Sort by fold (descending), then by offset (descending) - take the latest
        checkpoint_info.sort(key=lambda x: (x[0], x[1]), reverse=True)
        sample_checkpoint_path = checkpoint_info[0][2]
        logger.debug(f"Selected checkpoint with highest fold/offset (fold={checkpoint_info[0][0]}, offset={checkpoint_info[0][1]}): {os.path.basename(sample_checkpoint_path)}")
    
    if not sample_checkpoint_path:
        logger.debug("No existing checkpoint found to load metadata from")
        return None
    
    try:
        # Use provided model_handler or create a temporary one
        if model_handler is None:
            temp_model_handler = ModelHandler(config=config, device=get_device())
        else:
            temp_model_handler = model_handler
        
        metadata = temp_model_handler.load_checkpoint_metadata(sample_checkpoint_path)
        
        if metadata:
            trained_rooms = metadata.get('trained_rooms', [])
            last_room = metadata.get('last_room', None)
            room_complete = metadata.get('room_complete', {})
            logger.debug(f"Loaded metadata from checkpoint: trained_rooms={format_room_name(trained_rooms)}, last_room={format_room_name(last_room) if last_room else 'None'}, room_complete={room_complete}")
            return metadata
        else:
            logger.debug(f"Failed to load metadata from checkpoint: {sample_checkpoint_path}")
            return None
    except Exception as e:
        logger.warning(f"Error loading metadata from checkpoint {sample_checkpoint_path}: {e}")
        return None


def determine_training_step(trained_rooms, all_rooms):
    """
    Determine which training step we're in based on trained rooms.
    
    Args:
        trained_rooms (list): List of rooms that have been trained
        all_rooms (list): List of all rooms in the dataset
        
    Returns:
        str: 'first_step_in_progress', 'second_step_in_progress', or 'not_started'
    """
    if not trained_rooms:
        return 'not_started'
    
    # Normalize all room names for comparison
    normalized_trained = {format_room_name(r) for r in trained_rooms}
    normalized_all = {format_room_name(r) for r in all_rooms}
    
    # Check if only A and/or B are trained (first step)
    first_step_rooms = {'A', 'B'}
    if normalized_trained.issubset(first_step_rooms):
        return 'first_step_in_progress'
    
    # Otherwise, we're in second step (A+B + other rooms)
    return 'second_step_in_progress'


def scan_training_checkpoints(checkpoint_dir, base_pattern):
    """
    Scan checkpoint directory for all checkpoints matching base pattern.
    
    Room information comes from checkpoint metadata.
    
    Args:
        checkpoint_dir (str): Directory containing checkpoints
        base_pattern (str): Base pattern to match (e.g., 'ckpt_0970ced2')
        
    Returns:
        dict: {'checkpoints': dict of {(fold, offset): path}, 'files': list}
    """
    if not os.path.exists(checkpoint_dir):
        return {}
    
    # Pattern to match all checkpoints with base pattern: base_*.pth (exclude .tmp and final checkpoints)
    # Match checkpoints with fold/offset parameters
    pattern = os.path.join(checkpoint_dir, f"{base_pattern}*.pth")
    all_files = glob.glob(pattern)
    
    checkpoints = {}
    files = []
    
    for filepath in all_files:
        filename = os.path.basename(filepath)
        
        # Skip .tmp files (in-progress checkpoints)
        if filename.endswith('.tmp'):
            continue
        
        # Skip final checkpoints (no fold/offset parameters) - those are handled separately
        # Only process checkpoints with fold and offset parameters
        fold_match = re.search(r'fold=(\d+)@(\d+)', filename)
        offset_match = re.search(r'offset_B=(\d+)', filename)
        
        if fold_match and offset_match:
            fold = int(fold_match.group(1))
            offset = int(offset_match.group(1))
            checkpoints[(fold, offset)] = filepath
            files.append(filepath)
    
    return {
        'checkpoints': checkpoints,
        'files': files
    }


def detect_training_state(config):
    """
    Detect training state by checking checkpoint files and loading metadata.
    
    Checks final checkpoints first, then training checkpoints.
    Uses checkpoint metadata (trained_rooms, last_room) 
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        TrainingState: Current training state
    """
    logger = get_logger(__name__)
    
    # Get base checkpoint name
    checkpoint_name_base = config['model'].get('checkpoint_name', get_checkpoint_from_config(config))
    rooms = get_rooms_from_config(config)
    checkpoint_dir = get_directory(config, 'checkpoint', long_path=False)
    
    # Step 0: Quick check - if completion flag exists, training is complete
    if check_training_completion_flag(config):
        logger.info("Training state: complete - completion flag file exists")
        checkpoints_dict = ModelHandler.get_all_checkpoints(config)
        
        return TrainingState(
            status='complete',
            checkpoints=checkpoints_dict,
            last_room=None,
            trained_rooms=rooms,
            checkpoint_name_base=checkpoint_name_base,
            missing_checkpoints=[]
        )
    
    # Step 1: Check final checkpoints first
    all_final_checkpoints = ModelHandler.get_all_checkpoints(config)
    all_final_exist, missing_final = ModelHandler.check_checkpoints_exist(all_final_checkpoints)
    
    if not all_final_exist:
        # When any expected final checkpoint is missing, step 1 needs to rerun
        missing_final_keys = []
        for item in missing_final:
            if isinstance(item, dict):
                missing_final_keys.append((item.get('fold', 0), item.get('offset', 0)))
        logger.info(
            f"Missing Step 1 final checkpoints: {len(missing_final_keys)} fold/offset combinations. "
            "Forcing Step 1 to rerun."
        )
        metadata = get_metadata_from_checkpoints(config, all_final_checkpoints)
        trained_rooms_meta = metadata.get('trained_rooms', []) if metadata else []
        last_room_meta = metadata.get('last_room') if metadata else None
        existing_checkpoints = {
            key: path for key, path in all_final_checkpoints.items() if path and os.path.exists(path)
        }
        return TrainingState(
            status='first_step_in_progress',
            checkpoints=existing_checkpoints,
            last_room=last_room_meta,
            trained_rooms=trained_rooms_meta,
            checkpoint_name_base=checkpoint_name_base,
            missing_checkpoints=missing_final_keys
        )

    if all_final_exist:
        # Check metadata to verify all rooms are actually trained
        metadata = get_metadata_from_checkpoints(config, all_final_checkpoints)
        
        if metadata:
            trained_rooms = metadata.get('trained_rooms', [])
            last_room = metadata.get('last_room', None)
            room_complete = metadata.get('room_complete', {})
            
            if trained_rooms:
                # Normalize room names for comparison
                normalized_all_rooms = {format_room_name(r) for r in rooms}
                normalized_trained = {format_room_name(r) for r in trained_rooms}
            
            # Only return 'complete' if all rooms are trained AND marked as complete in room_complete
            # Check room_complete metadata to verify all rooms are truly complete
            all_rooms_complete = True
            if room_complete:
                for room in rooms:
                    if not room_complete.get(room, False):
                        all_rooms_complete = False
                        break
            else:
                # If room_complete doesn't exist, fall back to trained_rooms check
                all_rooms_complete = (normalized_trained == normalized_all_rooms)
            
            if normalized_trained == normalized_all_rooms and all_rooms_complete:
                logger.info(f"Training state: complete - all {len(all_final_checkpoints)} final checkpoints exist, all rooms trained, and room_complete verified")
                # Convert list of dicts to dict of (fold, offset): path
                checkpoints_dict = {}
                if isinstance(all_final_checkpoints, list):
                    for ckpt_info in all_final_checkpoints:
                        if isinstance(ckpt_info, dict):
                            key = (ckpt_info.get('fold', 0), ckpt_info.get('offset', 0))
                            checkpoints_dict[key] = ckpt_info.get('path', '')
                else:
                    checkpoints_dict = all_final_checkpoints
                
                return TrainingState(
                    status='complete',
                    checkpoints=checkpoints_dict,
                    last_room=last_room,
                    trained_rooms=trained_rooms,
                    checkpoint_name_base=checkpoint_name_base,
                    missing_checkpoints=[]
                )
            else:
                # Checkpoints exist but not all rooms trained - Step 1 is complete, ready for Step 2
                # Check if trained_rooms matches Step 1 (A and/or B) and verify room_complete
                normalized_trained = {format_room_name(r) for r in trained_rooms}
                first_step_rooms = {'A', 'B'}
                if normalized_trained.issubset(first_step_rooms):
                    # Check if Step 1 rooms are marked as complete in room_complete
                    step1_rooms_complete = True
                    if room_complete:
                        step1_rooms_complete = all(room_complete.get(room, False) for room in trained_rooms)
                    
                    # Step 1 is complete (A and/or B trained) and verified in room_complete, ready for Step 2
                    if step1_rooms_complete:
                        logger.info(f"Step 1 complete: trained_rooms={format_room_name(trained_rooms)}, room_complete verified, proceeding to Step 2")
                        # Convert list of dicts to dict of (fold, offset): path
                        checkpoints_dict = {}
                        if isinstance(all_final_checkpoints, list):
                            for ckpt_info in all_final_checkpoints:
                                if isinstance(ckpt_info, dict):
                                    key = (ckpt_info.get('fold', 0), ckpt_info.get('offset', 0))
                                    checkpoints_dict[key] = ckpt_info.get('path', '')
                        else:
                            checkpoints_dict = all_final_checkpoints
                        
                        return TrainingState(
                            status='second_step_in_progress',
                            checkpoints=checkpoints_dict,
                            last_room=last_room,
                            trained_rooms=trained_rooms,
                            checkpoint_name_base=checkpoint_name_base,
                            missing_checkpoints=[]
                        )
                    else:
                        logger.info(f"Step 1 rooms in trained_rooms but room_complete not verified. Continuing to scan...")
                        # Fall through to continue scanning
                else:
                    # Some other combination - continue to scan
                    logger.info(f"All checkpoints exist but metadata shows not all rooms trained. Trained: {format_room_name(trained_rooms)}, All: {format_room_name(rooms)}. Continuing to scan...")
        else:
            # Checkpoints exist but no metadata - continue to scan training checkpoints
            logger.info(f"All checkpoints exist but no metadata found. Continuing to scan training checkpoints...")
            # If all final checkpoints exist but no metadata, and we find no training checkpoints,
            # infer that Step 1 is complete (all checkpoints exist for current room)
            # This handles the case where checkpoints were created before metadata saving was implemented
            base_pattern = os.path.splitext(checkpoint_name_base)[0]
            training_data = scan_training_checkpoints(checkpoint_dir, base_pattern)
            training_checkpoints = training_data.get('checkpoints', {})
            if not training_checkpoints:
                # All final checkpoints exist, no training checkpoints, no metadata
                # Infer Step 1 completion from checkpoint existence
                current_room = config.get('training', {}).get('room', None)
                if current_room:
                    logger.info(f"All final checkpoints exist but no metadata. Inferring Step 1 complete for room {format_room_name(current_room)} based on checkpoint existence.")
                    # Return state indicating Step 1 is complete, which will allow progression to Step 2
                    inferred_trained_rooms = [current_room] if isinstance(current_room, str) else (current_room if isinstance(current_room, list) else [])
                    # Convert list of dicts to dict of (fold, offset): path
                    checkpoints_dict = {}
                    if isinstance(all_final_checkpoints, list):
                        for ckpt_info in all_final_checkpoints:
                            if isinstance(ckpt_info, dict):
                                key = (ckpt_info.get('fold', 0), ckpt_info.get('offset', 0))
                                checkpoints_dict[key] = ckpt_info.get('path', '')
                    else:
                        checkpoints_dict = all_final_checkpoints
                    
                    return TrainingState(
                        status='second_step_in_progress',  # Step 1 complete, ready for Step 2
                        checkpoints=checkpoints_dict,
                        last_room=current_room if isinstance(current_room, str) else (current_room[-1] if isinstance(current_room, list) and current_room else None),
                        trained_rooms=inferred_trained_rooms,
                        checkpoint_name_base=checkpoint_name_base,
                        missing_checkpoints=[]
                    )
    
    # Step 2: Scan for training checkpoints
    base_pattern = os.path.splitext(checkpoint_name_base)[0]  # Remove .pth extension
    training_data = scan_training_checkpoints(checkpoint_dir, base_pattern)
    checkpoints = training_data.get('checkpoints', {})
    
    if not checkpoints:
        logger.info("Training state: not_started - no checkpoints found")
        return TrainingState(
            status='not_started',
            checkpoints={},
            last_room=None,
            trained_rooms=[],
            checkpoint_name_base=checkpoint_name_base,
            missing_checkpoints=[]
        )
    
    # Step 3: Load metadata from a sample checkpoint to get trained_rooms and last_room
    metadata = get_metadata_from_checkpoints(config, checkpoints)
    if metadata:
        trained_rooms = metadata.get('trained_rooms', [])
        last_room = metadata.get('last_room', None)
    else:
        trained_rooms = []
        last_room = None
    
    # If no metadata, assume not started
    if not trained_rooms:
        logger.info("Training state: not_started - no metadata found in checkpoints")
        return TrainingState(
            status='not_started',
            checkpoints=checkpoints,
            last_room=None,
            trained_rooms=[],
            checkpoint_name_base=checkpoint_name_base,
            missing_checkpoints=[]
        )
    
    # Determine training step
    training_step = determine_training_step(trained_rooms, rooms)
    
    # Get expected total number of checkpoints
    k_folds = config['training'].get('cross_validation_folds', 0)
    offset_list, _ = get_fold_offset_range(config)
    
    expected_total = k_folds * len(offset_list)
    
    # Find missing checkpoints
    all_expected_keys = {(fold, offset) for fold in range(1, k_folds + 1) for offset in offset_list}
    missing_checkpoints = [key for key in all_expected_keys if key not in checkpoints]
    
    logger.info(
        f"Training state: {training_step} - {len(checkpoints)}/{expected_total} checkpoints for rooms {ModelHandler.format_rooms_string(trained_rooms)}, "
        f"last_room={format_room_name(last_room) if last_room else 'None'}"
    )
    
    return TrainingState(
        status=training_step,
        checkpoints=checkpoints,
        last_room=last_room,
        trained_rooms=trained_rooms,
        checkpoint_name_base=checkpoint_name_base,
        missing_checkpoints=missing_checkpoints
    )


def validate_state_transition(old_state, new_state):
    """
    Validate that state transition is valid.
    
    Args:
        old_state: TrainingState object
        new_state: TrainingState object
    
    Returns:
        tuple: (is_valid, error_message)
    """
    valid_transitions = {
        'not_started': ['first_step_in_progress', 'not_started'],
        'first_step_in_progress': ['second_step_in_progress', 'first_step_in_progress', 'complete'],
        'second_step_in_progress': ['second_step_in_progress', 'complete'],
        'complete': ['complete']  # Can only stay complete
    }
    
    old_status = old_state.status
    new_status = new_state.status
    
    if old_status not in valid_transitions:
        return False, f"Unknown old state: {old_status}"
    
    if new_status not in valid_transitions[old_status]:
        return False, f"Invalid transition: {old_status} -> {new_status}"
    
    return True, ""


def get_missing_checkpoints_for_state(config, state):
    """
    Get list of missing checkpoints for current state.
    
    Args:
        config: Configuration dictionary
        state: TrainingState object from detect_training_state()
    
    Returns:
        list: List of (fold, offset) tuples for missing checkpoints
    """
    if state.status == 'complete':
        return []
    
    return state.missing_checkpoints


def ensure_state_consistency(config, state):
    """
    Ensure checkpoint metadata matches filename-based state.
    
    Args:
        config: Configuration dictionary
        state: TrainingState object from detect_training_state()
    
    Returns:
        tuple: (is_consistent, error_message)
    """
    if state.status == 'not_started':
        return True, "State consistent"
    
    # If status is complete, no need to verify consistency
    if state.status == 'complete':
        return True, "State consistent (complete)"
    
    # Load a sample checkpoint to verify metadata
    checkpoints = state.checkpoints
    if not checkpoints:
        return True, "No checkpoints to verify"
    
    # Checkpoints can be a dict (from scan_training_checkpoints or get_all_checkpoints) or list (for backward compatibility)
    # For training state, it should be a dict
    if isinstance(checkpoints, list):
        # This shouldn't happen for training state, but handle it gracefully
        if not checkpoints:
            return True, "No checkpoints to verify"
        sample_path = checkpoints[0].get('path') if isinstance(checkpoints[0], dict) else checkpoints[0]
    else:
        # Checkpoints is a dict
        sample_path = list(checkpoints.values())[0]
    if not os.path.exists(sample_path):
        return True, "Sample checkpoint doesn't exist yet"
    
    try:
        checkpoint_state = torch.load(sample_path, map_location='cpu')
        metadata_rooms = checkpoint_state.get('trained_rooms', [])
        filename_rooms = state.trained_rooms
        
        # Normalize for comparison
        metadata_set = {format_room_name(r) for r in metadata_rooms}
        filename_set = {format_room_name(r) for r in filename_rooms}
        
        if metadata_set != filename_set:
            return False, f"Metadata mismatch: metadata={metadata_rooms}, filename={filename_rooms}"
        
        return True, "State consistent"
    except Exception as e:
        return False, f"Error checking consistency: {e}"


# ============================================================================
# Training Orchestration Helper Functions
# ============================================================================

def _validate_prerequisites(config):
    """
    Validate that all prerequisites for training are met.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        FileNotFoundError: If required preprocessed CSV inputs are missing
    """
    logger = get_logger(__name__)
    
    # Collect diagnostic information
    dataset_path = get_dataset_path(config, 'main')
    clusters_path = get_dataset_path(config, 'clusters')
    positions_path = get_dataset_path(config, 'positions')
    spike_rates_path = get_dataset_path(config, 'spike_rates')

    # Publication contract: data is already preprocessed. We only require concrete
    # input CSV files and do not gate on legacy preprocessing completion markers.
    missing_files = []
    for label, path in (
        ("dataset", dataset_path),
        ("clusters", clusters_path),
        ("positions", positions_path),
        ("spike_rates", spike_rates_path),
    ):
        if not os.path.exists(path):
            missing_files.append(f"  - {label}: {path}")

    if missing_files:
        error_msg = (
            "Missing required preprocessed input files:\n"
            + "\n".join(missing_files)
            + "\nPrepare CSVs under data/<session>/ (see README.md § Data layout)."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)


def _initialize_training(config):
    """
    Initialize all components needed for training.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        TrainingEnvironment: Initialized training environment
    """
    logger = setup_logger(config, __name__)
    start_time = time.time()
    
    # Ensure 'training' and 'evaluating' keys exist in config
    config.setdefault('training', {})
    config.setdefault('evaluating', {})
    
    # Load dataset
    dataset_path = get_dataset_path(config, 'main')
    try:
        df_data = pd.read_csv(dataset_path)
    except Exception as e:
        error_msg = f"Failed to load dataset from {dataset_path}: {e}"
        logger.error(error_msg)
        raise
    
    # Extract room information and setup
    rooms = get_rooms_from_config(config)
    rooms_to_indices = map_rooms_to_indices(config, df_data)
    boundary_points = get_boundary_points_from_csv(config)
    log_boundary_points(interpolate_boundary_points(boundary_points)[:, :2])
    
    # Setup device
    device = get_device()
    log_device_info(device)
    
    # Initialize model
    model_handler = ModelHandler(config=config, device=device)
    model = model_handler.get_model()
    
    # Save training config to file
    keys_to_ignore = [
        'model.templates', 'training.room',
        'evaluating.room', 'model.prev_room',
    ]
    config_path = save_config_to_file(config, keys_to_ignore=keys_to_ignore)
    config_file_name = os.path.basename(config_path)
    logger.info(f"Configuration saved to {config_file_name}")
    
    # Remove checkpoint_name from config (will be set per training step)
    config['model'].pop("checkpoint_name", None)
    
    return TrainingEnvironment(
        config=config,
        logger=logger,
        df_data=df_data,
        rooms=rooms,
        rooms_to_indices=rooms_to_indices,
        boundary_points=boundary_points,
        model_handler=model_handler,
        model=model,
        device=device,
        start_time=start_time
    )


def _handle_complete_state(config, training_state, rerun):
    """
    Handle the case when training is complete.
    
    Args:
        config: Configuration dictionary
        training_state: TrainingState object
        rerun: Whether to rerun training
        
    Returns:
        bool: True if should continue, False if should skip
    """
    logger = get_logger(__name__)
    
    if check_training_completion_flag(config):
        if rerun:
            logger.info("Training completion flag exists, but rerun is enabled. Rerunning training...")
            delete_training_completion_flag(config)
            return True
        else:
            logger.info("Training completion flag exists. Skipping training...")
            return False
    else:
        logger.info("Status is 'complete' but completion flag doesn't exist. Continuing training to finalize...")
        return True


def _train_step1(config, training_env, training_state, rerun):
    """
    Execute Step 1 training (A+B rooms).
    
    Args:
        config: Configuration dictionary
        training_env: TrainingEnvironment object
        training_state: TrainingState object
        rerun: Whether to rerun training
        
    Returns:
        TrainingState: Updated training state
    """
    logger = training_env.logger
    rooms = training_env.rooms
    model = training_env.model
    
    latent_train_room = [room for room in ['A', 'B'] if room in rooms]
    checkpoint_name = config['model'].get('checkpoint_name', get_checkpoint_from_config(config))
    
    # Set training rooms
    config['training']['room'] = latent_train_room
    config['evaluating']['room'] = latent_train_room
    
    # Get checkpoints for current state
    all_checkpoints_dict = ModelHandler.get_all_checkpoints(config, checkpoint_name=checkpoint_name)
    all_checkpoints_exist, missing_checkpoints = ModelHandler.check_checkpoints_exist(all_checkpoints_dict)
    
    # Train if needed
    if not all_checkpoints_exist or rerun:
        logger.info(f"Starting/resuming Step 1 training on rooms {format_room_name(latent_train_room)}")
        only_fold = config.get('training', {}).get('only_fold', None)
        train(config, checkpoint_name=checkpoint_name, init_model=model, plot=False,
              save_stats=False, only_fold=only_fold)

        # # Re-detect state after training
        # training_state = detect_training_state(config)
    else:
        logger.info("All checkpoints exist for Step 1. Skipping training iteration.")
        # if training_state.status == 'not_started':
        #     training_state = detect_training_state(config)
    
    return training_state


def _complete_step1(config, training_env, training_state):
    """
    Complete Step 1 and verify it's ready for Step 2.
    
    Args:
        config: Configuration dictionary
        training_env: TrainingEnvironment object
        training_state: TrainingState object
        
    Returns:
        TrainingState: Updated training state
    """
    logger = training_env.logger
    model_handler = training_env.model_handler
    rooms = training_env.rooms
    
    # Re-check state after Step 1 training
    old_state = TrainingState(
        status=training_state.status,
        checkpoints=training_state.checkpoints.copy(),
        last_room=training_state.last_room,
        trained_rooms=training_state.trained_rooms[:] if training_state.trained_rooms else [],
        checkpoint_name_base=training_state.checkpoint_name_base,
        missing_checkpoints=training_state.missing_checkpoints[:]
    )
    training_state = detect_training_state(config)

    # Validate state transition
    is_valid, transition_msg = validate_state_transition(old_state, training_state)
    if not is_valid:
        logger.warning(f"State transition validation: {transition_msg}")
    else:
        logger.debug(f"State transition validation: {transition_msg}")
    
    # Load metadata from checkpoints
    checkpoint_name = config['model'].get('checkpoint_name', get_checkpoint_from_config(config))
    all_checkpoints_dict = ModelHandler.get_all_checkpoints(config, checkpoint_name=checkpoint_name)
    metadata = get_metadata_from_checkpoints(config, all_checkpoints_dict, model_handler)
    
    if metadata:
        metadata_trained_rooms = metadata.get('trained_rooms', [])
        metadata_last_room = metadata.get('last_room', None)
        logger.info(f"Step 1 checkpoint metadata: trained_rooms={format_room_name(metadata_trained_rooms)}, last_room={format_room_name(metadata_last_room) if metadata_last_room else 'None'}")
        # Update state with metadata
        training_state.trained_rooms = metadata_trained_rooms
        training_state.last_room = metadata_last_room
    
    # Verify Step 1 is complete
    all_step1_checkpoints = all_checkpoints_dict
    all_step1_exist, missing_step1 = ModelHandler.check_checkpoints_exist(all_step1_checkpoints)
    
    if not all_step1_exist:
        missing_count = len(missing_step1)
        missing_list = get_missing_checkpoints_for_state(config, training_state)
        logger.error(f"Step 1 training incomplete. Missing {missing_count} checkpoints.")
        raise ValueError(f"Step 1 training incomplete. Missing {missing_count} checkpoints. Cannot continue to Step 2.")
    
    # Check if there are more rooms to train
    normalized_all_rooms = {format_room_name(r) for r in rooms}
    normalized_trained = {format_room_name(r) for r in training_state.trained_rooms}
    
    if normalized_trained != normalized_all_rooms:
        # More rooms to train - proceed to Step 2
        training_state.status = 'second_step_in_progress'
        logger.info(f"Proceeding to Step 2: trained_rooms={format_room_name(training_state.trained_rooms)}, remaining rooms={format_room_name([r for r in rooms if format_room_name(r) not in normalized_trained])}")
    else:
        logger.info(f"All rooms already trained according to metadata. trained_rooms={format_room_name(training_state.trained_rooms)}")
    
    return training_state


def _verify_room_training_status(config, train_room, all_checkpoints_dict, model_handler):
    """
    Verify if a room has already been trained by checking checkpoint metadata.
    RELIES ONLY ON room_complete metadata (not trained_rooms list) to avoid issues with multi-letter rooms.
    
    Args:
        config: Configuration dictionary
        train_room: Room to check
        all_checkpoints_dict: Dictionary of all checkpoints
        model_handler: ModelHandler instance
        
    Returns:
        bool: True if room should be skipped (already trained), False otherwise
    """
    logger = get_logger(__name__)
    
    # Load metadata from checkpoint
    metadata = get_metadata_from_checkpoints(config, all_checkpoints_dict, model_handler)
    if metadata:
        room_complete = metadata.get('room_complete', {})
        room_is_complete = room_complete.get(train_room, False) if room_complete else False
        if room_is_complete:
            logger.info(f"Skip training on {train_room}. Room marked as complete in room_complete metadata.")
            return True
        else:
            logger.info(f"Room {train_room} not marked as complete in room_complete. Will train.")
    else:
        logger.warning(f"Could not load metadata to verify if {train_room} is trained. Proceeding with training.")
    
    return False


def _clear_room_complete_for_room(config, room_to_clear):
    """
    Clear room_complete flag and remove room from trained_rooms for a specific room in all checkpoints.
    Also clears training_complete flag if set.
    
    Args:
        config: Configuration dictionary
        room_to_clear: Room name to clear (e.g., 'a')
    """
    logger = get_logger(__name__)
    checkpoint_name = config['model'].get('checkpoint_name', get_checkpoint_from_config(config))
    all_checkpoints_dict = ModelHandler.get_all_checkpoints(config, checkpoint_name=checkpoint_name)
    
    device = get_device()
    model_handler = ModelHandler(config=config, device=device)
    
    cleared_count = 0
    for checkpoint_path in all_checkpoints_dict.values():
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                model_handler.resume_from_last_checkpoint_safe(checkpoint_path)
                state_params = model_handler.get_model_state_params()
                room_complete = state_params.get('room_complete', {})
                trained_rooms = state_params.get('trained_rooms', [])
                
                needs_update = False
                
                # Clear room_complete flag for this room
                if room_complete.get(room_to_clear, False):
                    room_complete[room_to_clear] = False
                    state_params['room_complete'] = room_complete
                    needs_update = True
                
                # Remove room from trained_rooms if present
                if room_to_clear in trained_rooms:
                    trained_rooms = [r for r in trained_rooms if r != room_to_clear]
                    state_params['trained_rooms'] = trained_rooms
                    needs_update = True
                
                # Clear training_complete flag if set (since we're removing a room)
                if state_params.get('training_complete', False):
                    state_params['training_complete'] = False
                    needs_update = True
                
                # Update last_room if it was the room we're clearing
                if state_params.get('last_room') == room_to_clear:
                    state_params['last_room'] = None
                    needs_update = True
                
                # Reset training progress metadata to force retraining from scratch
                # This ensures the room will be retrained even if checkpoint exists
                if room_to_clear in trained_rooms or room_complete.get(room_to_clear, False):
                    state_params['epoch'] = 0
                    state_params['best_epoch'] = 0
                    state_params['best_val_loss'] = float('inf')
                    needs_update = True
                
                if needs_update:
                    model_handler.update_model_state_params(state_params, reset=False)
                    model_handler.save_checkpoint_atomic(checkpoint_path, save_as_temp=False)
                    cleared_count += 1
                    logger.debug(f"Cleared room_complete, removed from trained_rooms, and reset training progress for room {room_to_clear} in {os.path.basename(checkpoint_path)}")
            except Exception as e:
                logger.warning(f"Failed to clear room_complete for room {room_to_clear} in {checkpoint_path}: {e}")
    
    if cleared_count > 0:
        logger.info(f"Cleared room_complete flag and removed from trained_rooms for room {room_to_clear} in {cleared_count} checkpoint(s)")


def _process_step2_room(config, training_env, training_state, train_room, rerun):
    """
    Process training for a single room in Step 2.
    
    Args:
        config: Configuration dictionary
        training_env: TrainingEnvironment object
        training_state: TrainingState object
        train_room: Room to train
        rerun: Whether to rerun training
        
    Returns:
        tuple: (updated_trained_rooms, updated_last_room, should_continue)
    """
    logger = training_env.logger
    model_handler = training_env.model_handler
    model = training_env.model
    
    checkpoint_name = config['model'].get('checkpoint_name', get_checkpoint_from_config(config))
    trained_rooms = training_state.trained_rooms[:] if training_state.trained_rooms else []
    last_room = training_state.last_room
    
    # Normalize room name (needed for later checks)
    normalized_train_room = format_room_name(train_room)
    normalized_trained = {format_room_name(r) for r in trained_rooms}
    all_checkpoints_dict = ModelHandler.get_all_checkpoints(config, checkpoint_name=checkpoint_name)
    should_skip = _verify_room_training_status(config, train_room, all_checkpoints_dict, model_handler)
    
    if should_skip:
        # Still add room to trained_rooms even if skipping (to track that it was processed)
        # Use normalized comparison to avoid duplicates (e.g., "A" vs "a")
        if normalized_train_room not in normalized_trained:
            trained_rooms = trained_rooms + [train_room]
            normalized_trained.add(normalized_train_room)
        last_room = train_room
        logger.info(f"Skipping training for room {train_room} (already trained and complete according to room_complete metadata).")
        return trained_rooms, last_room, True
    
    # Set training room
    config['training']['room'] = train_room
    config['evaluating']['room'] = train_room
    logger.info(f'Continue training {config["model"]["name"]} Model on {train_room} (Step 2)')
    
    # Update trained_rooms list using normalized comparison to avoid duplicates
    if normalized_train_room not in normalized_trained:
        trained_rooms = trained_rooms + [train_room]
        normalized_trained.add(normalized_train_room)
    last_room = train_room
    
    # Recalculate checkpoints for Step 2
    original_training_room = config['training']['room']
    config['training']['room'] = trained_rooms
    all_step2_checkpoints = ModelHandler.get_all_checkpoints(config, checkpoint_name=checkpoint_name)
    config['training']['room'] = original_training_room
    all_step2_exist, missing_step2 = ModelHandler.check_checkpoints_exist(all_step2_checkpoints)
    
    # Check if all checkpoints exist AND have correct metadata (including room_complete)
    should_skip_training = all_step2_exist
    if all_step2_exist:
        # Find checkpoint with highest fold and offset (latest) to check metadata
        sample_checkpoint_path = None
        checkpoint_info = []
        for path in all_step2_checkpoints.values():
            if path and os.path.exists(path):
                try:
                    # Try to extract fold and offset from filename
                    filename = os.path.basename(path)
                    fold_match = re.search(r'fold=(\d+)@(\d+)', filename)
                    offset_match = re.search(r'offset_B=(\d+)', filename)
                    
                    fold = int(fold_match.group(1)) if fold_match else 0
                    offset = int(offset_match.group(1)) if offset_match else 0
                    
                    checkpoint_info.append((fold, offset, path))
                except Exception:
                    # If we can't extract, still include it but with fold=0, offset=0
                    checkpoint_info.append((0, 0, path))
        
        if checkpoint_info:
            # Sort by fold (descending), then by offset (descending) - take the latest
            checkpoint_info.sort(key=lambda x: (x[0], x[1]), reverse=True)
            sample_checkpoint_path = checkpoint_info[0][2]
        
        if sample_checkpoint_path:
            temp_model_handler = ModelHandler(config=config, device=training_env.device)
            metadata = temp_model_handler.load_checkpoint_metadata(sample_checkpoint_path)
            if metadata:
                room_complete_dict = metadata.get('room_complete', {})
                room_is_complete = room_complete_dict.get(train_room, False)

                if not room_is_complete:
                    logger.info(f"Checkpoints exist but room {train_room} not marked as complete in room_complete. Will train.")
                    should_skip_training = False
                else:
                    logger.info(f"Checkpoints exist and room {train_room} is marked as complete in room_complete. Will skip training.")
            else:
                logger.info(f"Checkpoints exist but no metadata found. Will train to add metadata.")
                should_skip_training = False
    
    # Update checkpoints for this room if needed
    if not should_skip_training or rerun:
        metadata_logged = False
        for fold_offset, checkpoint_path in all_step2_checkpoints.items():
            if checkpoint_path and os.path.exists(checkpoint_path):
                model_handler.resume_from_last_checkpoint_safe(checkpoint_path)
            else:
                logger.warning(f"Checkpoint not found for {fold_offset}: {checkpoint_path}. Skipping.")
                continue
            
            model_state_params = model_handler.get_model_state_params()
            checkpoint_trained_rooms = model_state_params.get('trained_rooms') or trained_rooms
            checkpoint_last_room = model_state_params.get('last_room') or last_room
            
            reset_flag = False
            normalized_checkpoint_trained = {format_room_name(r) for r in checkpoint_trained_rooms}
            
            # Check if room needs to be added
            if normalized_train_room not in normalized_checkpoint_trained:
                checkpoint_trained_rooms = checkpoint_trained_rooms + [train_room]
                checkpoint_last_room = train_room
                reset_flag = True
                model.reset_decoder_weights(train_room)
                # Recompute normalized set after adding
                normalized_checkpoint_trained = {format_room_name(r) for r in checkpoint_trained_rooms} 
            
            if not metadata_logged:
                logger.info(f"Last trained room: {format_room_name(checkpoint_last_room)}, Trained rooms: {format_room_name(checkpoint_trained_rooms)}")
                metadata_logged = True
            
            model.freeze_all_weights()
            model.unfreeze_decoder_weights(train_room)
            
            # Update checkpoint if adding new room or forcing rerun
            if reset_flag:
                updated_state_params = {
                    'epoch': 0,
                    'best_epoch': 0,
                    'best_val_loss': float('inf'),
                    'net': model_handler.get_model_state_dict(),
                    'last_room': checkpoint_last_room,
                    'trained_rooms': checkpoint_trained_rooms
                }
                model_handler.update_model_state_params(updated_state_params)
                model_handler.save_checkpoint_atomic(checkpoint_path)
    
    # Train if needed
    config['training']['room'] = train_room
    config['evaluating']['room'] = train_room
    
    if not should_skip_training or rerun:
        only_fold = config.get('training', {}).get('only_fold', None)
        train(config, checkpoint_name=checkpoint_name, init_model=model, plot=False,
              save_stats=False, only_fold=only_fold)
        
        # After training, reload trained_rooms from checkpoint metadata to ensure accuracy
        # Find a checkpoint to read metadata from
        sample_checkpoint_path = None
        for path in all_step2_checkpoints.values():
            if path and os.path.exists(path):
                sample_checkpoint_path = path
                break
        
        if sample_checkpoint_path:
            temp_model_handler = ModelHandler(config=config, device=training_env.device)
            metadata = temp_model_handler.load_checkpoint_metadata(sample_checkpoint_path)
            if metadata:
                metadata_trained_rooms = metadata.get('trained_rooms', [])
                if metadata_trained_rooms:
                    # Update trained_rooms from metadata, using normalized comparison to merge
                    normalized_metadata = {format_room_name(r) for r in metadata_trained_rooms}
                    normalized_current = {format_room_name(r) for r in trained_rooms}
                    # Add any rooms from metadata that aren't in current list (normalized)
                    for meta_room in metadata_trained_rooms:
                        if format_room_name(meta_room) not in normalized_current:
                            trained_rooms.append(meta_room)
                            normalized_current.add(format_room_name(meta_room))
                    logger.debug(f"Updated trained_rooms from checkpoint metadata: {format_room_name(trained_rooms)}")
    else:
        logger.info(f"All checkpoints exist for room {train_room}. Skipping training iteration.")
    
    return trained_rooms, last_room, True


def _train_step2(config, training_env, training_state, rerun):
    """
    Execute Step 2 training (remaining rooms).
    
    Args:
        config: Configuration dictionary
        training_env: TrainingEnvironment object
        training_state: TrainingState object
        rerun: Whether to rerun training
        
    Returns:
        TrainingState: Updated training state with all rooms trained
    """
    logger = training_env.logger
    rooms = training_env.rooms
    
    logger.info(f"Starting Step 2: Training on remaining rooms. Current trained_rooms={format_room_name(training_state.trained_rooms)}, all_rooms={format_room_name(rooms)}")
    
    trained_rooms = training_state.trained_rooms[:] if training_state.trained_rooms else []
    last_room = training_state.last_room
    
    # Get only remaining rooms that need to be trained
    remaining_rooms = training_state.get_remaining_rooms(rooms)
    
    # Filter out rooms that are in trained_rooms but not actually complete (check room_complete metadata)
    # This handles the case where training was interrupted and room was added to trained_rooms but not marked complete
    checkpoint_name = config['model'].get('checkpoint_name', get_checkpoint_from_config(config))
    all_checkpoints_dict = ModelHandler.get_all_checkpoints(config, checkpoint_name=checkpoint_name)
    metadata = get_metadata_from_checkpoints(config, all_checkpoints_dict, training_env.model_handler)
    
    if metadata:
        room_complete = metadata.get('room_complete', {})
        # Remove rooms from remaining_rooms if they're in trained_rooms but not marked as complete
        # This ensures we retrain rooms that were partially trained
        actually_complete_rooms = {r for r in trained_rooms if room_complete.get(r, False)}
        normalized_complete = {format_room_name(r) for r in actually_complete_rooms}
        # Keep only rooms that are either not in trained_rooms, or in trained_rooms but not complete
        remaining_rooms = [r for r in remaining_rooms if format_room_name(r) not in normalized_complete]
        
        # Also add rooms that are in trained_rooms but not complete (they need retraining)
        for room in trained_rooms:
            if not room_complete.get(room, False) and format_room_name(room) not in {format_room_name(r) for r in remaining_rooms}:
                remaining_rooms.append(room)
                logger.info(f"Room {room} is in trained_rooms but not marked as complete. Will retrain.")
    
    logger.info(f"Remaining rooms to train: {format_room_name(remaining_rooms)}")
    
    # Train on remaining rooms only
    for train_room in remaining_rooms:
        trained_rooms, last_room, should_continue = _process_step2_room(
            config, training_env, training_state, train_room, rerun
        )
        if not should_continue:
            break
    
    # Update training state
    training_state.trained_rooms = trained_rooms
    training_state.last_room = last_room
    
    return training_state


def _finalize_training(config, training_env, training_state):
    """
    Finalize training by marking checkpoints as complete and creating completion flag.
    
    Args:
        config: Configuration dictionary
        training_env: TrainingEnvironment object
        training_state: TrainingState object
    """
    logger = training_env.logger
    model_handler = training_env.model_handler
    
    rooms = training_env.rooms
    trained_rooms = training_state.trained_rooms
    
    # Reload trained_rooms from checkpoint metadata to ensure accuracy
    checkpoint_name = config['model'].get('checkpoint_name', get_checkpoint_from_config(config))
    all_checkpoints = ModelHandler.get_all_checkpoints(config, checkpoint_name=checkpoint_name)
    metadata = get_metadata_from_checkpoints(config, all_checkpoints, model_handler)
    
    if metadata:
        metadata_trained_rooms = metadata.get('trained_rooms', [])
        if metadata_trained_rooms:
            # Merge metadata trained_rooms with state trained_rooms using normalized comparison
            normalized_metadata = {format_room_name(r) for r in metadata_trained_rooms}
            normalized_current = {format_room_name(r) for r in trained_rooms}
            
            # Add any rooms from metadata that aren't in current list
            for meta_room in metadata_trained_rooms:
                if format_room_name(meta_room) not in normalized_current:
                    trained_rooms.append(meta_room)
                    normalized_current.add(format_room_name(meta_room))
            
            # Update training state with merged list
            training_state.trained_rooms = trained_rooms
            logger.info(f"Reloaded trained_rooms from checkpoint metadata before finalization: {format_room_name(trained_rooms)}")
    
    # Verify all rooms were trained
    normalized_all_rooms = {format_room_name(r) for r in rooms}
    normalized_trained = {format_room_name(r) for r in trained_rooms}
    
    if normalized_trained != normalized_all_rooms:
        missing_rooms = [r for r in rooms if format_room_name(r) not in normalized_trained]
        logger.error(f"Not all rooms were trained. Trained rooms: {format_room_name(trained_rooms)}, All rooms: {format_room_name(rooms)}, Missing: {format_room_name(missing_rooms)}")
        raise ValueError("Not all rooms were trained. Cannot finalize checkpoints.")
    
    # Mark all checkpoints as complete
    all_checkpoints = ModelHandler.get_all_checkpoints(config, checkpoint_name=checkpoint_name)
    all_exist, _ = ModelHandler.check_checkpoints_exist(all_checkpoints)

    if not all_exist:
        raise ValueError("Not all checkpoints exist. Cannot finalize checkpoints.")

    logger.info("Mark all checkpoints as complete.")
    for key, checkpoint_path in all_checkpoints.items():
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                model_handler.resume_from_last_checkpoint_safe(checkpoint_path)
                state_params = model_handler.get_model_state_params()
                state_params.update({'last_room': None, 'training_complete': True})
                model_handler.update_model_state_params(state_params, reset=False)
                model_handler.save_checkpoint_atomic(checkpoint_path)
            except Exception as e:
                logger.error(f"Failed to mark checkpoint: {checkpoint_path}, error: {e}")
                raise
        else:
            logger.error(f"No checkpoint found at {checkpoint_path}")
    
    # Create completion flag
    create_training_completion_flag(config)
    
def main(config):
    """
    Main training orchestrator.
    
    This function coordinates the entire training process:
    1. Validates prerequisites
    2. Initializes training environment
    3. Detects current training state
    4. Executes training based on state
    5. Finalizes training
    
    Args:
        config: Configuration dictionary
    """
    logger = setup_logger(config, __name__)
    
    # Ensure 'training' and 'evaluating' keys exist
    config.setdefault('training', {})
    config.setdefault('evaluating', {})
    
    model_name = config['model']['name']
    rerun = config.get('run', {}).get('rerun', False)
    # Handle rerun flag
    if rerun:
        delete_training_completion_flag(config)
    else:
        if check_training_completion_flag(config):
            logger.info("Training completion flag exists and rerun is False. Skipping training entirely.")
            return


    # Validate prerequisites
    _validate_prerequisites(config)
    
    # Detect training state
    training_state = detect_training_state(config)
    status = training_state.status
    
    # Validate state consistency
    is_consistent, consistency_msg = ensure_state_consistency(config, training_state)
    if not is_consistent:
        logger.warning(f"State consistency check: {consistency_msg}")
    else:
        logger.debug(f"State consistency check: {consistency_msg}")
    
    # Handle complete state
    if status == 'complete':
        if not _handle_complete_state(config, training_state, rerun):
            return
    else:
        # Log training progress
        if status == 'not_started':
            logger.info("No checkpoints found. Starting training from scratch...")
        elif status in ['first_step_in_progress', 'second_step_in_progress']:
            missing_count = len(training_state.missing_checkpoints)
            logger.info(f"Training in progress ({status}). Found {len(training_state.checkpoints)} checkpoints, missing {missing_count} checkpoints...")
    
    # Initialize training environment
    training_env = _initialize_training(config)
    logger.info(f'Train {model_name} Model')
    
    # Calculate total training intentions BEFORE starting training loops
    # Step 1: A&B together = 1 room combination
    # Step 2: All remaining rooms individually = n_rooms - 2 (excluding A and B already done in Step 1)
    # Total: 1 + (n_rooms - 2) = n_rooms - 1
    rooms = training_env.rooms
    n_rooms = len(rooms)
    k_folds = config['training'].get('cross_validation_folds', 0)
    offset_list, _ = get_fold_offset_range(config)
    fold_indices = [fold for fold in range(1, k_folds + 1)]
    only_fold = config.get('training', {}).get('only_fold', None)
    if only_fold is not None:
        fold_indices = [only_fold]
    
    # Total training room combinations: n_rooms - 1 (Step 1: A&B, Step 2: remaining rooms)
    total_room_combinations = n_rooms - 1
    # Total training intentions = room combinations × offsets × folds
    total_train_intentions = total_room_combinations * len(offset_list) * len(fold_indices)
    config['_total_train_intentions'] = total_train_intentions
    # Preserve completed_intentions across all steps
    if '_completed_train_intentions' not in config:
        config['_completed_train_intentions'] = 0
    
    # Execute training steps
    if status in ['not_started','first_step_in_progress']:
        # Train Step 1
        training_state = _train_step1(config, training_env, training_state, rerun)
        # Complete Step 1 and transition to Step 2 if needed
        training_state = _complete_step1(config, training_env, training_state)

    # Execute Step 2 if needed
    if training_state.status == 'second_step_in_progress':
        training_state = _train_step2(config, training_env, training_state, rerun)
    
    # Finalize training
    _finalize_training(config, training_env, training_state)
    
    log_completion_message(training_env.start_time)



def run_config_from_file(config, force_rerun=False):

    model_name = config['model']['name']
    run_config = config.get('run', {})
    run_config['rerun'] = force_rerun
    run_config['type'] = 'train'

    log_box_message(f'Train {model_name} Model')

    success = True
    try:
        main(config)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error in training model: {e}", exc_info=True)
        success = False
    finally:
        return  success
