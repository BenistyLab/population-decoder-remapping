# Population decoder remapping

This repository provides a standalone pipeline to **train, predict, evaluate, and compute cross-room mapping** for neural population dynamics. Using a neural network it predicts spatial positions from Neuropixels recordings in the medial entorhinal cortex (MEC) and subicular complex (SUBC).

For the main project, see [BenistyLab/population-decoder-remapping](https://github.com/BenistyLab/population-decoder-remapping).

## Key features
* **End-to-End Pipeline**: Handles model training, prediction, evaluation, and xy2xy statistical mapping.
* **Model Architecture (FR2XY)**: Uses a shared temporal encoder (LSTM + MLP) that projects to a low-dimensional shared latent space, followed by room-specific MLP decoder heads.
* **Minimal Configuration**: Driven by a single per-session YAML file.
* **Outputs**: Generates a canonical, reproducible set of artifacts.

## Prerequisites
* **Python 3.11+**
* Install dependencies:
  ```powershell
  pip install -r requirements.txt
  ```

## Data layout
For each session, create a directory matching your session name: `data/<session>/`.  
Required CSV files (exported from the preprocessing pipeline):
* `dataset.csv`
* `positions_dataset.csv`
* `spike_rate_dataset.csv`
* `clusters_dataset.csv`
* `config.yaml` (optional local config)

## Usage

1. **Configure your session**: Place a minimal YAML file at `config/sessions/config_<session>.yaml`.
2. **Run the pipeline**:
   ```powershell
   python cli/run.py --config_path .\config\sessions\config_<session>.yaml --stages train:xy2xy
   ```
3. **Re-run a stage** (ignore completion flags):
   ```powershell
   python cli/run.py --config_path .\config\sessions\config_<session>.yaml --stages xy2xy --rerun true
   ```

## Configuration (Parameters)
Most parameters are centrally defined in `config/common_config.yaml`. Your session YAML only needs to override session-specific keys:

* **`metadata.session`**: Must match your `data/<session>/` folder name.
* **`preprocessing.rooms_index`**: Dictionary mapping room labels to indices (e.g., `A: 0`, `B: 1`, `a: 0`).
* **`training.cross_validation_folds`**: Number of cross-validation folds (default 10).
* **`training.fold_offset_range`**: Number of temporal shifts applied to the test fold. Setting folds=10 and offset_range=10 generates 100 trained models per session for robust prediction.
* **`model.name`**: Use `FR2XY`.

*For the full list of parameters, default values, and templates, refer to the inline comments inside `config/common_config.yaml`.*

## Outputs

After a successful `train:xy2xy` run, artifacts are written under `output/<session>/`:

* **`configs/`** and **`checkpoints/`** directories.
* **`data_pred.csv`**: Predictions for evaluation.
* **`decoder_by_room.csv`**: Per-room decoder metrics (`room`, `r2`, `rmse`).
* **`decoder_r2_by_room.png`**: Decoder Accuracy (R²) by room; **`decoder_rmse_by_room.png`**: Decoder RMSE (cm) by room.
* **`mapping_stats.csv`**: Per room-pair affine remapping (`source_room`, `target_room`, `affinity_r2`, `rmse_cm`, `angle_deg`, `reflection`, `max_eigenvalue`, `min_eigenvalue`).
* **`nn_mapping.png`**: Cross-room mapping color-field figure (header stats from `mapping_stats.csv`).
