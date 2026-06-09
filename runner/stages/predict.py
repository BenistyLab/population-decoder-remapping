"""Prediction stage adapter."""

import os
import pandas as pd
from model.evaluation import predict_all_test_rooms
from utils.config import get_directory, get_target_columns_from_config, assign_room_column
from utils.helpers import get_prediction_columns, save_data_to_csv
from utils.logger import get_logger

logger = get_logger(__name__)


def run_predict_stage(config, force_rerun=False):
    """
    Run prediction stage - generates predictions for all rooms per timestamp.
    
    This stage:
    - Generates predictions for ALL rooms (A, B, a) for each timestamp
    - Creates data_pred.csv with format: timestamp, offset, fold, room, X, Y, X_pred_A, Y_pred_A, X_pred_B, Y_pred_B, ...
    - Key columns are timestamp + offset
    - Does not run evaluation metrics (see evaluate stage).
    """
    logger.info("Running prediction stage...")
    
    output_folder = get_directory(config, 'output')
    file_path = os.path.join(output_folder, 'data_pred.csv')
    rerun = config.get('run', {}).get('rerun', False) or force_rerun
    
    if not rerun and os.path.exists(file_path):
        logger.warning(f"File {file_path} already exists. Skipping to avoid overwriting.")
        return {'success': True}

    config.setdefault('run', {})['rerun'] = force_rerun
    
    df_all_results = predict_all_test_rooms(
        config=config,
        init_model=None
    )
    
    if df_all_results is None or len(df_all_results) == 0:
        logger.error("No results returned from predict_all_test_rooms. Cannot generate predictions.")
        raise ValueError("predict_all_test_rooms did not return any results. Cannot proceed with prediction generation.")
    
    if 'set' in df_all_results.columns:
        test_df = df_all_results[df_all_results['set'] == 'test'].copy()
    else:
        test_df = df_all_results.copy()
        
    target_columns = get_target_columns_from_config(config)
    pred_target_columns_standard = get_prediction_columns(target_columns)

    if 'offset' not in test_df.columns:
        test_df['offset'] = 0
    if 'fold' not in test_df.columns:
        test_df['fold'] = 0

    id_vars = ['timestamp', 'offset', 'fold', 'actual_room', 'predicted_room'] + target_columns
    value_vars = pred_target_columns_standard
    test_melted = test_df.melt(
        id_vars=[col for col in id_vars if col in test_df.columns],
        value_vars=value_vars,
        var_name='pred_col',
        value_name='pred_value'
    )
    
    test_melted['pivot_col'] = (
        test_melted['pred_col'].astype(str) + '_' + 
        test_melted['predicted_room'].astype(str)
    )
    
    pivot_index = ['timestamp', 'offset', 'fold']
    pivot_df = test_melted.pivot_table(
        index=[col for col in pivot_index if col in test_melted.columns],
        columns='pivot_col',
        values='pred_value',
        aggfunc='last'
    )
    
    pivot_df = pivot_df.reset_index()

    groupby_cols = ['timestamp', 'offset', 'fold']
    target_df = test_df.groupby([col for col in groupby_cols if col in test_df.columns])[target_columns].first().reset_index()
    
    merge_cols = [col for col in groupby_cols if col in target_df.columns and col in pivot_df.columns]
    df_data = pd.merge(target_df, pivot_df, on=merge_cols, how='outer')
    df_data = assign_room_column(df_data, config, room_column='room')

    sort_cols = [col for col in groupby_cols if col in df_data.columns]
    df_data = df_data.sort_values(sort_cols).reset_index(drop=True)

    save_data_to_csv(config, df_data, output_file='data_pred.csv', overwrite=True)
    
    logger.info("Prediction stage completed successfully.")
    return {'success': True}

