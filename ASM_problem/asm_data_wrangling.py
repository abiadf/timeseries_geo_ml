import os
import sys
import torch
from typing import List, Union
sys.path.append(os.path.abspath('.')) # to run files that are away
os.environ["WANDB_SILENT"] = "true"  # Suppress WandB logs

libraries = ["torch", "numpy", "polars"]
modules   = {lib: sys.modules.get(lib) for lib in libraries}

import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt

from load_and_rename_files import LogFilesProcessor, WaferFilesProcessor
from prediction_methods import MultiOutputModelPredictor, DataPreprocessor
from asm_utils import count_missing_values_in_df, remove_constant_valued_cols
from key_params import main_folder, NUM_WAFERS, dict_of_spatial_files, dict_of_log_files, step_col_name, COMMON_ID_COLS, COMMON_ID_COLS_MOD, parquet_folder_name

log_processor = LogFilesProcessor(COMMON_ID_COLS_MOD, COMMON_ID_COLS)
device        = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))


def load_spatial_csv_and_create_targets(dict_of_spatial_files, main_folder: str, save: bool = False) -> tuple:
    """Merge wafer files, split by RC, and create y and radius dataframes"""
    processor       = WaferFilesProcessor()
    master_spatial_df = processor.load_wafer_csv_files_and_merge_to_df(dict_of_spatial_files)
    if save:
        master_spatial_df.write_parquet(f"{main_folder}/{parquet_folder_name}/master_wafer_file.parquet")
    spatial_df_dict = processor.split_master_spatial_df_by_rc(master_spatial_df, "RC", "wafer")

    y_df_dict, radius_df_dict, wide_radius_df_dict = {}, {}, {}
    for idx, wafer_df in spatial_df_dict.items():
        y_df_dict[idx], radius_df_dict[idx] = processor.split_1_wafer_df_to_y_and_radius_df(wafer_df)
        radius_df_with_idx = radius_df_dict[idx].sort("marathon_run").with_columns(
            pl.arange(0, pl.len()).over("marathon_run").alias("radius_idx"))
        wide_radius_df_dict[idx] = radius_df_with_idx.pivot(values= "Radius (mm)",
                                                            index = "marathon_run",
                                                            on    = "radius_idx",
                                                            aggregate_function = "first").sort("marathon_run")
    return master_spatial_df, spatial_df_dict, y_df_dict, wide_radius_df_dict

def load_process_and_combine_log_csv_files(dict_of_log_files, log_processor: LogFilesProcessor, unique_marathon_runs_list: list, step_col_name: str,main_folder: str,save: bool = False) -> pl.DataFrame:
    """Read all log step file CSVs, concat, then optionally save to parquet"""
    df_list = []

    for log_file in dict_of_log_files.values():
        df = log_processor.read_csv_and_lowercase_cols_names(log_file['path'])
        df = log_processor.add_marathon_and_step_cols_to_df(df, log_file['marathon'], log_file['step'], step_col_name)
        df = log_processor.remove_marathon_runs_not_found_in_wafer_df(df, unique_marathon_runs_list)
        df = log_processor.insert_step_cols_after_run(df, step_col_name)
        df = log_processor.cast_df_cols_to_float64(df)
        df = log_processor.drop_single_value_cols(df, step_col_name)
        df = log_processor.append_step_suffix_to_cols(df, step_col_name, log_file['step'])
        df_list.append(df)

    master_log_df = pl.concat(df_list, how="diagonal")
    master_log_df = log_processor.reorder_cols(master_log_df, step_col_name)
    master_log_df = master_log_df.rename({col: col.strip().lower() for col in master_log_df.columns})

    if save:
        master_log_df.write_parquet(f"{main_folder}/{parquet_folder_name}/master_log_file.parquet")
    return master_log_df

def _reorder_cols(df, step_id_col, wafer_col):
    """reorder to put 'wafer' after 'step_id' """
    if step_id_col in df.columns and wafer_col in df.columns:
        cols = df.columns.copy()
        cols.remove(wafer_col)
        insert_idx = cols.index(step_id_col) + 1
        cols       = cols[:insert_idx] + [wafer_col] + cols[insert_idx:]
        return df.select(cols)
    return df #fallback if stepid_col or wafer_col not found

def split_log_df_by_wafer_and_save_to_parquet(log_df, num_wafers, main_folder, log_processor: LogFilesProcessor, overwrite = False) -> None:
    """Split log_df into parquet files (1 for each wafer) then save to parquet, easier to handle than a dict of dataframes"""
    wafer_col   = "wafer"
    rc_col      = "rc"
    step_id_col = "step_id"
    common_cols = [col for col in log_df.columns if not col.startswith(rc_col)]

    for i in range(num_wafers):
        wafer_cols= [col for col in log_df.columns if col.startswith(f"{rc_col}{i+1}")]
        df        = log_df.select(common_cols + wafer_cols).with_columns(pl.lit(i+1).alias(wafer_col))
        df        = _reorder_cols(df, step_id_col, wafer_col)

        filepath = f"{main_folder}/{parquet_folder_name}/{wafer_col}_{i+1}_log.parquet"
        if overwrite or not os.path.exists(filepath):
            df.write_parquet(filepath)

# new functionality
def combine_wafer_parquets(num_wafers: int, main_folder, parquet_folder_name, overwrite=False):
    dfs = []
    for i in range(1, num_wafers + 1):
        filepath = f"{main_folder}/{parquet_folder_name}/wafer_{i}_log.parquet"
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        df = pl.read_parquet(filepath)
        if "wafer" not in df.columns:
            df = df.with_columns(pl.lit(i).alias("wafer"))
        dfs.append(df)

    # combined_df = pl.concat(dfs, how="vertical")
    combined_df = pl.concat(dfs, how="diagonal")

    outpath = f"{main_folder}/{parquet_folder_name}/all_wafers_log.parquet"
    if overwrite or not os.path.exists(outpath):
        combined_df.write_parquet(outpath)
    return combined_df

# remove below when not using
def dont_split_log_df_by_wafer_and_save_to_parquet(log_df, main_folder, overwrite = False) -> None:
    """DONT Split log_df into parquet files (1 for each wafer) then save to parquet, easier to handle than a dict of dataframes"""
    log_with_Wafer_col = log_df.with_columns(pl.lit(wafer_idx).alias(wafer_col))

    filepath = f"{main_folder}/{parquet_folder_name}/all_wafers_log.parquet"
    if overwrite or not os.path.exists(filepath):
        log_df.write_parquet(filepath)

# remove below when not using
def infer_wafer_from_rc_values(log_df: pl.DataFrame, rc_prefix="rc", wafer_col="wafer", overwrite = False) -> pl.DataFrame:
    # Get all rc# column groups
    rc_cols = [col for col in log_df.columns if col.startswith(rc_prefix)]

    # Map: wafer # → list of its rc columns
    from collections import defaultdict
    wafer_rc_map = defaultdict(list)
    for col in rc_cols:
        # extract wafer number from column name like rc1_xx
        wafer_num = int(col[len(rc_prefix)])
        wafer_rc_map[wafer_num].append(col)

    # For each wafer group, create a mask column: True if any rc# column > 0
    masks = []
    for wafer, cols in wafer_rc_map.items():
        mask = pl.fold(
            acc=pl.lit(False),
            function=lambda acc, x: acc | (x > 0),
            exprs=[pl.col(c) for c in cols]
        ).alias(f"wafer_{wafer}_active")
        masks.append(mask)

    # Apply the masks to infer wafer number
    log_df = log_df.with_columns(masks)

    # Combine masks into wafer number (only one active assumed)
    wafer_expr = pl.select(
        [pl.when(pl.col(f"wafer_{w}_active")).then(w).otherwise(None) for w in sorted(wafer_rc_map)]
    ).hstack().sum(axis=1).alias(wafer_col)

    log_df = log_df.with_columns(wafer_expr)

    # Drop the intermediate boolean columns
    log_df_no_bool_cols = log_df.drop([f"wafer_{w}_active" for w in wafer_rc_map])

    filepath = f"{main_folder}/{parquet_folder_name}/all_wafers_log.parquet"
    if overwrite or not os.path.exists(filepath):
        log_df_no_bool_cols.write_parquet(filepath)

    return log_df_no_bool_cols

def fast_infer_wafer(df: pl.DataFrame, num_wafers: int) -> pl.DataFrame:
    wafer_sums    = []
    wafer_indices = []

    for i in range(1, num_wafers + 1):
        cols = [col for col in df.columns if col.startswith(f"rc{i}_")]
        if not cols:
            continue
        wafer_sums.append(
            pl.sum_horizontal([pl.col(c) for c in cols]).alias(f"wafer_{i}"))
        wafer_indices.append(i)

    df = df.with_columns(wafer_sums)
    df = df.with_columns(
        pl.struct([f"wafer_{i}" for i in wafer_indices]).arg_max().alias("wafer") + 1)

    return df.drop([f"wafer_{i}" for i in wafer_indices])


def _compute_log_df_grouped_stats(log_df: pl.DataFrame, col_to_group_by: Union[str, List[str]]):
    """Group log_df by specified column(s) and compute stats (mean, std, min, max, median, skew, kurtosis) for numeric columns
    Excludes columns containing certain substrings or non-numeric types. outputs a df with # of rows = runs"""

    exclude_substrings = ['process time', '#run', 'wafer', 'step_id', 'marathon_run']

    stat_map = {"mean":   lambda col: pl.col(col).mean(),
                "std":    lambda col: pl.col(col).std(),
                "min":    lambda col: pl.col(col).min(),
                "max":    lambda col: pl.col(col).max(),
                "median": lambda col: pl.col(col).median(),
                # "skew":   lambda col: pl.col(col).skew(),
                # "kurt":   lambda col: pl.col(col).kurtosis(),
                }
    
    allowed_types= {pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt64,
                    pl.UInt32, pl.Int16, pl.UInt16, pl.Int8, pl.UInt8}
    numeric_cols = [col for col in log_df.columns
                    if not any(sub in col.lower() for sub in exclude_substrings)
                    and log_df.schema[col] in allowed_types]
    agg_exprs    = [func(col).alias(f"{col}_{stat_name}")
                    for stat_name, func in stat_map.items()
                    for col in numeric_cols]
    result_df    = log_df.group_by(col_to_group_by).agg(agg_exprs)
    return result_df

def _flatten_last_n_rows(log_df: pl.DataFrame, col_to_group_by: Union[str, List[str]], num_of_last_rows:int = 1) -> pl.DataFrame:
    """considers last N rows of each marathon_run"""
    sort_by_col      = "process time"
    group_by_col     = "marathon_run"

    if num_of_last_rows == 1:
        sorted_df = log_df.sort(sort_by_col)
        result_df = (sorted_df.group_by(col_to_group_by).tail(1)).drop([sort_by_col, "step_id", "#run"])
        return result_df
    else:
        results       = []
        marathon_runs = []
        df            = log_df.sort(sort_by_col)
        numeric_cols  = [col for col, dtype in df.schema.items() if dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]

        for marathon_run, sub_df in df.group_by(group_by_col, maintain_order=True):
            last_rows = sub_df.select(numeric_cols).tail(num_of_last_rows)
            flat      = last_rows.to_numpy().flatten()
            results.append(flat)
            marathon_runs.append(marathon_run[0])

        # create flattened column names
        col_names = []
        for i in range(num_of_last_rows):
            col_names.extend([f"{col}_last{i+1}" for col in numeric_cols])

        # create polars DataFrame with marathon_run + flattened data
        data = {group_by_col: marathon_runs}
        for i, col_name in enumerate(col_names):
            data[col_name] = [row[i] for row in results]

        return pl.DataFrame(data)

def flatten_last_n_rows_per_wafer(log_df: pl.DataFrame,group_cols: list[str],time_col: str = "process time",num_last_rows: int = 3,) -> pl.DataFrame:
    # Sort by time
    df_sorted = log_df.sort(time_col)

    # Get numeric columns only
    numeric_cols = [
        col for col, dtype in df_sorted.schema.items()
        if dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]

    results    = []
    group_keys = []
    col_names  = [f"{col}_last{i+1}" for i in range(num_last_rows) for col in numeric_cols]

    for group_vals, group_df in df_sorted.group_by(group_cols, maintain_order=True):
        tail_df = group_df.select(numeric_cols).tail(num_last_rows)
        flat    = tail_df.to_numpy().flatten()
        if len(flat) < len(col_names):  # Pad if not enough rows
            flat = np.pad(flat, (0, len(col_names) - len(flat)), constant_values=np.nan)
        results.append(flat)
        group_keys.append([*group_vals])

    # Build final DataFrame
    key_cols = list(zip(*group_keys))
    data     = {col: vals for col, vals in zip(group_cols, key_cols)}
    for i, name in enumerate(col_names):
        data[name] = [row[i] for row in results]

    return pl.DataFrame(data)



def train_models(y_df_dict, radius_wide_dict, main_folder, num_wafers, device):
    predictor    = MultiOutputModelPredictor(device)
    preprocessor = DataPreprocessor()

    marathon_run_col = "marathon_run"
    wafer_col        = "wafer"

    total_rmse_lgb, total_rmse_cat, total_rmse_rf, total_rmse_linreg, total_rmse_ridge = 0, 0, 0, 0, 0
    for wafer_idx in range(num_wafers):
        wafer_log_df             = pl.read_parquet(f"{main_folder}/{parquet_folder_name}/{wafer_col}_{wafer_idx+1}_log.parquet")
        # processed_wafer_log_df   = _compute_log_df_grouped_stats(wafer_log_df, 'marathon_run')
        processed_wafer_log_df   = _flatten_last_n_rows(wafer_log_df, marathon_run_col, num_of_last_rows=1)
        wafer_log_df2            = processed_wafer_log_df.with_columns(pl.lit(wafer_idx+1).alias(wafer_col))
        wafer_log_df2_with_radius= wafer_log_df2.join(radius_wide_dict[wafer_idx], on=marathon_run_col, how="left")
        log_df_reordered         = wafer_log_df2_with_radius.select([wafer_col] + [c for c in wafer_log_df2_with_radius.columns if c != wafer_col])

        y = y_df_dict[wafer_idx].sort(marathon_run_col).drop(marathon_run_col).to_pandas()
        X = log_df_reordered.to_pandas().drop(columns=[wafer_col, marathon_run_col], errors='ignore')
        X = preprocessor.drop_certain_cols_from_df(X, [marathon_run_col])

        X_train: pd.DataFrame
        y_train: np.ndarray
        X_val: pd.DataFrame
        y_val: np.ndarray
        X_train, y_train, X_val, y_val, y_scaler = preprocessor.scale_and_split_data(X, y)

        zero_var_cols = X_train.columns[X_train.var() == 0].tolist()
        X_train       = X_train.drop(columns=zero_var_cols)
        X_val         = X_val.drop(columns=zero_var_cols)

        print(f"====== Wafer {wafer_idx+1} ======")
        rmse_linreg, _ = predictor.predict_linear_reg(X_train, y_train, X_val, y_val)
        print(f"LinReg RMSE: {rmse_linreg:.3f}")
        total_rmse_linreg += rmse_linreg

        rmse_ridge, _ = predictor.predict_linear_reg_ridge(X_train, y_train, X_val, y_val)
        print(f"Ridge RMSE: {rmse_ridge:.3f}")
        total_rmse_ridge += rmse_ridge

        rmse_lgb, _ = predictor.predict_lightgbm(X_train, y_train, X_val, y_val)
        print(f"LGBM RMSE: {rmse_lgb:.3f}")
        total_rmse_lgb += rmse_lgb

        rmse_cat, _ = predictor.predict_catboost(X_train, y_train, X_val, y_val)
        print(f"Catboost RMSE: {rmse_cat:.3f}")
        total_rmse_cat += rmse_cat

        rmse_rf, _  = predictor.predict_randomforest(X_train, y_train, X_val, y_val)
        print(f"RF RMSE: {rmse_rf:.3f}")
        total_rmse_rf += rmse_rf

    print("~~~~~~~~~")
    print(f"Avg linreg RMSE: {total_rmse_linreg / num_wafers:.3f}")
    print(f"Avg ridge RMSE: {total_rmse_ridge / num_wafers:.3f}")
    print(f"Avg LGBM RMSE: {total_rmse_lgb / num_wafers:.3f}")
    print(f"Avg RF RMSE: {total_rmse_rf / num_wafers:.3f}")
    print(f"Avg Catboost RMSE: {total_rmse_cat / num_wafers:.3f}")


"""ML predictions"""

if __name__ == '__main__':
    master_spatial_df, spatial_df_dict, y_df_dict, radius_wide_dict = load_spatial_csv_and_create_targets(dict_of_spatial_files, main_folder, save=False)
    unique_marathon_runs_list = list(master_spatial_df["marathon_run"].unique())
    master_log_df = load_process_and_combine_log_csv_files(dict_of_log_files, log_processor, unique_marathon_runs_list, step_col_name, main_folder, save=False)
    # master_log_df = master_log_df.fill_null(pl.lit(0))
    master_log_df = remove_constant_valued_cols(master_log_df)

    # split_log_df_by_wafer_and_save_to_parquet(master_log_df, NUM_WAFERS, main_folder, log_processor, overwrite = False)
    dont_split_log_df_by_wafer_and_save_to_parquet(master_log_df, main_folder, overwrite = True)

    train_models(y_df_dict, radius_wide_dict, main_folder, NUM_WAFERS, device)


# TODO: null values in log_df_with_wafer_col, what to do with them?
# NOTE: skew and kurt in _compute_log_df_grouped_stats were giving null values
# NOTE: made result_df into lasst row of each run, but still getting same results
# NOTE: Big assumption the code makes: assumes marathons are done on the same machine (do not split data per marathon)
# NOTE: is 'RC3 signal_7' correct? i get values of 65535