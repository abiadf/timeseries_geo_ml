import pandas as pd
import polars as pl
import numpy as np
from typing import Tuple, Union, List, Sequence

class LogAndSpatialProcessor:
    """Class dealing with processing log AND spatial files. Processes such that the whole data is in 1 df"""

    @staticmethod
    def explode_log_df_rows_by_wafer(log_df: pl.DataFrame, num_wafers: int) -> pl.DataFrame:
        """The original df had many wafers processed per row. This function expands each row to 4 rows, 1 row dedicated to each wafer,
        keeping common columns unchanged and zeroing out rc# columns not belonging to the current wafer. 
        Adds a 'wafer' column to identify the active wafer per row"""
        wafer_col       = "wafer"
        step_id_col     = "step_id"
        marathon_run_col= "marathon_run"
        process_time_col= "process time"
        rc_prefix       = "rc"

        rc_cols     = [c for c in log_df.columns if c.startswith(rc_prefix)]
        common_cols = [c for c in log_df.columns if not c.startswith(rc_prefix)]
        wafer_rows  = []

        for i in range(1, num_wafers + 1):
            wafer_prefix = f"{rc_prefix}{i}"
            zero_exprs   = []
            for c in rc_cols:
                if c.startswith(wafer_prefix):
                    zero_exprs.append(pl.col(c)) # keep original
                else:
                    dtype    = log_df.schema[c]
                    zero_val = 0.0 if dtype == pl.Float64 else 0
                    zero_exprs.append(pl.lit(zero_val).cast(dtype).alias(c))
            wafer_df = log_df.select([pl.col(c) for c in common_cols] + zero_exprs)
            wafer_df = wafer_df.with_columns(pl.lit(i).alias(wafer_col))

            # insert 'wafer' after 'step_id'
            if step_id_col in wafer_df.columns:
                cols = wafer_df.columns
                if wafer_col in cols:
                    cols.remove(wafer_col) # avoid duplicate
                idx      = cols.index(step_id_col)
                wafer_df = wafer_df.select(cols[:idx + 1] + [wafer_col] + cols[idx + 1:])
            wafer_rows.append(wafer_df)
        return pl.concat(wafer_rows, how = "vertical").sort([process_time_col, marathon_run_col, wafer_col])

    @staticmethod
    def load_spatial_csv_files_to_1_df(dict_of_spatial_files):
        """Load and vertically concat every spatial csv to a combined df"""
        rc_prefix    = "RC"
        run_col      = "#Run"
        marathon_col = "marathon"
        marathon_run_col = "marathon_run"
        wafer_col    = "wafer"

        all_dfs = []
        for file_info in dict_of_spatial_files.values():
            path       = file_info["path"]
            marathon   = file_info[marathon_col]
            spatial_df = pl.read_csv(path)
            spatial_df = (spatial_df.with_columns([
                    (pl.lit(str(marathon)) + "_" + pl.col(run_col).cast(pl.Utf8)).alias(marathon_run_col),]).rename({rc_prefix: wafer_col}))
            all_dfs.append(spatial_df)
        master_spatial_df = pl.concat(all_dfs, how = "vertical")
        
        unique_marathon_runs_list = list(master_spatial_df[marathon_run_col].unique())
        return master_spatial_df, unique_marathon_runs_list

    @staticmethod
    def create_target_df_from_spatial_df(master_spatial_df, main_folder: str, save: bool = False) -> tuple:
        """Given the combined spatial df, split it to 1 radius_df (that we join to X), and 1 y_df
        wide_radius_df: 1 row per (marathon_run, wafer) combo, column names = "Site ID" values"""
        wafer_col        = "wafer"
        marathon_run_col = "marathon_run"
        radius_col       = "Radius (mm)"
        site_id_col      = "Site #"
        spatial_prop_col = "Spatial property (nm)"

        wide_radius_df = (master_spatial_df
            .select([marathon_run_col, wafer_col, site_id_col, radius_col])
            .pivot(values  = radius_col,
                index   = [marathon_run_col, wafer_col],
                columns = site_id_col,
                aggregate_function = "first")
            .sort([marathon_run_col, wafer_col]))

        y_df = (master_spatial_df
            .select([marathon_run_col, wafer_col, site_id_col, spatial_prop_col])
            .pivot(values  = spatial_prop_col,
                index   = [marathon_run_col, wafer_col],
                columns = site_id_col,
                aggregate_function = "first")
            .sort([marathon_run_col, wafer_col]))

        # if save:
        #     master_spatial_df.write_parquet(f"{main_folder}/master_spatial_df.parquet")
        return wide_radius_df, y_df

    # to remove, adding the radii as cols in X does nothing
    @staticmethod
    def join_radius_df_to_exploded_log_df(master_log_df_exploded, wide_radius_df):
        """joining radius_df and exploded_log_df = X (used for predictions)"""
        return master_log_df_exploded.join(wide_radius_df,
                                        on  = ["marathon_run", "wafer"],
                                        how = "left")

    @staticmethod
    def expand_y_df_to_match_size_of_log_df(master_log_df_exploded, y_df):
        """This gives us y"""
        identifiers_list = ["marathon_run", "wafer"]
        # Step 1: ensure y_df has just the keys and values
        y_lookup = y_df.clone()

        # Step 2: select matching keys from log df in order
        keys_df = master_log_df_exploded.select(identifiers_list)

        # Step 3: join to align rows in y_df_expanded
        y_df_expanded = keys_df.join(y_lookup, on = identifiers_list, how = "left")
        return y_df_expanded

    @staticmethod
    def downsample_df_rows(log_df: pl.DataFrame, subsampling_factor: int) -> pl.DataFrame:
        """Downsamples the # of rows (grouped by marathon_run, wafer) by a factor. Larger factor = smaller resulting dataset
        NOTE: Grouping by wafer ensures we retain data from all wafers; without it, some wafers might be entirely excluded"""

        marathon_run_col = "marathon_run"
        wafer_col        = "wafer"
        time_col         = "process time"
        subsampled_parts = []

        for (_, _), group_df in log_df.group_by([marathon_run_col, wafer_col]):
            group_df = group_df.sort(time_col) # preserve time order within group
            subsampled_parts.append(group_df[::subsampling_factor])
        
        return pl.concat(subsampled_parts).sort(time_col)

    @staticmethod
    def keep_top_features_by_importance(X_train_clean: pd.DataFrame, X_val_clean: pd.DataFrame, importances: np.ndarray, top_features_fraction: float):
        """Select the top fraction of features from the dataframe, based on their mean importance scores (regarding predictions)
        Note that importances comes from something like so:
        importances  = np.array([est.get_feature_importance() for est in multi_model.estimators_])"""
        mean_importance        = importances.mean(axis=0)
        top_indices_sorted     = mean_importance.argsort()[::-1]
        feature_names          = X_train_clean.columns.to_list()
        features_by_importance = [feature_names[i] for i in top_indices_sorted]
        N_top_features         = int(top_features_fraction * len(features_by_importance))
        columns_to_keep        = features_by_importance[:N_top_features]
        X_train_clean_light    = X_train_clean[columns_to_keep]
        X_val_clean_light      = X_val_clean[columns_to_keep]
        return X_train_clean_light, X_val_clean_light


class LogFilesProcessor:
    """Class dealing with processing log files"""
    def __init__(self, common_id_cols_mod, common_id_cols, parquet_folder_name):
        self.COMMON_ID_COLS_MOD  = common_id_cols_mod
        self.COMMON_ID_COLS      = common_id_cols
        self.PARQUET_FOLDER_NAME = parquet_folder_name

    def read_csv_and_lowercase_cols_names(self, file_path: str) -> pl.DataFrame:
        """Read CSV into Polars DataFrame and title-case column names after stripping spaces
        NOTE: a known polars issue that it cant use the 'decimal' parameter in read_csv, so we load into pandas first"""
        pdf = pd.read_csv(file_path, decimal='.')
        df  = pl.from_pandas(pdf)
        df  = df.rename({c: c.strip().title() for c in df.columns})
        return df

    def add_marathon_and_step_cols_to_df(self, df: pl.DataFrame, marathon: Union[str, int], step_id: int, step_col_name: str) -> pl.DataFrame:
        """Add marathon, step_id columns, and create 'marathon_run' by combining marathon and '#Run'."""
        marathon_col     = 'marathon'
        run_col          = '#Run'
        marathon_run_col = 'marathon_run'
        df = df.with_columns([pl.lit(marathon).alias(marathon_col),
                              pl.lit(step_id).alias(step_col_name)])
        df = df.with_columns([
            (pl.col(marathon_col).cast(pl.Utf8) + "_" + pl.col(run_col).cast(pl.Utf8)).alias(marathon_run_col)])
        return df

    def split_log_df_by_marathon_run_existence(self, df: pl.DataFrame, unique_marathon_runs_list: list, keep_existing_runs: bool = True) -> pl.DataFrame:
        """Filter log df based on whether marathon_run is in the list (keep=True) or not (keep=False)
            - kept_df: rows where 'marathon_run' is in the list
            - dropped_df: rows where 'marathon_run' is NOT in the list"""
        condition = pl.col("marathon_run").is_in(unique_marathon_runs_list)
        return df.filter(condition if keep_existing_runs else ~condition)

    def insert_step_cols_after_run(self, df: pl.DataFrame, step_col_name: str) -> pl.DataFrame:
        """Reorder columns to insert step_col_name and 'marathon_run' after '#Run'."""
        run_col          = '#Run'
        marathon_run_col = 'marathon_run'

        cols       = df.columns
        insert_idx = cols.index(run_col) + 1
        cols.remove(step_col_name)
        cols.remove(marathon_run_col)
        cols.insert(insert_idx, step_col_name)
        cols.insert(insert_idx + 1, marathon_run_col)
        df = df.select(cols)
        return df

    def cast_df_cols_to_float64(self, df: pl.DataFrame) -> pl.DataFrame:
        """Cast integer and float columns to Float64 type."""
        df = df.with_columns([pl.col(col).cast(pl.Float64)
                              for col in df.columns
                              if df[col].dtype in [pl.Int64, pl.Float32, pl.Int32]])
        return df

    def drop_single_value_cols(self, df: pl.DataFrame, step_col_name: str) -> pl.DataFrame:
        """Drop columns with a single unique value, excluding common ID columns and step_col_name."""
        single_val_cols = [col for col in df.columns
                           if col not in self.COMMON_ID_COLS_MOD + [step_col_name]
                        #    if df[col].dtype.is_numeric()
                           and df[col].n_unique() == 1]
        df = df.drop(single_val_cols)
        return df

    def append_step_suffix_to_cols(self, df: pl.DataFrame, step_col_name: str, step_id: int) -> pl.DataFrame:
        """Rename non-ID columns by appending '_step{step_id}' suffix"""
        append_step_suffix_to_cols = {
            col: f"{col}_step{step_id}"
            for col in df.columns
            if col not in self.COMMON_ID_COLS_MOD + self.COMMON_ID_COLS + [step_col_name]}
        df = df.rename(append_step_suffix_to_cols)
        return df

    def reorder_cols(self, master_log_df: pl.DataFrame, step_col_name: str) -> pl.DataFrame:
        """Reorder columns to have common ID columns, step_col_name first, then others; warn if columns missing."""
        desired_order = self.COMMON_ID_COLS_MOD + [step_col_name] + [
            col for col in master_log_df.columns if col not in self.COMMON_ID_COLS_MOD + [step_col_name]]
        missing = [col for col in desired_order if col not in master_log_df.columns]
        if missing:
            print("Missing columns before select:", missing)
        master_log_df = master_log_df.select(desired_order)
        return master_log_df

    def load_and_process_and_combine_log_csv_files(self, dict_of_log_files, unique_marathon_runs_list: list, step_col_name: str,
                                                   main_folder: str, keep_existing_runs: bool = True, save_log_df_to_parquet: bool = False) -> pl.DataFrame:
        """Read all log step file CSVs, concat, then optionally save to parquet"""
        marathon_col = "marathon"
        df_list      = []

        for log_file in dict_of_log_files.values():
            df = self.read_csv_and_lowercase_cols_names(log_file['path'])
            df = self.add_marathon_and_step_cols_to_df(df, log_file[marathon_col], log_file['step'], step_col_name)
            df = self.split_log_df_by_marathon_run_existence(df, unique_marathon_runs_list, keep_existing_runs=keep_existing_runs)
            df = self.insert_step_cols_after_run(df, step_col_name)
            df = self.cast_df_cols_to_float64(df)
            df = self.drop_single_value_cols(df, step_col_name)
            df = self.append_step_suffix_to_cols(df, step_col_name, log_file['step'])
            df_list.append(df)

        master_log_df = pl.concat(df_list, how="diagonal")
        master_log_df = self.reorder_cols(master_log_df, step_col_name)
        master_log_df = master_log_df.rename({col: col.strip().lower() for col in master_log_df.columns})

        if save_log_df_to_parquet:
            master_log_df.write_parquet(f"{main_folder}/{self.PARQUET_FOLDER_NAME}/master_log_file.parquet")
        return master_log_df

class WaferFilesProcessor:
    """Class dealing with processing spatial files"""

    @staticmethod
    def _move_cols_in_df_after_target(cols, target: str, new_cols: Sequence[str]) -> list[str]:
        """Moves cols in a df right after a target col"""
        cols = cols.copy()
        for c in new_cols:
            if c in cols: # avoids duplicates
                cols.remove(c)
        idx = cols.index(target) + 1
        return cols[:idx] + new_cols + cols[idx:]

    @staticmethod
    def load_wafer_csv_files_and_merge_to_df(dict_of_spatial_files: dict) -> pl.DataFrame:
        """Loads WAFER csv files from different steps/marathons, adds marathon col, then merges into 1 big df"""
        dfs = []
        for val in dict_of_spatial_files.values():
            pdf = pd.read_csv(val['path'],
                              decimal   = '.',
                              na_values = ["", "NA", "null"],)
            df = pl.from_pandas(pdf).with_columns([pl.lit(val['marathon']).alias("marathon"),
                 (pl.lit(val['marathon']).cast(pl.Utf8) + "_" + pl.col("#Run").cast(pl.Utf8)).alias("marathon_run")])
            dfs.append(df)

        wafer_master_df= pl.concat(dfs, how = "vertical")
        cols           = WaferFilesProcessor._move_cols_in_df_after_target(wafer_master_df.columns, "#Run", ["marathon", "marathon_run"])

        return wafer_master_df.select(cols)

    @staticmethod
    def _spot_duplicates_in_wafer_df(wafer_df, desired_cols: list):
        counts = wafer_df.group_by(desired_cols).agg(pl.count()).rename({"count": "cnt"})
        dupes  = counts.filter(pl.col("cnt") > 1).select(desired_cols)
        if dupes.height > 0:
            raise KeyError(f"{dupes.height} Duplicate rows found in wafer_df")

    @staticmethod
    def split_1_wafer_df_to_y_and_radius_df(wafer_df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Pivot wafer_df to create:
        - y_df (output): Spatial property averaged by marathon_run and Site #
        - radius_df (output): marathon_run and Radius (mm) columns
        - wafer_df (pl.DataFrame): df with SINGLE wafer data"""

        radius_df = wafer_df.select(["marathon_run", "Radius (mm)"])
        y_df = wafer_df.pivot(values = "Spatial property (nm)",
                              index  = "marathon_run",
                              columns= "Site #",
                              aggregate_function="mean")
        y_df = y_df.sort("marathon_run")

        WaferFilesProcessor._spot_duplicates_in_wafer_df(wafer_df, ["marathon_run", "Site #"])
        return y_df, radius_df

    @staticmethod
    def split_master_spatial_df_by_rc(wafer_df: pl.DataFrame, col_to_group_by: str, new_col_name: str) -> dict:
        """Splits wafer_df by the 'RC' col values and returns a dict of dfs, each has as key (0-3) and wafer/RC value (1-4)
        dict of shape:   {int(rc_val) - 1 : df_subset} x 4"""

        spatial_df_dict = {}
        for rc_val in wafer_df[col_to_group_by].unique():
            df_subset = wafer_df.filter(pl.col(col_to_group_by) == rc_val).with_columns(
                pl.col(col_to_group_by).alias(new_col_name)) # keep original RC values in col
            spatial_df_dict[int(rc_val) - 1] = df_subset
        return spatial_df_dict

    @classmethod
    def load_spatial_csv_and_create_targets(cls, dict_of_spatial_files, main_folder: str, parquet_folder_name: str, save: bool = False) -> tuple:
        """Merge wafer files, split by RC, and create y and radius dataframes"""
        wafer_processor   = WaferFilesProcessor()
        master_spatial_df = wafer_processor.load_wafer_csv_files_and_merge_to_df(dict_of_spatial_files)
        if save:
            master_spatial_df.write_parquet(f"{main_folder}/{parquet_folder_name}/master_wafer_file.parquet")
        spatial_df_dict = wafer_processor.split_master_spatial_df_by_rc(master_spatial_df, "RC", "wafer")

        y_df_dict, radius_df_dict, wide_radius_df_dict = {}, {}, {}
        for idx, wafer_df in spatial_df_dict.items():
            y_df_dict[idx], radius_df_dict[idx] = wafer_processor.split_1_wafer_df_to_y_and_radius_df(wafer_df)
            radius_df_with_idx = radius_df_dict[idx].sort("marathon_run").with_columns(
                pl.arange(0, pl.len()).over("marathon_run").alias("radius_idx"))
            wide_radius_df_dict[idx] = radius_df_with_idx.pivot(values= "Radius (mm)",
                                                                index = "marathon_run",
                                                                on    = "radius_idx",
                                                                aggregate_function = "first").sort("marathon_run")
        return master_spatial_df, spatial_df_dict, y_df_dict, wide_radius_df_dict

