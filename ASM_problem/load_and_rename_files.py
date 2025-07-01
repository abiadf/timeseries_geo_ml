import pandas as pd
import polars as pl
from typing import Tuple, Union

class LogFilesProcessor:
    def __init__(self, common_id_cols_mod, common_id_cols):
        self.COMMON_ID_COLS_MOD = common_id_cols_mod
        self.COMMON_ID_COLS     = common_id_cols

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

    def remove_marathon_runs_not_found_in_wafer_df(self, df: pl.DataFrame, unique_marathon_runs_list: list) -> pl.DataFrame:
        """Keep only rows where 'marathon_run' exists in the provided list of valid wafer runs
        removing runs early on makes the processing faster/lighter"""
        df = df.filter(pl.col("marathon_run").is_in(unique_marathon_runs_list))
        return df

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

    # moved to utils file, remove this function if code is working fine
    # def remove_constant_valued_cols(self, df):
    #     """Drop numeric columns with a single unique value (constant-valued columns)."""
    #     if isinstance(df, pl.DataFrame):
    #         numeric_cols  = df.select(pl.selectors.numeric()).columns
    #         constant_cols = [col for col in numeric_cols if df[col].n_unique() == 1]
    #         return df.drop(constant_cols)
    #     elif isinstance(df, pd.DataFrame):
    #         constant_cols = [col for col in df.select_dtypes(include='number').columns
    #                         if df[col].nunique() == 1]
    #         return df.drop(columns=constant_cols)
    #     else:
    #         raise TypeError("Unsupported DataFrame type")

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
    

class WaferFilesProcessor:
    # def __init__(self, common_id_cols_mod):
        # self.COMMON_ID_COLS_MOD = common_id_cols_mod

    @staticmethod
    def _cols_in_df_after_target(cols, target, new_cols):
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
                              decimal  ='.',
                              na_values=["", "NA", "null"],)
            df = pl.from_pandas(pdf).with_columns([
                 pl.lit(val['marathon']).alias("marathon"),
                 (pl.lit(val['marathon']).cast(pl.Utf8) + "_" + pl.col("#Run").cast(pl.Utf8)).alias("marathon_run")])
            dfs.append(df)

        wafer_master_df= pl.concat(dfs, how="vertical")
        cols           = WaferFilesProcessor._cols_in_df_after_target(wafer_master_df.columns, "#Run", ["marathon", "marathon_run"])

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

        # =================
        # added this part
        # print(radius_df.shape)
        # radius_df = radius_df.join(
        #     radius_df.group_by(["marathon_run", "Radius (mm)"]).count().filter(pl.col("count") == 1),
        #     on=["marathon_run", "Radius (mm)"], how="inner")

        # print(radius_df.shape)
        # =================

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

        # spatial_df_dict = {i: df.with_columns(pl.lit(i).alias("wafer"))
        #                  for i, (_, df) in enumerate(wafer_df.partition_by("RC", as_dict=True).items())}
        # spatial_df_dict = {rc_val: df.with_columns(pl.col("RC").alias("wafer"))
        #                 for rc_val, df in wafer_df.partition_by("RC", as_dict=True).items()}
        # spatial_df_dict = {rc_val: df.rename({"RC": "wafer"})
        #                  for rc_val, df in wafer_df.partition_by("RC", as_dict=True).items()}
        # the 'rename' method^ was causing memory issues

        spatial_df_dict = {}
        for rc_val in wafer_df[col_to_group_by].unique():
            df_subset = wafer_df.filter(pl.col(col_to_group_by) == rc_val).with_columns(
                pl.col(col_to_group_by).alias(new_col_name)) # keep original RC values in col
            spatial_df_dict[int(rc_val) - 1] = df_subset
        return spatial_df_dict

