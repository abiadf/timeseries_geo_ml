import polars as pl
from typing import Optional, List
import math

class DataframeReducer:
    """Functions aiming to reduce the df size as it is too large to process properly.
       - Methods include subsampling, selecting a single step, keeping latest rows per run/step
       - Can also keep a certain fraction of unique runs based on specified columns."""

    @staticmethod
    def _select_1_step_from_df(df: pl.DataFrame, step_number, step_id_col) -> pl.DataFrame:
        """ASM-specific, gets out 1 step from df based on step_number"""
        return df.filter(pl.col(step_id_col) == step_number)

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
    def reduce_df(reduction_method: str, df: pl.DataFrame, step_id_col, process_time_col, marathon_run_col, wafer_col,
                  step_number: Optional[int] = None, N_downsampling: Optional[int] = None, N_last_rows_per_run: Optional[int] = None) -> pl.DataFrame:
        """multi-purpose function to reduce df size:
            - method: Reduction method, one of
                - "nothing": Do not reduce, return the original df
                - "subsample": Downsample rows (requires N_downsampling)
                - "subsample_1_step": Downsample single step
                - "latest_rows_per_run": Keep last N rows per (marathon_run, wafer) (requires N_last_rows_per_run)
                - "latest_rows_per_step_run": Keep last N rows per (step_id, marathon_run, wafer) (requires N_last_rows_per_run)
                - "1_step": Filter rows to a single step (requires step_number)
            - df: Input DataFrame to reduce.
            - step_number: Step number for "1_step" method.
            - N_downsampling: Downsampling factor for "subsample" method.
            - N_last_rows_per_run: Number of last rows to keep for "latest_rows_per_*" methods.
            - output: reduced polars df"""

        if reduction_method == "nothing":
            return df

        elif reduction_method == "subsample":
            subsampled_log_df = DataframeReducer.downsample_df_rows(df, N_downsampling)
            return subsampled_log_df

        elif reduction_method == "subsample_1_step":
            if step_number is None or N_downsampling is None:
                raise ValueError("step_number and N_downsampling must be provided for 'subsample_1_step'")
            df_step = DataframeReducer._select_1_step_from_df(df, step_number, step_id_col)
            return DataframeReducer.downsample_df_rows(df_step, N_downsampling)

        elif reduction_method == "1_step":
            return DataframeReducer._select_1_step_from_df(df, step_number, step_id_col)

        elif reduction_method == "latest_rows_per_run": # all runs
            latest_rows_per_run_df = (df
                                    .sort(process_time_col)
                                    .group_by([marathon_run_col, wafer_col])
                                    .tail(N_last_rows_per_run))
            return latest_rows_per_run_df

        elif reduction_method == "latest_rows_per_step_run": # all runs per step
            latest_rows_per_step_run_df = (df
                                        .sort(process_time_col)
                                        .group_by([step_id_col, marathon_run_col, wafer_col])
                                        .tail(N_last_rows_per_run))
            return latest_rows_per_step_run_df
        else:
            raise ValueError(f"Unknown reduction method: {reduction_method}")

    @staticmethod
    def keep_certain_number_of_runs(df: pl.DataFrame, cols_to_index: List[str], keep_fraction: float = 0.3) -> pl.DataFrame:
        """Keep certain fraction of unique combinations of runs in df. keep_fraction = 1, keep all runs
            - df (pl.DataFrame): Input df
            - cols_to_index (List[str]): Columns to consider for uniqueness
            - keep_fraction (float): Fraction of unique combinations to keep
            - Returns: filtered df with only the kept runs"""
        if keep_fraction == 1:
            return df

        unique_runs = df.unique(subset=cols_to_index)
        n_keep      = math.ceil(unique_runs.height * keep_fraction)
        keep_runs   = unique_runs.head(n_keep)

        # Semi-join to filter rows that match key columns, no duplication
        filtered_df = df.join(keep_runs, on=cols_to_index, how="semi")
        return filtered_df

