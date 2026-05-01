import os
from typing import Literal, Tuple, Optional, List
from lttb import downsample
from rdp import rdp
import polars as pl
import math

import numpy as np
import pandas as pd

class DataframeReducer:
    """Functions aiming to reduce the df size as it is too large to process properly.
       - Methods include subsampling, selecting a single step, keeping latest rows per run/step
       - Can also keep a certain fraction of unique runs based on specified columns."""

    @staticmethod
    def _select_1_step_from_df(df: pl.DataFrame, step_number, step_id_col) -> pl.DataFrame:
        """gets out 1 step from df based on step_number"""
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


class LTTBDownsampler:
    @staticmethod
    def downsample_using_lttb_union(df: pd.DataFrame, time_col: str, max_points_per_col: int) -> pd.DataFrame:
        """Downsample multivar timeseries using LTTB, keeping any point that is 'important' in at least 1 feature"""
        time        = df[time_col]
        kept_indices= set()

        is_time_col_datetime = pd.api.types.is_datetime64_any_dtype(time)
        if is_time_col_datetime:
            numeric_time = time.astype('int64')
        else:
            numeric_time = time

        # Apply LTTB to each col individually, then union indices
        for col in df.columns:
            if col == time_col:
                continue
            downsampled_col = downsample(np.column_stack((numeric_time.values, df[col].values)), max_points_per_col)
            
            # Find indices based on the original time column
            if is_time_col_datetime:
                # Reconstruct datetime from numeric_time to find indices
                downsampled_time = pd.to_datetime(downsampled_col[:, 0], unit='ns')
                col_indices      = df.index[df[time_col].isin(downsampled_time)]
            else:
                col_indices = df.index[df[time_col].isin(downsampled_col[:, 0])]
            kept_indices.update(col_indices)
        return df.loc[sorted(kept_indices)].reset_index(drop=True)

    @staticmethod
    def downsample_using_lttb_intersection(df: pd.DataFrame, time_col: str, max_points_per_col: int) -> pd.DataFrame:
        """Downsample multivar timeseries using LTTB, keeping only points important in all features (intersection)."""
        time = df[time_col]

        is_time_col_datetime = pd.api.types.is_datetime64_any_dtype(time)
        if is_time_col_datetime:
            numeric_time = time.astype('int64')
        else:
            numeric_time = time

        kept_indices = None

        for col in df.columns:
            if col == time_col:
                continue
            downsampled_col = downsample(np.column_stack((numeric_time.values, df[col].values)), max_points_per_col)
            
            if is_time_col_datetime:
                downsampled_time = pd.to_datetime(downsampled_col[:, 0], unit='ns')
                col_indices = set(df.index[df[time_col].isin(downsampled_time)])
            else:
                col_indices = set(df.index[df[time_col].isin(downsampled_col[:, 0])])

            if kept_indices is None:
                kept_indices = col_indices  # Initialize on first column
            else:
                kept_indices = kept_indices.intersection(col_indices)  # Intersection on others

        if kept_indices is None:
            # No columns other than time_col
            return df.copy()

        return df.loc[sorted(kept_indices)].reset_index(drop=True)

    @staticmethod
    def downsample_lttb_union_with_spikes(df: pd.DataFrame, time_col: str, max_points_per_col: int,
                                          spike_thresh: float = 4, spike_method: str = 'zscore') -> pd.DataFrame:
        """Downsample multivar timeseries using LTTB-union, but keep all 'spike' points
        'spike' = any point where abs(z-score) > spike_thresh for that col
        Works with numeric or datetime time col
        spike_method: "zscore" -> mean/std-based
                      "mad"    -> median/'Median Absolute Deviation'-based
        spike_thresh: threshold for spike detection (abs score > threshold)
                      ie. spike_thresh=4 means ~4 stdev away
        Works with numeric or datetime time col"""
        time        = df[time_col]
        kept_indices= set()
        MOD_ZSCORE  = 0.6745 # norm. constant
        # Convert datetime to numeric for LTTB if needed
        is_datetime = pd.api.types.is_datetime64_any_dtype(time)
        numeric_time= time.astype("int64") if is_datetime else time

        # 1: Spike detection
        for col in df.columns:
            if col == time_col:
                continue
            series = df[col].astype(float)

            if spike_method == "zscore":
                score  = (series - series.mean()) / series.std(ddof=0)
            elif spike_method == "mad":
                median = series.median()
                mad    = np.median(np.abs(series - median))
                if mad == 0:
                    continue  # avoid /0
                score = MOD_ZSCORE * (series - median) / mad
            else:
                raise ValueError("spike_method must be 'zscore' or 'mad'")
            spike_idx = df.index[np.abs(score) > spike_thresh]
            kept_indices.update(spike_idx)

        # 2: LTTB downsampling
        for col in df.columns:
            if col == time_col:
                continue
            downsampled = downsample(
                np.column_stack((numeric_time.values, df[col].values)), max_points_per_col)
            if is_datetime:
                down_t  = pd.to_datetime(downsampled[:, 0], unit="ns")
                col_idx = df.index[df[time_col].isin(down_t)]
            else:
                col_idx = df.index[df[time_col].isin(downsampled[:, 0])]
            kept_indices.update(col_idx)
        return df.loc[sorted(kept_indices)].reset_index(drop=True)

class OtherDownsamplers:
    @staticmethod
    def choose_rdp_epsilon(coords: np.ndarray, method: Literal["mad", "std", "fraction"] = "mad",
                        factor: float = 1.0, normalize_axes: bool = False) -> float:
        """Choose an RDP epsilon for coords (N x 2: [x, y])
        method:
        - "mad": epsilon = factor * MAD(y)
        - "std": epsilon = factor * std(y)
        - "fraction": epsilon = factor * (range(y))
        factor: multiplier (tune: 0.5-3 for mad/std, 1e-4-1e-2 for fraction)
        normalize_axes: if True, scale both axes to [0,1] before computing epsilon
        Returns: epsilon in the same units as coords (or normalized units if normalize_axes=True)"""
        if isinstance(coords, pd.DataFrame):
            x = coords.iloc[:, 0].values
            y = coords.iloc[:, 1].values
        else:
            x = coords[:, 0]
            y = coords[:, 1]

        # Convert datetime to int64 (nanoseconds)
        if np.issubdtype(x.dtype, np.datetime64):
            x = x.astype("int64")

        if normalize_axes:
            def _norm(v):
                r = v.max() - v.min()
                return (v - v.min()) / (r if r != 0 else 1.0)
            x = _norm(x)
            y = _norm(y)

        if method == "mad":
            med = np.median(y)
            base = np.median(np.abs(y - med))
        elif method == "std":
            base = np.std(y, ddof=0)
        elif method == "fraction":
            base = y.max() - y.min()
        else:
            raise ValueError("method must be 'mad', 'std' or 'fraction'")

        return float(max(base * factor, 1e-12))

    @staticmethod
    def downsample_using_rdp_union(df: pd.DataFrame, time_col: str, epsilon: float) -> pd.DataFrame:
        """Downsample multivar timeseries using Ramer-Douglas-Peucker algo, union of important points across cols
        - epsilon: max. distance between original points and simplified line (larger = more aggressive downsampling
        = less points)"""
        time = df[time_col]
        kept_indices = set()

        is_time_col_datetime = pd.api.types.is_datetime64_any_dtype(time)
        numeric_time = time.astype("int64") if is_time_col_datetime else time

        for col in df.columns:
            if col == time_col:
                continue
            coords           = np.column_stack((numeric_time, df[col].values))
            simplified_points= rdp(coords, epsilon=epsilon)
            simplified_times = simplified_points[:, 0]
            temp_df          = pd.DataFrame({time_col: numeric_time, "orig_idx": df.index})
            selected         = temp_df[temp_df[time_col].isin(simplified_times)]
            kept_indices.update(selected["orig_idx"].tolist())
        return df.loc[sorted(kept_indices)].reset_index(drop=True)

    @staticmethod
    def downsample_using_vw_union(df: pd.DataFrame, time_col: str, target_points: int) -> pd.DataFrame:
        """Downsample multivar timeseries using Visvalingam–Whyatt algorithm,
        union of important points across columns.
        Args:
            df: DataFrame containing the time series.
            time_col: Name of the time column.
            target_points: Approximate number of points to keep per column.
        Returns:
            Downsampled DataFrame with union of kept points."""

        def visvalingam_whyatt(coords: np.ndarray, keep_points: int) -> np.ndarray:
            """Visvalingam–Whyatt simplification preserving original timestamps."""
            N = len(coords)
            if N <= keep_points:
                return coords

            coords_f = coords.astype(float)
            x_orig = coords_f[:, 0].copy()
            coords_f[:, 0] -= coords_f[:, 0].min()  # Shift to avoid overflow

            areas = np.empty(N, dtype=float)
            areas[0] = areas[-1] = np.inf
            for i in range(1, N - 1):
                x1, y1 = coords_f[i - 1]
                x2, y2 = coords_f[i]
                x3, y3 = coords_f[i + 1]
                areas[i] = abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)

            to_remove = np.argsort(areas)[:-keep_points]
            mask = np.ones(N, dtype=bool)
            mask[to_remove] = False

            simplified = coords_f[mask]
            simplified[:, 0] = x_orig[mask]  # Restore original times

            return simplified

        time = df[time_col]
        kept_indices = set()

        is_time_col_datetime = pd.api.types.is_datetime64_any_dtype(time)
        numeric_time = time.astype("int64") if is_time_col_datetime else time.values

        for col in df.columns:
            if col == time_col:
                continue
            coords = np.column_stack((numeric_time, df[col].values))
            simplified_points = visvalingam_whyatt(coords, keep_points=target_points)
            simplified_times = simplified_points[:, 0]

            temp_df = pd.DataFrame({time_col: numeric_time, "orig_idx": df.index})
            selected = temp_df[temp_df[time_col].isin(simplified_times)]
            kept_indices.update(selected["orig_idx"].tolist())

        return df.loc[sorted(kept_indices)].reset_index(drop=True)

    @staticmethod
    def subsample_timeseries(df: pd.DataFrame, subsampling_ratio: int = 5) -> pd.DataFrame:
        if subsampling_ratio <= 0:
            raise ValueError("Subsampling ratio must be positive.")
        return df.iloc[::subsampling_ratio]

    @staticmethod
    def get_last_N_rows(df: pd.DataFrame, N_rows: int) -> pd.DataFrame:
        """Get the last N rows of the DataFrame."""
        if N_rows <= 0:
            raise ValueError("N must be a positive integer.")
        return df.tail(N_rows).reset_index(drop=True)
