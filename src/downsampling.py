import os
from typing import Literal, Tuple, Optional
from lttb import downsample
from rdp import rdp

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostRegressor

from lstm_network import LSTMModel, LSTMTrainer

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
