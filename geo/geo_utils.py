import __main__
import sys, os
project_root = os.path.abspath("..")  # adjust if notebook is elsewhere
sys.path.insert(0, project_root)
from typing import Dict, List, Literal, Tuple, Optional, Any, Union
import logging

import category_encoders as ce
import matplotlib.pyplot as plt

import numexpr as ne # makes numpy operations faster
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, mean_absolute_error, root_mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.random_projection import GaussianRandomProjection

from catboost import CatBoostRegressor, CatBoostClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # print(torch.cuda.memory_reserved(0) / 1e6, "MB reserved")
    # print(torch.cuda.memory_allocated(0) / 1e6, "MB allocated")

import src.param_config.config_paths as P

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.info("Starting process...")
logging.warning("Something looks off...")
logging.error("Something failed.")

def read_csv_and_skip_rows_then_save(file_loc: str, file_name: str, rows_to_skip: int, changed_file_name: str):
    "rows_to_skip: 0 (no skipping), 1 (skip every other), 2 (skip 2 out of 3), etc."
    df = pd.read_csv(file_loc+file_name, skiprows=lambda i: i % rows_to_skip == 1)
    df.to_csv(file_loc+changed_file_name, index=False)

def drop_low_variance_cols(X: pd.DataFrame, threshold: float = 1e-6) -> pd.DataFrame:
    """Drop columns in X whose variance is below threshold."""
    return X.loc[:, X.var(ddof=0) >= threshold]


class Periodicity:
    @staticmethod
    def _dominant_periods(x: np.ndarray, fs: float = 1.0, topk: int = 2, max_period: int = None) -> list[int]:
        """Return top-k dominant periods, ignoring near-DC artifacts."""
        n       = len(x)
        freqs   = np.fft.rfftfreq(n, d=1/fs)[1:]
        power   = np.abs(np.fft.rfft(x))[1:]**2
        periods = 1 / freqs

        if max_period is None:
            max_period = int(n * 0.25)

        mask = periods <= max_period
        idxs = np.argsort(power[mask])[-topk:]
        return sorted(int(round(periods[mask][i])) for i in idxs)



def scale_train_and_test_sets(train_set: Union[pd.DataFrame, pd.Series, np.ndarray],
                              test_set: Union[pd.DataFrame, pd.Series, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard-scale train/test sets; supports empty blocks; returns torch tensors."""

    # ---- handle empty blocks (DataFrame or ndarray) ----
    if (isinstance(train_set, pd.DataFrame) and train_set.shape[1] == 0) or \
       (isinstance(train_set, np.ndarray) and train_set.ndim == 2 and train_set.shape[1] == 0):
        n_train = len(train_set)
        n_test  = len(test_set)
        return (torch.empty((n_train, 0), dtype=torch.float32),
                torch.empty((n_test, 0), dtype=torch.float32))

    # ---- normalize input types ----
    if isinstance(train_set, pd.Series):
        train_set = train_set.to_frame()
        test_set  = test_set.to_frame()
    elif isinstance(train_set, np.ndarray) and train_set.ndim == 1:
        train_set = train_set.reshape(-1, 1)
        test_set  = test_set.reshape(-1, 1)

    scaler = StandardScaler()
    train_scaled = torch.tensor(scaler.fit_transform(train_set), dtype=torch.float32)
    test_scaled  = torch.tensor(scaler.transform(test_set), dtype=torch.float32)
    return train_scaled, test_scaled

def compute_cyclicity_score(time_series_1d: np.ndarray) -> float:
    """Compute a cyclicity score for a 1D time series. The cyclicity score measures how strongly periodic a signal is
    by identifying the fraction of total variance captured by the dominant freq.
    Steps:
    1. Remove mean from time series (zero-center).
    2. Compute real-valued Fourier transform (rFFT).
    3. Compute raw Fourier power (squared magnitude of FFT coeffs), NOT a normalized power spectral density (PSD)
    4. Ignore DC component (mean) to focus on cyclic variation.
    5. cyclicity score = fraction of strongest freq divided by total power.
    Args:
    - time_series : np.ndarray. 1D array of time series values.
    - Cyclicity score: fraction of total variance explained by the strongest frequency.
    - Cyclicity score = (Fourier power of strongest freq) / sum of all Fourier power (total variance of zero-mean time series)
        - Close to 1 → strongly cyclic
        - Close to 0 → weak or non-cyclic"""
    try:
        x        = np.asarray(time_series_1d, dtype=float)
        x        = x - x.mean()
        fft_vals = np.fft.rfft(x) # Fourier transform
        power    = np.abs(fft_vals) ** 2 # fourier power, not spectral density
        power[0] = 0.0 # remove DC component
        return power.max() / (power.sum() + 1e-8)
    except Exception as e:
        print(f"{e}, col type incompatible")
        return 0

def compute_multicyclicity_scores(time_series_1d: np.ndarray, top_k: int = 3, threshold: float = 0.1):
    """Compute the top_k dominant frequencies of a 1D signal using FFT.
    Args:
        time_series_1d: 1D signal shape (T,)
        top_k: number of dominant frequencies to return
        threshold: minimum power_ratio to consider the feature cyclic
    Returns:
        List of tuples (freq, power_ratio) sorted by descending power_ratio.
        freq:
            normalized frequency (cycles per sample)
        power_ratio:
            power at that frequency divided by total power
            (fraction of variance explained by that frequency)
        If power_ratio < threshold, the feature is considered non-cyclic."""
    x           = np.asarray(time_series_1d, dtype=float)
    x           = x - x.mean()
    fft_vals    = np.fft.rfft(x)
    power       = np.abs(fft_vals) ** 2
    power[0]    = 0
    power_total = power.sum() + 1e-12

    idx    = np.argsort(power)[-top_k:][::-1]
    freqs  = idx / len(x)
    ratios = power[idx] / power_total
    return [(float(freqs[i]), float(ratios[i])) for i in range(top_k) if ratios[i] >= threshold]

def split_dataset_to_linear_and_cyclic(dataset: pd.DataFrame, threshold: float = 0.5, verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset X columns into 2; cyclic and linear features, based on cyclicity score
    Returns column indices, NOT values
    dataset: pd/pl dataframe"""
    cyc_cols, lin_cols = [], []
    for col in dataset.columns:
        try:
            score = compute_cyclicity_score(dataset[col].to_numpy())
            (cyc_cols if score > threshold else lin_cols).append(col)
            if verbose:
                print(f"col {col} cycl. score: {score:.3f} → {'cyclic' if score > threshold else 'linear'}")
        except Exception as e:
            if verbose:
                print(f"Error processing col {col}: {e}")
    if verbose:
        print(f"{100*len(cyc_cols)/len(dataset.columns):.2f}% cyclic cols, {100*len(lin_cols)/len(dataset.columns):.2f}% linear cols")
    return dataset[lin_cols], dataset[cyc_cols]

class Windowing:
    @staticmethod
    def make_windows_from_X(X: torch.Tensor, window_size: int, sliding_size: int = 1, horizon: int = 1, task: str = "forecast") -> torch.Tensor:
        """Create X windows aligned with y windows
        Sliding window function, task agnostic.
        Converts:
        - 2D input (T, D) → 3D output (num_windows, window_size, D)
        - 3D input (N, T, D) → 3D output (N * num_windows, window_size, D)
        Parameters
        X : torch.Tensor
            Input tensor, 2D or 3D
        window_size : int
            Number of timesteps per window
        sliding_size : int
            Step size for sliding
        Returns: Windowed tensor as described above"""
        if isinstance(X, (pd.DataFrame, np.ndarray)):
            X = torch.tensor(np.array(X), dtype=torch.float32)
        if X.ndim == 2:  # (T, D)
            T, D = X.shape
            if task == "forecast":
                last_start = T - window_size - horizon + 1
            else:
                last_start = T - window_size + 1
            if last_start <= 0:
                return torch.empty((0, window_size, D))
            return torch.stack(
                [X[i:i + window_size] for i in range(0, last_start, sliding_size)])
        elif X.ndim == 3:  # (N, T, D)
            windows = [
                Windowing.make_windows_from_X(X[n], window_size, sliding_size, horizon, task)
                for n in range(X.shape[0])]
            windows = [w for w in windows if w.numel() > 0]
            return torch.cat(windows, dim=0) if windows else torch.empty((0, window_size, X.shape[-1]))
        else:
            raise ValueError(X.shape)

    @staticmethod
    def old_make_windows_from_y(y: np.ndarray, window_size: int, sliding_size: int, task: str,) -> np.ndarray:
        """Window labels to match X windows.
        Args:
            - y: np.ndarray, single sequence of shape (T,) or batch of sequences (N, T)
            - window_size: int
            - sliding_size: int
            - task: str, one of ["forecast", "nowcast", "tabular"]
                - "forecast": take last value of each window
                - "nowcast": take mean value of each window
                - "tabular": take first value of each window"""
        y = np.asarray(y)

        # ---- multi y-sequence ----
        if y.ndim == 3:
            return np.concatenate([Windowing.make_windows_from_y(y[i], window_size, sliding_size, task)
                                   for i in range(y.shape[0])], axis=0,)
        # ---- single y sequence ----
        windows = []
        T       = len(y)
        last_start_idx = (T - window_size) // sliding_size * sliding_size
        if last_start_idx < 0:
            return np.empty((0,) + y.shape[1:])

        for i in range(0, last_start_idx + 1, sliding_size):
            w = y[i : i + window_size]
            if task == "forecast": # take last value
                windows.append(w[-1])
            elif task == "nowcast": # take last value
                windows.append(w[-1])   # causal nowcasting
            elif task == "tabular": # take first value, doesnt matter since for tabular window_size = 1
                windows.append(w[0])
            else:
                raise ValueError(task)
        return np.asarray(windows)

    @staticmethod
    def make_windows_from_y(y: np.ndarray, window_size: int, sliding_size: int, task: str, horizon: int = 1) -> np.ndarray:
        """Create y windows aligned with X windows.
        forecast: predict next `horizon` steps
        nowcast: predict last value of window
        tabular: predict first value of window"""
        y = np.asarray(y)

        # batch of sequences
        if y.ndim == 3:
            return np.concatenate([Windowing.make_windows_from_y(y[i], window_size, sliding_size, task, horizon) for i in range(y.shape[0])], axis=0,)
        T       = len(y)
        windows = []

        if task == "forecast":
            last_start = T - window_size - horizon + 1
        else:
            last_start = T - window_size + 1
        if last_start <= 0:
            return np.empty((0, horizon) if task == "forecast" else (0,))

        for i in range(0, last_start, sliding_size):
            w = y[i : i + window_size]
            if task == "forecast":
                windows.append(y[i + window_size : i + window_size + horizon])
            elif task == "nowcast":
                windows.append(w[-1])
            elif task == "tabular":
                windows.append(w[0])
            else:
                raise ValueError(task)
        return np.asarray(windows)

    @staticmethod
    def split_timeseries_by_windows(X, y, window_size: int, sliding_fraction: float, test_ratio: float,):
        """to use when test set < window size. Split time series so test set contains ceil(test_ratio) of windows."""
        sliding_size = int(window_size * sliding_fraction)
        if sliding_size <= 0:
            raise ValueError("sliding_fraction too small")

        X_len = len(X)
        if X_len < window_size:
            raise ValueError("X shorter than window_size")

        num_windows  = 1 + (X_len - window_size) // sliding_size
        test_windows = max(1, int(np.ceil(num_windows * test_ratio)))

        test_samples = window_size + (test_windows - 1) * sliding_size
        split_idx    = X_len - test_samples

        if split_idx <= 0:
            raise ValueError("Test set consumes entire series")

        slicer = slice(None, split_idx)
        slicer_test = slice(split_idx, None)

        def _slice(obj, sl):
            return obj[sl] if isinstance(obj, np.ndarray) else obj.iloc[sl]

        return (
            _slice(X, slicer),
            _slice(X, slicer_test),
            _slice(y, slicer),
            _slice(y, slicer_test),)

