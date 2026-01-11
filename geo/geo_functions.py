import __main__
import sys, os
project_root = os.path.abspath("..")  # adjust if notebook is elsewhere
sys.path.insert(0, project_root)
from typing import Dict, List, Literal, Tuple, Optional, Any
import logging
from pathlib import Path

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
print(device)

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.info("Starting process...")
logging.warning("Something looks off...")
logging.error("Something failed.")

def compute_cyclicity_score(time_series: np.ndarray) -> float:
    """Compute a cyclicity score for a 1D time series.
    - time_series : np.ndarray. 1D array of time series values.
    - Cyclicity score: fraction of total variance explained by the strongest frequency.
        - Close to 1 → strongly cyclic
        - Close to 0 → weak or non-cyclic"""
    x        = np.asarray(time_series, dtype=float)
    x        = x - x.mean()
    fft_vals = np.fft.rfft(x)
    power    = np.abs(fft_vals) ** 2
    power[0] = 0.0  # remove DC component
    return power.max() / (power.sum() + 1e-8)

def split_dataset_to_linear_and_cyclic(dataset: pd.DataFrame, threshold: float = 0.5, verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset X columns into 2; cyclic and linear features, based on cyclicity score
    Returns column indices, NOT values
    dataset: pd/pl dataframe"""
    cyc_cols, lin_cols = [], []
    for col in dataset.columns:
        score = compute_cyclicity_score(dataset[col].to_numpy())
        (cyc_cols if score > threshold else lin_cols).append(col)
        if verbose:
            print(f"col {col} cycl. score: {score:.3f} → {'cyclic' if score > threshold else 'linear'}")
    if verbose:
        print(f"{100*len(cyc_cols)/len(dataset.columns):.2f}% cyclic cols, {100*len(lin_cols)/len(dataset.columns):.2f}% linear cols")
    return dataset[lin_cols], dataset[cyc_cols]

class Windowing:
    @staticmethod
    def make_windows_from_X(X: torch.Tensor, window_size: int, sliding_size: int = 1) -> torch.Tensor:
        """Sliding window function, task agnostic.
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

        if isinstance(X, pd.DataFrame) or isinstance(X, np.ndarray):
            X = torch.tensor(np.array(X), dtype=torch.float32)

        if X.ndim == 2:  # (T, D)
            timesteps, _ = X.shape
            num_windows  = (timesteps - window_size) // sliding_size + 1
            windows      = torch.stack([X[i:i + window_size] for i in range(0, num_windows*sliding_size, sliding_size)])
            return windows  # (num_windows, window_size, cols)
        elif X.ndim == 3:  # (N, T, D)
            pages, timesteps, _ = X.shape
            num_windows         = (timesteps - window_size) // sliding_size + 1
            windows_list        = []
            for n in range(pages):
                windows_list.append(torch.stack([X[n, i:i + window_size] for i in range(0, num_windows*sliding_size, sliding_size)]))
            return torch.cat(windows_list, dim=0)  # (N*num_windows, window_size, D)
        else:
            raise ValueError(f"X must be 2D or 3D, got {X.shape}")

    @staticmethod
    def make_windows_from_y(y: np.ndarray, window_size: int, sliding_size: int, task: str,) -> np.ndarray:
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

