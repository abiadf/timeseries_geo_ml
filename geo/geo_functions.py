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

def split_dataset_to_linear_and_cyclic(dataset: pd.DataFrame, thresh: float = 0.5):
    """Split dataset X columns into 2; cyclic and linear features, based on cyclicity score
    Returns column indices, NOT values
    dataset: pd/pl dataframe"""
    cyc_cols, lin_cols = [], []
    for col in dataset.columns:
        score = compute_cyclicity_score(dataset[col].to_numpy())
        (cyc_cols if score > thresh else lin_cols).append(col)
    return dataset[lin_cols], dataset[cyc_cols]

def make_windows_from_data(x: torch.Tensor, win: int) -> torch.Tensor:
    """x: (T, D) → (N, win, D)"""
    return torch.stack([x[i:i+win] for i in range(len(x) - win + 1)])

