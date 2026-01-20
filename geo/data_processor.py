"""Contains the logic for processing a dataset to X and y"""

import __main__
import sys, os
import time
project_root = os.path.abspath("..")  # adjust if notebook is elsewhere
sys.path.insert(0, project_root)
from typing import Dict, List, Literal, Tuple, Optional, Any, Union
import logging
import random
import umap
import yaml
from dataclasses import dataclass

import category_encoders as ce
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numexpr as ne # makes numpy operations faster
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

from scipy.signal import periodogram

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, mean_absolute_error, root_mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.random_projection import GaussianRandomProjection

from catboost import CatBoostRegressor, CatBoostClassifier
from pyriemann.tangentspace import TangentSpace

from geo.geo_utils import compute_cyclicity_score, split_dataset_to_linear_and_cyclic, scale_train_and_test_sets, drop_low_variance_cols, Windowing

from geo_encoders import EuclidEncoder, SphericalEncoder, Decoder, LSTMEncoderEuclid, LSTMSphericalEncoder, \
LSTMToroidalEncoder, LSTMDecoder, MLPDecoder, Reparam, MixedEncoder, WithSplit, NoSplit, kl_gaussian, kl_vmf_uniform, \
regularization_vmf, early_stop, fit_catboost_multi, evaluate_model_full, estimate_entropy, pool_latents

from ucimlrepo import fetch_ucirepo

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
from src.utils.io_utils import read_yaml_params

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.info("Starting process...")
logging.warning("Something looks off...")
logging.error("Something failed.")


def process_dataset_given_filename(file_loc, y_cols: List[str]):
    """GIven a dataset file location and target column names, process and return X and y"""
    if isinstance(y_cols, str):
        y_cols = [y_cols]
    X      = pd.read_csv(file_loc)
    y      = X[y_cols].values
    X      = X.drop(columns=y_cols)
    return X, y

def process_china_weather_dataset(file_location: str, y_col_indices: list[int]):
    """Process China weather tensor dataset."""
    if not all(isinstance(i, int) for i in y_col_indices):
        raise TypeError("y_col_indices must be integer feature indices")
    page_choice = 7
    china_weather_array = np.load(file_location).transpose(0, 2, 1)
    mask = np.ones(china_weather_array.shape[2], dtype=bool)
    mask[y_col_indices] = False
    X_cut = china_weather_array[:, :, mask]
    y_cut = china_weather_array[:, :, y_col_indices]
    X = pd.DataFrame(X_cut[page_choice])
    y = y_cut[page_choice]
    return X, y



def get_household_power_consumption(file_loc: str, y_cols: list[str]):
    """Load household power consumption dataset."""
    target = y_cols[0]
    if os.path.exists(file_loc):
        df = pl.read_parquet(file_loc)
    else:
        ds = fetch_ucirepo(id=235)
        df_pd = ds.data.original
        cols_to_fix = [c for c in df_pd.columns if c not in ["Date", "Time"]]
        for col in cols_to_fix:
            df_pd[col] = pd.to_numeric(df_pd[col], errors='coerce')
        df = pl.from_pandas(df_pd)
        df.write_parquet(file_loc)

    df = df.filter(pl.col(target).is_not_null())
    y  = df.get_column(target)
    X  = df.drop(target).fill_null(0)
    return X.to_pandas(), y.to_pandas()



