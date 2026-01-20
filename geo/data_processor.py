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
    X      = pd.read_csv(file_loc)
    y      = X[y_cols].values
    X      = X.drop(columns=y_cols)
    return X, y


def process_china_weather_dataset(file_location, y_col_indices, page_choice):
    china_weather_array = np.load(file_location).transpose(0, 2, 1)
    mask                = np.ones(china_weather_array.shape[2], dtype=bool)
    mask[y_col_indices] = False
    X_cut               = china_weather_array[:, :, mask]      # (stations, timesteps, n_features)
    y_cut               = china_weather_array[:, :, y_col_indices] # (stations, timesteps, n_targets)

    assert X_cut.shape[2] + y_cut.shape[2] == china_weather_array.shape[2]

    X = pd.DataFrame(X_cut[page_choice])
    y = y_cut[page_choice]
    return X, y









