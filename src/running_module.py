"still in dev"

import __main__
import os
from typing import Dict, List, Literal, Tuple, Optional
import logging
import math
import gc
import pickle
import time
import json
import random

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB

from datetime import datetime
from pathlib import Path

import numexpr as ne # makes numpy operations faster
import category_encoders as ce
from matplotlib import pyplot as plt
from momentfm import MOMENTPipeline
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import arff

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.random_projection import GaussianRandomProjection

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(torch.cuda.memory_reserved(0) / 1e6, "MB reserved")
    print(torch.cuda.memory_allocated(0) / 1e6, "MB allocated")

from benchmarks.ts2vec_runner import run_ts2vec, log_ts2vec_results
from benchmarks.timevae_runner import run_timevae, log_timevae_results
from benchmarks.moment_runner import MomentRunner

from methods.forecasting_module import TimeGPTForecaster, SARIMAXForecaster
from methods.cellsup import Cellsup, DeepClusterAndSwav
from methods.mlp_heads import _get_orthogonality_penalty, make_MLP_regression_head, evaluate_MLP_regressor, train_sup_head_per_encoder, train_sup_heads_joint

from utils.io_utils import JSONLogger, Notifiers, read_yaml_params, set_all_rand_seeds
from utils.metrics_utils import AutocorrMetrics, Preds, Losses, DimensionalityEstimator, ForecastUtils, SemiSupLearning
from utils.data_utils import Slicing, Bootstrapping, assign_encoder_weights, convert_numpy, select_top_X_features, Augmentations
from utils.model_utils import Decoder, ProjectionHead, TorchWrapper, schedule_learning_rate, norm_temp_xentropy_loss, profile_epoch

from preprocessing.dataset_preprocessors import DatasetPreprocessor, ECGLoader, GermanyDataset, WeatherDataset, process_argoverse_parquet
from preprocessing.data_loader import DatasetLoading, load_or_preprocess_dataset

from encoders.lstm_network import LSTMModel, LSTMTrainer, Seq2SeqLSTM
import encoders.autoencoders as ae
import encoders.train_autoencoders as train_ae
from encoders.ts2vec_encoder import TS2VecEncoder
from encoders.latents import Latents
from encoders.cnn import CnnAutoencoder

from src.param_config.config_file import interim_data_loc, public_data_loc, encoders_folder, ts2vec_params_loc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.info("Starting process...")
logging.warning("Something looks off...")
logging.error("Something failed.")





"Setting up params"

messager_params  = read_yaml_params("param_config/messager.yaml")
WEBHOOK_URL      = messager_params["webhook_url"]

# Default params path
params_path = "param_config/baseline_params.yaml"
params      = read_yaml_params(params_path)

# Override dataset if running from bash
dataset_from_env = os.getenv("DATASET")
if dataset_from_env is not None:
    params["basics"]["dataset"] = dataset_from_env
print("Dataset being used:", params["basics"]["desired_dataset"])
data_params = read_yaml_params("param_config/dataset_params.yaml")

# Optional: detect if running in notebook
running_in_notebook = not hasattr(__main__, "__file__")
print("Running in notebook:", running_in_notebook)
print("Params path:", params_path)

desired_dataset = params["basics"].get("dataset") or params["basics"]["desired_dataset"]
num_runs        = params["basics"]["num_runs"]
predictor_epochs= params["general_params"]["regressor"]["epochs_regressor"]
train_epochs    = data_params["general"]["train_epochs"]
NUM_PAGES_TO_USE= data_params["general"]["num_pages_to_use"]
WINDOWS_PER_PAGE= data_params[desired_dataset]["num_window_splits"]
NUM_ROWS        = data_params[desired_dataset]["num_rows_per_page"]

label_frac      = params["basics"]["label_frac"]
data_splitting  = params["basics"]["data_splitting"]
do_we_scale_y   = params["basics"]["do_we_scale_y"]
rand_seed       = params["basics"]["random_seed"]
set_all_rand_seeds(rand_seed)
# rng       = np.random.default_rng()
# rand_seed = rng.integers(42, 100)

# dataset_window = data_params[desired_dataset]["window_len"]

if "WINDOW_LEN" in os.environ:
    dataset_window = int(os.environ["WINDOW_LEN"])
else:
    # dataset_window = int(data_params[desired_dataset]["window_len"])
    dataset_window = int(data_params["general"]["window_len"])
print("Using window length:", dataset_window)

# method-specific params
AE_lr       = params["cellsup"]["AE_lr"]
weight_decay= params["cellsup"]["weight_decay"]
dropout     = params["cellsup"]["dropout"]
swav_iters  = params["cellsup"]["swav_iters"]
swav_temp   = params["cellsup"]["swav_temp"]
cluster_min = params["cellsup"]["clustering"]["cluster_min"]
cluster_max = params["cellsup"]["clustering"]["cluster_max"]

layer1_dim  = params["general_params"]["regressor"]["layer1_dim"]
layer2_dim  = params["general_params"]["regressor"]["layer2_dim"]
layer3_dim  = params["general_params"]["regressor"]["layer3_dim"]

regressor_epochs = params["general_params"]["regressor"]["epochs_regressor"]
lr_regressor     = params["general_params"]["regressor"]["lr"]


"""[RUN ME] Preprocess dataset, as class"""

if desired_dataset == "nasa": # Directly load NASA dataset
    X_train, X_test, y_train_scaled, y_test_scaled = DatasetLoading.load_nasa_data()
    window_size = "N/A"
else:
    X_train, X_test, y_train_scaled, y_test_scaled, window_size = load_or_preprocess_dataset(desired_dataset, NUM_PAGES_TO_USE, do_we_scale_y, 
                                                     dataset_window=dataset_window, random_seed=rand_seed, use_cache=False, num_rows_per_window=NUM_ROWS)
if X_train.shape[2] > 300:
    top_features_pct         = params["general_params"]["top_features_big_dataset_pct"] # % top features to select
    X_train, X_test, top_idx = select_top_X_features(X_train, y_train_scaled, X_test, top_features_pct)
elif X_train.shape[2] > 30:
    top_features_pct         = params["general_params"]["top_features_pct"] # % of top features to select
    X_train, X_test, top_idx = select_top_X_features(X_train, y_train_scaled, X_test, top_features_pct)

# save the model here

print(f"Random seed: {rand_seed}")
print(f"X_train: {X_train.shape} ({X_train.nbytes/1024**2:.1f} MB), X_test: {X_test.shape} ({X_test.nbytes/1024**2:.1f} MB)")
print(f"y_train: {y_train_scaled.shape} ({y_train_scaled.nbytes/1024**2:.1f} MB), y_test: {y_test_scaled.shape} ({y_test_scaled.nbytes/1024**2:.1f} MB)")
print(f"Mean: {X_train.mean():.2f}, {X_test.mean():.2f}, {y_train_scaled.mean():.2f}, {y_test_scaled.mean():.2f}")
print(f"Y is scaled: {do_we_scale_y}")
print("Min:", np.min(X_train), np.min(X_test))
print("Max:", np.max(X_train), np.max(X_test))


"[RUN ME] Setup step"
rng       = np.random.default_rng(rand_seed)
n_train   = len(X_train)
n_labeled = int(np.ceil(label_frac * n_train))
perm      = rng.permutation(n_train)

if data_splitting == "missing_labels":
    # Keep all of X_train, split y
    X_L = X_train[perm[:n_labeled]]
    y_L = y_train_scaled[perm[:n_labeled]]
    X_U = X_train[perm[n_labeled:]]
    y_U = y_train_scaled[perm[n_labeled:]]
elif data_splitting == "reduced_data":
    # Shrink X_train & y_train by fraction, no unlabeled
    X_L = X_train[perm[:n_labeled]]
    y_L = y_train_scaled[perm[:n_labeled]]
    X_U = np.empty((0, *X_L.shape[1:]), dtype=X_L.dtype)
    y_U = np.empty((0, *y_L.shape[1:]), dtype=y_L.dtype)

# X_train for supervised training is only the labeled portion
X_train        = X_L
y_train_scaled = y_L

# Optional: combine for TimeVAE or other use
X_small = np.concatenate([X_train, X_test], axis=0)
y_small = np.concatenate([y_train_scaled, y_test_scaled], axis=0)

print(f"{data_splitting=}, {label_frac=}")
print(f"X_L: {X_L.shape}, y_L: {y_L.shape}")
print(f"X_U: {X_U.shape}, y_U: {y_U.shape}")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"X_small: {X_small.shape}")

if params["run_console"]["timevae"] == True:
    timevae_file_name = f"X.npz"
    timevae_folder    = f"{interim_data_loc}/timevae/{desired_dataset}_frac{label_frac}"
    os.makedirs(timevae_folder, exist_ok=True)
    np.savez_compressed(f"{timevae_folder}/{timevae_file_name}", data=np.array(X_small, dtype=np.float32))
    print(f"Saved {timevae_file_name} to {timevae_folder}")
    timevae_file_path = f"{timevae_folder}/{timevae_file_name}"

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, X_small (TimeVAE): {X_small.shape}")
print(f"y_train: {y_train_scaled.shape}, y_test: {y_test_scaled.shape}, y_small (TimeVAE): {y_small.shape}")
print(f"Mean: {np.mean(X_train):.2f}, {np.mean(X_test):.2f}, {y_train_scaled.mean():.2f}, {y_test_scaled.mean():.2f}")
