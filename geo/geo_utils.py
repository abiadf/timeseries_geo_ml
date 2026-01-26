import __main__
import sys, os
project_root = os.path.abspath("..")  # adjust if notebook is elsewhere
sys.path.insert(0, project_root)
from typing import Dict, List, Literal, Tuple, Optional, Any, Union
import logging
from datetime import datetime

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
from geo_encoders import fit_catboost_multi, MLPPredHead

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.info("Starting process...")
logging.warning("Something looks off...")
logging.error("Something failed.")

def append_run_to_csv(params: Dict[str, Any], metrics: Dict[str, float], file_path: str, method: str) -> None:
    """Append one experiment run (date + method + metrics + params) to CSV."""
    row = {
        "time": datetime.now().strftime("%H:%M"),
        "method": method,
        **metrics,
        **params,}

    df = pd.DataFrame([row])
    write_header = not os.path.exists(file_path) or os.path.getsize(file_path) == 0
    df.to_csv(file_path, mode="a", index=False, header=write_header)
    print("💾 Appended run to", file_path)


def read_csv_and_skip_rows_then_save(file_loc: str, file_name: str, rows_to_skip: int, changed_file_name: str):
    "Skip rows = stride to make smaller. rows_to_skip: 0 (no skipping), 1 (skip every other), 2 (skip 2 out of 3), etc."
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
        print(f"{100*len(cyc_cols)/len(dataset.columns):.2f}% cyclic/{100*len(lin_cols)/len(dataset.columns):.2f}% linear cols")
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



class Predictors:
    @staticmethod
    def make_random_latent_prediction(Z_train, y_train, p, sliding_size, n_samples=1000):
        """Evaluate a random latent baseline by sampling Z from the training set.
        This preserves the exact empirical distribution and shape of Z_train,
        then predicts y using a CatBoost model trained on (Z_train, y_train_windows).
        Args:
            Z_train: torch.Tensor of latent vectors (N, D).
            y_train: target time series (torch.Tensor or np.array).
            p: config object with window_size and task attributes.
            n_samples: number of random latent samples to evaluate.
        Returns:
            rmse, r2: baseline performance of random latent samples."""
        Z_train_np = Z_train.detach().cpu().numpy()
        y_train_win = Windowing.make_windows_from_y(y_train, p.window_size, sliding_size, task=p.task).reshape(-1, 1)

        idx = np.random.randint(0, Z_train_np.shape[0], size=n_samples)
        Z_random_np = Z_train_np[idx]

        y_hat_random = fit_catboost_multi(Z_train_np, y_train_win, Z_random_np)
        y_dummy = np.tile(y_train_win.mean(axis=0), (n_samples, 1))
        rmse = np.sqrt(mean_squared_error(y_dummy, y_hat_random))
        r2 = r2_score(y_dummy, y_hat_random)
        return rmse, r2

    @staticmethod
    def make_latent_pca_prediction(Z_train: torch.Tensor, Z_test: torch.Tensor, y_train: np.ndarray, y_test: np.ndarray,
                                   p, n_components: float = 0.95, sliding_size=None) -> tuple[float, float]:
        """Apply PCA to latents and predict with CatBoost."""
        # window y to match latents
        y_train_win = Windowing.make_windows_from_y(y_train, p.window_size, sliding_size, task=p.task, horizon=p.horizon)
        y_test_win  = Windowing.make_windows_from_y(y_test,  p.window_size, sliding_size, task=p.task, horizon=p.horizon)

        # convert latents to numpy
        Z_train_np = Z_train.cpu().numpy()
        Z_test_np  = Z_test.cpu().numpy()

        # PCA fit on training latents
        pca = PCA(n_components=n_components)
        Z_train_pca = pca.fit_transform(Z_train_np)
        Z_test_pca  = pca.transform(Z_test_np)

        # regression
        model = LinearRegression()
        model.fit(Z_train_pca, y_train_win)
        y_hat = model.predict(Z_test_pca)

        # y_hat = fit_catboost_multi(Z_train_pca, y_train_win, Z_test_pca)
        rmse  = np.sqrt(mean_squared_error(y_test_win, y_hat))
        r2    = r2_score(y_test_win, y_hat)
        return rmse, r2

    @staticmethod
    def make_latent_umap_catboost(Z_train: torch.Tensor, Z_test: torch.Tensor, y_train: np.ndarray, y_test: np.ndarray,
                                p, n_components: int = 5,  sliding_size=None, n_neighbors: int = 15, min_dist: float = 0.1, random_state: int = 42,
                                catboost_params: dict = None) -> tuple[float, float]:
        """Apply UMAP to latents and predict with CatBoost regression."""
        # window y to match latents
        y_train_win = Windowing.make_windows_from_y(y_train, p.window_size, sliding_size, task=p.task, horizon=p.horizon)
        y_test_win  = Windowing.make_windows_from_y(y_test,  p.window_size, sliding_size, task=p.task, horizon=p.horizon)

        # convert latents to numpy
        Z_train_np = Z_train.cpu().numpy()
        Z_test_np  = Z_test.cpu().numpy()

        # scale before UMAP
        scaler = StandardScaler()
        Z_train_scaled = scaler.fit_transform(Z_train_np)
        Z_test_scaled  = scaler.transform(Z_test_np)

        # UMAP
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
        Z_train_umap = reducer.fit_transform(Z_train_scaled)
        Z_test_umap  = reducer.transform(Z_test_scaled)

        y_hat = fit_catboost_multi(Z_train_umap, y_train_win, Z_test_umap)
        rmse = np.sqrt(mean_squared_error(y_test_win, y_hat))
        r2   = r2_score(y_test_win, y_hat)
        return rmse, r2

    @staticmethod
    def run_riemann_catboost(Z_train, Z_test, y_train, y_test, window_size: int, sliding_size: int, prediction_task, eps: float = 1e-6):
        """Riemannian covariance → tangent space (PGA) → scaling → CatBoost → RMSE/R2."""
        # torch → numpy
        Z_train_np = Z_train.cpu().numpy() if Z_train.is_cuda else Z_train.numpy()
        Z_test_np  = Z_test.cpu().numpy()  if Z_test.is_cuda else Z_test.numpy()

        # covariance
        X_train_cov = compute_covariance_safe(Z_train_np)
        X_test_cov  = compute_covariance_safe(Z_test_np)

        # regularization
        n = X_train_cov.shape[1]
        X_train_cov += eps * np.eye(n)
        X_test_cov  += eps * np.eye(n)

        # tangent space (PGA)
        ts = TangentSpace(metric='riemann')
        Z_train_pga = ts.fit_transform(X_train_cov)
        Z_test_pga  = ts.transform(X_test_cov)

        # scale features
        Z_train_scaled, Z_test_scaled = scale_train_and_test_sets(Z_train_pga, Z_test_pga)

        # window + scale targets
        y_train_w = Windowing.make_windows_from_y(y_train, window_size, sliding_size, task=prediction_task, horizon=p.horizon)
        y_test_w  = Windowing.make_windows_from_y(y_test, window_size, sliding_size, task=prediction_task, horizon=p.horizon)
        y_train_scaled, y_test_scaled = scale_train_and_test_sets(y_train_w, y_test_w)

        # train + predict
        y_hat = fit_catboost_multi(Z_train_scaled, y_train_scaled, Z_test_scaled)

        # metrics
        rmse = np.sqrt(mean_squared_error(y_test_scaled, y_hat))
        r2   = r2_score(y_test_scaled, y_hat)
        return y_hat, rmse, r2

    @staticmethod
    def cosine_similarity_samples(z: torch.Tensor) -> torch.Tensor:
        """Mean pairwise cosine similarity between samples. z: (N, D), assumed normalized."""
        z = torch.nn.functional.normalize(z, dim=1)
        sim = z @ z.T
        N = z.shape[0]
        return (sim.sum() - N) / (N * (N - 1))

    @staticmethod
    def angular_variance_over_time(z: torch.Tensor) -> torch.Tensor:
        """Angular variance across time. z: (T, D), normalized."""
        z = torch.nn.functional.normalize(z, dim=1)
        mean_dir = torch.mean(z, dim=0)
        mean_dir = mean_dir / mean_dir.norm()
        cos_angles = (z @ mean_dir).clamp(-1, 1)
        angles = torch.acos(cos_angles)
        return angles.var()

    @staticmethod
    def make_direct_prediction(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                               sliding_size: int, prediction_task: str, p) -> tuple[float, float, float]:
        """Direct CatBoost baseline with task-consistent windowing."""
        print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

        if prediction_task == "tabular":
            X_train_flat, X_test_flat = X_train, X_test
            y_train_w, y_test_w       = y_train, y_test
        else:
            # --- window X ---
            y_train_w = Windowing.make_windows_from_y(y_train, p.window_size, sliding_size, task=prediction_task, horizon=p.horizon)
            y_test_w  = Windowing.make_windows_from_y(y_test, p.window_size, sliding_size, task=prediction_task, horizon=p.horizon)
            X_train_w = Windowing.make_windows_from_X(torch.from_numpy(X_train.to_numpy()).float(), p.window_size, sliding_size, horizon=p.horizon)
            X_test_w  = Windowing.make_windows_from_X(torch.from_numpy(X_test.to_numpy()).float(), p.window_size, sliding_size, horizon=p.horizon)

            X_train_flat = X_train_w.reshape(X_train_w.shape[0], -1).numpy()
            X_test_flat  = X_test_w.reshape(X_test_w.shape[0], -1).numpy()

        # --- scale ---
        X_train_flat, X_test_flat = scale_train_and_test_sets(X_train_flat, X_test_flat)
        y_train_prep = y_train_w.reshape(y_train_w.shape[0], -1)
        y_test_prep  = y_test_w.reshape(y_test_w.shape[0], -1)
        # y_train_prep = y_train_w.reshape(-1, 1) if y_train_w.ndim == 1 else y_train_w
        # y_test_prep  = y_test_w.reshape(-1, 1) if y_test_w.ndim == 1 else y_test_w
        y_train_scaled, y_test_scaled = scale_train_and_test_sets(y_train_prep, y_test_prep)

        assert y_train_prep.ndim == 2
        assert y_train_prep.shape[0] == X_train_flat.shape[0]

        # y_hat = LinearRegression().fit(X_train_flat, y_train_scaled).predict(X_test_flat)
        y_hat = fit_catboost_multi(X_train_flat, y_train_scaled, X_test_flat, cb_verbose=50)
        rmse  = root_mean_squared_error(y_test_scaled, y_hat)
        r2    = r2_score(y_test_scaled, y_hat)
        mae   = mean_absolute_error(y_test_scaled, y_hat)
        return rmse, r2, mae

    @staticmethod
    def make_direct_prediction_mlp(X_train, X_test, y_train, y_test, p, sliding_size, device="cuda"):
        # 1. Scaling (Crucial for Neural Nets)
        y_tr_s, y_te_s = scale_train_and_test_sets(y_train, y_test)
        X_tr_s, X_te_s = scale_train_and_test_sets(X_train, X_test)

        # 2. Windowing
        # Ensure we have Tensors for the windowing logic
        def to_tensor(data):
            if torch.is_tensor(data): return data.float()
            if hasattr(data, 'values'): return torch.from_numpy(data.values).float()
            return torch.from_numpy(data).float()

        X_tr_tensor = to_tensor(X_tr_s)
        X_te_tensor = to_tensor(X_te_s)

        X_tr_w = Windowing.make_windows_from_X(X_tr_tensor, p.window_size, sliding_size, horizon=p.horizon)
        X_te_w = Windowing.make_windows_from_X(X_te_tensor, p.window_size, sliding_size, horizon=p.horizon)
        
        y_tr_w = Windowing.make_windows_from_y(y_tr_s, p.window_size, sliding_size, task=p.task, horizon=p.horizon)
        y_te_w = Windowing.make_windows_from_y(y_te_s, p.window_size, sliding_size, task=p.task, horizon=p.horizon)

        # Flatten windows into vectors for the MLP
        X_tr_flat = X_tr_w.reshape(X_tr_w.size(0), -1)
        X_te_flat = X_te_w.reshape(X_te_w.size(0), -1)
        y_tr_flat = torch.tensor(y_tr_w.reshape(y_tr_w.shape[0], -1), dtype=torch.float32)
        y_te_flat = torch.tensor(y_te_w.reshape(y_te_w.shape[0], -1), dtype=torch.float32)

        # 3. Training Setup
        loader = DataLoader(TensorDataset(X_tr_flat, y_tr_flat), batch_size=p.batch_size, shuffle=True)
        # Hidden dim is doubled to handle the high-dimensional flattened window
        direct_head = MLPPredHead(X_tr_flat.shape[1], y_tr_flat.shape[1], hidden_dim=p.hidden_dim * 2).to(device)
        optimizer = torch.optim.AdamW(direct_head.parameters(), lr=p.lr_optimizer)

        direct_head.train()
        for epoch in range(p.epochs):
            epoch_loss = 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                y_hat = direct_head(xb)
                loss = F.mse_loss(y_hat, yb)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"Direct MLP Epoch {epoch}: Loss {epoch_loss/len(loader):.4f}")

        # 4. Evaluation
        direct_head.eval()
        with torch.no_grad():
            y_hat_final = direct_head(X_te_flat.to(device)).cpu().numpy()
        
        y_te_true = y_te_flat.numpy()
        rmse = root_mean_squared_error(y_te_true, y_hat_final)
        r2   = r2_score(y_te_true, y_hat_final)
        mae  = mean_absolute_error(y_te_true, y_hat_final)
        
        return rmse, r2, mae

    @staticmethod
    def make_direct_prediction_pca(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                                window_size: int, sliding_size: int, prediction_task: str, n_components: int = 0.95) -> tuple[float, float]:
        """Direct CatBoost baseline with optional PCA for dimensionality reduction."""

        if prediction_task == "tabular":
            X_train_flat, X_test_flat = X_train, X_test
            y_train_w, y_test_w       = y_train, y_test
        else:
            y_train_w = Windowing.make_windows_from_y(y_train, window_size, sliding_size, task=prediction_task, horizon=p.horizon)
            y_test_w  = Windowing.make_windows_from_y(y_test, window_size, sliding_size, task=prediction_task, horizon=p.horizon)

            X_train_w = Windowing.make_windows_from_X(torch.from_numpy(X_train.to_numpy()).float(), window_size, sliding_size, horizon=p.horizon)
            X_test_w  = Windowing.make_windows_from_X(torch.from_numpy(X_test.to_numpy()).float(), window_size, sliding_size, horizon=p.horizon)
            X_train_flat = X_train_w.reshape(X_train_w.shape[0], -1).numpy()
            X_test_flat  = X_test_w.reshape(X_test_w.shape[0], -1).numpy()

        # --- PCA ---
        pca = PCA(n_components=n_components)
        X_train_flat = pca.fit_transform(X_train_flat)
        X_test_flat  = pca.transform(X_test_flat)

        # --- scale ---
        X_train_flat, X_test_flat = scale_train_and_test_sets(X_train_flat, X_test_flat)

        y_train_prep = y_train_w.reshape(-1, 1) if y_train_w.ndim == 1 else y_train_w
        y_test_prep  = y_test_w.reshape(-1, 1) if y_test_w.ndim == 1 else y_test_w
        y_train_scaled, y_test_scaled = scale_train_and_test_sets(y_train_prep, y_test_prep)

        # --- regression ---
        y_hat = fit_catboost_multi(X_train_flat, y_train_scaled, X_test_flat)
        rmse  = root_mean_squared_error(y_test_scaled, y_hat)
        r2    = r2_score(y_test_scaled, y_hat)
        return rmse, r2

    @staticmethod
    def compute_covariance_safe(X, eps=1e-6):
        """
        Compute regularized covariance matrices for latent features.
        Handles 1D, 2D, or 3D inputs.
        Returns: [n_samples, n_channels, n_channels]
        """
        covs = []
        for x in X:
            # convert to 2D (samples × features)
            if x.ndim == 0:
                x_flat = x.reshape(1, 1)
            elif x.ndim == 1:
                x_flat = x.reshape(1, -1)
            elif x.ndim == 2:
                x_flat = x
            else:  # >2D
                x_flat = x.reshape(x.shape[0], -1)

            if x_flat.shape[0] == 1:
                # only one sample → covariance is outer product
                C = np.outer(x_flat[0], x_flat[0])
            else:
                C = np.cov(x_flat, rowvar=False)

            # regularize to make positive definite
            C += eps * np.eye(C.shape[0])
            covs.append(C)

        return np.array(covs)

    @staticmethod
    def make_direct_prediction_cov_pca(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                                    window_size: int, sliding_size: int, prediction_task: str, n_components: int = 0.95,
                                    method: str = "PCA") -> tuple[float, float]:
        """Direct CatBoost baseline with PCA or PGA on covariance matrices for fair comparison."""
        
        if prediction_task == "tabular":
            raise ValueError("Covariance-based PCA/PGA requires time-windowed data")
        
        # --- create windows ---
        y_train_w = Windowing.make_windows_from_y(y_train, window_size, sliding_size, task=prediction_task, horizon=p.horizon)
        y_test_w  = Windowing.make_windows_from_y(y_test, window_size, sliding_size, task=prediction_task, horizon=p.horizon)
        
        X_train_w = Windowing.make_windows_from_X(torch.from_numpy(X_train.to_numpy()).float(), window_size, sliding_size, horizon=p.horizon)
        X_test_w  = Windowing.make_windows_from_X(torch.from_numpy(X_test.to_numpy()).float(), window_size, sliding_size, horizon=p.horizon)
        
        # Ensure numpy
        X_train_w = X_train_w.numpy() if isinstance(X_train_w, torch.Tensor) else X_train_w
        X_test_w  = X_test_w.numpy() if isinstance(X_test_w, torch.Tensor) else X_test_w

        # --- compute covariance matrices ---
        X_train_cov = compute_covariance_safe(X_train_w)
        X_test_cov  = compute_covariance_safe(X_test_w)

        # --- dimensionality reduction ---
        if method.upper() == "PCA":
            # Flatten covariance matrices and apply linear PCA
            n_samples, n_channels, _ = X_train_cov.shape
            X_train_flat = X_train_cov.reshape(n_samples, -1)
            X_test_flat  = X_test_cov.reshape(X_test_cov.shape[0], -1)
            reducer = PCA(n_components=n_components)
            X_train_flat = reducer.fit_transform(X_train_flat)
            X_test_flat  = reducer.transform(X_test_flat)
        elif method.upper() == "PGA":
            ts = TangentSpace(metric='riemann')
            X_train_flat = ts.fit_transform(X_train_cov)
            X_test_flat  = ts.transform(X_test_cov)
        else:
            raise ValueError("method must be 'PCA' or 'PGA'")

        # --- scale ---
        X_train_flat, X_test_flat = scale_train_and_test_sets(X_train_flat, X_test_flat)

        y_train_prep = y_train_w.reshape(-1, 1) if y_train_w.ndim == 1 else y_train_w
        y_test_prep  = y_test_w.reshape(-1, 1) if y_test_w.ndim == 1 else y_test_w
        y_train_scaled, y_test_scaled = scale_train_and_test_sets(y_train_prep, y_test_prep)

        # --- regression ---
        y_hat = fit_catboost_multi(X_train_flat, y_train_scaled, X_test_flat)
        rmse  = root_mean_squared_error(y_test_scaled, y_hat)
        r2    = r2_score(y_test_scaled, y_hat)
        return rmse, r2

