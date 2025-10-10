from typing import Union, Generator, Tuple
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import torch
from torch import nn
import torch.nn.functional as F

from scipy.interpolate import PchipInterpolator
from scipy.linalg import sqrtm
from scipy.stats import kstest, wasserstein_distance as wasserstein
from skdim.id import MLE
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import acf
from tslearn.metrics import dtw

import yaml

def read_params(file_path: str) -> dict:
    """Read parameters from a YAML file."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def get_frechet_distance(array1: np.ndarray, array2: np.ndarray) -> float:
    """Compute the Fréchet Inception Distance (FID) between 2 arrays
    - array1: np.ndarray of shape (N, D)
    - array2: np.ndarray of shape (M, D)
    - fid (float): lower is better (= closer distributions)"""

    mu1, sigma1 = np.mean(array1, axis=0), np.cov(array1, rowvar=False)
    mu2, sigma2 = np.mean(array2, axis=0), np.cov(array2, rowvar=False)
    diff    = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)

def get_zscore_of_1D_or_2D_array(original_data: np.ndarray, generated_data: np.ndarray, num_samples: int = 1000):
    """Performs one-sample z-test comparing generated to original data. Assumes dependent samples
    Parameters:
    - original_data (1D or 2D): baseline real dataset
    - generated_data (1D or 2D): generated dataset
    - num_samples (int): number of generated samples per test iteration    
    Returns:
    - z_mean: average z-score over iterations
    - z_std: std dev of z-scores"""

    num_iterations     = 100
    original_data_mean = original_data.mean()
    original_data_std  = original_data.std()
    z_score_samples    = np.zeros(num_iterations)
    generated_data_flat= generated_data.flatten() if generated_data.ndim == 2 else generated_data

    eps = 1e-8 # threshold to catch near-0 std
    for i in range(num_iterations):
        max_available = len(generated_data_flat)
        sample_size   = min(num_samples, max_available)
        replace       = sample_size > max_available
        samples_array = np.random.choice(generated_data_flat, sample_size, replace=replace)
        mean_samples  = samples_array.mean()
        if original_data_std < eps:
            z_score_sample = 0.0
        else:
            # z_score_sample = (mean_samples - original_data_mean) / (original_data_std / (num_samples**0.5))
            z_score_sample = (mean_samples - original_data_mean) / (original_data_std / np.sqrt(sample_size))
        z_score_samples[i] = z_score_sample
    return z_score_samples.mean(), z_score_samples.std()

def evaluate_and_plot_autoencoder_metrics(X_scaled, X_reconstructed, should_we_plot):
    """Handles the stats evaluation and plotting for the Autoencoder"""

    X_tensor= torch.tensor(X_scaled, dtype=torch.float32)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reconstructed_array= np.array(X_reconstructed)
    n_features         = X_scaled.shape[1]  # Number of features
    ks_stats           = []
    wasserstein_dists  = []
    real_acfs          = []
    generated_acfs     = []
    dtw_distances      = []

    reconstruction_error = Losses.compute_MSE_loss(X_tensor.to(device),
        torch.tensor(reconstructed_array, dtype=torch.float32).to(device))

    for i in range(n_features):
        real_flat     = X_scaled[:, i].flatten()
        reconstr_flat = reconstructed_array[:, i].flatten()

        # Kolg-Smir Test
        ks_statistic, _ = kstest(real_flat, reconstr_flat)  # We only need the statistic
        ks_stats.append(ks_statistic)

        # Wasserstein dist (= how much work to move from 1 distr. to another)
        wasserstein_dist = wasserstein(np.sort(real_flat), np.sort(reconstr_flat))
        wasserstein_dists.append(wasserstein_dist)

        # Autocorr. (only calculate once and store)
        if i == 0:
            real_acfs      = acf(X_scaled[:, i], nlags=20)
            generated_acfs = acf(reconstructed_array[:, 0], nlags=20)

        # DTW
        dtw_distance = dtw(X_scaled[:, i], reconstructed_array[:, i])
        dtw_distances.append(dtw_distance)

    # Print the results, calculating the averages here
    print(f"Reconst. error: {reconstruction_error:.4f}")
    print(f"Avg Kolg-Smir Statistic: {np.mean(ks_stats):.4f}")
    print(f"Avg Wasserstein Distance: {np.mean(wasserstein_dists):.4f}")
    print(f"Real ACF (first 5 lags): {real_acfs[:5]}")
    print(f"Generated ACF (first 5 lags): {generated_acfs[:5]}")
    print(f"Avg DTW Distance: {np.mean(dtw_distances):.2f}")

    if should_we_plot == True:
        # Plotting the metrics
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        # KS Statistic per feature
        axs[0, 0].bar(range(n_features), ks_stats)
        axs[0, 0].set_title('Kolmogorov-Smirnov Statistic per Feature')
        axs[0, 0].set_xlabel('Feature')
        axs[0, 0].set_ylabel('KS Statistic')

        # Wasserstein Distance per feature
        axs[0, 1].bar(range(n_features), wasserstein_dists, color='orange')
        axs[0, 1].set_title('Wasserstein Distance per Feature')
        axs[0, 1].set_xlabel('Feature')
        axs[0, 1].set_ylabel('Distance')

        # DTW Distance per feature
        axs[1, 0].bar(range(n_features), dtw_distances, color='green')
        axs[1, 0].set_title('DTW Distance per Feature')
        axs[1, 0].set_xlabel('Feature')
        axs[1, 0].set_ylabel('DTW')

        # ACF comparison (first feature only)
        lags = np.arange(len(real_acfs))
        axs[1, 1].plot(lags, real_acfs, label='Real', marker='o')
        axs[1, 1].plot(lags, generated_acfs, label='Reconstructed', marker='x')
        axs[1, 1].set_title('Autocorrelation (Feature 0)')
        axs[1, 1].set_xlabel('Lag')
        axs[1, 1].set_ylabel('ACF')
        axs[1, 1].legend()

        plt.tight_layout()
        plt.show()

    return ks_stats, wasserstein_dists, real_acfs, generated_acfs, dtw_distances

class AutocorrMetrics:
    @staticmethod
    def autocorr(x, lag):
        x = x - x.mean()
        return torch.sum(x[:-lag] * x[lag:]) / torch.sum(x * x) if lag < len(x) else torch.tensor(0.0)

    @staticmethod
    def autocorr_diff(x1, x2, lag):
        diff = 0.0
        for i in range(x1.shape[1]):  # over features/variables
            ac1 = AutocorrMetrics.autocorr(x1[:, i], lag)
            ac2 = AutocorrMetrics.autocorr(x2[:, i], lag)
            diff += torch.abs(ac1 - ac2)
        return diff / x1.shape[1]


class Losses:
    @staticmethod
    def compute_MSE_loss(x_input: torch.Tensor, x_reconstructed: torch.Tensor) -> torch.Tensor:
        """Computes loss (MSE) between input and reconstructed output
        - x_in (torch.Tensor): Original input
        - x_out (torch.Tensor): Reconstructed input
        - torch.Tensor: Scalar loss value"""
        return F.mse_loss(x_reconstructed, x_input)

    @staticmethod
    def compute_MAE_loss(x_input: torch.Tensor, x_reconstructed: torch.Tensor) -> torch.Tensor:
        """Computes loss (MAE) between input and reconstructed output
        - x_in (torch.Tensor): Original input
        - x_out (torch.Tensor): Reconstructed input
        - torch.Tensor: Scalar loss value"""
        return F.l1_loss(x_reconstructed, x_input)

    @staticmethod
    def get_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
        """Compute NT-Xent contrastive loss between two batches of embeddings z1 and z2.
            Each row in z1/z2 is an augmented view of the same sample
            temperature (0.05-0.5), lower = sharper softmax, higher = smoother"""

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        N  = z1.size(0)

        z   = torch.cat([z1, z2], dim=0)  # [2N, D]
        sim = torch.matmul(z, z.T) / temperature  # [2N, 2N]
        sim_exp = torch.exp(sim)

        # Mask self-similarity
        mask    = ~torch.eye(2 * N, device=z.device).bool()
        sim_exp = sim_exp.masked_fill(~mask, 0)

        # Positive pairs: i with i+N and vice versa
        pos_sim = torch.exp(torch.sum(z1 * z2, dim=-1) / temperature)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)  # [2N]

        denom = sim_exp.sum(dim=1)  # [2N]
        loss  = -torch.log(pos_sim / denom)
        return loss.mean()


class DimensionalityEstimator:
    @staticmethod
    def estimate_dataset_dimensionality(dataset, neighbors: int = 10) -> int:
        """Estimate intrinsic dimensionality using scikit-dimension (ie best latent size). k is usually log(n_rows) to n/2
        - input: dataset (pd or pl df, or np array)
        - k: Number of neighbors for MLE estimator (default: 10).
        - output: estimated dataset dimensionality"""
        if isinstance(dataset, pd.DataFrame):
            dataset= dataset.select_dtypes(include=[np.number])
            X      = dataset.to_numpy()
        elif isinstance(dataset, pl.DataFrame):
            dataset= dataset.select(pl.col(pl.NUMERIC_DTYPES))
            X      = dataset.to_numpy()
        elif isinstance(dataset, (np.ndarray,)):
            X      = dataset
        elif isinstance(dataset, torch.Tensor):
            X      = dataset.numpy()
        else:
            raise TypeError("Unsupported dataset type")

        X        = X.astype(np.float32)
        estimator= MLE()
        estimator.set_params(K = neighbors)
        dim = estimator.fit_transform(X)
        return int(dim)

    @staticmethod
    def pca_components_explaining_variance(dataset, var: float = 0.95):
        """Return # of PCA components explaining given variance (default 95%)"""
        pca               = PCA().fit(dataset)
        explained_variance= pca.explained_variance_ratio_
        cum_var           = np.cumsum(explained_variance)
        n_components      = np.searchsorted(cum_var, var) + 1
        return n_components


class ForecastUtils:
    @staticmethod
    def infer_dominant_freq(df: pd.DataFrame, time_col: str) -> str:
        """Return the most common time difference as pandas freq string."""
        sorted_df = df.sort_values(time_col)
        td        = sorted_df[time_col].diff().dropna()
        dominant_delta = td.mode()[0]
        return pd.tseries.frequencies.to_offset(dominant_delta).freqstr

    @staticmethod
    def apply_pchip_interpolation(df: pd.DataFrame, time_col: str, freq: str) -> pd.DataFrame:
        """Apply PCHIP interpolation to numeric columns on uniform time grid
        If freq < original median spacing, will create more timestamps than the original series"""
        df            = df.copy()
        df[time_col]  = pd.to_datetime(df[time_col])
        df            = df.set_index(time_col).sort_index()
        uniform_index = pd.date_range(df.index.min(), df.index.max(), freq=freq)
        x             = (df.index - df.index[0]).total_seconds()
        x_new         = (uniform_index - df.index[0]).total_seconds()
        resampled     = pd.DataFrame(index=uniform_index)
        for col in df.select_dtypes(include="number").columns:
            y             = df[col].values
            interp        = PchipInterpolator(x, y, extrapolate=False)
            resampled[col]= interp(x_new)
        return resampled.reset_index().rename(columns={"index": time_col})

    @staticmethod
    def compute_horizon(series_length: int, fixed_points: int, pct: float) -> int:
        """Compute forecast horizon as min of fixed points or percentage of series."""
        num_steps = max(fixed_points, math.ceil(series_length * pct))
        print(f"Horizon = {num_steps} steps (fixed_points={fixed_points}, pct={pct})")
        return num_steps

    @staticmethod
    def make_windows(df: pd.DataFrame, n_windows = 10, horizon_len=None, horizon_frac=0.01, min_window_len = 300, window_frac = None, start_point= 0):
        """Create rolling-origin windows for a multivariate df.
        n_windows: Number of windows to generate.
        horizon_len: Fixed horizon length (overrides horizon_frac if set).
        horizon_frac: Fraction of total series length to use as horizon if horizon_len is None.
        min_window_len: Minimum training length (TimeGPT constraint). If None, defaults to max(horizon_len, TIMEGPT_MIN_INPUT).
        window_frac: Fraction of series length to use as maximum training length.
        start_point: Starting index for the first window.
        windows_list: list of tuples (train_df, test_df)"""
        N = len(df)
        TIMEGPT_MIN_INPUT = 1008  # TimeGPT minimum input size

        if horizon_len is None:
            horizon_len = max(1, int(N * horizon_frac))
        print(f"Horizon = {horizon_len} steps")

        min_window_len = min_window_len if min_window_len is not None else max(horizon_len, TIMEGPT_MIN_INPUT)
        max_window_len = int(N * window_frac) if window_frac is not None else N - horizon_len

        if max_window_len < min_window_len:
            raise ValueError(f"Series too short: max train length {max_window_len} < min train length {min_window_len}")

        window_sizes = np.linspace(min_window_len, max_window_len, n_windows, dtype=int)
        windows_list = []

        for w in window_sizes:
            train_df = df.iloc[start_point:w].copy()
            test_df  = df.iloc[w:w + horizon_len].copy()
            if len(test_df) == horizon_len:
                windows_list.append((train_df, test_df))
        return windows_list
    
def make_sample_splits(X: np.ndarray, y: np.ndarray, method: str = "holdout", train_ratio: float = 0.8, n_splits: int = 5,
                       random_state: int = None) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
    """Make random train/test splits across samples (pages). Each page is a separate time series. Methods:
    - 'holdout': first train_ratio of pages for training, rest for test
    - 'blocked': non-overlapping contiguous folds along pages"""
    n_pages = X.shape[0]
    rng     = np.random.default_rng(random_state)
    indices = rng.permutation(n_pages) 

    if method == "holdout":
        split_idx = int(n_pages * train_ratio)
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]
        X_train, X_test     = X[train_idx], X[test_idx]
        y_train, y_test     = y[train_idx], y[test_idx]
        yield X_train, X_test, y_train, y_test
    elif method == "kfold":
        fold_size = n_pages // n_splits
        for split_i in range(n_splits):
            test_idx        = indices[split_i * fold_size : (split_i + 1) * fold_size]
            train_idx       = np.concatenate([indices[:split_i * fold_size], indices[(split_i + 1) * fold_size:]])
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            yield X_train, X_test, y_train, y_test
    else:
        raise ValueError("method must be 1 of {'holdout', 'kfold'}")

