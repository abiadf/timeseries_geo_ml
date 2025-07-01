
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import torch
from torch import nn
import torch.nn.functional as F

from scipy.linalg import sqrtm
from scipy.stats import kstest, wasserstein_distance as wasserstein
from skdim.id import MLE
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import acf
from tslearn.metrics import dtw


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


class DimensionalityEstimator:
    @staticmethod
    def estimate_dataset_dimensionality(dataset):
        """Estimate intrinsic dimensionality using scikit-dimension (ie best latent size)
        input: dataset (pd or pl df, or np array)
        output: estimated dataset dimensionality"""
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.select_dtypes(include=[np.number])
            X = dataset.to_numpy()
        elif isinstance(dataset, pl.DataFrame):
            dataset = dataset.select(pl.col(pl.NUMERIC_DTYPES))
            X = dataset.to_numpy()
        elif isinstance(dataset, (np.ndarray,)):
            X = dataset
        else:
            raise TypeError("Unsupported dataset type")
        return MLE().fit(X).dimension_

    @staticmethod
    def pca_components_explaining_variance(dataset, var: float = 0.95):
        """Return # of PCA components explaining given variance (default 95%)"""
        pca               = PCA().fit(dataset)
        explained_variance= pca.explained_variance_ratio_
        cum_var           = np.cumsum(explained_variance)
        n_components      = np.searchsorted(cum_var, var) + 1
        return n_components


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
