from typing import Union, Generator, Tuple, Optional, List
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn.functional as F
from catboost import CatBoostRegressor, MetricVisualizer

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

from scipy.interpolate import PchipInterpolator
from scipy.linalg import sqrtm
from scipy.stats import kstest, wasserstein_distance as wasserstein
from skdim.id import MLE
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

    @staticmethod
    def compute_byol_loss(p_online: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        """Minimal BYOL loss: MSE between online predictions and target projections, adapted from the BYOL paper.
        Args:
            p_online: prediction from online network (batch, dim)
            z_target: projection from target network (batch, dim)
        Returns: Scalar loss"""
        # normalize for stability (optional but standard)
        p_online = F.normalize(p_online, dim=1)
        z_target = F.normalize(z_target, dim=1)

        z_target = z_target.detach() # stop gradients on target
        return 2 - 2 * (p_online * z_target).sum(dim=1).mean() # loss = MSE = 2 - 2 * cosine_sim



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

    @staticmethod
    def latent_pruner(z_train, threshold_frac: float = 0.05):
        """Source: "Auto-encoder based dimensionality reduction"
        Remove latent dimensions with variance < threshold_frac * max_variance
        z_array is 2d
        NOTE: use on train set only (not test set or combined set)"""
        latent_var = np.var(z_train, axis=0)
        for i, v in enumerate(latent_var):
            print(f"Latent dim {i}: variance = {v:.5f}")

        threshold        = threshold_frac * latent_var.max()
        active_dims_mask = latent_var > threshold
        print(f"Pruning latent dims with variance < {threshold:.2f}, kept {np.sum(active_dims_mask)}/{len(latent_var)} dims")
        return active_dims_mask



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


class Preds:
    "Class of predictors to predict y from X"
    def __init__(self):
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_str = "GPU" if self.device.type == "cuda" else "CPU"
        print(f"Using device: {self.device} ({self.device_str})")

    @staticmethod
    def predict_linreg(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Train linear predictor"""
        model  = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return root_mean_squared_error(y_test, y_pred)

    def predict_catboost_multioutput(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Optional[MultiOutputRegressor], np.ndarray, float, List[int]]:
        """Train multi-output CatBoost models and predict test set.
        Returns:
            model: trained MultiOutputRegressor (or None if all targets constant)
            y_pred: predictions on test set
            rmse: RMSE across all targets"""
        # flatten if X is 3D
        # if X_train.ndim == 3:
        #     X_train = X_train.reshape(X_train.shape[0], -1)
        #     X_test  = X_test.reshape(X_test.shape[0], -1)

        y_pred           = np.zeros_like(y_test, dtype=float)
        non_constant_idx = [i for i in range(y_train.shape[1])
                            if not np.all(y_train[:, i] == y_train[0, i])]
        if non_constant_idx:
            if self.device_str == "GPU":
                cb_params = dict(iterations=200, learning_rate=0.1, depth=4,
                                 task_type="GPU", devices='0', verbose=100, early_stopping_rounds=50)
            else:
                cb_params = dict(iterations=200, learning_rate=0.1, depth=4,
                                 thread_count=-1, verbose=100, early_stopping_rounds=50)
            model = MultiOutputRegressor(CatBoostRegressor(**cb_params))
            model.fit(X_train, y_train[:, non_constant_idx])
            # model.fit(X_train, y_train, eval_set=(X_val, y_val), )
            y_pred[:, non_constant_idx] = model.predict(X_test)
            for i in range(y_train.shape[1]):
                if np.all(y_train[:, i] == y_train[0, i]):
                    y_pred[:, i] = y_train[0, i]
        else:
            model = None
        rmse = root_mean_squared_error(y_test, y_pred)
        return model, y_pred, rmse, non_constant_idx

    @staticmethod
    def cluster_and_label(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                        y_test: np.ndarray, n_clusters: int = 10, random_state: int = 42) -> float:
        """Cluster train features with KMeans, assign representative y (mean per cluster), predict test labels by cluster assignment,
        and compute RMSE. this is considered unsupervised, as the clustering happens to X only
        - X_train: Training features (N_train, D).
        - y_train: Training targets (N_train, T).
        - X_test: Test features (N_test, D).
        - y_test: Test targets (N_test, T).
        - n_clusters: Number of KMeans clusters.
        - random_state: Random seed for reproducibility.
        Returns: Mean squared error on test set."""
        try:
            kmeans         = KMeans(n_clusters=n_clusters, random_state=random_state)
            train_clusters = kmeans.fit_predict(X_train)

            # mean target vector per cluster
            y_cluster     = {cluster: y_train[train_clusters == cluster].mean(axis=0)
                            for cluster in range(n_clusters)}
            test_clusters = kmeans.predict(X_test)
            y_pred        = np.stack([y_cluster[cluster] for cluster in test_clusters], axis=0)
            return root_mean_squared_error(y_test, y_pred)
        except Exception:
            return float('nan')

    def predict_rf_multioutput(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                               y_test: np.ndarray, return_model: bool = False):
        """Train multi-output Random Forest and compute RMSE."""
        # flatten if X is 3D
        if X_train.ndim == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test  = X_test.reshape(X_test.shape[0], -1)

        n_samples   = X_train.shape[0]
        max_depth   = int(np.log2(n_samples))  # reasonable default
        max_depth   = min(16, int(np.log2(n_samples)))  # cap at 16
        max_samples = min(5_000, n_samples)

        model  = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42, max_depth=max_depth, n_jobs=-1,
                                                            bootstrap=True, max_samples=max_samples, max_features="sqrt",
                                                            min_samples_leaf=2, min_samples_split=4))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # return root_mean_squared_error(y_test, y_pred), model
        rmse = root_mean_squared_error(y_test, y_pred)
        if return_model:
            return rmse, model
        return rmse

    @staticmethod
    def predict_elasticnet_multioutput(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                                       alpha: float = 0.1, l1_ratio: float = 0.5) -> Tuple[MultiOutputRegressor, np.ndarray, float]:
        """Train multi-output ElasticNet (L1+L2) and predict test set.
        Returns:
            model: trained MultiOutputRegressor
            y_pred: predictions on test set
            rmse: root mean squared error"""
        model  = MultiOutputRegressor(ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=1000, random_state=42))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse   = root_mean_squared_error(y_test, y_pred)
        return model, y_pred, rmse

    @staticmethod
    def predict_mlp_multioutput(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                                hidden_layer_sizes: Tuple[int, ...] = (128, 64), max_iter: int = 500,
                                random_state: int = 42) -> Tuple[MultiOutputRegressor, np.ndarray, float]:
        """Train multi-output MLPRegressor and predict test set.
        Returns:
            model: trained MultiOutputRegressor
            y_pred: predictions on test set
            rmse: root mean squared error"""
        base_mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                max_iter=max_iter,
                                random_state=random_state)
        model    = MultiOutputRegressor(base_mlp)
        model.fit(X_train, y_train)
        y_pred   = model.predict(X_test)
        rmse     = root_mean_squared_error(y_test, y_pred)
        return model, y_pred, rmse

    def evaluate_models_on_dataset(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate various models on the dataset and print RMSE results."""
        linreg_loss            = self.predict_linreg(X_train, y_train, X_test, y_test)
        print(f"Linear Regression done")
        _, _, catboost_loss, _ = self.predict_catboost_multioutput(X_train, y_train, X_test, y_test)
        print(f"CatBoost done")
        # unsupervised_rmse = Preds.cluster_and_label(X_train, y_train, X_test, y_test, n_clusters=5)
        rf_rmse, rf_model      = self.predict_rf_multioutput(X_train, y_train, X_test, y_test, return_model=True)
        print(f"Random Forest done")
        # _, _, el_rmse     = Preds.predict_elasticnet_multioutput(X_train, y_train, X_test, y_test, alpha=0.1, l1_ratio=0.5)
        return [linreg_loss, catboost_loss, rf_rmse], rf_model # unsupervised_rmse #, el_rmse


# unused class, consider removing
class SemiSupLearning:
    @staticmethod
    def barlow_twins_loss(z_a: torch.Tensor, z_b: torch.Tensor, lambd: float = 0.0051, eps: float = 1e-12):
        """z_a, z_b: (B, D) - embeddings for two views, assumed zero-meaned / normalized per-dim.
        Loss = sum_i (1 - C_ii)^2 + lambda * sum_{i!=j} C_ij^2
        where C is cross-correlation matrix between z_a and z_b (B-normalized)."""
        B, D     = z_a.shape
        z_a_norm = (z_a - z_a.mean(0)) / (z_a.std(0) + eps)
        z_b_norm = (z_b - z_b.mean(0)) / (z_b.std(0) + eps)
        xcorr    = (z_a_norm.T @ z_b_norm) / B
        on_diag  = torch.diagonal(xcorr).add_(-1).pow(2).sum()
        off_diag = (xcorr - torch.diag(torch.diagonal(xcorr))).pow(2).sum()
        return on_diag + lambd * off_diag

    @staticmethod
    def vicreg_loss(z_a: torch.Tensor, z_b: torch.Tensor, sim_coeff=25.0, 
                    var_coeff=25.0, cov_coeff=1.0, eps=1e-4):
        """z_a, z_b : (B, D)
        sim: mean squared error between z_a and z_b
        var: hinge on std per-dim (std should be > threshold)
        cov: off-diagonal terms of covariance matrix"""
        def variance_term(z):
            std      = torch.sqrt(z.var(dim=0) + eps)
            std_loss = torch.mean(F.relu(1.0 - std))
            return std_loss

        def covariance_term(z):
            z        = z - z.mean(dim=0)
            cov      = (z.T @ z) / (B - 1)   # (D, D)
            off_diag = cov - torch.diag(torch.diagonal(cov))
            return (off_diag.pow(2).sum()) / D

        B, D     = z_a.shape
        sim_loss = F.mse_loss(z_a, z_b)
        var_loss = variance_term(z_a) + variance_term(z_b)
        cov_loss = covariance_term(z_a) + covariance_term(z_b)
        return sim_coeff * sim_loss + var_coeff * var_loss + cov_coeff * cov_loss

    # not used anymore
    @staticmethod
    def train_ae_with_ssl(ae_model, X, device="cpu",
                        epochs=10, lr=1e-3, batch_size=128,
                        ssl_mode=None,      # None | 'barlow' | 'vicreg'
                        ssl_weight=1.0,     # weight applied to ssl loss
                        recon_weight=1.0,   # weight applied to reconstruction loss
                        augment_fn=None,    # function that given a torch tensor returns two views
                        print_every=10):
        """ae_model: must implement .forward(x) -> x_recon and .encode(x) -> z (torch modules)
        X: numpy array (N, T, C) or (N, D)
        augment_fn: function(X_tensor, device) -> (view1_tensor, view2_tensor)
        Returns: trained model (in-place)"""
        ae_model.to(device)
        ae_model.train()
        opt  = torch.optim.AdamW(ae_model.parameters(), lr=lr)
        N    = len(X)
        idxs = np.arange(N)

        # data to torch if needed
        X_tensor_all = torch.tensor(X, dtype=torch.float32, device=device)
        # if timeseries reshape handled by model; here we assume shape (B, T, C) or (B, D)

        for epoch in range(epochs):
            np.random.shuffle(idxs)
            for start in range(0, N, batch_size):
                batch_idx = idxs[start:start+batch_size]
                xb        = X_tensor_all[batch_idx]
                # if augment_fn provided and ssl_mode requested
                if ssl_mode is not None and augment_fn is not None:
                    v1, v2 = augment_fn(xb, device)    # expected torch tensors on device
                    z1 = ae_model.encode(v1).reshape(len(v1), -1)
                    z2 = ae_model.encode(v2).reshape(len(v2), -1)
                else:
                    # fallback: use two different random noisy versions of xb
                    v1 = xb
                    v2 = xb
                    z1 = ae_model.encode(v1).reshape(len(v1), -1)
                    z2 = ae_model.encode(v2).reshape(len(v2), -1)

                # recon: reconstruction from original (or v1)
                x_recon    = ae_model(xb)                      # assumes forward returns recon
                recon_loss = F.mse_loss(x_recon, xb)

                ssl_loss = 0.0
                if ssl_mode == "barlow":
                    ssl_loss = SemiSupLearning.barlow_twins_loss(z1, z2)
                elif ssl_mode == "vicreg":
                    ssl_loss = SemiSupLearning.vicreg_loss(z1, z2)
                elif ssl_mode is None:
                    ssl_loss = 0.0
                else:
                    raise ValueError("Unknown ssl_mode")

                loss = recon_weight * recon_loss + ssl_weight * ssl_loss
                opt.zero_grad()
                loss.backward()
                opt.step()
            if (epoch + 1) % print_every == 0 or epoch == epochs-1:
                print(f"[AE+SSL] epoch {epoch+1}/{epochs} recon={recon_loss.item():.4f} ssl={float(ssl_loss):.4f}")
        ae_model.eval()
        return ae_model
