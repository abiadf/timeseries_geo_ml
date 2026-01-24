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


# to use in main code
from geo_utils import Windowing, scale_train_and_test_sets
from geo_encoders import fit_catboost_multi, MLPPredHead

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



# %% VAE

class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, layer_dims: list = [128, 64],
                 distribution: str = "gaussian", activation = nn.Tanh):
        super().__init__()
        self.distribution = distribution
        # encoder
        enc_layers = []
        prev       = input_dim
        for h in layer_dims:
            enc_layers.append(nn.Linear(prev, h))
            enc_layers.append(activation()) # ReLU Tanh Sigmoid Softplus ELU
            prev = h
        self.enc    = nn.Sequential(*enc_layers)
        self.mu     = nn.Linear(prev, latent_dim)
        self.logvar = nn.Linear(prev, latent_dim)

        # Decoder (mirror encoder)
        dec_layers = []
        prev       = latent_dim
        for h in reversed(layer_dims):
            dec_layers.append(nn.Linear(prev, h))
            dec_layers.append(activation())
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.dec = nn.Sequential(*dec_layers)

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        """for gaussian or von Mises-Fisher"""
        if self.distribution.lower()[0] == "g": # gaussian
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        elif self.distribution.lower()[0] == "v": # von Mises-Fisher
            mu    = F.normalize(mu, dim=1)
            kappa = F.softplus(logvar) + 1e-6
            eps   = torch.randn_like(mu)
            eps   = F.normalize(eps, dim=1)
            return F.normalize(mu + eps / kappa, dim=1)
        elif self.distribution.lower()[0] == "s": # pure spherical 
            return F.normalize(mu, dim=1)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar if self.distribution.lower()[0] != "s" else z


def vae_loss(x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor,
             distribution: str = "gaussian", beta: float = 1.0, pb_weight: float = 0.03,
             decoder: nn.Module | None = None, z: torch.Tensor | None = None,):
    """VAE objective with optional β-KL and pullback-metric regularization.
    Loss = recon(x, x̂) + β · KL(q(z|x) || p(z)) + pb_weight · ||J_G(z)|| deviation
    The pullback term softly enforces local isometry of the decoder to prevent
    latent collapse or extreme volume distortion.
    Args:
        x: Input batch (N, D)
        x_hat: Reconstruction (N, D)
        mu: Latent mean (N, Z)
        logvar: Latent log-variance (N, Z)
        distribution: Latent distribution ("gaussian", "vMF", "spherical")
        beta: Weight on KL divergence (β-VAE)
        pb_weight: Weight on pullback/Jacobian regularizer
        decoder: Decoder network G(z) (required for pullback term)
        z: Latent samples used to evaluate decoder Jacobian
    Returns:
        total_loss: Scalar objective
        recon_loss: Reconstruction loss
        kl_loss: KL divergence (0 if non-Gaussian)"""
    recon = F.mse_loss(x_hat, x, reduction="mean")

    if distribution.lower()[0] == "g":
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    elif distribution.lower()[0] == "v":
        mu    = F.normalize(mu, dim=1)
        kappa = F.softplus(logvar) + 1e-6
        kl    = torch.mean(kappa)
    else:
        kl = torch.tensor(0.0, device=x.device)

    pb_loss = torch.tensor(0.0, device=x.device)
    if decoder is not None and z is not None:
        z_j = z.detach().requires_grad_(True)
        y   = decoder(z_j)
        v   = torch.randn_like(y)
        grad_v  = torch.autograd.grad(y, z_j, grad_outputs=v, create_graph=True)[0]
        pb_loss = torch.mean((grad_v.norm(dim=1) - 1.0) ** 2)

    total = recon + beta * kl + pb_weight * pb_loss
    return total, recon, kl

def train_vae(X_train, latent_dim: int, lr: float, epochs: int, 
              vae_layer_dims: list[int], beta: float = 1e-3):
    """Train a VAE using mini-batches (stochastic gradient descent).
    Args:
        X_train (np.ndarray or torch.Tensor): Training data.
        latent_dim (int): Dimension of latent space.
        lr (float): Learning rate.
        epochs (int): Number of epochs.
        vae_layer_dims (list[int]): Encoder/decoder hidden layer sizes.
        distribution (str): Distribution type ('gaussian', 'vMF', 'spherical').
        activation (nn.Module): Activation function.
        batch_size (int): Mini-batch size.
    Returns:
        VAE: Trained VAE model."""
    input_dim = X_train.shape[1]
    vae       = VAE(input_dim, latent_dim, layer_dims=vae_layer_dims).to(device)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr)
    X = torch.as_tensor(X_train, dtype=torch.float32, device=device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        x_hat, mu, logvar = vae(X)
        loss, recon, kl   = vae_loss(X, x_hat, mu, logvar,
                                     distribution="gaussian", beta=beta)
        loss.backward()
        optimizer.step()

        if epoch % 25 == 0:
            print(
                f"Epoch {epoch}: "
                f"Loss={loss.item():.4f} "
                f"Recon={recon.item():.4f} "
                f"KL={kl.item():.4f} β={beta}" )
    return vae


class Encoder(nn.Module):
    def __init__(self, x_dim: int, z_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, 64),
            nn.ReLU(),
            nn.Linear(64, z_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, z_dim: int, x_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, x_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# %% metrics

def evaluate_latent(z_train, z_test, y_train, y_test, n_clusters: int = 12, model_type: str = "linear", task_type: str = "regression",) -> Tuple:
    """Evaluate latent space with supervised model + KMeans clustering."""
    z_train_np = z_train.cpu().numpy()
    z_test_np  = z_test.cpu().numpy()

    task_type  = task_type.lower()
    model_type = model_type.lower()

    if task_type == "regression":
        if model_type == "linear":
            model = LinearRegression()
        elif model_type == "rf":
            model = RandomForestRegressor(n_estimators=200, random_state=42)
        elif model_type == "catboost":
            model = CatBoostRegressor(iterations=200, learning_rate=0.1, verbose=0, random_seed=42)
        else:
            raise ValueError(f"Unknown regressor: {model_type}")

        model.fit(z_train_np, y_train)
        y_pred = model.predict(z_test_np)
        rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
        r2     = r2_score(y_test, y_pred)
        supervised_metrics = (rmse, r2)

    elif task_type == "classification":
        if model_type == "linear":
            model = LogisticRegression(max_iter=1000)
        elif model_type == "rf":
            model = RandomForestClassifier(n_estimators=200, random_state=42)
        elif model_type == "catboost":
            model = CatBoostClassifier(iterations=200, learning_rate=0.1, verbose=0, random_seed=42)
        else:
            raise ValueError(f"Unknown classifier: {model_type}")

        model.fit(z_train_np, y_train)
        y_pred   = model.predict(z_test_np)
        accuracy = accuracy_score(y_test, y_pred)
        f1       = f1_score(y_test, y_pred, average="weighted")
        supervised_metrics = (accuracy, f1)
    else:
        raise ValueError("task_type must be 'regression' or 'classification'")

    # Clustering
    kmeans   = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(z_train_np)
    clusters = kmeans.predict(z_test_np)

    silhouette_score_val = silhouette_score(z_test_np, clusters)
    return (*supervised_metrics, clusters, silhouette_score_val)

def compute_latent_geometry_stats(decoder, z, n_samples=100):
    """Analyze latent geometry: Jacobian norm and pairwise distances.
    Args:
        decoder (nn.Module): Decoder G(z)
        z (torch.Tensor): Latent points (N x latent_dim)
        n_samples (int): Number of points to sample for pairwise distances
    Returns:
        dict: {'jacobian_mean': float, 'jacobian_std': float, 'dist_mean': float, 'dist_std': float}"""
    z_sample = z[:n_samples].detach().requires_grad_(True)
    y        = decoder(z_sample)

    jacobian_norms = []
    for i in range(y.shape[1]):  # iterate output dimensions
        grad = torch.autograd.grad(
            y[:, i].sum(), z_sample, create_graph=False, retain_graph=True)[0]
        jacobian_norms.append(grad.norm(dim=1))
    jacobian_norms    = torch.stack(jacobian_norms, dim=1)
    jac_mean, jac_std = jacobian_norms.mean().item(), jacobian_norms.std().item()

    # Pairwise Euclidean distances in latent space
    pairwise_dists = []
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            pairwise_dists.append((z_sample[i] - z_sample[j]).norm().item())
    dist_mean = float(torch.tensor(pairwise_dists).mean())
    dist_std  = float(torch.tensor(pairwise_dists).std())

    return {
        'jacobian_mean': jac_mean,
        'jacobian_std': jac_std,
        'dist_mean': dist_mean,
        'dist_std': dist_std}

# %% WAE

class WAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, layer_dims=[128,64], activation=nn.Tanh):
        super().__init__()
        # Encoder
        enc_layers = []
        prev       = input_dim
        for h in layer_dims:
            enc_layers.append(nn.Linear(prev, h))
            enc_layers.append(activation())
            prev = h
        self.enc   = nn.Sequential(*enc_layers)
        self.z_out = nn.Linear(prev, latent_dim)  # deterministic latent

        # Decoder
        dec_layers = []
        prev       = latent_dim
        for h in reversed(layer_dims):
            dec_layers.append(nn.Linear(prev, h))
            dec_layers.append(activation())
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.dec = nn.Sequential(*dec_layers)

    def encode(self, x):
        return self.z_out(self.enc(x))

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        z     = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

def compute_rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0, rbf_type: str = "gaussian",):
    """Compute RBF (Radial Basis Function) kernel matrices (K_xx, K_yy, K_xy) between two vector sets.
    RBF types: Gaussian, Laplacian, Multiquadric, Inverse Multiquadric.
    standard is Gaussian. K_.. = kernel (Gram) matrix"""
    x_dot_x = x @ x.T            # (N, N)
    y_dot_y = y @ y.T            # (M, M)
    x_dot_y = x @ y.T            # (N, M)

    # squared norms
    x_sq_norms = (x ** 2).sum(dim=1, keepdim=True)   # (N, 1)
    y_sq_norms = (y ** 2).sum(dim=1, keepdim=True).T # (1, M)

    # squared distances
    sq_dist_xx = x_sq_norms   - 2 * x_dot_x + x_sq_norms.T
    sq_dist_yy = y_sq_norms.T - 2 * y_dot_y + y_sq_norms
    sq_dist_xy = x_sq_norms   - 2 * x_dot_y + y_sq_norms

    if rbf_type == "gaussian":
        K_xx = torch.exp(-sq_dist_xx / (2 * sigma**2))
        K_yy = torch.exp(-sq_dist_yy / (2 * sigma**2))
        K_xy = torch.exp(-sq_dist_xy / (2 * sigma**2))
    elif rbf_type == "laplacian":
        K_xx = torch.exp(-torch.sqrt(sq_dist_xx + 1e-8) / sigma)
        K_yy = torch.exp(-torch.sqrt(sq_dist_yy + 1e-8) / sigma)
        K_xy = torch.exp(-torch.sqrt(sq_dist_xy + 1e-8) / sigma)
    elif rbf_type == "multiquadric":
        K_xx = torch.sqrt(sq_dist_xx + sigma**2)
        K_yy = torch.sqrt(sq_dist_yy + sigma**2)
        K_xy = torch.sqrt(sq_dist_xy + sigma**2)
    elif rbf_type == "inv_multiquadric":
        K_xx = 1.0 / torch.sqrt(sq_dist_xx + sigma**2)
        K_yy = 1.0 / torch.sqrt(sq_dist_yy + sigma**2)
        K_xy = 1.0 / torch.sqrt(sq_dist_xy + sigma**2)
    else:
        raise ValueError(f"Unknown rbf_type: {rbf_type}")
    return K_xx, K_yy, K_xy

def wae_loss(x, x_hat, z, prior_z, lambda_mmd=10.0, sigma=1.0):
    """Compute WAE loss = reconstruction error + MMD (Maximum Mean Discrepancy) regularization"""
    recon = F.mse_loss(x_hat, x)
    K_xx, K_yy, K_xy = compute_rbf_kernel(z, prior_z, sigma, "gaussian")
    mmd   = K_xx.mean() + K_yy.mean() - 2*K_xy.mean()
    return recon + lambda_mmd * mmd, recon, mmd

def train_wae(model, dataloader, epochs=50, lr=1e-3, lambda_mmd=10.0, device='cpu'):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss, total_recon, total_mmd = 0.0, 0.0, 0.0
        for batch in dataloader:
            x        = batch[0].float().to(device)
            optimizer.zero_grad()
            x_hat, z = model(x)
            prior_z  = torch.randn_like(z).to(device)
            loss, recon, mmd = wae_loss(x, x_hat, z, prior_z, lambda_mmd)
            # loss, recon, mmd = swae_loss(x, x_hat, z, prior_z, lambda_mmd)
            loss.backward()
            optimizer.step()
            total_loss  += loss.item() * x.size(0)
            total_recon += recon.item() * x.size(0)
            total_mmd   += mmd.item() * x.size(0)
        n = len(dataloader.dataset)
        if epoch % 25 == 0:
            print(f"Epoch {epoch+1}: Loss={total_loss/n:.4f}, Recon={total_recon/n:.4f}, MMD={total_mmd/n:.4f}")

def sliced_wasserstein_distance(z: torch.Tensor, prior_z: torch.Tensor, n_projections: int = 50) -> torch.Tensor:
    """Compute the sliced Wasserstein distance between two latent distributions."""
    z_dim = z.size(1)
    # Random directions
    directions = torch.randn(n_projections, z_dim, device=z.device)
    directions = directions / directions.norm(dim=1, keepdim=True)  # normalize

    swd = 0.0
    for d in directions:
        # Project onto direction
        proj_z      = z @ d
        proj_prior  = prior_z @ d
        # Sort projections (1D Wasserstein)
        proj_z_sorted     = torch.sort(proj_z, dim=0)[0]
        proj_prior_sorted = torch.sort(proj_prior, dim=0)[0]
        swd += ((proj_z_sorted - proj_prior_sorted) ** 2).mean()  # L2-Wasserstein
    return swd / n_projections

def swae_loss(x, x_hat, z, prior_z, lambda_swd=10.0):
    """SWAE loss: reconstruction + sliced Wasserstein distance"""
    recon = F.mse_loss(x_hat, x)
    swd   = sliced_wasserstein_distance(z, prior_z)
    return recon + lambda_swd * swd, recon, swd


# %% Riemannian stuff

def compute_G(z: torch.Tensor, decoder: nn.Module) -> torch.Tensor:
    """Compute metric G = J.T @ J for a single point z."""
    z_input = z.squeeze().detach().requires_grad_(True)
    J = torch.autograd.functional.jacobian(decoder, z_input)
    G = J.T @ J
    
    # Regularize to prevent det(G) = 0
    eye = torch.eye(G.size(-1), device=G.device)
    return G + 0.1 * eye

def midpoint_riemannian_distance_cached(z_u, z_l, G_u, G_l):
    """Vectorized midpoint metric using precomputed G(z)."""
    Nu, d = z_u.shape
    Nl, _ = z_l.shape
    dz = z_u[:, None, :] - z_l[None, :, :]
    G_mid = 0.5 * (G_u[:, None, :, :] + G_l[None, :, :, :])  # (Nu,Nl,d,d)
    # dz @ G_mid @ dz for each pair
    d2 = torch.einsum('uid, uijd, ujd -> ui', dz, G_mid, dz)
    return d2

def knn_regress(d2: torch.Tensor, y_L: torch.Tensor, k: int = 5) -> torch.Tensor:
    # idx = torch.topk(d2, k, largest=False).indices
    # return y_L[idx].mean(dim=1)
    vals, idx = torch.topk(d2, k, largest=False)
    weights = 1.0 / (vals + 1e-6)
    weights = weights / weights.sum(dim=1, keepdim=True)
    return (weights * y_L[idx]).sum(dim=1)

def visualize_latent_geometry(vae, latent_dim, res=30):
    vae.eval()
    grid_x = torch.linspace(-3, 3, res)
    grid_y = torch.linspace(-3, 3, res)
    xx, yy = torch.meshgrid(grid_x, grid_y, indexing='ij')
    
    z_grid = torch.zeros((res * res, latent_dim), device=device)
    z_grid[:, 0] = xx.reshape(-1)
    z_grid[:, 1] = yy.reshape(-1)
    
    volumes = []
    with torch.no_grad(): # Use no_grad for the loop, but compute_G enables it internally
        for i in range(z_grid.shape[0]):
            # Pass a single vector (latent_dim,)
            G = compute_G(z_grid[i], vae.decode)
            # log-det helps visualize the massive scale differences
            vol = torch.logdet(G)
            volumes.append(vol.item())
    
    volumes = np.array(volumes).reshape(res, res)
    
    plt.figure(figsize=(10, 8))
    cp = plt.contourf(xx.numpy(), yy.numpy(), volumes, cmap='viridis', levels=50)
    plt.colorbar(cp, label='Log-Volume log|G(z)|')
    plt.title("Latent Space Curvature (Expansion/Compression)")
    plt.xlabel("Latent Dim 1")
    plt.ylabel("Latent Dim 2")
    plt.show()
    
