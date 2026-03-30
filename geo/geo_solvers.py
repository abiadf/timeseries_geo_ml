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

# class Predictors:
#     @staticmethod
#     def make_random_latent_prediction(Z_train, y_train, p, sliding_size, n_samples=1000):
#         """Evaluate a random latent baseline by sampling Z from the training set.
#         This preserves the exact empirical distribution and shape of Z_train,
#         then predicts y using a CatBoost model trained on (Z_train, y_train_windows).
#         Args:
#             Z_train: torch.Tensor of latent vectors (N, D).
#             y_train: target time series (torch.Tensor or np.array).
#             p: config object with window_size and task attributes.
#             n_samples: number of random latent samples to evaluate.
#         Returns:
#             rmse, r2: baseline performance of random latent samples."""
#         Z_train_np = Z_train.detach().cpu().numpy()
#         y_train_win = Windowing.make_windows_from_y(y_train, p.window_size, sliding_size, task=p.task).reshape(-1, 1)

#         idx = np.random.randint(0, Z_train_np.shape[0], size=n_samples)
#         Z_random_np = Z_train_np[idx]

#         y_hat_random = fit_catboost_multi(Z_train_np, y_train_win, Z_random_np)
#         y_dummy = np.tile(y_train_win.mean(axis=0), (n_samples, 1))
#         rmse = np.sqrt(mean_squared_error(y_dummy, y_hat_random))
#         r2 = r2_score(y_dummy, y_hat_random)
#         return rmse, r2

#     @staticmethod
#     def make_latent_pca_prediction(Z_train: torch.Tensor, Z_test: torch.Tensor, y_train: np.ndarray, y_test: np.ndarray,
#                                    p, n_components: float = 0.95, sliding_size=None) -> tuple[float, float]:
#         """Apply PCA to latents and predict with CatBoost."""
#         # window y to match latents
#         y_train_win = Windowing.make_windows_from_y(y_train, p.window_size, sliding_size, task=p.task, horizon=p.horizon)
#         y_test_win  = Windowing.make_windows_from_y(y_test,  p.window_size, sliding_size, task=p.task, horizon=p.horizon)

#         # convert latents to numpy
#         Z_train_np = Z_train.cpu().numpy()
#         Z_test_np  = Z_test.cpu().numpy()

#         # PCA fit on training latents
#         pca = PCA(n_components=n_components)
#         Z_train_pca = pca.fit_transform(Z_train_np)
#         Z_test_pca  = pca.transform(Z_test_np)

#         # regression
#         model = LinearRegression()
#         model.fit(Z_train_pca, y_train_win)
#         y_hat = model.predict(Z_test_pca)

#         # y_hat = fit_catboost_multi(Z_train_pca, y_train_win, Z_test_pca)
#         rmse  = np.sqrt(mean_squared_error(y_test_win, y_hat))
#         r2    = r2_score(y_test_win, y_hat)
#         return rmse, r2

#     @staticmethod
#     def make_latent_umap_catboost(Z_train: torch.Tensor, Z_test: torch.Tensor, y_train: np.ndarray, y_test: np.ndarray,
#                                 p, n_components: int = 5,  sliding_size=None, n_neighbors: int = 15, min_dist: float = 0.1, random_state: int = 42,
#                                 catboost_params: dict = None) -> tuple[float, float]:
#         """Apply UMAP to latents and predict with CatBoost regression."""
#         # window y to match latents
#         y_train_win = Windowing.make_windows_from_y(y_train, p.window_size, sliding_size, task=p.task, horizon=p.horizon)
#         y_test_win  = Windowing.make_windows_from_y(y_test,  p.window_size, sliding_size, task=p.task, horizon=p.horizon)

#         # convert latents to numpy
#         Z_train_np = Z_train.cpu().numpy()
#         Z_test_np  = Z_test.cpu().numpy()

#         # scale before UMAP
#         scaler = StandardScaler()
#         Z_train_scaled = scaler.fit_transform(Z_train_np)
#         Z_test_scaled  = scaler.transform(Z_test_np)

#         # UMAP
#         reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
#         Z_train_umap = reducer.fit_transform(Z_train_scaled)
#         Z_test_umap  = reducer.transform(Z_test_scaled)

#         y_hat = fit_catboost_multi(Z_train_umap, y_train_win, Z_test_umap)
#         rmse = np.sqrt(mean_squared_error(y_test_win, y_hat))
#         r2   = r2_score(y_test_win, y_hat)
#         return rmse, r2

#     @staticmethod
#     def run_riemann_catboost(Z_train, Z_test, y_train, y_test, window_size: int, sliding_size: int, prediction_task, eps: float = 1e-6):
#         """Riemannian covariance → tangent space (PGA) → scaling → CatBoost → RMSE/R2."""
#         # torch → numpy
#         Z_train_np = Z_train.cpu().numpy() if Z_train.is_cuda else Z_train.numpy()
#         Z_test_np  = Z_test.cpu().numpy()  if Z_test.is_cuda else Z_test.numpy()

#         # covariance
#         X_train_cov = compute_covariance_safe(Z_train_np)
#         X_test_cov  = compute_covariance_safe(Z_test_np)

#         # regularization
#         n = X_train_cov.shape[1]
#         X_train_cov += eps * np.eye(n)
#         X_test_cov  += eps * np.eye(n)

#         # tangent space (PGA)
#         ts = TangentSpace(metric='riemann')
#         Z_train_pga = ts.fit_transform(X_train_cov)
#         Z_test_pga  = ts.transform(X_test_cov)

#         # scale features
#         Z_train_scaled, Z_test_scaled = scale_train_and_test_sets(Z_train_pga, Z_test_pga)

#         # window + scale targets
#         y_train_w = Windowing.make_windows_from_y(y_train, window_size, sliding_size, task=prediction_task, horizon=p.horizon)
#         y_test_w  = Windowing.make_windows_from_y(y_test, window_size, sliding_size, task=prediction_task, horizon=p.horizon)
#         y_train_scaled, y_test_scaled = scale_train_and_test_sets(y_train_w, y_test_w)

#         # train + predict
#         y_hat = fit_catboost_multi(Z_train_scaled, y_train_scaled, Z_test_scaled)

#         # metrics
#         rmse = np.sqrt(mean_squared_error(y_test_scaled, y_hat))
#         r2   = r2_score(y_test_scaled, y_hat)
#         return y_hat, rmse, r2

#     @staticmethod
#     def cosine_similarity_samples(z: torch.Tensor) -> torch.Tensor:
#         """Mean pairwise cosine similarity between samples. z: (N, D), assumed normalized."""
#         z = torch.nn.functional.normalize(z, dim=1)
#         sim = z @ z.T
#         N = z.shape[0]
#         return (sim.sum() - N) / (N * (N - 1))

#     @staticmethod
#     def angular_variance_over_time(z: torch.Tensor) -> torch.Tensor:
#         """Angular variance across time. z: (T, D), normalized."""
#         z = torch.nn.functional.normalize(z, dim=1)
#         mean_dir = torch.mean(z, dim=0)
#         mean_dir = mean_dir / mean_dir.norm()
#         cos_angles = (z @ mean_dir).clamp(-1, 1)
#         angles = torch.acos(cos_angles)
#         return angles.var()

#     @staticmethod
#     def make_direct_prediction(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
#                                sliding_size: int, prediction_task: str, p) -> tuple[float, float, float]:
#         """Direct CatBoost baseline with task-consistent windowing."""
#         print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

#         if prediction_task == "tabular":
#             X_train_flat, X_test_flat = X_train, X_test
#             y_train_w, y_test_w       = y_train, y_test
#         else:
#             # --- window X ---
#             y_train_w = Windowing.make_windows_from_y(y_train, p.window_size, sliding_size, task=prediction_task, horizon=p.horizon)
#             y_test_w  = Windowing.make_windows_from_y(y_test, p.window_size, sliding_size, task=prediction_task, horizon=p.horizon)
#             X_train_w = Windowing.make_windows_from_X(torch.from_numpy(X_train.to_numpy()).float(), p.window_size, sliding_size, horizon=p.horizon)
#             X_test_w  = Windowing.make_windows_from_X(torch.from_numpy(X_test.to_numpy()).float(), p.window_size, sliding_size, horizon=p.horizon)

#             X_train_flat = X_train_w.reshape(X_train_w.shape[0], -1).numpy()
#             X_test_flat  = X_test_w.reshape(X_test_w.shape[0], -1).numpy()

#         # --- scale ---
#         X_train_flat, X_test_flat = scale_train_and_test_sets(X_train_flat, X_test_flat)
#         y_train_prep = y_train_w.reshape(y_train_w.shape[0], -1)
#         y_test_prep  = y_test_w.reshape(y_test_w.shape[0], -1)
#         # y_train_prep = y_train_w.reshape(-1, 1) if y_train_w.ndim == 1 else y_train_w
#         # y_test_prep  = y_test_w.reshape(-1, 1) if y_test_w.ndim == 1 else y_test_w
#         y_train_scaled, y_test_scaled = scale_train_and_test_sets(y_train_prep, y_test_prep)

#         assert y_train_prep.ndim == 2
#         assert y_train_prep.shape[0] == X_train_flat.shape[0]

#         # y_hat = LinearRegression().fit(X_train_flat, y_train_scaled).predict(X_test_flat)
#         y_hat = fit_catboost_multi(X_train_flat, y_train_scaled, X_test_flat, cb_verbose=50)
#         rmse  = root_mean_squared_error(y_test_scaled, y_hat)
#         r2    = r2_score(y_test_scaled, y_hat)
#         mae   = mean_absolute_error(y_test_scaled, y_hat)
#         return rmse, r2, mae

#     @staticmethod
#     def make_direct_prediction_mlp(X_train, X_test, y_train, y_test, p, sliding_size, device="cuda"):
#         # 1. Scaling (Crucial for Neural Nets)
#         y_tr_s, y_te_s = scale_train_and_test_sets(y_train, y_test)
#         X_tr_s, X_te_s = scale_train_and_test_sets(X_train, X_test)

#         # 2. Windowing
#         # Ensure we have Tensors for the windowing logic
#         def to_tensor(data):
#             if torch.is_tensor(data): return data.float()
#             if hasattr(data, 'values'): return torch.from_numpy(data.values).float()
#             return torch.from_numpy(data).float()

#         X_tr_tensor = to_tensor(X_tr_s)
#         X_te_tensor = to_tensor(X_te_s)

#         X_tr_w = Windowing.make_windows_from_X(X_tr_tensor, p.window_size, sliding_size, horizon=p.horizon)
#         X_te_w = Windowing.make_windows_from_X(X_te_tensor, p.window_size, sliding_size, horizon=p.horizon)
        
#         y_tr_w = Windowing.make_windows_from_y(y_tr_s, p.window_size, sliding_size, task=p.task, horizon=p.horizon)
#         y_te_w = Windowing.make_windows_from_y(y_te_s, p.window_size, sliding_size, task=p.task, horizon=p.horizon)

#         # Flatten windows into vectors for the MLP
#         X_tr_flat = X_tr_w.reshape(X_tr_w.size(0), -1)
#         X_te_flat = X_te_w.reshape(X_te_w.size(0), -1)
#         y_tr_flat = torch.tensor(y_tr_w.reshape(y_tr_w.shape[0], -1), dtype=torch.float32)
#         y_te_flat = torch.tensor(y_te_w.reshape(y_te_w.shape[0], -1), dtype=torch.float32)

#         # 3. Training Setup
#         loader = DataLoader(TensorDataset(X_tr_flat, y_tr_flat), batch_size=p.batch_size, shuffle=True)
#         # Hidden dim is doubled to handle the high-dimensional flattened window
#         direct_head = MLPPredHead(X_tr_flat.shape[1], y_tr_flat.shape[1], hidden_dim=p.hidden_dim * 2).to(device)
#         optimizer = torch.optim.AdamW(direct_head.parameters(), lr=p.lr_optimizer)

#         direct_head.train()
#         for epoch in range(p.epochs):
#             epoch_loss = 0
#             for xb, yb in loader:
#                 xb, yb = xb.to(device), yb.to(device)
#                 y_hat = direct_head(xb)
#                 loss = F.mse_loss(y_hat, yb)
                
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 epoch_loss += loss.item()
            
#             if epoch % 10 == 0:
#                 print(f"Direct MLP Epoch {epoch}: Loss {epoch_loss/len(loader):.4f}")

#         # 4. Evaluation
#         direct_head.eval()
#         with torch.no_grad():
#             y_hat_final = direct_head(X_te_flat.to(device)).cpu().numpy()
        
#         y_te_true = y_te_flat.numpy()
#         rmse = root_mean_squared_error(y_te_true, y_hat_final)
#         r2   = r2_score(y_te_true, y_hat_final)
#         mae  = mean_absolute_error(y_te_true, y_hat_final)
        
#         return rmse, r2, mae

#     @staticmethod
#     def make_direct_prediction_pca(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
#                                 window_size: int, sliding_size: int, prediction_task: str, n_components: int = 0.95) -> tuple[float, float]:
#         """Direct CatBoost baseline with optional PCA for dimensionality reduction."""

#         if prediction_task == "tabular":
#             X_train_flat, X_test_flat = X_train, X_test
#             y_train_w, y_test_w       = y_train, y_test
#         else:
#             y_train_w = Windowing.make_windows_from_y(y_train, window_size, sliding_size, task=prediction_task, horizon=p.horizon)
#             y_test_w  = Windowing.make_windows_from_y(y_test, window_size, sliding_size, task=prediction_task, horizon=p.horizon)

#             X_train_w = Windowing.make_windows_from_X(torch.from_numpy(X_train.to_numpy()).float(), window_size, sliding_size, horizon=p.horizon)
#             X_test_w  = Windowing.make_windows_from_X(torch.from_numpy(X_test.to_numpy()).float(), window_size, sliding_size, horizon=p.horizon)
#             X_train_flat = X_train_w.reshape(X_train_w.shape[0], -1).numpy()
#             X_test_flat  = X_test_w.reshape(X_test_w.shape[0], -1).numpy()

#         # --- PCA ---
#         pca = PCA(n_components=n_components)
#         X_train_flat = pca.fit_transform(X_train_flat)
#         X_test_flat  = pca.transform(X_test_flat)

#         # --- scale ---
#         X_train_flat, X_test_flat = scale_train_and_test_sets(X_train_flat, X_test_flat)

#         y_train_prep = y_train_w.reshape(-1, 1) if y_train_w.ndim == 1 else y_train_w
#         y_test_prep  = y_test_w.reshape(-1, 1) if y_test_w.ndim == 1 else y_test_w
#         y_train_scaled, y_test_scaled = scale_train_and_test_sets(y_train_prep, y_test_prep)

#         # --- regression ---
#         y_hat = fit_catboost_multi(X_train_flat, y_train_scaled, X_test_flat)
#         rmse  = root_mean_squared_error(y_test_scaled, y_hat)
#         r2    = r2_score(y_test_scaled, y_hat)
#         return rmse, r2

#     @staticmethod
#     def compute_covariance_safe(X, eps=1e-6):
#         """
#         Compute regularized covariance matrices for latent features.
#         Handles 1D, 2D, or 3D inputs.
#         Returns: [n_samples, n_channels, n_channels]
#         """
#         covs = []
#         for x in X:
#             # convert to 2D (samples × features)
#             if x.ndim == 0:
#                 x_flat = x.reshape(1, 1)
#             elif x.ndim == 1:
#                 x_flat = x.reshape(1, -1)
#             elif x.ndim == 2:
#                 x_flat = x
#             else:  # >2D
#                 x_flat = x.reshape(x.shape[0], -1)

#             if x_flat.shape[0] == 1:
#                 # only one sample → covariance is outer product
#                 C = np.outer(x_flat[0], x_flat[0])
#             else:
#                 C = np.cov(x_flat, rowvar=False)

#             # regularize to make positive definite
#             C += eps * np.eye(C.shape[0])
#             covs.append(C)

#         return np.array(covs)

#     @staticmethod
#     def make_direct_prediction_cov_pca(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
#                                     window_size: int, sliding_size: int, prediction_task: str, n_components: int = 0.95,
#                                     method: str = "PCA") -> tuple[float, float]:
#         """Direct CatBoost baseline with PCA or PGA on covariance matrices for fair comparison."""
        
#         if prediction_task == "tabular":
#             raise ValueError("Covariance-based PCA/PGA requires time-windowed data")
        
#         # --- create windows ---
#         y_train_w = Windowing.make_windows_from_y(y_train, window_size, sliding_size, task=prediction_task, horizon=p.horizon)
#         y_test_w  = Windowing.make_windows_from_y(y_test, window_size, sliding_size, task=prediction_task, horizon=p.horizon)
        
#         X_train_w = Windowing.make_windows_from_X(torch.from_numpy(X_train.to_numpy()).float(), window_size, sliding_size, horizon=p.horizon)
#         X_test_w  = Windowing.make_windows_from_X(torch.from_numpy(X_test.to_numpy()).float(), window_size, sliding_size, horizon=p.horizon)
        
#         # Ensure numpy
#         X_train_w = X_train_w.numpy() if isinstance(X_train_w, torch.Tensor) else X_train_w
#         X_test_w  = X_test_w.numpy() if isinstance(X_test_w, torch.Tensor) else X_test_w

#         # --- compute covariance matrices ---
#         X_train_cov = compute_covariance_safe(X_train_w)
#         X_test_cov  = compute_covariance_safe(X_test_w)

#         # --- dimensionality reduction ---
#         if method.upper() == "PCA":
#             # Flatten covariance matrices and apply linear PCA
#             n_samples, n_channels, _ = X_train_cov.shape
#             X_train_flat = X_train_cov.reshape(n_samples, -1)
#             X_test_flat  = X_test_cov.reshape(X_test_cov.shape[0], -1)
#             reducer = PCA(n_components=n_components)
#             X_train_flat = reducer.fit_transform(X_train_flat)
#             X_test_flat  = reducer.transform(X_test_flat)
#         elif method.upper() == "PGA":
#             ts = TangentSpace(metric='riemann')
#             X_train_flat = ts.fit_transform(X_train_cov)
#             X_test_flat  = ts.transform(X_test_cov)
#         else:
#             raise ValueError("method must be 'PCA' or 'PGA'")

#         # --- scale ---
#         X_train_flat, X_test_flat = scale_train_and_test_sets(X_train_flat, X_test_flat)

#         y_train_prep = y_train_w.reshape(-1, 1) if y_train_w.ndim == 1 else y_train_w
#         y_test_prep  = y_test_w.reshape(-1, 1) if y_test_w.ndim == 1 else y_test_w
#         y_train_scaled, y_test_scaled = scale_train_and_test_sets(y_train_prep, y_test_prep)

#         # --- regression ---
#         y_hat = fit_catboost_multi(X_train_flat, y_train_scaled, X_test_flat)
#         rmse  = root_mean_squared_error(y_test_scaled, y_hat)
#         r2    = r2_score(y_test_scaled, y_hat)
#         return rmse, r2

