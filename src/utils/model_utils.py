"""Predictions and feature selection / dimensionality reduction"""
import time
import gc
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import tracemalloc
from collections import defaultdict
from typing import Tuple, List

from catboost import CatBoostRegressor
from fvcore.nn import FlopCountAnalysis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE

from src.methods.predictions import SingleOutputModelPredictor


def profile_epoch(model, loader, optimizer, criterion, device, warmup=False, measure_epochs=1):
    """Run one full pass (epoch) over `loader` for profiling purposes.
    Performs forward, backward, and optimizer steps for each batch.
    Optionally does warmup iterations and repeats timing for `measure_epochs`.
    Args:
        model: PyTorch model to profile.
        loader: DataLoader providing batches of (X, y).
        optimizer: Optimizer used for backward pass.
        criterion: Loss function.
        device: torch.device to run computations on.
        warmup: If True, performs warmup iterations before timing.
        measure_epochs: # full epochs to run for profiling; reported metrics
                are averaged per epoch over these runs.
    Returns:
        dict with timing/metric statistics averaged per epoch over measured_epochs runs"""
    model.to(device)
    model.train()

    def unpack(batch):
        return (batch[0].to(device), batch[1].to(device)) if len(batch) == 2 else (batch[0].to(device), None)

    if warmup:
        for batch in loader:
            x, y = unpack(batch)
            optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out, y) if y is not None else criterion(out, x)
            loss.backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
    runtimes, peak_mem = [], 0
    for _ in range(measure_epochs):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        else:
            tracemalloc.start()
        start = time.time()
        for batch in loader:
            x, y = unpack(batch)
            optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out, y) if y is not None else criterion(out, x)
            loss.backward()
            optimizer.step()
        end = time.time()
        runtimes.append(end - start)
        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / 1e6)
        else:
            _, peak  = tracemalloc.get_traced_memory()
            peak_mem = max(peak_mem, peak / 1e6)
            tracemalloc.stop()
    avg_runtime  = sum(runtimes) / len(runtimes)
    num_params   = sum(p.numel() for p in model.parameters()) / 1e6
    sample_input = next(iter(loader))[0][:1].to(device)
    flops        = FlopCountAnalysis(model, sample_input)
    flops_m      = flops.total() / 1e6

    print("===== Profiling =====")
    print(f"Avg Runtime [s]: {avg_runtime:.2f}")
    print(f"Peak Memory [MB]: {peak_mem:.2f}")
    print(f"# Params [x10^6]: {num_params:.2f}")
    print(f"FLOPs [x10^6]: {flops_m:.2f}")
    print("=====================")
    metrics = {
        'runtime_s': avg_runtime,
        'peak_memory_MB': peak_mem,
        'num_params_M': num_params,
        'flops_M': flops_m}
    return metrics

def clear_cuda_memory() -> None:
    """Release unreferenced GPU tensors and trigger CUDA memory cleanup."""
    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def schedule_learning_rate(step, max_steps, lr_0=1e-3, lr_end=1e-5, schedule_type="linear"):
    """Compute learning rate at given step with linear or cosine decay from lr0 to lr_end."""
    if schedule_type == "linear":
        return lr_0 - (lr_0 - lr_end) * (step / max_steps)
    elif schedule_type == "cosine":
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        return lr_end + (lr_0 - lr_end) * cosine_decay
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")

def norm_temp_xentropy_loss(z1, z2, temperature=0.5):
    """Normalized temperature-scaled cross entropy loss"""
    B   = z1.size(0)   # dynamically set batch size
    z1  = F.normalize(z1, dim=1)
    z2  = F.normalize(z2, dim=1)
    z   = torch.cat([z1, z2], dim=0)  # (2B, dim)

    sim = torch.matmul(z, z.T) / temperature
    mask= torch.eye(2*B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -9e15)

    labels = torch.cat([torch.arange(B) + B, torch.arange(B)], dim=0).to(z.device)
    return F.cross_entropy(sim, labels)


class ProjectionHead(nn.Module):
    """MLP projection head: maps latent z to projected space H for contrastive learning
    Notes:
    - Last layer is Linear only, **no BatchNorm** (kills contrastive loss), which is important for NT-Xent / cosine similarity loss.
    - Outputs are normalized with F.normalize to unit vectors for contrastive similarity."""
    def __init__(self, input_dim: int, proj_dim: int, hidden_sizes: list[int] = [256], dropout: float = 0.0):
        super().__init__()
        layers   = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        # Final projection layer: NO BatchNorm here!
        layers.append(nn.Linear(prev_dim, proj_dim))  # final projection
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Project latent z into normalized space H for contrastive loss.
        - z: (batch, latent_dim)
        - h: (batch, proj_dim), L2-normalized"""
        h = self.net(z)
        h = F.normalize(h, dim=1)  # normalize to unit vectors; ensures cosine similarity is meaningful
        return h


class Decoder(nn.Module):
    """Optional decoder to reconstruct the original input from latent embeddings z"""
    def __init__(self, latent_dim: int, output_shape: tuple[int, int], hidden_sizes=[128, 128], dropout: float = 0.0):
        """Args:
            latent_dim: Dimensionality of input latent z
            output_shape: Tuple (time_steps, channels) for reconstruction
            hidden_sizes: List of hidden layer sizes
            dropout: Dropout probability in hidden layers"""
        super().__init__()
        layers = []
        prev_dim = latent_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_shape[0] * output_shape[1]))  # flatten output
        self.net = nn.Sequential(*layers)
        self.output_shape = output_shape

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass: map latent z → reconstructed X
        Args:
            z: (batch, latent_dim)
        Returns:
            X_hat: (batch, time_steps, channels)"""
        x_hat = self.net(z)
        return x_hat.view(-1, *self.output_shape)  # reshape to (batch, time, channels)


class TorchWrapper(nn.Module):
    """Wraps a non-nn.Module model for PyTorch pipelines.
    Exposes the internal network as `.net` and also `.model` for compatibility."""
    def __init__(self, ts_model):
        super().__init__()
        self.net = ts_model.net if hasattr(ts_model, "net") else ts_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def model(self):
        return self.net


class PCA_analysis:
    """Class that contains all PCA methods. Given a df of F cols, the PCA df will also have F cols.
    This class looks at X only, not y"""

    @staticmethod
    def fit_pca(X: pd.DataFrame, var_threshold: float = 0.95):
        """Fit PCA retaining enough components to cover var_threshold variance."""
        pca_model = PCA(n_components=var_threshold)
        pca_model.fit(X)
        return pca_model

    @staticmethod
    def explain_pca_variance(pca, show_plot=False) -> Tuple[np.ndarray, int]:
        """Plot cumulative explained variance and return components and count."""
        explained_var = np.cumsum(pca.explained_variance_ratio_)
        N_pca_components = len(explained_var)  # all components in fitted pca

        if show_plot:
            plt.plot(explained_var)
            plt.axhline(y=explained_var[-1], color='r', linestyle='--')
            plt.xlabel("# of PCA Components")
            plt.ylabel("Fraction of Total Variance Explained")
            plt.title("Cumulative Explained Variance from PCA")
            plt.show()

        print(f"# PCA components covering {explained_var[-1]*100:.1f}% variance: {N_pca_components}")
        return pca.components_, N_pca_components

    @staticmethod
    def print_top_features_per_component(X_scaled, top_components, top_k_features: int = 3) -> None:
        """Print top contributing features for each PCA component"""
        feature_names = X_scaled.columns
        for i, comp in enumerate(top_components):
            abs_loadings    = np.abs(comp)
            percent_contrib = abs_loadings / abs_loadings.sum() * 100
            top_indices     = np.argsort(percent_contrib)[::-1]
            print(f'Component {i+1}:')
            for idx in top_indices[:top_k_features]:
                print(f'  {feature_names[idx]}: {percent_contrib[idx]:.2f}%')

    @staticmethod
    def summarize_feature_importance(X_scaled, top_components, top_k_features: int = 10):
        """Aggregate feature contributions across top components"""
        feature_names      = X_scaled.columns
        feature_importance = defaultdict(float)

        for comp in top_components:
            abs_loadings    = np.abs(comp)
            percent_contrib = abs_loadings / abs_loadings.sum() * 100
            for i, contrib in enumerate(percent_contrib):
                feature_importance[feature_names[i]] += contrib

        sorted_features = sorted(feature_importance.items(), key=lambda x: -x[1])
        print(f"\nTop {top_k_features} features across all components:")
        for feature, total_contrib in sorted_features[:top_k_features]:
            print(f"{feature}: {total_contrib:.2f}%")
        return sorted_features


class RFE_analysis():
    """RFE with Catboost"""
    def __init__(self, device):
        self.device     = torch.device(device) if isinstance(device, str) else device
        self.device_str = 'GPU' if self.device.type == 'cuda' else 'CPU'

    def apply_recursive_feature_elimination(self, X_train_scaled: pd.DataFrame, X_val_scaled: pd.DataFrame,
                                            y_train: pd.Series, y_val: pd.Series, single_predictor: SingleOutputModelPredictor,
                                            fraction_cols_to_keep: float = 0.95) -> Tuple[float, RFE]:
        """Apply Recursive Feature Elimination (RFE) using CatBoostRegressor on numeric features
        of training data. Returns RMSE on validation set + the fitted RFE model.
        - fraction_cols_to_keep (float): Fraction of numeric features to retain. Larger = less computations = faster"""

        numeric_feature_names = X_train_scaled.select_dtypes(include=np.number).columns

        X_train_num = X_train_scaled[numeric_feature_names]
        X_val_num   = X_val_scaled[numeric_feature_names]

        base_model  = CatBoostRegressor(verbose=0, random_state=42, early_stopping_rounds=10)
        rfe_model   = RFE(base_model, n_features_to_select=int(X_train_num.shape[1] * fraction_cols_to_keep))
        X_train_rfe = rfe_model.fit_transform(X_train_num, y_train.values.ravel())
        X_val_rfe   = rfe_model.transform(X_val_num)

        rmse_rfe, _, _ = single_predictor.predict_catboost_single_model(X_train_rfe, y_train, X_val_rfe,
                                                                        y_val, cat_features=None)
        print(f"CatBoost RMSE (RFE): {rmse_rfe:.3f} nm")
        return rmse_rfe, rfe_model

    def get_sorted_features_by_importance(self, rfe_model: RFE, X_train_num: pd.DataFrame):# -> List[str]:
        """Returns RFE-selected features sorted by importance (descending)"""

        selected_features = X_train_num.columns[rfe_model.get_support()]
        final_model: CatBoostRegressor = rfe_model.estimator_
        importances     = final_model.feature_importances_
        sorted_indices  = np.argsort(importances)[::-1]
        sorted_features = selected_features[sorted_indices]
        return sorted_features#.tolist()
