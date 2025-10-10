"""All predictions go here"""

from typing import Tuple

import catboost as cb
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

import itertools
from itertools import product
import lightgbm as lgb
from lightgbm import LGBMRegressor, early_stopping

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, GroupKFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# old, remove later
class X_MLPHead:
    """Supervised MLP predictor for embeddings with variable hidden layers. Predicts y from z"""
    def __init__(self, input_dim, output_dim, hidden_sizes=[64], lr=0.01, epochs=20, dropout=0.0, device="cpu"):
        self.device  = device
        self.epochs  = epochs

        layers   = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))  # output layer

        self.model     = nn.Sequential(*layers).to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.loss_fn   = nn.MSELoss()

    def train(self, z_train, y_train, z_test=None, y_test=None):
        z_train_t = torch.tensor(z_train, dtype=torch.float32, device=self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=self.device)

        if z_test is not None and y_test is not None:
            z_test_t = torch.tensor(z_test, dtype=torch.float32, device=self.device)
            y_test_t = torch.tensor(y_test, dtype=torch.float32, device=self.device)
        else:
            z_test_t = y_test_t = None

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            y_pred = self.model(z_train_t)
            loss   = self.loss_fn(y_pred, y_train_t)
            loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                if z_test_t is not None:
                    with torch.no_grad():
                        test_pred = self.model(z_test_t)
                        test_loss = self.loss_fn(test_pred, y_test_t)
                    print(f"Epoch {epoch}: Train {loss.item():.4f}, Test {test_loss.item():.4f}")
                else:
                    print(f"Epoch {epoch}: Train {loss.item():.4f}")

    def predict(self, z):
        z_t = torch.tensor(z, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.model(z_t).cpu().numpy()

    def evaluate(self, z_test, y_test):
        y_pred = self.predict(z_test)
        return root_mean_squared_error(y_test, y_pred)

class MLPHead:
    """Supervised MLP predictor for embeddings OR raw X. Avoids redundant tensor conversion."""

    def __init__(self, input_dim, output_dim, hidden_sizes=[64], lr=0.01,
                 epochs=20, dropout=0.0, device=None, early_stop_patience=10):
        self.PRINT_EVERY = 20
        self.device = device
        self.epochs = epochs
        self.early_stop_patience = early_stop_patience

        layers   = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))

        self.model     = nn.Sequential(*layers).to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.loss_fn   = nn.MSELoss()

    def _ensure_tensor(self, x):
        if isinstance(x, torch.Tensor):
            # Move to correct device and float32 if needed
            if x.device != torch.device(self.device) or x.dtype != torch.float32:
                return x.float().to(self.device)
            return x
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def train(self, X_train, y_train, X_test=None, y_test=None):
        X_train_t = self._ensure_tensor(X_train)
        y_train_t = self._ensure_tensor(y_train)
        X_test_t  = self._ensure_tensor(X_test)  if X_test is not None else None
        y_test_t  = self._ensure_tensor(y_test)  if y_test is not None else None

        best_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            y_pred = self.model(X_train_t)
            loss   = self.loss_fn(y_pred, y_train_t)
            loss.backward()
            self.optimizer.step()

            # if epoch % self.PRINT_EVERY == 0 or epoch == self.epochs - 1:
            #     if X_test_t is not None:
            #         with torch.no_grad():
            #             test_pred = self.model(X_test_t)
            #             test_loss = self.loss_fn(test_pred, y_test_t)
            #         print(f"Epoch {epoch}: Train {loss.item():.4f}, Test {test_loss.item():.4f}")
            #     else:
            #         print(f"Epoch {epoch}: Train {loss.item():.4f}")

            monitor_loss = loss.item()
            if X_test_t is not None:
                with torch.no_grad():
                    val_pred = self.model(X_test_t)
                    val_loss = self.loss_fn(val_pred, y_test_t).item()
                monitor_loss = val_loss

            # Early stopping
            if self.early_stop_patience is not None:
                if monitor_loss < best_loss:
                    best_loss = monitor_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.early_stop_patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

            if epoch % self.PRINT_EVERY == 0 or epoch == self.epochs - 1:
                if X_test_t is not None:
                    print(f"Epoch {epoch}: Train {loss.item():.4f}, Test {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch}: Train {loss.item():.4f}")

    def predict(self, X):
        X_t = self._ensure_tensor(X)
        with torch.no_grad():
            return self.model(X_t).cpu().numpy()

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return root_mean_squared_error(y_test, y_pred)


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
    
