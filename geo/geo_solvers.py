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
    
