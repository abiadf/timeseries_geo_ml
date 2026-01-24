"[☑️ RUN] utils functions"

import __main__
import sys, os
project_root = os.path.abspath("..")  # adjust if notebook is elsewhere
sys.path.insert(0, project_root)
from typing import Union
from dataclasses import dataclass

from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, mean_absolute_error, root_mean_squared_error, r2_score, silhouette_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import NearestNeighbors, KernelDensity

from catboost import CatBoostRegressor, CatBoostClassifier
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # print(torch.cuda.memory_reserved(0) / 1e6, "MB reserved")
    # print(torch.cuda.memory_allocated(0) / 1e6, "MB allocated")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% tabular
class EuclidEncoder(nn.Module):
    def __init__(self, window_size, input_dim, z_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window_size * input_dim, hidden),
            nn.ReLU())
            # nn.SiLU())
        self.mu     = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)

class SphericalEncoder(nn.Module):
    def __init__(self, window_size, input_dim, z_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window_size * input_dim, hidden),
            nn.ReLU())
            # nn.SiLU())

        self.mu_raw = nn.Linear(hidden, z_dim)
        self.kappa  = nn.Linear(hidden, 1)

    def forward(self, x):
        h      = self.net(x)
        mu_dir = F.normalize(self.mu_raw(h), dim=-1)
        kappa  = F.softplus(self.kappa(h)) + 1e-3
        return mu_dir, kappa

class Decoder(nn.Module):
    def __init__(self, z_dim_total, window_size, output_dim, hidden):
        super().__init__()
        self.window_size = window_size
        self.output_dim   = output_dim  # number of features (linear+cyclic)
        self.net = nn.Sequential(
            nn.Linear(z_dim_total, hidden),
            nn.ReLU(),
            # nn.SiLU(),
            nn.Linear(hidden, window_size * output_dim))  # << flattened output

    def forward(self, z):
        # z: (B, z_dim_total)
        return self.net(z)  # shape (B, window_size*output_dim)

# %% timeseries

# old
class old_LSTMSphericalEncoder(nn.Module):
    """Spherical latent LSTM encoder (vMF z_s)."""
    def __init__(self, input_dim, hidden_dim, z_dim):
        super().__init__()
        self.lstm   = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.mu_raw = nn.Linear(hidden_dim, z_dim)
        self.kappa  = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h      = h_n.squeeze(0)
        mu_dir = F.normalize(self.mu_raw(h), dim=-1)  # = cos theta, sin theta
        kappa  = F.softplus(self.kappa(h)) + 1e-3
        return mu_dir, kappa

# old
class old_LSTMToroidalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_cyc):
        super().__init__()
        self.n_cyc  = n_cyc
        self.lstm   = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.mu_raw = nn.Linear(hidden_dim, n_cyc * 2)
        self.kappa  = nn.Linear(hidden_dim, n_cyc)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h         = h.squeeze(0)
        
        # Reshape mu to [Batch, N_features, 2] and normalize each circle
        mu = self.mu_raw(h).view(-1, self.n_cyc, 2)
        mu = F.normalize(mu, dim=-1)
        
        # Kappa for each circle [Batch, N_features, 1]
        # kappa = F.softplus(self.kappa(h)).view(-1, self.n_cyc, 1) + 1e-3  # 3D
        kappa = F.softplus(self.kappa(h)).view(-1, self.n_cyc) + 1e-3  # 2D
        return mu, kappa

class LSTMEncoderEuclid(nn.Module):
    """Euclidean latent LSTM encoder (Gaussian z_e)."""
    def __init__(self, input_dim, hidden_dim, z_dim):
        super().__init__()
        self.lstm   = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.mu     = nn.Linear(hidden_dim, z_dim)
        self.logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        # x: [B, T, D]
        _, (lstm_hidden, _) = self.lstm(x)           # h_n: [1, B, H]
        encoder_hidden      = lstm_hidden.squeeze(0) # [B, H]
        return self.mu(encoder_hidden), self.logvar(encoder_hidden)  # [B, z_dim], [B, z_dim]

class LSTMSphericalEncoder(nn.Module):
    """Turns a sequence into a spherical latent representation.
    The LSTM collapses the time dimension and produces a final hidden state h_last.
    h_last is projected to:
      - mu_dir: a unit-length vector in R^z_dim (a direction on the (z_dim-1)-sphere)
      - kappa: a positive scalar controlling concentration around that direction"""
    def __init__(self, input_dim: int, hidden_dim: int, z_dim: int, n_layers: int =2, epsilon=1e-3):
        super().__init__()
        self.lstm    = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.mu_raw  = nn.Linear(hidden_dim, z_dim)
        self.kappa   = nn.Linear(hidden_dim, 1)
        self.epsilon = epsilon

    def forward(self, x):
        """x: (B, T, input_dim), B=batchsize (inferred from x, not defined)
        Returns:
        mu_dir: (B, z_dim) unit vectors (points on S^{z_dim-1})
        kappa:  (B, 1) positive concentration value"""
        _, (h, _) = self.lstm(x)
        h_last    = h[-1] 
        mu_dir    = F.normalize(self.mu_raw(h_last), dim=-1)
        # Ensure kappa is [B, 1] or [B]
        # kappa     = F.softplus(self.kappa(h_last)) + self.epsilon
        # return mu_dir, kappa
        logkappa  = self.kappa(h_last)
        return mu_dir, logkappa

class LSTMToroidalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_cyc_features, n_layers=2, epsilon=1e-3):
        """epsilon: small value to ensure kappa > 0"""
        super().__init__()
        self.n_cyc       = num_cyc_features
        self.epsilon     = epsilon
        self.lstm        = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.mu_layer    = nn.Linear(hidden_dim, num_cyc_features * 2)
        self.kappa_layer = nn.Linear(hidden_dim, num_cyc_features)

    def forward(self, x):
        # x: [Batch, Seq, Feats]
        _, (h, _) = self.lstm(x)
        h         = h[-1] # Take last layer's hidden state: [Batch, hidden_dim]
        
        # mu: [Batch, n_cyc, 2]
        mu = self.mu_layer(h).view(-1, self.n_cyc, 2)
        mu = F.normalize(mu, dim=-1)
        
        # kappa: [Batch, n_cyc]
        kappa = F.softplus(self.kappa_layer(h)).view(-1, self.n_cyc) + self.epsilon
        return mu, kappa

class LSTMDecoder(nn.Module):
    """LSTM decoder mapping latent z -> sequence output."""
    """Autoregressive LSTM decoder from latent z to sequence."""
    def __init__(self, z_dim, hidden_dim, output_dim, window_size):
        super().__init__()
        self.window_size = window_size
        self.hidden_dim  = hidden_dim
        self.output_dim  = output_dim
        self.lstm = nn.LSTM(output_dim, hidden_dim, batch_first=True)
        self.h0   = nn.Linear(z_dim, hidden_dim)
        self.c0   = nn.Linear(z_dim, hidden_dim)
        self.out  = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        B = z.size(0)
        h0 = self.h0(z).unsqueeze(0)
        c0 = self.c0(z).unsqueeze(0)
        x  = torch.zeros(B, self.window_size, self.output_dim, device=z.device)
        y, _ = self.lstm(x, (h0, c0))
        return self.out(y)


class MLPDecoder(nn.Module):
    """Decode concatenated latent vector z_e + z_s -> windowed features."""
    def __init__(self, z_dim_total, window_size, output_dim, hidden_dim):
        super().__init__()
        self.window_size = window_size
        self.output_dim  = output_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim_total, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
                nn.Linear(hidden_dim, window_size * output_dim))  # flattened output

    def forward(self, z):
        # z: [B, z_dim_total]
        return self.net(z)  # [B, window_size*output_dim]


# %% reparameterization tricks
class Reparam:
    @staticmethod
    def reparam_gaussian(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Gaussian reparameterization."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def reparam_vmf(mu_dir: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        """vMF has no closed form, so approximate vMF sampling: Gaussian noise + projection. Radius = 1 by construction.
        This is what Davidson (2018) does as well, fine because ELBO needs approximate sampling"""
        """Fixed: Uses broadcasting for kappa."""
        eps = torch.randn_like(mu_dir)
        # Ensure kappa is [B, 1] to broadcast with mu_dir [B, D]
        z = mu_dir + eps / (kappa.view(-1, 1) + 1e-6)
        return F.normalize(z, dim=-1)

    @staticmethod # old
    def sample_vmf(mu: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        """Sample from vMF distribution on S^{d-1} with direction mu and concentration kappa.
        mu: (batch, dim), must be unit norm
        kappa: (batch, 1)
        returns: (batch, dim)"""
        batch, dim = mu.shape
        # for very small kappa, approximate uniform sampling
        eps = torch.rand(batch, 1, device=mu.device)
        w   = 1 + (torch.log(eps + (1 - eps) * torch.exp(-2*kappa))) / kappa  # simplified
        w   = w.clamp(-1+1e-7, 1-1e-7)
        
        # sample v ~ uniform on unit sphere orthogonal to mu
        v = torch.randn(batch, dim, device=mu.device)
        v = v - (v*mu).sum(dim=1, keepdim=True) * mu  # make orthogonal
        v = F.normalize(v, dim=1)
        
        # combine
        z = w * mu + torch.sqrt(1 - w**2) * v
        return F.normalize(z, dim=-1)

    @staticmethod
    def sample_vmf_approximate(mu, kappa):
        """Fixed: Uses broadcasting for kappa."""
        noise = torch.randn_like(mu) * (1.0 / (kappa.view(-1, 1) + 1e-6))
        return F.normalize(mu + noise, dim=-1)

    @staticmethod
    def sample_vmf_exact(mu_xy: torch.Tensor, kappa: torch.Tensor):
        """Exact S¹ sampler."""
        mu_xy = F.normalize(mu_xy, dim=-1)
        mu_angle = torch.atan2(mu_xy[:, 1], mu_xy[:, 0])
        # Note: VonMises.rsample() is only available in newer PyTorch versions
        # If rsample() fails, use sample(), but it won't be backprop-friendly.
        dist = torch.distributions.VonMises(mu_angle, kappa.view(-1))
        theta = dist.sample() 
        return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)


def kl_gaussian(mu, logvar):
    """KL divergence between Gaussian distr. and standard normal distr."""
    return -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1).mean()

# remove if kl_vmf_uniform works
def regularization_vmf(kappa, dim):
    """Spherical has no closed form of the KL term, so this functino is a proxy regularizer
    This is more of a concentration regularizer encouraging proximity to a uniform hyperspherical prior than a KL divergence."""
    # return (kappa - (dim - 1) * torch.log(kappa + 1e-6)).mean() # my old one
    return (kappa**2).mean()

def kl_vmf_uniform(mu: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """Approximate KL(vMF(mu, kappa) || uniform(S^{d-1})), from Davidson et al. 2018
    mu: (batch, dim)
    kappa: (batch, 1)"""
    d     = mu.shape[1]
    kappa = kappa.clamp_min(1e-3) # avoid log(0)
    if not torch.is_tensor(kappa):
        kappa = torch.tensor(kappa, device=mu.device)
    # Approximate log C_d(kappa)
    log_c = (d/2 - 1) * torch.log(kappa) - (d/2) * torch.log(torch.tensor(2*torch.pi, device=mu.device)) - kappa
    kl    = kappa.squeeze(-1) - log_c  # per-batch, KL ≈ kappa * (μ · μ) + log C_d(kappa)  (simplified)
    return kl.mean()

class MixedEncoder(nn.Module):
    """Encoder producing mixed latent variables:
      - Euclidean (Gaussian) z_e with mean `mu_e` and log-variance `logvar_e`
      - Hyperspherical (vMF) z_s with direction `mu_s` and concentration `kappa`
    Args:
        input_dim (int): Number of input features
        hidden (int): Size of shared hidden layers
        z_e_dim (int): Dimension of Euclidean latent z_e
        z_s_dim (int): Dimension of hyperspherical latent z_s"""
    def __init__(self, input_dim, hidden_dim, z_e_dim, z_s_dim):
        super().__init__()
        # Shared hidden layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU())
        # Euclidean latent parameters
        self.mu_e    = nn.Linear(hidden_dim, z_e_dim) # mean vector (euclidean)
        self.logvar_e= nn.Linear(hidden_dim, z_e_dim) # log variance (euclidean)
        # Hyperspherical latent parameters
        self.mu_s    = nn.Linear(hidden_dim, z_s_dim) # mean direction (spherical)
        self.kappa   = nn.Linear(hidden_dim, 1)       # vMF concentration param (spherical)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        Args: x (Tensor): Input tensor of shape [batch_size, input_dim]
        * hidden_layer (Tensor) is a shared representation computed as self.shared(x)
        Returns:
            mu_e (Tensor): Mean of Gaussian latent z_e [batch_size, z_e_dim]
            logvar_e (Tensor): Log-variance of z_e [batch_size, z_e_dim]
            mu_s (Tensor): Unit vector direction of vMF latent z_s [batch_size, z_s_dim]
            kappa (Tensor): Concentration of vMF latent z_s [batch_size, 1], positive"""
        hidden_layer = self.shared(x)              # Shared hidden representation
        mu_e         = self.mu_e(hidden_layer)     # Gaussian mean
        logvar_e     = self.logvar_e(hidden_layer) # Gaussian log-variance
        mu_s         = F.normalize(self.mu_s(hidden_layer), dim=-1) # Unit vector for vMF
        kappa        = F.softplus(self.kappa(hidden_layer)) + 1e-3  # Positive concentration
        return mu_e, logvar_e, mu_s, kappa


class WithSplit:
    @staticmethod
    @torch.no_grad()
    def encode_tabular_dataset(lin_windows, cyc_windows, lin_encoder, cyc_encoder, pooling="mean"):
        """Encode linear + cyclic windows.
        lin_windows: (num_windows, win, D_lin)
        cyc_windows: (num_windows, win, D_cyc)"""
        N, win, D_lin  = lin_windows.shape
        D_cyc          = cyc_windows.shape[-1]

        lin_flat       = lin_windows.view(N, win*D_lin)
        cyc_flat       = cyc_windows.view(N, win*D_cyc)

        mu_e, logvar_e = lin_encoder(lin_flat)
        mu_s, kappa    = cyc_encoder(cyc_flat)

        z = torch.cat([mu_e, mu_s], dim=-1)

        if pooling is None:
            return z.view(N, 1, -1)  # add dummy W dim
        if pooling == "mean":
            return z.mean(dim=0, keepdim=True)  # only one sequence
        if pooling == "max":
            return z.max(dim=0, keepdim=True).values
        raise ValueError(f"Unknown pooling: {pooling}")

    @staticmethod
    @torch.no_grad()
    def encode_timeseries_dataset(X_lin_win, X_cyc_win, lin_encoder, cyc_encoder, pooling="mean"):
        """Encode linear + cyclic windows.
        X_lin_win: (num_windows, win, D_lin)
        X_cyc_win: (num_windows, win, D_cyc)"""
        num_windows, win, D_lin  = X_lin_win.shape
        D_cyc          = X_cyc_win.shape[-1]
        mu_e, logvar_e = lin_encoder(X_lin_win)
        mu_s, kappa    = cyc_encoder(X_cyc_win)

        z = torch.cat([mu_e, mu_s], dim=-1)

        if pooling is None:
            return z.view(num_windows, 1, -1)  # add dummy W dim
        if pooling == "mean":
            return z.mean(dim=0, keepdim=True)  # only one sequence
        if pooling == "max":
            return z.max(dim=0, keepdim=True).values
        raise ValueError(f"Unknown pooling: {pooling}")

    @staticmethod
    def vae_train_step_for_tabular(x_lin: torch.Tensor, x_cyc: torch.Tensor, lin_encoder: nn.Module, cyc_encoder: nn.Module,
                    decoder: nn.Module, lambdas: dict) -> torch.Tensor:
        """Single VAE training step.
        - x_lin, x_cyc : (batch, window_size, features)"""
        B, win, D_lin = x_lin.shape
        D_cyc         = x_cyc.shape[-1]

        x_lin_flat    = x_lin.view(B, win*D_lin)
        x_cyc_flat    = x_cyc.view(B, win*D_cyc)

        mu_e, logvar_e= lin_encoder(x_lin_flat)
        mu_s, kappa   = cyc_encoder(x_cyc_flat)
        
        # ensure kappa is Tensor
        if not isinstance(kappa, torch.Tensor):
            kappa = torch.tensor(kappa, dtype=torch.float32, device=x_cyc.device)

        z_e = Reparam.reparam_gaussian(mu_e, logvar_e)
        z_s = Reparam.reparam_vmf(mu_s, kappa)
        z   = torch.cat([z_e, z_s], dim=-1)

        x_hat       = decoder(z)
        x_full_flat = torch.cat([x_lin_flat, x_cyc_flat], dim=-1)
        L_recon     = F.mse_loss(x_hat, x_full_flat)

        L_kl_e   = kl_gaussian(mu_e, logvar_e)
        # L_reg_s  = regularization_vmf(kappa, z_s.size(-1))
        L_kl_s   = kl_vmf_uniform(mu_s, kappa)  # replaces old regularization_vmf()

        # return lambdas["reconstr"]*L_recon + lambdas["euc"]*L_kl_e + lambdas["sph"]*L_reg_s
        return lambdas["reconstr"]*L_recon + lambdas["euc"]*L_kl_e + lambdas["sph"]*L_kl_s

    @staticmethod
    def vae_train_step_for_timeseries(x_lin, x_cyc, lin_encoder, cyc_encoder, decoder, lambdas):
        """Supports cases where lin_encoder or cyc_encoder are None."""
        z_parts = []
        kl_e = torch.tensor(0.0, device=x_lin.device)
        kl_s = torch.tensor(0.0, device=x_lin.device)

        # Euclidean branch: Skip if encoder is None or input has no features
        if lin_encoder is not None and x_lin.shape[-1] > 0:
            mu_e, logvar_e = lin_encoder(x_lin)
            z_e = Reparam.reparam_gaussian(mu_e, logvar_e)
            z_parts.append(z_e)
            kl_e = kl_gaussian(mu_e, logvar_e)

        # Spherical branch: Skip if encoder is None or input has no features
        if cyc_encoder is None or x_cyc.shape[-1] == 0:
            mu_s = kappa = z_s = None
        else:
            mu_s, kappa = cyc_encoder(x_cyc)
            # Pass kappa explicitly reshaped to ensure no broadcast errors
            z_s = Reparam.reparam_vmf(mu_s, kappa.view(-1, 1))
            kl_s = kl_vmf_uniform(mu_s, kappa.view(-1))

        # Combine latents
        z = torch.cat(z_parts, dim=-1)
        x_hat = decoder(z)

        # Flatten inputs to match MLPDecoder output (B, win * D_total)
        # Only concatenate if both have features; otherwise just reshape the active one
        inputs_to_concat = []
        if x_lin.shape[-1] > 0: inputs_to_concat.append(x_lin.reshape(x_lin.size(0), -1))
        if x_cyc.shape[-1] > 0: inputs_to_concat.append(x_cyc.reshape(x_cyc.size(0), -1))
        
        x_target = torch.cat(inputs_to_concat, dim=-1)
        recon_loss = F.mse_loss(x_hat, x_target)

        return (lambdas["reconstr"] * recon_loss + 
                lambdas["euc"] * kl_e + 
                lambdas["sph"] * kl_s)

    @staticmethod
    def old_vae_train_step_for_timeseries(x_lin: torch.Tensor, x_cyc: torch.Tensor,
                    lin_encoder: nn.Module, cyc_encoder: nn.Module,
                    decoder: nn.Module, lambdas: dict) -> torch.Tensor:
        """Single VAE training step for split latent VAE.
        - x_lin, x_cyc : (batch, window_size, features)
        - decoder expects concatenated z_e + z_s of shape (batch, z_total)
        and outputs flattened reconstruction of shape (batch, window_size*D_total)"""
        B, win, D_lin = x_lin.shape
        D_cyc         = x_cyc.shape[-1]

        # flatten per-window inputs for LSTM
        x_lin_flat = x_lin
        x_cyc_flat = x_cyc

        # ---- Euclidean branch ----
        if lin_encoder is not None:
            mu_e, logvar_e = lin_encoder(x_lin_flat)
        else:
            mu_e = logvar_e = None

        # ---- Spherical branch ----
        if cyc_encoder is not None:
            mu_s, kappa = cyc_encoder(x_cyc_flat)
        else:
            mu_s = kappa = None

        # ---- reparameterize ----
        z_e = Reparam.reparam_gaussian(mu_e, logvar_e)  # [B, z_e]
        z_s = Reparam.reparam_vmf(mu_s, kappa)         # [B, z_s]
        z   = torch.cat([z_e, z_s], dim=-1)  # [B, z_total]

        # ---- decode ----
        x_hat = decoder(z)                  # [B, window_size * D_total]

        # ---- flatten original inputs for reconstruction loss ----
        x_full_flat = torch.cat([x_lin_flat.reshape(B, -1), x_cyc_flat.reshape(B, -1)], dim=-1)

        # ---- losses ----
        L_recon = F.mse_loss(x_hat, x_full_flat)
        L_kl_e  = kl_gaussian(mu_e, logvar_e)
        L_kl_s  = kl_vmf_uniform(mu_s, kappa)
        loss    = lambdas["reconstr"] * L_recon + lambdas["euc"] * L_kl_e + lambdas["sph"] * L_kl_s
        return loss

    @staticmethod
    def train_linear_and_cyclic_vaes_for_1_epoch_timeseries(data_loader: DataLoader, lin_encoder: nn.Module, cyc_encoder: nn.Module,
                    decoder: nn.Module, optimizer: torch.optim.Optimizer, lambdas: dict):
        """Full training loop over one epoch for the VAE.
        Args:
            data_loader : DataLoader yielding (x_lin, x_cyc). note x_lin and x_cyc are separated
            lin_encoder : Linear encoder
            cyc_encoder : Cyclic encoder
            decoder     : Decoder
            optimizer   : Optimizer
            lambdas     : dict of loss weights"""

        if lin_encoder is not None:
            lin_encoder.train()
        if cyc_encoder is not None:
            cyc_encoder.train()
        decoder.train()

        epoch_loss = 0
        for x_linear, x_cyclic in data_loader:
            optimizer.zero_grad()
            loss = WithSplit.vae_train_step_for_timeseries(x_linear, x_cyclic, lin_encoder, cyc_encoder, decoder, lambdas)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(data_loader)
        return epoch_loss

    @staticmethod
    def train_linear_and_cyclic_vaes_for_1_epoch_tabular(data_loader: DataLoader, lin_encoder: nn.Module, cyc_encoder: nn.Module,
                    decoder: nn.Module, optimizer: torch.optim.Optimizer, lambdas: dict):
        """Full training loop over one epoch for the VAE.
        Args:
            data_loader : DataLoader yielding (x_lin, x_cyc). note x_lin and x_cyc are separated
            lin_encoder : Linear encoder
            cyc_encoder : Cyclic encoder
            decoder     : Decoder
            optimizer   : Optimizer
            lambdas     : dict of loss weights"""
        lin_encoder.train()
        cyc_encoder.train()
        decoder.train()

        epoch_loss = 0
        for x_linear, x_cyclic in data_loader:
            optimizer.zero_grad()
            loss = WithSplit.vae_train_step_for_tabular(x_linear, x_cyclic, lin_encoder, cyc_encoder, decoder, lambdas)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(data_loader)
        return epoch_loss


class NoSplit:
    @staticmethod
    @torch.no_grad()
    def encode_dataset_no_split(X_win: torch.Tensor, encoder: nn.Module, pooling: str | None = "mean") -> torch.Tensor:
        """Encode a dataset without linear/cyclic split.
        x : (num_windows, window_size, num_features)
        pooling : "mean" or None
        Returns : (num_windows, z_dim) if pooled, else (num_windows, window_size, z_dim)"""
        N_w, win, D = X_win.shape
        x_flat      = X_win.view(N_w, win*D)
        mu, _       = encoder(x_flat)
        
        if pooling is None:
            return mu
        if pooling == "mean":
            return mu.mean(dim=0, keepdim=True)  # average over windows
        raise ValueError(f"Unknown pooling: {pooling}")

    @staticmethod
    def vae_train_step_no_split(x: torch.Tensor, encoder: nn.Module, decoder: nn.Module, lambdas: dict) -> torch.Tensor:
        """Single VAE step for dataset without linear/cyclic split.
        x : (num_windows, window_size, num_features)"""
        N_w, win, D = x.shape
        x_flat      = x.view(N_w, win*D)

        mu, logvar  = encoder(x_flat)
        z           = Reparam.reparam_gaussian(mu, logvar)

        x_hat   = decoder(z)
        L_recon = F.mse_loss(x_hat, x_flat)
        L_kl    = kl_gaussian(mu, logvar)

        return lambdas["reconstr"]*L_recon + lambdas["euc"]*L_kl

    @staticmethod
    def train_vae_no_split(loader, encoder, decoder, optimizer, lambdas):
        encoder.train(); decoder.train()
        epoch_loss = 0
        for x, in loader:
            optimizer.zero_grad()
            loss = NoSplit.vae_train_step_no_split(x, encoder, decoder, lambdas)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(loader)
        return epoch_loss


# general
def early_stop(current_loss: float, best_loss: float, counter: int, patience: int = 5):
    """Quick early stopping tracker.
    Returns updated best_loss, counter, and stop flag."""
    if current_loss < best_loss:
        return current_loss, 0, False
    else:
        counter += 1
        stop = counter >= patience
        return best_loss, counter, stop

def fit_catboost_multi(X_train: Union[np.ndarray, torch.Tensor], y_train: Union[np.ndarray, torch.Tensor],
                       X_test: Union[np.ndarray, torch.Tensor], cb_verbose = 0) -> np.ndarray:
    """Fit CatBoost for multi-output regression. logging_level = 0, 1, 2 for silent, info, debug."""
    if isinstance(X_train, torch.Tensor):
        X_train = X_train.detach().cpu().numpy()
    if isinstance(y_train, torch.Tensor):
        y_train = y_train.detach().cpu().numpy()
    if isinstance(X_test, torch.Tensor):
        X_test = X_test.detach().cpu().numpy()

    # model = MultiOutputRegressor(CatBoostRegressor(verbose=cb_verbose))
    model = MultiOutputRegressor(CatBoostRegressor(iterations=300, depth=5, learning_rate=0.1, task_type="GPU", verbose=cb_verbose))
    model.fit(X_train, y_train)
    return model.predict(X_test)

def evaluate_model_full(Z_train: np.ndarray, Z_test: np.ndarray, y_train_win: np.ndarray, y_test_win: np.ndarray, z_dim_e: int,
                        X_cyc_test_w: torch.Tensor | None = None, recon_cyc_test: torch.Tensor | None = None) -> dict:
    """Regression + geometry-aware metrics. Lower is better unless stated."""
    metrics = {}

    # --- downstream ---
    y_hat           = fit_catboost_multi(Z_train, y_train_win, Z_test)
    metrics["rmse"] = root_mean_squared_error(y_test_win, y_hat)
    metrics["mae"]  = mean_absolute_error(y_test_win, y_hat)
    metrics["r2"]   = r2_score(y_test_win, y_hat)

    # --- cyclic reconstruction ---
    if recon_cyc_test is not None and X_cyc_test_w is not None:
        metrics["cyc_recon_mse"] = F.mse_loss(
            recon_cyc_test, X_cyc_test_w).item()

    # --- angular stability (torus only) ---
    z_torus = Z_test[:, z_dim_e:]
    if z_torus.shape[1] >= 2:
        angles = np.arctan2(z_torus[:, 1::2], z_torus[:, 0::2])
        metrics["theta_std"] = angles.std(axis=0).mean()
    return metrics

def _cyclic_recon_mse(x_cyc_true: torch.Tensor, x_cyc_recon: torch.Tensor) -> float:
    """MSE on cyclic features only."""
    return F.mse_loss(x_cyc_recon, x_cyc_true).item()

def _angular_std(z_torus: np.ndarray) -> float:
    """Mean angular std across all S¹ dimensions."""
    angles = np.arctan2(z_torus[:, 1::2], z_torus[:, 0::2])
    return angles.std(axis=0).mean()

def _phase_alignment(theta: np.ndarray, t: np.ndarray, P: float) -> float:
    ref = np.sin(2 * np.pi * t / P)
    return np.corrcoef(theta.flatten(), ref[:len(theta.flatten())])[0,1]


def estimate_entropy(x, bandwidth: float = 0.2) -> float:
    """KDE-based differential entropy estimator for 1D samples."""
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("estimate_entropy expects 1D input")
    x   = x[:, None]
    kde = KernelDensity(bandwidth=bandwidth).fit(x)
    return -kde.score_samples(x).mean()

def pool_latents(z: torch.Tensor, z_e_dim: int) -> torch.Tensor:
    """Pool mixed latents over windows, depending on latent type (spherical vs Euclidean)
    z: (batch, windows, z_total)
    z_e_dim: dimension of Euclidean part
    NOTE: assumes z is concat like so: [linear, cyclic]"""
    z_euclid = z[..., :z_e_dim]  # Euclidean
    z_spher  = z[..., z_e_dim:]  # Spherical
    z_e_mean = z_euclid.mean(dim=1)  # Euclidean mean
    z_s_mean = F.normalize(z_spher.sum(dim=1), dim=-1)  # Spherical mean
    return torch.cat([z_e_mean, z_s_mean], dim=-1)
