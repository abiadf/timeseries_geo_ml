import __main__
import logging

from catboost import CatBoostRegressor
from momentfm import MOMENTPipeline
from pycatch22 import catch22_all
from scipy.special import softmax

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(torch.cuda.memory_reserved(0) / 1e6, "MB reserved")
    print(torch.cuda.memory_allocated(0) / 1e6, "MB allocated")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.info("Starting process...")
logging.warning("Something looks off...")
logging.error("Something failed.")


class CnnAutoencoder(nn.Module):
    """Flexible CNN autoencoder for 1D time-series.
    Allows variable number of Conv+Pool layers."""
    def __init__(
        self,
        n_features: int,
        n_timesteps: int,
        latent_dim: int,
        channels: list[int] = [64, 128],
        kernel_size: int = 3,
        pool_kernel: int = 2,):
        super().__init__()
        self.n_features  = n_features
        self.n_timesteps = n_timesteps
        self.kernel_size = kernel_size
        self.pool_kernel = pool_kernel
        self.channels    = channels
        self.latent_dim  = latent_dim

        # --- ENCODER ---
        convs, pools = [], []
        in_ch = n_features
        for out_ch in channels:
            convs.append(nn.Conv1d(in_ch, out_ch, kernel_size))
            pools.append(nn.MaxPool1d(pool_kernel))
            in_ch = out_ch
        self.convs = nn.ModuleList(convs)
        self.pools = nn.ModuleList(pools)

        # --- Compute flat size dynamically ---
        with torch.no_grad():
            x = torch.zeros(1, n_timesteps, n_features).permute(0, 2, 1)
            for conv, pool in zip(self.convs, self.pools):
                x = pool(F.relu(conv(x)))
            self.flat_size = x.numel()
            self.final_channels = x.shape[1]
            self.final_time = x.shape[2]

        self.enc_linear = nn.Linear(self.flat_size, latent_dim)
        self.dec_linear = nn.Linear(latent_dim, self.flat_size)

        # --- DECODER ---
        deconvs = []
        rev_channels = channels[::-1]
        for i in range(len(rev_channels) - 1):
            deconvs.append(nn.ConvTranspose1d(rev_channels[i], rev_channels[i + 1], kernel_size))
        deconvs.append(nn.ConvTranspose1d(rev_channels[-1], n_features, kernel_size))
        self.deconvs = nn.ModuleList(deconvs)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # (B, F, T)
        for conv, pool in zip(self.convs, self.pools):
            x = pool(F.relu(conv(x)))
        z = self.enc_linear(x.reshape(x.size(0), -1))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.dec_linear(z))
        x = x.view(-1, self.final_channels, self.final_time)
        for deconv in self.deconvs[:-1]:
            x = F.interpolate(x, scale_factor=self.pool_kernel, mode="nearest")
            x = F.relu(deconv(x))
        x = F.interpolate(x, scale_factor=self.pool_kernel, mode="nearest")
        x = self.deconvs[-1](x)
        diff = self.n_timesteps - x.shape[2]
        if diff > 0:
            x = F.pad(x, (0, diff))
        elif diff < 0:
            x = x[:, :, :self.n_timesteps]
        return x.permute(0, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
