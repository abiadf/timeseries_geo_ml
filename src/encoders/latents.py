import __main__

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from typing import Dict, List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# moved from previously unused part of the notebok code
def latent_target_corr_multi(z: np.ndarray, y: np.ndarray, max_n: int = 3000) -> List[float]:
    """Compute Spearman correlation between pairwise distances in z
    and absolute differences in each y-dimension.
    z: (n, d) latent array
    y: (n, m) targets
    max_n: subsample size for speed
    
    Returns: list of correlations, length m"""
    n = len(y)
    if n > max_n:
        idx = np.random.choice(n, size=max_n, replace=False)
        z, y = z[idx], y[idx]
    # latent distances
    pdist = np.sqrt(((z[:, None, :] - z[None, :, :])**2).sum(-1)).ravel()
    corrs = []
    for j in range(y.shape[1]):
        ydist = np.abs(y[:, None, j] - y[None, :, j]).ravel()
        corr, _ = spearmanr(pdist, ydist)
        corrs.append(float(corr))
    return corrs

class Latents:
    @staticmethod
    def flatten_X(X: np.ndarray) -> np.ndarray:
        """Flatten 2D or 3D X to (N, D) for torch feeding."""
        if X.ndim > 2:
            return X.reshape(len(X), -1)
        return X

    @staticmethod
    def get_latent_from_encoder(encoder, X, device="cpu") -> np.ndarray:
        """Return latent z for any encoder type with shape (N, latent_dim)."""
        if isinstance(encoder, nn.Module):
            X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
            if X_tensor.ndim > 2:
                X_tensor = X_tensor.reshape(len(X_tensor), -1)
            if type(encoder).__name__ == "VAE":
                mu, _ = encoder.encode(X_tensor)
                z     = mu.detach().cpu().numpy()
            else: # AE / DAE: return latent layer instead of reconstruction
                z = encoder.encode(X_tensor).detach().cpu().numpy()
        elif type(encoder).__name__ == "TS2VecEncoder":
            z = encoder.encode(X)  # returns (N, T, latent_dim)
            z = z.mean(axis=1)      # temporal pooling
        else:
            raise ValueError(f"Unknown encoder type: {type(encoder).__name__}")
        return z

    @staticmethod
    def get_concat_latents(encoders_dict, X, device=device):
        """Return concatenated latent vectors from all encoders for X."""
        latents = [Latents.get_latent_tensor(encoder, X, train_encoder=False, device=device).cpu().numpy()
                for encoder in encoders_dict.values()]
        return np.concatenate(latents, axis=1)  # shape (N, sum(latent_dims))

    @staticmethod
    def get_weighted_latents(encoders_dict, X, encoder_weights=None, device=device) -> np.ndarray:
        """Return concatenated latent vectors from all encoders, optionally weighted and normalized per encoder.
        - encoders_dict: dict of encoders {name: model}
        - X: input array, shape (N, T, F) or (N, F)
        - encoder_weights: dict {name: float}, defaults to 1.0 if None"""
        latents = []
        for name, encoder in encoders_dict.items():
            z = Latents.get_latent_tensor(encoder, X, train_encoder=False, device=device).cpu().numpy()
            # normalize per encoder to [0,1]
            z      = (z - z.min(axis=0, keepdims=True)) / (z.max(axis=0, keepdims=True) - z.min(axis=0, keepdims=True) + 1e-8)
            weight = 1.0 if encoder_weights is None else encoder_weights.get(name, 1.0)
            latents.append(z * weight)
        return np.concatenate(latents, axis=1)

    @staticmethod
    def get_latent_tensor(encoder, X, train_encoder=False, device=device) -> torch.Tensor:
        """Return 2D tensor (N, latent_dim) for SupHead/CatBoost."""
        if train_encoder and isinstance(encoder, nn.Module):
            encoder.train()
            X_tensor = torch.tensor(Latents.flatten_X(X), dtype=torch.float32, device=device)
            z = encoder.encode(X_tensor)
            if isinstance(z, tuple):  # for VAE
                z = z[0]
            if z.ndim > 2:
                z = z.mean(dim=1)
            return z
        else:
            z = Latents.get_latent_from_encoder(encoder, X, device=device)
            if z.ndim > 2:
                z = z.mean(axis=1)
            return torch.tensor(z, dtype=torch.float32, device=device) if isinstance(z, np.ndarray) else z
