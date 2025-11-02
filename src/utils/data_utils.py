import __main__
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def assign_encoder_weights(encoders_dict: dict, sup_head_rmse, weight_encoding_method: str = "uniform"):
    """Compute normalized encoder weights using one of three methods:
        - "uniform": equal weights
        - "inverse_rmse": proportional to 1/RMSE
        - "softmax": softmax over 1/RMSE"""
    if weight_encoding_method == "uniform":
        encoder_weights = {name: 1.0 for name in encoders_dict.keys()}
        total           = sum(encoder_weights.values())
        encoder_weights = {k: v / total for k, v in encoder_weights.items()}
    elif weight_encoding_method == "inverse_rmse": # RMSE-based weights: better encoders get higher weight
        encoder_weights = {name: 1/rmse for name, rmse in sup_head_rmse.items()}
        total           = sum(encoder_weights.values())
        encoder_weights = {k: v/total for k,v in encoder_weights.items()}
    elif weight_encoding_method == "softmax": # softmax-based weights
        inv_rmse        = np.array([1/r for r in sup_head_rmse.values()])
        weights_softmax = np.exp(inv_rmse) / np.sum(np.exp(inv_rmse))
        encoder_weights = {name: w for name, w in zip(sup_head_rmse.keys(), weights_softmax)}
    return encoder_weights


class Bootstrapping:
    @staticmethod
    def bootstrap_sample(X, sample_frac=0.8):
        """Draw a bootstrap sample from X with replacement. Used to approximate sampling variability when the
        true population dist is unknown
            - X: Input array of shape (n_samples, ...).
            - sample_frac: Fraction of samples to draw (default=0.8).
            - returns: bootstrap sample array of shape (int(n_samples * sample_frac), ...)"""
        idx = np.random.choice(len(X), size=int(len(X)*sample_frac), replace=True)
        return X[idx]

    @staticmethod
    def train_ae_with_bootstraps(model, X_train, num_epochs=5, lr=1e-3, sample_frac=0.8, weight_decay=0.0, device="cpu"):
        """Train AE with bootstrap sampling."""
        optimizer     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        X_tensor_full = torch.tensor(X_train, dtype=torch.float32, device=device).reshape(len(X_train), -1)
        
        for _ in range(num_epochs):
            X_boot        = Bootstrapping.bootstrap_sample(X_train, sample_frac)
            X_tensor_boot = torch.tensor(X_boot, dtype=torch.float32, device=device).reshape(len(X_boot), -1)
            optimizer.zero_grad()
            X_recon = model(X_tensor_boot)
            loss    = F.mse_loss(X_recon, X_tensor_boot)
            loss.backward()
            optimizer.step()
        return model


class Slicing:
    """Slice arrays to feed into encoder"""
    @staticmethod
    def get_sliced_data(X: np.ndarray, num_slices: int, slice_idx: int) -> np.ndarray:
        """Return a station slice of X for a given encoder.
        Slices along 'pages' (axis 0)."""
        pages, _, _ = X.shape
        slice_len   = pages // num_slices
        start, end  = slice_idx * slice_len, (slice_idx + 1) * slice_len
        return X[start:end, :, :]

    @staticmethod
    def get_weighted_slices(X, weights):
        """Split X along pages axis proportional to AE weights."""
        weights = np.array(weights) / np.sum(weights)
        cumsum  = np.cumsum(np.round(weights * X.shape[0])).astype(int)
        starts  = np.concatenate(([0], cumsum[:-1]))
        return [X[start:end] for start, end in zip(starts, cumsum)]

    @staticmethod
    def get_weighted_slices_sqrt(X: np.ndarray, latent_dims: list[int]) -> list[np.ndarray]:
        """Split X along pages axis using sqrt(latent_dim) weighting."""
        pages   = X.shape[0]
        weights = np.sqrt(np.array(latent_dims))
        weights = weights / weights.sum()
        cum     = np.cumsum(np.round(weights * pages)).astype(int)
        starts  = np.concatenate(([0], cum[:-1]))
        return [X[start:end] for start, end in zip(starts, cum)]

