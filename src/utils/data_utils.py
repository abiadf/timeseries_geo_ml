import __main__
import numpy as np
import torch
import torch.nn.functional as F
from pycatch22 import catch22_all

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def catch22_features_from_windows(X_windows: np.ndarray, y_windows: np.ndarray, which_y: str) -> tuple[np.ndarray, np.ndarray]:
    """Apply catch22 to each window/channel and return (features, targets).
    X_windows: shape (n_windows, window_size, n_channels)
    y_windows: shape (n_windows, window_size, n_targets)"""
    
    n_windows, _, n_channels = X_windows.shape
    feats = np.empty((n_windows, n_channels * 22), dtype=float)
    
    for i in range(n_windows):
        channel_feats = [catch22_all(X_windows[i, :, ch])['values']
                         for ch in range(n_channels)]
        feats[i] = np.concatenate(channel_feats)
    
    # Take last value in each target window
    if which_y == "last":
        y_out = y_windows[:, -1, :]
    elif which_y == "mean":
        y_out = np.mean(y_windows, axis=1)
    return feats, y_out


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


def select_top_X_features(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, top_features_pct: int):
    """Select top features via mutual info along last dimension. 
    - 3D X: (stations, timesteps, features), keeps last dim shrunk
    - 2D X: (samples, features)
    Removes rows with all-zero before scoring"""
    from sklearn.feature_selection import mutual_info_regression

    X_dims = len(X_train.shape)
    if X_dims == 3:
        _, n_timesteps, n_features = X_train.shape
        X_flat = X_train.reshape(-1, n_features)
        y_flat = np.repeat(y_train.mean(axis=1), n_timesteps, axis=0)
    elif X_dims == 2:
        n_features = X_train.shape[1]
        X_flat = X_train
        y_flat = y_train.mean(axis=1) if y_train.ndim > 1 else y_train
    else:
        raise ValueError("X_train must be 2D or 3D")

    mask = ~(np.all(X_flat == 0, axis=1))
    X_flat, y_flat = X_flat[mask], y_flat[mask]

    mi         = np.array([mutual_info_regression(X_flat[:, [i]], y_flat)[0] for i in range(n_features)])
    k_features = max(1, int(top_features_pct * n_features / 100))
    top_idx    = np.argsort(mi)[-k_features:]

    if X_dims == 3:
        X_train_sel = X_train[:, :, top_idx]
        X_test_sel  = X_test[:, :, top_idx]
    else:
        X_train_sel = X_train[:, top_idx]
        X_test_sel  = X_test[:, top_idx]
    print(f"Keeping top {k_features}/{n_features} features")
    return X_train_sel, X_test_sel, top_idx



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

