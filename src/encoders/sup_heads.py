import __main__
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import root_mean_squared_error

from encoders.latents import Latents

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _get_orthogonality_penalty(encoders, X_batch, device):
    """Compute sum of squared correlations between encoder latent batches.
       encoders: dict[name]->encoder; X_batch: same X for all encoders (or views applied externally)"""
    z_list = []
    for enc in encoders.values():
        z = Latents.get_latent_tensor(enc, X_batch, train_encoder=False, device=device)  # (B, d)
        z = z - z.mean(0)
        # l2-normalize per feature to reduce scale issues
        z = z / (z.std(0) + 1e-8)
        z_list.append(z)  # torch tensors
    # compute pairwise dot products of mean latent vectors (or flattened)
    penalty = 0.0
    for i in range(len(z_list)):
        for j in range(i+1, len(z_list)):
            # compute covariance between latent dims (sum of squared correlations)
            C = (z_list[i].T @ z_list[j]) / z_list[i].shape[0]  # (d_i, d_j)
            penalty = penalty + (C ** 2).sum()
    return penalty

def train_sup_head_per_encoder(encoder, X_L, y_L, X_test, y_test, dropout, train_encoder=True,
                               all_encoders=None, reg_ortho=5e-4,   # tune this
                               device="cpu", epochs=5, hidden_sizes=[64,32]):
    """Train a small supervised MLP head on top of EACH encoder's z.
    Args:
        encoder: AE/VAE/DAE/TS2Vec encoder
        X_L, y_L: labelled training data
        X_test, y_test: test data
        train_encoder: whether to finetune encoder
        device: "cpu"/"cuda"
        epochs: training epochs
        hidden_sizes: list of hidden layer sizes
    Returns:
        rmse on test set"""
    encoder_type  = type(encoder).__name__
    train_encoder = train_encoder and isinstance(encoder, nn.Module) and encoder_type in ["FlexibleAutoencoder","AE","VAE","DenoisingAE"]

    encoder.eval()
    z_sample   = Latents.get_latent_tensor(encoder, X_L[:2], train_encoder=False, device=device)
    latent_dim = z_sample.shape[-1]
    sup_head   = SupHead(latent_dim, y_L.shape[1], hidden_sizes, dropout).to(device)
    params     = list(sup_head.parameters())
    if train_encoder:
        params += list(encoder.parameters())
        # =========
        for other_name, other_enc in all_encoders.items():
            if other_enc is not encoder:
                params += list(other_enc.parameters())
        # =========
    optimizer  = torch.optim.AdamW(params, lr=1e-3)
    y_tensor   = torch.tensor(y_L, dtype=torch.float32, device=device)
    
    for _ in range(epochs):
        sup_head.train()
        if train_encoder:
            encoder.train()
        z      = Latents.get_latent_tensor(encoder, X_L, train_encoder=train_encoder, device=device)
        y_pred = sup_head(z)
        loss   = F.mse_loss(y_pred, y_tensor)

        # ====== add orthogonality penalty across encoder ensemble (very cheap)
        if all_encoders is not None and reg_ortho > 0:
            # use a small random batch for speed
            idx       = np.random.choice(len(X_L), size=min(128, len(X_L)), replace=False)
            X_batch   = X_L[idx]
            ortho_pen = _get_orthogonality_penalty(all_encoders, X_batch, device=device)
            loss      = loss + reg_ortho * ortho_pen
        # ======

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    sup_head.eval()
    with torch.no_grad():
        z_test      = Latents.get_latent_tensor(encoder, X_test, train_encoder=False, device=device)
        y_pred_test = sup_head(z_test).cpu().numpy()
    return root_mean_squared_error(y_test, y_pred_test)

def train_sup_heads_joint(encoders_dict, X_train, y_train, X_val, y_val,
                          hidden_sizes=[64], lr=0.001, epochs=50, device="cpu",
                          reg_ortho=1e-3, train_encoders=True, batch_size=128):
    """Jointly train supervised heads for each encoder with optional finetuning and orthogonality.
    Updates encoders_dict in-place with finetuned encoders.
    Returns dict of RMSE metrics."""
    metrics = {}
    for name, encoder in encoders_dict.items():
        rmse = train_sup_head_per_encoder(
            encoder, X_train, y_train, X_val, y_val,
            dropout=0.1, train_encoder=train_encoders,
            all_encoders=encoders_dict, reg_ortho=reg_ortho,
            device=device, epochs=epochs, hidden_sizes=hidden_sizes)
        metrics[name] = rmse
    return metrics


class SupHead(nn.Module):
    """Small supervised head: maps latent z -> target y"""
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes=[64, 32], dropout=1e-3):
        super().__init__()
        layers, prev_dim = [], input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


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

