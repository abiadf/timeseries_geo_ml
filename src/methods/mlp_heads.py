import __main__
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import root_mean_squared_error

from src.encoders.latents import Latents
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


def make_MLP_regression_head(embedding_dim: int, hidden_dims_list: list,
                             y_train_tensor: torch.Tensor, dropout: float, device: str):
    """Creates a variable-layer MLP regression head"""
    layers = []
    in_dim = embedding_dim
    for hidden_layer in hidden_dims_list:
        layers.append(nn.Linear(in_dim, hidden_layer))
        layers.append(nn.SiLU())
        layers.append(nn.Dropout(dropout))
        in_dim = hidden_layer
    layers.append(nn.Linear(in_dim, y_train_tensor.shape[1]))
    return nn.Sequential(*layers).to(device)

def evaluate_MLP_regressor(model: nn.Module, z_train: torch.Tensor, z_test: torch.Tensor, y_train: torch.Tensor,
                       y_test: torch.Tensor, epochs: int, lr: float, device: str):
    """Train a regression MLP on z and return RMSE on test split."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    # z_train = z_train.to(device).float()
    # z_test  = z_test.to(device).float()
    # y_train = y_train.to(device).float()
    # y_test  = y_test.to(device).float()
    z_train = torch.tensor(z_train, device=device, dtype=torch.float32)
    z_test  = torch.tensor(z_test, device=device, dtype=torch.float32)
    y_train = torch.tensor(y_train, device=device, dtype=torch.float32)
    y_test  = torch.tensor(y_test, device=device, dtype=torch.float32)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(z_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 2 == 0:
            print(f"{epoch} | loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        test_pred = model(z_test)
        test_rmse = torch.sqrt(loss_fn(test_pred, y_test)).item()
    return test_rmse

