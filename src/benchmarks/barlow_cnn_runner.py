import __main__
import torch
import torch.nn as nn
import torch.nn.functional as F
from methods.mlp_heads import make_MLP_regression_head
from utils.metrics_utils import Preds
from utils.data_utils import Augmentations
from encoders.cnn import CnnAutoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BarlowCNNRunner:
    @staticmethod
    def encode_in_batches(model: nn.Module, X: torch.Tensor, batch_size: int, device: str) -> torch.Tensor:
        """Encode X in batches to avoid OOM."""
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, X.size(0), batch_size):
                batch = X[i:i+batch_size].to(device)
                out.append(model.encode(batch).detach().cpu())
        return torch.cat(out, dim=0)

    @staticmethod
    def compute_barlow_loss(z1: torch.Tensor, z2: torch.Tensor, ssl_lambda: float) -> torch.Tensor:
        """Compute Barlow Twins loss."""
        B, _     = z1.shape

        # ---- Spherical normalization (L2 onto unit sphere) ----
        # z1 = z1 / (z1.norm(dim=-1, keepdim=True) + 1e-12)
        # z2 = z2 / (z2.norm(dim=-1, keepdim=True) + 1e-12)
        # ========
        z1       = (z1 - z1.mean(0)) / (z1.std(0) + 1e-12)
        z2       = (z2 - z2.mean(0)) / (z2.std(0) + 1e-12)

        # corr. matrix:
        c        = (z1.T @ z2) / B
        on_diag  = (torch.diag(c) - 1).pow(2).sum()
        off_diag = (c - torch.diag(torch.diag(c))).pow(2).sum()
        return on_diag + ssl_lambda * off_diag

    @staticmethod
    def train_encoder(cnn: nn.Module, X: torch.Tensor, *, epochs: int, lr: float,
                      ssl_lambda: float, ssl_weight: float, recon_weight: float,
                      augment_const: float, device: str, rand_seed: int) -> dict:
        """Train CNN encoder with Barlow Twins + reconstruction loss, log per-epoch losses."""
        optimizer    = torch.optim.AdamW(cnn.parameters(), lr=lr)
        ssl_losses   = []
        recon_losses = []
        cnn.train()

        for epoch in range(epochs):
            optimizer.zero_grad()
            v1, v2     = Augmentations.make_two_views_augmentation(X, device, augment_const, seed=rand_seed)
            z1         = cnn.encode(v1)
            z2         = cnn.encode(v2)
            ssl_loss   = BarlowCNNRunner.compute_barlow_loss(z1, z2, ssl_lambda)
            recon_loss = F.mse_loss(cnn.decode(z1), v1) + F.mse_loss(cnn.decode(z2), v2)
            loss       = ssl_weight * ssl_loss + recon_weight * recon_loss
            loss.backward()
            optimizer.step()

            ssl_losses.append(ssl_loss.item())
            recon_losses.append(recon_loss.item())

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"[Barlow CNN] epoch {epoch}, loss={loss.item():.4f} "
                    f"(SSL={ssl_loss.item():.4f}, Recon={recon_loss.item():.4f})")
        cnn.eval()
        return {"ssl_losses": ssl_losses, "recon_losses": recon_losses}

    @staticmethod
    def run_barlow_cnn(X_train, X_test, y_train, y_test, *, model_cfg: dict, train_cfg: dict, device):
        """Train Barlow CNN encoder, evaluate models, compute final reconstruction losses.
        Returns (losses, r2, metrics, final_recon_train, final_recon_test)."""
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_test  = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
        y_test  = torch.tensor(y_test, dtype=torch.float32).to(device)

        n_rows, n_cols = X_train.shape[1], X_train.shape[2]

        cnn = CnnAutoencoder(
            n_cols, n_rows, model_cfg["latent_dim"],
            channels=model_cfg["channels"],
            kernel_size=model_cfg["kernel_size"],
            pool_kernel=model_cfg["pool_kernel"]).to(device)

        # train encoder with Barlow + reconstruction loss
        metrics = BarlowCNNRunner.train_encoder(
            cnn, X_train,
            epochs=train_cfg["epochs"],
            lr=train_cfg["lr"],
            ssl_lambda=model_cfg["ssl_lambda"],
            ssl_weight=model_cfg["ssl_weight"],
            recon_weight=model_cfg["recon_weight"],
            augment_const=model_cfg["augment_const"],
            device=device,
            rand_seed=train_cfg["rand_seed"])

        cnn.eval()
        with torch.no_grad():
            z_train = cnn.encode(X_train)
            z_test  = cnn.encode(X_test)

            final_recon_train = F.mse_loss(cnn.decode(z_train), X_train).item()
            final_recon_test  = F.mse_loss(cnn.decode(z_test), X_test).item()

            # convert latent codes to numpy for downstream models
            z_train_np = z_train.cpu().numpy()
            z_test_np  = z_test.cpu().numpy()
            y_train_np = y_train.cpu().numpy()
            y_test_np  = y_test.cpu().numpy()

            # evaluate downstream models (RMSE, R2, etc.)
            losses, rf_model = Preds().evaluate_models_on_dataset(z_train_np, y_train_np, z_test_np, y_test_np)
            r2 = rf_model.score(z_test_np, y_test_np)

            # MLP head evaluation
            head  = make_MLP_regression_head(z_train_np.shape[1], model_cfg["head_dims_list"], y_train, 0.0, device)
            head.eval()
            preds = head(torch.tensor(z_test_np, dtype=torch.float32, device=device))
            rmse  = torch.sqrt(F.mse_loss(preds, y_test)).item()
            losses.append(rmse)
        return losses, r2, metrics, final_recon_train, final_recon_test

    @staticmethod
    def log_barlow_cnn_results(dataset_name, losses, r2, model_cfg, train_cfg, metrics,
                                filename="results/hyperparam_search_barlow_cnn.txt"):
        """Log Barlow CNN results including last SSL and reconstruction loss."""
        rmse, linreg, catboost, rforest = losses[:4]
        last_ssl   = metrics["ssl_losses"][-1]
        last_recon = metrics["recon_losses"][-1]

        line1 = (
            f"barlow_cnn/{dataset_name}: "
            f"c1={model_cfg['channels'][0]} c2={model_cfg['channels'][1]} "
            f"k={model_cfg['kernel_size']} pool={model_cfg['pool_kernel']} "
            f"latent={model_cfg['latent_dim']} "
            f"sslλ={model_cfg['ssl_lambda']} sslw={model_cfg['ssl_weight']} "
            f"aug={model_cfg['augment_const']} "
            f"epochs={train_cfg['epochs']} lr={train_cfg['lr']} "
            f"(last SSL={last_ssl:.4f}, last Recon={last_recon:.4f})")

        with open(filename, "a") as f:
            f.write(line1 + "\n")
            f.write(f"& Z (barlow_cnn) & {rmse:.4f} & {linreg:.4f} & {catboost:.4f} & {rforest:.4f}\n")
            f.write(f"R² (barlow_cnn): {r2:.3f}\n\n")
        print(line1)
        print(f"& Z (barlow_cnn) & {rmse:.4f} & {linreg:.4f} & {catboost:.4f} & {rforest:.4f}")
        print(f"R² (barlow_cnn): {r2:.3f}\n")


