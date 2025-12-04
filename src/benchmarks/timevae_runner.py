""""timevae running functions"""
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from src.utils.model_utils import profile_epoch
from src.utils.metrics_utils import Preds
import src.param_config.config_paths as P
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def _encode_timevae_in_batches(model: torch.nn.Module, X: np.ndarray, batch_size: int = 32, return_mean=False) -> np.ndarray:
    """Encode X to latent z in batches to avoid OOM."""
    zs = []
    device = next(model.parameters()).device
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i+batch_size]).float().to(device)
        # zs.append(z.cpu().numpy())
        z_mean, z_log_var, z_sample = model.encoder(xb)
        zs.append(z_mean.cpu().numpy() if return_mean else z_sample.cpu().numpy())
    return np.concatenate(zs, 0)


def run_timevae(X_train, X_test, y_train_scaled, y_test_scaled, *,
                timevae_file_path, device, batch_size, train_epochs, lr_training, latent_dim, hidden_layer_sizes,
                reconstruction_wt, desired_dataset, force_train=False):
    """Train or load TimeVAE, compute latent codes, downstream metrics, and final reconstruction losses on REAL train/test data.
    Returns:
        losses, r2, profiling_metrics,
        recon_loss_train, recon_loss_test,
        z_train, z_test"""
    # make sure timevae_torch/src is importable
    src_path = P.SRC_ROOT / "timevae_torch" / "src"
    if str(src_path) not in sys.path:
        sys.path.append(str(src_path))
    from vae_pipeline import run_vae_pipeline
    from vae.timevae import TimeVAE

    model_dir       = str(Path(timevae_file_path).parent)
    checkpoint_path = os.path.join(model_dir, "TimeVAE_weights.pth")
    model_exists    = os.path.exists(checkpoint_path)

    # Train or load
    if force_train or not model_exists:
        z_train, z_test, recon_loss_train, profiling_metrics, timevae_model = run_vae_pipeline(
            timevae_file_path, desired_dataset,
            vae_type="timeVAE", train_epochs=train_epochs,
            lr_training=lr_training, latent_dim=latent_dim,
            hidden_layer_sizes=hidden_layer_sizes, reconstruction_wt=reconstruction_wt)
        timevae_model = timevae_model.to(device)
        timevae_model.eval()
    else:
        # Load model from checkpoint
        try:
            timevae_model = TimeVAE(
                seq_len=X_train.shape[1], feat_dim=X_train.shape[2],
                latent_dim=latent_dim, hidden_layer_sizes=hidden_layer_sizes,
                batch_size=batch_size, reconstruction_wt=reconstruction_wt).to(device)
            timevae_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            timevae_model.eval()
            profiling_metrics = {}
            z_train = _encode_timevae_in_batches(timevae_model, X_train, batch_size=batch_size)
            z_test  = _encode_timevae_in_batches(timevae_model, X_test, batch_size=batch_size)
            recon_loss_train = None
        except RuntimeError:
            # Checkpoint mismatch; retrain safely
            print("Checkpoint mismatch detected; retraining model to match hyperparameters.")
            z_train, z_test, recon_loss_train, profiling_metrics, trained_model = run_vae_pipeline(
                timevae_file_path, desired_dataset,
                vae_type="timeVAE", train_epochs=train_epochs,
                lr_training=lr_training, latent_dim=latent_dim,
                hidden_layer_sizes=hidden_layer_sizes, reconstruction_wt=reconstruction_wt,
                return_model=True)
            timevae_model = trained_model.to(device)
            timevae_model.eval()
            model_exists = False

    # Compute final recon losses
    with torch.no_grad():
        X_train_t = torch.from_numpy(X_train).float().to(device)
        X_test_t  = torch.from_numpy(X_test).float().to(device)
        z_mean_tr, z_log_tr, z_samp_tr = timevae_model.encoder(X_train_t)
        z_mean_te, z_log_te, z_samp_te = timevae_model.encoder(X_test_t)
        recon_train = timevae_model.decoder(z_samp_tr)
        recon_test  = timevae_model.decoder(z_samp_te)
        recon_loss_train = F.mse_loss(recon_train, X_train_t).item()
        recon_loss_test  = F.mse_loss(recon_test,  X_test_t).item()

    # Clip extreme values and reshape
    z_train = np.clip(z_train, -1e3, 1e3).reshape(z_train.shape[0], -1)
    z_test  = np.clip(z_test, -1e3, 1e3).reshape(z_test.shape[0], -1)

    def _align_latent_and_target_length(z_train, z_test, y_train_scaled, y_test_scaled):
        """Ensure z and y have matching lengths by truncating the longer one."""
        min_test_len   = min(z_test.shape[0], y_test_scaled.shape[0])
        z_test         = z_test[:min_test_len]
        y_test_scaled  = y_test_scaled[:min_test_len]

        min_train_len  = min(z_train.shape[0], y_train_scaled.shape[0])
        z_train        = z_train[:min_train_len]
        y_train_scaled = y_train_scaled[:min_train_len]
        return z_train, z_test, y_train_scaled, y_test_scaled

    z_train, z_test, y_train_scaled, y_test_scaled = _align_latent_and_target_length(z_train, z_test, y_train_scaled, y_test_scaled)

    # Evaluate downstream predictors
    losses, rf_model = Preds().evaluate_models_on_dataset(z_train, y_train_scaled, z_test, y_test_scaled)
    r2 = rf_model.score(z_test, y_test_scaled)

    return losses, r2, profiling_metrics, recon_loss_train, recon_loss_test, z_train, z_test


def log_timevae_results(dataset_name, window_size, losses, r2, metrics, recon_loss,
                        model_cfg, train_cfg, filename=P.timevae_hyperparam_file):
    """Log TimeVAE results with hyperparameters, both to file and stdout."""
    linreg, catboost, rf = losses[:3]

    with open(filename, 'a') as f:
        f.write(f"timevae/{dataset_name}: hidden_layers={model_cfg['hidden_layers']} "
                f"latent_dim={model_cfg['latent_dim']} reconstr_wt={model_cfg['reconstruction_wt']} train_epochs={train_cfg['train_epochs']} lr={train_cfg['lr']} batch_size={train_cfg['batch_size']} window_size={window_size}\n")
        f.write(f"linreg={linreg:.4f} catboost={catboost:.4f} rf={rf:.4f} ")
        f.write(f"R²={r2:.3f} L_recons={recon_loss:.3f}\n")
        f.write("time & params & flops & memory\n")
        f.write(f"{metrics['runtime_s']:.3f} & {metrics['num_params_M']:.3f} & "
                f"{metrics['flops_M']:.3f} & {metrics['peak_memory_MB']:.3f}\n\n")

    print(f"timevae/{dataset_name}: hidden_layers={model_cfg['hidden_layers']} "
          f"latent_dim={model_cfg['latent_dim']} reconstruction_wt={model_cfg['reconstruction_wt']}")
    print(f"& Z (timevae) & {linreg:.4f} & {catboost:.4f} & {rf:.4f} ")
    print(f"R²: {r2:.3f}, L_recons: {recon_loss:.3f}")
    print(f"train_epochs={train_cfg['train_epochs']} lr={train_cfg['lr']} batch_size={train_cfg['batch_size']}")
    print("time & params & flops & memory")
    print(f"{metrics['runtime_s']:.3f} & {metrics['num_params_M']:.3f} & "
          f"{metrics['flops_M']:.3f} & {metrics['peak_memory_MB']:.3f}\n")


