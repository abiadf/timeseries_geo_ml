""""TS2Vec running functions"""
import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset 
from utils.model_utils import profile_epoch
from utils.metrics_utils import Preds
from pathlib import Path

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
                timevae_file_path, device, batch_size, train_epochs, lr_training, latent_dim,
                hidden_layer_sizes, reconstruction_wt, desired_dataset, train=True):
    """Train (or load) TimeVAE, encode X in batches, evaluate predictors, and return metrics.
    Returns: losses, r2, profiling_metrics"""
    # sys.path.append("./timevae_torch/src")
    # from vae_pipeline import run_vae_pipeline
    # from vae.timevae import TimeVAE

    notebook_dir = Path().resolve()  # the directory of your notebook
    src_path = notebook_dir / "timevae_torch" / "src"
    if str(src_path) not in sys.path:
        sys.path.append(str(src_path))
    from vae_pipeline import run_vae_pipeline
    from vae.timevae import TimeVAE

    if train: # always train a new model
        z_train, z_test, timevae_recon_loss, profiling_metrics = run_vae_pipeline(
            timevae_file_path, desired_dataset,
            vae_type="timeVAE",
            train_epochs=train_epochs,
            lr_training=lr_training,
            latent_dim=latent_dim,
            hidden_layer_sizes=hidden_layer_sizes,
            reconstruction_wt=reconstruction_wt)
        timevae_model = None
    else: # optionally, load existing model
        timevae_model = TimeVAE.load(timevae_file_path.replace(".npz", "_model")).to(device).eval()
        timevae_model._print_model_param_summary()
        z_train = _encode_timevae_in_batches(timevae_model, X_train, batch_size=batch_size)
        z_test  = _encode_timevae_in_batches(timevae_model, X_test, batch_size=batch_size)
        timevae_recon_loss = None
        profiling_metrics = {}

    # print("z_train stats: ", np.nanmin(z_train), np.nanmax(z_train), np.isnan(z_train).any(), np.isinf(z_train).any())
    # print("z_test stats: ", np.nanmin(z_test), np.nanmax(z_test), np.isnan(z_test).any(), np.isinf(z_test).any())    

    # fondue_latent_dim = DimensionalityEstimator.estimate_latent_dim_using_fondue(z_train, z_train, verbose=True)
    # active_dims_mask  = DimensionalityEstimator.prune_latent_dims(z_train, threshold_frac=0.05)
    # z_train = z_train[:, active_dims_mask]
    # z_test  = z_test[:, active_dims_mask]

    # Clip extreme values
    z_train = np.clip(z_train, -1e3, 1e3)
    z_test  = np.clip(z_test, -1e3, 1e3)

    # after encoding
    z_train = z_train.reshape(z_train.shape[0], -1)  # (n_samples, sequence_length * latent_dim)
    z_test  = z_test.reshape(z_test.shape[0], -1)

    losses, rf_model = Preds().evaluate_models_on_dataset(z_train, y_train_scaled, z_test, y_test_scaled)
    r2 = rf_model.score(z_test, y_test_scaled)

    # # Convert to tensors
    # z_train_tensor = torch.tensor(z_train, dtype=torch.float32).to(device)
    # z_test_tensor  = torch.tensor(z_test,  dtype=torch.float32).to(device)
    # y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
    # y_test_tensor  = torch.tensor(y_test_scaled,  dtype=torch.float32).to(device)
    # Train + evaluate neural regressor
    # embedding_dim   = z_train_tensor.shape[1]
    # regression_head = make_MLP_regression_head(embedding_dim, layer1_dim, layer2_dim, layer3_dim,
    #                                        y_train_tensor, dropout, device)
    # nn_loss = evaluate_MLP_regressor(regression_head, z_train_tensor, z_test_tensor,
    #                              y_train_tensor, y_test_tensor, regressor_epochs, lr_regressor)
    # timevae_losses.append(nn_loss)
    return losses, r2, profiling_metrics, timevae_recon_loss, z_train, z_test

def log_timevae_results(dataset_name, window_size, losses, r2, metrics, recon_loss,
                        model_cfg, train_cfg, filename="results/hyperparam_search.txt"):
    """Log TimeVAE results with hyperparameters, both to file and stdout."""
    linreg, catboost, rf = losses[:3]

    with open(filename, 'a') as f:
        f.write(f"timevae/{dataset_name}: hidden_layers={model_cfg['hidden_layers']} "
                f"latent_dim={model_cfg['latent_dim']} reconstr_wt={model_cfg['reconstruction_wt']} train_epochs={train_cfg['train_epochs']} lr={train_cfg['lr']} batch_size={train_cfg['batch_size']} window_size={window_size}\n")
        f.write(f"linreg={linreg:.4f} catboost={catboost:.4f} rf={rf:.4f}")
        f.write(f"R²={r2:.3f} L_recons={recon_loss:.3f}\n")
        f.write("time & params & flops & memory\n")
        f.write(f"{metrics['runtime_s']:.3f} & {metrics['num_params_M']:.3f} & "
                f"{metrics['flops_M']:.3f} & {metrics['peak_memory_MB']:.3f}\n\n")

    print(f"timevae/{dataset_name}: hidden_layers={model_cfg['hidden_layers']} "
          f"latent_dim={model_cfg['latent_dim']} reconstruction_wt={model_cfg['reconstruction_wt']}")
    print(f"& Z (timevae) & {linreg:.4f} & {catboost:.4f} & {rf:.4f}")
    print(f"R²: {r2:.3f}, L_recons: {recon_loss:.3f}")
    print(f"train_epochs={train_cfg['train_epochs']} lr={train_cfg['lr']} batch_size={train_cfg['batch_size']}")
    print("time & params & flops & memory")
    print(f"{metrics['runtime_s']:.3f} & {metrics['num_params_M']:.3f} & "
          f"{metrics['flops_M']:.3f} & {metrics['peak_memory_MB']:.3f}\n")

