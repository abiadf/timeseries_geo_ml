""""timevae running functions"""
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
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


def run_timevae(
    X_train, X_test, y_train_scaled, y_test_scaled, *,
    timevae_file_path, device, batch_size, train_epochs, lr_training, latent_dim, hidden_layer_sizes,
    reconstruction_wt, desired_dataset, train=True):
    """Train or load TimeVAE, compute latent codes, downstream metrics, and final reconstruction losses on REAL train/test data.
    Returns:
        losses, r2, profiling_metrics,
        recon_loss_train, recon_loss_test,
        z_train, z_test"""
    # make sure timevae_torch/src is importable
    notebook_dir = Path().resolve()
    src_path = notebook_dir / "timevae_torch" / "src"
    if str(src_path) not in sys.path:
        sys.path.append(str(src_path))

    from vae_pipeline import run_vae_pipeline
    from vae.timevae import TimeVAE

    if train:
        z_train, z_test, timevae_recon_loss_train, profiling_metrics, timevae_model = run_vae_pipeline(
            timevae_file_path, desired_dataset,
            vae_type="timeVAE", train_epochs=train_epochs,
            lr_training=lr_training,
            latent_dim=latent_dim,
            hidden_layer_sizes=hidden_layer_sizes,
            reconstruction_wt=reconstruction_wt)

        # load the freshly trained model from the folder
        model_dir = str(Path(timevae_file_path).parent)
        timevae_model = TimeVAE(
            seq_len=X_train.shape[1],
            feat_dim=X_train.shape[2],
            latent_dim=latent_dim,
            hidden_layer_sizes=hidden_layer_sizes,
            batch_size=batch_size,
            reconstruction_wt=reconstruction_wt).to(device)
        weights_path = os.path.join(model_dir, "TimeVAE_weights.pth")
        timevae_model.load_state_dict(torch.load(weights_path, map_location=device))
        timevae_model.eval()
    else:
        # pkl_path = timevae_file_path.replace(".npz", ".pkl")
        # timevae_model = TimeVAE.load(pkl_path).to(device).eval()
        model_dir = str(Path(timevae_file_path).parent)
        timevae_model = TimeVAE(
            seq_len=X_train.shape[1],
            feat_dim=X_train.shape[2],
            latent_dim=latent_dim,
            hidden_layer_sizes=hidden_layer_sizes,
            batch_size=batch_size,
            reconstruction_wt=reconstruction_wt).to(device)

        weights_path = os.path.join(model_dir, "TimeVAE_weights.pth")
        timevae_model.load_state_dict(torch.load(weights_path, map_location=device))
        timevae_model.eval()

        timevae_model._print_model_param_summary()
        z_train = _encode_timevae_in_batches(timevae_model, X_train, batch_size=batch_size)
        z_test  = _encode_timevae_in_batches(timevae_model, X_test,  batch_size=batch_size)
        timevae_recon_loss_train = None
        profiling_metrics = {}

    with torch.no_grad():
        X_train_t = torch.from_numpy(X_train).float().to(device)
        X_test_t  = torch.from_numpy(X_test).float().to(device)

        # encode
        z_mean_tr, z_log_tr, z_samp_tr = timevae_model.encoder(X_train_t)
        z_mean_te, z_log_te, z_samp_te = timevae_model.encoder(X_test_t)

        # decode
        recon_train = timevae_model.decoder(z_samp_tr)
        recon_test  = timevae_model.decoder(z_samp_te)

        recon_loss_train = F.mse_loss(recon_train, X_train_t).item()
        recon_loss_test  = F.mse_loss(recon_test,  X_test_t).item()

    # the pipeline already returned a training loss, so overwrite with consistent metric
    timevae_recon_loss_train = recon_loss_train

    z_train = np.clip(z_train, -1e3, 1e3).reshape(z_train.shape[0], -1)
    z_test  = np.clip(z_test,  -1e3, 1e3).reshape(z_test.shape[0], -1)

    losses, rf_model = Preds().evaluate_models_on_dataset(z_train, y_train_scaled, z_test, y_test_scaled)
    r2               = rf_model.score(z_test, y_test_scaled)

    return (
        losses,
        r2,
        profiling_metrics,
        timevae_recon_loss_train,   # final train reconstruction
        recon_loss_test,            # final test reconstruction
        z_train,
        z_test)


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


# if __name__ == "__main__":
#     from utils.io_utils import JSONLogger, Notifiers, read_yaml_params, set_all_rand_seeds
#     from param_config.config_paths import interim_data_loc, public_data_loc, encoders_folder, params_path

#     params = read_yaml_params(params_path)

#     def main(desired_dataset, timevae_file_path, window_size):
#         """TimeVAE"""

#         if params["run_console"]["timevae"]:
#             # train_epochs = data_params["general"]["train_epochs"]
#             # lr_training  = data_params[desired_dataset]["timevae"]["lr_training"]
#             # latent_dim   = data_params[desired_dataset]["timevae"]["latent_dim"]
#             # hidden_layer_sizes= data_params[desired_dataset]["timevae"]["hidden_layer_sizes"]
#             # batch_size        = data_params[desired_dataset]["timevae"]["batch_size"]
#             # reconstruction_wt = data_params[desired_dataset]["timevae"]["reconstruction_wt"]
#             train_epochs = 100
#             lr_training  = 0.05
#             latent_dim   = 8
#             hidden_layer_sizes= [12, 16, 20]
#             batch_size        = 1024
#             reconstruction_wt = 3.5

#             losses, r2, metrics, recon_loss, z_train, z_test = run_timevae(
#                 X_train, X_test, y_train_scaled, y_test_scaled,
#                 timevae_file_path=timevae_file_path,
#                 device=device,
#                 batch_size=batch_size,
#                 train_epochs=train_epochs,
#                 lr_training=lr_training,
#                 latent_dim=latent_dim,
#                 hidden_layer_sizes=hidden_layer_sizes,
#                 reconstruction_wt=reconstruction_wt, desired_dataset=desired_dataset)
            
#             model_cfg = {"hidden_layers": hidden_layer_sizes,
#                         "latent_dim": latent_dim,
#                         "reconstruction_wt": reconstruction_wt}

#             train_cfg = {"train_epochs": train_epochs,
#                         "lr": lr_training,
#                         "batch_size": batch_size}

#             log_timevae_results(desired_dataset, window_size, losses, r2, metrics, recon_loss, model_cfg, train_cfg,
#                                 filename="results/hyperparam_search_timevae.txt")

#             return losses, r2, metrics, recon_loss, z_train, z_test

