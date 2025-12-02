"""TimeVAE Script"""
from typing import Dict, Any, Tuple
from pathlib import Path
from benchmarks.timevae_runner import run_timevae, log_timevae_results

def run_timevae_block(
    X_train, X_test, y_train_scaled, y_test_scaled, timevae_file_path,
    params: Dict[str, Any], desired_dataset: str,
    window_size: int, device: str,
    force_train: bool = False) -> Tuple[Any, Any, Any, Dict[str, Any], Dict[str, Any]]:
    """Run TimeVAE end-to-end: train/load model, encode latent space, compute downstream metrics, and log results.
    Returns:
        losses, r2, profiling_metrics, model_cfg, train_cfg"""
    # Hyperparameters
    lr_training       = params["timevae"].get("lr_training", 0.001)
    latent_dim        = params["timevae"].get("latent_dim", 8)
    hidden_layer_sizes= params["timevae"].get("hidden_layer_sizes", [18, 12, 8])
    batch_size        = params["timevae"].get("batch_size", 64)
    reconstruction_wt = params["timevae"].get("reconstruction_wt", 3.5)
    train_epochs      = params["timevae"].get("train_epochs", 100)

    # Run TimeVAE
    losses, r2, profiling_metrics, recon_loss_train, recon_loss_test, z_train, z_test = run_timevae(
        X_train, X_test, y_train_scaled, y_test_scaled,
        timevae_file_path=timevae_file_path,
        device=device,
        batch_size=batch_size,
        train_epochs=train_epochs,
        lr_training=lr_training,
        latent_dim=latent_dim,
        hidden_layer_sizes=hidden_layer_sizes,
        reconstruction_wt=reconstruction_wt,
        desired_dataset=desired_dataset,
        force_train=force_train)   # True if you want to retrain

    # Config dictionaries
    model_cfg = {
        "hidden_layers": hidden_layer_sizes,
        "latent_dim": latent_dim,
        "reconstruction_wt": reconstruction_wt}

    train_cfg = {
        "train_epochs": train_epochs,
        "lr": lr_training,
        "batch_size": batch_size}

    log_timevae_results(
        dataset_name=desired_dataset,
        window_size=window_size,
        losses=losses,
        r2=r2,
        metrics=profiling_metrics,
        recon_loss=recon_loss_test,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        filename="results/hyperparam_search_timevae.txt")

    return losses, r2, profiling_metrics, model_cfg, train_cfg

