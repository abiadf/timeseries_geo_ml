"""TS2Vec"""
from typing import Dict, Any
from benchmarks.ts2vec_runner import run_ts2vec, log_ts2vec_results
from param_config.config_paths import ts2vec_hyperparam_file

def run_ts2vec_block(X_train, X_test, y_train_scaled, y_test_scaled, params: Dict[str, Any], desired_dataset: str,
                     window_size: int, device: str):
    """Run TS2Vec end-to-end and return metrics."""
    z_pooling_method   = params["ts2vec"]["z_pooling_method"]
    ts2vec_epochs      = params["ts2vec"]["ts2vec_epochs"]
    ts2vec_hidden_dims = 16
    ts2vec_depth       = 2
    patience           = 25
    ts2vec_lr          = 0.05
    ts2vec_latent_dims = 16
    ts2vec_batch_size  = 16

    model_cfg = {
        "z_pooling_method": z_pooling_method,
        "hidden_dims":      ts2vec_hidden_dims,
        "latent_dims":      ts2vec_latent_dims,
        "depth":            ts2vec_depth,}

    train_cfg = {
        "lr":          ts2vec_lr,
        "patience":    patience,
        "epochs":      ts2vec_epochs,
        "batch_size":  ts2vec_batch_size,
        "window_size": window_size,}

    losses, r2, metrics = run_ts2vec(X_train, X_test, y_train_scaled, y_test_scaled,
        model_cfg=model_cfg, train_cfg=train_cfg, device=device)
    log_ts2vec_results(desired_dataset, window_size, losses, r2, metrics, model_cfg, train_cfg,
                       filename=ts2vec_hyperparam_file)

    return losses, r2, metrics, model_cfg, train_cfg
