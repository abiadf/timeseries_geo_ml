"Barlow (CNN) runner"
from typing import Dict, Any, Tuple
from benchmarks.barlow_cnn_runner import BarlowCNNRunner

def run_barlow_cnn_block(X_train, X_test, y_train_scaled, y_test_scaled, params: Dict[str, Any],
                    desired_dataset: str, device: str) -> Tuple[Any, Any, Any, Any, Any, Dict[str, Any], Dict[str, Any]]:
    """Run Barlow CNN end-to-end: train, encode, regress, log results."""

    c              = params["barlow"]["cnn"]
    latent_dim     = c.get("latent_dim", 12)
    channels       = c.get("channels", [8, 32])
    kernel_size    = c.get("kernel_size", 7)
    pool_kernel    = c.get("pool_kernel", 2)
    ssl_lambda     = c.get("ssl_lambda", 0.1)
    ssl_weight     = c.get("ssl_weight", 0.1)
    recon_weight   = c.get("recon_weight", 0.5)
    augment_const  = c.get("augment_const", 0.05)
    head_dims_list = c.get("head_dims_list", [64, 32])

    epochs         = params["barlow"].get("epochs", 100)
    lr             = params["barlow"].get("lr", 0.01)
    rand_seed      = params.get("rand_seed", 0)

    model_cfg = {
        "latent_dim":     latent_dim,
        "channels":       channels,
        "kernel_size":    kernel_size,
        "pool_kernel":    pool_kernel,
        "ssl_lambda":     ssl_lambda,
        "ssl_weight":     ssl_weight,
        "recon_weight":   recon_weight,
        "augment_const":  augment_const,
        "head_dims_list": head_dims_list,}

    train_cfg = {
        "epochs":    epochs,
        "lr":        lr,
        "rand_seed": rand_seed,}

    losses, r2, metrics, recon_train, recon_test = BarlowCNNRunner.run_barlow_cnn(X_train, X_test, y_train_scaled,
                                            y_test_scaled, model_cfg=model_cfg, train_cfg=train_cfg, device=device)
    BarlowCNNRunner.log_barlow_cnn_results(dataset_name=desired_dataset, losses=losses, r2=r2, model_cfg=model_cfg,
                                train_cfg=train_cfg, metrics=metrics, filename="results/hyperparam_search_barlow.txt")

    return losses, r2, metrics, recon_train, recon_test, model_cfg, train_cfg
