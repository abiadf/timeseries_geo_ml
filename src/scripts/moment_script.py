"""MOMENT (centralized)
there is no regression task in MOMENT (see 'moment_model.task_name'), so we make our own head"""
from typing import Dict, Any, Tuple
from benchmarks.moment_runner import MomentRunner

def run_moment_block(X_train, X_test, y_train_scaled, y_test_scaled, params: Dict[str, Any],
                     desired_dataset: str, device: str) -> Tuple[Any, Any, Any, Dict[str, Any], Dict[str, Any]]:
    """Run MOMENT end-to-end: finetune, embed, regress, and log results."""

    # Hyperparameters
    model_name        = params["moment"]["model_name"]
    model_task        = params["moment"]["model_task"]        # "classification" (= regression)
    latent_layers     = params["moment"]["hidden_layer_sizes"]
    unfreeze_last_n   = params["moment"]["unfreeze_last_n"]
    dropout           = params["moment"]["predictor_dropout"]

    epochs_finetune   = params["moment"]["epochs_finetune"]
    lr_moment_encoder = params["moment"]["lr_encoder"]
    lr_moment_pred    = params["moment"]["lr_predictor"]
    batch_size        = params["moment"]["batch_size"]
    fine_tune_bool    = params["moment"]["fine_tune"]

    # Config dictionaries (same format as TimeVAE)
    model_cfg = {
        "model_name":      model_name,
        "task_name":       model_task,
        "hidden_layers":   latent_layers,
        "unfreeze_last_n": unfreeze_last_n,
        "dropout":         dropout,}

    train_cfg = {
        "epochs":     epochs_finetune,
        "batch_size": batch_size,
        "lr_encoder": lr_moment_encoder,
        "lr_head":    lr_moment_pred,
        "fine_tune":  fine_tune_bool,}

    losses, r2, metrics = MomentRunner.run_moment(X_train, X_test, y_train_scaled, y_test_scaled,
                                                  model_cfg=model_cfg, train_cfg=train_cfg, device=device)

    MomentRunner.log_moment_results(dataset_name=desired_dataset, losses=losses, r2=r2, model_cfg=model_cfg,
                                    train_cfg=train_cfg, filename="results/hyperparam_search_moment.txt")

    return losses, r2, metrics, model_cfg, train_cfg
