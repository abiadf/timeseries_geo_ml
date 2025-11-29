""""TS2Vec running functions"""
import torch
from torch.utils.data import DataLoader, TensorDataset 
from encoders.ts2vec_encoder import TS2VecEncoder
from utils.model_utils import profile_epoch
from utils.metrics_utils import Preds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_ts2vec(X_train, X_test, y_train_scaled, y_test_scaled, *,
               model_cfg: dict, train_cfg: dict, device, save_ts2vec_encoder=False,):
    """Train TS2Vec with config dicts. Returns (losses, profiling_metrics)."""

    ts2vec = TS2VecEncoder(
        z_pooling= model_cfg["z_pooling_method"],
        lr       = train_cfg["lr"],
        device   = device,
        patience = train_cfg["patience"],
        max_train_length=train_cfg["window_size"],)

    metrics = ts2vec.fit_ts2vec(X_train,
        hidden_dims= model_cfg["hidden_dims"],
        output_dims= model_cfg["latent_dims"],
        depth      = model_cfg["depth"],
        batch_size = train_cfg["batch_size"],
        n_epochs   = train_cfg["epochs"],)

    z_train = ts2vec.encode(X_train, pooling=None)
    z_test  = ts2vec.encode(X_test,  pooling=None)

    z_train_flat = z_train.reshape(len(z_train), -1)
    z_test_flat  = z_test.reshape(len(z_test),  -1)

    losses, rf_model= Preds().evaluate_models_on_dataset(z_train_flat, y_train_scaled, z_test_flat,  y_test_scaled)
    r2              = rf_model.score(z_test_flat, y_test_scaled)
    # ====== supervised MLP predictor ======
    # z_train_tensor = torch.tensor(z_train_flat, dtype=torch.float32, device=device)
    # z_test_tensor  = torch.tensor(z_test_flat, dtype=torch.float32, device=device)
    # y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32, device=device)
    # y_test_tensor  = torch.tensor(y_test_scaled, dtype=torch.float32, device=device)

    # embedding_dim   = z_train_tensor.shape[1]
    # regression_head = make_MLP_regression_head(embedding_dim, layer1_dim, layer2_dim, layer3_dim,
    #                                        y_train_tensor, dropout, device)
    # test_loss = evaluate_MLP_regressor(regression_head, z_train_tensor, z_test_tensor,
    #                                y_train_tensor, y_test_tensor, regressor_epochs, lr_regressor)
    # ts2vec_losses.append(test_loss)
    return losses, r2, metrics

def log_ts2vec_results(dataset_name, window_size, losses, r2, metrics, model_cfg, train_cfg, filename="results/hyperparam_search_ts2vec.txt"):
    """Log TS2Vec results in the desired format, both to file and stdout."""
    rmse, linreg, catboost = losses[:3]  # only 3 models

    with open(filename, 'a') as f:
        f.write(f"ts2vec/{dataset_name}: hidden_dims={model_cfg['hidden_dims']} "
                f"latent_dims={model_cfg['latent_dims']} depth={model_cfg['depth']} "
                f"batch_size={train_cfg['batch_size']} lr={train_cfg['lr']} "
                f"window_size={window_size}\n")
        f.write(f"& Z (ts2vec) & {rmse:.4f} & {linreg:.4f}  & {catboost:.4f}\n")
        f.write(f"R² (TS2Vec): {r2:.3f}\n")
        f.write("time & params & flops & memory \n")
        f.write(f"{metrics['runtime_s']:.3f} & {metrics['num_params_M']:.3f} & "
                f"{metrics['flops_M']:.3f} & {metrics['peak_memory_MB']:.3f}\n\n")

    print(f"ts2vec/{dataset_name}: hidden_dims={model_cfg['hidden_dims']} "
          f"latent_dims={model_cfg['latent_dims']} depth={model_cfg['depth']} "
          f"batch_size={train_cfg['batch_size']} lr={train_cfg['lr']} "
          f"window_size={window_size}")
    print(f"& Z (ts2vec) & {rmse:.4f} & {linreg:.4f}  & {catboost:.4f}")
    print(f"R² (TS2Vec): {r2:.3f}")
    print("time & params & flops & memory")
    print(f"{metrics['runtime_s']:.3f} & {metrics['num_params_M']:.3f} & "
          f"{metrics['flops_M']:.3f} & {metrics['peak_memory_MB']:.3f}\n")



