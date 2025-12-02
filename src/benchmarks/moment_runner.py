""""TS2Vec running functions"""
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

from momentfm import MOMENTPipeline

from methods.mlp_heads import make_MLP_regression_head
from utils.metrics_utils import Preds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MomentRunner:
    @staticmethod
    def pad_to_moment_patch_size(x: torch.Tensor, patch_size: int) -> torch.Tensor:
        "MOMENT has a patch embedding layer, so need to pad the input's #rows to be a multiple of patch_size"
        pad_len     = (patch_size - x.shape[1] % patch_size) % patch_size
        if pad_len == 0: return x
        zero_padding= torch.zeros(x.shape[0], pad_len, x.shape[2], device=x.device)
        return torch.cat([x, zero_padding], dim=1)

    @staticmethod
    def encode_x_to_z_in_batches(model, X, batch_size=32):
        """Direct encoding X to z (using MOMENT) is too heavy causing OOM, so do in batches"""
        from torch.cuda.amp import autocast
        model.eval()
        all_embeds = []
        use_amp = device == "cuda"  # autocast only on GPU
        with torch.no_grad():
            for i in range(0, X.size(0), batch_size):
                batch = X[i:i+batch_size].to(device)
                if use_amp:
                    with autocast(device_type='cuda'):
                        z = model.embed(x_enc=batch).embeddings
                else:
                    z = model.embed(x_enc=batch).embeddings#.detach().cpu()
                all_embeds.append(z.detach().cpu())  # move batch to CPU to relieve GPU memory
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        return torch.cat(all_embeds, dim=0)

    @staticmethod
    def train_moment_encoders(encoders: list, heads: list, X_splits: list, y_data,
                            batch_size: int, epochs: int, lr_encoder: float, lr_head: float,
                            unfreeze_last_n: int, fine_tune: bool, device: str, criterion):
        """Finetunes the pretrained Moment model encoder and prediction head. Trains the pre-trained Moment backbone
        along with a new randomly-initialized prediction head on the task-specific data.
        model: Moment model instance containing the pre-trained encoder backbone."""
        if isinstance(y_data, torch.Tensor):
            y_splits = [y_data] * len(X_splits)
        else:
            y_splits = y_data

        for i, (encoder, head, X_split, y_split) in enumerate(zip(encoders, heads, X_splits, y_splits)):
            if fine_tune:
                # Unfreeze last N blocks
                num_blocks = len(encoder.encoder.block)
                for j in range(num_blocks - unfreeze_last_n, num_blocks):
                    block_name = f"encoder.block.{j}"
                    for name, param in encoder.named_parameters():
                        if block_name in name:
                            param.requires_grad = True
                # Unfreeze final_layer_norm
                for name, param in encoder.named_parameters():
                    if "final_layer_norm" in name:
                        param.requires_grad = True

            optimizer = torch.optim.AdamW([
                {'params': [p for p in encoder.parameters() if p.requires_grad], 'lr': lr_encoder},
                {'params': head.parameters(), 'lr': lr_head}])

            encoder.train()
            head.train()
            X_split, y_split = X_split.to(device), y_split.to(device)

            for epoch in range(epochs):
                permutation = torch.randperm(X_split.size(0))
                epoch_loss  = 0.0
                for idx in range(0, X_split.size(0), batch_size):
                    batch_idx = permutation[idx:idx+batch_size]
                    batch_X   = X_split[batch_idx]
                    batch_y   = y_split[batch_idx]
                    batch_z   = encoder.embed(x_enc=batch_X).embeddings
                    preds     = head(batch_z)
                    optimizer.zero_grad()
                    loss      = criterion(preds, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * batch_X.size(0)
                epoch_loss /= X_split.size(0)
                if epoch % 2 == 0:
                    print(f"Encoder {i+1}, Epoch {epoch}, Loss: {epoch_loss:.4f}")
            encoder.eval()
            head.eval()

    @staticmethod
    def _unfreeze_last_n_blocks(moment_model, unfreeze_last_n):
        """Unfreezes the last N transformer blocks + final layer norm in the MOMENT encoder."""
        num_blocks = len(moment_model.encoder.block)
        for i in range(num_blocks - unfreeze_last_n, num_blocks):
            block_name = f"encoder.block.{i}"
            for name, param in moment_model.named_parameters():
                if block_name in name:
                    param.requires_grad = True
        for name, param in moment_model.named_parameters(): # unfreeze final_layer_norm
            if "final_layer_norm" in name:
                param.requires_grad = True

    @staticmethod
    def evaluate_moment_rmse(moment_model: nn.Module, regressor: nn.Module, X_train: torch.Tensor, X_test: torch.Tensor,
                            y_train: torch.Tensor, y_test: torch.Tensor, batch_size: int, device: str, criterion) -> tuple:
        """Returns (train_rmse, test_rmse) using MOMENT encoder + MLP head."""
        moment_model.eval()
        regressor.eval()
        bs = batch_size if device == "cuda" else 2

        with torch.no_grad():
            z_train = MomentRunner.encode_x_to_z_in_batches(moment_model, X_train, batch_size=bs)
            z_test  = MomentRunner.encode_x_to_z_in_batches(moment_model, X_test,  batch_size=bs)
            preds_train = regressor(z_train.to(device))
            preds_test  = regressor(z_test.to(device))
            train_rmse  = torch.sqrt(criterion(preds_train, y_train.to(device))).item()
            test_rmse   = torch.sqrt(criterion(preds_test,  y_test.to(device))).item()
        return z_train, z_test, preds_test, train_rmse, test_rmse

    @staticmethod
    def log_moment_results(dataset_name, losses, r2, model_cfg, train_cfg, filename="results/hyperparam_search_moment.txt"):
        """Log MOMENT results with hyperparameters, both to file and stdout.
        Args:
            dataset_name (str): Name of the dataset.
            losses (list[float]): [RMSE_train, RMSE_test, linreg_RMSE, catboost_RMSE, ...]
            r2 (float): R² on test set for the MLP head.
            metrics (dict): Runtime, parameters, FLOPS, memory info.
            model_cfg (dict): Model config like {'hidden_layers': [...], 'latent_dim': int, 'unfreeze_last_n': int}.
            train_cfg (dict): Training config like {'epochs': int, 'lr_encoder': float, 'lr_head': float, 'batch_size': int}.
            filename (str): File path to append results."""
        rmse_train, rmse_test = losses[:2]

        with open(filename, 'a') as f:
            f.write(f"moment/{dataset_name}: hidden_layers={model_cfg['hidden_layers']} "
                    # f"latent_dim={model_cfg['latent_dim']} unfreeze_last_n={model_cfg['unfreeze_last_n']}\n")
                    f"unfreeze_last_n={model_cfg['unfreeze_last_n']}\n")

            f.write(f"train_epochs={train_cfg['epochs']} lr_encoder={train_cfg['lr_encoder']} "
                    f"lr_head={train_cfg['lr_head']} batch_size={train_cfg['batch_size']}\n")
            f.write(f"& Z (moment) & train_RMSE={rmse_train:.4f} & test_RMSE={rmse_test:.4f}\n")
            f.write(f"R² (MLP head): {r2:.3f}\n\n")
            # f.write("time & params & flops & memory\n")
            # f.write(f"{metrics['runtime_s']:.3f} & {metrics['num_params_M']:.3f} & "
            #         f"{metrics['flops_M']:.3f} & {metrics['peak_memory_MB']:.3f}\n\n")

        print(f"moment/{dataset_name}: hidden_layers={model_cfg['hidden_layers']} "
            f"latent_dim={model_cfg['latent_dim']} unfreeze_last_n={model_cfg['unfreeze_last_n']}")
        print( "    RMSE   | LinReg   |   CatBoost  |   RForest |   NN")
        print(f"& {losses[0]:.4f} & {losses[1]:.4f} & {losses[2]:.4f} & {losses[3]:.4f}")
        # print(f"& Z (moment) & train_RMSE={rmse_train:.4f} & test_RMSE={rmse_test:.4f}")
        print(f"R² (MLP head): {r2:.3f}")
        # print(f"R² (MOMENT): {rf_model.score(z_test_final.cpu().numpy(), y_test_scaled):.3f}")
        print(f"train_epochs={train_cfg['epochs']} lr_encoder={train_cfg['lr_encoder']} "
            f"lr_head={train_cfg['lr_head']} batch_size={train_cfg['batch_size']}")
        # print("time & params & flops & memory")
        # print(f"{metrics['runtime_s']:.3f} & {metrics['num_params_M']:.3f} & "
        #       f"{metrics['flops_M']:.3f} & {metrics['peak_memory_MB']:.3f}\n")

    @staticmethod
    def run_moment0(X_train: torch.Tensor, X_test: torch.Tensor, y_train: torch.Tensor, y_test: torch.Tensor, *,
                model_cfg: dict, train_cfg: dict, device):
        """Full MOMENT training wrapper. Returns (losses, r2, metrics)."""
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test  = torch.tensor(X_test,  dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test  = torch.tensor(y_test,  dtype=torch.float32)
        torch.cuda.empty_cache()
        
        moment_model = MOMENTPipeline.from_pretrained(
            f"AutonLab/{model_cfg['model_name']}", model_kwargs={'task_name': model_cfg['task_name'], 'n_channels': X_train.shape[2],},).to(device)

        patch_size = (getattr(moment_model.tokenizer, "patch_size", None) or getattr(moment_model.tokenizer, "patch_len", None))
        if patch_size is None:
            raise ValueError("Missing patch size")

        X_train = MomentRunner.pad_to_moment_patch_size(X_train, patch_size).permute(0, 2, 1)
        X_test  = MomentRunner.pad_to_moment_patch_size(X_test,  patch_size).permute(0, 2, 1)

        # Freeze / unfreeze
        for p in moment_model.parameters(): p.requires_grad = False
        if train_cfg["fine_tune"]:
            MomentRunner._unfreeze_last_n_blocks(moment_model, model_cfg["unfreeze_last_n"])
            moment_model.train()
        else:
            moment_model.eval()

        with torch.no_grad():
            z_sample = moment_model.embed(x_enc=X_train[:1].to(device)).embeddings
        embedding_dim= z_sample.shape[1]
        head         = make_MLP_regression_head(embedding_dim, model_cfg["hidden_layers"], y_train, model_cfg["dropout"], device)
        model_cfg["latent_dim"] = embedding_dim

        criterion = nn.MSELoss()

        MomentRunner.train_moment_encoders(
            [moment_model], [head], [X_train], y_train,
            train_cfg["batch_size"], train_cfg["epochs"],
            train_cfg["lr_encoder"], train_cfg["lr_head"],
            model_cfg["unfreeze_last_n"], train_cfg["fine_tune"], device, criterion)

        z_train, z_test, preds_test, train_rmse, test_rmse = MomentRunner.evaluate_moment_rmse(moment_model, head, X_train, X_test, y_train, 
                                                                                y_test, train_cfg["batch_size"], device, criterion)
        losses, rf_model = Preds().evaluate_models_on_dataset(z_train.cpu().numpy(), y_train.cpu().numpy(),
                                                            z_test.cpu().numpy(),  y_test.cpu().numpy())
        losses.append(test_rmse)
        r2      = r2_score(y_test.cpu().numpy(), preds_test.cpu().numpy())
        metrics = {}
        return losses, r2, metrics

    @staticmethod
    def run_moment(X_train, X_test, y_train, y_test, *,
                   model_cfg, train_cfg, device, moment_model=None, head=None):
        """Full MOMENT training wrapper. Returns (losses, r2, metrics).
        - device: torch.device or str ("cuda" / "cpu")"""
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test  = torch.tensor(X_test,  dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test  = torch.tensor(y_test, dtype=torch.float32)
        # torch.cuda.empty_cache()

        # Ensure device is torch.device
        if isinstance(device, str):
            device = torch.device(device)

        if moment_model is None:
            moment_model = MOMENTPipeline.from_pretrained(f"AutonLab/{model_cfg['model_name']}",
                model_kwargs={'task_name': model_cfg['task_name'], 'n_channels': X_train.shape[2]}).to(device)
        else: # Already preloaded, don’t touch requires_grad
            moment_model.train() if train_cfg.get("fine_tune", True) else moment_model.eval()

        patch_size = getattr(moment_model.tokenizer, "patch_size", None) or getattr(moment_model.tokenizer, "patch_len", None)
        if patch_size is None:
            raise ValueError("Missing patch size")

        X_train = MomentRunner.pad_to_moment_patch_size(X_train, patch_size).permute(0, 2, 1)
        X_test  = MomentRunner.pad_to_moment_patch_size(X_test,  patch_size).permute(0, 2, 1)

        # Freeze / unfreeze
        for p in moment_model.parameters(): p.requires_grad = False
        if train_cfg.get("fine_tune", True):
            MomentRunner._unfreeze_last_n_blocks(moment_model, model_cfg["unfreeze_last_n"])
            moment_model.train()
        else:
            moment_model.eval()

        with torch.no_grad():
            z_sample = moment_model.embed(x_enc=X_train[:1].to(device)).embeddings
        embedding_dim = z_sample.shape[1]

        model_cfg["latent_dim"] = embedding_dim
        criterion = nn.MSELoss()

        if head is None:
            head = make_MLP_regression_head(embedding_dim, model_cfg["hidden_layers"], y_train, model_cfg.get("dropout", 0.0), device)

        MomentRunner.train_moment_encoders([moment_model], [head], [X_train], y_train,
                            train_cfg["batch_size"], train_cfg["epochs"],
                            train_cfg["lr_encoder"], train_cfg["lr_head"],
                            model_cfg["unfreeze_last_n"], train_cfg.get("fine_tune", True),
                            device, criterion)

        z_train, z_test, preds_test, train_rmse, test_rmse = MomentRunner.evaluate_moment_rmse(moment_model, head, X_train, X_test, y_train,
                                                                                y_test, train_cfg["batch_size"], device, criterion)
        losses, rf_model = Preds().evaluate_models_on_dataset(z_train.cpu().numpy(), y_train.cpu().numpy(),
                                                              z_test.cpu().numpy(),  y_test.cpu().numpy())
        losses.append(test_rmse)
        r2 = r2_score(y_test.cpu().numpy(), preds_test.cpu().numpy())
        metrics = {}
        return losses, r2, metrics

