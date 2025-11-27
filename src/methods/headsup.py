import math
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from utils.model_utils import norm_temp_xentropy_loss, TorchWrapper

class Headsup:
    def __init__(self, encoder, proj_head, decoder, supervised_head, device, contrast_temp: float = 0.5,
                 aug1: str = "jitter", aug2: str = "mag_warp", aug1_strength: float = 0.1, aug2_strength: float = 0.1,
                 last_block_lr: float = 1e-3, default_lr: float = 1e-4):
        """Wrapper for pretraining + fine-tuning an encoder with projection, decoder, and supervised head
            encoder: nn.Module
            proj_head: nn.Module
            decoder: nn.Module
            supervised_head: wrapper with .model attribute
            device: torch.device
            contrast_temp: float, temperature for NT-Xent loss
            jitter_strength: float, strength of jitter augmentation
            mag_warp_strength: float, strength of mag_warp augmentation
            last_block_lr: float, learning rate for last block when frac < 0.5
            default_lr: float, default learning rate for other params"""
        self.MIN_BATCH_SIZE   = 2 # for contrastive loss equation
        self.PRINT_EVERY      = 3
        self.lr_min           = 1e-5

        self.encoder          = encoder
        self.proj_head        = proj_head
        self.decoder          = decoder
        self.supervised_head  = supervised_head
        self.device           = device

        self.contrast_temp      = contrast_temp
        self.aug1               = aug1
        self.aug2               = aug2
        self.aug1_strength      = aug1_strength
        self.aug2_strength      = aug2_strength
        self.last_block_lr      = last_block_lr
        self.default_lr         = default_lr

    def _augment(self, X):
        X1 = make_augmentations(X, self.aug1, self.device, self.aug1_strength)
        X2 = make_augmentations(X, self.aug2, self.device, self.aug2_strength)
        return X1, X2

    def _early_stop_check(self, loss_total, best_loss, wait, patience):
        """Stop when loss isnt getting better. Returns updated best_loss, wait counter, and a boolean flag indicating whether to stop."""
        if loss_total < best_loss:
            best_loss = loss_total
            wait = 0
            stop = False
        else:
            wait += 1
            stop = wait >= patience
        return best_loss, wait, stop

    def _make_lr_cos_scheduler(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float):
        """Cosine LR scheduler with linear warmup.
            - optimizer: torch optimizer
            - warmup_steps: steps to linearly ramp up LR
            - total_steps: total training steps
            - min_lr: minimum LR at the end of cosine decay"""
        schedulers = []
        for group in optimizer.param_groups:
            base_lr = group["lr"]

            def lr_lambda(step, base_lr=base_lr):
                if step < warmup_steps:
                    return step / float(max(1, warmup_steps))
                progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return (min_lr / base_lr) + (1 - min_lr / base_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            schedulers.append(lr_lambda)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedulers)

    def _pretrain_single_epoch(self, X_train, batch_size, weights, optimizer, scheduler):
        """Runs one epoch of pretraining on the encoder."""
        loss_recon, loss_contrast, loss_total = 0, 0, 0
        for i in range(0, len(X_train), batch_size):
            X_batch = torch.tensor(X_train[i:i+batch_size], dtype=torch.float32, device=self.device)
            if X_batch.size(0) < self.MIN_BATCH_SIZE:
                continue

            X1, X2    = self._augment(X_batch)
            z1, z2    = self.encoder(X1), self.encoder(X2)
            h1, h2    = self.proj_head(z1).mean(dim=1), self.proj_head(z2).mean(dim=1)
            z1_pooled = z1.mean(dim=1)
            x_recon   = self.decoder(z1_pooled)

            loss_contrast = norm_temp_xentropy_loss(h1, h2, temperature=self.contrast_temp)
            loss_recon    = F.mse_loss(x_recon, X_batch)
            loss_total    = weights["recon"] * loss_recon + weights["contrast"] * loss_contrast

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            scheduler.step()
        return loss_recon.item(), loss_contrast.item(), loss_total.item()

    def pretrain(self, X_train, batch_size, epochs, warmup_frac_pretrain, weights=weights_pretrain, lr=lr_pretrain, patience = None):
        """Pretrains encoder with contrastive + reconstruction loss. Pretrain usually has lots of steps, and finetuning has few"""
        optimizer_pretrain = torch.optim.AdamW(list(self.encoder.parameters()) +
                                               list(self.proj_head.parameters()) +
                                               list(self.decoder.parameters()), lr=lr)
        max_steps          = train_epochs_pretrain * (len(X_train) // batch_size)
        warmup_steps       = int(warmup_frac_pretrain * max_steps)
        scheduler_pretrain = self._make_lr_cos_scheduler(optimizer_pretrain, warmup_steps=warmup_steps, total_steps=max_steps, min_lr=self.lr_min)

        best_loss, wait = float("inf"), 0
        for epoch in range(epochs):
            loss_recon, loss_contrast, loss_total = self._pretrain_single_epoch(X_train, batch_size, weights, optimizer_pretrain, scheduler_pretrain)
            if epoch % self.PRINT_EVERY == 0:
                print(f"[Pretrain] Epoch {epoch+1}/{epochs}: "
                      f"recon={loss_recon:.4f}, contrast={loss_contrast:.4f}, total={loss_total:.4f}")

            if patience is not None:
                best_loss, wait, stop_flag = self._early_stop_check(loss_total, best_loss, wait, patience)
                if stop_flag:
                    print(f"Early stopping triggered @ epoch {epoch+1}")
                    break
        return self.encoder, self.proj_head, self.decoder

    def _setup_encoder_optimizer(self, frac: float):
        """Sets encoder layers' requires_grad according to labeled fraction.
        Returns weights for the loss components and optimizer"""
        if frac == 1.0:
            for p in self.encoder.parameters():
                p.requires_grad = True # unfrozen encoder
            weights_train   = weights_train_100 #{"pred": 1.0, "recon": 0.5, "contrast": 0.5}
            params_to_opt   = list(self.encoder.parameters()) + list(self.proj_head.parameters()) + \
                              list(self.decoder.parameters()) + list(self.supervised_head.model.parameters())
            optimizer_train = torch.optim.AdamW(params_to_opt, lr=lr_train)
        elif frac >= 0.5:
            for p in self.encoder.parameters():
                p.requires_grad = False # frozen encoder
            weights_train   = weights_train_50 #{"pred": 1.0, "recon": 0.1, "contrast": 0.1}
            params_to_opt   = list(self.encoder.parameters()) + list(self.proj_head.parameters()) + \
                              list(self.decoder.parameters()) + list(self.supervised_head.model.parameters())
            optimizer_train = torch.optim.AdamW(params_to_opt, lr=lr_train)
        else:
            for name, p in self.encoder.named_parameters():
                p.requires_grad = False # frozen encoder
                if name.startswith("encoder_layers") and "last_block" in name:
                    p.requires_grad = True # unfreeze last block only
            weights_train = weights_train_other #{"pred": 2.0, "recon": 0.0, "contrast": 0.0}

            last_block_params = [p for n, p in self.encoder.named_parameters()
                                 if n.startswith("encoder_layers") and "last_block" in n]
            params_to_opt     = list(self.proj_head.parameters()) + list(self.decoder.parameters()) + \
                                list(self.supervised_head.model.parameters())
            optimizer_train   = torch.optim.AdamW([{"params": last_block_params, "lr": self.last_block_lr}, {"params": params_to_opt, "lr": self.default_lr}])
        return weights_train, optimizer_train

    def _train_single_epoch(self, X_L, y_L, X_train, optimizer_train, scheduler_train, batch_size, weights_train):
        """Runs one epoch of fine-tuning on labeled + unlabeled data (tensor conversion done once)."""
        X_L     = X_L.to(self.device) if not isinstance(X_L, torch.Tensor) else X_L
        y_L     = y_L.to(self.device) if not isinstance(y_L, torch.Tensor) else y_L
        X_train = torch.tensor(X_train, dtype=torch.float32, device=self.device) if not isinstance(X_train, torch.Tensor) else X_train

        loss_pred, loss_recon, loss_contrast, loss_total = 0, 0, 0, 0

        num_batches = (len(X_train) + batch_size - 1) // batch_size
        for i in range(num_batches):
            start = i * batch_size
            end_l = min(start + batch_size, len(X_L))
            end_u = min(start + batch_size, len(X_train))

            X_batch_l = X_L[start:end_l]
            y_batch_l = y_L[start:end_l]
            X_batch_u = X_train[start:end_u]

            if X_batch_l.size(0) < self.MIN_BATCH_SIZE or X_batch_u.size(0) < self.MIN_BATCH_SIZE:
                continue

            # encode
            z_l, z_u = self.encoder(X_batch_l), self.encoder(X_batch_u)
            z_l_pooled, z_u_pooled = z_l.mean(dim=1), z_u.mean(dim=1)

            # supervised loss
            y_hat     = self.supervised_head.model(z_l_pooled)
            loss_pred = F.mse_loss(y_hat, y_batch_l)

            # reconstruction loss
            x_recon_l, x_recon_u = self.decoder(z_l_pooled), self.decoder(z_u_pooled)
            loss_recon = (F.mse_loss(x_recon_l, X_batch_l) + F.mse_loss(x_recon_u, X_batch_u)) / 2

            # contrastive loss
            X1_L, X2_L = self._augment(X_batch_l)
            X1_U, X2_U = self._augment(X_batch_u)
            z1_L, z2_L = self.encoder(X1_L), self.encoder(X2_L)
            z1_U, z2_U = self.encoder(X1_U), self.encoder(X2_U)
            h1_L, h2_L = self.proj_head(z1_L).mean(dim=1), self.proj_head(z2_L).mean(dim=1)
            h1_U, h2_U = self.proj_head(z1_U).mean(dim=1), self.proj_head(z2_U).mean(dim=1)
            loss_contrast = (norm_temp_xentropy_loss(h1_L, h2_L, self.contrast_temp) + norm_temp_xentropy_loss(h1_U, h2_U, self.contrast_temp)) / 2

            # backward
            loss_total = weights_train["pred"] * loss_pred + weights_train["recon"] * loss_recon + weights_train["contrast"] * loss_contrast
            optimizer_train.zero_grad()
            loss_total.backward()
            optimizer_train.step()
            scheduler_train.step()
        return loss_pred.item(), loss_recon.item(), loss_contrast.item(), loss_total.item()

    def training_loop(self, X_train, y_train_scaled, X_test, y_test_scaled, batch_size, train_epochs_finetune, warmup_frac_train, label_fractions, patience = None):
        """Fine-tunes encoder + heads over all labeled fractions. Pretrain usually has lots of steps, and finetuning has few"""
        results_dict = {}
        z_train_dict = {}
        z_test_dict  = {}
        y_L_dict     = {}
        for frac in label_fractions:
            n_samples = int(len(X_train) * frac)
            X_L       = torch.tensor(X_train[:n_samples], dtype=torch.float32, device=self.device)
            y_L       = torch.tensor(y_train_scaled[:n_samples], dtype=torch.float32, device=self.device)

            weights_train, optimizer_train = self._setup_encoder_optimizer(frac)
            # scheduler_train = CosineAnnealingLR(optimizer_train, T_max=train_epochs_finetune * (len(X_train)//batch_size), eta_min=self.lr_min)
            max_steps       = train_epochs_finetune * (len(X_train) // batch_size)
            warmup_steps    = int(warmup_frac_train * max_steps)
            scheduler_train = self._make_lr_cos_scheduler(optimizer_train, warmup_steps=warmup_steps, total_steps=max_steps, min_lr=self.lr_min)

            best_loss, wait = float("inf"), 0
            for epoch in range(train_epochs_finetune):
                loss_pred, loss_recon, loss_contrast, loss_total = self._train_single_epoch(
                    X_L, y_L, X_train, optimizer_train, scheduler_train, batch_size, weights_train)
                if epoch % self.PRINT_EVERY == 0:
                    print(f"[Finetune {frac*100:.0f}%] Epoch {epoch+1}/{train_epochs_finetune}: "
                          f"pred={loss_pred:.4f}, recon={loss_recon:.4f}, "
                          f"contrast={loss_contrast:.4f}, total={loss_total:.4f}")
                if patience is not None:
                    best_loss, wait, stop_flag = self._early_stop_check(loss_total, best_loss, wait, patience)
                    if stop_flag:
                        print(f"Early stopping triggered @ epoch {epoch+1} (label frac={frac*100:.0f}%)")
                        break

            #  ====== internal evaluation ======
            self.encoder.eval()
            self.proj_head.eval()
            self.decoder.eval()
            self.supervised_head.model.eval()
            with torch.no_grad():
                z_test             = self.encoder(torch.tensor(X_test, dtype=torch.float32, device=self.device))
                z_test_pooled      = z_test.mean(dim=1)
                y_pred             = self.supervised_head.model(z_test_pooled).cpu().numpy()
                results_dict[frac] = root_mean_squared_error(y_test_scaled, y_pred)

                z_train            = self.encoder(X_L).mean(dim=1).cpu().numpy()
                z_train_dict[frac] = z_train
                z_test_dict[frac]  = z_test_pooled.cpu().numpy()
                y_L_dict[frac]     = y_L.cpu().numpy()
                print(f"Sup. head RMSE ({frac*100:.0f}% labels): {results_dict[frac]:.4f}")
        return results_dict, z_train_dict, z_test_dict, y_L_dict


class HeadsupAE(Headsup):
    """Wrapper for FlexibleAutoencoder with projection + supervised head.
    - Encoder/decoder work on sequences [B, T, F].
    - Decoder reconstructs full sequence.
    - Supervised head works on pooled latent (mean over time).
    """
    def __init__(self, ae_model, device: torch.device,
                 contrast_temp: float = 0.5, jitter_strength: float = 0.1,
                 mag_warp_strength: float = 0.1, last_block_lr: float = 5e-4,
                 default_lr: float = 1e-4, supervised_head=None):

        if supervised_head is None:
            supervised_head = TorchWrapper(ae_model.prediction_head)

        super().__init__(
            encoder=None,  # we override
            proj_head=ae_model.projection_head,
            decoder=ae_model.decoder,
            supervised_head=supervised_head,
            device=device,
            contrast_temp=contrast_temp,
            jitter_strength=jitter_strength,
            mag_warp_strength=mag_warp_strength,
            last_block_lr=last_block_lr,
            default_lr=default_lr)
        self.ae_model = ae_model
        self.encoder  = ae_model.encoder

    def _encode(self, X: torch.Tensor) -> torch.Tensor:
        """Encode each timestep independently to satisfy BN layers"""
        B, T, F = X.shape
        Z_list = [self.ae_model.encoder(X[:, t, :]) for t in range(T)]  # list of [B, H]
        Z = torch.stack(Z_list, dim=1)  # [B, T, H]
        return Z

    def _pool_latents(self, Z: torch.Tensor) -> torch.Tensor:
        """Pool latent sequence for supervised head (mean over time)."""
        return Z.mean(dim=1)  # [B, H]

    def _pretrain_single_epoch(self, X_train, batch_size, weights, optimizer, scheduler):
        loss_recon, loss_contrast, loss_total = 0, 0, 0
        for i in range(0, len(X_train), batch_size):
            X_batch = torch.tensor(X_train[i:i+batch_size], dtype=torch.float32, device=self.device)
            if X_batch.size(0) < self.MIN_BATCH_SIZE:
                continue

            # augmentations
            X1, X2 = self._augment(X_batch)

            # encode sequences
            z1, z2 = self._encode(X1), self._encode(X2)

            # project pooled
            h1, h2 = self.proj_head(self._pool_latents(z1)), self.proj_head(self._pool_latents(z2))

            # reconstruct sequence
            x_recon = self.decoder(z1)

            # losses
            loss_contrast = norm_temp_xentropy_loss(h1, h2, temperature=self.contrast_temp)
            loss_recon    = F.mse_loss(x_recon, X_batch)
            loss_total    = weights["recon"]*loss_recon + weights["contrast"]*loss_contrast

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            scheduler.step()
        return loss_recon.item(), loss_contrast.item(), loss_total.item()

    def pretrain(self, X_train, batch_size, epochs,
                 weights={"recon":0.1,"contrast":1.0}, lr=1e-3):
        optimizer_pretrain = torch.optim.AdamW(
            list(self.ae_model.encoder.parameters()) +
            list(self.proj_head.parameters()) +
            list(self.decoder.parameters()),
            lr=lr
        )
        max_steps = epochs * (len(X_train)//batch_size)
        scheduler_pretrain = CosineAnnealingLR(
            optimizer_pretrain, T_max=max_steps, eta_min=self.lr_min
        )

        for epoch in range(epochs):
            loss_recon, loss_contrast, loss_total = self._pretrain_single_epoch(
                X_train, batch_size, weights, optimizer_pretrain, scheduler_pretrain
            )
            if epoch % self.PRINT_EVERY == 0:
                print(f"[Pretrain] Epoch {epoch+1}/{epochs}: "
                      f"recon={loss_recon:.4f}, contrast={loss_contrast:.4f}, total={loss_total:.4f}")
        return self.ae_model.encoder, self.proj_head, self.decoder

    def _setup_encoder_optimizer(self, frac: float, lr_train: float = 1e-4):
        """Freeze/unfreeze encoder depending on labeled fraction."""
        if frac == 1.0:
            for p in self.ae_model.encoder.parameters(): p.requires_grad = True
            weights_train = {"pred": 1.0, "recon": 0.5, "contrast": 0.5}
            params = list(self.ae_model.encoder.parameters()) + list(self.proj_head.parameters()) + list(self.decoder.parameters())
            if self.supervised_head: params += list(self.supervised_head.parameters())
            opt = torch.optim.AdamW(params, lr=lr_train)
        elif frac >= 0.5:
            for p in self.ae_model.encoder.parameters(): p.requires_grad = False
            weights_train = {"pred": 1.0, "recon": 0.1, "contrast": 0.1}
            params = list(self.proj_head.parameters()) + list(self.decoder.parameters())
            if self.supervised_head: params += list(self.supervised_head.parameters())
            opt = torch.optim.AdamW(params, lr=lr_train)
        else:
            for p in self.ae_model.encoder.parameters(): p.requires_grad = False
            weights_train = {"pred": 2.0, "recon": 0.0, "contrast": 0.0}
            params = list(self.proj_head.parameters())
            if self.supervised_head: params += list(self.supervised_head.parameters())
            opt = torch.optim.AdamW(params, lr=self.default_lr)
        return weights_train, opt

    def _train_single_epoch(self, X_L, y_L, X_train, optimizer_train, scheduler_train, batch_size, weights_train):
        loss_pred, loss_recon, loss_contrast, loss_total = 0, 0, 0, 0
        for i in range(0, len(X_train), batch_size):
            X_batch_l = X_L[i:i+batch_size]
            y_batch_l = y_L[i:i+batch_size]
            if X_batch_l.size(0) < self.MIN_BATCH_SIZE: continue

            start_unlab = i % len(X_train)
            X_batch_u = torch.tensor(X_train[start_unlab:start_unlab+batch_size], dtype=torch.float32, device=self.device)
            if X_batch_u.size(0) < self.MIN_BATCH_SIZE: continue

            # encode sequences
            z_l, z_u = self._encode(X_batch_l), self._encode(X_batch_u)

            # supervised prediction
            if self.supervised_head:
                y_hat = self.supervised_head(self._pool_latents(z_l))
                loss_pred = F.mse_loss(y_hat, y_batch_l)

            # reconstruction
            x_recon_l, x_recon_u = self.decoder(z_l), self.decoder(z_u)
            loss_recon = (F.mse_loss(x_recon_l, X_batch_l) + F.mse_loss(x_recon_u, X_batch_u)) / 2

            # contrastive
            X1_l, X2_l = self._augment(X_batch_l)
            X1_u, X2_u = self._augment(X_batch_u)
            z1_l, z2_l = self._encode(X1_l), self._encode(X2_l)
            z1_u, z2_u = self._encode(X1_u), self._encode(X2_u)
            h1_l, h2_l = self.proj_head(self._pool_latents(z1_l)), self.proj_head(self._pool_latents(z2_l))
            h1_u, h2_u = self.proj_head(self._pool_latents(z1_u)), self.proj_head(self._pool_latents(z2_u))
            loss_contrast = (norm_temp_xentropy_loss(h1_l, h2_l, self.contrast_temp) +
                             norm_temp_xentropy_loss(h1_u, h2_u, self.contrast_temp)) / 2

            # total loss
            loss_total = (weights_train["pred"]*loss_pred +
                          weights_train["recon"]*loss_recon +
                          weights_train["contrast"]*loss_contrast)

            optimizer_train.zero_grad()
            loss_total.backward()
            optimizer_train.step()
            scheduler_train.step()
        return loss_pred.item(), loss_recon.item(), loss_contrast.item(), loss_total.item()



# to remove (doesnt work well, catboost doesnt backprop)
class HeadsCatboost(Headsup):
    def __init__(self, encoder, proj_head, decoder, device, catboost_params=None):
        super().__init__(encoder, proj_head, decoder, supervised_head=None, device=device)
        self.catboost_params = catboost_params or dict(iterations=500, learning_rate=0.05, depth=6, verbose=0)
        self.catboost_model = None
        self.non_constant_idx = []

    def pretrain_encoder(self, X_train, batch_size, epochs):
        """Run only the pretraining stage."""
        self.pretrain(X_train, batch_size=batch_size, epochs=epochs)

    def predict_catboost_multioutput(self, X_train: np.ndarray, y_train: np.ndarray, 
                                     X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Optional[MultiOutputRegressor], np.ndarray, float, list]:
        """Train multi-output CatBoost and return predictions, RMSE, and non-constant indices."""
        y_pred = np.zeros_like(y_test, dtype=float)
        non_constant_idx = [i for i in range(y_train.shape[1])
                            if not np.all(y_train[:, i] == y_train[0, i])]
        if non_constant_idx:
            model = MultiOutputRegressor(CatBoostRegressor(**self.catboost_params))
            model.fit(X_train, y_train[:, non_constant_idx])
            y_pred[:, non_constant_idx] = model.predict(X_test)
            for i in range(y_train.shape[1]):
                if i not in non_constant_idx:
                    y_pred[:, i] = y_train[0, i]
        else:
            model = None
        rmse = root_mean_squared_error(y_test, y_pred)
        return model, y_pred, rmse, non_constant_idx

    def train_supervised(self, X_train, y_train_scaled, X_test, y_test_scaled, label_fractions):
        """Train CatBoost on frozen encoder features for multiple label fractions."""
        results_dict, z_train_dict, z_test_dict, y_L_dict = {}, {}, {}, {}

        self.encoder.eval()
        self.proj_head.eval()
        self.decoder.eval()

        for frac in label_fractions:
            n_samples = int(len(X_train) * frac)
            X_L, y_L = X_train[:n_samples], y_train_scaled[:n_samples]

            # Frozen encoder features
            with torch.no_grad():
                z_train = self.encoder(torch.tensor(X_L, dtype=torch.float32, device=self.device)).mean(dim=1).cpu().numpy()
                z_test  = self.encoder(torch.tensor(X_test, dtype=torch.float32, device=self.device)).mean(dim=1).cpu().numpy()

            # Train CatBoost
            model, y_pred, rmse, non_constant_idx = self.predict_catboost_multioutput(z_train, y_L, z_test, y_test_scaled)
            self.catboost_model = model
            self.non_constant_idx = non_constant_idx

            results_dict[frac]  = rmse
            z_train_dict[frac]  = z_train
            z_test_dict[frac]   = z_test
            y_L_dict[frac]      = y_L
            print(f"RMSE with {frac*100:.0f}% labeled (CatBoost): {rmse:.4f}")

        return results_dict, z_train_dict, z_test_dict, y_L_dict

    def predict(self, X, y_ref=None):
        """Predict using trained CatBoost on pooled latent features."""
        self.encoder.eval()
        with torch.no_grad():
            z = self.encoder(torch.tensor(X, dtype=torch.float32, device=self.device)).mean(dim=1).cpu().numpy()

        if self.catboost_model is None:
            raise ValueError("CatBoost model not trained yet.")

        n_targets = y_ref.shape[1] if y_ref is not None else max(self.non_constant_idx)+1
        y_pred_full = np.zeros((len(X), n_targets))

        if self.non_constant_idx:
            y_pred_full[:, self.non_constant_idx] = self.catboost_model.predict(z)

        if y_ref is not None:
            for i in range(n_targets):
                if i not in self.non_constant_idx:
                    y_pred_full[:, i] = y_ref[0, i]

        return y_pred_full

