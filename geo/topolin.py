
import __main__
import sys, os
project_root = os.path.abspath("..")  # adjust if notebook is elsewhere
sys.path.insert(0, project_root)
from typing import Tuple
import logging

import category_encoders as ce
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numexpr as ne # makes numpy operations faster
import numpy as np
import pandas as pd

from scipy.signal import periodogram

from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from pyriemann.tangentspace import TangentSpace

from geo.geo_utils import compute_cyclicity_score, split_dataset_to_linear_and_cyclic, scale_train_and_test_sets, drop_low_variance_cols, Windowing
from geo_encoders import LSTMEncoderEuclid, LSTMSphericalEncoder, \
LSTMToroidalEncoder, MLPDecoder, Reparam, MLPPredHead, early_stop, fit_catboost_multi

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # print(torch.cuda.memory_reserved(0) / 1e6, "MB reserved")
    # print(torch.cuda.memory_allocated(0) / 1e6, "MB allocated")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Sphlin:
    @staticmethod
    def sample_gaussian(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    @staticmethod
    def sample_vmf(mu: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        """Approximate vMF sampling by adding Gaussian noise and normalizing.
        This is NOT exact vMF sampling; it is a heuristic used for speed.
        Valid for large kappa (high concentration) where vMF ≈ Gaussian on the sphere."""
        # kappa is [B, 1], mu is [B, D]
        # heuristic version to make things faster
        # z = mu + torch.randn_like(mu) / (kappa.view(-1,1) + 1e-6)
        # return F.normalize(z, dim=-1)

        # abither heuristic approimation to make things faster
        # kappa   = F.softplus(kappa) + 20.0
        # epsilon = torch.randn_like(mu) / torch.sqrt(kappa)
        # return F.normalize(mu + epsilon, dim=-1)

        """EXACT vMF sampler using Wood's algorithm. “Simulation of the von Mises Fisher Distribution” (Wood 1994)
        it is also used in davidson 2018 and Gopal & Yang, 2014
        mu: [B, D] unit vectors
        kappa: [B, 1] concentration (>=0)
        Returns z: [B, D] unit vectors."""
        B, D = mu.shape
        # kappa = kappa.view(-1).clamp(min=0.5)
        # kappa = torch.exp(kappa).clamp(min=2.0, max=20.0).view(-1)
        # kappa = F.softplus(kappa).clamp(max=20.0).view(-1)
        # kappa = 1 + F.elu(kappa)
        kappa = F.softplus(kappa) #+ 20.0

        b = (-2 * kappa + torch.sqrt(4 * kappa ** 2 + (D - 1) ** 2)) / (D - 1)
        x0 = (1 - b) / (1 + b)
        c = kappa * x0 + (D - 1) * torch.log1p(-x0 ** 2)

        w = torch.zeros(B, device=mu.device)
        for i in range(B):
            while True:
                u1 = torch.rand(1, device=mu.device)
                u2 = torch.rand(1, device=mu.device)
                z = 1 - (1 + b[i]) * u1 / (1 + b[i] * u1)
                t = kappa[i] * z + (D - 1) * torch.log1p(-x0[i] * z) - c[i]
                if torch.log(u2) <= t:
                    w[i] = z
                    break

        v = torch.randn(B, D - 1, device=mu.device)
        v = F.normalize(v, dim=-1)
        w = w.view(-1, 1)
        z = torch.cat([w, torch.sqrt(1 - w ** 2) * v], dim=1)

        # rotate to mu
        u = torch.randn(B, D, device=mu.device)
        u = F.normalize(u, dim=-1)
        # Householder reflection
        mu0 = torch.tensor([1.] + [0.] * (D - 1), device=mu.device).view(1, -1)
        v_h = F.normalize(mu0 - mu, dim=-1)
        z = z - 2 * (z * v_h).sum(dim=1, keepdim=True) * v_h
        return F.normalize(z, dim=-1)

    @staticmethod
    def kl_gaussian(mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

    @staticmethod
    def kl_vmf(mu: torch.Tensor, logkappa: torch.Tensor) -> torch.Tensor:
        # method 1: Standard vMF KL approximation: mu is normalized, so mu^2 sum is 1. problematic as mu=1, so its =0
        # return (kappa.view(-1) * (1 - torch.sum(mu ** 2, dim=-1))).mean()

        # method 2: use this (davidson also doesnt use the exact KL, but a concentr. penalty proport. to kappa)
        # why: Exact KL involves modified Bessel functions of fractional order. Numerically unstable. Adds nothing
        # empirically beyond “don’t collapse to high κ”
        # kappa = kappa.view(-1, 1).clamp(min=0.2)  # prevent collapse
        # return kappa.mean()

        """Compute vMF KL divergence between q(z|mu,kappa) and p(z)=Uniform(S^{d-1}).
        Uses a stable 'standard ratio approximation' for A_d(kappa)=I_{d/2}(kappa)/I_{d/2-1}(kappa).
        Source: Banerjee 2005 and Davidson 2018. see §'concerntr. param' of https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution
            - mu: unit vectors, shape [B, D]
            - kappa: concentration, shape [B, 1]
            - scalar KL mean over batch"""
        # method 3, KL between vMF(q(z|mu,kappa)) and uniform p(z), where encoder outputs logkappa (why logkappa? cause its +ve)
        # kappa = torch.exp(logkappa).clamp(min=2.0, max=20.0) + 1e-6
        # kappa = F.softplus(logkappa).clamp(max=20.0) + 1e-6
        # kappa = 1 + F.elu(logkappa)
        kappa = F.softplus(logkappa) + 5.0


        D = mu.shape[-1]
        kappa = kappa.view(-1)
        nu = D/2 - 1
        A = kappa / (nu + torch.sqrt(nu**2 + kappa**2) + 1e-20)
        pi = torch.tensor(np.pi, device=kappa.device, dtype=kappa.dtype)
        log_2pi = torch.log(2*pi)
        log_c = nu*torch.log(kappa + 1e-20) - (D/2)*log_2pi - torch.log(torch.special.i0(kappa) + 1e-20)
        log_c0 = -(D/2)*log_2pi
        kl = kappa*A - log_c + log_c0
        return kl.mean()

        # # method 4
        # D      = mu.shape[-1]
        # kappa  = kappa.view(-1)
        # nu     = D / 2 - 1
        # A      = kappa / (nu + torch.sqrt(nu**2 + kappa**2) + 1e-20)  # approximation of I_{nu+1}/I_nu
        # log_c  = (nu) * torch.log(kappa + 1e-20) - (D / 2) * torch.log(torch.tensor(2 * torch.pi, device=mu.device)) - torch.log(torch.special.i0(kappa) + 1e-20)
        # log_c0 = - (D / 2) * torch.log(torch.tensor(2 * torch.pi, device=mu.device))
        # kl     = (kappa * A) - log_c + log_c0
        # return kl.mean()

    @staticmethod
    def encode_full_dataset(X_lin, X_cyc, encoder_e, encoder_s, z_dim_euclid, device, batch_size=64):
        """Encodes the full dataset in batches to avoid OOM errors, handling cases where one input is None."""
        if encoder_e is not None: encoder_e.eval()
        if encoder_s is not None: encoder_s.eval()
        
        # Use whichever one is not None to get the total length
        n_samples = len(X_lin) if X_lin is not None else len(X_cyc)
        
        zs = []
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                x_lin = X_lin[i:i+batch_size].to(device) if X_lin is not None else None
                x_cyc = X_cyc[i:i+batch_size].to(device) if X_cyc is not None else None
                
                z_batch = []
                # Check for encoder existence AND actual features in the tensor
                if encoder_e is not None and x_lin is not None and x_lin.shape[-1] > 0:
                    mu, _ = encoder_e(x_lin)
                    z_batch.append(mu)
                if encoder_s is not None and x_cyc is not None and x_cyc.shape[-1] > 0:
                    mu_s, _ = encoder_s(x_cyc)
                    z_batch.append(mu_s)
                
                zs.append(torch.cat(z_batch, dim=-1).cpu())
        return torch.cat(zs, dim=0)

    @staticmethod
    def train_step(x_lin: torch.Tensor, x_cyc: torch.Tensor, y_win: torch.Tensor,
                encoder_e, encoder_s, decoder, pred_head,
                lambdas: dict, epoch: int) -> dict:
        """One VAE + prediction step with hard shape guards."""
        device = x_lin.device
        B = x_lin.size(0)

        mu_s, logkappa, z_s = None, None, None
        mu_e, z_e           = None, None
        z_parts             = []
        kl_e                = torch.tensor(0., device=device)
        kl_s                = torch.tensor(0., device=device)

        if encoder_e is not None and x_lin.shape[-1] > 0:
            mu_e, logvar_e = encoder_e(x_lin)
            z_e = Sphlin.sample_gaussian(mu_e, logvar_e)
            z_parts.append(z_e)
            kl_e = Sphlin.kl_gaussian(mu_e, logvar_e)

        if encoder_s is not None and x_cyc.shape[-1] > 0:
            mu_s, logkappa = encoder_s(x_cyc)
            z_s = Sphlin.sample_vmf(mu_s, logkappa)
            z_parts.append(z_s)
            kl_s = Sphlin.kl_vmf(mu_s, logkappa)

        if len(z_parts) == 0:
            z_parts.append(torch.zeros((B, 0), device=device))

        z = torch.cat(z_parts, dim=-1)

        targets = []
        if x_lin.shape[-1] > 0: targets.append(x_lin.reshape(B, -1))
        if x_cyc.shape[-1] > 0: targets.append(x_cyc.reshape(B, -1))
        x_target = torch.cat(targets, dim=-1)

        assert z.dim() == 2, f"z must be [B, latent], got {z.shape}"
        assert z.shape[0] == B, f"z batch mismatch: {z.shape[0]} vs {B}"

        x_hat_raw = decoder(z)
        assert x_hat_raw.shape[0] == B, f"decoder batch mismatch: {x_hat_raw.shape}"
        assert x_hat_raw.numel() % B == 0, f"decoder output not divisible by batch: {x_hat_raw.shape}"

        x_hat = x_hat_raw.reshape(B, -1)
        assert x_hat.shape == x_target.shape, (
            f"RECON SHAPE MISMATCH\n"
            f"x_hat    : {x_hat.shape}\n"
            f"x_target : {x_target.shape}\n"
            f"window   : {x_lin.shape[1]}\n"
            f"n_lin    : {x_lin.shape[-1]}\n"
            f"n_cyc    : {x_cyc.shape[-1]}"
        )

        y_hat = pred_head(z)
        assert y_hat.shape[0] == B, f"pred batch mismatch: {y_hat.shape}"

        recon_loss = F.mse_loss(x_hat, x_target)

        if y_win.dim() == 1:
            y_win = y_win.unsqueeze(-1)

        mask = ~torch.isnan(y_win)
        if mask.sum() == 0:
            pred_loss = torch.tensor(0., device=device)
        else:
            pred_loss = F.mse_loss(y_hat[mask], y_win[mask])

        kl_weight = 0.1 if epoch < 20 else 0.5 if epoch < 60 else 1.0
        total_loss = (
            lambdas["reconstr"] * recon_loss
            + kl_weight * (lambdas["euc"] * kl_e + lambdas["sph"] * kl_s)
            + lambdas["pred"] * pred_loss
        )

        kappa = None if logkappa is None else 1 + F.elu(logkappa)

        return {
            "total": total_loss,
            "recon": recon_loss,
            "kl_e": kl_e,
            "kl_s": kl_s,
            "pred": pred_loss,
            "mu_s": mu_s,
            "logkappa": logkappa,
            "kappa": kappa,
            "z_s": z_s,
            "z_e": z_e,
        }

    @staticmethod
    def train_epoch(loader, encoder_e, encoder_s, decoder, pred_head, optimizer, lambdas, device, epoch):
        totals = {}
        counts = {}
        for x_lin, x_cyc, y_win in loader:
            y_win = y_win.squeeze(-1)
            x_lin, x_cyc, y_win = x_lin.to(device), x_cyc.to(device), y_win.to(device)
            optimizer.zero_grad()
            losses = Sphlin.train_step(x_lin, x_cyc, y_win, encoder_e, encoder_s, decoder, pred_head, lambdas, epoch)
            losses["total"].backward()
            optimizer.step()

            for k, v in losses.items():
                if v is None: 
                    continue
                v = v.detach().mean() if torch.is_tensor(v) else v
                totals[k] = totals.get(k, 0.0) + float(v)
                counts[k] = counts.get(k, 0) + 1
        return {k: totals[k] / counts[k] for k in totals}

    @staticmethod
    def train_loop(XL_w, XC_w, y_w, encoder_e, encoder_s, decoder, pred_head, optimizer, p, device):
        # Determine total samples from whatever is available
        batch_size_total = y_w.shape[0]

        # Safely convert or create empty placeholders for the DataLoader
        if XL_w is None:
            XL_w = torch.zeros((batch_size_total, p.window_size, 0), device=device)
        elif not torch.is_tensor(XL_w):
            XL_w = torch.tensor(XL_w, dtype=torch.float32, device=device)
            
        if XC_w is None:
            XC_w = torch.zeros((batch_size_total, p.window_size, 0), device=device)
        elif not torch.is_tensor(XC_w):
            XC_w = torch.tensor(XC_w, dtype=torch.float32, device=device)
            
        if not torch.is_tensor(y_w):
            y_w = torch.tensor(y_w, dtype=torch.float32, device=device)

        # Now the loader will receive Tensors, not None
        loader  = DataLoader(TensorDataset(XL_w, XC_w, y_w), batch_size=p.batch_size, shuffle=True)
        lambdas = {"reconstr": p.lambda_recon,
                   "pred": p.lambda_pred,
                   "euc": p.lambda_kl_euc, # 0.1 #0. if encoder_e is None else p.lambda_latent / np.sqrt(p.z_dim_total),
                   "sph": p.lambda_kl_sph #0.25 #0. if encoder_s is None else p.lambda_latent / np.sqrt(p.z_dim_total)
                   }

        best, counter = float("inf"), 0
        logs = {"total": [], "recon": [], "kl_e": [], "kl_s": [], "pred": [], "kappa": [], "mu_s": [], "z_s": [], "z_e": []}

        for epoch in range(p.epochs):
            losses = Sphlin.train_epoch(loader, encoder_e, encoder_s, decoder, pred_head, optimizer, lambdas, device, epoch)
            for k in logs:
                if k in losses and losses[k] is not None:
                    logs[k].append(losses[k])

            # best, counter, stop = early_stop(losses["total"], best, counter, p.earlystop_patience)
            best, counter, stop = early_stop(losses["pred"], best, counter, p.earlystop_patience)
            if epoch % 10 == 0:
                # print(f"Epoch {epoch+1}/{p.epochs}, Total={losses['total']:.4f}, Recon={losses['recon']:.4f}, KL_e={losses['kl_e']:.4f}, KL_s={losses['kl_s']:.4f}, Pred={losses['pred']:.4f}, logkappa={losses.get('logkappa',float('nan')):.4f}")
                print(f"Epoch {epoch+1}/{p.epochs}, Total={losses['total']:.4f}, Recon={losses['recon']:.4f}, KL_e={losses['kl_e']:.4f}, KL_s={losses['kl_s']:.4f}, Pred={losses['pred']:.4f}, kappa={losses.get('kappa',float('nan')):.4f}")

            if stop:
                print(f"Early stopping at epoch {epoch+1}")
                break
        return logs

    @staticmethod
    def latent_dim_handler(X_train, X_lin_tr, X_cyc_tr, p):
        n_tot, n_lin, n_cyc = X_train.shape[1], X_lin_tr.shape[1], X_cyc_tr.shape[1]
        if n_cyc == 0:
            print("⬜️ 100% Euclidean VAE ⬜️")
            z_euc, z_sph = p.z_dim_total, 0
        elif n_lin == 0:
            z_euc, z_sph = 0, p.z_dim_total
            print("🌐 100% Spherical VAE 🌐")
        else:
            print("⚽️ 🟪 Mixed Euclidean-Spherical VAE 🟪 ⚽️")
            z_sph = max(1, min(int(np.ceil(p.z_dim_total * n_cyc / n_tot)), p.z_dim_total - 1))
            # z_sph = 4*n_cyc
            z_euc = p.z_dim_total - z_sph
        print(f"z_euc={z_euc}, z_sph={z_sph}, {n_cyc=}, %feats_cyc={100*n_cyc/n_tot:.2f}")
        return n_tot, n_lin, n_cyc, z_euc, z_sph

    @staticmethod # uses LSTM decoder
    def run_sphlin_LSTM(X_train, X_test, y_train, y_test, p, sliding_size=10, manually_set_cols: list[str] | None = None, prewindowed: bool = False):
        """Run sphlin LSTM with optional manual cyclic column names.
        If prewindowed=True, X_train/X_test must be tuples (X_lin_w, X_cyc_w) and y_train/y_test are already windowed."""

        if prewindowed:
            X_lin_tr_w, X_cyc_tr_w = X_train
            X_lin_te_w, X_cyc_te_w = X_test
            y_tr_w, y_te_w = y_train, y_test

            n_lin = X_lin_tr_w.shape[-1]
            n_cyc = X_cyc_tr_w.shape[-1]
            n_tot = n_lin + n_cyc

            y_tr_w = torch.tensor(y_tr_w, dtype=torch.float32, device=device)
            y_te_w = torch.tensor(y_te_w, dtype=torch.float32, device=device)

            n_tot, n_lin, n_cyc, z_euc, z_sph = Sphlin.latent_dim_handler(torch.zeros((1, n_tot)), torch.zeros((1, n_lin)), torch.zeros((1, n_cyc)), p)

        else:
            if manually_set_cols is not None:
                X_cyc_tr = X_train[manually_set_cols]
                X_lin_tr = X_train.drop(columns=manually_set_cols)
            else:
                X_lin_tr, X_cyc_tr = split_dataset_to_linear_and_cyclic(X_train, threshold=p.cyclic_threshold, verbose=True)

            X_lin_te = X_test[X_lin_tr.columns]
            X_cyc_te = X_test[X_cyc_tr.columns]

            y_tr_s, y_te_s = scale_train_and_test_sets(y_train, y_test)
            X_lin_tr, X_lin_te = scale_train_and_test_sets(X_lin_tr, X_lin_te)
            X_cyc_tr, X_cyc_te = scale_train_and_test_sets(X_cyc_tr, X_cyc_te)

            n_tot, n_lin, n_cyc, z_euc, z_sph = Sphlin.latent_dim_handler(X_train, X_lin_tr, X_cyc_tr, p)

            def make_w(X, d):
                if d == 0: return None
                return Windowing.make_windows_from_X(X, p.window_size, sliding_size, horizon=p.horizon).to(device).view(-1, p.window_size, d)

            X_lin_tr_w, X_lin_te_w = make_w(X_lin_tr, n_lin), make_w(X_lin_te, n_lin)
            X_cyc_tr_w, X_cyc_te_w = make_w(X_cyc_tr, n_cyc), make_w(X_cyc_te, n_cyc)

            y_tr_w = Windowing.make_windows_from_y(y_tr_s, p.window_size, sliding_size, task=p.task, horizon=p.horizon)
            y_te_w = Windowing.make_windows_from_y(y_te_s, p.window_size, sliding_size, task=p.task, horizon=p.horizon)
            y_tr_w = y_tr_w.reshape(y_tr_w.shape[0], -1)
            y_te_w = y_te_w.reshape(y_te_w.shape[0], -1)

            y_tr_w = torch.tensor(y_tr_w, dtype=torch.float32, device=device)
            y_te_w = torch.tensor(y_te_w, dtype=torch.float32, device=device)

        ref_tr = X_lin_tr_w if X_lin_tr_w is not None else X_cyc_tr_w
        ref_te = X_lin_te_w if X_lin_te_w is not None else X_cyc_te_w
        if X_lin_tr_w is None: X_lin_tr_w = torch.zeros((ref_tr.shape[0], p.window_size, 0), device=device)
        if X_cyc_tr_w is None: X_cyc_tr_w = torch.zeros((ref_tr.shape[0], p.window_size, 0), device=device)
        if X_lin_te_w is None: X_lin_te_w = torch.zeros((ref_te.shape[0], p.window_size, 0), device=device)
        if X_cyc_te_w is None: X_cyc_te_w = torch.zeros((ref_te.shape[0], p.window_size, 0), device=device)

        h_split = int(p.hidden_dim / np.sqrt(2))
        enc_e = LSTMEncoderEuclid(n_lin, h_split, z_euc, n_layers=2).to(device) if z_euc > 0 else None
        enc_s = LSTMSphericalEncoder(n_cyc, h_split, z_sph, n_layers=2).to(device) if z_sph > 0 else None

        latent_dim = z_euc + z_sph
        dec = MLPDecoder(latent_dim, p.window_size, n_lin + n_cyc, p.hidden_dim).to(device)
        pred_head = MLPPredHead(latent_dim, y_tr_w.shape[1], hidden_dim=p.hidden_dim * 2).to(device)

        params = list(dec.parameters()) + list(pred_head.parameters())
        if enc_e: params += list(enc_e.parameters())
        if enc_s: params += list(enc_s.parameters())
        opt = torch.optim.AdamW(params, lr=p.lr_optimizer)

        logs = Sphlin.train_loop(X_lin_tr_w, X_cyc_tr_w, y_tr_w, enc_e, enc_s, dec, pred_head, opt, p, device)

        torch.cuda.empty_cache()
        X_lin_tr_w = X_lin_tr_w.cpu()
        X_cyc_tr_w = X_cyc_tr_w.cpu()
        X_lin_te_w = X_lin_te_w.cpu()
        X_cyc_te_w = X_cyc_te_w.cpu()

        Z_train = Sphlin.encode_full_dataset(X_lin_tr_w, X_cyc_tr_w, enc_e, enc_s, z_euc, device)
        torch.cuda.empty_cache()
        Z_test = Sphlin.encode_full_dataset(X_lin_te_w, X_cyc_te_w, enc_e, enc_s, z_euc, device)

        pred_head.eval()
        with torch.no_grad():
            Z_test_device = Z_test.to(device)
            y_hat_torch = pred_head(Z_test_device)
            y_hat = y_hat_torch.cpu().numpy()

        y_te_w = y_te_w.reshape(y_te_w.shape[0], -1).detach().cpu().numpy()
        rmse = root_mean_squared_error(y_te_w, y_hat)
        r2 = r2_score(y_te_w, y_hat)
        mae = mean_absolute_error(y_te_w, y_hat)

        return rmse, r2, mae, (Z_train, Z_test), (y_hat, y_tr_w.detach().cpu().numpy(), y_te_w), (z_euc, z_sph), logs, \
            (enc_e, enc_s, pred_head, X_lin_te_w, X_cyc_te_w)

    @staticmethod #uses MLP decoder
    def XX_run_sphlin_LSTM(X_train, X_test, y_train, y_test, p, sliding_size=10, manually_set_cols: list[str] | None = None):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Feature Splitting
        if manually_set_cols is not None:
            X_cyc_tr = X_train[manually_set_cols]
            X_lin_tr = X_train.drop(columns=manually_set_cols)
        else:
            X_lin_tr, X_cyc_tr = split_dataset_to_linear_and_cyclic(X_train, threshold=p.cyclic_threshold)

        X_lin_te, X_cyc_te = X_test[X_lin_tr.columns], X_test[X_cyc_tr.columns]

        # 2. Scaling
        y_tr_s, y_te_s = scale_train_and_test_sets(y_train, y_test)
        X_lin_tr, X_lin_te = scale_train_and_test_sets(X_lin_tr, X_lin_te)
        X_cyc_tr, X_cyc_te = scale_train_and_test_sets(X_cyc_tr, X_cyc_te)

        n_tot, n_lin, n_cyc, z_euc, z_sph = Sphlin.latent_dim_handler(X_train, X_lin_tr, X_cyc_tr, p)

        # 3. Windowing (Keep d as None if 0 features)
        def make_w(X, d):
            if d == 0: return None
            return Windowing.make_windows_from_X(X, p.window_size, sliding_size, horizon=p.horizon).to(device)

        X_lin_tr_w, X_lin_te_w = make_w(X_lin_tr, n_lin), make_w(X_lin_te, n_lin)
        X_cyc_tr_w, X_cyc_te_w = make_w(X_cyc_tr, n_cyc), make_w(X_cyc_te, n_cyc)

        y_tr_w = Windowing.make_windows_from_y(y_tr_s, p.window_size, sliding_size, task=p.task, horizon=p.horizon)
        y_te_w = Windowing.make_windows_from_y(y_te_s, p.window_size, sliding_size, task=p.task, horizon=p.horizon)
        y_tr_w = torch.tensor(y_tr_w.reshape(y_tr_w.shape[0], -1), dtype=torch.float32, device=device)
        y_te_w_tensor = torch.tensor(y_te_w.reshape(y_te_w.shape[0], -1), dtype=torch.float32, device=device)

        # 4. Model Initialization
        h_split = int(p.hidden_dim / np.sqrt(2))
        enc_e = LSTMEncoderEuclid(n_lin, h_split, z_euc).to(device) if z_euc > 0 else None
        enc_s = LSTMSphericalEncoder(n_cyc, h_split, z_sph).to(device) if z_sph > 0 else None
        
        # decoder: z_dim_total -> window_size * n_tot
        dec = MLPDecoder(p.z_dim_total, p.window_size, n_tot, p.hidden_dim).to(device)
        # pred_head: z_dim_total -> prediction target
        pred_head = MLPPredHead(p.z_dim_total, y_tr_w.shape[1], hidden_dim=p.hidden_dim * 2).to(device)
        
        params = list(dec.parameters()) + list(pred_head.parameters())
        if enc_e: params += list(enc_e.parameters())
        if enc_s: params += list(enc_s.parameters())
        opt = torch.optim.AdamW(params, lr=p.lr_optimizer)

        # 5. Training Loop
        logs = Sphlin.train_loop(X_lin_tr_w, X_cyc_tr_w, y_tr_w, enc_e, enc_s, dec, pred_head, opt, p, device)

        # 6. Neural Inference
        with torch.no_grad():
            # Get test latents (passing back to CPU then back to device is safer for memory)
            Z_test = Sphlin.encode_full_dataset(X_lin_te_w, X_cyc_te_w, enc_e, enc_s, z_euc, device)
            pred_head.eval()
            y_hat = pred_head(Z_test.to(device)).cpu().numpy()

        # 7. Metrics
        y_te_true = y_te_w_tensor.cpu().numpy()
        rmse = root_mean_squared_error(y_te_true, y_hat)
        r2   = r2_score(y_te_true, y_hat)
        mae  = mean_absolute_error(y_te_true, y_hat)

        # Fetch training latents for the return tuple if needed for visualization
        Z_train = Sphlin.encode_full_dataset(X_lin_tr_w, X_cyc_tr_w, enc_e, enc_s, z_euc, device)
        return rmse, r2, mae, (Z_train, Z_test), (y_hat, y_tr_w.detach().cpu().numpy(), y_te_true), (z_euc, z_sph), logs


class TopolinPlots:
    @staticmethod
    def old_plot_torus(Z_numpy, z_dim_euc, feat_a=0, feat_b=1):
        z_torus = Z_numpy[:, z_dim_euc:]
        u1, v1 = z_torus[:, 2*feat_a], z_torus[:, 2*feat_a+1]
        u2, v2 = z_torus[:, 2*feat_b], z_torus[:, 2*feat_b+1]
        
        theta = np.arctan2(v1, u1)
        phi   = np.arctan2(v2, u2)
        
        R, r = 3, 1
        x = (R + r*np.cos(theta))*np.cos(phi)
        y = (R + r*np.cos(theta))*np.sin(phi)
        z = r*np.sin(theta)
        
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(x, y, z, c=theta, cmap='twilight', s=5, alpha=0.6)
        ax.set_axis_off()
        ax.set_xlim(-4,4); ax.set_ylim(-4,4); ax.set_zlim(-4,4)
        plt.title(f"Torus latent (feat {feat_a} & {feat_b})")
        plt.show()

    @staticmethod
    def plot_torus_latent(Z_train, num_cyc_features, z_dim_euclid):
        fig, axes = plt.subplots(1, num_cyc_features, figsize=(num_cyc_features * 4, 4))
        if num_cyc_features == 1: axes = [axes]
        
        for i in range(num_cyc_features):
            # Indexing: Euclid uses first z_dim_euclid dims. 
            # Torus follows in pairs of 2.
            idx = z_dim_euclid + (i * 2)
            x = Z_train[:, idx]     # cos
            y = Z_train[:, idx + 1] # sin
            
            axes[i].scatter(x, y, s=2, alpha=0.5)
            axes[i].set_aspect('equal')
            axes[i].set_title(f"Circle {i} (Latent Space)")
            axes[i].set_xlim(-1.1, 1.1); axes[i].set_ylim(-1.1, 1.1)
        plt.show()

    @staticmethod
    def plot_kappa_dist(encoder_t, loader, device):
        encoder_t.eval()
        all_kappas = []
        with torch.no_grad():
            for _, b_cyc in loader:
                _, kappa = encoder_t(b_cyc.to(device))
                all_kappas.append(kappa.cpu())
        kappas = torch.cat(all_kappas).numpy()
        plt.hist(kappas, bins=50)
        plt.title("Distribution of Concentration (Kappa)")
        plt.show()

