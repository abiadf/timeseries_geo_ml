
import __main__
import sys, os
project_root = os.path.abspath("..")  # adjust if notebook is elsewhere
sys.path.insert(0, project_root)
from typing import Dict, List, Literal, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass

import category_encoders as ce
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numexpr as ne # makes numpy operations faster
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

from scipy.signal import periodogram

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, mean_absolute_error, root_mean_squared_error, r2_score, silhouette_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.random_projection import GaussianRandomProjection

from catboost import CatBoostRegressor, CatBoostClassifier
from pyriemann.tangentspace import TangentSpace

from geo.geo_utils import compute_cyclicity_score, split_dataset_to_linear_and_cyclic, scale_train_and_test_sets, drop_low_variance_cols, Windowing

from geo_encoders import EuclidEncoder, SphericalEncoder, Decoder, LSTMEncoderEuclid, LSTMSphericalEncoder, \
LSTMToroidalEncoder, LSTMDecoder, MLPDecoder, Reparam, MixedEncoder, WithSplit, NoSplit, kl_gaussian, kl_vmf_uniform, \
regularization_vmf, early_stop, fit_catboost_multi, evaluate_model_full, estimate_entropy, pool_latents

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

class Oldtor:
    @staticmethod
    def run_oldtor_LSTM(X_train, X_test, y_train, y_test, p):
        # 1. Split & Identify dimensions
        X_lin_train, X_cyc_train = split_dataset_to_linear_and_cyclic(X_train, threshold=p.cyclic_threshold, verbose=False)
        X_lin_test, X_cyc_test   = X_test[X_lin_train.columns], X_test[X_cyc_train.columns]
        
        # 2. Torus latent
        num_cyc_features = X_cyc_train.shape[1]
        z_dim_torus      = 2 * num_cyc_features #this is fixed for torus VAE
        z_dim_euclid     = p.z_dim_total - z_dim_torus
        
        if (z_dim_euclid < 2) or (p.z_dim_total < z_dim_torus):
            print(f"Warning: z_dim_total ({p.z_dim_total}) too small for {num_cyc_features} cyclic features")
            z_dim_euclid  = max(8, p.z_dim_total - z_dim_torus)
            p.z_dim_total = z_dim_euclid + z_dim_torus

        # print(f">>>>> {X_lin_train.shape=} {z_dim_torus=}")

        # 3. Scaling & Windowing (Standard)
        X_lin_train, X_lin_test = scale_train_and_test_sets(X_lin_train, X_lin_test)
        X_cyc_train, X_cyc_test = scale_train_and_test_sets(X_cyc_train, X_cyc_test)
        y_train_s, y_test_s     = scale_train_and_test_sets(y_train, y_test)

        X_lin_tr_w = Windowing.make_windows_from_X(X_lin_train, p.window_size, p.sliding_size).to(device)
        X_cyc_tr_w = Windowing.make_windows_from_X(X_cyc_train, p.window_size, p.sliding_size).to(device)
        X_lin_te_w = Windowing.make_windows_from_X(X_lin_test, p.window_size, p.sliding_size).to(device)
        X_cyc_te_w = Windowing.make_windows_from_X(X_cyc_test, p.window_size, p.sliding_size).to(device)
        y_train_win= Windowing.make_windows_from_y(y_train_s, p.window_size, p.sliding_size, task=p.task)
        y_test_win = Windowing.make_windows_from_y(y_test_s, p.window_size, p.sliding_size, task=p.task)

        # 4. Initialize Models with dynamic dims
        hidden_dim_split = int(p.hidden_dim / np.sqrt(2))
        encoder_e = LSTMEncoderEuclid(X_lin_train.shape[1], hidden_dim_split, z_dim_euclid).to(device)
        encoder_t = LSTMToroidalEncoder(num_cyc_features, hidden_dim_split, num_cyc_features).to(device)
        decoder   = MLPDecoder(z_dim_total=p.z_dim_total, window_size=p.window_size, output_dim=X_train.shape[1],
                            hidden_dim=p.hidden_dim,).to(device) #decoder should use full hidden_dim
        optimizer = torch.optim.AdamW(list(encoder_e.parameters()) + list(encoder_t.parameters()) + list(decoder.parameters()), lr=p.lr_optimizer)
        loader    = DataLoader(TensorDataset(X_lin_tr_w, X_cyc_tr_w), batch_size=p.batch_size, shuffle=False)

        # 5. Training
        lambdas = {'reconstr': p.lambda_recon,
                'euc': p.lambda_latent/np.sqrt(z_dim_euclid),
                'sph': p.lambda_latent/np.sqrt(z_dim_torus)}

        for _ in range(p.epochs):
            train_linear_and_toroidal_vaes_1_epoch(loader, encoder_e, encoder_t, decoder, optimizer, lambdas, num_cyc_features)

        # 6. Inference (Correcting concatenation)
        with torch.no_grad():
            encoder_e.eval()
            encoder_t.eval()
            
            def get_z_joint(l_win, c_win):
                mu_e, _ = encoder_e(l_win)     # [Batch, z_dim_euc]
                mu_t, _ = encoder_t(c_win)     # [Batch, num_cyc, 2]
                zt_flat = mu_t.reshape(mu_t.size(0), -1) # [Batch, z_dim_torus]
                return torch.cat([mu_e, zt_flat], dim=-1).cpu().numpy()

            Z_train = get_z_joint(X_lin_tr_w, X_cyc_tr_w)
            Z_test  = get_z_joint(X_lin_te_w, X_cyc_te_w)

        # 7. Prediction
        y_hat = fit_catboost_multi(Z_train, y_train_win, Z_test)
        rmse  = np.sqrt(mean_squared_error(y_test_win, y_hat))
        r2    = r2_score(y_test_win, y_hat)
        mae   = mean_absolute_error(y_test_win, y_hat)
        return rmse, r2, mae, (Z_train, Z_test)

    @staticmethod
    def train_linear_and_toroidal_vaes_1_epoch(train_loader, encoder_euc, encoder_torus, decoder, optimizer, lambdas, n_cyc):
        encoder_euc.train()
        encoder_torus.train()
        decoder.train()
        
        total_loss = 0
        for b_lin, b_cyc in train_loader:
            optimizer.zero_grad()
            
            # 1. Encode Euclidean
            mu_euc, logvar_euc = encoder_euc(b_lin)
            z_euclid           = Reparam.reparam_gaussian(mu_euc, logvar_euc)
            
            # 2. Encode Toroidal (Product of N circles)
            # mu_t: [B, N, 2], kappa_t: [B, N, 1]
            mu_t, kappa_t = encoder_torus(b_cyc) 
            
            # Sample each circle on the torus independently
            z_torus_list = []
            for i in range(n_cyc):
                # mu_t[:, i, :] is the [B, 2] direction for the i-th circle
                z_torus_i = Reparam.sample_vmf(mu_t[:, i, :], kappa_t[:, i])
                z_torus_list.append(z_torus_i)
            
            # Concat all circles into the Torus latent [B, 2*N]
            z_torus      = torch.cat(z_torus_list, dim=-1)
            z_joint      = torch.cat([z_euclid, z_torus], dim=-1)
            x_recon_flat = decoder(z_joint) # Decode
            
            # 4. Compute Losses
            # A. Reconstruction (Compare against combined original X)
            x_orig     = torch.cat([b_lin, b_cyc], dim=-1).view(b_lin.size(0), -1)
            recon_loss = F.mse_loss(x_recon_flat, x_orig)
            kl_euc     = -0.5 * torch.sum(1 + logvar_euc - mu_euc.pow(2) - logvar_euc.exp(), dim=1).mean()
            
            # C. Toroidal KL (Sum of vMF KLs for each circle)
            # Analytical KL for vMF in 2D (S^1) involves Bessel functions, 
            # but a common ICML-acceptable approximation for S^1 is:
            kl_torus = 0
            for i in range(n_cyc):
                # For S^1, KL is roughly proportional to kappa
                # This is a standard prior-to-uniform KL for vMF
                k = kappa_t[:, i]
                kl_torus += (k - torch.log(k + 1e-6)).mean() 

            # 5. Backprop
            loss = (lambdas['reconstr'] * recon_loss + lambdas['euc'] * kl_euc + lambdas['sph'] * kl_torus)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    @staticmethod
    def plot_publication_torus(Z_numpy, z_dim_euc, feat_a=0, feat_b=1):
        """
        Z_numpy: [Batch, Total_Dim] 
        z_dim_euc: The index where the cyclic features start
        feat_a: Index of first cyclic feature (0 to N-1)
        feat_b: Index of second cyclic feature (0 to N-1)
        """
        # 1. Slice out the toroidal part ONLY
        z_torus = Z_numpy[:, z_dim_euc:] 
        
        # 2. Extract (u, v) pairs. Each feature has 2 dims.
        # Feature 0 is at [0,1], Feature 1 at [2,3], etc.
        u1, v1 = z_torus[:, 2*feat_a], z_torus[:, 2*feat_a + 1]
        u2, v2 = z_torus[:, 2*feat_b], z_torus[:, 2*feat_b + 1]
        
        # 3. Convert to angles
        theta = np.arctan2(v1, u1) # Angle around the tube
        phi = np.arctan2(v2, u2)   # Angle around the donut center
        
        # 4. Geometry
        R, r = 3, 1
        x = (R + r * np.cos(theta)) * np.cos(phi)
        y = (R + r * np.cos(theta)) * np.sin(phi)
        z = r * np.sin(theta)
        
        # 5. Plotting
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the points
        sc = ax.scatter(x, y, z, c=theta, cmap='twilight', s=2, alpha=0.5)
        
        # Aesthetics for ICML
        ax.set_axis_off()
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_zlim(-4, 4)
        plt.title(f"Latent Torus Projection (Cyc Features {feat_a} & {feat_b})")
        plt.show()

class Sphlin:
    @staticmethod
    def sample_gaussian(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    @staticmethod
    def kl_gaussian(mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

    @staticmethod
    def sample_vmf(mu: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        """Approximate vMF sampling by adding Gaussian noise and normalizing.
        This is NOT exact vMF sampling; it is a heuristic used for speed.
        Valid for large kappa (high concentration) where vMF ≈ Gaussian on the sphere."""
        # kappa is [B, 1], mu is [B, D]
        # eps = torch.randn_like(mu)
        # z   = mu + eps / (kappa.view(-1, 1) + 1e-6)
        # return F.normalize(z, dim=-1)

        """EXACT vMF sampler using Wood's algorithm. “Simulation of the von Mises Fisher Distribution” (Wood 1994)
        it is also used in davidson 2018 and Gopal & Yang, 2014
        mu: [B, D] unit vectors
        kappa: [B, 1] concentration (>=0)
        Returns z: [B, D] unit vectors."""
        B, D = mu.shape
        kappa = kappa.view(-1)
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
    def kl_vmf(mu: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        # method 1: Standard vMF KL approximation: mu is normalized, so mu^2 sum is 1. problematic as mu=1, so its =0
        # return (kappa.view(-1) * (1 - torch.sum(mu ** 2, dim=-1))).mean()

        # method 2: use this (davidson also doesnt use the exact KL, but a concentr. penalty proport. to kappa)
        # why: Exact KL involves modified Bessel functions of fractional order. Numerically unstable. Adds nothing
        # empirically beyond “don’t collapse to high κ”
        return kappa.mean()

        """Compute vMF KL divergence between q(z|mu,kappa) and p(z)=Uniform(S^{d-1}).
        Uses a stable 'standard ratio approximation' for A_d(kappa)=I_{d/2}(kappa)/I_{d/2-1}(kappa).
        Source: Banerjee 2005 and Davidson 2018. see §'concerntr. param' of https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution
            - mu: unit vectors, shape [B, D]
            - kappa: concentration, shape [B, 1]
            - scalar KL mean over batch"""
        # method 3
        # D = mu.shape[-1]
        # kappa = kappa.view(-1)
        # # log normalization constant for vMF: log C_d(kappa)
        # # C_d(kappa) = kappa^{d/2-1} / ((2π)^{d/2} I_{d/2-1}(kappa))
        # log_bessel = torch.log(torch.special.iv(D/2 - 1, kappa) + 1e-20)
        # log_c = (D/2 - 1) * torch.log(kappa + 1e-20) - (D/2) * torch.log(torch.tensor(2 * torch.pi, device=mu.device)) - log_bessel
        # log_c0 = - (D/2) * torch.log(torch.tensor(2 * torch.pi, device=mu.device))  # kappa=0 uniform
        # # expected value of mu^T z under vMF is A_d(kappa)=I_{d/2}(kappa)/I_{d/2-1}(kappa)
        # A  = torch.special.iv(D/2, kappa) / (torch.special.iv(D/2 - 1, kappa) + 1e-20)
        # kl = (kappa * A) - log_c + log_c0
        # return kl.mean()

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
        """encodes the full dataset in batches to avoid OOM errors."""
        if encoder_e is not None: encoder_e.eval()
        if encoder_s is not None: encoder_s.eval()
        zs = []
        with torch.no_grad():
            for i in range(0, len(X_lin), batch_size):
                x_lin   = X_lin[i:i+batch_size].to(device) if X_lin is not None else None
                x_cyc   = X_cyc[i:i+batch_size].to(device) if X_cyc is not None else None
                z_batch = []
                if encoder_e is not None and x_lin is not None and x_lin.shape[-1] > 0:
                    mu, _ = encoder_e(x_lin)
                    z_batch.append(mu)
                if encoder_s is not None and x_cyc is not None and x_cyc.shape[-1] > 0:
                    mu_s, _ = encoder_s(x_cyc)
                    z_batch.append(mu_s)
                zs.append(torch.cat(z_batch, dim=-1).cpu())
        return torch.cat(zs, dim=0)

    @staticmethod
    def train_step(x_lin, x_cyc, y_win, encoder_e, encoder_s, decoder, pred_head, lambdas):
        """One VAE + prediction step."""
        z_parts, kl_e, kl_s = [], torch.tensor(0., device=x_lin.device), torch.tensor(0., device=x_lin.device)

        if encoder_e and x_lin.shape[-1] > 0:
            mu_e, logvar_e = encoder_e(x_lin)
            z_e  = Sphlin.sample_gaussian(mu_e, logvar_e)
            z_parts.append(z_e)
            kl_e = Sphlin.kl_gaussian(mu_e, logvar_e)

        if encoder_s and x_cyc.shape[-1] > 0:
            mu_s, kappa = encoder_s(x_cyc)
            z_s  = Sphlin.sample_vmf(mu_s, kappa)
            z_parts.append(z_s)
            kl_s = Sphlin.kl_vmf(mu_s, kappa)

        z = torch.cat(z_parts, dim=-1)

        targets = []
        if x_lin.shape[-1] > 0: targets.append(x_lin.reshape(x_lin.size(0), -1))
        if x_cyc.shape[-1] > 0: targets.append(x_cyc.reshape(x_cyc.size(0), -1))
        x_target = torch.cat(targets, dim=-1)

        x_hat      = decoder(z)
        recon_loss = F.mse_loss(x_hat, x_target)

        y_hat     = pred_head(z)
        pred_loss = F.mse_loss(y_hat, y_win)

        total_loss = (lambdas["reconstr"] * recon_loss +
                    lambdas["euc"] * kl_e +
                    lambdas["sph"] * kl_s +
                    lambdas["pred"] * pred_loss)

        return {
            "total": total_loss,
            "recon": recon_loss,
            "kl_e": kl_e,
            "kl_s": kl_s,
            "pred": pred_loss,
            "mu_s": mu_s if encoder_s else None,
            "kappa": kappa if encoder_s else None,
            "z_s": z_s if encoder_s else None,
            "z_e": z_e if encoder_e else None
        }


    @staticmethod
    def train_epoch(loader, encoder_e, encoder_s, decoder, pred_head, optimizer, lambdas, device):
        totals = {"total": 0., "recon": 0., "kl_e": 0., "kl_s": 0., "pred": 0., "kappa": 0., "mu_s": 0., "z_s": 0., "z_e": 0.}
        counts = {"kappa": 0, "mu_s": 0, "z_s": 0, "z_e": 0}
        for x_lin, x_cyc, y_win in loader:
            x_lin, x_cyc, y_win = x_lin.to(device), x_cyc.to(device), y_win.to(device)
            optimizer.zero_grad()
            losses = Sphlin.train_step(x_lin, x_cyc, y_win, encoder_e, encoder_s, decoder, pred_head, lambdas)
            losses["total"].backward()
            optimizer.step()

            totals["total"] += losses["total"].item()
            totals["recon"] += losses["recon"].item()
            totals["kl_e"] += losses["kl_e"].item()
            totals["kl_s"] += losses["kl_s"].item()
            totals["pred"] += losses["pred"].item()

            if losses["kappa"] is not None:
                totals["kappa"] += losses["kappa"].mean().item()
                counts["kappa"] += 1
            if losses["mu_s"] is not None:
                totals["mu_s"] += losses["mu_s"].mean().item()
                counts["mu_s"] += 1
            if losses["z_s"] is not None:
                totals["z_s"] += losses["z_s"].mean().item()
                counts["z_s"] += 1
            if losses["z_e"] is not None:
                totals["z_e"] += losses["z_e"].mean().item()
                counts["z_e"] += 1

        out = {k: totals[k] / len(loader) for k in ["total", "recon", "kl_e", "kl_s", "pred"]}
        if counts["kappa"]>0: out["kappa"] = totals["kappa"] / counts["kappa"]
        if counts["mu_s"]>0: out["mu_s"] = totals["mu_s"] / counts["mu_s"]
        if counts["z_s"]>0: out["z_s"] = totals["z_s"] / counts["z_s"]
        if counts["z_e"]>0: out["z_e"] = totals["z_e"] / counts["z_e"]
        return out

    @staticmethod
    def train_loop(XL_w, XC_w, y_w, encoder_e, encoder_s, decoder, pred_head, optimizer, p, device):
        if not torch.is_tensor(XL_w): XL_w = torch.tensor(XL_w, dtype=torch.float32, device=device)
        if not torch.is_tensor(XC_w): XC_w = torch.tensor(XC_w, dtype=torch.float32, device=device)
        if not torch.is_tensor(y_w): y_w = torch.tensor(y_w, dtype=torch.float32, device=device)

        loader  = DataLoader(TensorDataset(XL_w, XC_w, y_w), batch_size=p.batch_size, shuffle=True)
        lambdas = {"reconstr": p.lambda_recon, "pred": p.lambda_pred,
                "euc": 0. if encoder_e is None else p.lambda_latent / np.sqrt(p.z_dim_total),
                "sph": 0. if encoder_s is None else p.lambda_latent / np.sqrt(p.z_dim_total)}

        best, counter = float("inf"), 0
        logs = {"total": [], "recon": [], "kl_e": [], "kl_s": [], "pred": [], "kappa": []}
        for epoch in range(p.epochs):
            losses = Sphlin.train_epoch(loader, encoder_e, encoder_s, decoder, pred_head, optimizer, lambdas, device)
            for k in logs: 
                if k in losses: logs[k].append(losses[k])

            best, counter, stop = early_stop(losses["total"], best, counter, p.earlystop_patience)
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{p.epochs}, Total={losses['total']:.4f}, Recon={losses['recon']:.4f}, KL_e={losses['kl_e']:.4f}, KL_s={losses['kl_s']:.4f}, Pred={losses['pred']:.4f}, κ={losses.get('kappa',float('nan')):.4f}")
            if stop:
                print(f"Early stopping at epoch {epoch+1}")
                break
        return logs

    @staticmethod
    def run_sphlin_LSTM(X_train, X_test, y_train, y_test, p):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        X_lin_tr, X_cyc_tr = split_dataset_to_linear_and_cyclic(X_train, threshold=p.cyclic_threshold, verbose=False)
        X_lin_te, X_cyc_te = X_test[X_lin_tr.columns], X_test[X_cyc_tr.columns]

        y_tr_s, y_te_s     = scale_train_and_test_sets(y_train, y_test)
        X_lin_tr, X_lin_te = scale_train_and_test_sets(X_lin_tr, X_lin_te)
        X_cyc_tr, X_cyc_te = scale_train_and_test_sets(X_cyc_tr, X_cyc_te)

        n_tot, n_lin, n_cyc = X_train.shape[1], X_lin_tr.shape[1], X_cyc_tr.shape[1]
        if n_cyc == 0:
            z_euc, z_sph = p.z_dim_total, 0
        elif n_lin == 0:
            z_euc, z_sph = 0, p.z_dim_total
        else:
            z_sph = max(1, min(int(round(p.z_dim_total * n_cyc / n_tot)), p.z_dim_total - 1))
            z_euc = p.z_dim_total - z_sph

        print(f"z_euc={z_euc}, z_sph={z_sph}, %feats_cyc={100*n_cyc/n_tot:.2f}")

        def make_w(X, d):
            if d == 0: return None
            return Windowing.make_windows_from_X(X, p.window_size, p.sliding_size).to(device).view(-1, p.window_size, d)

        X_lin_tr_w, X_lin_te_w = make_w(X_lin_tr, n_lin), make_w(X_lin_te, n_lin)
        X_cyc_tr_w, X_cyc_te_w = make_w(X_cyc_tr, n_cyc), make_w(X_cyc_te, n_cyc)

        y_tr_w = Windowing.make_windows_from_y(y_tr_s, p.window_size, p.sliding_size, task=p.task)
        y_te_w = Windowing.make_windows_from_y(y_te_s, p.window_size, p.sliding_size, task=p.task)
        y_tr_w = torch.tensor(y_tr_w, dtype=torch.float32, device=device)
        y_te_w = torch.tensor(y_te_w, dtype=torch.float32, device=device)

        ref_tr = X_lin_tr_w if X_lin_tr_w is not None else X_cyc_tr_w
        ref_te = X_lin_te_w if X_lin_te_w is not None else X_cyc_te_w
        if X_lin_tr_w is None: X_lin_tr_w = torch.zeros((ref_tr.shape[0], p.window_size, 0), device=device)
        if X_cyc_tr_w is None: X_cyc_tr_w = torch.zeros((ref_tr.shape[0], p.window_size, 0), device=device)
        if X_lin_te_w is None: X_lin_te_w = torch.zeros((ref_te.shape[0], p.window_size, 0), device=device)
        if X_cyc_te_w is None: X_cyc_te_w = torch.zeros((ref_te.shape[0], p.window_size, 0), device=device)

        h_split   = int(p.hidden_dim / np.sqrt(2))
        enc_e     = LSTMEncoderEuclid(n_lin, h_split, z_euc).to(device) if z_euc > 0 else None
        enc_s     = LSTMSphericalEncoder(n_cyc, h_split, z_sph).to(device) if z_sph > 0 else None
        dec       = MLPDecoder(p.z_dim_total, p.window_size, n_tot, p.hidden_dim).to(device)
        pred_head = torch.nn.Linear(p.z_dim_total, y_tr_w.shape[1]).to(device)

        params = list(dec.parameters()) + list(pred_head.parameters())
        if enc_e: params += list(enc_e.parameters())
        if enc_s: params += list(enc_s.parameters())
        opt = torch.optim.AdamW(params, lr=p.lr_optimizer)

        Sphlin.train_loop(X_lin_tr_w, X_cyc_tr_w, y_tr_w, enc_e, enc_s, dec, pred_head, opt, p, device)

        torch.cuda.empty_cache()
        X_lin_tr_w = X_lin_tr_w.cpu()
        X_cyc_tr_w = X_cyc_tr_w.cpu()
        X_lin_te_w = X_lin_te_w.cpu()
        X_cyc_te_w = X_cyc_te_w.cpu()

        Z_train = Sphlin.encode_full_dataset(X_lin_tr_w, X_cyc_tr_w, enc_e, enc_s, z_euc, device)
        torch.cuda.empty_cache()
        Z_test  = Sphlin.encode_full_dataset(X_lin_te_w, X_cyc_te_w, enc_e, enc_s, z_euc, device)

        y_hat  = fit_catboost_multi(Z_train.detach().cpu().numpy(), y_tr_w.detach().cpu().numpy(), Z_test.detach().cpu().numpy())
        y_te_w = y_te_w.detach().cpu().numpy()

        rmse = root_mean_squared_error(y_te_w, y_hat)
        r2   = r2_score(y_te_w, y_hat)
        mae  = mean_absolute_error(y_te_w, y_hat)
        # return rmse, r2, mae, (Z_train, Z_test), (y_hat, y_tr_w, y_te_w)
        return rmse, r2, mae, (Z_train, Z_test), (y_hat, y_tr_w.detach().cpu().numpy(), y_te_w), (z_euc, z_sph)


class Torlin:
    @staticmethod
    def sample_vmf_approximate(mu, kappa):
        "Approximate vMF sampler (S^1 sampling in 2D)"
        noise = torch.randn_like(mu) * (1.0/kappa)
        return F.normalize(mu + noise, dim=-1)

    @staticmethod
    def sample_vmf_exact(mu_xy: torch.Tensor, kappa: torch.Tensor):
        """Exact vMF sampler on S¹ using VonMises.
        mu_xy: (B, 2)
        kappa: (B,)"""
        mu_xy    = F.normalize(mu_xy, dim=-1)
        mu_angle = torch.atan2(mu_xy[:, 1], mu_xy[:, 0])
        # theta    = torch.distributions.VonMises(mu_angle, kappa).rsample()
        theta    = torch.distributions.VonMises(mu_angle, kappa).sample()
        return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

    @staticmethod
    def vmf_kl_s1(kappa: torch.Tensor):
        """Exact KL(vMF || Uniform) on S¹."""
        i0 = torch.special.i0(kappa)
        i1 = torch.special.i1(kappa)
        return kappa * (i1 / (i0 + 1e-8)) - torch.log(i0 + 1e-8)

    @staticmethod # this is the good one (works wwith the old versio nof lstm encoder)! but gives warnings
    def X_run_torlin_pipeline(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: np.ndarray,
                              y_test: np.ndarray, p, angular_features_list: list) -> Tuple:
        """Full Toroidal VAE pipeline with Reconstruction Decoder and original latent dimension splitting logic."""
        if len(angular_features_list) > 0:
            X_cyc_train = X_train[angular_features_list]
            X_cyc_test = X_test[angular_features_list]
            X_linear_train = X_train.drop(columns=angular_features_list)
            X_linear_test = X_test.drop(columns=angular_features_list)
        else:
            X_cyc_train = pd.DataFrame()
            X_cyc_test = pd.DataFrame()
            X_linear_train = X_train.copy()
            X_linear_test = X_test.copy()

        num_cyc_features = X_cyc_train.shape[1]
        z_dim_torus = 2 * num_cyc_features
        z_dim_euclid = p.z_dim_total - z_dim_torus
        if z_dim_euclid < 8:
            z_dim_euclid = 8
        print(f"Latent Split: Euclid={z_dim_euclid}, Torus Circles={num_cyc_features}, Total Dim={z_dim_euclid + z_dim_torus}")

        X_lin_train, X_lin_test = scale_train_and_test_sets(X_linear_train, X_linear_test)
        if num_cyc_features > 0:
            X_cyc_train, X_cyc_test = scale_train_and_test_sets(X_cyc_train, X_cyc_test)
        else:
            X_cyc_train = torch.empty((len(X_lin_train), 0), dtype=torch.float32)
            X_cyc_test = torch.empty((len(X_lin_test), 0), dtype=torch.float32)

        X_lin_tr_w = Windowing.make_windows_from_X(X_lin_train, p.window_size, p.sliding_size).to(device)
        X_cyc_tr_w = Windowing.make_windows_from_X(X_cyc_train, p.window_size, p.sliding_size).to(device)
        X_lin_te_w = Windowing.make_windows_from_X(X_lin_test, p.window_size, p.sliding_size).to(device)
        X_cyc_te_w = Windowing.make_windows_from_X(X_cyc_test, p.window_size, p.sliding_size).to(device)

        hidden_dim_split = int(p.hidden_dim / np.sqrt(2))
        enc_e = LSTMEncoderEuclid(X_lin_train.shape[1], hidden_dim_split, z_dim_euclid).to(device)
        enc_t = LSTMToroidalEncoder(X_cyc_train.shape[1], hidden_dim_split, num_cyc_features).to(device) if num_cyc_features > 0 else None

        output_dim = X_lin_train.shape[1] + X_cyc_train.shape[1]
        decoder = MLPDecoder(z_dim_total=z_dim_euclid + z_dim_torus, window_size=p.window_size, output_dim=output_dim, hidden_dim=p.hidden_dim).to(device)
        optimizer = torch.optim.AdamW(list(enc_e.parameters()) + (list(enc_t.parameters()) if enc_t else []) + list(decoder.parameters()), lr=p.lr_optimizer)

        loader = DataLoader(TensorDataset(X_lin_tr_w, X_cyc_tr_w), batch_size=p.batch_size, shuffle=True)
        lambdas = {'recon': p.lambda_recon, 'euc': p.lambda_latent / np.sqrt(z_dim_euclid), 'tor': p.lambda_latent / np.sqrt(z_dim_torus if z_dim_torus > 0 else 1)}
        criterion = nn.MSELoss()
        best_loss = float("inf")
        counter = 0

        for epoch in range(p.epochs):
            enc_e.train()
            if enc_t: enc_t.train()
            decoder.train()
            epoch_loss = 0

            for b_lin, b_cyc in loader:
                optimizer.zero_grad()
                b_lin, b_cyc = b_lin.to(device), b_cyc.to(device)

                mu_e, logvar_e = enc_e(b_lin)
                if mu_e.dim() == 3:
                    mu_e = mu_e[:, -1, :]
                    logvar_e = logvar_e[:, -1, :]
                z_e = Reparam.reparam_gaussian(mu_e, logvar_e)

                if enc_t:
                    mu_t, kappa_t = enc_t(b_cyc)
                    if mu_t.dim() == 4:
                        mu_t = mu_t[:, -1, :, :]
                        kappa_t = kappa_t[:, -1, :]
                    elif mu_t.dim() == 3:
                        mu_t = mu_t[:, -1, :].unsqueeze(1)
                        kappa_t = kappa_t[:, -1].unsqueeze(1)
                    elif mu_t.dim() == 2:
                        mu_t = mu_t.unsqueeze(1)
                        kappa_t = kappa_t.unsqueeze(1)

                    n_cyc_enc = mu_t.shape[1]
                    if n_cyc_enc != num_cyc_features:
                        print("WARNING: mismatch", num_cyc_features, "vs", n_cyc_enc)

                    z_t_list = [Torlin.sample_vmf_exact(mu_t[:, i, :], kappa_t[:, i]) for i in range(min(n_cyc_enc, num_cyc_features))]
                    if len(z_t_list) < num_cyc_features:
                        pad = torch.zeros((b_lin.size(0), 2 * (num_cyc_features - len(z_t_list))), device=device)
                        z_t = torch.cat(z_t_list + [pad], dim=-1)
                    else:
                        z_t = torch.cat(z_t_list[:num_cyc_features], dim=-1)
                else:
                    z_t = torch.zeros((b_lin.size(0), z_dim_torus), device=device)

                z_combined = torch.cat([z_e, z_t], dim=-1)
                recon = decoder(z_combined)

                target = torch.cat([b_lin, b_cyc], dim=-1)
                loss_recon = criterion(recon.view(recon.size(0), -1), target.view(target.size(0), -1))
                kl_euc = -0.5 * torch.sum(1 + logvar_e - mu_e.pow(2) - logvar_e.exp(), dim=1).mean()
                kl_tor = Torlin.vmf_kl_s1(kappa_t).mean() if enc_t else 0

                total_loss = lambdas['recon'] * loss_recon + lambdas['euc'] * kl_euc + lambdas['tor'] * kl_tor
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()

            avg_epoch_loss = epoch_loss / len(loader)
            best_loss, counter, stop = early_stop(avg_epoch_loss, best_loss, counter, p.earlystop_patience)
            if stop:
                break

        enc_e.eval()
        if enc_t: enc_t.eval()

        with torch.no_grad():
            mu_e_te, _ = enc_e(X_lin_te_w)
            if mu_e_te.dim() == 3:
                mu_e_te = mu_e_te[:, -1, :]
            if enc_t:
                mu_t_te, _ = enc_t(X_cyc_te_w)
                if mu_t_te.dim() == 4:
                    mu_t_te = mu_t_te[:, -1, :, :]
                elif mu_t_te.dim() == 3:
                    mu_t_te = mu_t_te[:, -1, :].unsqueeze(1)
                elif mu_t_te.dim() == 2:
                    mu_t_te = mu_t_te.unsqueeze(1)

                z_t_te = torch.cat([F.normalize(mu_t_te[:, i, :], dim=-1) for i in range(min(mu_t_te.shape[1], num_cyc_features))], dim=-1)
                if z_t_te.shape[1] < z_dim_torus:
                    pad = torch.zeros((z_t_te.size(0), z_dim_torus - z_t_te.shape[1]), device=device)
                    z_t_te = torch.cat([z_t_te, pad], dim=-1)
            else:
                z_t_te = torch.zeros((mu_e_te.size(0), z_dim_torus), device=device)

            Z_test = torch.cat([mu_e_te, z_t_te], dim=-1).cpu().numpy()

            mu_e_tr, _ = enc_e(X_lin_tr_w)
            if mu_e_tr.dim() == 3:
                mu_e_tr = mu_e_tr[:, -1, :]
            if enc_t:
                mu_t_tr, _ = enc_t(X_cyc_tr_w)
                if mu_t_tr.dim() == 4:
                    mu_t_tr = mu_t_tr[:, -1, :, :]
                elif mu_t_tr.dim() == 3:
                    mu_t_tr = mu_t_tr[:, -1, :].unsqueeze(1)
                elif mu_t_tr.dim() == 2:
                    mu_t_tr = mu_t_tr.unsqueeze(1)

                z_t_tr = torch.cat([F.normalize(mu_t_tr[:, i, :], dim=-1) for i in range(min(mu_t_tr.shape[1], num_cyc_features))], dim=-1)
                if z_t_tr.shape[1] < z_dim_torus:
                    pad = torch.zeros((z_t_tr.size(0), z_dim_torus - z_t_tr.shape[1]), device=device)
                    z_t_tr = torch.cat([z_t_tr, pad], dim=-1)
            else:
                z_t_tr = torch.zeros((mu_e_tr.size(0), z_dim_torus), device=device)

            Z_train = torch.cat([mu_e_tr, z_t_tr], dim=-1).cpu().numpy()

        y_train_s, y_test_s = scale_train_and_test_sets(y_train, y_test)
        y_train_win = Windowing.make_windows_from_y(y_train_s, p.window_size, p.sliding_size, task=p.task)
        y_test_win = Windowing.make_windows_from_y(y_test_s, p.window_size, p.sliding_size, task=p.task)

        y_hat = fit_catboost_multi(Z_train, y_train_win, Z_test)
        rmse  = root_mean_squared_error(y_test_win, y_hat)
        r2    = r2_score(y_test_win, y_hat)
        mae   = mean_absolute_error(y_test_win, y_hat)
        return rmse, r2, mae, (Z_train, Z_test)

    @staticmethod  # old (no warmup)
    def X_train_torlin(encoder_e, encoder_t, decoder, loader, optimizer, p, lambdas, num_cyc_features, z_dim_torus, device):
        criterion          = nn.MSELoss()
        best_loss, counter = float("inf"), 0
        
        for epoch in range(p.epochs):
            encoder_e.train(); decoder.train()
            if encoder_t: encoder_t.train()
            epoch_loss   = 0
            total_kl_tor = 0

            for b_lin, b_cyc in loader:
                optimizer.zero_grad()
                b_lin, b_cyc = b_lin.to(device), b_cyc.to(device)

                # 1. Encode
                mu_e, logvar_e = encoder_e(b_lin)
                z_e = Reparam.reparam_gaussian(mu_e, logvar_e)

                if encoder_t:
                    mu_t, kappa_t = encoder_t(b_cyc)
                    # Sample each circle on the torus
                    z_t    = torch.cat([Torlin.sample_vmf_exact(mu_t[:, i, :], kappa_t[:, i]) 
                                       for i in range(num_cyc_features)], dim=-1)
                    kl_tor = Torlin.vmf_kl_s1(kappa_t).mean()
                    total_kl_tor += kl_tor.item()
                else:
                    z_t    = torch.zeros((b_lin.size(0), z_dim_torus), device=device)
                    kl_tor = 0

                # 2. Decode & Loss
                z_combined = torch.cat([z_e, z_t], dim=-1)
                recon      = decoder(z_combined)
                
                target     = torch.cat([b_lin, b_cyc], dim=-1).view(b_lin.size(0), -1)
                loss_recon = criterion(recon.view(recon.size(0), -1), target)
                kl_euc     = -0.5 * torch.sum(1 + logvar_e - mu_e.pow(2) - logvar_e.exp(), dim=1).mean()
                
                total_loss = (lambdas['recon'] * loss_recon + 
                            lambdas['euc'] * kl_euc + 
                            lambdas['tor'] * kl_tor)
                
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()

            avg_loss = epoch_loss / len(loader)
            
            # Monitor progress every 10 epochs
            if epoch % 10 == 0 and encoder_t:
                print(f"Epoch {epoch} | KL Tor: {total_kl_tor/len(loader):.4f} | Recon: {avg_loss:.4f}")

            best_loss, counter, stop = early_stop(avg_loss, best_loss, counter, p.earlystop_patience)
            if stop: 
                break
        return encoder_e, encoder_t, decoder

    @staticmethod
    def _train_torlin(encoder_e, encoder_t, decoder, loader, optimizer, p, lambdas, num_cyc_features, z_dim_torus, device, use_warmup=True):
        criterion    = nn.MSELoss(reduction='none')
        best_loss, counter = float("inf"), 0
        total_steps  = p.epochs * len(loader)
        current_step = 0
        
        for epoch in range(p.epochs):
            encoder_e.train(); decoder.train()
            if encoder_t: encoder_t.train()
            
            epoch_mse_lin, epoch_mse_cyc = 0, 0
            epoch_kl_euc, epoch_kl_tor = 0, 0

            for b_lin, b_cyc in loader:
                optimizer.zero_grad()
                b_lin, b_cyc = b_lin.to(device), b_cyc.to(device)
                current_step += 1
                beta = min(1.0, current_step / (0.3 * total_steps + 1e-9)) if use_warmup else 1.0

                # 1. Encode
                mu_e, logvar_e = encoder_e(b_lin)
                z_e = Reparam.reparam_gaussian(mu_e, logvar_e)
                
                mu_t, kappa_t = encoder_t(b_cyc)
                z_t = torch.cat([Torlin.sample_vmf_exact(mu_t[:, i, :], kappa_t[:, i]) for i in range(num_cyc_features)], dim=-1)
                
                # 2. Decode
                z_combined = torch.cat([z_e, z_t], dim=-1)
                recon      = decoder(z_combined)
                
                # 3. Component Losses
                target  = torch.cat([b_lin, b_cyc], dim=-1).view(b_lin.size(0), -1)
                raw_mse = criterion(recon.view(recon.size(0), -1), target)
                
                flat_dim_lin = b_lin.shape[1] * b_lin.shape[2]
                mse_lin = raw_mse[:, :flat_dim_lin].mean()
                mse_cyc = raw_mse[:, flat_dim_lin:].mean()
                
                kl_euc = -0.5 * torch.sum(1 + logvar_e - mu_e.pow(2) - logvar_e.exp(), dim=1).mean()
                kl_tor = Torlin.vmf_kl_s1(kappa_t).mean()

                loss = (lambdas['recon'] * (mse_lin + mse_cyc)) + \
                       (lambdas['euc'] * beta * kl_euc) + \
                       (lambdas['tor'] * beta * kl_tor)
                
                loss.backward()
                optimizer.step()
                
                epoch_mse_lin += mse_lin.item()
                epoch_mse_cyc += mse_cyc.item()
                epoch_kl_euc  += kl_euc.item()
                epoch_kl_tor  += kl_tor.item()

            if epoch % 10 == 0:
                L = len(loader)
                print(f"Epoch {epoch} | MSE(lin/cyc): {epoch_mse_lin/L:.3f}/{epoch_mse_cyc/L:.3f} | KL(euc/tor): {epoch_kl_euc/L:.3f}/{epoch_kl_tor/L:.4f}")
        return encoder_e, encoder_t, decoder

    @staticmethod
    def run_torlin(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: np.ndarray, 
                   y_test: np.ndarray, p, angular_features_list: list, epsilon=1e-3) -> Tuple:
        # 1. Feature Splitting
        if len(angular_features_list) > 0:
            X_cyc_train    = X_train[angular_features_list]
            X_cyc_test     = X_test[angular_features_list]
            X_linear_train = X_train.drop(columns=angular_features_list)
            X_linear_test  = X_test.drop(columns=angular_features_list)
        else:
            X_cyc_train, X_cyc_test       = pd.DataFrame(), pd.DataFrame()
            X_linear_train, X_linear_test = X_train.copy(), X_test.copy()

        num_cyc_features = X_cyc_train.shape[1]
        z_dim_torus      = 2 * num_cyc_features
        z_dim_euclid     = max(8, p.z_dim_total - z_dim_torus)

        # 2. Scaling & Windowing
        X_lin_train, X_lin_test = scale_train_and_test_sets(X_linear_train, X_linear_test)
        if num_cyc_features > 0:
            X_cyc_train, X_cyc_test = scale_train_and_test_sets(X_cyc_train, X_cyc_test)
        else:
            X_cyc_train = torch.empty((len(X_lin_train), 0))
            X_cyc_test  = torch.empty((len(X_lin_test), 0))

        X_lin_tr_w = Windowing.make_windows_from_X(X_lin_train, p.window_size, p.sliding_size).to(device)
        X_cyc_tr_w = Windowing.make_windows_from_X(X_cyc_train, p.window_size, p.sliding_size).to(device)
        X_lin_te_w = Windowing.make_windows_from_X(X_lin_test, p.window_size, p.sliding_size).to(device)
        X_cyc_te_w = Windowing.make_windows_from_X(X_cyc_test, p.window_size, p.sliding_size).to(device)

        # 3. Model Init
        hidden_dim_split = int(p.hidden_dim / np.sqrt(2))
        output_dim       = X_lin_train.shape[1] + X_cyc_train.shape[1]

        encoder_e  = LSTMEncoderEuclid(X_lin_train.shape[1], hidden_dim_split, z_dim_euclid).to(device)
        encoder_t  = LSTMToroidalEncoder(X_cyc_train.shape[1], hidden_dim_split, num_cyc_features, n_layers=2, epsilon=epsilon).to(device) if num_cyc_features > 0 else None
        decoder    = MLPDecoder(z_dim_total=z_dim_euclid + z_dim_torus, window_size=p.window_size, 
                                output_dim=output_dim, hidden_dim=p.hidden_dim).to(device)
        
        params    = list(encoder_e.parameters()) + list(decoder.parameters())
        if encoder_t: params += list(encoder_t.parameters())
        optimizer = torch.optim.AdamW(params, lr=p.lr_optimizer)

        # 4. Training Loop
        loader  = DataLoader(TensorDataset(X_lin_tr_w, X_cyc_tr_w), batch_size=p.batch_size, shuffle=True)
        lambdas = {'recon': p.lambda_recon,
                   'euc': p.lambda_latent / np.sqrt(z_dim_euclid),
                   'tor': p.lambda_latent / np.sqrt(z_dim_torus if z_dim_torus > 0 else 1)}
        encoder_e, encoder_t, decoder = Torlin._train_torlin(encoder_e, encoder_t, decoder, loader, optimizer, p, lambdas, num_cyc_features, z_dim_torus, device)

        # 5. Inference
        encoder_e.eval()
        if encoder_t: encoder_t.eval()
        
        with torch.no_grad():
            def get_z(l_w, c_w):
                mu_e_val, _ = encoder_e(l_w)
                if encoder_t:
                    mu_t_val, _ = encoder_t(c_w)
                    # For inference,  use the normalized mean direction (mu)
                    z_t_val = torch.cat([F.normalize(mu_t_val[:, i, :], dim=-1) 
                                        for i in range(num_cyc_features)], dim=-1)
                else:
                    z_t_val = torch.zeros((mu_e_val.size(0), z_dim_torus), device=device)
                return torch.cat([mu_e_val, z_t_val], dim=-1).cpu().numpy()

            Z_train = get_z(X_lin_tr_w, X_cyc_tr_w)
            Z_test  = get_z(X_lin_te_w, X_cyc_te_w)

        # 6. Downstream
        y_tr_s, y_te_s = scale_train_and_test_sets(y_train, y_test)
        y_train_w      = Windowing.make_windows_from_y(y_tr_s, p.window_size, p.sliding_size, task=p.task)
        y_test_w       = Windowing.make_windows_from_y(y_te_s, p.window_size, p.sliding_size, task=p.task)

        y_hat = fit_catboost_multi(Z_train, y_train_w, Z_test)
        rmse  = root_mean_squared_error(y_test_w, y_hat)
        r2    = r2_score(y_test_w, y_hat)
        mae   = mean_absolute_error(y_test_w, y_hat)

        TopolinPlots.plot_kappa_dist(encoder_t, loader, device)
        return rmse, r2, mae, (Z_train, Z_test), (y_hat, y_train_w, y_test_w), (z_dim_euclid, z_dim_torus)


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

