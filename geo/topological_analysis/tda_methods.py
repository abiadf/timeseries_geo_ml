"""Module for topological data analysis (TDA) methods."""

from datetime import datetime
import math
import yaml

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from ripser import ripser
from persim import PersistenceImager, plot_diagrams, PersLandscapeApprox, PersLandscapeExact
from persim.persistent_entropy import persistent_entropy
from gtda.diagrams import BettiCurve

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.regression import R2Score

from scipy.stats import wasserstein_distance

class LSTMAutoencoder(nn.Module):
    """LSTM autoencoder for time-series reconstruction + latent embedding. Good accuracy comes from
    teacher forcing in "decode": feeding the true previous value at each step instead of the predicted one
    Encodes X -> Z (sequence) and reconstructs X."""

    def __init__(self, input_dim: int, latent_dim: int,
                 encoder_hidden_dim: int = 64,
                 decoder_hidden_dim: int = 64):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # encoder
        self.encoder = nn.LSTM(input_dim, encoder_hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(encoder_hidden_dim, latent_dim)

        # decoder
        self.decoder = nn.LSTM(input_dim + latent_dim, decoder_hidden_dim, batch_first=True)
        self.to_h0 = nn.Linear(latent_dim, decoder_hidden_dim)
        self.to_c0 = nn.Linear(latent_dim, decoder_hidden_dim)
        self.output_layer = nn.Linear(decoder_hidden_dim, input_dim)

        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.encoder(x)
        return self.to_latent(out)

    def decode(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x_shift = torch.zeros_like(x)
        x_shift[:, 1:, :] = x[:, :-1, :]

        dec_in = torch.cat([x_shift, z], dim=-1)

        h0 = self.to_h0(z[:, -1]).unsqueeze(0)
        c0 = self.to_c0(z[:, -1]).unsqueeze(0)

        out, _ = self.decoder(dec_in, (h0, c0))
        return self.output_layer(out)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z, x)
        return x_hat, z


class PersistenceAnalysis:
    """Class for computing + plotting persistence tools: persistence diagrams, entropy, images, and Betti curves"""

    def __init__(self, dataset: np.ndarray = None, max_dim: int = 2, pixel_size: float = 0.1):
        """max_dim = max homology dim to compute (0 = connected components, 1 = loops...)
        pixel_size is for persistence images: smaller = higher res, larger = more smoothing"""
        self.dataset = dataset
        self.max_dim = max_dim
        self._pimgr  = PersistenceImager(pixel_size=pixel_size)
        self._bc     = BettiCurve()

    def compute_persistence_diagrams(self, X: np.ndarray | torch.Tensor) -> list[np.ndarray]:
        """Returns list of persistence diagrams (one per homology dimension). Ripser expects np array, not tensor"""
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        return ripser(X, maxdim=self.max_dim)['dgms']

    def compute_entropy(self, diagrams_list: list[np.ndarray], feature_dim=None) -> np.ndarray:
        """Return entropy per homology dimension or specific one."""
        if feature_dim is None:
            return persistent_entropy(diagrams_list)
        return persistent_entropy([diagrams_list[feature_dim]])

    def compute_persistence_image(self, diagrams_list: list[np.ndarray]) -> np.ndarray:
        """Convert diagrams to persistence images."""
        diagrams_list = [d for d in diagrams_list if len(d) > 0]

        if len(diagrams_list) == 0 or all(len(d) == 0 for d in diagrams_list):
            return [np.zeros((10, 10))]

        self._pimgr.fit(diagrams_list)
        return self._pimgr.transform(diagrams_list)

    def _compute_persistence_images_global(self, dgms_all: list[list[np.ndarray]]) -> np.ndarray:
        """Compute persistence images for many diagrams with ONE global fit
        output shape: (n_diagrams, H, W)"""

        # flatten all diagrams
        all_dgms = [d for dgms in dgms_all for d in dgms if len(d) > 0]

        if len(all_dgms) == 0:
            return np.zeros((len(dgms_all), 10, 10))

        self._pimgr.fit(all_dgms)

        imgs = []
        for dgms in dgms_all:
            dgms = [d for d in dgms if len(d) > 0]

            if len(dgms) == 0:
                imgs.append(np.zeros((10, 10)))
            else:
                imgs.append(self._pimgr.transform(dgms)[0])
        return np.stack(imgs)

    def compute_persistence_landscape(self, diagrams_list: list[np.ndarray], approx: bool = True, num_steps: int = 100, K_layers: int = 3, flatten: bool = True) -> np.ndarray:
        """Compute persistence landscape. See https://persim.scikit-tda.org/en/latest/notebooks/Persistence%20landscapes.html 
        K = #layers in the landscape (the k-th layer is the k-th largest "tent function" at each point in the grid)
        approx=True -> PersLandscapeApprox (ML-ready grid)
        approx=False -> PersLandscapeExact (list of critical points)
        returns array if approx else object"""
        diagrams_list = [d for d in diagrams_list if len(d) > 0]
        if len(diagrams_list) == 0:
            return np.zeros((0, num_steps)) if approx else None

        if approx:
            persist_landscape = PersLandscapeApprox(dgms=diagrams_list, num_steps=num_steps)#, k=K_layers)
            vals              = persist_landscape.values  # shape: (k_layers, num_steps)
            return vals.reshape(-1) if flatten else vals
        return PersLandscapeExact(dgms=diagrams_list)

    def _compute_persistence_landscapes_global(self, dgms_all: list[list[np.ndarray]], num_steps: int = 100, K_layers: int = 3) -> np.ndarray:
        """Compute landscapes for many diagrams → (N, D). This functino is called by compute_persistence_features"""
        landscapes = []
        for dgms in dgms_all:
            # vals = self.compute_persistence_landscape(dgms, approx=True, K_layers=K_layers, num_steps=num_steps, flatten=False)
            # vals = self.compute_persistence_landscape(dgms, approx=True, num_steps=num_steps, flatten=False)
            vals = self.compute_persistence_landscape(dgms, approx=True, num_steps=num_steps, K_layers=K_layers, flatten=False)
            k    = vals.shape[0]

            if k == 0:
                vals = np.zeros((K_layers, num_steps))  # <-- FIX (important)
            elif k < K_layers:
                vals = np.vstack([vals, np.zeros((K_layers - k, num_steps))])
            else:
                vals = vals[:K_layers]
            landscapes.append(vals.reshape(-1))

        return np.stack(landscapes)

    def compute_betti_curves(self, diagrams_list: list[np.ndarray]) -> np.ndarray:
        """Return Betti curves."""
        X = self.ripser_to_gtda(diagrams_list)
        X = X[None, :, :]  # add batch dimension -> (1, n_points, 3)
        return self._bc.fit_transform(X)

    def ripser_to_gtda(self, diagrams_list: list[np.ndarray]) -> np.ndarray:
        """Convert ripser diagrams to giotto format."""
        out = []
        for dim, dgm in enumerate(diagrams_list):
            if dgm is None or len(dgm) == 0:
                continue
            labels = np.full((dgm.shape[0], 1), dim)
            out.append(np.hstack([dgm, labels]))
        if len(out) == 0:
            return np.zeros((0, 3))
        return np.vstack(out)

    def convert_persistence_diagrams_to_tensor(self, diagrams_list: list[np.ndarray]) -> np.ndarray:
        """Convert list of diagrams -> padded tensor (N, P, 2)."""
        max_len = max(len(dgm) for dgm in diagrams_list)
        out     = np.zeros((len(diagrams_list), max_len, 2), dtype=np.float32)

        for i, dgm in enumerate(diagrams_list):
            n = len(dgm)
            out[i, :n] = dgm
        return out

    def remove_inf(self, dgms: list[np.ndarray]) -> list[np.ndarray]:
        """Remove points with infinite death times from each diagram."""
        return [dgm[np.isfinite(dgm).all(axis=1)] for dgm in dgms]


class PersistencePlotter(PersistenceAnalysis):
    """Class for plotting persistence diagrams, images, Betti curves, and entropy, for single diagrams or grids across windows."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _cache_diagrams(self, z_windows):
        if isinstance(z_windows, torch.Tensor):
            z_windows = z_windows.detach().cpu().numpy()

        return [self.remove_inf(self.compute_persistence_diagrams(z_windows[i]))
                for i in range(z_windows.shape[0])]

    def _window_grid(self, z_windows, transform_fn, plot_fn, max_windows=8, n_cols=4):
        if isinstance(z_windows, torch.Tensor):
            z_windows = z_windows.detach().cpu().numpy()

        n_pages = min(z_windows.shape[0], max_windows)
        n_cols  = min(n_cols, n_pages)
        n_rows  = math.ceil(n_pages / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        axes      = axes.flatten()

        for i in range(n_pages):
            data = transform_fn(i)
            plot_fn(axes[i], data, i)

        for j in range(n_pages, len(axes)):
            axes[j].axis("off")
        return fig, axes

    def plot_persistence_diagrams(self, diagrams_list: list[np.ndarray], title="Persistence Diagrams"):
        """Plot persistence diagrams using the persim command"""
        plot_diagrams(diagrams_list, show=True, title=title)

    def plot_persistence_image(self, pimg):
        """Visualize persistence image."""
        plt.imshow(pimg, origin='lower')
        plt.colorbar()
        plt.title("Persistence Image")
        plt.show()

    def plot_betti_curves(self, betti_curves: np.ndarray):
        """Plot Betti curves from giotto output."""
        curves = betti_curves[0]  # remove batch dim

        for dim in range(curves.shape[0]):
            plt.plot(curves[dim], label=f"H{dim}")

        plt.legend()
        plt.xlabel("Filtration step")
        plt.ylabel("Betti number")
        plt.title("Betti Curves")
        plt.show()

    def plot_entropy(self, entropy):
        """Bar plot of entropy per dimension."""
        plt.bar(range(len(entropy)), entropy)
        plt.xlabel("Homology dimension")
        plt.ylabel("Entropy")
        plt.title("Persistent Entropy")
        plt.show()

    def plot_persistence_grid(self, z_windows, max_windows=8, n_cols=4):
        dgms_all = self._cache_diagrams(z_windows)

        def _transform(i):
            return dgms_all[i]

        def plot(ax, diagrams, i):
            for d, dgms in enumerate(diagrams):
                ax.scatter(dgms[:, 0], dgms[:, 1], s=10, label=f"H{d}")

            ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
            ax.set_title(f"window {i}")
            ax.set_xlabel("")
            ax.set_ylabel("")

        fig, axes       = self._window_grid(z_windows, _transform, plot, max_windows, n_cols)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', frameon=False)
        fig.suptitle("Persistence diagrams across windows", fontsize=16)
        plt.show()

    def plot_persistence_image_grid(self, z_windows, max_windows=8, n_cols=4):
        dgms_all = self._cache_diagrams(z_windows)

        # flatten all diagrams for fitting (skip empty safely)
        all_dgms = [d for dgms in dgms_all for d in dgms if len(d) > 0]

        if len(all_dgms) == 0:
            print("No valid diagrams")
            return

        self._pimgr.fit(all_dgms)

        def _transform(i):
            dgms = dgms_all[i]
            dgms = [d for d in dgms if len(d) > 0]

            if len(dgms) == 0:
                return np.zeros((10, 10))

            return self._pimgr.transform(dgms)[0]

        def plot(ax, img, i):
            ax.imshow(img, origin='lower', cmap='viridis')
            ax.set_title(f"window {i}")
            ax.set_xticks([])
            ax.set_yticks([])

        fig, axes = self._window_grid(z_windows, _transform, plot, max_windows, n_cols)
        fig.colorbar(axes[0].images[0], ax=axes[:max_windows], shrink=0.7)
        fig.suptitle("Persistence images across windows", fontsize=16)
        plt.show()

    def plot_betti_grid(self, z_windows, max_windows=8, n_cols=4):
        dgms_all = self._cache_diagrams(z_windows)

        def _transform(i):
            return self.compute_betti_curves(dgms_all[i])[0]

        def plot(ax, curves, i):
            for d in range(curves.shape[0]):
                ax.plot(curves[d], label=f"H{d}")

            ax.set_title(f"window {i}")
            ax.set_xticks([])
            ax.set_yticks([])

        fig, axes = self._window_grid(z_windows, _transform, plot, max_windows, n_cols)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', frameon=False)
        fig.suptitle("Betti curves across windows", fontsize=16)
        plt.show()

    def plot_landscape(self, landscape: np.ndarray):
        """Plot persistence landscape (approx only)"""
        for i in range(landscape.shape[0]):
            plt.plot(landscape[i], label=f"λ_{i}")
        plt.legend()
        plt.title("Persistence Landscape")
        plt.show()

    def plot_landscape_grid(self, z_windows, max_windows=8, n_cols=4, num_steps=100, K_layers=3):
        """Plot landscapes across windows. Uses the approximate entry for ML stuff
        num_steps = #steps in the landscape grid (x-axis resolution), larger = smoother and slower
        K_layers = #layers in landscape (user-set), larger = richer representation but higher dim, smaller = more compressed but faster. We pad
        with zeros if there are less than K layers, and we cut if there are more than K layers, to get a fixed-size output for ML.
        n_layers = computed #layers in the landscape"""
        dgms_all = self._cache_diagrams(z_windows)

        def _transform(i):
            """n_layers = computed #layers in the landscape. We pad with zeros if there are less than k layers, and we cut if there are more than k layers.
            This way we get a fixed-size output for ML"""
            vals     = self.compute_persistence_landscape(dgms_all[i], approx=True, num_steps=num_steps, flatten=False)
            n_layers = vals.shape[0]

            if n_layers == 0:
                vals = np.zeros((K_layers, num_steps))
            elif n_layers < K_layers:
                vals = np.vstack([vals, np.zeros((K_layers-n_layers, num_steps))])
            else:
                vals = vals[:K_layers]
            return vals

        def plot(ax, land, i):
            """plot mean landscape per plot, not individual lines"""
            mean_land = land.mean(axis=0)
            ax.plot(mean_land)
            ax.set_title(f"window {i}")
            ax.set_xticks([])
            ax.set_yticks([])

        fig, axes = self._window_grid(z_windows, _transform, plot, max_windows, n_cols)
        fig.suptitle("Persistence Landscapes across windows", fontsize=16)
        plt.show()

    def compute_persistence_features(self, z_windows, mode: str = "image") -> np.ndarray:
        """Return ML-ready features (N, D). This function runs the whole pipeline: diagrams + feature extraction, for many windows.
        mode options: image, betti, diagram, landscape"""

        dgms_all = self._cache_diagrams(z_windows)
        dgms_all = [
            d if len(d) > 0 else [np.zeros((1, 2), dtype=np.float32)]
            for d in dgms_all]

        if mode == "image":
            imgs = self._compute_persistence_images_global(dgms_all)
            return imgs.reshape(imgs.shape[0], -1)

        if mode == "betti":
            curves_list = [self.compute_betti_curves(d)[0] for d in dgms_all]

            max_dims = max(c.shape[0] for c in curves_list)
            max_len  = max(c.shape[1] for c in curves_list)

            curves = np.zeros((len(curves_list), max_dims, max_len), dtype=np.float32)

            for i, c in enumerate(curves_list):
                d, t = c.shape
                curves[i, :d, :t] = c
            return curves.reshape(curves.shape[0], -1)

        if mode == "diagram":
            tensors = [self.convert_persistence_diagrams_to_tensor(d) for d in dgms_all]
            X       = np.stack(tensors)
            return X.reshape(X.shape[0], -1)

        if mode == "landscape":
            return self._compute_persistence_landscapes_global(dgms_all)

        raise ValueError(mode)


class TakensEmbedding:
    """Class for time-delay embedding of time series data, related to Takens' embedding theorem."""
    def __init__(self, delay: int = 1, embedding_dim: int = 3):
        """delay τ: time delay (number of steps), >=1. Rule of thumb; delay =~ period/10
        embedding_dim: embedding dimension (usually 2D or 3D), the higher the richer topology structure"""
        self.delay         = delay
        self.embedding_dim = embedding_dim

    def _embed_1d(self, x_1d: np.ndarray) -> np.ndarray:
        """Embed 1D time series → (n_points, embedding_dim)."""
        n = len(x_1d) - (self.embedding_dim - 1) * self.delay
        if n <= 0:
            raise ValueError(f"Time series too short: len={len(x_1d)}, delay={self.delay}, dim={self.embedding_dim}")
        return np.stack([x_1d[i:i+n] for i in range(0, self.embedding_dim * self.delay, self.delay)], axis=1)

    def make_timedelay_embeddings(self, x: np.ndarray) -> np.ndarray:
        """Time-delay embedding: returns shape (n_points, dim).
        x: 1D/2D array of time series values
        1D -> (n_points, embedding_dim)
        2D -> (n_points, embedding_dim * n_features) via per-channel embedding + concat"""

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if x.ndim == 1:
            return self._embed_1d(x)
        elif x.ndim == 2:
            pcs = [self._embed_1d(x[:, i]) for i in range(x.shape[1])]
            return np.concatenate(pcs, axis=1)  # (n, dim * D)
        else:
            raise ValueError("x must be 1D or 2D")

    def make_timedelay_embeddings_grid(self, z_windows: np.ndarray | torch.Tensor) -> list[np.ndarray]:
        """Apply time-delay embedding to each window in a grid.
        Output is converted to np because persistence libraries dont like torch
        z_windows: 3D array of shape (n_windows, n_timesteps, n_features)"""

        if isinstance(z_windows, torch.Tensor):
            z_windows = z_windows.detach().cpu().numpy()

        point_cloud_2d = [self.make_timedelay_embeddings(z_windows[i])
                          for i in range(z_windows.shape[0])]
        return np.stack(point_cloud_2d, axis=0)


class WassersteinDistance:
    """Class that computes wasserstein distance between 2 consecutive windows"""

    @staticmethod
    def compute_wasserstein_distance(array1, array2, delay: int) -> np.ndarray:
        """computes Wasserstein distance between consecutive windows for desired persistence feature type
        output: 1D Wasser. distance vector of size (num_windows - delay)

        Case 1: only 1 array input (array1) → temporal Wasserstein across windows
        Case 2: 2 arrays input (array1, array2) → Wasserstein between two sequences (aligned)"""

        def safe_wass(x, y):
            if len(x) == 0 or len(y) == 0:
                return 0.0
            return wasserstein_distance(x, y)

        # CASE 1: single input
        if array2 is None:
            return np.array([safe_wass(array1[i - delay], array1[i])
                             for i in range(delay, array1.shape[0])])

        # CASE 2: compare two sequences
        assert array1.shape[0] == array2.shape[0], "Sequences must be aligned"

        return np.array([safe_wass(array1[i], array2[i])
                         for i in range(array1.shape[0])])

    @staticmethod
    def plot_all_wasserstein_distances(betti_dist_train, image_dist_train, diagram_dist_train, landscape_dist_train):
        """plots all wasserstein distances for each of the input arrays. Each plot is a single line, since the
        wasserstein distance function outputs a 1D vector for each representation type."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0, 0].plot(betti_dist_train, color='blue')
        axes[0, 0].set_title("Betti")

        axes[0, 1].plot(image_dist_train, color='green')
        axes[0, 1].set_title("Image")

        axes[1, 0].plot(diagram_dist_train, color='red')
        axes[1, 0].set_title("Diagram")

        axes[1, 1].plot(landscape_dist_train, color='orange')
        axes[1, 1].set_title("Landscape")

        fig.suptitle("Wasserstein Distances Across Representations", fontsize=16)
        fig.supxlabel("Window")
        fig.supylabel("d_wasser")

        plt.subplots_adjust(
            left=0.07,
            right=0.98,
            bottom=0.08,
            top=0.88,
            wspace=0.15,
            hspace=0.25)
        plt.show()


class GeometryConverter:
    """Class for angle conversions, ie angles-3D coords, for torus and sphere."""
    def __init__(self, R_major, r_tube):
        self.R_major = R_major
        self.r_tube  = r_tube

    def convert_angles_to_torus_xyz(self, u, v):
        """Convert rad angles to torus coords"""
        x = np.cos(u) * (self.R_major + self.r_tube * np.cos(v))
        y = np.sin(u) * (self.R_major + self.r_tube * np.cos(v))
        z = self.r_tube * np.sin(v)
        return x, y, z

    def convert_angles_to_2torus_xyz(self, u, v, d_shift=0.8):
        """Approx genus-2 surface (connected double torus)
        Toruses both start at 0, and 
        d_shift =  distance between 2 toruses (both start from 0); smaller = more connected, larger = more separate"""

        # torus 1
        x1 = (self.R_major + self.r_tube*np.cos(v)) * np.cos(u)
        y1 = (self.R_major + self.r_tube*np.cos(v)) * np.sin(u) - d_shift
        z1 = self.r_tube * np.sin(v)

        # torus 2 (shifted, slightly rotated)
        x2 = (self.R_major + self.r_tube*np.cos(v)) * np.cos(u)
        y2 = (self.R_major + self.r_tube*np.cos(v)) * np.sin(u) + d_shift
        z2 = self.r_tube * np.sin(v)
        return x1, y1, z1, x2, y2, z2

    def convert_angles_to_sphere_xyz(self, u, v):
        """Convert rad angles to sphere coords
        u = azimuth, v = polar angle"""
        x = self.R_major * np.cos(u) * np.sin(v)
        y = self.R_major * np.sin(u) * np.sin(v)
        z = self.R_major * np.cos(v)
        return x, y, z

    def convert_xyz_to_angles(self, x, y, z):
        """Convert coords to rad angles, but doesnt project them onto torus/sphere;
        this needs to be converted to coords again using the torus/sphere formulae"""
        u = np.arctan2(z, np.sqrt(x**2 + y**2))
        v = np.arctan2(y, x)
        return u, v


def plot_3d_points(*clouds, colors=None, figsize=(8, 12), size=3, alpha=0.7):
    """Plot one or multiple 3D point clouds with equal axis scaling. If input is torch, converts to np
    clouds: tuples of (x, y, z)
    colors: list of colors (optional)"""
    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(projection='3d')

    if colors is None:
        colors = ['red'] * len(clouds)

    clouds_np = [tuple(torch_to_numpy(c_i) for c_i in c) for c in clouds]
    all_x = np.concatenate([c[0] for c in clouds_np])
    all_y = np.concatenate([c[1] for c in clouds_np])
    all_z = np.concatenate([c[2] for c in clouds_np])

    max_range = np.array([
        all_x.max() - all_x.min(),
        all_y.max() - all_y.min(),
        all_z.max() - all_z.min()]).max() / 2.0

    mid_x = (all_x.max() + all_x.min()) * 0.5
    mid_y = (all_y.max() + all_y.min()) * 0.5
    mid_z = (all_z.max() + all_z.min()) * 0.5

    for (x, y, z), c in zip(clouds_np, colors):
        ax.scatter(x, y, z, color=c, alpha=alpha, s=size)

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.margins(0)
    plt.show()

def plot_all_features_in_2d_array(dataset, window_number=0):
    """plots everything in 2d array. If array is 3d, plots features of a given input window"""
    plt.figure(figsize=(12, 7))

    # 3d array
    if dataset.ndim == 3 and dataset.shape[0] > 1:
        print(f"Plotting features for window {window_number} (dataset shape: {dataset.shape})")
        title = f"Features for window {window_number}"

        shape_idx_of_interest = 2 # 0 = windows, 1 = timesteps, 2 (or -1) = features
        for i in range(dataset.shape[shape_idx_of_interest]):
            plt.plot(dataset[window_number, :, i].cpu(), label=f'z{i}')

    # 2d array
    elif dataset.ndim == 2 or (dataset.ndim == 3 and dataset.shape[0] == 1):
        title = f"Features for entire array"

        shape_idx_of_interest = 1
        for i in range(dataset.shape[shape_idx_of_interest]):
            plt.plot(dataset[:, i].cpu(), label=f'z{i}')

    plt.title(title)
    plt.legend(fontsize=6)
    plt.show()

def plot_latent_evolution_grid(z: np.ndarray | torch.Tensor, max_latent_dims: int = 8, n_plots_per_row: int = 4):
    """Given 3D array, plot Heatmaps for latent tensors (W, C, L) as per-dimension heatmaps in a grid.
    Args:
        z: (W, C, L) array or tensor.
        max_latent_dims: max L to plot.
        n_plots_per_row: grid width.
    Note: Each subplot is independently scaled."""

    if isinstance(z, torch.Tensor):
        z = z.detach().cpu().numpy()

    L     = min(z.shape[2], max_latent_dims)
    nrows = int(np.ceil(L / n_plots_per_row))
    
    fig, axes = plt.subplots(
        nrows,
        n_plots_per_row,
        figsize=(4 * n_plots_per_row + 1, 3.6 * nrows),
        constrained_layout=True)
    axes = axes.flatten()

    im = None
    for i in range(L):
        im = axes[i].imshow(z[:, :, i], aspect='auto', cmap='hot')
        axes[i].set_title(f"z[{i}]")
        axes[i].yaxis.set_major_locator(MaxNLocator(integer=True))

        # remove per-plot axis labels
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")

    for j in range(L, len(axes)):
        axes[j].axis('off')

    fig.subplots_adjust(right=0.88) # right-side colorbar

    cbar = fig.colorbar(
        im,
        ax=axes[:L],
        orientation='vertical',
        fraction=0.03,
        pad=0.02,
        shrink=0.9)
    cbar.set_label("activation")

    # global axis labels
    fig.text(0.5, -0.02, "chunk t", ha='center', va='bottom')
    fig.text(0.0, 0.5, "window #", va='center', rotation='vertical')

    # note
    fig.text(
        0.99,
        -0.02,
        "Colors are normalized per latent feature plot; not comparable across subplots.",
        ha='right',
        va='bottom',
        fontsize=8,
        alpha=0.7)

    fig.suptitle("z across windows + chunks", fontsize=16)
    plt.show()

def numpy_to_torch(x: np.ndarray) -> torch.Tensor:
    """converts np array to torch tensor (works for both CPU and GPU tensors)"""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    return x

def torch_to_numpy(x: torch.Tensor) -> np.ndarray:
    """converts torch tensor to np array (works for both CPU and GPU tensors)"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def window_2d_sequence_to_3d(sequence, window_size: int, stride: int = 1):
    """windows a 2d sequence to a 3d array of shape (num_windows, window_size, num_features).
    stride = window_size means no overlap, stride = 1 means maximum overlap"""
    if type(sequence) == np.ndarray:
        return np.array([sequence[i:i+window_size] for i in range(0, len(sequence) - window_size + 1, stride)])
    elif type(sequence) == torch.Tensor:
        return torch.stack([sequence[i:i+window_size] for i in range(0, len(sequence) - window_size + 1, stride)]) 

def write_results_to_file(file: str, line: str):
    """appends line to file + adds time"""
    with open(file, "a") as f:
        current_time = datetime.now().strftime("%H:%M")
        f.write(current_time + " " + line)

def read_yaml_params(file_path: str) -> dict:
    """Read parameters from a YAML file."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

