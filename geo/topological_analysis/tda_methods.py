"""Module for topological data analysis (TDA) methods."""

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ripser import ripser
from persim import PersistenceImager, plot_diagrams
from persim.persistent_entropy import persistent_entropy
from gtda.diagrams import BettiCurve

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


class PersistenceAnalysis:
    """Class for computing + plotting persistence tools: persistence diagrams, entropy, images, and Betti curves"""

    def __init__(self, dataset: np.ndarray = None, max_dim: int = 2):
        """max_dim = max homology dim to compute (0 = connected components, 1 = loops...)"""
        self.dataset = dataset
        self.max_dim = max_dim
        self._pimgr  = PersistenceImager()
        self._bc     = BettiCurve()

    def compute_persistence_diagrams(self, X: np.ndarray | torch.Tensor) -> list[np.ndarray]:
        """Returns list of persistence diagrams (one per homology dimension). Ripser expects np array, not tensor"""
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        return ripser(X, maxdim=self.max_dim)['dgms']

    def plot_persistence_diagrams(self, diagrams_list: list[np.ndarray], title="Persistence Diagrams"):
        """Plot persistence diagrams using the persim command"""
        plot_diagrams(diagrams_list, show=True, title=title)

    def compute_entropy(self, diagrams_list: list[np.ndarray], feature_dim=None) -> np.ndarray:
        """Return entropy per homology dimension or specific one."""
        if feature_dim is None:
            return persistent_entropy(diagrams_list)
        return persistent_entropy([diagrams_list[feature_dim]])

    def plot_entropy(self, entropy):
        """Bar plot of entropy per dimension."""
        plt.bar(range(len(entropy)), entropy)
        plt.xlabel("Homology dimension")
        plt.ylabel("Entropy")
        plt.title("Persistent Entropy")
        plt.show()

    def compute_persistence_image(self, diagrams_list: list[np.ndarray]) -> np.ndarray:
        """Convert diagrams to persistence images."""
        self._pimgr.fit(diagrams_list)
        return self._pimgr.transform(diagrams_list)

    def plot_persistence_image(self, pimg):
        """Visualize persistence image."""
        plt.imshow(pimg, origin='lower')
        plt.colorbar()
        plt.title("Persistence Image")
        plt.show()

    def compute_betti_curves(self, diagrams_list: list[np.ndarray]) -> np.ndarray:
        """Return Betti curves."""
        X = self.ripser_to_gtda(diagrams_list)
        return self._bc.fit_transform([X])

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

    def ripser_to_gtda(self, diagrams_list: list[np.ndarray]) -> np.ndarray:
        """Convert ripser diagrams to giotto format."""
        out = []
        for dim, dgm in enumerate(diagrams_list):
            if len(dgm) == 0:
                continue
            labels = np.full((dgm.shape[0], 1), dim)
            out.append(np.hstack([dgm, labels]))
        return np.vstack(out)

    def diagrams_to_tensor(self, diagrams_list: list[np.ndarray]) -> np.ndarray:
        """Convert list of diagrams -> padded tensor (N, P, 2)."""
        max_len = max(len(dgm) for dgm in diagrams_list)
        out     = np.zeros((len(diagrams_list), max_len, 2), dtype=np.float32)

        for i, dgm in enumerate(diagrams_list):
            n = len(dgm)
            out[i, :n] = dgm
        return out

    def remove_inf(self, dgms: list[np.ndarray]) -> list[np.ndarray]:
        """Remove points with infinite death times from each diagram."""
        return [dgm[np.isfinite(dgm[:, 1])] for dgm in dgms]


def make_timedelay_embeddings(y: np.ndarray, tau: int, dim: int) -> np.ndarray:
    """Time-delay embedding: returns shape (n_points, dim). Related to Takens' embedding theorem
    y: 1D array of time series values
    tau: time delay (number of steps), >=1
    dim: embedding dimension (usually 2D)"""
    n = len(y) - (dim - 1) * tau
    return np.stack([y[i:i+n] for i in range(0, dim * tau, tau)], axis=1)

def plot_3d_points(*clouds, colors=None, figsize=(8, 12), size=3, alpha=0.7):
    """Plot one or multiple 3D point clouds with equal axis scaling.
    clouds: tuples of (x, y, z)
    colors: list of colors (optional)"""
    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(projection='3d')

    if colors is None:
        colors = ['red'] * len(clouds)

    all_x = np.concatenate([c[0] for c in clouds])
    all_y = np.concatenate([c[1] for c in clouds])
    all_z = np.concatenate([c[2] for c in clouds])

    max_range = np.array([
        all_x.max() - all_x.min(),
        all_y.max() - all_y.min(),
        all_z.max() - all_z.min()]).max() / 2.0

    mid_x = (all_x.max() + all_x.min()) * 0.5
    mid_y = (all_y.max() + all_y.min()) * 0.5
    mid_z = (all_z.max() + all_z.min()) * 0.5

    for (x, y, z), c in zip(clouds, colors):
        ax.scatter(x, y, z, color=c, alpha=alpha, s=size)

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.margins(0)
    plt.show()
