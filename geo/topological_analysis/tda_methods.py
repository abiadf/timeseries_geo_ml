"""Module for topological data analysis (TDA) methods."""

import numpy as np
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
        self.dataset       = dataset
        self.max_dim       = max_dim
        self.diagrams_list = None
        self.entropy_array = None
        self.persistence_images_list = None
        self.betti_curves_array      = None

    def plot_persistence_diagrams(self, dataset: np.ndarray, to_plot: bool = True):
        """Make persistent homology plot for a dataset using the Vietoris-Rips filtration
        If no dataset is provided, it will use random data
        max_dim = max homology dimension to compute (0 = connected components, 1 = loops...)"""
        if dataset is None:
            print("No dataset provided, using random data")
            num_points, dim_points = 800, 3
            dataset = np.random.randn(num_points, dim_points)
        diagrams_list = ripser(dataset, maxdim=self.max_dim)['dgms']
        if to_plot:
            plot_diagrams(diagrams_list, show=True)
        return diagrams_list

    def compute_entropy(self, dgms, feature_dim=None):
        """Return entropy per homology dimension or specific one."""
        if feature_dim is None:
            return persistent_entropy(dgms)
        return persistent_entropy([dgms[feature_dim]])

    def plot_entropy(self, entropy):
        """Bar plot of entropy per dimension."""
        plt.bar(range(len(entropy)), entropy)
        plt.xlabel("Homology dimension")
        plt.ylabel("Entropy")
        plt.title("Persistent Entropy")
        plt.show()

    def compute_persistence_image(self, dgms):
        """Convert diagrams to persistence images."""
        pimgr = PersistenceImager()
        return pimgr.fit_transform(dgms)

    def plot_persistence_image(self, pimg):
        """Visualize persistence image."""
        plt.imshow(pimg, origin='lower')
        plt.colorbar()
        plt.title("Persistence Image")
        plt.show()

    def compute_betti_curves(self, dgms):
        """Return Betti curves."""
        bc = BettiCurve()
        X  = self.ripser_to_gtda(dgms)
        return bc.fit_transform([X])

    def plot_betti_curves(self, betti_curves):
        """Plot Betti curves from giotto output."""
        curves = betti_curves[0]  # remove batch dim

        for dim in range(curves.shape[0]):
            plt.plot(curves[dim], label=f"H{dim}")

        plt.legend()
        plt.xlabel("Filtration step")
        plt.ylabel("Betti number")
        plt.title("Betti Curves")
        plt.show()

    def ripser_to_gtda(self, dgms):
        """Convert ripser diagrams to giotto format."""
        out = []
        for dim, dgm in enumerate(dgms):
            if len(dgm) == 0:
                continue
            labels = np.full((dgm.shape[0], 1), dim)
            out.append(np.hstack([dgm, labels]))
        return np.vstack(out)

    def remove_inf(self, dgms):
        """Remove points with infinite death times."""
        cleaned = []
        for dgm in dgms:
            mask = np.isfinite(dgm[:, 1])
            cleaned.append(dgm[mask])
        return cleaned


