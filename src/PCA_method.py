from collections import defaultdict
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class PCA_analysis:
    """Class that contains all PCA methods. Given a df of F cols, the PCA df will also have F cols.
    This class looks at X only, not y"""

    @staticmethod
    def fit_pca(X: pd.DataFrame):
        """Fit PCA model on data. It also sorts PCA components in descending order of eigenvalue magnitude (= explained variance)"""
        pca_model = PCA()
        pca_model.fit(X)
        return pca_model

    @staticmethod
    def explain_pca_variance(pca, var_threshold: float = 0.95, show_plot = False) -> Tuple[np.ndarray, int]:
        """Return top PCA components covering desired variance and plot explained variance"""
        explained_var    = pca.explained_variance_ratio_.cumsum()
        N_pca_components = np.argmax(explained_var >= var_threshold) + 1

        if show_plot:
            plt.plot(explained_var)
            plt.axhline(y=var_threshold, color='r', linestyle='--')
            plt.xlabel("# of PCA Components")
            plt.ylabel("Fraction of Total Variance in X Explained")
            plt.title("Cumulative Explained Variance from PCA on X")
            plt.show()

        print(f"# of PCA components covering {var_threshold*100:.1f}% variance: {N_pca_components}")
        top_N_pca_components = pca.components_[:N_pca_components]
        return top_N_pca_components, N_pca_components

    @staticmethod
    def print_top_features_per_component(X_scaled, top_components, top_k_features: int = 3) -> None:
        """Print top contributing features for each PCA component"""
        feature_names = X_scaled.columns
        for i, comp in enumerate(top_components):
            abs_loadings    = np.abs(comp)
            percent_contrib = abs_loadings / abs_loadings.sum() * 100
            top_indices     = np.argsort(percent_contrib)[::-1]
            print(f'Component {i+1}:')
            for idx in top_indices[:top_k_features]:
                print(f'  {feature_names[idx]}: {percent_contrib[idx]:.2f}%')

    @staticmethod
    def summarize_feature_importance(X_scaled, top_components, top_k_features: int = 10):
        """Aggregate feature contributions across top components"""
        feature_names      = X_scaled.columns
        feature_importance = defaultdict(float)

        for comp in top_components:
            abs_loadings    = np.abs(comp)
            percent_contrib = abs_loadings / abs_loadings.sum() * 100
            for i, contrib in enumerate(percent_contrib):
                feature_importance[feature_names[i]] += contrib

        sorted_features = sorted(feature_importance.items(), key=lambda x: -x[1])
        print(f"\nTop {top_k_features} features across all components:")
        for feature, total_contrib in sorted_features[:top_k_features]:
            print(f"{feature}: {total_contrib:.2f}%")
        return sorted_features
