"""Module concerned with feature selection / dimensionality reduction"""

from collections import defaultdict
from typing import Tuple, List

import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from catboost import CatBoostRegressor
from predictions import SingleOutputModelPredictor

class PCA_analysis:
    """Class that contains all PCA methods. Given a df of F cols, the PCA df will also have F cols.
    This class looks at X only, not y"""

    @staticmethod
    def fit_pca(X: pd.DataFrame, var_threshold: float = 0.95):
        """Fit PCA retaining enough components to cover var_threshold variance."""
        pca_model = PCA(n_components=var_threshold)
        pca_model.fit(X)
        return pca_model

    @staticmethod
    def explain_pca_variance(pca, show_plot=False) -> Tuple[np.ndarray, int]:
        """Plot cumulative explained variance and return components and count."""
        explained_var = np.cumsum(pca.explained_variance_ratio_)
        N_pca_components = len(explained_var)  # all components in fitted pca

        if show_plot:
            plt.plot(explained_var)
            plt.axhline(y=explained_var[-1], color='r', linestyle='--')
            plt.xlabel("# of PCA Components")
            plt.ylabel("Fraction of Total Variance Explained")
            plt.title("Cumulative Explained Variance from PCA")
            plt.show()

        print(f"# PCA components covering {explained_var[-1]*100:.1f}% variance: {N_pca_components}")
        return pca.components_, N_pca_components

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



class RFE_analysis():
    """RFE with Catboost"""
    def __init__(self, device):
        self.device     = torch.device(device) if isinstance(device, str) else device
        self.device_str = 'GPU' if self.device.type == 'cuda' else 'CPU'

    def apply_recursive_feature_elimination(self, X_train_scaled: pd.DataFrame, X_val_scaled: pd.DataFrame,
                                            y_train: pd.Series, y_val: pd.Series, single_predictor: SingleOutputModelPredictor,
                                            fraction_cols_to_keep: float = 0.95) -> Tuple[float, RFE]:
        """Apply Recursive Feature Elimination (RFE) using CatBoostRegressor on numeric features
        of training data. Returns RMSE on validation set + the fitted RFE model.
        - fraction_cols_to_keep (float): Fraction of numeric features to retain. Larger = less computations = faster"""

        numeric_feature_names = X_train_scaled.select_dtypes(include=np.number).columns

        X_train_num = X_train_scaled[numeric_feature_names]
        X_val_num   = X_val_scaled[numeric_feature_names]

        base_model  = CatBoostRegressor(verbose=0, random_state=42, early_stopping_rounds=10)
        rfe_model   = RFE(base_model, n_features_to_select=int(X_train_num.shape[1] * fraction_cols_to_keep))
        X_train_rfe = rfe_model.fit_transform(X_train_num, y_train.values.ravel())
        X_val_rfe   = rfe_model.transform(X_val_num)

        rmse_rfe, _, _ = single_predictor.predict_catboost_single_model(X_train_rfe, y_train, X_val_rfe,
                                                                        y_val, cat_features=None)
        print(f"CatBoost RMSE (RFE): {rmse_rfe:.3f} nm")
        return rmse_rfe, rfe_model

    def get_sorted_features_by_importance(self, rfe_model: RFE, X_train_num: pd.DataFrame):# -> List[str]:
        """Returns RFE-selected features sorted by importance (descending)"""

        selected_features = X_train_num.columns[rfe_model.get_support()]
        final_model: CatBoostRegressor = rfe_model.estimator_
        importances     = final_model.feature_importances_
        sorted_indices  = np.argsort(importances)[::-1]
        sorted_features = selected_features[sorted_indices]
        return sorted_features#.tolist()
