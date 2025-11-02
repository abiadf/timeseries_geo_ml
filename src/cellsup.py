import __main__
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, root_mean_squared_error

from other_encoders.latents import Latents

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Cellsup:
    """Ensemble clustering across multiple encoders."""
    def __init__(self, encoders_dict: dict, n_clusters: int = 5, device: str = "cpu",
                 cluster_assignment: str = "soft", cluster_metric: str = "ch", random_state: int = 42):
        self.encoders_dict      = encoders_dict
        self.n_clusters         = n_clusters
        self.device             = device
        self.cluster_assignment = cluster_assignment
        self.cluster_prob_matrix= None
        self.cluster_metric     = cluster_metric.lower()
        self.rng                = np.random.RandomState(random_state)
        self.clusterers         = {}  # stores KMeans per encoder

    def score_clusters(self, z: np.ndarray, labels: np.ndarray) -> float:
        """Return clustering quality score depending on metric."""
        try:
            if self.cluster_metric == "silhouette":
                return silhouette_score(z, labels)
            elif self.cluster_metric == "ch":
                return calinski_harabasz_score(z, labels)
            elif self.cluster_metric == "db":
                return -davies_bouldin_score(z, labels)  # negate so higher = better
            else:
                raise ValueError(f"Unknown metric {self.cluster_metric}")
        except ValueError:
            return -1  # invalid clustering (e.g. only 1 cluster found)

    def _fit_best_kmeans_for_latent(self, z: np.ndarray, cluster_range: tuple[int,int], n_restarts: int = 5) -> KMeans:
        """Fit KMeans on latent `z` using separate best-k selection per encoder.
        1) Tune best cluster number k (average score over n_restarts)
        2) Fit KMeans with n_restarts for that k to select the best initialization
        Args:
            z (np.ndarray): Latent representations, shape (N, D)
            cluster_range (tuple[int,int]): Min and max+1 clusters to try
            n_restarts (int): KMeans restarts per candidate cluster number and for final best k
        Returns:
            KMeans: Fitted KMeans model with optimal cluster number"""
        # Step 1: pick best k using mean score across n_restarts
        best_score, best_k = -np.inf, None
        for k in range(cluster_range[0], cluster_range[1]):
            scores = []
            for _ in range(n_restarts):
                kmeans_try = KMeans(n_clusters=k, random_state=self.rng.randint(0,10000)).fit(z)
                scores.append(self.score_clusters(z, kmeans_try.labels_))
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score, best_k = mean_score, k

        # Step 2: fit KMeans with multiple restarts for that best k
        best_score_restart, best_kmeans = -np.inf, None
        for _ in range(n_restarts):
            kmeans = KMeans(n_clusters=best_k, random_state=self.rng.randint(0,10000)).fit(z)
            score  = self.score_clusters(z, kmeans.labels_)
            if score > best_score_restart:
                best_score_restart, best_kmeans = score, kmeans

        print(f"Selected cluster size: {best_kmeans.n_clusters}")
        return best_kmeans

    def fit_kmeans_on_encoder_latents(self, X: np.ndarray, encoder_weights: dict = None,
                                      cluster_range=(4,16)):
        """Fit KMeans on each encoder's latent space, select the best cluster number per encoder,
        and concatenate per-encoder cluster features into a single matrix.
        Cluster features:
        - 'soft': standardized softmax of negative distances (approx. probabilities)
        - 'hard': one-hot cluster labels
        Optional encoder weights scale each encoder's features.
        Args:
            X (np.ndarray): Input data, shape (N, D) or (N, T, D)
            encoder_weights (dict, optional): Encoder name → scalar weight (normalized)
            cluster_range (tuple, optional): Range of clusters to try (min, max+1)
        Returns:
            self: Adds `self.clusterers` (dict of fitted KMeans) and 
                `self.prob_matrix` (concatenated cluster features, shape (N, sum_k))"""
        cluster_features_list = []

        if encoder_weights is not None: # normalize encoder weights if provided
            total = sum(encoder_weights.values())
            encoder_weights = {k: v / total for k, v in encoder_weights.items()}

        for name, encoder in self.encoders_dict.items():
            z = Latents.get_latent_from_encoder(encoder, X, device=self.device)
            kmeans               = self._fit_best_kmeans_for_latent(z, cluster_range, n_restarts=5)
            self.clusterers[name]= kmeans
            labels               = kmeans.labels_

            # --- compute soft/hard assignments ---
            if self.cluster_assignment == "soft":
                distances        = kmeans.transform(z)
                cluster_features = softmax(-distances, axis=1)
                # standardize to prevent dominance by encoders with more clusters
                cluster_features = (cluster_features - cluster_features.mean(axis=0)) / (cluster_features.std(axis=0) + 1e-8)
            else: # hard
                cluster_features = np.eye(kmeans.n_clusters)[labels]

            if encoder_weights is not None: # apply encoder weight
                cluster_features *= encoder_weights.get(name, 1.0)
            cluster_features_list.append(cluster_features)

        # concatenate horizontally across encoders
        self.cluster_prob_matrix = np.concatenate(cluster_features_list, axis=1)  # (N, sum_k)
        return self

    def apply_kmeans_to_new_data(self, X: np.ndarray) -> np.ndarray:
        """Apply EXISTING fitted KMeans models to new data X and return concat cluster features
        Cluster features:
        - 'soft': softmax of negative distances
        - 'hard': one-hot labels
        Args: X (np.ndarray): Input data, shape (N, D) or (N, T, D)
        Returns: np.ndarray: Concatenated cluster features, shape (N, sum_k)"""
        cluster_features_list = []
        for name, encoder in self.encoders_dict.items():
            z      = Latents.get_latent_from_encoder(encoder, X, device=self.device)
            kmeans = self.clusterers[name]
            labels = kmeans.predict(z)
            if self.cluster_assignment == "soft":
                distances        = kmeans.transform(z) # (n_samples, n_clusters)
                cluster_features = softmax(-distances, axis=1) # convert distance → probability
            elif self.cluster_assignment == "hard":
                cluster_features = np.eye(kmeans.n_clusters)[labels]
            cluster_features_list.append(cluster_features)
        # return np.mean(cluster_features_list, axis=0)
        # fix
        return np.concatenate(cluster_features_list, axis=1)  # (N, E*K)

    def assign_pseudo_labels(self, X_labeled, y_labeled, X_unlabeled, confidence_thresh=0.5):
        """Assign pseudo-labels to unlabeled data using KMeans cluster averages."""
        pseudo_labels_all = []

        if X_unlabeled is None or len(X_unlabeled) == 0:
            return np.zeros((0, y_labeled.shape[1]))

        for name, kmeans in self.clusterers.items():
            z_L      = Latents.get_latent_from_encoder(self.encoders_dict[name], X_labeled, device=device)
            z_U      = Latents.get_latent_from_encoder(self.encoders_dict[name], X_unlabeled, device=device)
            labels_L = kmeans.predict(z_L)
            labels_U = kmeans.predict(z_U)

            # cluster -> mean label map
            cluster_to_mean = {c: y_labeled[labels_L == c].mean(axis=0) for c in np.unique(labels_L)}
            fallback = y_labeled.mean(axis=0)  # use global mean instead of NaNs
            y_pseudo = np.stack([cluster_to_mean.get(c, fallback) for c in labels_U], axis=0)
            # ==============
            # # compute soft assignment probabilities
            # distances  = kmeans.transform(z_U)
            # soft_probs = softmax(-distances, axis=1)
            # max_probs  = soft_probs.max(axis=1)
            # # mask low-confidence pseudo-labels
            # mask = max_probs >= confidence_thresh
            # y_pseudo[~mask] = np.nan  # mark low-confidence samples as NaN
            # ==============
            pseudo_labels_all.append(y_pseudo)

        # combine across encoders
        y_pseudo_final = np.nanmean(np.stack(pseudo_labels_all, axis=0), axis=0)
        return y_pseudo_final

    # to replace the above one
    def assign_ensemble_pseudo_labels(self, X_labeled, y_labeled, X_unlabeled,
                                      cluster_sizes=(4, 6, 8), encoder_weights=None):
        """Assign pseudo-labels by averaging across multiple cluster sizes and encoders.
        Args:
            X_labeled (np.ndarray): labeled features
            y_labeled (np.ndarray): labels
            X_unlabeled (np.ndarray): unlabeled features
            cluster_sizes (tuple[int]): cluster numbers to ensemble
            encoder_weights (dict, optional): encoder name -> scalar weight
        Returns:
            np.ndarray: pseudo-labels for unlabeled data (shape: n_samples, y_dim)"""
        all_pseudo_labels = []

        for name, encoder in self.encoders_dict.items():
            z_L = Latents.get_latent_from_encoder(encoder, X_labeled, device=self.device)
            z_U = Latents.get_latent_from_encoder(encoder, X_unlabeled, device=self.device)

            for k in cluster_sizes:
                kmeans = KMeans(n_clusters=k, random_state=42).fit(z_L)
                labels_L = kmeans.predict(z_L)
                labels_U = kmeans.predict(z_U)

                cluster_to_mean = {c: y_labeled[labels_L == c].mean(axis=0) for c in np.unique(labels_L)}
                fallback = y_labeled.mean(axis=0)
                y_pseudo = np.stack([cluster_to_mean.get(c, fallback) for c in labels_U], axis=0)

                # optional encoder weight
                if encoder_weights is not None:
                    y_pseudo *= encoder_weights.get(name, 1.0)
                all_pseudo_labels.append(y_pseudo)

        # ensemble average across encoders and cluster sizes
        return np.nanmean(np.stack(all_pseudo_labels, axis=0), axis=0)



class DeepClusterAndSwav(Cellsup):
    """SWAV-style DeepCluster wrapper on top of Cellsup base."""
    def __init__(self, encoders_dict: dict, n_clusters: int = 5, device: str = "cpu", cluster_assignment: str = "soft",
                 cluster_metric: str = "ch", random_state: int = 42):
        super().__init__(encoders_dict=encoders_dict, n_clusters=n_clusters, device=device, 
                         cluster_assignment=cluster_assignment, cluster_metric=cluster_metric, random_state=random_state,)

    def target_distribution(self, q: np.ndarray) -> np.ndarray:
        """compute DEC target p from soft assignments q (N,K).
        squares q to amplify confident assignments then re-normalizes."""
        weight = (q ** 2) / q.sum(axis=0)
        return (weight.T / weight.sum(axis=1)).T

    # consider removing
    def refine_clusters_DEC(self, X, encoder, name: str, n_iters: int = 10, lr: float = 1e-4):
        """Refine encoder so that latent z matches clusters better (DEC refinement).
        Assumes encoder is a torch.nn.Module."""
        optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)

        # get cluster centers from fitted KMeans
        # centers = torch.tensor(self.clusterers['AE_8'].cluster_centers_, dtype=torch.float32)
        centers = torch.tensor(self.clusterers[name].cluster_centers_, dtype=torch.float32)

        for _ in range(n_iters):
            z = Latents.get_latent_tensor(encoder, X, train_encoder=True, device=self.device)  # (n, d)

            # soft assignment of z to centers
            q = torch.softmax(-torch.cdist(z, centers.to(z.device)), dim=1)

            # sharpened target distribution
            p = torch.tensor(self.target_distribution(q.detach().cpu().numpy()), device=z.device)

            # KL divergence loss
            loss = F.kl_div(q.log(), p, reduction="batchmean")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # to replace with SWAV
    def deepcluster_step(self, X, n_iters=1, refine_encoder=False, lr=1e-3, epochs_finetune=5):
        """Iterative DeepCluster-style for lightweight modular pseudo-labels."""
        for _ in range(n_iters):  # iterate DeepCluster steps
            for name, encoder in self.encoders_dict.items():
                z = Latents.get_latent_from_encoder(encoder, X, device=self.device)

                # --- fit KMeans ---
                kmeans = self._fit_best_kmeans_for_latent(z, cluster_range=(4,16))
                self.clusterers[name] = kmeans

                # --- soft cluster assignments ---
                distances        = kmeans.transform(z)
                cluster_features = softmax(-distances, axis=1)
                cluster_features = (cluster_features - cluster_features.mean(axis=0)) / (cluster_features.std(axis=0)+1e-8)

                # --- optional lightweight refinement ---
                if refine_encoder:
                    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
                    centers   = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=self.device)
                    for _ in range(epochs_finetune):
                        # compute z fresh every step
                        z_tensor = Latents.get_latent_tensor(encoder, X, train_encoder=True, device=self.device)
                        q        = torch.softmax(-torch.cdist(z_tensor, centers), dim=1)
                        p        = torch.tensor(self.target_distribution(q.detach().cpu().numpy()), device=q.device)
                        loss     = F.kl_div(q.log(), p, reduction="batchmean")
                        optimizer.zero_grad()
                        loss.backward()  # no retain_graph
                        optimizer.step()
                # store/concatenate cluster features across encoders
                if self.cluster_prob_matrix is None:
                    self.cluster_prob_matrix = cluster_features
                else:
                    self.cluster_prob_matrix = np.concatenate([self.cluster_prob_matrix, cluster_features], axis=1)
    
    # remove
    def swav_soft_assign(self, X, cluster_range=(4,16), temperature=0.1):
        """Compute SWAV-style normalized soft cluster assignments for all encoders.
        Each encoder gets a moving-target distribution for pseudo-labels.
        Args:
            X (np.ndarray): input features
            cluster_range (tuple[int]): min/max clusters to try for KMeans
            temperature (float): softmax temperature to sharpen assignments
        Returns:
            dict: encoder name -> soft pseudo-label matrix (n_samples, sum_k)"""
        swav_features = {}
        for name, encoder in self.encoders_dict.items():
            z = Latents.get_latent_from_encoder(encoder, X, device=self.device)
            kmeans = self._fit_best_kmeans_for_latent(z, cluster_range)
            self.clusterers[name] = kmeans
            distances = kmeans.transform(z)  # (N, k)
            q = softmax(-distances / temperature, axis=1)
            q = q / q.sum(axis=1, keepdims=True)
            swav_features[name] = q
        return swav_features

    def deepcluster_step_swav(self, X, n_iters=1, cluster_range=(4,16), temperature=0.1, refine_encoder=False, lr=1e-3, epochs_finetune=5):
        """DeepCluster step using SWAV-style soft cluster assignments.
        Replaces hard pseudo-labels with normalized moving-target distributions.
        Args:
            X (np.ndarray): input features
            n_iters (int): number of DeepCluster iterations
            cluster_range (tuple[int]): min/max clusters for KMeans
            temperature (float): softmax temperature
            refine_encoder (bool): whether to fine-tune encoder toward clusters
            lr (float): learning rate for encoder refinement
            epochs_finetune (int): fine-tune steps per iteration"""
        if X is None or len(X) == 0:
            print("[DeepCluster] Skipping: no data provided.")
            self.cluster_prob_matrix = np.zeros((0, 0))
            return
        for _ in range(n_iters):
            for name, encoder in self.encoders_dict.items():
                # --- fit KMeans ---
                z      = Latents.get_latent_from_encoder(encoder, X, device=self.device)
                kmeans = self._fit_best_kmeans_for_latent(z, cluster_range)
                self.clusterers[name] = kmeans

                # --- SWAV-style soft assignment ---
                distances = kmeans.transform(z)
                q = softmax(-distances / temperature, axis=1)
                q = q / q.sum(axis=1, keepdims=True)  # normalize

                # store features
                if self.cluster_prob_matrix is None:
                    self.cluster_prob_matrix = q
                else:
                    self.cluster_prob_matrix = np.concatenate([self.cluster_prob_matrix, q], axis=1)

                # --- optional lightweight refinement ---
                if refine_encoder:
                    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
                    centers   = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=self.device)
                    for _ in range(epochs_finetune):
                        z_tensor = Latents.get_latent_tensor(encoder, X, train_encoder=True, device=self.device)
                        q_tensor = torch.softmax(-torch.cdist(z_tensor, centers), dim=1)
                        p_tensor = torch.tensor(self.target_distribution(q_tensor.detach().cpu().numpy()), device=q_tensor.device)
                        loss = F.kl_div(q_tensor.log(), p_tensor, reduction="batchmean")
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

