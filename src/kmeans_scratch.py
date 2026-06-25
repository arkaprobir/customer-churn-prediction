"""
K-Means implemented from scratch using only NumPy.

This is the piece that proves you understand the algorithm at the
level interviewers actually probe -- not just sklearn.fit().

Key DSA concepts on display:
- Euclidean distance computation (vectorized, O(n*k*d) per iteration)
- Iterative convergence (like a fixed-point algorithm)
- Centroid update = mean of assigned points (reduce operation)
- Stopping condition based on centroid movement (epsilon threshold)
"""

import numpy as np


class KMeansScratch:
    def __init__(self, n_clusters=4, max_iters=300, tol=1e-4, random_state=42):
        self.k = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None

    def _init_centroids(self, X):
        """K-Means++ initialization: spreads out initial centroids
        instead of pure random picks, which avoids bad local minima
        and speeds up convergence. This is the detail interviewers
        love to ask about."""
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]

        centroids = [X[rng.randint(n_samples)]]
        for _ in range(1, self.k):
            dist_sq = np.min(
                [np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0
            )
            probs = dist_sq / dist_sq.sum()
            next_idx = rng.choice(n_samples, p=probs)
            centroids.append(X[next_idx])
        return np.array(centroids)

    def _assign_clusters(self, X):
        # Distance of every point to every centroid: shape (n_samples, k)
        distances = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2)
        return np.argmin(distances, axis=1), distances

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.centroids = self._init_centroids(X)

        for iteration in range(self.max_iters):
            labels, distances = self._assign_clusters(X)

            new_centroids = np.array([
                X[labels == j].mean(axis=0) if np.any(labels == j) else self.centroids[j]
                for j in range(self.k)
            ])

            shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            if shift < self.tol:
                break

        self.labels_ = labels
        self.inertia_ = sum(
            np.sum((X[labels == j] - self.centroids[j]) ** 2)
            for j in range(self.k)
        )
        self.n_iter_ = iteration + 1
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        labels, _ = self._assign_clusters(X)
        return labels


def elbow_method(X, k_range=range(2, 9)):
    """Helper to justify choice of k -- plot inertia vs k and look for the bend."""
    inertias = []
    for k in k_range:
        km = KMeansScratch(n_clusters=k).fit(X)
        inertias.append(km.inertia_)
    return list(k_range), inertias


if __name__ == "__main__":
    # Quick sanity check against sklearn
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=500, centers=4, random_state=42)

    custom = KMeansScratch(n_clusters=4).fit(X)
    sk = KMeans(n_clusters=4, n_init=10, random_state=42).fit(X)

    print(f"Custom K-Means inertia: {custom.inertia_:.2f}  (iters: {custom.n_iter_})")
    print(f"Sklearn K-Means inertia: {sk.inertia_:.2f}")
