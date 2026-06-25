"""
kmeans_from_scratch.py
------------------------
A from-scratch implementation of the K-Means clustering algorithm.

Why implement this manually instead of just using sklearn.cluster.KMeans?
  - Interviewers will ask "do you actually understand how K-Means works?"
    This proves it.
  - It lets us reason clearly about time complexity: O(n * k * i * d)
      n = number of points, k = number of clusters,
      i = number of iterations, d = number of dimensions/features
  - It shows understanding of WHY initialization matters (K-Means++ vs random)
    and WHY the algorithm can converge to a local optimum.

Algorithm steps (Lloyd's algorithm):
  1. Initialize k centroids (we use K-Means++ for smarter initialization)
  2. Assign each point to its nearest centroid (Assignment step)
  3. Recompute each centroid as the mean of points assigned to it (Update step)
  4. Repeat steps 2-3 until centroids stop moving (convergence) or max_iters hit
"""

import numpy as np


class KMeansScratch:
    def __init__(self, n_clusters=4, max_iters=100, tol=1e-4, random_state=42, n_init=10):
        self.k = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init  # number of random restarts; keep the best (lowest inertia) run
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None  # sum of squared distances to nearest centroid

    def _kmeans_plus_plus_init(self, X):
        """
        Smarter initialization than pure random choice.
        Picks the first centroid randomly, then each subsequent centroid
        is chosen with probability proportional to its squared distance
        from the nearest already-chosen centroid. This spreads out the
        initial centroids and avoids poor local optima.
        """
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        centroids = [X[rng.integers(n_samples)]]

        for _ in range(1, self.k):
            dist_sq = np.min(
                [np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0
            )
            probs = dist_sq / dist_sq.sum()
            next_idx = rng.choice(n_samples, p=probs)
            centroids.append(X[next_idx])

        return np.array(centroids)

    def _assign_clusters(self, X):
        # Compute distance from every point to every centroid: shape (n, k)
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1), distances

    def _fit_once(self, X, seed):
        rng_state = self.random_state
        self.random_state = seed
        centroids = self._kmeans_plus_plus_init(X)
        self.random_state = rng_state

        for iteration in range(self.max_iters):
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.array([
                X[labels == j].mean(axis=0) if np.any(labels == j)
                else centroids[j]  # handle empty cluster edge case
                for j in range(self.k)
            ])

            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids

            if shift < self.tol:
                break

        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        inertia = np.sum(np.min(distances, axis=1) ** 2)
        return centroids, labels, inertia, iteration + 1

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        best_inertia = np.inf
        best_result = None

        for restart in range(self.n_init):
            seed = self.random_state + restart  # different init each restart
            centroids, labels, inertia, n_iter = self._fit_once(X, seed)
            if inertia < best_inertia:
                best_inertia = inertia
                best_result = (centroids, labels, inertia, n_iter)

        self.centroids, self.labels_, self.inertia_, self.n_iter_ = best_result
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        labels, _ = self._assign_clusters(X)
        return labels


def elbow_method(X, k_range=range(1, 11), random_state=42):
    """
    Helper to find optimal k by plotting inertia vs k.
    The 'elbow' point (where inertia stops decreasing sharply) is a good
    candidate for the number of clusters.
    """
    inertias = []
    for k in k_range:
        km = KMeansScratch(n_clusters=k, random_state=random_state)
        km.fit(X)
        inertias.append(km.inertia_)
    return list(k_range), inertias


if __name__ == "__main__":
    # quick smoke test against sklearn's implementation
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans as SkKMeans

    X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

    custom = KMeansScratch(n_clusters=4, random_state=42).fit(X)
    sk = SkKMeans(n_clusters=4, random_state=42, n_init=10).fit(X)

    print("Custom inertia :", round(custom.inertia_, 2))
    print("Sklearn inertia:", round(sk.inertia_, 2))
    print("(Close inertia values confirm the from-scratch implementation works correctly)")
