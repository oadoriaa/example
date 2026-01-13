import random
import math
from typing import List, Tuple, Optional

TimeSeries = List[float]


def sq_euclid_weighted(x: TimeSeries, c: TimeSeries, w: TimeSeries) -> float:
    """
    Weighted squared Euclidean distance:
        sum_t w_t * (x_t - c_t)^2
    """
    s = 0.0
    for xt, ct, wt in zip(x, c, w):
        d = xt - ct
        s += wt * d * d
    return s


def inertia_weighted(timeseries: List[TimeSeries],
                     centroids: List[TimeSeries],
                     labels: List[int],
                     weights_by_cluster: List[TimeSeries]) -> float:
    """
    Weighted inertia:
        sum_i sum_t w_{label_i,t} * (x_{i,t} - c_{label_i,t})^2
    """
    total = 0.0
    for x, k in zip(timeseries, labels):
        total += sq_euclid_weighted(x, centroids[k], weights_by_cluster[k])
    return total


def compute_cluster_stats(timeseries: List[TimeSeries],
                          labels: List[int],
                          k: int,
                          length: int) -> Tuple[Optional[TimeSeries], Optional[TimeSeries], int]:
    """
    For cluster k:
      - mu_t = mean across members at each time t
      - sigma2_t = (population) variance across members at each time t
    Returns (mu, sigma2, m). If cluster empty: (None, None, 0).
    """
    members = [x for x, lab in zip(timeseries, labels) if lab == k]
    m = len(members)
    if m == 0:
        return None, None, 0

    mu = [0.0] * length
    inv_m = 1.0 / m

    for x in members:
        for t in range(length):
            mu[t] += x[t]
    for t in range(length):
        mu[t] *= inv_m

    sigma2 = [0.0] * length
    for x in members:
        for t in range(length):
            d = x[t] - mu[t]
            sigma2[t] += d * d
    for t in range(length):
        sigma2[t] *= inv_m  # population variance

    return mu, sigma2, m


def weighted_barycenter(timeseries: List[TimeSeries],
                        labels: List[int],
                        k: int,
                        length: int,
                        p: float = 1.0,
                        eps: float = 1e-12) -> Tuple[Optional[TimeSeries], Optional[TimeSeries]]:
    """
    Compute weighted barycenter for cluster k under the weighted SSE objective:
        sum_i sum_t w_t * (x_{i,t} - c_t)^2
    where w_t = 1 / (sigma_t^p + eps)

    Returns (centroid, weights). If cluster is empty: (None, None).

    Note:
      With per-time weights that are constant across members, the centroid that minimizes
      weighted SSE is still the per-time mean mu_t. We still compute and return weights
      because they matter in the assignment/distance step and the weighted inertia.
    """
    if p <= 0:
        raise ValueError("p must be > 0.")

    mu, sigma2, m = compute_cluster_stats(timeseries, labels, k, length)
    if m == 0:
        return None, None

    # sigma_t = sqrt(sigma2_t), and w_t = 1 / (sigma_t^p + eps)
    # equivalently w_t = 1 / ( (sigma2_t)^(p/2) + eps )
    weights = [0.0] * length
    half_p = 0.5 * p
    for t in range(length):
        weights[t] = 1.0 / (math.pow(sigma2[t] + eps, half_p) + eps)

    centroid = mu  # minimizer of weighted SSE given these weights
    return centroid, weights


def assign_weighted(timeseries: List[TimeSeries],
                    centroids: List[TimeSeries],
                    weights_by_cluster: List[TimeSeries]) -> List[int]:
    """
    Assign each series to the nearest centroid using that cluster's weights.
    label_i = argmin_k sum_t w_{k,t} * (x_{i,t} - c_{k,t})^2
    """
    labels = []
    K = len(centroids)
    for x in timeseries:
        best_k = 0
        best_d = sq_euclid_weighted(x, centroids[0], weights_by_cluster[0])
        for k in range(1, K):
            d = sq_euclid_weighted(x, centroids[k], weights_by_cluster[k])
            if d < best_d:
                best_d = d
                best_k = k
        labels.append(best_k)
    return labels


def kmeans_timeseries_weighted(timeseries: List[TimeSeries],
                               K: int,
                               p: float = 1.0,
                               eps: float = 1e-12,
                               max_iter: int = 100,
                               tol: float = 1e-8,
                               seed: int = 0):
    """
    K-means-like clustering for equal-length time series with per-cluster, per-time weights:
        w_{k,t} = 1 / (sigma_{k,t}^p + eps)

    Objective minimized (alternating):
        J = sum_i sum_t w_{label_i,t} * (x_{i,t} - c_{label_i,t})^2
    """
    if K <= 0:
        raise ValueError("K must be positive.")
    if len(timeseries) == 0:
        raise ValueError("timeseries is empty.")
    if p <= 0:
        raise ValueError("p must be > 0.")

    n = len(timeseries)
    length = len(timeseries[0])
    for x in timeseries:
        if len(x) != length:
            raise ValueError("All time series must have the same length for this implementation.")
    if K > n:
        raise ValueError("K cannot exceed number of time series.")

    random.seed(seed)

    # Initialize centroids by sampling K distinct series
    init_idx = random.sample(range(n), K)
    centroids = [list(timeseries[i]) for i in init_idx]

    # Start with uniform weights (all ones); they will be updated after first assignment
    weights_by_cluster = [[1.0] * length for _ in range(K)]

    labels = [0] * n
    inertia_hist = []
    prev_inertia = None

    for _ in range(max_iter):
        # 1) Assignment step (uses current centroids + weights)
        labels = assign_weighted(timeseries, centroids, weights_by_cluster)

        # 2) Update step: recompute centroid + weights per cluster
        new_centroids = []
        new_weights = []
        for k in range(K):
            c, w = weighted_barycenter(timeseries, labels, k, length, p=p, eps=eps)
            if c is None:
                # Empty cluster: reseed centroid; keep weights uniform initially
                c = list(timeseries[random.randrange(n)])
                w = [1.0] * length
            new_centroids.append(c)
            new_weights.append(w)

        centroids = new_centroids
        weights_by_cluster = new_weights

        # 3) Track weighted inertia
        cur_inertia = inertia_weighted(timeseries, centroids, labels, weights_by_cluster)
        inertia_hist.append(cur_inertia)

        # 4) Convergence check
        if prev_inertia is not None:
            if abs(prev_inertia - cur_inertia) <= tol * (1.0 + prev_inertia):
                break
        prev_inertia = cur_inertia

    return centroids, labels, weights_by_cluster, inertia_hist


if __name__ == "__main__":
    X = [
        [1.0, 2.0, 1.0],
        [2.0, 1.0, 2.0],
        [8.0, 9.0, 8.0],
        [9.0, 8.0, 9.0],
    ]

    centroids, labels, weights, hist = kmeans_timeseries_weighted(
        X, K=2, p=1.0, eps=1e-12, max_iter=50, seed=42
    )

    print("labels:", labels)
    print("final weighted inertia:", hist[-1])
    print("centroids:", centroids)
    print("weights_by_cluster:", weights)
