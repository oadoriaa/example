import random
from typing import List, Optional, Tuple

TimeSeries = List[float]


def _median(values: List[float]) -> float:
    """Median without external libraries."""
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def mom_barycenter(timeseries: List[TimeSeries],
                   labels: List[int],
                   k: int,
                   length: int,
                   n_blocks: int = 5,
                   seed: int = 0) -> Optional[TimeSeries]:
    """
    Median-of-means (MoM) barycenter for cluster k.

    For each time index t:
      - split cluster members into n_blocks blocks
      - compute block means at t
      - set c[t] = median(block_means)

    This is robust to outliers vs. the plain mean barycenter.

    Parameters
    ----------
    timeseries : list of list of float
    labels     : list of int
    k          : cluster id to update
    length     : time series length (all must match)
    n_blocks   : number of blocks (>=1). Use min(n_blocks, m) internally.
    seed       : random seed for shuffling members before blocking

    Returns
    -------
    centroid : list[float] or None if cluster empty
    """
    # Collect cluster members
    members = [x for x, lab in zip(timeseries, labels) if lab == k]
    m = len(members)
    if m == 0:
        return None

    # Ensure blocks make sense
    B = max(1, min(n_blocks, m))

    rng = random.Random(seed)
    rng.shuffle(members)

    # Split into B blocks as evenly as possible
    blocks = [[] for _ in range(B)]
    for idx, x in enumerate(members):
        blocks[idx % B].append(x)

    centroid = [0.0] * length

    # For each time index, compute MoM: median of block means
    for t in range(length):
        block_means = []
        for b in blocks:
            s = 0.0
            for x in b:
                s += x[t]
            block_means.append(s / len(b))
        centroid[t] = _median(block_means)

    return centroid


# ----------------------------
# Integration into k-means loop
# ----------------------------

def sq_euclid(x: TimeSeries, c: TimeSeries) -> float:
    s = 0.0
    for xt, ct in zip(x, c):
        d = xt - ct
        s += d * d
    return s


def inertia(timeseries: List[TimeSeries],
            centroids: List[TimeSeries],
            labels: List[int]) -> float:
    total = 0.0
    for x, k in zip(timeseries, labels):
        total += sq_euclid(x, centroids[k])
    return total


def assign(timeseries: List[TimeSeries],
           centroids: List[TimeSeries]) -> List[int]:
    labels = []
    for x in timeseries:
        best_k = 0
        best_d = sq_euclid(x, centroids[0])
        for k in range(1, len(centroids)):
            d = sq_euclid(x, centroids[k])
            if d < best_d:
                best_d = d
                best_k = k
        labels.append(best_k)
    return labels


def kmeans_timeseries_mom(timeseries: List[TimeSeries],
                          K: int,
                          n_blocks: int = 5,
                          max_iter: int = 100,
                          tol: float = 1e-8,
                          seed: int = 0) -> Tuple[List[TimeSeries], List[int], List[float]]:
    """
    K-means-style clustering with MoM barycenter update.
    Distance: squared Euclidean (aligned).
    Update: MoM barycenter (median-of-means) per time index.

    Note: This is a robust k-means variant; it may not strictly minimize standard SSE each step,
    but is typically more robust to outliers.
    """
    if K <= 0:
        raise ValueError("K must be positive.")
    if not timeseries:
        raise ValueError("timeseries is empty.")
    n = len(timeseries)
    if K > n:
        raise ValueError("K cannot exceed number of time series.")

    length = len(timeseries[0])
    for x in timeseries:
        if len(x) != length:
            raise ValueError("All time series must have the same length.")

    rng = random.Random(seed)

    # Initialize centroids by sampling K distinct series
    init_idx = rng.sample(range(n), K)
    centroids = [list(timeseries[i]) for i in init_idx]

    labels = [0] * n
    inertia_hist = []
    prev_inertia = None

    for it in range(max_iter):
        # 1) Assignment
        labels = assign(timeseries, centroids)

        # 2) Update (MoM barycenters)
        new_centroids = []
        # Use different seeds per (iteration, cluster) to avoid identical blockings each time
        for k in range(K):
            c = mom_barycenter(
                timeseries, labels, k, length,
                n_blocks=n_blocks,
                seed=seed + 100000 * it + 1000 * k
            )
            if c is None:
                # Empty cluster: reseed centroid to random point
                c = list(timeseries[rng.randrange(n)])
            new_centroids.append(c)
        centroids = new_centroids

        # 3) Track standard SSE inertia (for monitoring)
        cur_inertia = inertia(timeseries, centroids, labels)
        inertia_hist.append(cur_inertia)

        # 4) Stop if stable
        if prev_inertia is not None:
            if abs(prev_inertia - cur_inertia) <= tol * (1.0 + prev_inertia):
                break
        prev_inertia = cur_inertia

    return centroids, labels, inertia_hist


if __name__ == "__main__":
    X = [
        [1.0, 2.0, 1.0],
        [2.0, 1.0, 2.0],
        [8.0, 9.0, 8.0],
        [9.0, 8.0, 9.0],
        # Add an outlier to see robustness effects
        [50.0, 50.0, 50.0],
    ]

    centroids, labels, hist = kmeans_timeseries_mom(
        X, K=2, n_blocks=3, max_iter=50, seed=42
    )

    print("labels:", labels)
    print("final inertia:", hist[-1])
    print("centroids:", centroids)
