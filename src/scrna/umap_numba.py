from __future__ import annotations

import logging
from typing import Literal

import numba
import numpy as np
from pynndescent import NNDescent
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


@numba.njit(fastmath=True, cache=True)
def _smooth_knn_dist(
    distances: np.ndarray,
    n_neighbors: int,
    local_connectivity: float = 1.0,
    n_iter: int = 64,
    bandwidth: float = 1.0,
    tol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    n = distances.shape[0]
    rho = np.zeros(n, dtype=np.float64)
    sigmas = np.ones(n, dtype=np.float64)

    for i in range(n):
        lo = 0.0
        hi = np.inf
        mid = 1.0

        rho[i] = distances[i, 0] if distances[i, 0] > 0 else 0.0

        target = np.log2(n_neighbors) * bandwidth

        for _ in range(n_iter):
            psum = 0.0
            for j in range(n_neighbors):
                d = distances[i, j] - rho[i]
                if d > 0:
                    psum += np.exp(-(d * d) / (mid * mid))
                else:
                    psum += 1.0

            if np.abs(psum - target) < tol:
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == np.inf:
                    mid *= 2.0
                else:
                    mid = (lo + hi) / 2.0

        sigmas[i] = max(mid, 1e-12)

    return rho, sigmas


@numba.njit(parallel=True, fastmath=True, cache=True)
def _compute_membership_strengths(
    indices: np.ndarray,
    distances: np.ndarray,
    rho: np.ndarray,
    sigmas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = indices.shape[0]
    n_neighbors = indices.shape[1]
    nnz = n * n_neighbors

    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty(nnz, dtype=np.int64)
    vals = np.empty(nnz, dtype=np.float64)

    for i in numba.prange(n):
        for j in range(n_neighbors):
            idx = i * n_neighbors + j
            rows[idx] = i
            cols[idx] = indices[i, j]
            d = distances[i, j] - rho[i]
            if d > 0:
                vals[idx] = np.exp(-(d * d) / (sigmas[i] * sigmas[i]))
            else:
                vals[idx] = 1.0

    return rows, cols, vals


@numba.njit(parallel=True, fastmath=True, cache=True)
def _optimize_layout(
    head: np.ndarray,
    tail: np.ndarray,
    weights: np.ndarray,
    n_vertices: int,
    n_components: int = 2,
    n_epochs: int = 500,
    learning_rate: float = 1.0,
    min_dist: float = 0.1,
    spread: float = 1.0,
    negative_sample_rate: int = 5,
    random_state: int = 42,
    repulsion_strength: float = 1.0,
) -> np.ndarray:
    a = 1.577
    b = 0.8951

    np.random.seed(random_state)
    embedding = np.random.randn(n_vertices, n_components).astype(np.float64) * 0.0001

    n_edges = len(head)
    alpha = learning_rate

    for epoch in range(n_epochs):
        alpha = learning_rate * (1.0 - epoch / n_epochs)
        if alpha < learning_rate * 0.01:
            alpha = learning_rate * 0.01

        for e in range(n_edges):
            i = head[e]
            j = tail[e]
            w = weights[e]

            for d in range(n_components):
                dist_sq = 0.0
                for d2 in range(n_components):
                    diff = embedding[i, d2] - embedding[j, d2]
                    dist_sq += diff * diff
                dist_sq = max(dist_sq, 1e-12)

                grad_coeff = -2.0 * a * b * pow(dist_sq, b - 1.0)
                grad_coeff /= (a * pow(dist_sq, b) + 1.0)

                diff = embedding[i, d] - embedding[j, d]
                grad = grad_coeff * diff * w * alpha
                embedding[i, d] += grad

            for _ in range(negative_sample_rate):
                k = np.random.randint(0, n_vertices)
                if k == i:
                    continue

                dist_sq = 0.0
                for d2 in range(n_components):
                    diff = embedding[i, d2] - embedding[k, d2]
                    dist_sq += diff * diff
                dist_sq = max(dist_sq, 1e-12)

                grad_coeff = 2.0 * b / (0.001 + dist_sq * (a * pow(dist_sq, b - 1.0) + 1.0))
                grad_coeff *= repulsion_strength

                for d in range(n_components):
                    diff = embedding[i, d] - embedding[k, d]
                    grad = grad_coeff * diff * alpha / negative_sample_rate
                    embedding[i, d] += grad

    return embedding


def _pynndescent_knn(
    X: np.ndarray,
    n_neighbors: int,
    metric: str = "euclidean",
    random_state: int = 42,
    n_jobs: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    logger.info("  PyNNDescent: building ANN index on %d samples ...", X.shape[0])
    index = NNDescent(
        X,
        n_neighbors=n_neighbors,
        metric=metric,
        random_state=random_state,
        n_jobs=n_jobs,
        compressed=True,
    )
    indices, distances = index.neighbor_graph
    indices = indices[:, 1:].astype(np.int64, copy=True)
    distances = distances[:, 1:].astype(np.float64, copy=True)
    return indices, distances


def umap_numba(
    X: np.ndarray,
    n_neighbors: int = 15,
    n_components: int = 2,
    n_epochs: int = 500,
    learning_rate: float = 1.0,
    min_dist: float = 0.1,
    spread: float = 1.0,
    metric: Literal["euclidean", "cosine"] = "euclidean",
    negative_sample_rate: int = 5,
    random_state: int = 42,
    n_pcs: int | None = None,
) -> np.ndarray:
    if X.ndim == 1:
        raise ValueError("Input must be 2D array")

    if n_pcs is not None and X.shape[1] > n_pcs:
        logger.info("Using first %d columns as pre-computed PCA embeddings", n_pcs)
        X_work = X[:, :n_pcs].astype(np.float64, copy=True)
    else:
        X_work = np.asarray(X, dtype=np.float64)

    n_samples = X_work.shape[0]
    logger.info("UMAP: %d samples, n_neighbors=%d (PyNNDescent backend)", n_samples, n_neighbors)

    n_neighbors = min(n_neighbors, n_samples - 1)

    logger.info("  Step 1/4: Computing k-NN graph via PyNNDescent ...")
    indices, distances = _pynndescent_knn(
        X_work, n_neighbors, metric=metric, random_state=random_state,
    )

    logger.info("  Step 2/4: Smoothing k-NN distances ...")
    rho, sigmas = _smooth_knn_dist(distances, n_neighbors)

    logger.info("  Step 3/4: Building fuzzy simplicial set ...")
    rows, cols, vals = _compute_membership_strengths(indices, distances, rho, sigmas)

    graph = csr_matrix((vals, (rows, cols)), shape=(n_samples, n_samples))
    graph_T = graph.transpose()
    product = graph.multiply(graph_T)
    sym_graph = graph + graph_T - product
    sym_graph.eliminate_zeros()

    coo = sym_graph.tocoo()
    head = coo.row.astype(np.int64)
    tail = coo.col.astype(np.int64)
    weights = coo.data.astype(np.float64)

    weight_sum = weights.sum()
    if weight_sum > 0:
        weights = weights / weight_sum * len(weights)

    logger.info("  Step 4/4: Optimizing layout (%d edges, %d epochs) ...", len(head), n_epochs)
    embedding = _optimize_layout(
        head=head,
        tail=tail,
        weights=weights,
        n_vertices=n_samples,
        n_components=n_components,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        min_dist=min_dist,
        spread=spread,
        negative_sample_rate=negative_sample_rate,
        random_state=random_state,
    )

    logger.info("UMAP complete: embedding shape=%s", embedding.shape)
    return embedding
