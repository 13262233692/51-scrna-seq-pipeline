from __future__ import annotations

import logging

import anndata as ad
import igraph as ig
import leidenalg
import numpy as np
from scipy import sparse
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)


def build_knn_graph(
    X: np.ndarray,
    n_neighbors: int = 15,
) -> sparse.csr_matrix:
    n = X.shape[0]
    n_neighbors = min(n_neighbors, n - 1)

    logger.info("Building k-NN graph: n=%d, k=%d", n, n_neighbors)

    tree = KDTree(X)
    distances, indices = tree.query(X, k=n_neighbors + 1)
    indices = indices[:, 1:]
    distances = distances[:, 1:]

    rows = np.repeat(np.arange(n), n_neighbors)
    cols = indices.ravel()
    data = np.exp(-distances.ravel() / (distances.ravel().mean() + 1e-12))

    connectivities = sparse.csr_matrix(
        (data, (rows, cols)), shape=(n, n)
    )
    connectivities = (connectivities + connectivities.T) / 2.0
    connectivities.eliminate_zeros()

    logger.info("k-NN graph: %d edges", connectivities.nnz)
    return connectivities


def leiden_cluster(
    adata: ad.AnnData,
    n_neighbors: int = 15,
    resolution: float = 1.0,
    n_iterations: int = -1,
    use_pca: bool = True,
    key_added: str = "leiden",
) -> ad.AnnData:
    if use_pca and "X_pca" in adata.obsm:
        X = adata.obsm["X_pca"]
        logger.info("Using PCA embeddings for clustering (shape=%s)", X.shape)
    else:
        X = adata.X
        if sparse.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float64)

    connectivities = build_knn_graph(X, n_neighbors=n_neighbors)
    sources, targets = connectivities.nonzero()
    weights = connectivities.data

    g = ig.Graph(n=connectivities.shape[0], edges=list(zip(sources, targets)), directed=False)
    g.es["weight"] = weights

    logger.info("Running Leiden clustering (resolution=%.2f) ...", resolution)
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        n_iterations=n_iterations,
    )

    labels = np.array(partition.membership)
    adata.obs[key_added] = labels.astype(str)
    adata.obs[key_added] = adata.obs[key_added].astype("category")

    adata.obsp["connectivities"] = connectivities

    n_clusters = len(set(labels))
    logger.info("Leiden clustering: %d clusters detected", n_clusters)
    return adata
