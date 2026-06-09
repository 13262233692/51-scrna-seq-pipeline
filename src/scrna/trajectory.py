from __future__ import annotations

import logging

import anndata as ad
import igraph as ig
import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)


def compute_cluster_centroids(
    embedding: np.ndarray,
    cluster_labels: np.ndarray | list[str],
) -> tuple[np.ndarray, list[str]]:
    unique_clusters = sorted(set(str(c) for c in cluster_labels))
    n_clusters = len(unique_clusters)
    n_dims = embedding.shape[1]
    centroids = np.zeros((n_clusters, n_dims), dtype=np.float64)

    for i, c in enumerate(unique_clusters):
        mask = np.array([str(lb) == c for lb in cluster_labels])
        centroids[i] = embedding[mask].mean(axis=0)

    logger.info("Computed %d cluster centroids in %dD space", n_clusters, n_dims)
    return centroids, unique_clusters


def compute_centroid_distance_matrix(centroids: np.ndarray) -> np.ndarray:
    dist_matrix = squareform(pdist(centroids, metric="euclidean"))
    logger.info("Centroid distance matrix: %dx%d", dist_matrix.shape[0], dist_matrix.shape[1])
    return dist_matrix


def build_mst(
    dist_matrix: np.ndarray,
    cluster_labels: list[str],
) -> tuple[ig.Graph, list[tuple[int, int]], np.ndarray]:
    n = dist_matrix.shape[0]
    edges = []
    weights = []

    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
            weights.append(dist_matrix[i, j])

    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es["weight"] = weights

    mst = g.spanning_tree(weights="weight")

    mst_edges = []
    for e in mst.es:
        mst_edges.append((e.source, e.target))

    logger.info("MST: %d nodes, %d edges", n, len(mst_edges))

    mst_adj = np.full((n, n), np.inf)
    for e in mst.es:
        i, j = e.source, e.target
        mst_adj[i, j] = dist_matrix[i, j]
        mst_adj[j, i] = dist_matrix[i, j]

    return mst, mst_edges, mst_adj


def compute_pseudotime(
    mst_adj: np.ndarray,
    root_cluster_idx: int,
    cluster_labels_list: list[str],
) -> np.ndarray:
    n = mst_adj.shape[0]
    np.fill_diagonal(mst_adj, 0.0)

    geodesic_dists = shortest_path(mst_adj, method="D", directed=False,
                                   unweighted=False)

    root_dists = geodesic_dists[root_cluster_idx]
    root_name = cluster_labels_list[root_cluster_idx]

    logger.info(
        "Pseudotime from root cluster '%s' (idx=%d): max=%.2f, mean=%.2f",
        root_name, root_cluster_idx, root_dists.max(), root_dists.mean(),
    )
    return root_dists


def assign_cell_pseudotime(
    cluster_labels: np.ndarray | list[str],
    cluster_pseudotime: np.ndarray,
    unique_clusters: list[str],
) -> np.ndarray:
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    cell_pseudotime = np.array([
        cluster_pseudotime[cluster_to_idx[str(c)]]
        for c in cluster_labels
    ], dtype=np.float64)
    return cell_pseudotime


def compute_smooth_trajectory(
    centroids: np.ndarray,
    mst_edges: list[tuple[int, int]],
    n_interp: int = 20,
) -> list[list[float]]:
    curves = []
    for src, tgt in mst_edges:
        p0 = centroids[src]
        p1 = centroids[tgt]
        for t in np.linspace(0, 1, n_interp):
            pt = p0 * (1 - t) + p1 * t
            curves.append([float(pt[0]), float(pt[1])])
    return curves


def trajectory_inference(
    adata: ad.AnnData,
    root_cluster: str | None = None,
    use_umap: bool = True,
    key_added: str = "pseudotime",
) -> ad.AnnData:
    if use_umap and "X_umap" in adata.obsm:
        embedding = adata.obsm["X_umap"]
        logger.info("Using UMAP embedding for trajectory inference (shape=%s)", embedding.shape)
    elif "X_pca" in adata.obsm:
        embedding = adata.obsm["X_pca"]
        logger.info("Using PCA embedding for trajectory inference (shape=%s)", embedding.shape)
    else:
        raise ValueError("No embedding found. Run UMAP or PCA first.")

    if "leiden" not in adata.obs.columns:
        raise ValueError("Leiden clustering must be run before trajectory inference.")

    cluster_labels = adata.obs["leiden"].values

    centroids, unique_clusters = compute_cluster_centroids(embedding, cluster_labels)

    dist_matrix = compute_centroid_distance_matrix(centroids)

    mst, mst_edges, mst_adj = build_mst(dist_matrix, unique_clusters)

    if root_cluster is not None:
        if root_cluster not in unique_clusters:
            raise ValueError(
                f"Root cluster '{root_cluster}' not found. "
                f"Available clusters: {unique_clusters}"
            )
        root_idx = unique_clusters.index(root_cluster)
    else:
        root_idx = int(np.argmin(centroids[:, 1]))
        logger.info("Auto-selecting root cluster: '%s' (lowest centroid Y)", unique_clusters[root_idx])

    cluster_pseudotime = compute_pseudotime(mst_adj, root_idx, unique_clusters)

    cell_pseudotime = assign_cell_pseudotime(cluster_labels, cluster_pseudotime, unique_clusters)

    adata.obs[key_added] = cell_pseudotime
    adata.uns["trajectory"] = {
        "centroids": centroids,
        "unique_clusters": unique_clusters,
        "mst_edges": mst_edges,
        "dist_matrix": dist_matrix,
        "cluster_pseudotime": cluster_pseudotime,
        "root_cluster": unique_clusters[root_idx],
        "root_cluster_idx": root_idx,
    }

    n_clusters = len(unique_clusters)
    n_edges = len(mst_edges)
    logger.info(
        "Trajectory inference complete: %d clusters, %d MST edges, "
        "root='%s', pseudotime range=[0, %.2f]",
        n_clusters, n_edges, unique_clusters[root_idx],
        cluster_pseudotime.max(),
    )
    return adata
