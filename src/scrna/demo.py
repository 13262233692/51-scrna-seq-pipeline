"""Generate a synthetic .h5ad dataset for testing the scRNA-seq pipeline.

Creates a sparse expression matrix with configurable cell/gene counts,
embedding mitochondrial gene markers, and distinct cluster structure.
"""

import argparse
import logging
import sys

import anndata as ad
import numpy as np
from scipy import sparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def generate_synthetic(
    n_cells: int = 10_000,
    n_genes: int = 5_000,
    n_mito_genes: int = 15,
    n_clusters: int = 8,
    dead_cell_fraction: float = 0.05,
    sparsity: float = 0.80,
    random_state: int = 42,
) -> ad.AnnData:
    rng = np.random.default_rng(random_state)

    logger.info("Generating synthetic dataset: %d cells x %d genes", n_cells, n_genes)

    gene_names = np.array([f"GENE{i:05d}" for i in range(n_genes)])
    for i in range(n_mito_genes):
        gene_names[i] = f"MT-{chr(65 + i % 26)}{i // 26 + 1}"
    cell_names = np.array([f"CELL{i:07d}" for i in range(n_cells)])

    cluster_labels = rng.integers(0, n_clusters, size=n_cells)

    n_dead = int(n_cells * dead_cell_fraction)
    is_dead = np.zeros(n_cells, dtype=bool)
    dead_idx = rng.choice(n_cells, size=n_dead, replace=False)
    is_dead[dead_idx] = True

    cluster_bases = rng.exponential(scale=5.0, size=(n_clusters, n_genes)).astype(np.float32)
    for c in range(n_clusters):
        marker_start = int(c * n_genes / n_clusters)
        marker_end = min(marker_start + 100, n_genes)
        cluster_bases[c, marker_start:marker_end] += rng.exponential(scale=30.0, size=marker_end - marker_start).astype(np.float32)

    counts = np.zeros((n_cells, n_genes), dtype=np.float32)
    for c in range(n_clusters):
        cell_idx = np.where(cluster_labels == c)[0]
        n_c = len(cell_idx)
        if n_c == 0:
            continue
        base = cluster_bases[c]
        lib_sizes = rng.lognormal(mean=np.log(8000), sigma=0.4, size=n_c).astype(np.float32)
        for i, ci in enumerate(cell_idx):
            lam = base * lib_sizes[i] / (base.sum() + 1e-12)
            counts[ci] = rng.poisson(np.maximum(lam, 0.05))

    for idx in dead_idx:
        mito_boost = rng.uniform(3, 8)
        counts[idx, :n_mito_genes] = (counts[idx, :n_mito_genes] * mito_boost).astype(np.float32)
        counts[idx, n_mito_genes:] = (counts[idx, n_mito_genes:] * 0.1).astype(np.float32)

    drop_mask = rng.random((n_cells, n_genes)) > sparsity
    counts *= drop_mask

    X = sparse.csr_matrix(counts, dtype=np.float32)

    obs = ad.AnnData(X).obs
    adata = ad.AnnData(
        X=X,
        obs={"cluster_true": cluster_labels.astype(str), "is_dead": is_dead},
        var={"gene_ids": gene_names},
    )
    adata.obs_names = cell_names
    adata.var_names = gene_names

    logger.info("Synthetic dataset: %d cells, %d genes, %d dead cells, sparsity=%.2f%%",
                n_cells, n_genes, n_dead, (1 - drop_mask.mean()) * 100)
    return adata


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic .h5ad for pipeline testing")
    parser.add_argument("-n", "--n-cells", type=int, default=10_000, help="Number of cells")
    parser.add_argument("-g", "--n-genes", type=int, default=5_000, help="Number of genes")
    parser.add_argument("-c", "--n-clusters", type=int, default=8, help="Number of clusters")
    parser.add_argument("-o", "--output", default="data/synthetic.h5ad", help="Output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    adata = generate_synthetic(
        n_cells=args.n_cells,
        n_genes=args.n_genes,
        n_clusters=args.n_clusters,
        random_state=args.seed,
    )

    from pathlib import Path
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(str(out))
    logger.info("Written to %s", out)


if __name__ == "__main__":
    main()
