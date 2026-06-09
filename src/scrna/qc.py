from __future__ import annotations

import logging

import anndata as ad
import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)

MITO_PREFIXES = ("MT-", "mt-", "Mt-")


def _identify_mito_genes(gene_names: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(gene_names), dtype=bool)
    for prefix in MITO_PREFIXES:
        mask |= np.char.startswith(gene_names.astype(str), prefix)
    return mask


def calculate_qc_metrics(adata: ad.AnnData) -> ad.AnnData:
    gene_names = np.array(adata.var_names, dtype=str)
    mito_mask = _identify_mito_genes(gene_names)
    adata.var["mito"] = mito_mask

    X = adata.X
    if sparse.issparse(X):
        total_counts = np.asarray(X.sum(axis=1)).ravel()
        n_genes_by_counts = np.asarray((X > 0).sum(axis=1)).ravel()
        if mito_mask.any():
            mito_counts = np.asarray(X[:, mito_mask].sum(axis=1)).ravel()
        else:
            mito_counts = np.zeros(adata.n_obs)
    else:
        X_arr = np.asarray(X)
        total_counts = X_arr.sum(axis=1)
        n_genes_by_counts = (X_arr > 0).sum(axis=1)
        if mito_mask.any():
            mito_counts = X_arr[:, mito_mask].sum(axis=1)
        else:
            mito_counts = np.zeros(adata.n_obs)

    adata.obs["total_counts"] = total_counts
    adata.obs["n_genes_by_counts"] = n_genes_by_counts
    adata.obs["pct_counts_mito"] = (
        mito_counts / np.maximum(total_counts, 1.0) * 100.0
    )

    logger.info(
        "QC metrics: total_counts median=%.0f, n_genes median=%.0f, "
        "pct_mito median=%.1f%%",
        np.median(total_counts),
        np.median(n_genes_by_counts),
        np.median(adata.obs["pct_counts_mito"]),
    )
    return adata


def filter_cells(
    adata: ad.AnnData,
    min_genes: int = 200,
    max_genes: int | None = 8000,
    min_counts: int | None = None,
    max_counts: int | None = None,
    max_pct_mito: float = 20.0,
) -> ad.AnnData:
    if "pct_counts_mito" not in adata.obs.columns:
        adata = calculate_qc_metrics(adata)

    mask = np.ones(adata.n_obs, dtype=bool)
    mask &= adata.obs["n_genes_by_counts"].values >= min_genes
    if max_genes is not None:
        mask &= adata.obs["n_genes_by_counts"].values <= max_genes
    if min_counts is not None:
        mask &= adata.obs["total_counts"].values >= min_counts
    if max_counts is not None:
        mask &= adata.obs["total_counts"].values <= max_counts
    mask &= adata.obs["pct_counts_mito"].values <= max_pct_mito

    n_removed = adata.n_obs - mask.sum()
    adata = adata[mask].copy()
    logger.info(
        "Filtered cells: removed %d (%.1f%%), kept %d",
        n_removed,
        n_removed / (n_removed + adata.n_obs) * 100 if (n_removed + adata.n_obs) > 0 else 0,
        adata.n_obs,
    )
    return adata


def filter_genes(
    adata: ad.AnnData,
    min_cells: int = 3,
) -> ad.AnnData:
    if sparse.issparse(adata.X):
        n_cells_per_gene = np.asarray((adata.X > 0).sum(axis=0)).ravel()
    else:
        n_cells_per_gene = (np.asarray(adata.X) > 0).sum(axis=0)

    gene_mask = n_cells_per_gene >= min_cells
    n_removed = adata.n_vars - gene_mask.sum()
    adata = adata[:, gene_mask].copy()
    logger.info("Filtered genes: removed %d, kept %d", n_removed, adata.n_vars)
    return adata
