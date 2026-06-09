from __future__ import annotations

import logging

import anndata as ad
import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)


def highly_variable_genes(
    adata: ad.AnnData,
    n_top_genes: int = 2000,
    flavor: str = "seurat_v3",
    batch_key: str | None = None,
    span: float = 0.3,
) -> ad.AnnData:
    X = adata.X
    if sparse.issparse(X):
        X_dense = np.asarray(X.tocsr()[:, :].todense())
    else:
        X_dense = np.asarray(X, dtype=np.float64)

    if flavor == "seurat_v3":
        mean_expr = X_dense.mean(axis=0)
        var_expr = X_dense.var(axis=0)

        mean_expr = np.maximum(mean_expr, 1e-12)
        variance_std = np.sqrt(var_expr)

        expected_var = _loess_variance(mean_expr, variance_std, span=span)
        clip_val = np.maximum(expected_var, 1e-12)
        variance_std_clipped = np.minimum(variance_std, clip_val * 10)

        cv = variance_std_clipped / np.maximum(mean_expr, 1e-12)

        sorted_indices = np.argsort(-cv)
        hvg_mask = np.zeros(adata.n_vars, dtype=bool)
        n_select = min(n_top_genes, adata.n_vars)
        hvg_mask[sorted_indices[:n_select]] = True

    else:
        mean_expr = X_dense.mean(axis=0)
        var_expr = X_dense.var(axis=0)
        dispersion = var_expr / np.maximum(mean_expr, 1e-12)

        sorted_indices = np.argsort(-dispersion)
        hvg_mask = np.zeros(adata.n_vars, dtype=bool)
        n_select = min(n_top_genes, adata.n_vars)
        hvg_mask[sorted_indices[:n_select]] = True

    adata.var["highly_variable"] = hvg_mask
    adata.var["means"] = mean_expr
    adata.var["variances"] = var_expr

    n_hvg = hvg_mask.sum()
    logger.info("Selected %d highly variable genes (flavor=%s)", n_hvg, flavor)
    return adata


def _loess_variance(
    mean_expr: np.ndarray,
    variance_std: np.ndarray,
    span: float = 0.3,
) -> np.ndarray:
    n = len(mean_expr)
    window = max(int(n * span), 10)

    order = np.argsort(mean_expr)
    sorted_mean = mean_expr[order]
    sorted_var = variance_std[order]

    result = np.empty(n, dtype=np.float64)

    for i in range(n):
        left = max(0, i - window // 2)
        right = min(n, i + window // 2 + 1)
        local_mean = sorted_mean[left:right]
        local_var = sorted_var[left:right]
        weights = np.exp(-0.5 * ((local_mean - sorted_mean[i]) / (np.std(local_mean) + 1e-12)) ** 2)
        weights /= weights.sum() + 1e-12
        result[i] = np.dot(weights, local_var)

    unsort_order = np.argsort(order)
    return result[unsort_order]


def subset_to_hvg(adata: ad.AnnData) -> ad.AnnData:
    if "highly_variable" not in adata.var.columns:
        raise ValueError("Run highly_variable_genes() first")

    hvg_mask = adata.var["highly_variable"].values
    adata = adata[:, hvg_mask].copy()
    logger.info("Subset to %d HVGs", adata.n_vars)
    return adata
