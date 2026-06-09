from __future__ import annotations

import logging

import anndata as ad
import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)


def log_normalize(
    adata: ad.AnnData,
    target_sum: float = 1e4,
    layer: str | None = None,
    inplace: bool = True,
) -> ad.AnnData | np.ndarray:
    if not inplace:
        adata = adata.copy()

    X = adata.X if layer is None else adata.layers[layer]

    if sparse.issparse(X):
        X = X.tocsr()
        counts_per_cell = np.asarray(X.sum(axis=1)).ravel()
        size_factors = target_sum / np.maximum(counts_per_cell, 1.0)

        for i in range(X.shape[0]):
            start, end = X.indptr[i], X.indptr[i + 1]
            X.data[start:end] *= size_factors[i]

        X.data = np.log1p(X.data)
        adata.X = X
    else:
        X_arr = np.asarray(X, dtype=np.float64)
        counts_per_cell = X_arr.sum(axis=1, keepdims=True)
        size_factors = target_sum / np.maximum(counts_per_cell, 1.0)
        X_arr = X_arr * size_factors
        X_arr = np.log1p(X_arr)
        adata.X = sparse.csr_matrix(X_arr) if sparse.issparse(X) else X_arr

    logger.info("Log-normalization complete (target_sum=%.0f)", target_sum)
    return adata


def normalize_total(
    adata: ad.AnnData,
    target_sum: float = 1e4,
    inplace: bool = True,
) -> ad.AnnData:
    if not inplace:
        adata = adata.copy()

    X = adata.X
    if sparse.issparse(X):
        X = X.tocsr()
        counts_per_cell = np.asarray(X.sum(axis=1)).ravel()
        size_factors = target_sum / np.maximum(counts_per_cell, 1.0)
        for i in range(X.shape[0]):
            start, end = X.indptr[i], X.indptr[i + 1]
            X.data[start:end] *= size_factors[i]
        adata.X = X
    else:
        X_arr = np.asarray(X, dtype=np.float64)
        counts_per_cell = X_arr.sum(axis=1, keepdims=True)
        size_factors = target_sum / np.maximum(counts_per_cell, 1.0)
        adata.X = X_arr * size_factors

    logger.info("Total-count normalization complete (target_sum=%.0f)", target_sum)
    return adata
