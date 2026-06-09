from __future__ import annotations

import logging

import anndata as ad
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)


def run_pca(
    adata: ad.AnnData,
    n_pcs: int = 50,
    layer: str | None = None,
    random_state: int = 42,
    use_highly_variable: bool = True,
    algorithm: str = "arpack",
) -> ad.AnnData:
    if use_highly_variable and "highly_variable" in adata.var.columns:
        hvg_mask = adata.var["highly_variable"].values
        logger.info("Using %d HVGs for PCA (sparse-safe)", hvg_mask.sum())
        X = adata[:, hvg_mask].X
    else:
        X = adata.X if layer is None else adata.layers[layer]

    n_pcs = min(n_pcs, min(X.shape) - 1)
    if n_pcs < 1:
        raise ValueError(
            f"Cannot run PCA: insufficient data (shape={X.shape}). "
            "Check that QC filtering did not remove all cells/genes."
        )

    if not sparse.issparse(X):
        X = sparse.csr_matrix(np.asarray(X, dtype=np.float64))

    logger.info("Running TruncatedSVD: n_pcs=%d, matrix shape=%s, nnz=%d", n_pcs, X.shape, X.nnz)
    svd = TruncatedSVD(
        n_components=n_pcs,
        algorithm=algorithm,
        random_state=random_state,
    )
    X_pca = svd.fit_transform(X)

    explained_var = svd.explained_variance_
    total_var = np.sum(svd.singular_values_ ** 2)
    explained_var_ratio = explained_var / max(total_var, 1e-12)

    adata.obsm["X_pca"] = X_pca
    adata.uns["pca"] = {
        "variance": explained_var,
        "variance_ratio": explained_var_ratio,
    }

    if use_highly_variable and "highly_variable" in adata.var.columns:
        adata.varm["PCs"] = np.zeros((adata.n_vars, n_pcs))
        adata.varm["PCs"][hvg_mask] = svd.components_.T

    logger.info(
        "TruncatedSVD complete: top-10 variance ratio = %.4f (no densification)",
        explained_var_ratio[:10].sum(),
    )
    return adata
