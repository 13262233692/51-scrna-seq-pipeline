from __future__ import annotations

import logging

import anndata as ad
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def run_pca(
    adata: ad.AnnData,
    n_pcs: int = 50,
    layer: str | None = None,
    svd_solver: str = "arpack",
    random_state: int = 42,
    use_highly_variable: bool = True,
) -> ad.AnnData:
    if use_highly_variable and "highly_variable" in adata.var.columns:
        hvg_mask = adata.var["highly_variable"].values
        logger.info("Using %d HVGs for PCA", hvg_mask.sum())
        X = adata[:, hvg_mask].X
    else:
        X = adata.X if layer is None else adata.layers[layer]

    if sparse.issparse(X):
        X_input = X.toarray()
    else:
        X_input = np.asarray(X, dtype=np.float64)

    n_pcs = min(n_pcs, min(X_input.shape) - 1)
    if n_pcs < 1:
        raise ValueError(
            f"Cannot run PCA: insufficient data (shape={X_input.shape}). "
            "Check that QC filtering did not remove all cells/genes."
        )

    logger.info("Running PCA: n_pcs=%d, matrix shape=%s", n_pcs, X_input.shape)
    pca = PCA(
        n_components=n_pcs,
        svd_solver=svd_solver,
        random_state=random_state,
    )
    X_pca = pca.fit_transform(X_input)

    adata.obsm["X_pca"] = X_pca
    adata.uns["pca"] = {
        "variance": pca.explained_variance_,
        "variance_ratio": pca.explained_variance_ratio_,
    }

    if use_highly_variable and "highly_variable" in adata.var.columns:
        hvg_names = adata.var_names[hvg_mask]
        adata.varm["PCs"] = np.zeros((adata.n_vars, n_pcs))
        adata.varm["PCs"][hvg_mask] = pca.components_.T

    logger.info(
        "PCA complete: top-10 variance ratio = %.4f",
        pca.explained_variance_ratio_[:10].sum(),
    )
    return adata
