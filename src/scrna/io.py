from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator

import anndata as ad
import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)


def load_h5ad(
    path: str | Path,
    backed: str = "r",
) -> ad.AnnData:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f".h5ad file not found: {path}")
    logger.info("Loading .h5ad from %s (backed=%s)", path, backed)
    adata = ad.read_h5ad(path, backed=backed)
    logger.info("Loaded AnnData: %s cells x %s genes", adata.n_obs, adata.n_vars)
    return adata


def chunked_load(
    path: str | Path,
    chunk_size: int = 50_000,
) -> Generator[ad.AnnData, None, None]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f".h5ad file not found: {path}")

    logger.info("Opening .h5ad for chunked read: %s", path)
    adata_backed = ad.read_h5ad(path, backed="r")

    n_obs = adata_backed.n_obs
    for start in range(0, n_obs, chunk_size):
        end = min(start + chunk_size, n_obs)
        logger.debug("Reading chunk [%d:%d]", start, end)
        chunk = adata_backed[start:end].to_memory()
        yield chunk

    adata_backed.file.close()


def load_to_memory(
    path: str | Path,
    chunk_size: int = 50_000,
) -> ad.AnnData:
    path = Path(path)
    chunks = list(chunked_load(path, chunk_size=chunk_size))
    if not chunks:
        raise ValueError("No data chunks read from file")

    if len(chunks) == 1:
        return chunks[0]

    logger.info("Concatenating %d chunks ...", len(chunks))
    combined = ad.concat(chunks, merge="same")
    return combined


def ensure_sparse(adata: ad.AnnData) -> ad.AnnData:
    if not sparse.issparse(adata.X):
        logger.info("Converting dense matrix to CSR sparse format")
        adata.X = sparse.csr_matrix(adata.X)
    return adata


def get_gene_names(adata: ad.AnnData) -> np.ndarray:
    if adata.var_names is not None:
        return np.array(adata.var_names, dtype=str)
    if "gene_ids" in adata.var:
        return np.array(adata.var["gene_ids"], dtype=str)
    return np.array([f"gene_{i}" for i in range(adata.n_vars)], dtype=str)


def get_cell_ids(adata: ad.AnnData) -> np.ndarray:
    if adata.obs_names is not None:
        return np.array(adata.obs_names, dtype=str)
    return np.array([f"cell_{i}" for i in range(adata.n_obs)], dtype=str)
