from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import anndata as ad
import numpy as np

from scrna.cluster import leiden_cluster
from scrna.export import export_deckgl_json
from scrna.hvg import highly_variable_genes, subset_to_hvg
from scrna.io import ensure_sparse, load_to_memory, get_cell_ids
from scrna.normalize import log_normalize
from scrna.pca import run_pca
from scrna.qc import calculate_qc_metrics, filter_cells, filter_genes
from scrna.umap_numba import umap_numba

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    h5ad_path: str = ""
    chunk_size: int = 50_000

    min_genes: int = 200
    max_genes: int = 8000
    max_pct_mito: float = 20.0
    min_cells_per_gene: int = 3

    target_sum: float = 1e4
    n_top_genes: int = 2000
    hvg_flavor: str = "seurat_v3"

    n_pcs: int = 50
    pca_random_state: int = 42

    n_neighbors: int = 15
    umap_n_epochs: int = 500
    umap_min_dist: float = 0.1
    umap_learning_rate: float = 1.0
    umap_random_state: int = 42

    leiden_resolution: float = 1.0

    output_json: str = "output/umap_deckgl.json"


class ScRNAPipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.adata: ad.AnnData | None = None
        self.embedding: np.ndarray | None = None
        self._timings: dict[str, float] = field(default_factory=dict)

    def _step(self, name: str):
        return _StepTimer(name, self._timings)

    def run(self, h5ad_path: str | None = None) -> dict:
        if h5ad_path is not None:
            self.config.h5ad_path = h5ad_path

        if not self.config.h5ad_path:
            raise ValueError("h5ad_path must be provided")

        self._timings = {}

        with self._step("load"):
            logger.info("=" * 60)
            logger.info("STEP 1: Loading .h5ad data")
            logger.info("=" * 60)
            self.adata = load_to_memory(
                self.config.h5ad_path,
                chunk_size=self.config.chunk_size,
            )
            self.adata = ensure_sparse(self.adata)

        with self._step("qc"):
            logger.info("=" * 60)
            logger.info("STEP 2: Quality control")
            logger.info("=" * 60)
            self.adata = calculate_qc_metrics(self.adata)
            self.adata = filter_cells(
                self.adata,
                min_genes=self.config.min_genes,
                max_genes=self.config.max_genes,
                max_pct_mito=self.config.max_pct_mito,
            )
            self.adata = filter_genes(
                self.adata,
                min_cells=self.config.min_cells_per_gene,
            )

        with self._step("normalize"):
            logger.info("=" * 60)
            logger.info("STEP 3: Log-normalization")
            logger.info("=" * 60)
            self.adata = log_normalize(self.adata, target_sum=self.config.target_sum)

        with self._step("hvg"):
            logger.info("=" * 60)
            logger.info("STEP 4: Highly variable gene selection")
            logger.info("=" * 60)
            self.adata = highly_variable_genes(
                self.adata,
                n_top_genes=self.config.n_top_genes,
                flavor=self.config.hvg_flavor,
            )
            self.adata = subset_to_hvg(self.adata)

        with self._step("pca"):
            logger.info("=" * 60)
            logger.info("STEP 5: PCA")
            logger.info("=" * 60)
            self.adata = run_pca(
                self.adata,
                n_pcs=self.config.n_pcs,
                random_state=self.config.pca_random_state,
            )

        with self._step("umap"):
            logger.info("=" * 60)
            logger.info("STEP 6: UMAP (Numba-accelerated)")
            logger.info("=" * 60)
            X_pca = self.adata.obsm["X_pca"]
            self.embedding = umap_numba(
                X_pca,
                n_neighbors=self.config.n_neighbors,
                n_epochs=self.config.umap_n_epochs,
                min_dist=self.config.umap_min_dist,
                learning_rate=self.config.umap_learning_rate,
                random_state=self.config.umap_random_state,
            )
            self.adata.obsm["X_umap"] = self.embedding

        with self._step("cluster"):
            logger.info("=" * 60)
            logger.info("STEP 7: Leiden clustering")
            logger.info("=" * 60)
            self.adata = leiden_cluster(
                self.adata,
                n_neighbors=self.config.n_neighbors,
                resolution=self.config.leiden_resolution,
            )

        with self._step("export"):
            logger.info("=" * 60)
            logger.info("STEP 8: Exporting Deck.gl JSON")
            logger.info("=" * 60)
            cell_ids = get_cell_ids(self.adata)
            pct_mito = self.adata.obs["pct_counts_mito"].values if "pct_counts_mito" in self.adata.obs else None
            n_genes = self.adata.obs["n_genes_by_counts"].values if "n_genes_by_counts" in self.adata.obs else None
            total_counts = self.adata.obs["total_counts"].values if "total_counts" in self.adata.obs else None
            cluster_labels = self.adata.obs["leiden"].values

            export_deckgl_json(
                embedding=self.embedding,
                cluster_labels=cluster_labels,
                output_path=self.config.output_json,
                cell_ids=cell_ids,
                pct_mito=pct_mito,
                n_genes=n_genes,
                total_counts=total_counts,
            )

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        for step, elapsed in self._timings.items():
            logger.info("  %-15s %.2f s", step, elapsed)

        return {
            "adata": self.adata,
            "embedding": self.embedding,
            "timings": self._timings,
        }


class _StepTimer:
    def __init__(self, name: str, timings: dict):
        self.name = name
        self.timings = timings

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        self.timings[self.name] = elapsed
        logger.info("  [Timer] %s: %.2f s", self.name, elapsed)
