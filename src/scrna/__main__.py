from __future__ import annotations

import argparse
import logging
import sys

from scrna.pipeline import PipelineConfig, ScRNAPipeline


def main():
    parser = argparse.ArgumentParser(
        description="scRNA-seq analysis pipeline — Numba-accelerated UMAP + Deck.gl visualization",
    )
    parser.add_argument("h5ad_path", help="Path to .h5ad input file")
    parser.add_argument("-o", "--output", default="output/umap_deckgl.json", help="Output JSON path")
    parser.add_argument("--min-genes", type=int, default=200, help="Min genes per cell")
    parser.add_argument("--max-genes", type=int, default=8000, help="Max genes per cell")
    parser.add_argument("--max-pct-mito", type=float, default=20.0, help="Max mitochondrial gene %%")
    parser.add_argument("--n-top-genes", type=int, default=2000, help="Number of HVGs")
    parser.add_argument("--n-pcs", type=int, default=50, help="Number of PCA components")
    parser.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--umap-epochs", type=int, default=500, help="UMAP optimization epochs")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist")
    parser.add_argument("--leiden-resolution", type=float, default=1.0, help="Leiden resolution")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Chunk size for .h5ad loading")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    config = PipelineConfig(
        h5ad_path=args.h5ad_path,
        chunk_size=args.chunk_size,
        min_genes=args.min_genes,
        max_genes=args.max_genes,
        max_pct_mito=args.max_pct_mito,
        n_top_genes=args.n_top_genes,
        n_pcs=args.n_pcs,
        n_neighbors=args.n_neighbors,
        umap_n_epochs=args.umap_epochs,
        umap_min_dist=args.umap_min_dist,
        leiden_resolution=args.leiden_resolution,
        output_json=args.output,
    )

    pipeline = ScRNAPipeline(config)
    result = pipeline.run()

    adata = result["adata"]
    print(f"\n{'=' * 60}")
    print(f"Pipeline completed successfully")
    print(f"  Cells processed:  {adata.n_obs:,}")
    print(f"  Genes retained:   {adata.n_vars:,}")
    print(f"  Clusters found:   {adata.obs['leiden'].nunique()}")
    print(f"  Output JSON:      {config.output_json}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
