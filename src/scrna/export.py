from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    "#5254a3", "#6b6ecf", "#b5cf6b", "#e7ba52", "#d6616b",
]


def _get_color_for_cluster(cluster_id: int, n_clusters: int) -> str:
    if n_clusters <= len(PALETTE):
        return PALETTE[cluster_id % len(PALETTE)]
    hue = (cluster_id * 137.508) % 360
    s = 0.7
    l = 0.55
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((hue / 60) % 2 - 1))
    m = l - c / 2

    if hue < 60:
        r, g, b = c, x, 0
    elif hue < 120:
        r, g, b = x, c, 0
    elif hue < 180:
        r, g, b = 0, c, x
    elif hue < 240:
        r, g, b = 0, x, c
    elif hue < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    ri = int((r + m) * 255)
    gi = int((g + m) * 255)
    bi = int((b + m) * 255)
    return f"#{ri:02x}{gi:02x}{bi:02x}"


def export_deckgl_json(
    embedding: np.ndarray,
    cluster_labels: np.ndarray | list[str],
    output_path: str | Path,
    cell_ids: np.ndarray | list[str] | None = None,
    pct_mito: np.ndarray | None = None,
    n_genes: np.ndarray | None = None,
    total_counts: np.ndarray | None = None,
    pseudotime: np.ndarray | None = None,
    trajectory: dict | None = None,
) -> Path:
    output_path = Path(output_path)
    n_cells = embedding.shape[0]

    unique_clusters = sorted(set(str(c) for c in cluster_labels))
    n_clusters = len(unique_clusters)
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    color_map = {c: _get_color_for_cluster(i, n_clusters) for i, c in enumerate(unique_clusters)}

    logger.info("Exporting %d cells, %d clusters to Deck.gl JSON", n_cells, n_clusters)

    points = []
    for i in range(n_cells):
        point = {
            "position": [float(embedding[i, 0]), float(embedding[i, 1])],
            "cluster": str(cluster_labels[i]),
            "color": color_map[str(cluster_labels[i])],
        }
        if cell_ids is not None:
            point["cell_id"] = str(cell_ids[i])
        if pct_mito is not None:
            point["pct_mito"] = float(pct_mito[i])
        if n_genes is not None:
            point["n_genes"] = int(n_genes[i])
        if total_counts is not None:
            point["total_counts"] = float(total_counts[i])
        if pseudotime is not None:
            point["pseudotime"] = float(pseudotime[i])
        points.append(point)

    payload = {
        "metadata": {
            "n_cells": n_cells,
            "n_clusters": n_clusters,
            "clusters": [
                {"id": c, "label": c, "color": color_map[c]}
                for c in unique_clusters
            ],
        },
        "points": points,
    }

    if trajectory is not None:
        payload["trajectory"] = trajectory
        logger.info("  Trajectory data: %d MST edges, root='%s'",
                     len(trajectory["mst_edges"]), trajectory["root_cluster"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Exported Deck.gl JSON: %s (%.1f MB)", output_path, size_mb)
    return output_path


def export_parquet(
    embedding: np.ndarray,
    cluster_labels: np.ndarray | list[str],
    output_path: str | Path,
    cell_ids: np.ndarray | list[str] | None = None,
) -> Path:
    import pandas as pd

    output_path = Path(output_path)
    n_cells = embedding.shape[0]

    df = pd.DataFrame({
        "x": embedding[:, 0],
        "y": embedding[:, 1],
        "cluster": [str(c) for c in cluster_labels],
    })
    if cell_ids is not None:
        df["cell_id"] = [str(c) for c in cell_ids]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Exported Parquet: %s (%.1f MB)", output_path, size_mb)
    return output_path
