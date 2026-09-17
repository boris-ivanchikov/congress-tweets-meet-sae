#!/usr/bin/env python3
"""Score a graph layout's agreement with Louvain colors and its geometric shape."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layout")
    parser.add_argument("--graph", default="viewer/data/activation_graph.json")
    parser.add_argument("--neighbors", type=int, default=10)
    args = parser.parse_args()

    graph = json.loads(Path(args.graph).read_text())
    layout = json.loads(Path(args.layout).read_text())
    ids = np.array([node["id"] for node in graph["nodes"]], dtype=np.int64)
    clusters = np.array([node["cluster"] for node in graph["nodes"]], dtype=np.int64)
    points = np.array([layout["positions"][str(index)] for index in ids], dtype=np.float64)
    points -= np.median(points, axis=0)
    distances, neighbors = cKDTree(points).query(points, k=args.neighbors + 1)
    neighbor_clusters = clusters[neighbors[:, 1:]]
    purity = float(np.mean(neighbor_clusters == clusters[:, None]))
    emergent_communities = layout.get("communities")
    if emergent_communities:
        emergent = np.array([
            emergent_communities[str(int(node_id))] for node_id in ids
        ], dtype=np.int64)
        emergent_purity = float(np.mean(emergent[neighbors[:, 1:]] == emergent[:, None]))
        community_count = int(len(set(emergent.tolist())))
    else:
        emergent_purity = None
        community_count = None

    id_to_position = {int(node_id): index for index, node_id in enumerate(ids)}
    edge_pairs = np.array([
        [id_to_position[int(edge["source"])], id_to_position[int(edge["target"])]]
        for edge in graph["edges"]
    ], dtype=np.int64)
    raw_weights = np.array([
        max(float(edge.get("weight", 0.0)), 1e-12) for edge in graph["edges"]
    ])
    log_weights = np.log(raw_weights)
    low_weight, high_weight = np.quantile(log_weights, [0.08, 0.98])
    visual_weights = np.clip(
        (log_weights - low_weight) / max(high_weight - low_weight, 1e-12),
        0.03,
        1.0,
    )
    edge_lengths = np.linalg.norm(
        points[edge_pairs[:, 0]] - points[edge_pairs[:, 1]], axis=1,
    )
    median_edge_length = float(np.median(edge_lengths))
    strong = visual_weights >= np.quantile(visual_weights, 0.90)
    strong_median = float(np.median(edge_lengths[strong]))
    strong_p95 = float(np.quantile(edge_lengths[strong], 0.95))
    weighted_mean = float(np.average(edge_lengths, weights=visual_weights))
    ordinary_mean = float(np.mean(edge_lengths))

    adjacency = [set() for _ in ids]
    for source, target in edge_pairs:
        adjacency[source].add(target)
        adjacency[target].add(source)
    direct_neighbor_hits = []
    neighbor_recall = []
    for index, spatial_neighbors in enumerate(neighbors[:, 1:]):
        linked = adjacency[index]
        if not linked:
            continue
        hits = len(linked.intersection(map(int, spatial_neighbors)))
        direct_neighbor_hits.append(hits / args.neighbors)
        neighbor_recall.append(hits / min(args.neighbors, len(linked)))

    radii = np.linalg.norm(points, axis=1)
    radius_ratio = float(np.quantile(radii, 0.99) / max(np.median(radii), 1e-12))
    local_scale = float(np.median(distances[:, 1:]))
    diameter = float(np.linalg.norm(np.ptp(points, axis=0)))
    fill = float(local_scale / max(diameter, 1e-12))
    # Reward community-local neighborhoods; gently penalize tendrils and layouts
    # whose useful local scale becomes microscopic relative to their diameter.
    score = purity - 0.025 * max(0.0, radius_ratio - 3.0) - 0.5 * max(0.0, 0.0006 - fill)
    print(json.dumps({
        "layout": args.layout,
        "settings": layout.get("settings", {}),
        "elapsed_seconds": layout.get("elapsed_seconds"),
        "neighbor_purity": round(purity, 5),
        "emergent_community_count": community_count,
        "emergent_neighbor_purity": round(emergent_purity, 5) if emergent_purity is not None else None,
        "emergent_modularity": round(float(layout["modularity"]), 5) if layout.get("modularity") is not None else None,
        "spatial_neighbor_edge_precision": round(float(np.mean(direct_neighbor_hits)), 5),
        "spatial_neighbor_edge_recall": round(float(np.mean(neighbor_recall)), 5),
        "strong_edge_median_over_all": round(strong_median / max(median_edge_length, 1e-12), 5),
        "strong_edge_p95_over_all_median": round(strong_p95 / max(median_edge_length, 1e-12), 5),
        "weighted_edge_length_over_mean": round(weighted_mean / max(ordinary_mean, 1e-12), 5),
        "p99_median_radius": round(radius_ratio, 4),
        "local_scale_over_diameter": round(fill, 7),
        "score": round(score, 5),
    }))


if __name__ == "__main__":
    main()
