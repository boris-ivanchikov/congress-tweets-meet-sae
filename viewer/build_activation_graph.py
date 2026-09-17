"""Build a sparse, clustered activation co-occurrence graph for the viewer."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import networkx as nx
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-dir", default="viewer/backend_data")
    parser.add_argument("--manifest", default="viewer/data/manifest.json")
    parser.add_argument("--output", default="viewer/data/activation_graph.json")
    parser.add_argument("--sample-per-activation", type=int, default=1024)
    parser.add_argument("--neighbors", type=int, default=30)
    parser.add_argument("--edge-mode", choices=("presence", "activation"), default="presence")
    parser.add_argument("--layout-iterations", type=int, default=90)
    parser.add_argument("--seed", type=int, default=73)
    return parser.parse_args()


def row_entry_positions(rows: np.ndarray, indptr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.asarray(indptr[rows], dtype=np.int64)
    ends = np.asarray(indptr[rows + 1], dtype=np.int64)
    counts = ends - starts
    total = int(counts.sum())
    if not total:
        return np.empty(0, dtype=np.int64), counts
    group_starts = np.repeat(starts, counts)
    local_starts = np.repeat(np.cumsum(counts) - counts, counts)
    positions = group_starts + np.arange(total, dtype=np.int64) - local_starts
    return positions, counts


def activation_moments(indices: np.ndarray, values: np.ndarray, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    totals = np.zeros(n_features, dtype=np.float64)
    squares = np.zeros(n_features, dtype=np.float64)
    chunk_size = 10_000_000
    for start in range(0, len(indices), chunk_size):
        end = min(len(indices), start + chunk_size)
        chunk_indices = np.asarray(indices[start:end])
        chunk_values = np.asarray(values[start:end], dtype=np.float64)
        totals += np.bincount(chunk_indices, weights=chunk_values, minlength=n_features)
        squares += np.bincount(chunk_indices, weights=chunk_values * chunk_values, minlength=n_features)
    return totals, squares


def candidate_edges(
    row_indptr: np.ndarray,
    row_indices: np.ndarray,
    feature_indptr: np.ndarray,
    feature_rows: np.ndarray,
    row_values: np.ndarray | None,
    activation_totals: np.ndarray | None,
    activation_squares: np.ndarray | None,
    sample_size: int,
    neighbors: int,
    seed: int,
) -> tuple[np.ndarray, dict[tuple[int, int], tuple[float, float, int]]]:
    n_features = len(feature_indptr) - 1
    n_tweets = len(row_indptr) - 1
    frequencies = np.diff(feature_indptr).astype(np.int64)
    edges: dict[tuple[int, int], tuple[float, float, int]] = {}

    for feature in range(n_features):
        start, end = int(feature_indptr[feature]), int(feature_indptr[feature + 1])
        rows = np.asarray(feature_rows[start:end])
        if not len(rows):
            continue
        if len(rows) > sample_size:
            rng = np.random.default_rng(seed + feature)
            rows = np.sort(rng.choice(rows, size=sample_size, replace=False))
        positions, counts = row_entry_positions(rows, row_indptr)
        cofeatures = np.asarray(row_indices[positions])
        observed = np.bincount(cofeatures, minlength=n_features).astype(np.float64)
        observed[feature] = 0

        # Estimate corpus intersections from a uniform sample of tweets carrying
        # this activation. Cosine is the normalized co-occurrence strength; a mild
        # lift adjustment separates meaningful associations from ubiquitous generic
        # activations, and confidence suppresses coincidences seen only once or twice.
        intersection = observed * frequencies[feature] / len(rows)
        intersection = np.minimum(intersection, np.minimum(frequencies[feature], frequencies))
        if row_values is None:
            denominator = np.sqrt(np.maximum(frequencies[feature] * frequencies, 1))
            cosine = intersection / denominator
            lift = intersection * n_tweets / np.maximum(frequencies[feature] * frequencies, 1)
        else:
            coactivation_values = np.asarray(row_values[positions], dtype=np.float64)
            group_ids = np.repeat(np.arange(len(rows), dtype=np.int32), counts)
            source_values = np.bincount(
                group_ids,
                weights=coactivation_values * (cofeatures == feature),
                minlength=len(rows),
            )
            products = coactivation_values * source_values[group_ids]
            dot_product = np.bincount(cofeatures, weights=products, minlength=n_features)
            dot_product *= frequencies[feature] / len(rows)
            dot_product[feature] = 0
            denominator = np.sqrt(np.maximum(activation_squares[feature] * activation_squares, 1e-12))
            cosine = dot_product / denominator
            lift = dot_product * n_tweets / np.maximum(
                activation_totals[feature] * activation_totals,
                1e-12,
            )
        confidence = observed / (observed + 4.0)
        association = cosine * np.sqrt(np.log1p(np.maximum(lift, 0))) * confidence
        association[observed < (1 if len(rows) <= 20 else 3)] = 0

        candidates = np.flatnonzero(association > 0)
        if len(candidates) > neighbors:
            candidates = candidates[np.argpartition(association[candidates], -neighbors)[-neighbors:]]
        candidates = candidates[np.argsort(association[candidates])[::-1]]
        for other in candidates:
            pair = (feature, int(other)) if feature < other else (int(other), feature)
            strength = float(association[other])
            overlap = float(intersection[other])
            previous = edges.get(pair)
            if previous is None:
                edges[pair] = (strength, overlap, 1)
            else:
                count = previous[2] + 1
                edges[pair] = (
                    max(previous[0], strength),
                    (previous[1] * previous[2] + overlap) / count,
                    count,
                )
        if feature and feature % 512 == 0:
            print(f"Co-occurrence candidates: {feature:,}/{n_features:,}", flush=True)
    return frequencies, edges


def force_layout(
    n_features: int,
    edges: dict[tuple[int, int], tuple[float, float, int]],
    seed: int,
    iterations: int,
) -> tuple[dict[int, int], dict[int, tuple[float, float]], dict[int, int]]:
    """Detect communities for color and independently lay out the full graph."""
    graph = nx.Graph()
    graph.add_nodes_from(range(n_features))
    graph.add_weighted_edges_from(
        (left, right, values[0]) for (left, right), values in edges.items()
    )
    nonisolated = graph.subgraph([node for node in graph if graph.degree(node)]).copy()
    largest_component = max(nx.connected_components(nonisolated), key=len)
    connected = nonisolated.subgraph(largest_component).copy()
    communities = list(nx.community.louvain_communities(connected, weight="weight", seed=seed))
    communities.sort(key=lambda group: (-len(group), min(group)))
    cluster_for: dict[int, int] = {}
    for cluster, community in enumerate(communities):
        for node in community:
            cluster_for[node] = cluster
    layout_graph = connected.copy()
    for _, _, data in layout_graph.edges(data=True):
        data["layout_weight"] = float(data["weight"]) ** 1.6
    positions = nx.spring_layout(
        layout_graph,
        weight="layout_weight",
        seed=seed,
        iterations=iterations,
        method="energy",
        gravity=1.0,
        k=2.0 / math.sqrt(len(layout_graph)),
        scale=100,
    )
    return (
        cluster_for,
        {node: tuple(np.asarray(point, dtype=float).tolist()) for node, point in positions.items()},
        dict(connected.degree()),
    )


def main() -> None:
    args = parse_args()
    backend = Path(args.backend_dir)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    metadata = {int(item["id"]) - 1: item for item in manifest["features"]}
    row_indptr = np.load(backend / "activation_indptr.npy", mmap_mode="r")
    row_indices = np.load(backend / "activation_indices.npy", mmap_mode="r")
    row_values = np.load(backend / "activation_values.npy", mmap_mode="r") if args.edge_mode == "activation" else None
    feature_indptr = np.load(backend / "activation_tweet_indptr.npy", mmap_mode="r")
    feature_rows = np.load(backend / "activation_tweet_indices.npy", mmap_mode="r")

    activation_totals = activation_squares = None
    if row_values is not None:
        print("Computing activation magnitude moments…", flush=True)
        activation_totals, activation_squares = activation_moments(
            row_indices, row_values, len(feature_indptr) - 1
        )
    frequencies, edges = candidate_edges(
        row_indptr, row_indices, feature_indptr, feature_rows,
        row_values, activation_totals, activation_squares,
        args.sample_per_activation, args.neighbors, args.seed,
    )
    # Retain mutual nearest-neighbor bonds only. A one-sided top-neighbor relation is
    # often caused by a very broad activation and creates long, misleading edges;
    # mutuality yields a substantially cleaner semantic graph.
    edges = {pair: values for pair, values in edges.items() if values[2] == 2}
    cluster_for, positions, degrees = force_layout(
        len(frequencies), edges, args.seed, args.layout_iterations
    )
    maximum = max((values[0] for values in edges.values()), default=1.0)
    connected_nodes = sorted(positions)
    connected_set = set(connected_nodes)
    payload = {
        "schema_version": 2,
        "method": {
            "name": "sampled weighted co-occurrence graph",
            "sample_per_activation": args.sample_per_activation,
            "neighbors": args.neighbors,
            "edge_mode": args.edge_mode,
            "mutual_neighbors": True,
            "edge_weight": (
                "activation-weighted cosine × sqrt(log(1 + weighted lift)) × sampling confidence"
                if args.edge_mode == "activation"
                else "presence cosine × sqrt(log(1 + lift)) × sampling confidence"
            ),
            "layout": "global weighted force-directed layout (independent of communities)",
        },
        "nodes": [
            {
                "id": index + 1,
                "label": (
                    metadata.get(index, {}).get("broad_label")
                    or metadata.get(index, {}).get("core_label")
                    or "Uninterpreted activation"
                ),
                "cluster": cluster_for[index],
                "x": round(float(positions[index][0]), 4),
                "y": round(float(positions[index][1]), 4),
                "n": int(frequencies[index]),
                "degree": int(degrees[index]),
            }
            for index in connected_nodes
        ],
        "edges": [
            {
                "source": left + 1,
                "target": right + 1,
                "weight": round(values[0] / maximum, 6),
                "cooccurrence": int(round(values[1])),
            }
            for (left, right), values in sorted(edges.items())
            if left in connected_set and right in connected_set
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    temporary.replace(output)
    print(
        f"Wrote {output}: {len(payload['nodes']):,} nodes, "
        f"{len(payload['edges']):,} edges, "
        f"{len(set(cluster_for.values()) - {-1}):,} clusters",
        flush=True,
    )


if __name__ == "__main__":
    main()
