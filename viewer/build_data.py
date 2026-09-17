"""Build the compact, static dataset consumed by the Hugging Face viewer.

This runs locally. The hosted Space only receives interpretations and the
selected tweet examples; it never needs the full tweet corpus, SAE weights, or
activation matrix.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
TWEET_COLUMNS = ["tweet_id", "posted_at", "name", "twitter", "text"]
INTERPRETATION_FIELDS = [
    "model",
    "core_label",
    "core_explanation",
    "broad_label",
    "broad_explanation",
    "relation",
    "tail_coherence",
]
SCORE_FIELDS = [
    "score_core",
    "score_broad",
    *(f"score_core_q{q}" for q in range(1, 5)),
    *(f"score_broad_q{q}" for q in range(1, 5)),
    "score_neg_core",
    "score_neg_broad",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a compact static dataset for the SAE activation viewer."
    )
    parser.add_argument("--path", required=True, help="Trained SAE run directory")
    parser.add_argument("--interpretations", required=True, help="Autointerp CSV")
    parser.add_argument("--tweets", default="data/tweets.csv", help="Tweet CSV")
    parser.add_argument("--out-dir", default="viewer/data", help="Generated data directory")
    parser.add_argument("--model", default=None, help="Keep only this interpreter model")
    parser.add_argument("--features", nargs="+", type=int, default=None,
                        help="Optional activation-index subset for a test build")
    parser.add_argument("--n-top", type=int, default=10)
    parser.add_argument("--n-random", type=int, default=10)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--tweet-chunksize", type=int, default=200_000)
    return parser.parse_args()


def clean_value(value):
    """Convert pandas/numpy scalars to strict JSON values."""
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return str(value)


def viewer_language(value):
    """Keep generated prose consistent with the viewer's core/broad terminology."""
    text = clean_value(value)
    if text is None:
        return None
    text = re.sub(
        r"\b(?:broader|broad|activation|semantic) envelope\b",
        lambda match: "Broad interpretation" if match.group(0)[0].isupper() else "broad interpretation",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\benvelope\b", "interpretation", text, flags=re.IGNORECASE)


def load_interpretations(path: Path, model: str | None) -> dict[int, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Interpretations not found: {path}")

    df = pd.read_csv(path)
    if "feature" not in df:
        raise ValueError(f"{path} has no 'feature' column")
    if model is not None:
        if "model" not in df:
            raise ValueError("--model was supplied but the CSV has no 'model' column")
        df = df[df["model"] == model]
    df = df.drop_duplicates("feature", keep="last")

    records = {}
    for row in df.to_dict("records"):
        feature = int(row["feature"])
        record = {field: clean_value(row.get(field)) for field in INTERPRETATION_FIELDS}
        record["scores"] = {field: clean_value(row.get(field)) for field in SCORE_FIELDS}
        records[feature] = record
    return records


def ordered_positions(values: np.ndarray, positions: np.ndarray) -> list[int]:
    if len(positions) == 0:
        return []
    order = np.lexsort((positions, -values[positions]))
    return [int(x) for x in positions[order]]


def select_positions(values: np.ndarray, n_top: int, n_random: int, seed: int):
    """Return strongest positions and a reproducible sample from the remainder."""
    n = len(values)
    if n == 0:
        return [], []

    top_n = min(n_top, n)
    if top_n == n:
        top = np.arange(n, dtype=np.int64)
    elif top_n:
        top = np.argpartition(values, n - top_n)[n - top_n:]
    else:
        top = np.empty(0, dtype=np.int64)
    top_list = ordered_positions(values, top)
    top_set = set(top_list)

    remaining_n = n - len(top_list)
    random_n = min(n_random, remaining_n)
    if random_n == 0:
        return top_list, []

    if remaining_n <= n_random:
        random_positions = np.array([i for i in range(n) if i not in top_set], dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        chosen = set()
        while len(chosen) < random_n:
            candidate = int(rng.integers(0, n))
            if candidate not in top_set:
                chosen.add(candidate)
        random_positions = np.fromiter(chosen, dtype=np.int64)
    return top_list, ordered_positions(values, random_positions)


def collect_samples(activation_path: Path, features: list[int], n_top: int, n_random: int):
    """Select sparse activation rows without constructing dense feature columns."""
    selected = {}
    needed_tweet_ids: set[str] = set()

    with h5py.File(activation_path, "r") as handle:
        total_tweets, total_features = (int(x) for x in handle.attrs["shape"])
        indptr = handle["indptr"][:]
        ids = handle["ids"][:]
        indices = handle["indices"]
        data = handle["data"]

        for feature in features:
            start, end = int(indptr[feature - 1]), int(indptr[feature])
            rows = indices[start:end]
            values = data[start:end]
            top, random = select_positions(values, n_top, n_random, seed=feature)

            examples = {"top": [], "random": []}
            for group, positions in (("top", top), ("random", random)):
                for position in positions:
                    tweet_id = str(int(ids[int(rows[position])]))
                    needed_tweet_ids.add(tweet_id)
                    examples[group].append({
                        "tweet_id": tweet_id,
                        "activation": round(float(values[position]), 6),
                    })

            n_activating = end - start
            selected[feature] = {
                "examples": examples,
                "n_activating": n_activating,
                "pct_activating": round(100.0 * n_activating / total_tweets, 6),
            }

    return selected, needed_tweet_ids, total_tweets, total_features


def load_selected_tweets(path: Path, needed_ids: set[str], chunksize: int) -> dict[str, dict]:
    tweets = {}
    for chunk in pd.read_csv(
        path,
        usecols=TWEET_COLUMNS,
        dtype={"tweet_id": "string"},
        chunksize=chunksize,
    ):
        wanted = chunk[chunk["tweet_id"].isin(needed_ids)]
        for row in wanted.to_dict("records"):
            tweet_id = str(row.pop("tweet_id"))
            tweets[tweet_id] = {key: clean_value(value) or "" for key, value in row.items()}
    return tweets


def attach_tweet_metadata(selected: dict[int, dict], tweets: dict[str, dict]) -> None:
    missing = []
    for feature_data in selected.values():
        for examples in feature_data["examples"].values():
            for example in examples:
                tweet = tweets.get(example["tweet_id"])
                if tweet is None:
                    missing.append(example["tweet_id"])
                else:
                    example.update(tweet)
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Could not find {len(missing)} selected tweets in the CSV: {preview}")


def interpretation_payload(raw: dict | None) -> dict | None:
    if raw is None:
        return None
    return {
        "model": raw["model"],
        "core": {
            "label": raw["core_label"],
            "explanation": raw["core_explanation"],
        },
        "broad": {
            "label": raw["broad_label"],
            "explanation": viewer_language(raw["broad_explanation"]),
        },
        "relation": viewer_language(raw["relation"]),
        "tail_coherence": raw["tail_coherence"],
        "scores": raw["scores"],
    }


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    os.replace(temporary, path)


def build(args: argparse.Namespace) -> None:
    run_path = Path(args.path)
    activation_path = run_path / "activations.h5"
    if not activation_path.exists():
        raise FileNotFoundError(f"Activations not found: {activation_path}")
    if args.n_top < 0 or args.n_random < 0:
        raise ValueError("Example counts must be non-negative")
    if args.shard_size < 1:
        raise ValueError("--shard-size must be positive")

    interpretations = load_interpretations(Path(args.interpretations), args.model)
    with h5py.File(activation_path, "r") as handle:
        total_features = int(handle.attrs["shape"][1])
    features = sorted(set(args.features or range(1, total_features + 1)))
    invalid = [feature for feature in features if not 1 <= feature <= total_features]
    if invalid:
        raise ValueError(f"Activation indices outside 1..{total_features}: {invalid[:5]}")

    print(f"Selecting examples for {len(features):,} activations...")
    selected, needed_ids, total_tweets, _ = collect_samples(
        activation_path, features, args.n_top, args.n_random
    )
    print(f"Loading metadata for {len(needed_ids):,} unique tweets...")
    tweets = load_selected_tweets(Path(args.tweets), needed_ids, args.tweet_chunksize)
    attach_tweet_metadata(selected, tweets)

    shards = defaultdict(list)
    manifest_features = []
    for feature in features:
        activation = selected[feature]
        interpretation = interpretation_payload(interpretations.get(feature))
        shard = f"{(feature - 1) // args.shard_size:03d}"
        payload = {
            "id": feature,
            "interpretation": interpretation,
            "stats": {
                "n_activating": activation["n_activating"],
                "pct_activating": activation["pct_activating"],
            },
            "examples": activation["examples"],
        }
        shards[shard].append(payload)
        manifest_features.append({
            "id": feature,
            "shard": shard,
            "core_label": interpretation["core"]["label"] if interpretation else None,
            "core_explanation": interpretation["core"]["explanation"] if interpretation else None,
            "broad_label": interpretation["broad"]["label"] if interpretation else None,
            "broad_explanation": interpretation["broad"]["explanation"] if interpretation else None,
            "relation": interpretation["relation"] if interpretation else None,
            "n_activating": activation["n_activating"],
            "pct_activating": activation["pct_activating"],
        })

    out_dir = Path(args.out_dir)
    for shard, payload in sorted(shards.items()):
        write_json(out_dir / "features" / f"{shard}.json", {
            "schema_version": SCHEMA_VERSION,
            "features": payload,
        })

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run": run_path.name,
        "total_features": total_features,
        "total_tweets": total_tweets,
        "exported_features": len(features),
        "interpreted_features": sum(feature in interpretations for feature in features),
        "n_top": args.n_top,
        "n_random": args.n_random,
        "features": manifest_features,
    }
    # Write the manifest last so an interrupted rebuild never points at missing shards.
    write_json(out_dir / "manifest.json", manifest)

    total_bytes = sum(path.stat().st_size for path in out_dir.rglob("*.json"))
    print(
        f"Wrote {len(shards):,} shards and manifest to {out_dir} "
        f"({total_bytes / 1024 / 1024:.1f} MiB)"
    )


if __name__ == "__main__":
    build(parse_args())
