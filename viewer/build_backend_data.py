"""Build the immutable tweet and row-oriented activation data for the viewer API."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse


TWEET_COLUMNS = ["tweet_id", "posted_at", "name", "twitter", "text"]
REPRESENTATIVE_COLUMNS = ["bioguide", "twitter", "party", "state"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tweets", default="data/tweets.csv")
    parser.add_argument("--representatives", default="data/representatives.csv")
    parser.add_argument("--activations", default="sae/runs/chief-dream-73/activations.h5")
    parser.add_argument("--out-dir", default="viewer/backend_data")
    parser.add_argument("--skip-tweets", action="store_true")
    parser.add_argument("--skip-activations", action="store_true")
    parser.add_argument(
        "--compact-existing",
        action="store_true",
        help="Build top-activation arrays from existing row-oriented arrays",
    )
    parser.add_argument(
        "--filter-index-only",
        action="store_true",
        help="Build only the activation-to-tweet index used by viewer filters",
    )
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def build_tweet_corpus(
    tweets_path: Path, representatives_path: Path, activation_path: Path, out_dir: Path
) -> None:
    with h5py.File(activation_path, "r") as handle:
        activation_ids = handle["ids"][:]
    if not np.all(activation_ids[1:] >= activation_ids[:-1]):
        raise ValueError("Activation tweet IDs must be sorted for lookup")

    print("Loading and ordering tweet corpus...", flush=True)
    tweets = pd.read_csv(
        tweets_path,
        usecols=TWEET_COLUMNS,
        dtype={column: "string" for column in TWEET_COLUMNS},
        keep_default_na=False,
        dtype_backend="pyarrow",
    )
    numeric_ids = pd.to_numeric(tweets["tweet_id"], errors="raise").to_numpy(dtype=np.int64)
    positions = np.searchsorted(activation_ids, numeric_ids)
    valid = positions < len(activation_ids)
    valid[valid] &= activation_ids[positions[valid]] == numeric_ids[valid]
    if not valid.all() or len(tweets) != len(activation_ids):
        raise ValueError(
            f"Expected exactly {len(activation_ids):,} activation tweets; "
            f"matched {int(valid.sum()):,}/{len(tweets):,} CSV rows"
        )

    tweets["row_id"] = positions
    tweets["name"] = tweets["name"].str.strip()
    tweets["twitter"] = tweets["twitter"].str.strip().str.lower()
    representatives = pd.read_csv(
        representatives_path,
        usecols=REPRESENTATIVE_COLUMNS,
        dtype="string",
        keep_default_na=False,
        dtype_backend="pyarrow",
    )
    representatives["twitter"] = representatives["twitter"].str.strip().str.lower()
    if representatives["twitter"].duplicated().any():
        raise ValueError("Representative Twitter handles must be unique")
    tweets = tweets.merge(representatives, on="twitter", how="left", validate="many_to_one")
    for column in ("bioguide", "party", "state"):
        tweets[column] = tweets[column].fillna("")
    tweets["year"] = pd.to_numeric(tweets["posted_at"].str.slice(0, 4), errors="raise").astype("int16")
    tweets.sort_values("row_id", inplace=True)
    if not np.array_equal(tweets["row_id"].to_numpy(), np.arange(len(activation_ids))):
        raise ValueError("Tweet corpus does not map one-to-one to activation rows")
    tweets.drop(columns="row_id", inplace=True)

    corpus_path = out_dir / "tweets.parquet"
    temporary_path = out_dir / "tweets.parquet.tmp"
    tweets.to_parquet(
        temporary_path,
        index=False,
        engine="pyarrow",
        compression="zstd",
        compression_level=10,
        row_group_size=100_000,
    )
    temporary_path.replace(corpus_path)
    print(f"Wrote {corpus_path} ({corpus_path.stat().st_size / 1024**2:.1f} MiB)", flush=True)


def write_top_activations(indptr, indices, values, out_dir: Path, top_k: int) -> None:
    n_rows = len(indptr) - 1
    top_indices = np.lib.format.open_memmap(
        out_dir / "top_activation_indices.npy", mode="w+", dtype=np.uint16, shape=(n_rows, top_k)
    )
    top_values = np.lib.format.open_memmap(
        out_dir / "top_activation_values.npy", mode="w+", dtype=np.float16, shape=(n_rows, top_k)
    )
    counts = np.lib.format.open_memmap(
        out_dir / "activation_counts.npy", mode="w+", dtype=np.uint8, shape=(n_rows,)
    )
    top_indices[:] = 0
    top_values[:] = 0
    for row in range(n_rows):
        start, end = int(indptr[row]), int(indptr[row + 1])
        row_values = values[start:end]
        count = end - start
        counts[row] = min(count, np.iinfo(np.uint8).max)
        if count > top_k:
            selected = np.argpartition(row_values, -top_k)[-top_k:]
        else:
            selected = np.arange(count)
        selected = selected[np.argsort(row_values[selected])[::-1]]
        kept = len(selected)
        top_indices[row, :kept] = indices[start:end][selected]
        top_values[row, :kept] = row_values[selected]
        if row and row % 500_000 == 0:
            print(f"Compacted activations: {row:,}/{n_rows:,} tweets", flush=True)
    del top_indices, top_values, counts
    total_size = sum(path.stat().st_size for path in out_dir.glob("top_activation_*.npy"))
    total_size += (out_dir / "activation_counts.npy").stat().st_size
    print(f"Wrote strongest {top_k} activations per tweet ({total_size / 1024**2:.1f} MiB)", flush=True)


def write_full_activations(indptr, indices, values, out_dir: Path) -> None:
    arrays = {
        "activation_indptr.npy": np.asarray(indptr, dtype=np.uint32),
        "activation_indices.npy": np.asarray(indices, dtype=np.uint16),
        "activation_values.npy": np.asarray(values, dtype=np.float16),
    }
    for name, array in arrays.items():
        np.save(out_dir / name, array)
    size = sum((out_dir / name).stat().st_size for name in arrays)
    print(f"Wrote complete row-oriented activations ({size / 1024**2:.1f} MiB)", flush=True)


def write_filter_index(activation_path: Path, out_dir: Path) -> None:
    """Store the feature-oriented row index compactly without duplicating values."""
    print("Building activation-to-tweet filter index...", flush=True)
    with h5py.File(activation_path, "r") as handle:
        source_rows = handle["indices"]
        source_indptr = handle["indptr"][:]
        if len(source_rows) >= np.iinfo(np.uint32).max:
            raise ValueError("Activation-to-tweet index no longer fits uint32")
        np.save(out_dir / "activation_tweet_indptr.npy", source_indptr.astype(np.uint32))
        target_rows = np.lib.format.open_memmap(
            out_dir / "activation_tweet_indices.npy",
            mode="w+",
            dtype=np.uint32,
            shape=source_rows.shape,
        )
        chunk_size = 10_000_000
        for start in range(0, len(source_rows), chunk_size):
            end = min(start + chunk_size, len(source_rows))
            target_rows[start:end] = source_rows[start:end]
        del target_rows
    size = sum(
        (out_dir / name).stat().st_size
        for name in ("activation_tweet_indptr.npy", "activation_tweet_indices.npy")
    )
    print(f"Wrote activation filter index ({size / 1024**2:.1f} MiB)", flush=True)


def build_row_activations(activation_path: Path, out_dir: Path, top_k: int) -> None:
    print("Loading feature-oriented sparse activations...", flush=True)
    with h5py.File(activation_path, "r") as handle:
        shape = tuple(int(value) for value in handle.attrs["shape"])
        values = handle["data"][:]
        row_indices = handle["indices"][:]
        feature_indptr = handle["indptr"][:]

    matrix = sparse.csc_matrix((values, row_indices, feature_indptr), shape=shape)
    print("Converting activations to tweet-oriented CSR...", flush=True)
    row_matrix = matrix.tocsr()
    del matrix, values, row_indices, feature_indptr

    if row_matrix.nnz >= np.iinfo(np.uint32).max:
        raise ValueError("Activation count no longer fits uint32 row pointers")
    if shape[1] > np.iinfo(np.uint16).max:
        raise ValueError("Activation indices no longer fit uint16")

    write_full_activations(row_matrix.indptr, row_matrix.indices, row_matrix.data, out_dir)
    write_top_activations(
        row_matrix.indptr,
        row_matrix.indices.astype(np.uint16, copy=False),
        row_matrix.data.astype(np.float16, copy=False),
        out_dir,
        top_k,
    )
    write_filter_index(activation_path, out_dir)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tweets_path = Path(args.tweets)
    activation_path = Path(args.activations)
    if args.filter_index_only:
        write_filter_index(activation_path, out_dir)
        return
    if not args.skip_tweets:
        build_tweet_corpus(tweets_path, Path(args.representatives), activation_path, out_dir)
    if not args.skip_activations:
        if args.compact_existing:
            write_top_activations(
                np.load(out_dir / "activation_indptr.npy", mmap_mode="r"),
                np.load(out_dir / "activation_indices.npy", mmap_mode="r"),
                np.load(out_dir / "activation_values.npy", mmap_mode="r"),
                out_dir,
                args.top_k,
            )
        else:
            build_row_activations(activation_path, out_dir, args.top_k)


if __name__ == "__main__":
    main()
