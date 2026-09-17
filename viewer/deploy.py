"""Create or update the Docker Hugging Face Space for the SAE viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, Volume


DEFAULT_REPO = "ivanchikovboris/congress-tweets-sae-explorer"
DEFAULT_DATA_BUCKET = "ivanchikovboris/congress-tweets-sae-explorer-data"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy the generated SAE viewer backend.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--data-bucket-id", default=DEFAULT_DATA_BUCKET)
    parser.add_argument(
        "--upload-data",
        action="store_true",
        help="Upload backend artifacts with the Space code",
    )
    parser.add_argument(
        "--visibility",
        required=True,
        choices=("public", "protected", "private"),
        help="HF Space visibility; this must be chosen explicitly",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    viewer_dir = Path(__file__).resolve().parent
    manifest_path = viewer_dir / "data" / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            "viewer/data/manifest.json is missing. Run viewer/build_data.py before deploying."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("exported_features") != manifest.get("total_features"):
        raise SystemExit(
            "Refusing to deploy a partial activation export: "
            f"{manifest.get('exported_features')} of {manifest.get('total_features')} activations."
        )
    backend_dir = viewer_dir / "backend_data"
    data_files = [
        backend_dir / "tweets.parquet",
        backend_dir / "activation_counts.npy.zst",
        backend_dir / "top_activation_indices.npy.zst",
        backend_dir / "top_activation_values.npy.zst",
        backend_dir / "activation_indptr.npy.zst",
        backend_dir / "activation_indices.npy.zst",
        backend_dir / "activation_values.npy.zst",
        backend_dir / "activation_tweet_indptr.npy.zst",
        backend_dir / "activation_tweet_indices.npy.zst",
    ]
    missing_backend = [path.name for path in data_files if not path.exists()]
    if missing_backend:
        raise SystemExit(f"Backend data is missing: {', '.join(missing_backend)}")

    api = HfApi()
    account = api.whoami()["name"]
    namespace = args.repo_id.split("/", 1)[0]
    if account != namespace:
        print(f"Authenticated as {account}; deploying to namespace {namespace}.")

    api.create_repo(
        repo_id=args.repo_id,
        repo_type="space",
        space_sdk="docker",
        visibility=args.visibility,
        exist_ok=True,
    )
    if args.upload_data:
        api.create_bucket(bucket_id=args.data_bucket_id, private=True, exist_ok=True)
        destinations = [f"files/{path.name}" for path in data_files]
        existing = {
            item.path: item.size
            for item in api.get_bucket_paths_info(args.data_bucket_id, destinations)
        }
        pending = [
            (path, destination)
            for path, destination in zip(data_files, destinations)
            if existing.get(destination) != path.stat().st_size
        ]
        if pending:
            print(f"Uploading {len(pending)} backend file(s) directly to private storage...", flush=True)
            for position, item in enumerate(pending, start=1):
                path, destination = item
                print(f"Backend file {position}/{len(pending)}: {path.name}", flush=True)
                api.batch_bucket_files(args.data_bucket_id, add=[item])
        else:
            print("Private backend storage is already current.", flush=True)

    source_paths = [
        (viewer_dir / "Dockerfile", "Dockerfile"),
        (viewer_dir / "requirements.txt", "requirements.txt"),
        (viewer_dir / "server.py", "server.py"),
        (viewer_dir / "entrypoint.sh", "entrypoint.sh"),
        (viewer_dir / "index.html", "index.html"),
        (viewer_dir / "app.js", "app.js"),
        (viewer_dir / "styles.css", "styles.css"),
        (viewer_dir / "vendor" / "forceatlas2.js", "vendor/forceatlas2.js"),
        (viewer_dir / "data" / "activation_graph.json", "data/activation_graph.json"),
        (viewer_dir / "data" / "activation_graph_weighted.json", "data/activation_graph_weighted.json"),
        # Upload the Space metadata last, after every Docker input is present.
        (viewer_dir / "README.md", "README.md"),
    ]
    remote_files = set(api.list_repo_files(args.repo_id, repo_type="space"))
    if "data/manifest.json" not in remote_files:
        raise SystemExit("Space is missing viewer/data; run the static data deployment first.")
    print(f"Uploading {len(source_paths)} application files in one commit...", flush=True)
    api.create_commit(
        repo_id=args.repo_id,
        repo_type="space",
        operations=[
            CommitOperationAdd(path_in_repo=destination, path_or_fileobj=path)
            for path, destination in source_paths
        ] + ([CommitOperationDelete(path_in_repo="introduction.md")]
             if "introduction.md" in remote_files else []),
        commit_message=(
            f"Update SAE viewer ({manifest['exported_features']:,} activations, "
            f"{manifest['interpreted_features']:,} interpreted)"
        ),
    )
    api.set_space_volumes(
        args.repo_id,
        volumes=[
            Volume(
                type="bucket",
                source=args.data_bucket_id,
                mount_path="/app/backend_data",
                read_only=True,
            )
        ],
    )
    print(f"Space: https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
