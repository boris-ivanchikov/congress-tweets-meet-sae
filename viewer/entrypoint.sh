#!/bin/sh
set -eu

runtime_dir=${VIEWER_RUNTIME_DATA_DIR:-/tmp/backend_data}
volume_dir=${VIEWER_VOLUME_DATA_DIR:-/app/backend_data}
mkdir -p "$runtime_dir"

if [ ! -f "$runtime_dir/tweets.parquet" ]; then
  cp "$volume_dir/files/tweets.parquet" "$runtime_dir/tweets.parquet"
fi

for name in activation_counts.npy top_activation_indices.npy top_activation_values.npy; do
  if [ ! -f "$runtime_dir/$name" ]; then
    archive="$runtime_dir/$name.zst"
    cp "$volume_dir/files/$name.zst" "$archive"
    zstd -d -q -f "$archive" -o "$runtime_dir/$name"
    rm -f "$archive"
  fi
done

for name in activation_indptr.npy activation_indices.npy activation_values.npy activation_tweet_indptr.npy activation_tweet_indices.npy; do
  if [ ! -f "$runtime_dir/$name" ] && [ -f "$volume_dir/files/$name.zst" ]; then
    archive="$runtime_dir/$name.zst"
    cp "$volume_dir/files/$name.zst" "$archive"
    zstd -d -q -f "$archive" -o "$runtime_dir/$name"
    rm -f "$archive"
  fi
done

exec python -m uvicorn server:app --host 0.0.0.0 --port 7860 --workers 1
