"""Embed tweets into one resumable HDF5 file (ids and embeddings datasets).

Embeddings are stored as float16; tweet IDs remain exact int64 values.
Only successful responses enter the arrays. Batch status, attempts, usage and
errors live inside the same file. Resume uses the same input and skips finished
batches, including exhausted failures. One process may write this file at a time.
"""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import heapq
import json
from pathlib import Path
import random
import threading
import time

import h5py
import numpy as np
import requests
from tqdm import tqdm


MODEL = "Qwen/Qwen3-Embedding-8B"
BASE_URL = "https://api.tokenfactory.nebius.com/v1"
INSTRUCTION = "Represent this tweet."
PRICE_PER_MILLION_TOKENS = 0.01
MAX_ATTEMPTS = 5  # Initial request plus at most four retries.
MAX_CONCURRENCY = 8
_REQUEST_SESSIONS = threading.local()


def prepare_text(text, instruction=INSTRUCTION):
    return f"Instruct: {instruction}\nQuery:{text}" if instruction else text


def request_batch(texts, key_file, base_url, model, proxy):
    """One attempt. Workers do network I/O only; the main thread writes HDF5."""
    key = key_file.read_text().strip()
    session = getattr(_REQUEST_SESSIONS, "session", None)
    if session is None:
        session = requests.Session()
        _REQUEST_SESSIONS.session = session
    # One reusable connection pool per worker; no session is shared by threads.
    session.headers["Authorization"] = f"Bearer {key}"
    if proxy:
        session.trust_env = False
        session.proxies = {"http": proxy, "https": proxy}
    try:
        response = session.post(
            base_url.rstrip("/") + "/embeddings",
            json={"model": model, "input": texts, "encoding_format": "float"},
            timeout=(15, 120),
        )
    except (requests.Timeout, requests.ConnectionError,
            requests.exceptions.ChunkedEncodingError) as exc:
        return {"error": type(exc).__name__, "retry": True, "delay": 0}
    if not response.ok:
        delay = 0
        header = response.headers.get("Retry-After", "0")
        try:
            delay = float(header)
        except ValueError:
            try:
                delay = (parsedate_to_datetime(header)
                         - datetime.now(timezone.utc)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                pass
        if not np.isfinite(delay):
            delay = 0
        reason = f"HTTP {response.status_code}"
        try:
            detail = response.json().get("error", {})
            if isinstance(detail, dict):
                reason += ": " + str(detail.get("message", ""))[:400]
        except (ValueError, AttributeError):
            pass
        return {"error": reason.replace(key, "[redacted]"),
                "retry": response.status_code in (408, 429, 500, 502, 503, 504),
                "fatal": response.status_code not in (408, 429, 500, 502, 503, 504),
                "delay": max(0, delay)}
    try:
        result = response.json()
        data = sorted(result["data"], key=lambda row: row["index"])
        if [row["index"] for row in data] != list(range(len(texts))):
            raise ValueError("Missing/duplicate indices")
        vectors = np.asarray([row["embedding"] for row in data], dtype=np.float32)
        tokens = int(result["usage"]["prompt_tokens"])
        if (vectors.ndim != 2 or vectors.shape[1] == 0 or tokens < 0
                or not np.isfinite(vectors).all()
                or np.any(np.linalg.norm(vectors, axis=1) == 0)):
            raise ValueError("Invalid embeddings")
    except (ValueError, KeyError, TypeError, AttributeError):
        # A malformed successful response may already have been billed.
        return {"error": "Malformed successful response", "retry": False,
                "fatal": True, "delay": 0}
    return {"vectors": vectors, "tokens": tokens}


def commit(file, batch, ids, vectors, tokens):
    """Append then commit a batch; uncommitted tail is removed on resume."""
    ledger = file["batches"]
    record = ledger[batch]
    start = int(file.attrs["saved_rows"])
    end = start + len(ids)
    if "embeddings" not in file:
        file.create_dataset("embeddings", shape=(0, vectors.shape[1]),
                            maxshape=(None, vectors.shape[1]), dtype="f2",
                            chunks=(32, vectors.shape[1]))
    embeddings = file["embeddings"]
    if embeddings.shape[1] != vectors.shape[1]:
        raise RuntimeError("API embedding dimensions changed")
    file["ids"].resize((end,))
    embeddings.resize((end, vectors.shape[1]))
    file["ids"][start:end] = ids
    embeddings[start:end] = vectors
    file.flush()
    record["status"] = 1
    record["end"] = end
    record["tokens"] = tokens
    ledger[batch] = record
    file.flush()
    file.attrs["saved_rows"] = end
    file.flush()


def recover(file):
    ledger = file["batches"][:]
    saved = int(ledger["end"].max(initial=0))
    if len(file["ids"]) < saved:
        raise RuntimeError("Output data is shorter than its committed checkpoint")
    file["ids"].resize((saved,))
    if "embeddings" in file:
        matrix = file["embeddings"]
        if matrix.dtype != np.float16:
            raise RuntimeError("Existing embeddings are not float16; use a new output file")
        if len(matrix) < saved:
            raise RuntimeError("Embeddings are shorter than the checkpoint")
        matrix.resize((saved, matrix.shape[1]))
    file.attrs["saved_rows"] = saved
    file.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output .h5 file")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-concurrency", type=int, default=MAX_CONCURRENCY)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--key-file", type=Path, default=Path.home() / ".nebius/api_key")
    parser.add_argument("--proxy-file", type=Path)
    parser.add_argument("--max-attempted-tokens", type=int, default=None,
                        help="Optional UTF-8 byte-based token bound, counting retries")
    args = parser.parse_args()
    if (args.batch_size < 1 or args.max_concurrency < 1
            or (args.limit is not None and args.limit < 1)):
        parser.error("Batch size, concurrency and limit must be positive")
    stat = args.input.stat()
    config = {"input": str(args.input.resolve()), "size": stat.st_size,
              "mtime_ns": stat.st_mtime_ns, "instruction": INSTRUCTION,
              "model": args.model, "base_url": args.base_url,
              "batch_size": args.batch_size, "limit": args.limit}
    csv.field_size_limit(10_000_000)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        total = 0
        with args.input.open(newline="") as source:
            reader = csv.DictReader(source)
            if not {"tweet_id", "text"}.issubset(reader.fieldnames or []):
                raise RuntimeError("Input must contain tweet_id and text")
            for _ in tqdm(reader, desc="Count tweets", unit="tweet"):
                total += 1
                if args.limit and total == args.limit:
                    break
        if not total:
            raise RuntimeError("Input contains no tweets")
        with h5py.File(args.output, "x") as file:
            file.attrs.update(format="tweets-h5-v1", config=json.dumps(config),
                              total_rows=total, saved_rows=0, attempted_token_bound=0)
            file.create_dataset("ids", shape=(0,), maxshape=(None,), dtype="i8")
            count = (total + args.batch_size - 1) // args.batch_size
            dtype = np.dtype([("status", "u1"), ("attempts", "u1"),
                              ("end", "i8"), ("tokens", "i8")])
            file.create_dataset("batches", data=np.zeros(count, dtype=dtype))
            file.create_dataset("errors", shape=(count,), dtype=h5py.string_dtype())
    proxy = args.proxy_file.read_text().strip() if args.proxy_file else None
    with h5py.File(args.output, "r+") as file:
        if (file.attrs.get("format") != "tweets-h5-v1"
                or json.loads(file.attrs["config"]) != config):
            raise RuntimeError("Output format/settings/input changed; use a new file")
        recover(file)
        ledger = file["batches"]
        total = int(file.attrs["total_rows"])
        records = ledger[:]
        # An interrupted in-flight request consumes an attempt: never reset the cap.
        exhausted = (records["status"] == 0) & (records["attempts"] >= MAX_ATTEMPTS)
        for i in np.flatnonzero(exhausted):
            rec = records[i]; rec["status"] = 2; ledger[i] = rec
            file["errors"][i] = "Attempt cap reached, possibly interrupted in flight"
        records = ledger[:]
        def size(batch):
            return min(args.batch_size, total - batch * args.batch_size)
        finished = sum(size(i) for i in np.flatnonzero(records["status"]))
        failed = sum(size(i) for i in np.flatnonzero(records["status"] == 2))
        tokens = int(records["tokens"].sum())
        retries = int(np.maximum(records["attempts"].astype(int) - 1, 0).sum())
        concurrency = min(2, args.max_concurrency)
        healthy = 0
        stopped = False
        with args.input.open(newline="") as source, tqdm(
            total=total, initial=finished, unit="tweet", desc="Embed tweets",
            smoothing=0
        ) as bar, ThreadPoolExecutor(max_workers=args.max_concurrency) as pool:
            reader = iter(csv.DictReader(source))
            next_batch = 0
            pending = []  # (retry deadline, batch, ids, texts)
            futures = {}
            cooldown_until = 0.0
            last_slowdown = float("-inf")

            def show():
                remaining = max(0, cooldown_until - time.monotonic())
                status = ("draining" if stopped else
                          f"backoff {remaining:.0f}s" if remaining else "running")
                bar.set_postfix(saved=int(file.attrs["saved_rows"]), failed=failed,
                                cost_est=f"${tokens * PRICE_PER_MILLION_TOKENS / 1e6:.4f}",
                                retries=retries, active=len(futures),
                                workers=concurrency, status=status, refresh=False)
                bar.refresh()

            show()
            while futures or (not stopped and (pending or next_batch < len(ledger))):
                # Refill each free slot without waiting for the slowest request.
                while (not stopped and len(futures) < concurrency
                       and time.monotonic() >= cooldown_until):
                    if pending and pending[0][0] <= time.monotonic():
                        _, batch, ids, texts = heapq.heappop(pending)
                    elif next_batch < len(ledger):
                        batch = next_batch
                        next_batch += 1
                        rows = [next(reader) for _ in range(size(batch))]
                        if records[batch]["status"]:
                            continue
                        ids = np.asarray([int(r["tweet_id"]) for r in rows], dtype=np.int64)
                        texts = [prepare_text(r["text"]) for r in rows]
                    else:
                        break
                    bound = sum(len(t.encode("utf-8")) + 2 for t in texts)
                    attempted = int(file.attrs["attempted_token_bound"]) + bound
                    if args.max_attempted_tokens is not None and attempted > args.max_attempted_tokens:
                        stopped = True
                        tqdm.write("Token guard reached; saving in-flight results and stopping")
                        break
                    rec = ledger[batch]
                    if rec["attempts"]:
                        retries += 1
                    rec["attempts"] += 1
                    ledger[batch] = rec
                    file.attrs["attempted_token_bound"] = attempted
                    file.flush()
                    submitted = time.monotonic()
                    future = pool.submit(request_batch, texts, args.key_file,
                                         args.base_url, args.model, proxy)
                    futures[future] = (batch, ids, texts, submitted)
                if futures:
                    done, _ = wait(futures, timeout=0.5, return_when=FIRST_COMPLETED)
                else:
                    done = ()
                    if not stopped and (pending or next_batch < len(ledger)):
                        time.sleep(0.1)
                for future in done:
                    batch, ids, texts, submitted = futures.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {"error": f"Client error: {type(exc).__name__}",
                                  "retry": False, "fatal": True, "delay": 0}
                    if "vectors" in result:
                        commit(file, batch, ids, result["vectors"], result["tokens"])
                        tokens += result["tokens"]
                        file["errors"][batch] = ""
                        bar.update(len(ids))
                        if submitted >= last_slowdown:
                            healthy += 1
                        if (healthy >= max(8, concurrency)
                                and time.monotonic() >= cooldown_until
                                and concurrency < args.max_concurrency):
                            concurrency += 1
                            healthy = 0
                    else:
                        reason = result["error"]
                        rec = ledger[batch]
                        file["errors"][batch] = reason
                        if result.get("fatal"):
                            stopped = True
                        if result["retry"]:
                            now = time.monotonic()
                            delay = max(result["delay"], min(60, 2 ** int(rec["attempts"])))
                            deadline = now + delay + random.random()
                            cooldown_until = max(cooldown_until, deadline)
                            # Errors from the same in-flight cohort trigger only
                            # one reduction, not repeated halvings down to one.
                            if submitted >= last_slowdown:
                                concurrency = max(1, concurrency // 2)
                                last_slowdown = now
                                healthy = 0
                        if result["retry"] and rec["attempts"] < MAX_ATTEMPTS:
                            heapq.heappush(pending, (deadline, batch, ids, texts))
                        else:
                            rec["status"] = 2
                            ledger[batch] = rec
                            failed += len(ids)
                            bar.update(len(ids))
                            if result["retry"]:
                                tqdm.write("Retries exhausted; stopping for inspection")
                                stopped = True
                        tqdm.write(f"Batch {batch}, attempt {rec['attempts']}/{MAX_ATTEMPTS}: {reason}")
                    file.flush()
                # Refresh even during quiet periods. The rate/ETA includes time
                # spent waiting, rather than smoothing just a burst of commits.
                show()
            print(f"Saved {int(file.attrs['saved_rows']):,}; failed {failed:,}; "
                  f"estimated cost ${tokens * PRICE_PER_MILLION_TOKENS / 1e6:.4f}")
            if stopped:
                print("Stopped early. Inspect the errors dataset before resuming.")


if __name__ == "__main__":
    main()
