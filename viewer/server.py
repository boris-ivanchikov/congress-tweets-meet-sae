"""FastAPI backend for full-corpus tweet search and activation lookup."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


ROOT = Path(__file__).resolve().parent
STATIC_DIR = Path(os.environ.get("VIEWER_STATIC_DIR", ROOT / "static"))
DATA_DIR = Path(os.environ.get("VIEWER_BACKEND_DATA", ROOT / "backend_data"))

print("Loading tweet corpus...", flush=True)
TWEETS = pq.read_table(DATA_DIR / "tweets.parquet").combine_chunks()
SEARCH_COLUMNS = {
    name: pc.utf8_lower(TWEETS[name]) for name in ("text", "name", "twitter")
}
YEARS = (
    TWEETS["year"].to_numpy(zero_copy_only=False)
    if "year" in TWEETS.column_names
    else np.array([int(value.as_py()[:4]) for value in TWEETS["posted_at"]], dtype=np.int16)
)
POSTED_DAYS = (
    TWEETS["posted_at"].to_numpy(zero_copy_only=False)
    .astype("datetime64[D]")
    .astype(np.int32)
)
PARTIES = TWEETS["party"] if "party" in TWEETS.column_names else pa.array([""] * len(TWEETS))
STATES = TWEETS["state"] if "state" in TWEETS.column_names else pa.array([""] * len(TWEETS))

ACTIVATION_COUNTS = np.load(DATA_DIR / "activation_counts.npy", mmap_mode="r")
ACTIVATION_INDICES = np.load(DATA_DIR / "top_activation_indices.npy", mmap_mode="r")
ACTIVATION_VALUES = np.load(DATA_DIR / "top_activation_values.npy", mmap_mode="r")
FULL_ACTIVATIONS = all(
    (DATA_DIR / name).exists()
    for name in ("activation_indptr.npy", "activation_indices.npy", "activation_values.npy")
)
if FULL_ACTIVATIONS:
    ACTIVATION_INDPTR = np.load(DATA_DIR / "activation_indptr.npy", mmap_mode="r")
    FULL_ACTIVATION_INDICES = np.load(DATA_DIR / "activation_indices.npy", mmap_mode="r")
    FULL_ACTIVATION_VALUES = np.load(DATA_DIR / "activation_values.npy", mmap_mode="r")
FILTERABLE_ACTIVATIONS = all(
    (DATA_DIR / name).exists()
    for name in ("activation_tweet_indptr.npy", "activation_tweet_indices.npy")
)
if FILTERABLE_ACTIVATIONS:
    ACTIVATION_TWEET_INDPTR = np.load(DATA_DIR / "activation_tweet_indptr.npy", mmap_mode="r")
    ACTIVATION_TWEET_INDICES = np.load(DATA_DIR / "activation_tweet_indices.npy", mmap_mode="r")

manifest = json.loads((STATIC_DIR / "data" / "manifest.json").read_text(encoding="utf-8"))
ACTIVATIONS = {int(item["id"]): item for item in manifest["features"]}
TOTAL_TWEETS = int(manifest["total_tweets"])
if len(TWEETS) != TOTAL_TWEETS:
    raise RuntimeError(f"Tweet corpus has {len(TWEETS):,} rows; expected {TOTAL_TWEETS:,}")

app = FastAPI(title="Congress Tweets SAE Explorer", docs_url="/api/docs", redoc_url=None)

LEXICAL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:lexical|lexeme|lemma|orthograph\w*|morpholog\w*|morpheme\w*|substring|prefix\w*|suffix\w*|digraph|trigraph|spelling|punctuation|capitalization|uppercase|lowercase|inflection\w*|homographs?|homophones?)\b",
        r"\bwords?\s+(?:beginning|starting|ending|containing|formed from|derived from|with (?:the )?(?:root|stem|prefix|suffix)|sharing (?:a )?(?:root|stem))\b",
        r"\b(?:word|verb|noun|adjective|adverb)\s+famil(?:y|ies)\b",
        r"\b(?:word|verb|noun|adjective|adverb)\s+forms?\b",
        r"\b(?:forms?|variants?)\s+of\s+(?:the\s+)?(?:word|root|stem|lexeme)\b",
        r"\b(?:letter|character)\s+(?:pair|sequence|pattern|cluster|combination|presence|occurrence)\b",
        r"\b(?:initial|final)\s+(?:letter|character|token)s?\b",
        r"\b(?:[a-z]{1,4}-initial|initial [a-z]{1,4})\b",
        r"\b(?:capitalized|uppercase|lowercase)\s+(?:words?|tokens?|names?|acronyms?|letters?)\b",
        r"\b(?:capitalized|uppercase|lowercase)\s+[\"']?[a-z]{1,4}[\"']?\s+(?:as\s+an?\s+)?initial\b",
        r"\b(?:bi|tri|n-?)gram\b",
        r"\b(?:text|tweets?|tokens?|strings?|proper nouns?|names?|acronyms?)\s+(?:containing|beginning|starting|ending)\b",
        r"\b(?:presence|occurrences?)\s+of\s+(?:the\s+)?(?:a\s+)?(?:letter|digit|substring|string|sequence|bigram|trigram|word|token)\b",
        r"\b(?:letter|digit|character)\s+[\"']?[a-z0-9]{1,4}[\"']?\s+(?:in|within|anywhere|presence|occurrence)\b",
        r"\bthe\s+(?:specific\s+|literal\s+)?word\b",
        r"\b(?:root|stem)[- ](?:based|related)\s+(?:words?|vocabulary|forms?)\b",
        r"\b(?:vocabulary|wording)\s+(?:patterns?|forms?|related to)\b",
        r"\b(?:names?|proper nouns?|acronyms?)\s+(?:beginning|starting|ending)\s+with\b",
        r"\b(?:word|token)[- ]initial\b",
    )
]


def is_lexical_activation(metadata: dict) -> bool:
    text = metadata.get("broad_label") or metadata.get("core_label") or ""
    return any(pattern.search(text) for pattern in LEXICAL_PATTERNS)


LEXICAL_ACTIVATIONS = {
    activation_id for activation_id, metadata in ACTIVATIONS.items()
    if is_lexical_activation(metadata)
}

INDEX_HTML = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
INDEX_HTML = INDEX_HTML.replace(
    '<link rel="stylesheet" href="styles.css">',
    f"<style>\n{(STATIC_DIR / 'styles.css').read_text(encoding='utf-8')}\n</style>",
)
INDEX_HTML = INDEX_HTML.replace(
    '<script src="app.js" defer></script>',
    f"<script>\n{(STATIC_DIR / 'app.js').read_text(encoding='utf-8')}\n</script>",
)


def search_tokens(value: str) -> list[str]:
    return re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE)[:12]


def tweet_payload(row_id: int) -> dict:
    return {
        "row_id": row_id,
        "tweet_id": TWEETS["tweet_id"][row_id].as_py(),
        "posted_at": TWEETS["posted_at"][row_id].as_py(),
        "name": TWEETS["name"][row_id].as_py(),
        "twitter": TWEETS["twitter"][row_id].as_py(),
        "text": TWEETS["text"][row_id].as_py(),
        "year": int(YEARS[row_id]),
        "party": PARTIES[row_id].as_py() or None,
        "state": STATES[row_id].as_py() or None,
    }


def matching_rows(
    tokens: list[str], year_from: int | None, year_to: int | None, party: str,
    representative: str, activation_ids: list[int], activation_mode: str,
) -> np.ndarray:
    matches: pa.Array | None = None
    for token in tokens:
        token_matches: pa.Array | None = None
        for column in SEARCH_COLUMNS.values():
            field_matches = pc.match_substring(column, token)
            token_matches = field_matches if token_matches is None else pc.or_(token_matches, field_matches)
        matches = token_matches if matches is None else pc.and_(matches, token_matches)
    if year_from is not None:
        year_match = pa.array(YEARS >= year_from)
        matches = year_match if matches is None else pc.and_(matches, year_match)
    if year_to is not None:
        year_match = pa.array(YEARS <= year_to)
        matches = year_match if matches is None else pc.and_(matches, year_match)
    if party:
        party_match = pc.equal(PARTIES, party.upper())
        matches = party_match if matches is None else pc.and_(matches, party_match)
    if representative:
        representative_match = pc.equal(SEARCH_COLUMNS["twitter"], representative.casefold().lstrip("@"))
        matches = representative_match if matches is None else pc.and_(matches, representative_match)
    if matches is None:
        rows = np.arange(TOTAL_TWEETS - 1, -1, -1, dtype=np.int64)
    else:
        # Tweet IDs are Twitter snowflakes, so reverse row order is newest first.
        rows = np.flatnonzero(matches.to_numpy(zero_copy_only=False))[::-1]
    if activation_ids:
        if not FILTERABLE_ACTIVATIONS:
            raise HTTPException(status_code=503, detail="Activation filtering is unavailable")
        groups = []
        for activation_id in activation_ids:
            start = int(ACTIVATION_TWEET_INDPTR[activation_id - 1])
            end = int(ACTIVATION_TWEET_INDPTR[activation_id])
            groups.append(np.asarray(ACTIVATION_TWEET_INDICES[start:end]))
        activation_rows = groups[0]
        for group in groups[1:]:
            if activation_mode == "all":
                activation_rows = np.intersect1d(activation_rows, group, assume_unique=True)
            else:
                activation_rows = np.union1d(activation_rows, group)
        activation_match = np.zeros(TOTAL_TWEETS, dtype=bool)
        activation_match[activation_rows] = True
        rows = rows[activation_match[rows]]
    return rows


def filter_options() -> dict:
    encoded = TWEETS["twitter"].combine_chunks().dictionary_encode()
    _, first_rows = np.unique(encoded.indices.to_numpy(zero_copy_only=False), return_index=True)
    people = []
    for row_id in first_rows:
        handle = TWEETS["twitter"][int(row_id)].as_py()
        if not handle:
            continue
        people.append({
            "twitter": handle,
            "name": TWEETS["name"][int(row_id)].as_py(),
            "party": PARTIES[int(row_id)].as_py() or None,
            "state": STATES[int(row_id)].as_py() or None,
        })
    return {
        "year_min": int(YEARS.min()),
        "year_max": int(YEARS.max()),
        "parties": ["D", "R"],
        "representatives": sorted(
            people, key=lambda item: ((item["name"] or "").casefold(), item["twitter"])
        ),
    }


FILTER_OPTIONS = filter_options()


def adaptive_time_bins(rows: np.ndarray) -> tuple[str, np.ndarray, np.ndarray, list[dict]]:
    days = POSTED_DAYS[rows]
    span_days = int(days.max() - days.min()) if len(days) else 0
    if span_days <= 90:
        period = "day"
        keys = days
        unit = "D"
        step = 1
    elif span_days <= 730:
        period = "week"
        # 1970-01-01 was a Thursday; offset by three to obtain Monday starts.
        keys = days - ((days + 3) % 7)
        unit = "D"
        step = 7
    else:
        period = "month"
        keys = days.astype("datetime64[D]").astype("datetime64[M]").astype(np.int32)
        unit = "M"
        step = 1
    first_key = int(keys.min())
    last_key = int(keys.max())
    all_keys = np.arange(first_key, last_key + step, step, dtype=np.int32)
    inverse = ((keys - first_key) // step).astype(np.int32)
    counts = np.bincount(inverse, minlength=len(all_keys))
    bins = [
        {
            "start": str(np.datetime64(int(key), unit)),
            "count": int(count),
        }
        for key, count in zip(all_keys, counts)
    ]
    return period, all_keys, inverse, bins


def activation_distribution(
    rows: np.ndarray, n_top: int = 20, hide_lexical: bool = False
) -> dict:
    if not len(rows):
        return {"time_bins": [], "top_activations": []}
    ordered_rows = np.sort(rows)
    period, time_keys, row_time_indices, time_bins = adaptive_time_bins(ordered_rows)
    if not FULL_ACTIVATIONS:
        return {"period": period, "time_bins": time_bins, "top_activations": []}
    n_features = len(ACTIVATIONS)
    time_feature_sums = np.zeros((len(time_keys), n_features), dtype=np.float64)
    time_totals = np.bincount(row_time_indices, minlength=len(time_keys))

    for start in range(0, len(ordered_rows), 25_000):
        batch_rows = ordered_rows[start : start + 25_000]
        batch_time_indices = row_time_indices[start : start + len(batch_rows)]
        entry_starts = np.asarray(ACTIVATION_INDPTR[batch_rows], dtype=np.int64)
        entry_ends = np.asarray(ACTIVATION_INDPTR[batch_rows + 1], dtype=np.int64)
        counts = entry_ends - entry_starts
        total_entries = int(counts.sum())
        if not total_entries:
            continue
        group_starts = np.repeat(entry_starts, counts)
        local_starts = np.repeat(np.cumsum(counts) - counts, counts)
        positions = group_starts + np.arange(total_entries, dtype=np.int64) - local_starts
        indices = np.asarray(FULL_ACTIVATION_INDICES[positions], dtype=np.int64)
        values = np.asarray(FULL_ACTIVATION_VALUES[positions], dtype=np.float64)
        repeated_times = np.repeat(batch_time_indices, counts)
        encoded = repeated_times * n_features + indices
        time_feature_sums += np.bincount(
            encoded, weights=values, minlength=len(time_keys) * n_features
        ).reshape(len(time_keys), n_features)

    totals = time_feature_sums.sum(axis=0)
    ranked_indices = np.argsort(totals)[::-1]
    if hide_lexical:
        ranked_indices = np.asarray([
            index for index in ranked_indices
            if int(index) + 1 not in LEXICAL_ACTIVATIONS
        ])
    top_indices = ranked_indices[:n_top]
    top = []
    for feature_index in top_indices:
        activation_id = int(feature_index) + 1
        metadata = ACTIVATIONS.get(activation_id, {})
        timeline = []
        for offset, denominator in enumerate(time_totals):
            timeline.append({
                "start": time_bins[offset]["start"],
                "mean": (
                    float(time_feature_sums[offset, feature_index] / int(denominator))
                    if denominator else None
                ),
            })
        top.append({
            "id": activation_id,
            "label": metadata.get("broad_label") or metadata.get("core_label") or "Uninterpreted activation",
            "mean": float(totals[feature_index] / len(rows)),
            "timeline": timeline,
        })
    return {
        "period": period,
        "time_bins": time_bins,
        "top_activations": top,
        "basis": "mean activation across all matching tweets, counting non-activation as zero",
    }


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True, "tweets": TOTAL_TWEETS, "activations": len(ACTIVATIONS),
        "full_activations": FULL_ACTIVATIONS, "filterable_activations": FILTERABLE_ACTIVATIONS,
    }


@app.get("/api/tweets/options")
def tweet_filter_options() -> dict:
    return FILTER_OPTIONS


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    # Private HF Spaces authenticate the iframe URL with a query signature.
    # Inline critical assets so the signature cannot be lost on subrequests.
    return INDEX_HTML


@app.get("/api/tweets/search")
def search_tweets(
    q: str = Query(default="", max_length=300),
    limit: int = Query(default=30, ge=1, le=100),
    offset: int = Query(default=0, ge=0, le=10_000),
    year_from: int | None = Query(default=None, ge=1900, le=2100),
    year_to: int | None = Query(default=None, ge=1900, le=2100),
    party: str = Query(default="", pattern="^(|D|R)$"),
    representative: str = Query(default="", max_length=80),
    activations: str = Query(default="", max_length=300),
    activation_mode: str = Query(default="all", pattern="^(all|any)$"),
    hide_lexical: bool = Query(default=False),
) -> dict:
    if year_from is not None and year_to is not None and year_from > year_to:
        raise HTTPException(status_code=400, detail="year_from must not exceed year_to")
    try:
        activation_ids = [int(value) for value in activations.split(",") if value]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="activations must be comma-separated integers") from exc
    if len(activation_ids) > 20 or any(value not in ACTIVATIONS for value in activation_ids):
        raise HTTPException(status_code=400, detail="invalid activation filter")
    activation_ids = list(dict.fromkeys(activation_ids))
    rows = matching_rows(
        search_tokens(q), year_from, year_to, party, representative,
        activation_ids, activation_mode,
    )
    selected = rows[offset : offset + limit]
    return {
        "query": q,
        "total": int(len(rows)),
        "offset": offset,
        "limit": limit,
        "tweets": [tweet_payload(int(row_id)) for row_id in selected],
        "stats": activation_distribution(rows, hide_lexical=hide_lexical),
    }


@app.get("/api/tweets/{row_id}")
def get_tweet(row_id: int) -> dict:
    if not 0 <= row_id < TOTAL_TWEETS:
        raise HTTPException(status_code=404, detail="Tweet not found")

    if FULL_ACTIVATIONS:
        start, end = int(ACTIVATION_INDPTR[row_id]), int(ACTIVATION_INDPTR[row_id + 1])
        indices = np.asarray(FULL_ACTIVATION_INDICES[start:end])
        values = np.asarray(FULL_ACTIVATION_VALUES[start:end])
        order = np.argsort(values)[::-1]
        indices, values = indices[order], values[order]
    else:
        stored_count = min(int(ACTIVATION_COUNTS[row_id]), ACTIVATION_INDICES.shape[1])
        values = np.asarray(ACTIVATION_VALUES[row_id, :stored_count])
        indices = np.asarray(ACTIVATION_INDICES[row_id, :stored_count])
    active = []
    for position in range(len(indices)):
        activation_id = int(indices[position]) + 1
        metadata = ACTIVATIONS.get(activation_id, {})
        active.append({
            "id": activation_id,
            "value": float(values[position]),
            "core_label": metadata.get("core_label"),
            "broad_label": metadata.get("broad_label"),
        })
    return {
        "tweet": tweet_payload(row_id),
        "n_activations": int(ACTIVATION_COUNTS[row_id]),
        "activations": active,
        "complete": FULL_ACTIVATIONS,
    }


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
