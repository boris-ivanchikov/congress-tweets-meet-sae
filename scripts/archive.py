"""Compile CongressTweets without requesting any URL contained in a tweet.

The upstream collector already expands RT/QT content and media into `text`.
List-valued CSV cells are JSON arrays for lossless storage, not model prompts.
"""

import csv
import json
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

import requests
from tqdm import tqdm

REVISION = "f5c837de11a67f07a8a8565d2961202c28885155"
ACCOUNTS_REVISION = "002e949c85341c0d222241be70331817f896477f"
URL_RE = re.compile(r'https?://[^\s<>"\\]+')
STATUS_RE = re.compile(
    r"https?://(?:www\.|mobile\.)?(?:twitter\.com|x\.com)/([\w]+)/status(?:es)?/(\d+)",
    re.I,
)
RT_RE = re.compile(r"^RT @([A-Za-z0-9_]+)\s*:?[ ]*")
QT_RE = re.compile(r"(?:^| )QT @([A-Za-z0-9_]+) ")
LIST_FIELDS = ("images", "videos", "urls")
FIELDS = (
    "tweet_id",
    "posted_at",
    "name",
    "twitter",
    "text",
    "user_id",
    "account_bioguide",
    "account_type",
    "tweet_url",
    "archive_link",
    "is_retweet",
    "retweet_id",
    *LIST_FIELDS,
    "source",
    "archive_file",
)


def get_json(url, cache=None):
    if cache and cache.exists():
        return json.loads(cache.read_bytes())
    for attempt in range(5):
        try:
            response = requests.get(url, timeout=(15, 90))
            response.raise_for_status()
            value = response.json()
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache.with_suffix(".tmp")
                tmp.write_bytes(response.content)
                tmp.replace(cache)
            return value
        except (requests.RequestException, ValueError):
            if attempt == 4:
                raise
            time.sleep(2**attempt)


def unique(values):
    return list(dict.fromkeys(values))


def unpack_records(record):
    # One archived daily file contains an unflattened collector response.
    if "id" not in record and set(record) <= {"yesterday", "today"} and record:
        for key in ("yesterday", "today"):
            for child in record.get(key, []):
                yield from unpack_records(child)
    else:
        yield record


def extract_urls(text):
    # Preserve URLs as supplied by the archive; trim only surrounding prose.
    return unique(u.rstrip(".,;!?)") for u in URL_RE.findall(text))


def partition_media(urls):
    """Separate Twitter CDN media; leave other providers' URLs untouched."""
    media = {field: [] for field in LIST_FIELDS}
    for url in unique(urls):
        field = "urls"
        try:
            parsed = urlsplit(url)
        except ValueError:
            media[field].append(url)
            continue
        host, path = (parsed.hostname or "").lower(), parsed.path.lower()
        if host == "video.twimg.com":
            field = "videos"
        elif host == "pbs.twimg.com" and any(
            s in path
            for s in (
                "/ext_tw_video_thumb/",
                "/amplify_video_thumb/",
                "/tweet_video_thumb/",
            )
        ):
            continue  # Video frames are intentionally excluded.
        elif host == "pbs.twimg.com":
            field = "images"
        media[field].append(url)
    return media


def extract_media(text):
    return partition_media(extract_urls(text))


def upgrade_media_columns(path, work_dir):
    """Upgrade the old raw CSV locally, without fetching the archive again."""
    counts = Counter()
    signature = (path.stat().st_size, path.stat().st_mtime_ns)
    temporary = Path(work_dir) / "tweets_media.csv"
    with (
        path.open(encoding="utf-8", newline="") as src,
        temporary.open("w", encoding="utf-8", newline="") as dst,
        tqdm(total=signature[0], desc="Separate Twitter media", unit="B", unit_scale=True) as bar,
    ):
        reader = csv.DictReader(src)
        fields = list(reader.fieldnames)
        fields = [field for field in fields if field != "thumbnails"]
        if "videos" not in fields:
            fields.insert(fields.index("images") + 1, "videos")
        writer = csv.DictWriter(dst, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("CSV row does not match its header")
            urls = []
            for field in ("urls", "images", "videos", "thumbnails"):
                values = json.loads(row.get(field) or "[]")
                if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
                    raise ValueError(f"Invalid {field} list for tweet {row['tweet_id']}")
                urls.extend(values)
            media = partition_media(urls)
            counts["tweets"] += 1
            for field, values in media.items():
                counts["with_" + field] += bool(values)
                row[field] = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
            row.pop("thumbnails", None)
            writer.writerow(row)
            if counts["tweets"] % 1000 == 0:
                bar.update(src.buffer.tell() - bar.n)
                bar.set_postfix(tweets=counts["tweets"], refresh=False)
        bar.update(bar.total - bar.n)
    if not counts["tweets"]:
        raise ValueError("Empty input; refusing to replace tweets/raw.csv")
    if (path.stat().st_size, path.stat().st_mtime_ns) != signature:
        raise ValueError("Input changed during media upgrade")
    temporary.replace(path)
    tqdm.write(json.dumps(dict(counts), indent=2))
    return counts


def account_lookup(records):
    lookup = {}
    for person in records:
        for account in person.get("accounts", []):
            user_id = str(account.get("id", ""))
            if not user_id:
                continue
            value = {
                "name": person.get("name", ""),
                "bioguide": person.get("id", {}).get("bioguide", ""),
                "account_type": account.get("account_type", ""),
                "twitter": account.get("screen_name", "").lower(),
            }
            if (
                user_id in lookup
                and lookup[user_id]["bioguide"] != value["bioguide"]
            ):
                raise ValueError(
                    f"Conflicting bioguide mapping for account {user_id}"
                )
            lookup[user_id] = value
    return lookup


def parse_tweet(raw, archive_file, accounts):
    tweet_id = str(raw.get("id", ""))
    if not tweet_id.isdigit() or not isinstance(raw.get("text"), str):
        raise ValueError("missing/invalid tweet ID or text")
    text = raw["text"]
    user_id = str(raw.get("user_id", ""))
    screen_name = raw.get("screen_name", "")
    metadata = accounts.get(user_id, {})
    rt = RT_RE.match(text)
    link = raw.get("link", "")
    target = STATUS_RE.match(link)
    is_retweet = bool(rt and target and target.group(2) != tweet_id)
    row = dict(
        tweet_id=tweet_id,
        posted_at=raw.get("time", ""),
        name=metadata.get("name", screen_name),
        twitter=screen_name.lower(),
        text=text,
        user_id=user_id,
        account_bioguide=metadata.get("bioguide", ""),
        account_type=metadata.get("account_type", ""),
        tweet_url=f"https://twitter.com/{screen_name}/status/{tweet_id}",
        archive_link=link,
        is_retweet=is_retweet,
        retweet_id=target.group(2) if is_retweet else "",
        source=raw.get("source", ""),
        archive_file=archive_file,
    )
    # This includes the original post's media for RTs, and appended QT media.
    row.update(extract_media(text))
    return row


def missing_content_target(row):
    """Find an empty inline RT/QT body with an unambiguous local target."""
    text = row["text"]
    markers = list(QT_RE.finditer(text))
    if markers:
        marker = markers[-1]
        targets = {
            match.group(2) for match in STATUS_RE.finditer(text[:marker.start()])
            if match.group(1).lower() == marker.group(1).lower()
        }
        target = next(iter(targets)) if len(targets) == 1 else ""
    else:
        marker = RT_RE.match(text)
        target = row["retweet_id"]
    if marker and target and not URL_RE.sub("", text[marker.end():]).strip():
        return target, marker.end()
    return None


def recover_inline_text(path, candidates):
    """Recover only absent RT/QT bodies from originals present in this CSV."""
    targets = {target for _, target, _ in candidates.values()}
    originals = {}
    if not targets:
        return {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            if row["tweet_id"] in targets:
                originals[row["tweet_id"]] = row["text"]
    fixes = {}
    for tweet_id, (text, target, offset) in candidates.items():
        original = originals.get(target, "")
        if not URL_RE.sub("", original).strip():
            continue
        extra = [u for u in extract_urls(text[offset:]) if u not in original]
        fixes[tweet_id] = text[:offset] + original + (" " + " ".join(extra) if extra else "")
    return fixes


def write_reduced_tweets(source, destination, fixes):
    count = 0
    with source.open() as src, destination.open("w", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in csv.DictReader(src):
            if row["tweet_id"] in fixes:
                row["text"] = fixes[row["tweet_id"]]
                for field, values in extract_media(row["text"]).items():
                    row[field] = json.dumps(unique(json.loads(row[field]) + values))
            writer.writerow(row)
            count += 1
    return count


def compile_tweets(
    data_dir, work_dir, workers=12, revision=REVISION
):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    output = data_dir / "tweets" / "raw.csv"
    if output.exists():
        with output.open(encoding="utf-8", newline="") as stream:
            fields = csv.DictReader(stream).fieldnames
        if fields and ("videos" not in fields or "thumbnails" in fields):
            upgrade_media_columns(output, work_dir)
            return
        print(f"{output} already exists, skipping.", flush=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(work_dir)
    cache = work_dir / "archive_cache"
    records = get_json(
        f"https://raw.githubusercontent.com/alexlitel/congresstweets-automator/{ACCOUNTS_REVISION}/data/historical-users-filtered.json",
        cache / ACCOUNTS_REVISION / "accounts.json",
    )
    accounts = account_lookup(records)
    tree = get_json(
        f"https://api.github.com/repos/alexlitel/congresstweets/git/trees/{revision}?recursive=1",
        cache / revision / "tree.json",
    )
    if tree.get("truncated"):
        raise ValueError("GitHub tree was truncated")
    files = sorted(
        x["path"]
        for x in tree["tree"]
        if re.fullmatch(r"data/\d{4}-\d{2}-\d{2}\.json", x["path"])
    )
    if not files:
        raise ValueError("No daily archives found")

    def fetch(path):
        return path, get_json(
            f"https://raw.githubusercontent.com/alexlitel/congresstweets/{revision}/{path}",
            cache / revision / path,
        )

    seen, counts, account_counts = set(), Counter(), Counter()
    invalid = []
    missing = {}
    tmp = work_dir / "tweets.csv"
    start = time.monotonic()
    with (
        tmp.open("w", newline="", encoding="utf-8") as stream,
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        # Bound prefetched responses rather than buffering the entire archive.
        for offset in range(0, len(files), workers * 4):
            for path, daily in pool.map(
                fetch, files[offset : offset + workers * 4]
            ):
                counts["raw_top_level_rows"] += len(daily)
                expanded = [
                    raw for entry in daily for raw in unpack_records(entry)
                ]
                counts["unpacked_extra_rows"] += len(expanded) - len(daily)
                for index, raw in enumerate(expanded):
                    counts["raw_rows"] += 1
                    try:
                        row = parse_tweet(raw, path, accounts)
                    except ValueError as error:
                        invalid.append(
                            {"file": path, "row": index, "reason": str(error)}
                        )
                        continue
                    if row["tweet_id"] in seen:
                        counts["duplicate_ids"] += 1
                        continue
                    target = missing_content_target(row)
                    if target:
                        missing[row["tweet_id"]] = (row["text"], *target)
                    seen.add(row["tweet_id"])
                    counts["tweets"] += 1
                    counts["retweets"] += row["is_retweet"]
                    counts["quotes"] += bool(QT_RE.search(row["text"]))
                    counts["mapped_bioguide"] += bool(row["account_bioguide"])
                    year = row["posted_at"][:4]
                    account_counts[(row["user_id"], row["twitter"], year)] += 1
                    for field in LIST_FIELDS:
                        counts["with_" + field] += bool(row[field])
                        row[field] = json.dumps(
                            row[field],
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    writer.writerow(row)
            print(
                f"{min(offset + workers * 4, len(files))}/{len(files)} files; "
                f"{counts['tweets']:,} tweets; "
                f"{time.monotonic() - start:.0f}s",
                flush=True,
            )
    if not counts["tweets"]:
        raise ValueError("Empty output; refusing to replace tweets/raw.csv")
    fixes = recover_inline_text(tmp, missing)
    counts["recovered_inline_text"] = len(fixes)
    if fixes:
        resolved = work_dir / "tweets_resolved.csv"
        if write_reduced_tweets(tmp, resolved, fixes) != counts["tweets"]:
            raise ValueError("Row count changed during inline text recovery")
        resolved.replace(tmp)
    tmp.replace(output)
    with (work_dir / "tweet_account_counts.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "user_id",
                "twitter",
                "year",
                "tweets",
                "bioguide",
                "name",
                "account_type",
            ]
        )
        for (uid, handle, year), n in sorted(account_counts.items()):
            meta = accounts.get(uid, {})
            writer.writerow(
                [
                    uid,
                    handle,
                    year,
                    n,
                    meta.get("bioguide", ""),
                    meta.get("name", ""),
                    meta.get("account_type", ""),
                ]
            )
    report = dict(
        source="alexlitel/congresstweets",
        revision=revision,
        accounts_revision=ACCOUNTS_REVISION,
        daily_files=len(files),
        first_file=files[0],
        last_file=files[-1],
        counts=dict(counts),
        invalid_rows=invalid,
        list_fields=LIST_FIELDS,
        retweet_resolution=(
            "Text and RT/QT media are already embedded by the upstream "
            "collector; no external lookups."
        ),
    )
    (work_dir / "tweets_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


def audit_handles(data_dir, work_dir):
    """Compare study handles with ID-backed archive account mappings."""
    data_dir = Path(data_dir)
    work_dir = Path(work_dir)
    if (
        not (work_dir / "tweet_account_counts.csv").exists()
        or not (data_dir / "representatives.csv").exists()
    ):
        return
    with (data_dir / "representatives.csv").open() as stream:
        reps = list(csv.DictReader(stream))
    existing = defaultdict(set)
    names = {}
    for rep in reps:
        handles = json.loads(rep["twitter"])
        existing[rep["bioguide"]].update(handle.lower() for handle in handles)
        names[rep["bioguide"]] = rep["name"]
    observed = defaultdict(Counter)
    types = {}
    with (work_dir / "tweet_account_counts.csv").open() as stream:
        for row in csv.DictReader(stream):
            key = (row["bioguide"], row["user_id"], row["twitter"])
            observed[key]["all"] += int(row["tweets"])
            if "2019" <= row["year"] <= "2022":
                observed[key]["study"] += int(row["tweets"])
            types[key] = row["account_type"]
    fields = [
        "bioguide",
        "name",
        "user_id",
        "twitter",
        "account_type",
        "already_mapped",
        "tweets",
        "tweets_2019_2022",
    ]
    additions = []
    with (work_dir / "handle_audit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key, counts in sorted(observed.items()):
            bio, uid, handle = key
            if bio not in existing:
                continue
            mapped = handle in existing[bio]
            result = dict(
                bioguide=bio,
                name=names[bio],
                user_id=uid,
                twitter=handle,
                account_type=types[key],
                already_mapped=mapped,
                tweets=counts["all"],
                tweets_2019_2022=counts["study"],
            )
            writer.writerow(result)
            if not mapped:
                additions.append(result)
    report = {
        "study_reps": len(existing),
        "existing_handles": sum(map(len, existing.values())),
        "additional_observed_handles": len(
            {(r["bioguide"], r["twitter"]) for r in additions}
        ),
        "additional_account_handle_pairs": len(additions),
        "reps_with_additional_handles": len(
            {r["bioguide"] for r in additions}
        ),
        "additional_tweets_2019_2022": sum(
            r["tweets_2019_2022"] for r in additions
        ),
        "largest_additions": sorted(
            additions, key=lambda r: r["tweets_2019_2022"], reverse=True
        )[:20],
    }
    (work_dir / "handle_audit_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print("Handle audit:", json.dumps(report, indent=2), flush=True)
