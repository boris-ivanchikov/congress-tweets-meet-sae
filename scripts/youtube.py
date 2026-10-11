"""Fetch YouTube oEmbed metadata into data/tweets/youtube.csv.

Run: python scripts/youtube.py
Read the union of YouTube URLs in raw.csv and resolved.csv without changing either.
Fetch and write once per video ID, using canonical youtube.com/watch URLs.
"""

import csv
import hashlib
import heapq
import json
import random
import re
import sqlite3
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/tweets/raw.csv'
RESOLVED = ROOT / 'data/tweets/resolved.csv'
OUTPUT = ROOT / 'data/tweets/youtube.csv'
WORKERS = 8
RPS = 4.0
ATTEMPTS = 3
TIMEOUT = (5, 10)
MAX_COOLDOWN = 60
RETRYABLE = {408, 429, 500, 502, 503, 504}
VIDEO_ID = re.compile(r'[A-Za-z0-9_-]{11}\Z')


def youtube_url(url):
    try:
        parsed = urlsplit(url)
        name = (parsed.hostname or '').lower()
        return parsed.scheme in {'http', 'https'} and (
            name in {'youtu.be', 'youtube.com', 'youtube-nocookie.com'}
            or name.endswith(('.youtube.com', '.youtube-nocookie.com'))
        )
    except ValueError:
        return False


def video_id(url):
    if not youtube_url(url):
        return ''
    parsed = urlsplit(url)
    parts = parsed.path.strip('/').split('/')
    if parsed.hostname.lower() == 'youtu.be':
        candidate = parts[0]
    elif parts[0] == 'watch':
        candidate = parse_qs(parsed.query).get('v', [''])[0]
    elif len(parts) >= 2 and parts[0] in {'shorts', 'live', 'embed', 'v'}:
        candidate = parts[1]
    else:
        return ''
    return candidate if VIDEO_ID.fullmatch(candidate) else ''


def signatures():
    return [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in (RAW, RESOLVED)]


def collect_urls():
    with RESOLVED.open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['short_url', 'resolved_url', 'error']:
            raise ValueError('resolved.csv must be the short_url,resolved_url,error mapping table')
        mapping, resolved_urls = {}, set()
        for row in reader:
            if youtube_url(row['short_url']):
                resolved_urls.add(row['short_url'])
            if row['resolved_url'] and not row['error']:
                mapping[row['short_url']] = row['resolved_url']
                if youtube_url(row['resolved_url']):
                    resolved_urls.add(row['resolved_url'])
    urls, original = set(resolved_urls), set()
    rows = tweets = 0
    with RAW.open(encoding='utf-8', newline='') as f, tqdm(
        total=RAW.stat().st_size, desc='Scan tweets', unit='B', unit_scale=True,
    ) as bar:
        for rows, row in enumerate(csv.DictReader(f), 1):
            values = json.loads(row['urls'] or '[]')
            if not isinstance(values, list) or any(not isinstance(u, str) for u in values):
                raise ValueError(f'Invalid urls in tweet row {rows}')
            matched = False
            for url in values:
                if youtube_url(url):
                    original.add(url)
                    urls.add(url)
                    matched = True
                effective = mapping.get(url, url)
                if youtube_url(effective):
                    urls.add(effective)
                    matched = True
            tweets += matched
            if rows % 10000 == 0:
                bar.update(f.buffer.tell() - bar.n)
        bar.update(bar.total - bar.n)
    parsed = {url: video_id(url) for url in urls}
    counts = dict(tweets=rows, tweets_with_youtube=tweets,
                  distinct_raw_youtube_urls=len(original),
                  distinct_resolved_youtube_urls=len(resolved_urls),
                  distinct_youtube_urls=len(urls),
                  distinct_videos=len(set(parsed.values()) - {''}),
                  non_video_urls=sum(not value for value in parsed.values()))
    raw_ids = {video_id(url) for url in original} - {''}
    all_ids = set(parsed.values()) - {''}
    counts['videos_added_by_resolved'] = len(all_ids - raw_ids)
    tqdm.write(json.dumps(counts, indent=2))
    tqdm.write(f"Found {len(urls):,} URL variants; {len(all_ids):,} distinct videos to fetch/export; "
               f"skipping {counts['non_video_urls']:,} unsupported/non-video URLs.")
    return {f'https://www.youtube.com/watch?v={vid}': vid for vid in all_ids}


class Fetcher:
    def __init__(self):
        self.local = threading.local()
        self.sessions = []

    def fetch(self, vid):
        if not hasattr(self.local, 'session'):
            self.local.session = requests.Session()
            self.local.session.headers['User-Agent'] = 'CongressTweets-YouTubeMetadata/1.0'
            self.sessions.append(self.local.session)
        try:
            with self.local.session.get('https://www.youtube.com/oembed', params={
                'url': f'https://www.youtube.com/watch?v={vid}', 'format': 'json',
            }, timeout=TIMEOUT) as response:
                if response.status_code == 200:
                    try:
                        data = response.json()
                        title, channel, thumb = data.get('title'), data.get('author_name'), data.get('thumbnail_url')
                        if not isinstance(title, str) or not title.strip() or not isinstance(channel, str) or not channel.strip() or not isinstance(thumb, str) or not thumb.startswith(('https://', 'http://')):
                            raise ValueError('missing title, channel, or thumbnail')
                        return title, channel, thumb, '', False, 0
                    except (ValueError, AttributeError):
                        return '', '', '', 'invalid oEmbed response', False, 0
                retry_after = response.headers.get('Retry-After', '')
                delay = 0
                if retry_after:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        try:
                            delay = parsedate_to_datetime(retry_after).timestamp() - time.time()
                        except (ValueError, TypeError, OverflowError):
                            pass
                return '', '', '', f'HTTP {response.status_code}', response.status_code in RETRYABLE, max(0, delay)
        except requests.RequestException as error:
            permanent = isinstance(error, requests.exceptions.SSLError)
            return '', '', '', type(error).__name__, not permanent, 0

    def close(self):
        for session in self.sessions:
            session.close()


def fetch_metadata(ids, cache):
    cache.execute('CREATE TABLE IF NOT EXISTS metadata (video_id TEXT PRIMARY KEY, title TEXT, channel TEXT, thumb_url TEXT, error TEXT)')
    columns = {row[1] for row in cache.execute('PRAGMA table_info(metadata)')}
    if 'channel' not in columns:
        cache.execute('ALTER TABLE metadata ADD COLUMN channel TEXT')
    results = {vid: (title, channel or '', thumb, error)
               for vid, title, channel, thumb, error in cache.execute(
                   'SELECT video_id, title, channel, thumb_url, error FROM metadata')
               if vid in ids and (channel is not None or error)}
    remaining = list(ids - results.keys())
    random.shuffle(remaining)
    queue = [(0, i, vid, 0) for i, vid in enumerate(remaining)]
    heapq.heapify(queue)
    seq = len(queue)
    fetcher = Fetcher()
    executor = ThreadPoolExecutor(max_workers=WORKERS)
    pending = {}
    next_request = cooldown = 0
    interval = 1 / RPS
    blocked = ''
    failures = sum(bool(result[3]) for result in results.values())
    transient_streak = 0

    def save(vid, result):
        nonlocal failures
        failures += bool(result[3]) - bool(results.get(vid, ("", "", "", ""))[3])
        results[vid] = result[:4]
        cache.execute('INSERT OR REPLACE INTO metadata (video_id, title, channel, thumb_url, error) VALUES (?, ?, ?, ?, ?)', (vid, *result[:4]))

    try:
        with tqdm(total=len(ids), initial=len(results), desc='YouTube oEmbed', unit='video') as bar:
            while queue or pending:
                now = time.monotonic()
                if blocked:
                    while queue:
                        _, _, vid, _ = heapq.heappop(queue)
                        # Deferred outcomes are exported but retried on a later run.
                        results[vid] = ('', '', '', blocked)
                        failures += 1
                        bar.update()
                elif queue and len(pending) < WORKERS and now >= max(queue[0][0], next_request, cooldown):
                    _, _, vid, attempt = heapq.heappop(queue)
                    pending[executor.submit(fetcher.fetch, vid)] = (vid, attempt)
                    next_request = now + interval
                if pending:
                    done, _ = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                else:
                    time.sleep(0.1)
                    done = set()
                for future in done:
                    vid, attempt = pending.pop(future)
                    result = future.result()
                    title, channel, thumb, error, transient, retry_after = result
                    transient_streak = transient_streak + 1 if transient else 0
                    if transient_streak >= 12:
                        blocked = 'deferred: 12 consecutive transient failures; check network or VPN'
                    delay = max(retry_after, random.uniform(1, 2 ** (attempt + 1)))
                    if error == 'HTTP 429':
                        interval = min(5, interval * 2)
                        cooldown = max(cooldown, time.monotonic() + delay)
                    if transient and retry_after > MAX_COOLDOWN:
                        blocked = f'deferred: server requested {retry_after:g}s cooldown'
                    if transient and attempt + 1 < ATTEMPTS and not blocked:
                        seq += 1
                        heapq.heappush(queue, (time.monotonic() + delay, seq, vid, attempt + 1))
                    else:
                        save(vid, result)
                        bar.update()
                cache.commit()
                bar.set_postfix(success=len(results) - failures, errors=failures, active=len(pending), refresh=False)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        for future, (vid, _) in pending.items():
            if not future.cancelled() and future.exception() is None:
                result = future.result()
                if not result[4]:
                    save(vid, result)
        fetcher.close()
        cache.commit()
    return results


def write_results(path, urls, results):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['url', 'title', 'channel', 'thumb_url', 'error'])
        for url, vid in sorted(urls.items()):
            result = results[vid]
            writer.writerow([url, *result])


def main():
    if OUTPUT.exists():
        tqdm.write(f'{OUTPUT} already exists; delete it to rerun.')
        return
    csv.field_size_limit(16 * 1024 * 1024)
    signature = signatures()
    urls = collect_urls()
    key = hashlib.sha256(json.dumps(signature).encode()).hexdigest()[:16]
    cache_path = Path(tempfile.gettempdir()) / f'congress-youtube-{key}.sqlite3'
    tqdm.write(f'Resume cache: {cache_path}')
    with sqlite3.connect(cache_path) as cache:
        results = fetch_metadata(set(urls.values()) - {''}, cache)
    if signatures() != signature:
        raise ValueError('Inputs changed during metadata fetching')
    with tempfile.TemporaryDirectory(prefix='congress-youtube-') as work:
        partial = Path(work) / 'youtube.csv'
        write_results(partial, urls, results)
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        partial.replace(OUTPUT)
    errors = Counter(result[3] for result in results.values() if result[3])
    tqdm.write(f'Saved {len(urls):,} URL rows to {OUTPUT}')
    for reason, count in errors.most_common():
        tqdm.write(f'{reason}: {count:,} videos')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        tqdm.write('Interrupted. Completed results are cached; rerun to resume.')
        raise SystemExit(130)
    except (OSError, ValueError, sqlite3.Error) as error:
        tqdm.write(f'Error: {error}')
        raise SystemExit(1)
