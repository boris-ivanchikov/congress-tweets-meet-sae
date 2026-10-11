"""Resolve known short URLs from raw.csv into a separate URL-results table.

Run with `python scripts/resolve.py`. Output columns are short_url,
resolved_url, error. raw.csv is read-only. Successes and failures are cached
in temporary storage for resuming. Article destinations are never requested.
"""

import csv
import hashlib
import json
import random
import shutil
import sqlite3
import tempfile
import threading
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import requests
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/tweets/raw.csv"
OUTPUT = ROOT / "data/tweets/resolved.csv"
WORKERS = 48
PER_HOST = 2
INITIAL_RPS = 2.0
MIN_RPS = 0.25
MAX_RPS = 4.0
RATE_DECREASE = 0.5
RATE_INCREASE = 0.25
RECOVERY_SUCCESSES = 10
BACKOFF_FACTOR = 2.0
MAX_BACKOFF = 5.0
MAX_HOST_ERRORS = 3
MAX_HOST_WAIT = 15.0
ATTEMPTS = 1
TIMEOUT = (3, 5)
MAX_HOPS = 12
REDIRECTS = {301, 302, 303, 307, 308}
RETRYABLE = {408, 429, 500, 502, 503, 504}

# Explicit hosts, not a short-path heuristic: /123 can be a real article URL.
# Add other branded shorteners here when encountered in the input.
SHORT_HOSTS = set("""
    t.co bit.ly bitly.com bitly.is j.mp goo.gl ow.ly buff.ly tinyurl.com
    tiny.cc is.gd v.gd trib.al dlvr.it ift.tt lnkd.in amzn.to fb.me fb.watch
    youtu.be pic.twitter.com instagr.am po.st spr.ly shorturl.at short.gy
    rebrand.ly rb.gy cutt.ly s.id
    45.wh.gov cs.pn spkrryan.us washex.am nydn.us dems.me on.nwlc.org
    bayareane.ws urbn.is sgfnow.co cnnmon.ie nyp.st thndr.me shar.es
    cor.bo bzfd.it read.bi kaiserf.am s.al.com pbpo.st pgj.cc 1.usa.gov
    toi.sr on.wusa9.com nbc4dc.com bsun.md nixle.us
    nyti.ms wapo.st hill.cm politi.co cnn.it reut.rs tmsnrt.rs apne.ws n.pr
    nbcnews.to on.msnbc.com on.today.com cbsn.ws cnb.cx fxn.ws foxs.pt
    on.wsj.com on.mktw.net usat.ly ti.me huff.to huffp.st huffpo.st bloom.bg
    buswk.co for.tn axios.link abcn.ws abc7.la lat.ms w.bos.gl bos.gl
    chtrib.com sacb.ee hrld.us nj-ne.ws dallasne.ws dpo.st azc.cc states.mn
    ksl.fm slnm.us slate.me nym.ag theatln.tc es.pn snpy.tv gu.com ind.pn
    econ.st bbc.in dailym.ai mol.im fw.to ampr.gs
    go.nasa.gov go.nps.gov go.uscis.gov go.ice.gov go.va.gov
    go.wh.gov wh.gov sen.gov wp.me on.fb.me conta.cc eepurl.com act.ly
""".split())


def host(url):
    try:
        parsed = urlsplit(url)
        if parsed.scheme.lower() not in {"http", "https"}:
            return ""
        return (parsed.hostname or "").lower().removeprefix("www.")
    except ValueError:
        return ""


def is_short(url):
    name = host(url)
    return name in SHORT_HOSTS or name.endswith((".short.gy", ".trib.al"))


def url_list(row, number):
    try:
        values = json.loads(row["urls"] or "[]")
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Row {number}: invalid urls JSON") from error
    if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
        raise ValueError(f"Row {number}: urls must be an array of strings")
    return values


def scan(path):
    short_urls, count = set(), 0
    with path.open(encoding="utf-8", newline="") as stream, tqdm(
        total=path.stat().st_size, desc="Scan tweets", unit="B", unit_scale=True,
    ) as bar:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields or "urls" not in fields:
            raise ValueError("Input must contain a urls column")
        for count, row in enumerate(reader, 1):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"Row {count}: columns do not match the header")
            values = url_list(row, count)
            short_urls.update(url for url in values if is_short(url))
            if count % 1000 == 0:
                bar.update(stream.buffer.tell() - bar.n)
                bar.set_postfix(rows=count, short=len(short_urls), refresh=False)
        bar.update(bar.total - bar.n)
    return fields, count, short_urls


class HostPacer:
    """Shared request-start pacing for all workers and redirect hops on a host."""
    def __init__(self):
        self.lock = threading.Lock()
        self.rps = INITIAL_RPS
        self.next_request = 0.0
        self.cooldown = 0.0
        self.successes = 0
        self.streak = 0
        self.errors = 0
        self.requests = 0

    def delay(self):
        with self.lock:
            return max(0.0, max(self.next_request, self.cooldown) - time.monotonic())

    def reserve(self):
        """Called only by the scheduler when this host is ready."""
        self.next_request = time.monotonic() + 1.0 / self.rps
        self.requests += 1

    def feedback(self, transient=False, success=False, retry_after=0.0):
        with self.lock:
            now = time.monotonic()
            if transient:
                self.errors += 1
                self.streak += 1
                self.successes = 0
                self.rps = max(MIN_RPS, self.rps * RATE_DECREASE)
                backoff = min(MAX_BACKOFF, BACKOFF_FACTOR * 2 ** min(self.streak - 1, 10))
                # Shared cooldown, including Retry-After, applies to new URLs too.
                self.cooldown = max(self.cooldown, now + max(retry_after, random.uniform(backoff / 2, backoff)))
                self.next_request = max(self.next_request, now + 1.0 / self.rps)
            elif success:
                self.streak = 0
                self.successes += 1
                if self.successes >= RECOVERY_SUCCESSES:
                    self.rps = min(MAX_RPS, self.rps + RATE_INCREASE)
                    self.successes = 0
            else:
                # A 404 or non-redirect response does not imply overload.
                self.successes = 0
                self.streak = 0


def retry_after_seconds(value):
    if not value:
        return 0.0
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            return 0.0
    return max(0.0, seconds)


@dataclass
class Job:
    original: str
    current: str
    seen: set = field(default_factory=set)
    hops: int = 0
    attempts: int = 0
    retries: int = 0


@dataclass
class Attempt:
    target: str = ""
    error: str = ""
    transient: bool = False
    retry_after: float = 0.0


class Resolver:
    """Workers perform exactly one HTTP request; only the scheduler waits."""
    def __init__(self):
        self.local = threading.local()
        self.sessions = []
        self.stop = threading.Event()

    def session(self):
        if not hasattr(self.local, "session"):
            session = requests.Session()
            session.headers["User-Agent"] = "CongressTweets-LinkResolver/1.0"
            self.local.session = session
            self.sessions.append(session)
        return self.local.session

    def request(self, url):
        if self.stop.is_set():
            return Attempt(error="interrupted")
        try:
            with self.session().get(
                url, allow_redirects=False, stream=True, timeout=TIMEOUT,
            ) as response:
                status = response.status_code
                location = response.headers.get("Location")
                if status in REDIRECTS and location:
                    target = urljoin(response.url, location.strip())
                    return Attempt(target=target) if host(target) else Attempt(error="invalid redirect")
                error = f"HTTP {status}" if status != 200 else "no redirect"
                return Attempt(error=error, transient=status in RETRYABLE,
                               retry_after=retry_after_seconds(response.headers.get("Retry-After")))
        except requests.RequestException as exception:
            permanent = isinstance(exception, (
                requests.exceptions.SSLError, requests.exceptions.InvalidURL,
                requests.exceptions.InvalidSchema, requests.exceptions.MissingSchema,
            ))
            return Attempt(error=type(exception).__name__, transient=not permanent)
        except ValueError:
            return Attempt(error="invalid URL")

    def close(self):
        for session in self.sessions:
            session.close()


def print_results(stats, failures):
    rows = [["Original host", "Success", "Errors", "Error %"]]
    for name, counts in sorted(stats.items(), key=lambda item: (-item[1]["failed"], item[0])):
        attempted = counts["resolved"] + counts["failed"]
        rate = f"{100 * counts['failed'] / attempted:.1f}%" if attempted else "—"
        rows.append([name, str(counts["resolved"]), str(counts["failed"]), rate])
    tables = [("Total results by original short-link host (including cache)", rows)]
    if failures:
        reasons = [["Original host", "Failing redirect host", "Reason", "Count"]]
        reasons.extend([origin, failed_host, reason, str(count)]
                       for (origin, failed_host, reason), count in failures.most_common())
        tables.append(("Failure reasons", reasons))
    for title, table in tables:
        widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
        tqdm.write("\n" + title)
        for i, row in enumerate(table):
            tqdm.write("  ".join(value.ljust(width) for value, width in zip(row, widths)))
            if i == 0:
                tqdm.write("  ".join("-" * width for width in widths))


def resolve_urls(urls, cache):
    resolved = {url: target for url, target in cache.execute("SELECT url, target FROM redirects")
                if url in urls and host(target) and not is_short(target)}
    cache.execute("CREATE TABLE IF NOT EXISTS failures (url TEXT PRIMARY KEY, error TEXT NOT NULL, error_host TEXT NOT NULL)")
    previous_errors = {url: (error, error_host) for url, error, error_host in
                       cache.execute("SELECT url, error, error_host FROM failures")
                       if url in urls and url not in resolved}
    remaining = list(urls - resolved.keys() - previous_errors.keys())
    random.shuffle(remaining)
    queues = defaultdict(deque)
    for url in remaining:
        queues[host(url)].append(Job(url, url, {url}))
    ready = deque(queues)
    active = Counter()
    pacers = defaultdict(HostPacer)
    stats, failures = defaultdict(Counter), Counter()
    for url in resolved:
        stats[host(url)]["resolved"] += 1
    for url, (error, error_host) in previous_errors.items():
        stats[host(url)]["failed"] += 1
        failures[(host(url), error_host, error)] += 1
    tqdm.write(f"Loaded totals: success={len(resolved):,}; errors={len(previous_errors):,}; unprocessed={len(remaining):,}")
    for name in queues:
        stats[name]
    failed = len(previous_errors)
    resolver = Resolver()
    executor = ThreadPoolExecutor(max_workers=WORKERS)
    pending = {}
    stopped_hosts = {}

    def enqueue(job, retry=False):
        name = host(job.current)
        if not queues[name]:
            ready.append(name)
        if retry:
            queues[name].appendleft(job)
        else:
            queues[name].append(job)

    def fill_workers():
        for _ in range(len(ready)):
            if len(pending) >= WORKERS:
                break
            name = ready.popleft()
            if name in stopped_hosts:
                while queues[name]:
                    record(queues[name].popleft(), error=stopped_hosts[name])
                continue
            if active[name] < PER_HOST and pacers[name].delay() <= 0:
                job = queues[name].popleft()
                pacers[name].reserve()
                job.attempts += 1
                pending[executor.submit(resolver.request, job.current)] = job
                active[name] += 1
            if queues[name]:
                ready.append(name)

    def record(job, error="", target=""):
        nonlocal failed
        name = host(job.original)
        if error:
            failed += 1
            stats[name]["failed"] += 1
            error_host = host(job.current)
            failures[(name, error_host, error)] += 1
            cache.execute("INSERT OR REPLACE INTO failures VALUES (?, ?, ?)",
                          (job.original, error, error_host))
        else:
            stats[name]["resolved"] += 1
            resolved[job.original] = target
            cache.execute("INSERT OR REPLACE INTO redirects VALUES (?, ?)", (job.original, target))
            cache.execute("DELETE FROM failures WHERE url = ?", (job.original,))

    def consume(future):
        job = pending.pop(future)
        name = host(job.current)
        active[name] -= 1
        result = future.result()
        if result.error == "interrupted":
            return False
        pacers[name].feedback(transient=result.transient, success=bool(result.target),
                              retry_after=result.retry_after)
        if result.transient and (pacers[name].streak >= MAX_HOST_ERRORS or result.retry_after > MAX_HOST_WAIT):
            stopped_hosts[name] = (
                f"host skipped: repeated transient failures ({result.error}); URL not fully tested"
                if result.retry_after <= MAX_HOST_WAIT else
                f"host skipped: Retry-After {result.retry_after:g}s exceeds wait budget; URL not fully tested"
            )
        if result.target:
            job.hops += 1
            if host(result.target) == "go.usa.gov":
                job.current = result.target
                record(job, error="excluded host: go.usa.gov")
            elif not is_short(result.target):
                record(job, target=result.target)
            elif result.target in job.seen:
                record(job, error="redirect loop")
            elif job.hops >= MAX_HOPS:
                record(job, error="too many redirects")
            else:
                job.current = result.target
                job.seen.add(result.target)
                job.attempts = 0
                enqueue(job)
                return False
        elif result.transient and job.attempts < ATTEMPTS:
            job.retries += 1
            enqueue(job, retry=True)
            return False
        else:
            record(job, error=result.error)
        return True

    try:
        with tqdm(total=len(urls), initial=len(resolved) + failed, desc="Resolve short URLs", unit="url") as bar:
            while pending or ready:
                fill_workers()
                bar.update(len(resolved) + failed - bar.n)
                delays = [pacers[name].delay() for name in ready if active[name] < PER_HOST]
                timeout = min(0.5, min(delays, default=0.5))
                if pending:
                    done, _ = wait(pending, timeout=max(0.01, timeout), return_when=FIRST_COMPLETED)
                else:
                    resolver.stop.wait(max(0.01, timeout))
                    done = set()
                for future in done:
                    consume(future)
                bar.update(len(resolved) + failed - bar.n)
                cache.commit()
                bar.set_postfix(success=len(resolved), errors=failed,
                                active=len(pending),
                                slowed=sum(p.rps < INITIAL_RPS for p in pacers.values()),
                                refresh=False)
                bar.refresh()
    finally:
        resolver.stop.set()
        executor.shutdown(wait=True, cancel_futures=True)
        # Save completed requests; queued retries/redirects remain unprocessed.
        for future in list(pending):
            if not future.cancelled() and future.exception() is None:
                consume(future)
        resolver.close()
        cache.commit()
        print_results(stats, failures)
        tqdm.write(f"Total success={len(resolved):,}; total errors={failed:,}; unprocessed={len(urls) - len(resolved) - failed:,}")
    return resolved


def write_results(destination, urls, resolved, cache):
    errors = dict(cache.execute("SELECT url, error FROM failures"))
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["short_url", "resolved_url", "error"])
        for url in tqdm(sorted(urls), desc="Write URL results", unit="url"):
            target = resolved.get(url, "")
            error = "" if target else errors.get(url, "")
            if not target and not error:
                raise ValueError(f"Missing resolution outcome for {url}")
            writer.writerow([url, target, error])


def main():
    if OUTPUT.exists():
        tqdm.write(f"{OUTPUT} already exists; delete it to rerun.")
        return
    csv.field_size_limit(16 * 1024 * 1024)
    signature = (INPUT.stat().st_size, INPUT.stat().st_mtime_ns)
    _, count, urls = scan(INPUT)
    key = hashlib.sha256(f"{INPUT}:{signature}".encode()).hexdigest()[:16]
    cache_path = Path(tempfile.gettempdir()) / f"congress-short-urls-{key}.sqlite3"
    tqdm.write(f"{count:,} tweets; {len(urls):,} distinct short URLs. Resume cache: {cache_path}")
    cache = sqlite3.connect(cache_path)
    try:
        cache.execute("CREATE TABLE IF NOT EXISTS redirects (url TEXT PRIMARY KEY, target TEXT NOT NULL)")
        resolved = resolve_urls(urls, cache)
        if (INPUT.stat().st_size, INPUT.stat().st_mtime_ns) != signature:
            raise ValueError("Input changed during resolution; rerun against the new input")
        with tempfile.TemporaryDirectory(prefix="congress-resolved-") as work:
            partial = Path(work) / "resolved.csv"
            write_results(partial, urls, resolved, cache)
            if (INPUT.stat().st_size, INPUT.stat().st_mtime_ns) != signature:
                raise ValueError("Input changed while writing")
            OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(partial), OUTPUT)
            except BaseException:
                OUTPUT.unlink(missing_ok=True)
                raise
    finally:
        cache.close()
    tqdm.write(f"Saved {len(urls):,} URL results to {OUTPUT}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        tqdm.write("Interrupted. Completed successes and errors remain cached; rerun to resume.")
        raise SystemExit(130)
    except (OSError, ValueError, sqlite3.Error) as error:
        tqdm.write(f"Error: {error}")
        raise SystemExit(1)
