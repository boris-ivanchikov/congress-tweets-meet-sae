"""Build formated.csv offline; preserve all raw.csv fields except text.

Run: python scripts/format.py
Uses resolved.csv, youtube.csv and matches.csv. A reproducible sample report
and validation statistics go to temporary storage, printed on completion.
"""

import csv
import hashlib
import html
import json
import random
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import date
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlencode, urlsplit, urlunsplit

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/tweets"
INPUT = DATA / "raw.csv"
OUTPUT = DATA / "formated.csv"
URL = re.compile(
    r"(?:https?://|(?<![\w@])(?:www\.|(?:[a-z0-9-]+\.)+"
    r"(?:com|org|net|gov|edu|us|co|io|ly)/))[^\s<>\"\\]*", re.I,
)
RT = re.compile(r"^RT @([A-Za-z0-9_]+)\s*:?[ ]*")
QT = re.compile(r"(?<!\S)QT @([A-Za-z0-9_]+)\s+")
STATUS_PATH = re.compile(r"^/(?:([\w]+)/status(?:es)?|i/web/status)/(\d+)(?:/|$)")
BAD_TITLE = re.compile(
    r"^(?:404|403|access denied|page not found|just a moment|attention required|"
    r"request rejected|robot or human)", re.I,
)
GENERIC = set("index default home story stories article articles news newsroom media press "
              "release releases public politics opinion business us world local national "
              "international latest breaking update updates live video videos watch print "
              "amp html htm aspx asp php cfm feed rss search tags topics category archives".split())
ACRONYMS = set("us usa uk un eu gop fbi cia doj dhs hhs irs ice cbp epa fda cdc nih va "
               "nato nasa nyc dc isis daca lgbt lgbtq covid nafta usmca ppp sba".split())
MISC = {"instagram.com", "facebook.com", "fb.com", "zoom.us", "google.com",
        "docs.google.com", "forms.gle", "actblue.com", "secure.actblue.com",
        "winred.com", "secure.winred.com", "secure.anedot.com", "eventbrite.com",
        "congress.gov", "govtrack.us", "fec.gov", "opensecrets.org", "legiscan.com"}


def url_parts(value):
    try:
        return urlsplit(value if re.match(r"https?://", value, re.I) else "https://" + value)
    except ValueError:
        return urlsplit("")


@lru_cache(maxsize=100000)
def host(value):
    return (url_parts(value).hostname or "").lower().removeprefix("www.")


@lru_cache(maxsize=100000)
def status(value):
    p = url_parts(value)
    if host(value) not in {"twitter.com", "mobile.twitter.com", "x.com", "mobile.x.com"}:
        return None
    m = STATUS_PATH.match(p.path)
    return (m[1] or "", m[2]) if m else None


def split_url(value):
    # Keep punctuation outside the replacement, including balanced URL parentheses.
    tail = ""
    while value and (value[-1] in ".,;!:?" or
                     value[-1] == ")" and value.count(")") > value.count("(")):
        tail = value[-1] + tail
        value = value[:-1]
    return value, tail


@lru_cache(maxsize=100000)
def canonical(value):
    p = url_parts(value)
    query = [(k, v) for k, values in parse_qs(p.query, keep_blank_values=True).items()
             if not k.lower().startswith("utm_") and k.lower() not in
             {"fbclid", "gclid", "ref_src"} for v in values]
    return urlunsplit(("", host(value), p.path.rstrip("/"), urlencode(sorted(query)), ""))


@lru_cache(maxsize=100000)
def youtube_id(value):
    p = url_parts(value)
    h = host(value)
    if h == "youtu.be":
        result = p.path.strip("/").split("/")[0]
    elif h in {"youtube.com", "m.youtube.com", "youtube-nocookie.com"}:
        result = parse_qs(p.query).get("v", [""])[0]
        m = re.match(r"/(?:embed|shorts|live|v)/([^/]+)", p.path)
        if m:
            result = m[1]
    else:
        return ""
    return result if re.fullmatch(r"[\w-]{11}", result) else ""


@lru_cache(maxsize=100000)
def slug_title(value):
    """Extract a URL label, never assert that it is a publisher's headline."""
    p = url_parts(value)
    h = host(value)
    path = unquote(p.path)
    if h in MISC or any(h.endswith("." + x) for x in MISC):
        return ""
    if re.search(r"…|\.\.\.|Ã|Â|�", path):
        return ""
    if re.search(r"\.(?:pdf|docx?|xlsx?|pptx?|jpe?g|png|gif|mp[34]|zip|xml|json|csv|txt)$", path, re.I):
        return ""
    if re.search(r"/(?:tags?|topics?|categor(?:y|ies)|search|authors?|people|profiles?|"
                 r"events?|calendar|donate|donation|contact|about|privacy|terms)(?:/|$)", path, re.I):
        return ""
    parts = [x for x in path.split("/") if x]
    q = parse_qs(p.query)
    candidates = [v for k in ("title", "slug", "storytitle", "headline") for v in q.get(k, [])]
    candidates += [v.rsplit("/", 1)[-1] for k in ("p", "article", "story")
                   for v in q.get(k, []) if "-" in v and len(v) > 20]
    if parts:
        leaf = parts[-1]
        wrapper = re.fullmatch(r"(?:index|default)(?:\.(?:html?|aspx?))?|amp", leaf, re.I)
        identifier = re.fullmatch(r"(?:article_)?[a-f0-9-]{24,}(?:_story)?(?:\.html?)?|\d{5,}", leaf, re.I)
        candidates.append(parts[-2] if (wrapper or identifier) and len(parts) > 1 else leaf)
    for v in candidates:
        v = v.lower().strip()
        v = re.sub(r"\.(?:s?html?|aspx?|php|cfm|cms|amp|cnn)$", "", v)
        v = re.sub(r"[_-](?:story|article)$", "", v)
        if h == "stripes.com":
            v = re.sub(r"-\d\.\d{5,}$", "", v)
        if h == "wsj.com":
            v = re.sub(r"-[a-f0-9]{8}$", "", v)
        if h == "medium.com":
            v = re.sub(r"-[a-f0-9]{10,}$", "", v)
        v = re.sub(r"[_-]id(?:us|uk|ca)?[a-z0-9]+$", "", v)
        v = re.sub(r"[_-](?:n|rcna|rcn|cbs|b|a)\d{5,}$", "", v)
        v = re.sub(r"[_-][a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}$", "", v)
        v = re.sub(r"[_-][a-f0-9]{12,}$|^[0-9]{5,}[-_]|[-_]\d{5,}$", "", v)
        tokens = [x for x in re.split(r"[-_]+", v) if x]
        alpha = [x for x in tokens if re.fullmatch("[a-z]+", x)]
        if len(alpha) < 3 or len([x for x in alpha if x not in GENERIC]) < 2 or sum(map(len, alpha)) < 15:
            continue
        if any(len(x) >= 20 or re.fullmatch(r"[a-f0-9]{10,}", x) for x in tokens):
            continue
        if any(re.search("[a-z]", x) and re.search("[0-9]", x) and not
               re.fullmatch(r"\d+(?:st|nd|rd|th|s|k|m|b|bn|g)|(?:fy|covid|h|hr)\d+", x) for x in tokens):
            continue
        return " ".join(x.upper() if x in ACRONYMS else x.capitalize() for x in tokens)
    return ""


def read_rows(path, label):
    with path.open(newline="", encoding="utf-8") as stream, tqdm(
        total=path.stat().st_size, desc=label, unit="B", unit_scale=True,
    ) as bar:
        for i, row in enumerate(csv.DictReader(stream), 1):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"Invalid CSV row {i} in {path}")
            yield row
            if i % 2000 == 0:
                bar.update(stream.buffer.tell() - bar.n)
        bar.update(bar.total - bar.n)


def clean_metadata(value):
    value = re.sub(r"<[^>]+>", " ", html.unescape(value or ""))
    value = URL.sub(lambda m: "[link to " + (host(m[0]) or "website") + "]", value)
    return re.sub(r"\s+", " ", value).strip()


class Formatter:
    def __init__(self, redirects, articles, youtube, originals):
        self.redirects = redirects
        self.articles = articles
        self.youtube = youtube
        self.originals = originals

    @lru_cache(maxsize=100000)
    def resolve(self, value):
        seen = set()
        while value in self.redirects and value not in seen:
            seen.add(value)
            value = self.redirects[value]
        return value

    def format(self, row):
        events = Counter()
        images, videos = set(), set()
        inferred_images, inferred_videos = {}, set()
        seen = {row['tweet_id']}
        inline_handles = {m[1].lower() for m in QT.finditer(row['text'])}
        rt_marker = RT.match(row['text'])
        if rt_marker:
            inline_handles.add(rt_marker[1].lower())
        if row.get("retweet_id"):
            seen.add(row["retweet_id"])

        def media(value):
            p = url_parts(value)
            h = host(value)
            if h == "pbs.twimg.com":
                m = re.search(r"/(?:ext_tw_video_thumb|amplify_video_thumb)/(\d+)/", p.path)
                if m:
                    videos.add(m[1])
                elif "/tweet_video_thumb/" in p.path:
                    videos.add("gif:" + Path(p.path).stem)
                else:
                    images.add(re.sub(r"\.(?:jpg|jpeg|png|webp)(?::\w+)?$", "", p.path, flags=re.I))
                return True
            if h == "video.twimg.com":
                m = re.search(r"/(?:ext_tw_video|amplify_video)/(\d+)/", p.path)
                videos.add(m[1] if m else "gif:" + Path(p.path).stem)
                return True
            return False

        def add_media(record):
            for field in ("images", "videos"):
                for value in json.loads(record.get(field) or "[]"):
                    media(value)

        def link(value, depth, suppressed):
            original = value
            value = self.resolve(value)
            events['resolved_short_links'] += value != original
            if media(value):
                return ""
            target = status(value)
            if target:
                author, ident = target
                attachment = re.search(r'/(photo|video)/(\d+)(?:/|$)', url_parts(value).path)
                if attachment and ident in seen:
                    if attachment[1] == 'photo':
                        inferred_images[ident] = max(inferred_images.get(ident, 0), int(attachment[2]))
                    else:
                        inferred_videos.add(ident)
                    events['attachment_links_removed'] += 1
                    return ''
                if ident in suppressed:
                    events['quote_links_removed'] += 1
                    return ""
                if (ident not in seen and depth < 1 and ident in self.originals
                        and author.lower() not in inline_handles):
                    seen.add(ident)
                    record = self.originals[ident]
                    add_media(record)
                    body = render_record(record, depth + 1)
                    events['linked_tweets_expanded'] += 1
                    return f'[linked tweet by @{record["twitter"] or author}: "{body}"]'
                events['twitter_fallbacks'] += 1
                return f'[link to tweet by @{author}]' if author else '[link to tweet]'
            h = host(value) or "website"
            if h in {"twitter.com", "x.com", "mobile.twitter.com", "mobile.x.com"}:
                name = url_parts(value).path.strip("/")
                events['twitter_fallbacks'] += 1
                return f'[link to profile @{name}]' if re.fullmatch(r"[\w]{1,15}", name) and name not in {"home", "search", "intent", "i"} else '[link to Twitter]'
            y = self.youtube.get(youtube_id(value))
            if y:
                events['youtube_titles'] += 1
                suffix = f' by {y[1]}' if y[1] else ''
                return f'[link to video "{y[0]}"{suffix}]'
            a = self.articles.get(value) or self.articles.get(original) or self.articles.get(canonical(value))
            if a:
                title, author, publisher, published = a
                events['article_titles'] += 1
                details = f' by {author}' if author else ''
                if publisher and publisher.casefold() != author.casefold():
                    details += f', {publisher}'
                if published:
                    details += f', {published}'
                return f'[link to article "{title}"{details}]'
            label = slug_title(value)
            if label:
                events['url_labels'] += 1
                return f'[link to {h}; URL-derived text: "{label}"]'
            events['domain_fallbacks'] += 1
            return f'[link to {h}]'

        def replace_links(text, depth, suppressed, remove_spans=()):
            def replace(m):
                value, tail = split_url(m[0])
                if (m.start(), m.end()) in remove_spans:
                    events['quote_links_removed'] += 1
                    return tail
                return link(value, depth, suppressed) + tail
            return URL.sub(replace, text)

        def render_text(text, owner_id, depth, suppressed, nesting=0):
            if nesting < 12:
                for marker in QT.finditer(text):
                    prefix = text[:marker.start()]
                    candidates = []
                    for m in URL.finditer(prefix):
                        target = status(self.resolve(split_url(m[0])[0]))
                        if target and target[0].lower() == marker[1].lower() and target[1] != owner_id:
                            candidates.append((m, target[1]))
                    ids = {ident for _, ident in candidates}
                    if len(ids) == 1:
                        ident = next(iter(ids))
                        events['quotes_formatted'] += 1
                        seen.add(ident)
                        before = replace_links(prefix, depth, suppressed | ids,
                                               {(m.start(), m.end()) for m, _ in candidates})
                        quoted_text = text[marker.end():]
                        if not URL.sub('', quoted_text).strip() and ident in self.originals:
                            quoted_record = self.originals[ident]
                            add_media(quoted_record)
                            quoted_text = html.unescape(quoted_record['text'])
                            events['quote_text_recovered'] += 1
                        if re.search(r'\n[ \t\r]*\n', quoted_text):
                            events['quote_bodies_with_blank_lines'] += 1
                            events['rows_with_quote_blank_lines'] = 1
                        body = render_text(quoted_text, ident, depth, suppressed | ids, nesting + 1)
                        lines = [line.strip() for line in body.splitlines() if line.strip()]
                        quoted = [f'> Quoting @{marker[1]}:']
                        for line in lines:
                            if re.match(r'^>+ Quoting @\w+:', line):
                                quoted.append('>' + line)
                            else:
                                quoted[-1] += ' ' + line
                        prefix = before.rstrip()
                        return (prefix + '\n' if prefix else '') + '\n'.join(quoted)
            if QT.search(text):
                events['unconfirmed_qt_segments'] += 1
            return replace_links(text, depth, suppressed)

        def render_record(record, depth):
            text = html.unescape(record['text'])
            m = RT.match(text)
            target = status(record.get('archive_link', ''))
            confirmed = bool(m and target and target[1] != record['tweet_id'] and
                             target[0].lower() == m[1].lower())
            if confirmed:
                events['retweets_formatted'] += 1
                body = render_text(text[m.end():], target[1], depth, set())
                body = body.strip()
                separator = '\n' if body.startswith('>') else ' '
                return f'Retweet of @{m[1]}:' + separator + body
            if m:
                events['unconfirmed_rt'] += 1
            return render_text(text, record['tweet_id'], depth, set())

        add_media(row)
        result = render_record(row, 0).strip()
        attachments = []
        image_count = len(images) or sum(inferred_images.values())
        video_count = len(videos) or len(inferred_videos)
        if inferred_images and not images or inferred_videos and not videos:
            events['rows_with_inferred_media'] += 1
        if image_count:
            attachments.append('attached image' if image_count == 1 else f'{image_count} attached images')
        if video_count:
            attachments.append('attached video' if video_count == 1 else f'{video_count} attached videos')
        if attachments:
            separator = '\n' if result.split('\n')[-1].startswith('>') else ' '
            result += separator + '[' + ' and '.join(attachments) + ']'
            events['rows_with_media'] += 1
        result = re.sub(r'[ \t]+', ' ', result)
        result = re.sub(r' *\n *', '\n', result).strip()
        if re.search(r'https?://|(?<![\w@.-])www\.', result, re.I):
            raise ValueError(f"URL remains in tweet {row['tweet_id']}")
        return result, events


def load_metadata():
    redirects = {r['short_url']: r['resolved_url'] for r in
                 read_rows(DATA / 'resolved.csv', 'Load redirects') if r['resolved_url']}
    articles, aliases = {}, {}
    for r in read_rows(DATA / 'matches.csv', 'Load article metadata'):
        title = clean_metadata(r['title'])
        if not title or BAD_TITLE.match(title):
            continue
        author = clean_metadata(r['author'])
        if len(author) > 200:
            author = ''
        publisher = clean_metadata(r['publisher']) or host(r['matched_url'])
        if publisher.casefold() == title.casefold() or len(publisher) > 120:
            publisher = host(r['matched_url'])
        if publisher:
            author = re.sub(r'[,;|]\s*' + re.escape(publisher) + r'$', '', author, flags=re.I).strip()
        published = ''
        try:
            published = date.fromisoformat(r['date'][:10]).strftime('%d/%m/%Y')
        except ValueError:
            pass
        metadata = (title, author, publisher, published)
        articles[r['url']] = metadata
        key = canonical(r['url'])
        if key not in aliases:
            aliases[key] = metadata
        elif aliases[key] != metadata:
            aliases[key] = None
    articles.update({k: v for k, v in aliases.items() if v is not None})
    youtube = {}
    for r in read_rows(DATA / 'youtube.csv', 'Load YouTube metadata'):
        ident = youtube_id(r['url'])
        if ident and r['title']:
            youtube[ident] = (clean_metadata(r['title']), clean_metadata(r['channel']))
    return redirects, articles, youtube


def fingerprint(row):
    return json.dumps({k: v for k, v in row.items() if k != 'text'},
                      ensure_ascii=False, sort_keys=True).encode() + b'\n'


def write_report(path, samples, interesting, counts):
    def card(row, label):
        return (f'<article><h3>{html.escape(label)}</h3><a href="{html.escape(row["url"], quote=True)}" '
                f'target="_blank">Open original tweet</a><div class="pair"><section><h4>Original</h4>'
                f'<pre>{html.escape(row["original"])}</pre></section><section><h4>Formatted</h4>'
                f'<pre>{html.escape(row["formatted"])}</pre></section></div></article>')
    body = '<h1>Tweet formatting — plain retweets and quoted blocks</h1><p>Seeded random sample of 20 rows, followed by selected cases. '
    body += 'Only the text column changes. Unconfirmed RT/QT markers remain literal. '
    body += 'Retweets are inline; each quote is one paragraph introduced by &gt; Quoting (nested quotes use &gt;&gt; Quoting). '
    body += 'Attachment counts combine archived media, including quoted/linked content; ownership is not inferred.</p>'
    body += '<details><summary>Counts and validation</summary><pre>' + html.escape(json.dumps(counts, indent=2)) + '</pre></details>'
    body += '<h2>Random sample</h2>' + ''.join(card(r, f'Random {i:02}') for i, r in enumerate(samples, 1))
    body += '<h2>Interesting cases</h2>'
    for kind, rows in interesting.items():
        body += ''.join(card(r, kind) for r in rows)
    path.write_text('<!doctype html><meta charset="utf-8"><title>Tweet formatting</title><style>'
                    'body{font:16px system-ui;background:#f4f5f7;color:#192334;margin:32px auto;max-width:1400px;padding:0 24px}'
                    'article{background:white;border:1px solid #ddd;border-radius:12px;padding:20px;margin:20px 0}'
                    '.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:15px/1.55 system-ui}'
                    'h4{color:#637083}a{color:#1462a6}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style>' + body,
                    encoding='utf-8')


def main():
    csv.field_size_limit(16 * 1024 * 1024)
    if OUTPUT.exists():
        raise SystemExit(f'{OUTPUT} exists; remove it explicitly before rerunning.')
    sources = [INPUT, DATA / 'resolved.csv', DATA / 'youtube.csv', DATA / 'matches.csv']
    signatures = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in sources}
    work = Path(tempfile.mkdtemp(prefix='congress-formatted-'))
    redirects, articles, youtube = load_metadata()
    formatter = Formatter(redirects, articles, youtube, {})
    targets = set()
    for row in read_rows(INPUT, 'Find linked tweets'):
        for m in URL.finditer(html.unescape(row['text'])):
            target = status(formatter.resolve(split_url(m[0])[0]))
            if target:
                targets.add(target[1])
    for row in read_rows(INPUT, 'Index local originals'):
        if row['tweet_id'] in targets:
            formatter.originals[row['tweet_id']] = {k: row[k] for k in
                ('tweet_id', 'twitter', 'text', 'archive_link', 'retweet_id', 'images', 'videos')}
    tqdm.write(f'{len(targets):,} linked IDs; {len(formatter.originals):,} found locally.')
    counts = Counter()
    samples, interesting = [], defaultdict(list)
    rng = random.Random(20261010)
    original_digest = hashlib.sha256()
    partial = work / 'formated.csv'
    with partial.open('w', newline='', encoding='utf-8') as stream:
        writer = None
        for n, row in enumerate(read_rows(INPUT, 'Format tweets'), 1):
            if writer is None:
                fields = list(row)
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
            original_digest.update(fingerprint(row))
            original = row['text']
            row['text'], events = formatter.format(row)
            writer.writerow(row)
            counts.update(events)
            counts['rows'] += 1
            counts['text_changed'] += row['text'] != original
            sample = {'url': row['tweet_url'], 'original': original, 'formatted': row['text']}
            if len(samples) < 20:
                samples.append(sample)
            else:
                index = rng.randrange(n)
                if index < 20:
                    samples[index] = sample
            categories = [k for k in ('youtube_titles', 'article_titles', 'url_labels', 'linked_tweets_expanded',
                          'twitter_fallbacks', 'unconfirmed_qt_segments', 'unconfirmed_rt', 'rows_with_media') if events[k]]
            if events['quotes_formatted']:
                categories.append('retweet containing quote' if events['retweets_formatted'] else 'quote')
            if events['retweets_formatted'] and not events['quotes_formatted']:
                categories.append('retweet')
            for category in categories:
                if len(interesting[category]) < 2:
                    interesting[category].append(sample)
    output_digest = hashlib.sha256()
    verified = 0
    for row in read_rows(partial, 'Validate output'):
        if list(row) != fields or re.search(r'https?://|(?<![\w@.-])www\.', row['text'], re.I):
            raise ValueError('Output schema or URL validation failed')
        output_digest.update(fingerprint(row))
        verified += 1
    if verified != counts['rows'] or output_digest.digest() != original_digest.digest():
        raise ValueError('Row count or non-text columns changed')
    if any((p.stat().st_size, p.stat().st_mtime_ns) != sig for p, sig in signatures.items()):
        raise ValueError('An input changed during formatting')
    counts['verified_rows'] = verified
    counts['non_text_columns_unchanged'] = True
    counts['remaining_http_urls'] = 0
    write_report(work / 'report.html', samples, interesting, dict(counts))
    (work / 'counts.json').write_text(json.dumps(counts, indent=2))
    (work / 'samples.json').write_text(json.dumps({'random': samples, 'interesting': interesting}, indent=2))
    shutil.move(str(partial), OUTPUT)
    tqdm.write(json.dumps(counts, indent=2))
    tqdm.write(f'Saved {OUTPUT}\nReport: {work / "report.html"}')


if __name__ == '__main__':
    main()
