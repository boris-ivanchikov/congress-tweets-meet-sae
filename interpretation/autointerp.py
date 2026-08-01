"""Auto-interpret SAE features through the DeepSeek API and write interpretations.csv.

Reads a trained SAE run directory (config, weights, activations) and writes all
artifacts to the mirrored interpretation/runs/<run-name>/ directory; the input
directory is never written to. Features are selected by activation-rate band
(--min-pct/--max-pct), explicitly (--features), or by random sample (--random).

Examples per feature are deduplicated, capped per author, and stratified across
the activation range (strongest first, activation values shown to the model).
Each explanation is optionally scored by held-out detection: the model classifies
unseen positives (spread across the activation range) against random
non-activating negatives.
"""

import os
import re
import json
import time
import random
import argparse
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import Activations, artifacts_dir, normalize_for_dedup

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_KEY_PATH = "~/.deepseek/key"
MODEL = "deepseek-v4-flash"
# $ per 1M tokens: (input cache miss, input cache hit, output)
PRICES = {
    "deepseek-v4-flash": (0.14, 0.0028, 0.28),
    "deepseek-v4-pro": (0.435, 0.003625, 0.87),
}

N_EXAMPLES = 15
TOP_FRACTION = 0.4          # share of examples taken from the very top of the range
AUTHOR_CAP = 2              # max examples per twitter handle
MAX_TWEET_CHARS = 280
SCORE_PER_CLASS = 10
INTERP_MAX_TOKENS = 512
INTERP_THINKING_MAX_TOKENS = 4096

INTERP_SYSTEM = (
    "Task\n"
    "Interpret one feature of a sparse autoencoder trained on embeddings of congressional tweets. "
    "You will receive activating tweets sorted from strongest to weakest, with activation values. "
    "Infer the single property best represented by the feature.\n\n"
    "Activation types\n"
    "Choose exactly one type. The examples below are fabricated and are not evaluation data.\n"
    "- topic: a recurring subject, activity, or policy domain. Example: tweets about coral bleaching, "
    "reef restoration, and ocean warming represent the topic 'coral reef conservation'.\n"
    "- entity: one specific person, organization, place, event, or named program. Example: tweets about "
    "performances at, renovations to, and directions to Maple Hall represent the entity 'Maple Hall'.\n"
    "- lexical: the same word, phrase, hashtag, handle, or character sequence used across otherwise "
    "unrelated meanings or subjects. Example: 'seal the envelope', 'a harbor seal', and 'the official "
    "seal' represent the lexical item 'seal'.\n"
    "- style: a recurring tone, communicative purpose, rhetorical form, or layout. Example: several "
    "questions challenging a decision represent the style 'rhetorical questions'.\n"
    "- other: a stable property that does not fit the four types above. Example: tweets written "
    "primarily in Spanish represent the other property 'Spanish-language text'.\n\n"
    "Special instructions\n"
    "- Treat the strongest activators as the primary evidence. Use the weaker activation tail to refine "
    "the boundary, but allow very weak activations to be noise; do not let them outvote a clear strong "
    "pattern.\n"
    "- Choose the narrowest interpretation supported by several strong examples. Do not anchor on a "
    "detail found in only one or two examples, and do not broaden a coherent pattern unnecessarily.\n"
    "- A repeated word is not automatically lexical. Choose lexical only when the shared surface form "
    "persists across semantically unrelated contexts. If the surrounding tweets concern the same "
    "real-world subject or referent, choose topic or entity even when they repeat the same terminology.\n"
    "- Base the interpretation only on the tweet text. Be concise and politically neutral. Give one "
    "interpretation, not a list of alternatives.\n\n"
    "Output format\n"
    "Return one JSON object and nothing else:\n"
    '{"analysis": "<reasoning>", "label": "<3-6 word noun phrase>", '
    '"type": "<topic|entity|lexical|style|other>", "explanation": "<1-2 sentences>"}\n'
    "- analysis: at most 100 words. Identify the strongest coherent signal, use the weaker examples to "
    "check its scope, and justify the selected type. Do not quote tweets.\n"
    "- label: lowercase except proper nouns and acronyms; no trailing period.\n"
    "- explanation: state the general property directly. Do not mention features, examples, evidence, "
    "or activation strengths, and do not quote the supplied tweets."
)

SCORE_SYSTEM = (
    "You are a meticulous AI interpretability researcher. You are given a description of a feature "
    "(for example 'male pronouns' or 'support for military aid') and a numbered list of "
    "tweets. For each tweet, decide whether it genuinely exhibits the described property.\n\n"
    "Judge STRICTLY: mark a tweet 1 only if it actually matches the description, not merely because "
    "it shares a word or broad topic with it; otherwise mark 0.\n\n"
    "Return only a JSON array of integers (1 for match, 0 for no match), one per tweet, in order, "
    "and nothing else. Example: [1,0,0,1,0]"
)


# ---------------------------------------------------------------- example selection

def select_interp_examples(texts, handles, active_rows, active_values, k,
                           max_chars=MAX_TWEET_CHARS,
                           author_cap=AUTHOR_CAP, top_frac=TOP_FRACTION):
    """Pick k distinct, author-capped (text, activation) examples: the strongest
    activators first, then a sample spread evenly across the rest of the range.
    active_rows and active_values must be sorted by descending activation."""

    seen, per_author, chosen = set(), {}, []

    def take(i):
        j = int(active_rows[i])
        t = str(texts[j]).strip()
        norm = normalize_for_dedup(t)
        h = str(handles[j])
        if not norm or norm in seen or per_author.get(h, 0) >= author_cap:
            return False
        seen.add(norm)
        per_author[h] = per_author.get(h, 0) + 1
        chosen.append((t[:max_chars], float(active_values[i])))
        return True

    k = min(k, len(active_rows))
    k_top = max(1, round(k * top_frac)) if k else 0
    i = 0
    while i < len(active_rows) and len(chosen) < k_top:
        take(i)
        i += 1
    n_top = len(chosen)

    k_rest = k - n_top
    if k_rest > 0 and i < len(active_rows):
        taken = set()
        for target in np.linspace(i, len(active_rows) - 1, k_rest).round().astype(int):
            j = int(target)
            while j < len(active_rows):
                if j not in taken and take(j):
                    taken.add(j)
                    break
                j += 1

    chosen.sort(key=lambda p: -p[1])
    return chosen, n_top, seen


def _take_distinct(indices, texts, seen, max_chars, k):
    """Distinct (truncated text, source index) pairs, in the given order."""
    out = []
    for j in indices:
        t = str(texts[j]).strip()
        norm = normalize_for_dedup(t)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append((t[:max_chars], int(j)))
        if len(out) >= k:
            break
    return out


def select_score_pool(active_rows, active_values, n_tweets, texts, exclude_norms,
                      n, seed, max_chars=MAX_TWEET_CHARS):
    """Held-out positives spread across the activation range (as (text, activation)
    pairs); random non-activating negatives, balanced to the number of positives."""
    rng = random.Random(seed)
    seen = set(exclude_norms)

    positives = []
    n_act = len(active_rows)
    if n_act:
        ranks = np.linspace(0, n_act - 1, min(n_act, n * 8)).round().astype(int)
        held = []
        for rank in ranks:
            j = int(active_rows[rank])
            t = str(texts[j]).strip()
            norm = normalize_for_dedup(t)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            held.append((t[:max_chars], float(active_values[rank])))
        if len(held) > n:
            held = [held[i] for i in np.linspace(0, len(held) - 1, n).round().astype(int)]
        positives = held

    target = len(positives)
    negatives = []
    n_nonact = n_tweets - n_act
    if n_nonact and target:
        active_sorted = np.sort(active_rows)
        candidates, sampled = [], set()
        while len(candidates) < min(n_nonact, target * 4):
            j = rng.randrange(n_tweets)
            if j in sampled:
                continue
            sampled.add(j)
            at = np.searchsorted(active_sorted, j)
            if at == len(active_sorted) or active_sorted[at] != j:
                candidates.append(j)
        negatives = [t for t, _ in _take_distinct(
            np.asarray(candidates), texts, seen, max_chars, target)]
    return positives, negatives


# ---------------------------------------------------------------- prompts & parsing

def build_interp_messages(examples, n_top):
    numbered = "\n".join(f"{i + 1}. [activation={act:.2f}] {t}"
                         for i, (t, act) in enumerate(examples))
    user = (f"Here are {len(examples)} tweets that activate this SAE feature, sorted by activation "
            f"strength (strongest first). The first {n_top} are the strongest activators overall; "
            f"the remaining {len(examples) - n_top} are sampled evenly across the rest of the "
            f"activation range.\n\n{numbered}\n\n"
            "What property does this feature represent? Reply with only the JSON object.")
    return [{"role": "system", "content": INTERP_SYSTEM}, {"role": "user", "content": user}]


def build_score_messages(explanation, texts):
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    user = (f"Concept: {explanation}\n\nTweets:\n{numbered}\n\n"
            f"Reply with a JSON array of {len(texts)} integers (0 or 1), one per tweet in order.")
    return [{"role": "system", "content": SCORE_SYSTEM}, {"role": "user", "content": user}]


def _clean_label(label):
    label = label.strip().rstrip(".").strip()
    if len(label) > 1 and label[0] == label[-1] and label[0] in "\"'":
        label = label[1:-1].strip()
    return label


FEATURE_TYPES = {"topic", "entity", "lexical", "style", "other"}


def parse_interp_output(raw):
    """Returns (label, type, explanation), or None if no valid JSON was produced."""
    text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL).strip()
    # last JSON object wins: reasoning transcripts may contain earlier drafts
    candidates = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)[::-1]
    if not candidates:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        candidates = [m.group(0)] if m else []
    for cand in candidates:
        try:
            obj = json.loads(cand)
            label = _clean_label(str(obj.get("label", "")))
            ftype = str(obj.get("type", "")).strip().lower()
            if ftype not in FEATURE_TYPES:
                ftype = "other"
            explanation = str(obj.get("explanation", "")).strip()
            if label:
                return label[:80], ftype, explanation
        except (json.JSONDecodeError, AttributeError, TypeError):
            continue
    return None


def parse_score_output(raw, n):
    text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL)
    m = re.search(r"\[[\s\S]*?\]", text)
    if m:
        try:
            nums = [int(x) for x in json.loads(m.group(0))]
        except (json.JSONDecodeError, ValueError, TypeError):
            nums = [int(x) for x in re.findall(r"[01]", m.group(0))]
    else:
        nums = [int(x) for x in re.findall(r"[01]", text)]
    nums = [1 if x else 0 for x in nums][:n]
    return nums if len(nums) == n else None


# ---------------------------------------------------------------- API client

class DeepSeekClient:
    def __init__(self, model, max_cost):
        from openai import OpenAI

        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            with open(os.path.expanduser(DEEPSEEK_KEY_PATH)) as f:
                key = f.read().strip()
        self.client = OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL)
        self.model = model
        self.max_cost = max_cost
        self.lock = threading.Lock()
        self.usage = {"miss": 0, "hit": 0, "out": 0}

    def cost(self):
        miss_p, hit_p, out_p = PRICES[self.model]
        u = self.usage
        return (u["miss"] * miss_p + u["hit"] * hit_p + u["out"] * out_p) / 1e6

    def _record(self, u):
        extra = getattr(u, "model_extra", None) or {}
        hit = extra.get("prompt_cache_hit_tokens", 0) or 0
        miss = u.prompt_tokens - hit
        with self.lock:
            self.usage["hit"] += hit
            self.usage["miss"] += miss
            self.usage["out"] += u.completion_tokens
        miss_p, hit_p, out_p = PRICES[self.model]
        return (miss * miss_p + hit * hit_p + u.completion_tokens * out_p) / 1e6

    def chat(self, messages, max_tokens, thinking, json_object=False):
        """Returns (text, $ spent on this call including retries)."""
        if self.cost() >= self.max_cost:
            raise RuntimeError(f"cost cap ${self.max_cost} reached for {self.model}")
        last_exc, spent = None, 0.0
        for attempt in range(5):
            try:
                kwargs = {"response_format": {"type": "json_object"}} if json_object else {}
                r = self.client.chat.completions.create(
                    model=self.model, messages=messages, stream=False, max_tokens=max_tokens,
                    extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
                    **kwargs,
                )
                spent += self._record(r.usage)
                choice = r.choices[0]
                text = (choice.message.content or "").strip()
                if not text and choice.finish_reason == "length" and max_tokens < 4 * INTERP_MAX_TOKENS:
                    # reasoning ate the whole budget before any answer was emitted
                    max_tokens *= 2
                    continue
                if not text and not json_object:
                    # the answer sometimes lands in reasoning_content with empty content
                    text = (getattr(choice.message, "reasoning_content", None) or "").strip()
                return text, spent
            except Exception as exc:
                last_exc = exc
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"API call failed after retries: {last_exc}")


# ---------------------------------------------------------------- per-feature job

def interpret_feature(client, feature, examples, n_top, positives, negatives,
                      thinking_interp=False):
    if not examples:
        return {"label": "(no activating tweets)", "type": "other", "explanation": "",
                "score": None, "score_weighted": None,
                "n_pos": 0, "n_neg": 0, "cost_usd": 0.0}

    cost, parsed = 0.0, None
    max_tokens = INTERP_THINKING_MAX_TOKENS if thinking_interp else INTERP_MAX_TOKENS
    for _ in range(3):
        raw, spent = client.chat(build_interp_messages(examples, n_top), max_tokens,
                                 thinking=thinking_interp, json_object=True)
        cost += spent
        parsed = parse_interp_output(raw)
        if parsed:
            break
    if parsed is None:
        return {"label": "(parse failed)", "type": "other", "explanation": "",
                "score": None, "score_weighted": None,
                "n_pos": 0, "n_neg": 0, "cost_usd": round(cost, 6)}
    label, ftype, explanation = parsed

    score = score_weighted = None
    if explanation and positives and negatives:
        items = [(t, 1, a) for t, a in positives] + [(t, 0, 0.0) for t in negatives]
        random.Random(feature).shuffle(items)
        texts = [t for t, _, _ in items]
        raw, spent = client.chat(build_score_messages(explanation, texts),
                                 len(texts) * 3 + 64, thinking=False)
        cost += spent
        preds = parse_score_output(raw, len(texts))
        if preds is not None:
            records = [(l, a, int(p == l)) for p, (_, l, a) in zip(preds, items)]
            score = round(sum(c for _, _, c in records) / len(records), 3)
            # balanced accuracy with positives weighted by activation strength:
            # high when the label fits the strong activators even if the weak tail drifts
            pos = [(a, c) for l, a, c in records if l == 1]
            neg = [c for l, _, c in records if l == 0]
            wsum = sum(a for a, _ in pos)
            if wsum > 0 and neg:
                pos_acc = sum(a * c for a, c in pos) / wsum
                neg_acc = sum(neg) / len(neg)
                score_weighted = round((pos_acc + neg_acc) / 2, 3)

    return {"label": label, "type": ftype, "explanation": explanation,
            "score": score, "score_weighted": score_weighted,
            "n_pos": len(positives), "n_neg": len(negatives), "cost_usd": round(cost, 6)}


# ---------------------------------------------------------------- main

def resolve_features(args, acts):
    """1-indexed features to interpret, chosen without any dependency on
    downstream analyses: explicit list, activation-rate band, or random sample."""
    if args.features:
        return [int(f) for f in args.features]
    in_band = np.where((acts.pct_active >= args.min_pct)
                       & (acts.pct_active <= args.max_pct))[0] + 1
    features = in_band.tolist()
    if args.random:
        rng = random.Random(args.seed)
        features = sorted(rng.sample(features, min(args.random, len(features))))
    if args.limit:
        features = features[:args.limit]
    print(f"Interpreting {len(features)} of {len(in_band)} features in "
          f"[{args.min_pct}%, {args.max_pct}%] activation band")
    return features


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=str, required=True,
                        help="Path to trained SAE run directory (read-only input)")
    parser.add_argument("--features", nargs="+", type=int, default=None,
                        help="Explicit 1-indexed feature ids (overrides band selection)")
    parser.add_argument("--min-pct", type=float, default=0.05,
                        help="Min %% of tweets a feature must activate on")
    parser.add_argument("--max-pct", type=float, default=100.0,
                        help="Max %% of tweets a feature may activate on")
    parser.add_argument("--random", type=int, default=None,
                        help="Random sample of X features from the band")
    parser.add_argument("--seed", type=int, default=0, help="Seed for --random")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only interpret the first X selected features")
    parser.add_argument("--n-examples", type=int, default=N_EXAMPLES)
    parser.add_argument("--interp-thinking", action="store_true",
                        help="Enable thinking for interpretation (disabled by default)")
    parser.add_argument("--no-score", action="store_true", help="Skip detection scoring")
    parser.add_argument("--score-per-class", type=int, default=SCORE_PER_CLASS)
    parser.add_argument("--out", type=str, default=None,
                        help="Output CSV (default: interpretation/runs/<run>/interpretations.csv)")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-cost", type=float, default=2.0,
                        help="Abort new requests once estimated $ per model exceeds this")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-interpret (feature, model) pairs already in the output CSV")
    return parser.parse_args()


def main(args):
    out_path = args.out or os.path.join(artifacts_dir(args.path), "interpretations.csv")
    acts = Activations(args.path)
    features = resolve_features(args, acts)

    done = set()
    if os.path.exists(out_path) and not args.overwrite:
        prev = pd.read_csv(out_path)
        done = set(zip(prev["feature"].astype(int), prev["model"]))
        print(f"Resuming: {len(done)} (feature, model) rows already in {out_path}")

    print("Loading tweets...")
    tweets_df = pd.read_csv("data/tweets.csv")
    tweet_ids = tweets_df["tweet_id"].to_numpy()
    texts = tweets_df["text"].to_numpy()
    handles = tweets_df["twitter"].astype(str).to_numpy()

    csv_lock = threading.Lock()

    def append_row(row):
        with csv_lock:
            pd.DataFrame([row]).to_csv(out_path, mode="a", index=False,
                                       header=not os.path.exists(out_path))

    todo = [f for f in features if (f, MODEL) not in done]
    if not todo:
        print("nothing to do")
        return

    print("Aligning activation rows to tweets once...")
    tweet_row_of_id = pd.Series(np.arange(len(tweet_ids)), index=tweet_ids)
    act_to_tweet = tweet_row_of_id.reindex(acts.ids).to_numpy()
    if pd.isna(act_to_tweet).any():
        raise ValueError("activation ids and tweet ids do not match")
    act_to_tweet = act_to_tweet.astype(np.int64, copy=False)

    def prepare(feature):
        act_rows, act_values = acts.sparse_col(feature)
        tweet_rows = act_to_tweet[act_rows]
        positive = act_values > 0
        tweet_rows = tweet_rows[positive]
        act_values = act_values[positive]
        order = np.lexsort((tweet_rows, -act_values))
        tweet_rows = tweet_rows[order]
        act_values = act_values[order]

        examples, n_top, ex_norms = select_interp_examples(
            texts, handles, tweet_rows, act_values, args.n_examples)
        positives, negatives = [], []
        if not args.no_score:
            positives, negatives = select_score_pool(
                tweet_rows, act_values, len(texts), texts, ex_norms,
                args.score_per_class, seed=feature)
        n_act = len(tweet_rows)
        return {"feature": feature, "examples": examples, "n_top": n_top,
                "positives": positives, "negatives": negatives,
                "n_act": n_act, "pct_act": 100 * n_act / len(texts)}

    client = DeepSeekClient(MODEL, args.max_cost)
    api_bar = tqdm(total=len(todo), desc=MODEL, position=1)

    def work(j):
        try:
            res = interpret_feature(client, j["feature"], j["examples"], j["n_top"],
                                    j["positives"], j["negatives"],
                                    thinking_interp=args.interp_thinking)
            append_row({"feature": j["feature"], "model": MODEL, **res,
                        "n_examples": len(j["examples"]), "n_activating": j["n_act"],
                        "pct_activating": round(j["pct_act"], 4)})
        except Exception as exc:
            print(f"\nfeature {j['feature']} failed: {exc}")
        api_bar.update(1)

    max_pending = max(args.concurrency * 2, 1)
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        pending = set()
        for feature in tqdm(todo, desc="Pipeline", position=0):
            pending.add(ex.submit(work, prepare(feature)))
            if len(pending) >= max_pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    future.result()
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                future.result()
    api_bar.close()
    u = client.usage
    print(f"{MODEL}: {u['miss']:,} input (miss) + {u['hit']:,} input (hit) + "
          f"{u['out']:,} output tokens -> ${client.cost():.4f}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main(parse_args())
