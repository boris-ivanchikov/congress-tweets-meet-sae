"""Auto-interpret SAE features through the DeepSeek API and write interpretations.csv.

Reads a trained SAE run directory (config, weights, activations) and writes all
artifacts to the mirrored interpretation/runs/<run-name>/ directory; the input
directory is never written to. Features are selected by activation-rate band
(--min-pct/--max-pct), explicitly (--features), or by random sample (--random).

Examples per feature are deduplicated, capped per author, and stratified across
the activation range (strongest first, activation values shown to the model).
Each feature receives a narrow core interpretation for its strongest activations
and a broader activation-envelope interpretation for its weaker semantic
neighborhood. Held-out scoring reports both coverages by activation-rank quartile
plus specificity on random non-activating tweets.
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

N_EXAMPLES = 25
TOP_FRACTION = 0.4          # share of examples taken from the very top of the range
AUTHOR_CAP = 2              # max examples per twitter handle
MAX_TWEET_CHARS = 280
SCORE_PER_QUARTILE = 10
SCORE_NEGATIVES = 40
SCORE_CHUNK_SIZE = 20
INTERP_MAX_TOKENS = 512
INTERP_THINKING_MAX_TOKENS = 4096
INTERP_RETRY_MAX_TOKENS = 800

INTERP_SYSTEM = (
    "Task\n"
    "Interpret one feature of a sparse autoencoder trained on embeddings of congressional tweets. "
    "You will receive activating tweets sorted from strongest to weakest, with activation values. "
    "Describe its meaning at two nested levels.\n\n"
    "Interpretation levels\n"
    "- Core: the narrow, specific property that best explains the strongest activations. It may be a "
    "topic, entity, lexical pattern, communicative function, tone, or other stable property. Do not "
    "weaken this interpretation merely to cover the tail.\n"
    "- Broad envelope: the smallest meaningful generalization of the core that explains as many weaker "
    "activations as possible. It may generalize subject matter, rhetorical function, morphology, or "
    "another dimension; it need not be a topical category. The core must be a genuine instance of it.\n\n"
    "Reasoning rules\n"
    "1. Infer the core from the strongest group first and hold it fixed. Choose the most specific pattern "
    "supported by several strong, author-diverse tweets rather than one vivid example.\n"
    "2. Then inspect the activation-range sample and generalize only as far as needed. Look for a clear "
    "semantic, functional, stylistic, lexical, or morphological relationship to the core.\n"
    "3. If the weaker tweets have several unrelated patterns and no useful common generalization, set "
    "the broad label and explanation to null and mark the tail as noise. Do not invent an envelope such "
    "as 'politics' that would apply to most congressional tweets.\n"
    "4. Do not add a stance, causal claim, audience, or political purpose unless it recurs. Base both "
    "interpretations only on the tweet text; be concise and politically neutral.\n\n"
    "Output format\n"
    "Return one JSON object and nothing else:\n"
    '{"analysis": "<reasoning>", "core_label": "<3-8 word noun phrase>", '
    '"core_explanation": "<1-2 sentences>", "broad_label": "<3-8 word noun phrase or null>", '
    '"broad_explanation": "<1-2 sentences or null>", '
    '"tail_coherence": "<coherent|partial|noise>", "relation": "<one sentence>"}\n'
    "- analysis: at most 100 words. Establish the core, then test the smallest broader envelope. Do not "
    "quote tweets.\n"
    "- labels: lowercase except proper nouns and acronyms; no trailing period.\n"
    "- explanations: state each property directly. Never write 'This feature represents', "
    "'The feature represents', 'This feature captures', 'The feature captures', or similar framing. "
    "Do not mention features, examples, evidence, or activation strengths, and do not quote the "
    "supplied tweets.\n"
    "- relation: explain how the core is nested inside the broad envelope; if the broad interpretation "
    "is null, briefly state that the weaker tail is not coherent."
)

SCORE_SYSTEM = (
    "Task\n"
    "Evaluate how well each tweet matches two nested descriptions of an SAE feature. Judge every "
    "tweet independently; other tweets in the batch must not change the standard.\n\n"
    "Decisions\n"
    "Answer these two yes/no questions for each tweet, in this order:\n"
    "1. Broad: Does the tweet genuinely fall anywhere within the broad label and description? The "
    "broad label defines the category; details in its explanation are illustrative rather than an "
    "exhaustive list. A real subtopic or instance counts even if it uses different words or differs "
    "from the narrow core.\n"
    "2. Core: Does the tweet satisfy the narrower core label and description?\n\n"
    "Rules\n"
    "- Judge semantic properties by meaning, not word overlap. Ordinary implications stated or clearly "
    "identified in the text count, but do not invent absent context.\n"
    "- Judge lexical, orthographic, numeric, structural, and other formal properties directly from the "
    "text, regardless of its topic.\n"
    "- The core is nested inside the broad category: core=yes requires broad=yes.\n"
    "- Be precise, but do not reject a genuine category member merely because it is not one of the "
    "specific cases named in the descriptions.\n\n"
    "Output\n"
    "Return only one JSON array containing one [broad, core] pair per tweet, in order. Use 1 for yes "
    "and 0 for no. Valid pairs are [1,1], [1,0], and [0,0]. "
    "Example: [[1,1],[1,0],[0,0]]"
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
                      per_quartile, n_negatives, seed, max_chars=MAX_TWEET_CHARS):
    """Select held-out positives from each activation-rank quartile plus random
    exact-zero negatives.

    Q1 uses the strongest unseen tweets so it is a stable test of whether the
    interpretation describes the feature's head. Q2--Q4 are sampled uniformly
    within their quartiles to diagnose semantic decay through the weaker tail.
    """
    rng = random.Random(seed)
    seen = set(exclude_norms)

    positives = []
    n_act = len(active_rows)
    for quartile in range(4):
        start = quartile * n_act // 4
        stop = (quartile + 1) * n_act // 4
        width = stop - start
        if not width:
            continue
        # The head score must actually test the held-out head. Uniform sampling
        # from a very dense top quartile can land far below the coherent region.
        # Lower quartiles remain uniform diagnostic samples.
        ranks = (range(start, stop) if quartile == 0 else
                 rng.sample(range(start, stop), min(width, per_quartile * 50)))
        held = 0
        for rank in ranks:
            j = int(active_rows[rank])
            t = str(texts[j]).strip()
            norm = normalize_for_dedup(t)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            positives.append((t[:max_chars], float(active_values[rank]), quartile + 1))
            held += 1
            if held >= per_quartile:
                break

    negatives = []
    n_nonact = n_tweets - n_act
    if n_nonact and n_negatives:
        active_sorted = np.sort(active_rows)
        candidates, sampled = [], set()
        while len(candidates) < min(n_nonact, n_negatives * 4):
            j = rng.randrange(n_tweets)
            if j in sampled:
                continue
            sampled.add(j)
            at = np.searchsorted(active_sorted, j)
            if at == len(active_sorted) or active_sorted[at] != j:
                candidates.append(j)
        negatives = [t for t, _ in _take_distinct(
            np.asarray(candidates), texts, seen, max_chars, n_negatives)]
    return positives, negatives


# ---------------------------------------------------------------- prompts & parsing

def build_interp_messages(examples, n_top):
    numbered = "\n".join(f"{i + 1}. [activation={act:.4g}] {t}"
                         for i, (t, act) in enumerate(examples))
    user = (f"Here are {len(examples)} tweets that activate this SAE feature, sorted by activation "
            f"strength (strongest first). The first {n_top} are the strongest activators overall; "
            f"the remaining {len(examples) - n_top} are sampled evenly across the rest of the "
            f"activation range.\n\n{numbered}\n\n"
            "What latent property best explains these activations? Reply with only the JSON object.")
    return [{"role": "system", "content": INTERP_SYSTEM}, {"role": "user", "content": user}]


def build_score_messages(core_label, core_explanation, broad_label, broad_explanation, texts):
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    broad_label = broad_label or core_label
    broad_explanation = broad_explanation or core_explanation
    user = (f"Broad label: {broad_label}\nBroad description: {broad_explanation}\n\n"
            f"Core label: {core_label}\nCore description: {core_explanation}\n\n"
            f"Tweets:\n{numbered}\n\nReturn exactly {len(texts)} [broad, core] pairs.")
    return [{"role": "system", "content": SCORE_SYSTEM}, {"role": "user", "content": user}]


def _clean_label(label):
    label = label.strip().rstrip(".").strip()
    if len(label) > 1 and label[0] == label[-1] and label[0] in "\"'":
        label = label[1:-1].strip()
    return label


def parse_interp_output(raw):
    """Return the flat core/envelope interpretation, or None for invalid JSON."""
    text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL).strip()
    # last JSON object wins: reasoning transcripts may contain earlier drafts
    candidates = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)[::-1]
    if not candidates:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        candidates = [m.group(0)] if m else []
    for cand in candidates:
        try:
            obj = json.loads(cand)
            core_label = _clean_label(str(obj.get("core_label", "")))
            core_explanation = str(obj.get("core_explanation", "")).strip()
            raw_broad_label = obj.get("broad_label")
            raw_broad_explanation = obj.get("broad_explanation")
            broad_label = (_clean_label(str(raw_broad_label))
                           if raw_broad_label is not None else "")
            broad_explanation = (str(raw_broad_explanation).strip()
                                 if raw_broad_explanation is not None else "")
            tail_coherence = str(obj.get("tail_coherence", "partial")).strip().lower()
            if tail_coherence not in {"coherent", "partial", "noise"}:
                tail_coherence = "partial"
            relation = str(obj.get("relation", "")).strip()
            if core_label and core_explanation:
                return {
                    "core_label": core_label[:100],
                    "core_explanation": core_explanation,
                    "broad_label": broad_label[:100],
                    "broad_explanation": broad_explanation,
                    "tail_coherence": tail_coherence,
                    "relation": relation,
                }
        except (json.JSONDecodeError, AttributeError, TypeError):
            continue
    return None


def parse_score_output(raw, n):
    """Parse broad-first [broad, core] pairs into 0/1/2 decisions."""
    text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL).strip()
    start, stop = text.find("["), text.rfind("]")
    if start < 0 or stop < start:
        return None
    try:
        pairs = json.loads(text[start:stop + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(pairs, list) or len(pairs) != n:
        return None
    predictions = []
    for pair in pairs:
        if not isinstance(pair, list) or len(pair) != 2:
            return None
        try:
            broad, core = int(pair[0]), int(pair[1])
        except (TypeError, ValueError):
            return None
        if broad not in {0, 1} or core not in {0, 1} or (core and not broad):
            return None
        predictions.append(2 if core else 1 if broad else 0)
    return predictions


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
        """Return text, reasoning trace, finish reason, and dollars spent.

        The loop retries transport/API errors only. Invalid model output is
        handled once by the caller so it cannot trigger repeated thinking runs.
        """
        if self.cost() >= self.max_cost:
            raise RuntimeError(f"cost cap ${self.max_cost} reached for {self.model}")
        last_exc, spent = None, 0.0
        for attempt in range(5):
            try:
                kwargs = {"response_format": {"type": "json_object"}} if json_object else {}
                if not thinking:
                    kwargs["temperature"] = 0
                r = self.client.chat.completions.create(
                    model=self.model, messages=messages, stream=False, max_tokens=max_tokens,
                    extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
                    **kwargs,
                )
                spent += self._record(r.usage)
                choice = r.choices[0]
                text = (choice.message.content or "").strip()
                reasoning = (getattr(choice.message, "reasoning_content", None) or "").strip()
                if not text and not json_object:
                    # the answer sometimes lands in reasoning_content with empty content
                    text = reasoning
                return text, reasoning, choice.finish_reason, spent
            except Exception as exc:
                last_exc = exc
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"API call failed after retries: {last_exc}")


# ---------------------------------------------------------------- per-feature job

def interpret_feature(client, feature, examples, n_top, positives, negatives,
                      thinking_interp=False,
                      thinking_max_tokens=INTERP_THINKING_MAX_TOKENS):
    empty_scores = {"score_core": None, "score_broad": None,
                    **{f"score_core_q{q}": None for q in range(1, 5)},
                    **{f"score_broad_q{q}": None for q in range(1, 5)},
                    "score_neg_core": None, "score_neg_broad": None}
    if not examples:
        return {"core_label": "(no activating tweets)", "core_explanation": "",
                "broad_label": "", "broad_explanation": "",
                "tail_coherence": "noise", "relation": "",
                **empty_scores, "cost_usd": 0.0, "score_details": [],
                "reasoning_debug": None}

    cost, parsed = 0.0, None
    messages = build_interp_messages(examples, n_top)
    max_tokens = thinking_max_tokens if thinking_interp else INTERP_MAX_TOKENS
    raw, reasoning, finish_reason, spent = client.chat(
        messages, max_tokens, thinking=thinking_interp, json_object=True)
    cost += spent
    initial_raw = raw
    initial_finish_reason = finish_reason
    recovery_used = False
    parsed = parse_interp_output(raw)

    if parsed is None:
        # Preserve the useful work from a truncated or malformed attempt, but
        # make the recovery itself deterministic and cheap. DeepSeek ignores
        # reasoning_content in an ordinary next turn, so expose it as text.
        notes = reasoning
        if raw:
            notes += ("\n\nAttempted answer:\n" if notes else "") + raw
        if not notes:
            notes = "The previous attempt did not produce usable working notes."
        retry_messages = messages + [
            {"role": "assistant",
             "content": "Incomplete working notes from the previous attempt:\n" + notes},
            {"role": "user",
             "content": ("Use those notes only as a draft. Without further deliberation, "
                         "return the requested final JSON object now.")},
        ]
        raw, _, finish_reason, spent = client.chat(
            retry_messages, INTERP_RETRY_MAX_TOKENS,
            thinking=False, json_object=True)
        cost += spent
        recovery_used = True
        parsed = parse_interp_output(raw)
    reasoning_debug = {
        "reasoning": reasoning,
        "initial_answer": initial_raw,
        "initial_finish_reason": initial_finish_reason,
        "recovery_used": int(recovery_used),
        "final_answer": raw,
        "final_finish_reason": finish_reason,
    }
    if parsed is None:
        return {"core_label": "(parse failed)", "core_explanation": "",
                "broad_label": "", "broad_explanation": "",
                "tail_coherence": "noise", "relation": "",
                **empty_scores, "cost_usd": round(cost, 6), "score_details": [],
                "reasoning_debug": reasoning_debug}

    scores = dict(empty_scores)
    score_details = []
    if parsed["core_explanation"] and positives and negatives:
        items = [(t, float(act), f"q{quartile}") for t, act, quartile in positives]
        items += [(t, 0.0, "neg") for t in negatives]
        random.Random(feature).shuffle(items)
        texts = [t for t, _, _ in items]
        preds = []
        for start in range(0, len(texts), SCORE_CHUNK_SIZE):
            chunk = texts[start:start + SCORE_CHUNK_SIZE]
            raw, _, _, spent = client.chat(build_score_messages(
                                               parsed["core_label"], parsed["core_explanation"],
                                               parsed["broad_label"], parsed["broad_explanation"], chunk),
                                           len(chunk) * 8 + 64, thinking=False)
            cost += spent
            chunk_preds = parse_score_output(raw, len(chunk))
            if chunk_preds is None:
                preds = None
                break
            preds.extend(chunk_preds)
        if preds is not None:
            records = [(pred, act, group) for pred, (_, act, group) in zip(preds, items)]
            for quartile in range(1, 5):
                q_preds = [pred for pred, _, group in records if group == f"q{quartile}"]
                if q_preds:
                    scores[f"score_core_q{quartile}"] = round(
                        sum(pred == 2 for pred in q_preds) / len(q_preds), 3)
                    scores[f"score_broad_q{quartile}"] = round(
                        sum(pred >= 1 for pred in q_preds) / len(q_preds), 3)
            neg_preds = [pred for pred, _, group in records if group == "neg"]
            if neg_preds:
                scores["score_neg_core"] = round(
                    sum(pred != 2 for pred in neg_preds) / len(neg_preds), 3)
                scores["score_neg_broad"] = round(
                    sum(pred == 0 for pred in neg_preds) / len(neg_preds), 3)
            positive = [(pred, act) for pred, act, group in records if group != "neg"]
            weight = sum(act for _, act in positive)
            if weight:
                scores["score_core"] = round(
                    sum(act for pred, act in positive if pred == 2) / weight, 3)
                scores["score_broad"] = round(
                    sum(act for pred, act in positive if pred >= 1) / weight, 3)
            decision = {0: "neither", 1: "broad-only", 2: "core"}
            score_details = [
                {"text": text, "activation": act, "group": group,
                 "broad_match": int(pred >= 1), "core_match": int(pred == 2),
                 "prediction": pred, "decision": decision[pred]}
                for pred, (text, act, group) in zip(preds, items)
            ]

    return {**parsed, **scores, "cost_usd": round(cost, 6),
            "score_details": score_details, "reasoning_debug": reasoning_debug}


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
    parser.add_argument("--interp-thinking-max-tokens", type=int,
                        default=INTERP_THINKING_MAX_TOKENS,
                        help="Output-token ceiling for thinking interpretation")
    parser.add_argument("--no-score", action="store_true", help="Skip detection scoring")
    parser.add_argument("--score-per-quartile", type=int, default=SCORE_PER_QUARTILE,
                        help="Held-out activating tweets from each rank quartile")
    parser.add_argument("--score-negatives", type=int, default=SCORE_NEGATIVES,
                        help="Random exact-zero tweets used as negative controls")
    parser.add_argument("--score-details-out", type=str, default=None,
                        help="Optional CSV with every scorer decision for manual auditing")
    parser.add_argument("--reasoning-out", type=str, default=None,
                        help="Optional CSV with interpreter reasoning and raw answers for debugging")
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
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if args.score_details_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.score_details_out)), exist_ok=True)
        if args.overwrite and os.path.exists(args.score_details_out):
            os.remove(args.score_details_out)
    if args.reasoning_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.reasoning_out)), exist_ok=True)
        if args.overwrite and os.path.exists(args.reasoning_out):
            os.remove(args.reasoning_out)
    acts = Activations(args.path)
    features = resolve_features(args, acts)

    done = set()
    if os.path.exists(out_path) and not args.overwrite:
        prev = pd.read_csv(out_path)
        valid = prev
        if "core_label" in prev:
            valid = prev[prev["core_label"].fillna("") != "(parse failed)"]
        done = set(zip(valid["feature"].astype(int), valid["model"]))
        print(f"Resuming: {len(done)} (feature, model) rows already in {out_path}")

    print("Loading tweets...")
    tweets_df = pd.read_csv("data/tweets.csv")
    tweet_ids = tweets_df["tweet_id"].to_numpy()
    texts = tweets_df["text"].to_numpy()
    handles = tweets_df["twitter"].astype(str).to_numpy()

    csv_lock = threading.Lock()

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
                args.score_per_quartile, args.score_negatives, seed=feature)
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
                                    thinking_interp=args.interp_thinking,
                                    thinking_max_tokens=args.interp_thinking_max_tokens)
            score_details = res.pop("score_details", [])
            reasoning_debug = res.pop("reasoning_debug", None)
            main_row = {"feature": j["feature"], "model": MODEL, **res,
                        "n_examples": len(j["examples"]), "n_activating": j["n_act"],
                        "pct_activating": round(j["pct_act"], 4)}
            # Auxiliary records are written first. The compact main row is the
            # completion checkpoint used by resume, so a completed feature can
            # never be skipped while its requested debug records are missing.
            with csv_lock:
                if args.score_details_out and score_details:
                    detail_rows = [{"feature": j["feature"], **item}
                                   for item in score_details]
                    pd.DataFrame(detail_rows).to_csv(
                        args.score_details_out, mode="a", index=False,
                        header=not os.path.exists(args.score_details_out))
                if args.reasoning_out and reasoning_debug is not None:
                    reasoning_row = {"feature": j["feature"], "model": MODEL,
                                     **reasoning_debug,
                                     "total_feature_cost_usd": res["cost_usd"]}
                    pd.DataFrame([reasoning_row]).to_csv(
                        args.reasoning_out, mode="a", index=False,
                        header=not os.path.exists(args.reasoning_out))
                pd.DataFrame([main_row]).to_csv(
                    out_path, mode="a", index=False,
                    header=not os.path.exists(out_path))
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
