import os
import re
import sys
import json
import html
import random
import zipfile
import argparse
from itertools import zip_longest
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
from numpy.lib import format as npy_format
import scipy.sparse
import pandas as pd
import jinja2
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sae.model import init_sae

MODEL = "Qwen/Qwen3-Embedding-8B"
EMBEDDINGS = "data/embeddings.npz"
MAX_LENGTH = 100
OCCLUSION_BATCH = 64
PREACT_BATCH = 200_000
ATTR_REF_QUANTILE = 0.99
ATTR_ACTIVATION_GAMMA = 1.0
QUARTILE_LABELS = {1: "Q1 — strongest activation", 2: "Q2", 3: "Q3", 4: "Q4 — weakest activation"}

AUTOINTERP_MODEL = "Qwen/Qwen3-32B"
AUTOINTERP_N_EXAMPLES = 25
AUTOINTERP_MAX_TWEET_CHARS = 280
AUTOINTERP_MAX_NEW_TOKENS = 512
AUTOINTERP_SCORE_PER_CLASS = 16

INTERP_SYSTEM = (
    "You are a meticulous AI interpretability researcher studying features of a sparse autoencoder "
    "trained on embeddings of whole tweets from United States members of Congress. Each feature "
    "activates on one shared property of a tweet: a topic, event, named entity, policy area, "
    "rhetorical style, or recurring phrase.\n\n"
    "You will be shown the tweets that most strongly activate a single feature. Describe the "
    "property that is common to the examples.\n\n"
    "Guidelines:\n"
    "- Explain the pattern shared by MOST of the examples, not a detail present in only one or two. "
    "Do NOT over-anchor on a single vivid or attention-grabbing word; identify the general category "
    "that accounts for the whole set. (Illustration: if most examples mention a cat, a dog, and a "
    "hamster, the property is 'pets', even if one example also happens to mention a tarantula.)\n"
    "- Be as specific as the evidence supports, and no more. (Illustration: if the examples describe "
    "rain, snow, and wind, the property is 'weather' — not the narrower 'snowstorms' that only a few "
    "show, nor the vaguer 'nature'.)\n"
    "- Base your answer only on the tweet text. Be concise and politically neutral. Do not make a "
    "list of possible explanations.\n\n"
    "Reason briefly in this order, then stop:\n"
    "1. What stands out across the tweets.\n"
    "2. The pattern shared by most of them.\n"
    "3. Your final answer.\n\n"
    "The LAST line of your response must be a single JSON object and nothing after it:\n"
    '{"label": "<3-6 word noun phrase>", "explanation": "<1-2 sentences>"}'
)

SCORE_SYSTEM = (
    "You are a meticulous AI interpretability researcher. You are given a description of a feature "
    "(for example 'male pronouns' or 'support for Ukraine military aid') and a numbered list of "
    "tweets. For each tweet, decide whether it genuinely exhibits the described property.\n\n"
    "Judge STRICTLY: mark a tweet 1 only if it actually matches the description, not merely because "
    "it shares a word or broad topic with it; otherwise mark 0.\n\n"
    "Return only a JSON array of integers (1 for match, 0 for no match), one per tweet, in order, "
    "and nothing else. Example: [1,0,0,1,0]"
)


def load_activations(path):
    with h5py.File(os.path.join(path, "activations.h5"), "r") as f:
        mat = scipy.sparse.csc_matrix(
            (f["data"][:], f["indices"][:], f["indptr"][:]),
            shape=f.attrs["shape"][:],
        )
        ids = f["ids"][:]
    return ids, mat


def align_to_tweets(col, act_ids, tweet_ids):
    row_of_id = pd.Series(np.arange(len(act_ids)), index=act_ids)
    rows = row_of_id.reindex(tweet_ids).to_numpy()
    return col[rows]


def load_sae(run_path, device):
    import torch

    with open(os.path.join(run_path, "config.json")) as f:
        config = json.load(f)
    sae = init_sae(config["model"]).to(device)
    sae.load_state_dict(torch.load(os.path.join(run_path, "weights.pt"), weights_only=False))
    sae.eval()
    return sae


def read_threshold(run_path):
    import torch

    sd = torch.load(os.path.join(run_path, "weights.pt"), weights_only=False, map_location="cpu")
    return float(sd["threshold"]) if "threshold" in sd else 0.0


def embeddings_memmap(path=EMBEDDINGS):
    zi = zipfile.ZipFile(path).getinfo("embeddings.npy")
    with open(path, "rb") as fh:
        fh.seek(zi.header_offset)
        head = fh.read(30)
        fnl = int.from_bytes(head[26:28], "little")
        efl = int.from_bytes(head[28:30], "little")
        fh.seek(zi.header_offset + 30 + fnl + efl)
        version = npy_format.read_magic(fh)
        readers = {(1, 0): npy_format.read_array_header_1_0, (2, 0): npy_format.read_array_header_2_0}
        shape, fortran, dtype = readers[version](fh)
        offset = fh.tell()
    return np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=shape)


def compute_preacts(run_path, feature_ids):
    import torch

    with np.load(EMBEDDINGS) as data:
        pre_ids = data["ids"][:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae = load_sae(run_path, device)
    with torch.no_grad():
        idx = torch.tensor([f - 1 for f in feature_ids], device=device)
        w_sub = sae.w_enc.weight[idx].float()
        b_sub = sae.b_enc[idx].float()
        b_pre = sae.b_pre.float()
        mm = embeddings_memmap()
        n = mm.shape[0]
        out = np.empty((n, len(feature_ids)), dtype=np.float32)
        for s in tqdm(range(0, n, PREACT_BATCH), desc="Pre-activations"):
            e = min(s + PREACT_BATCH, n)
            x = torch.from_numpy(np.array(mm[s:e])).to(device).float()
            out[s:e] = ((x - b_pre) @ w_sub.T + b_sub).cpu().numpy()
    del sae

    return pre_ids, {f: out[:, k] for k, f in enumerate(feature_ids)}


def human_date(value):
    return str(value)[:16]


def pick_tweets(tweets_df, activations, n):
    df = tweets_df.assign(act=activations)
    activating = df[df["act"] > 0].sort_values("act", ascending=False)

    if len(activating) == 0:
        return [], df[df["act"] == 0].sample(n=0).to_dict("records")

    take_all = len(activating) <= 4 * n
    bounds = np.linspace(0, len(activating), 5, dtype=int)
    parts = []
    for q in range(4):
        g = activating.iloc[bounds[q]:bounds[q + 1]]
        if not take_all:
            g = g.sample(n=n)
        parts.append(g.sort_values("act", ascending=False).assign(quartile=q + 1))
    active = pd.concat(parts)

    non_activating = df[df["act"] == 0]
    not_active = non_activating.sample(n=min(len(active), len(non_activating)))
    return active.to_dict("records"), not_active.to_dict("records")


def gpu_pool():
    import torch

    if not torch.cuda.is_available():
        return [torch.device("cpu")]
    return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]


class FeatureAttributor:
    def __init__(self, base_model, run_path, device):
        import torch
        import transformers

        self.torch = torch
        self.device = device

        with open(os.path.join(run_path, "config.json")) as f:
            config = json.load(f)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL, padding_side="left")
        self.base_model = base_model.eval()

        self.sae = init_sae(config["model"]).to(device)
        self.sae.load_state_dict(
            torch.load(os.path.join(run_path, "weights.pt"), weights_only=False, map_location="cpu"))
        self.sae.eval()
        self.sae_dtype = next(self.sae.parameters()).dtype

        self.embeddings = self.base_model.get_input_embeddings()
        self.mean_emb = self.embeddings.weight.mean(dim=0).detach()

    def _last_token_pool(self, last_hidden_states, attention_mask):
        if attention_mask[:, -1].sum() == attention_mask.shape[0]:
            return last_hidden_states[:, -1]
        lengths = attention_mask.sum(dim=1) - 1
        return last_hidden_states[self.torch.arange(last_hidden_states.shape[0]), lengths]

    def _feature_score(self, inputs_embeds, attention_mask, feature_idx):
        import torch.nn.functional as F
        out = self.base_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        pooled = self._last_token_pool(out.last_hidden_state, attention_mask)
        pooled = F.normalize(pooled, p=2, dim=1).to(self.sae_dtype)
        return self.sae.encode(pooled)[:, feature_idx]

    def attribute(self, text, feature_idx):
        torch = self.torch
        enc = self.tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(self.device)
        input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]
        tokens = [self.tokenizer.decode([i]) for i in input_ids[0].tolist()]

        special = self.tokenizer.get_special_tokens_mask(input_ids[0].tolist(), already_has_special_tokens=True)
        content = [i for i, s in enumerate(special) if s == 0]
        if not content:
            return []

        with torch.inference_mode():
            embeds = self.embeddings(input_ids)
            batch = embeds.repeat(len(content) + 1, 1, 1)
            for row, pos in enumerate(content, start=1):
                batch[row, pos] = self.mean_emb
            mask = attention_mask.repeat(batch.shape[0], 1)
            scores = torch.cat([
                self._feature_score(batch[i:i + OCCLUSION_BATCH], mask[i:i + OCCLUSION_BATCH], feature_idx)
                for i in range(0, batch.shape[0], OCCLUSION_BATCH)
            ]).float()

        deltas = (scores[0] - scores[1:]).cpu()
        denom = deltas.abs().max()
        if denom > 0:
            deltas = deltas / denom
        return [(tokens[pos], deltas[k].item()) for k, pos in enumerate(content)]


class ParallelAttributor:
    def __init__(self, run_path, devices, dtype):
        import copy
        import transformers

        print(f"Loading {MODEL} once ({dtype})...")
        base_cpu = transformers.AutoModel.from_pretrained(MODEL, dtype=dtype)

        print(f"Replicating to {len(devices)} device(s) {[str(d) for d in devices]}...")
        self.replicas = []
        for device in devices:
            base = copy.deepcopy(base_cpu).to(device)
            self.replicas.append(FeatureAttributor(base, run_path, device))
        del base_cpu

    def attribute_all(self, jobs):
        chunks = [[] for _ in self.replicas]
        for i, job in enumerate(jobs):
            chunks[i % len(self.replicas)].append(job)

        bar = tqdm(total=len(jobs), desc="Attribution")

        def work(replica, chunk):
            local = {}
            for key, text, feature_idx in chunk:
                try:
                    local[key] = replica.attribute(text, feature_idx)
                except Exception as exc:
                    print(f"  attribution failed for {key}: {exc}")
                    local[key] = []
                bar.update(1)
            return local

        results = {}
        with ThreadPoolExecutor(max_workers=len(self.replicas)) as ex:
            futures = [ex.submit(work, r, c) for r, c in zip(self.replicas, chunks)]
            for fut in futures:
                results.update(fut.result())
        bar.close()
        return results


def build_attr_jobs(features_raw):
    jobs = []
    for f in features_raw:
        for tweet in f["active"] + f["not_active"]:
            scale = activation_scale(tweet.get("preact", 0.0), f["threshold"], f["attr_ref"])
            if scale > 0:
                jobs.append(((f["id"], tweet["tweet_id"]), str(tweet["text"]), f["id"] - 1))
    return jobs


def preact_hist_data(preact, activated, bins=80):
    edges = np.linspace(float(preact.min()), float(preact.max()), bins + 1)
    inactive, _ = np.histogram(preact[~activated], bins=edges)
    active, _ = np.histogram(preact[activated], bins=edges)
    return {
        "edges": [round(float(x), 6) for x in edges],
        "active": active.astype(int).tolist(),
        "inactive": inactive.astype(int).tolist(),
    }


def user_hist_data(handles, top=10):
    counts = pd.Series(handles).value_counts().head(top)
    return [{"handle": str(h), "count": int(c)} for h, c in counts.items()]


def plain_text_html(text):
    return html.escape(text)


def attribution_html(token_scores, scale=1.0):
    spans = []
    for tok, score in token_scores:
        r, g, b = (46, 160, 67) if score >= 0 else (220, 68, 55)
        alpha = min(abs(score), 1.0) * 0.85 * scale
        spans.append(f'<span class="tok" style="background:rgba({r},{g},{b},{alpha:.3f})">{html.escape(tok)}</span>')
    return "".join(spans)


def make_card(tweet, text_html):
    act = tweet["act"]
    zero = act == 0
    preact = tweet.get("preact")
    return {
        "name": (tweet.get("name") or "").strip(),
        "handle": "@" + str(tweet.get("twitter", "")).strip(),
        "date": human_date(tweet.get("posted_at", "")),
        "badge": f"{preact:+.3f}" if zero and preact is not None else f"{act:.3f}",
        "title": "pre-activation (post-threshold = 0)" if zero else "activation",
        "zero": zero,
        "text_html": text_html,
    }


TEMPLATE = jinja2.Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SAE feature visualization</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         background: #f6f8fa; color: #1f2328; margin: 0; padding: 32px 24px; line-height: 1.5; }
  .wrap { max-width: 1180px; margin: 0 auto; }
  h1 { font-size: 20px; font-weight: 600; margin: 0 0 4px; }
  .sub { color: #656d76; font-size: 13px; margin-bottom: 14px; }
  .controls { grid-column: 1 / -1; display: flex; flex-wrap: wrap; align-items: center; gap: 16px;
              margin: 4px 0 2px; font-size: 12px; color: #656d76; }
  .toggle { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; user-select: none; }
  .tabbar { display: flex; flex-wrap: wrap; gap: 4px; border-bottom: 1px solid #d0d7de; margin-bottom: 22px; }
  .tab { appearance: none; border: 0; background: transparent; cursor: pointer; font: inherit; font-size: 13px;
         color: #656d76; padding: 8px 14px; border-radius: 8px 8px 0 0; border-bottom: 2px solid transparent; margin-bottom: -1px; }
  .tab:hover { color: #1f2328; background: #eef1f4; }
  .tab[aria-selected="true"] { color: #0969da; font-weight: 600; border-bottom-color: #0969da; }
  .panel[hidden] { display: none; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; column-gap: 20px; row-gap: 12px; align-items: stretch; }
  .col-label { font-size: 11px; font-weight: 600; letter-spacing: .04em; text-transform: uppercase;
               color: #656d76; margin: 0; }
  .qhead { grid-column: 1 / -1; font-size: 12px; font-weight: 600; color: #424a53;
           background: #eaedf1; border-radius: 6px; padding: 5px 10px; margin-top: 10px; }
  .card { background: #fff; border: 1px solid #d8dee4; border-radius: 10px; padding: 12px 14px;
          height: 100%; box-shadow: 0 1px 2px rgba(31,35,40,.04); }
  .card-head { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; margin-bottom: 6px; }
  .who { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .name { font-weight: 600; font-size: 13px; }
  .handle, .date { color: #656d76; font-size: 12px; }
  .badge { font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; background: #eaf3ff; color: #0969da;
           border-radius: 999px; padding: 2px 8px; white-space: nowrap; flex: none; }
  .badge.zero { background: #f0f1f3; color: #656d76; }
  .text { font-size: 14px; white-space: pre-wrap; overflow-wrap: anywhere; }
  .tok { border-radius: 3px; transition: background .12s; }
  body.attr-off .tok { background: transparent !important; }
  .chip { display: inline-block; width: 11px; height: 11px; border-radius: 3px; vertical-align: middle; margin: 0 3px 0 10px; }
  .summary { margin-bottom: 22px; }
  .stat { font-size: 14px; margin: 0 0 14px; color: #424a53; }
  .stat b { color: #1f2328; font-size: 16px; }
  .autointerp { background: linear-gradient(180deg,#fff,#fbfcfe); border: 1px solid #c8d3e0;
                border-left: 4px solid #0969da; border-radius: 10px; padding: 14px 16px;
                margin-bottom: 16px; box-shadow: 0 1px 2px rgba(31,35,40,.05); }
  .ai-kicker { font-size: 10px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
               color: #0969da; margin: 0 0 4px; }
  .ai-label { font-size: 18px; font-weight: 700; color: #1f2328; margin: 0 0 6px; }
  .ai-expl { font-size: 14px; color: #424a53; margin: 0; }
  .ai-meta { font-size: 11px; color: #8a929b; margin: 8px 0 0;
             font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  .chart { background: #fff; border: 1px solid #d8dee4; border-radius: 10px; padding: 12px 14px;
           box-shadow: 0 1px 2px rgba(31,35,40,.04); margin-bottom: 14px; }
  .chart-head { display: flex; align-items: baseline; justify-content: space-between; gap: 12px;
                flex-wrap: wrap; margin-bottom: 6px; }
  .chart-title { font-size: 12px; font-weight: 600; letter-spacing: .03em; text-transform: uppercase; color: #424a53; }
  .legend { font-size: 11px; color: #656d76; display: flex; align-items: center; gap: 4px; flex-wrap: wrap; }
  .legend .chip { margin: 0 2px 0 8px; }
  .chart-svg { width: 100%; }
  .chart svg { display: block; width: 100%; height: auto; }
  .chart svg text { fill: #656d76; font: 10px ui-monospace, SFMono-Regular, Menlo, monospace; }
  .axis { stroke: #d0d7de; }
  .gridline { stroke: #eef1f4; }
  .bar-inactive { fill: #8b97a3; }
  .bar-active { fill: #2ea043; }
  .bar-user { fill: #0969da; }
  .hoverband { fill: #1f2328; fill-opacity: 0; cursor: default; }
  .hoverband:hover { fill-opacity: .05; }
  .chart-tip { position: fixed; pointer-events: none; z-index: 20; background: #1f2328; color: #fff;
               font: 11px ui-monospace, SFMono-Regular, Menlo, monospace; line-height: 1.45;
               padding: 6px 9px; border-radius: 7px; opacity: 0; transition: opacity .08s;
               box-shadow: 0 4px 14px rgba(31,35,40,.25); white-space: nowrap; }
  .chart-tip b { color: #fff; font-weight: 600; }
  .chart-tip .tip-g { color: #57d977; } .chart-tip .tip-x { color: #adb5bd; }
</style>
</head>
<body>
<div class="wrap">
  <h1>SAE feature visualization</h1>
  <div class="sub">{{ run }} &middot; {{ features|length }} feature(s){% if attributed %} &middot; token attribution via occlusion{% endif %}</div>

  <div class="tabbar" role="tablist">
    {% for f in features %}<button class="tab" role="tab" data-target="panel-{{ f.id }}" aria-selected="{{ 'true' if loop.first else 'false' }}">Feature {{ f.id }}</button>
    {% endfor %}
  </div>

  {% macro card(c) %}
  <div class="card">
    <div class="card-head">
      <div class="who"><span class="name">{{ c.name }}</span>
        <span class="handle">{{ c.handle }}</span> &middot; <span class="date">{{ c.date }}</span></div>
      <span class="badge {{ 'zero' if c.zero }}" title="{{ c.title }}">{{ c.badge }}</span>
    </div>
    <div class="text">{{ c.text_html|safe }}</div>
  </div>
  {% endmacro %}

  {% for f in features %}
  <section class="panel" id="panel-{{ f.id }}" role="tabpanel"{% if not loop.first %} hidden{% endif %}>
    <div class="summary">
      {% if f.autointerp %}
      <div class="autointerp">
        <p class="ai-kicker">Auto-interpretation</p>
        <p class="ai-label">{{ f.autointerp.label }}</p>
        <p class="ai-expl">{{ f.autointerp.explanation }}</p>
        <p class="ai-meta">{% if f.autointerp.score is not none %}detection acc {{ f.autointerp.score }} ({{ f.autointerp.n_pos + f.autointerp.n_neg }} held-out) &middot; {% endif %}{{ f.autointerp.n_examples }} examples &middot; {{ f.autointerp.model }}</p>
      </div>
      {% endif %}
      <p class="stat">Activates on <b>{{ f.summary.n_act }}</b> / {{ f.summary.total }} tweets ({{ f.summary.pct }})</p>
      <div class="chart" data-kind="preact" data-fid="{{ f.id }}">
        <div class="chart-head">
          <span class="chart-title">Pre-activation distribution</span>
          <span class="legend">
            <span class="chip" style="background:#2ea043"></span>activating
            <span class="chip" style="background:#8b97a3"></span>not activating
          </span>
        </div>
        <div class="chart-svg"></div>
      </div>
      <div class="chart" data-kind="users" data-fid="{{ f.id }}">
        <div class="chart-head"><span class="chart-title">Top users by activating tweets</span></div>
        <div class="chart-svg"></div>
      </div>
    </div>
    <div class="grid">
      <p class="col-label">Activating</p>
      <p class="col-label">Not activating (random)</p>
      {% if attributed %}
      <div class="controls">
        <label class="toggle"><input type="checkbox" class="attrToggle" checked> Token attribution</label>
        <span><span class="chip" style="background:rgba(46,160,67,.7)"></span>pushes feature up
          <span class="chip" style="background:rgba(220,68,55,.7)"></span>pushes feature down</span>
        <span>&middot; intensity &prop; |pre-activation &minus; threshold|</span>
      </div>
      {% endif %}
      {% for b in f.blocks %}
      <div class="qhead">{{ b.label }}</div>
      {% for row in b.rows %}{% if row[0] %}{{ card(row[0]) }}{% else %}<div></div>{% endif %}{% if row[1] %}{{ card(row[1]) }}{% else %}<div></div>{% endif %}{% endfor %}
      {% endfor %}
    </div>
  </section>
  {% endfor %}
</div>
<div class="chart-tip" id="tip"></div>
<script>
  var CHART_DATA = {{ chart_json|safe }};
  var NS = "http://www.w3.org/2000/svg";
  var tip = document.getElementById('tip');

  function svgEl(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function fmt(n) {
    if (n >= 1e6) return (n / 1e6).toFixed(n % 1e6 ? 1 : 0) + 'M';
    if (n >= 1e3) return (n / 1e3).toFixed(n % 1e3 ? 1 : 0) + 'k';
    return '' + n;
  }
  function showTip(html, evt) {
    tip.innerHTML = html;
    tip.style.opacity = 1;
    var x = evt.clientX + 14, y = evt.clientY + 14;
    if (x + tip.offsetWidth > window.innerWidth - 8) x = evt.clientX - tip.offsetWidth - 14;
    if (y + tip.offsetHeight > window.innerHeight - 8) y = evt.clientY - tip.offsetHeight - 14;
    tip.style.left = x + 'px';
    tip.style.top = y + 'px';
  }
  function hideTip() { tip.style.opacity = 0; }

  function renderPreact(host, d) {
    var W = host.clientWidth, H = 300;
    var m = { t: 14, r: 14, b: 34, l: 50 };
    var pw = W - m.l - m.r, ph = H - m.t - m.b;
    var nb = d.active.length;
    var x0 = d.edges[0], x1 = d.edges[nb];
    var sx = function (v) { return m.l + (v - x0) / (x1 - x0) * pw; };
    var maxc = 1;
    for (var i = 0; i < nb; i++) maxc = Math.max(maxc, d.active[i], d.inactive[i]);
    var lu = Math.log10(maxc + 1);
    var sh = function (c) { return c <= 0 ? 0 : Math.log10(c + 1) / lu * ph; };
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H }, host);

    for (var p = 1; p <= maxc; p *= 10) {
      var gy = m.t + ph - sh(p);
      svgEl('line', { class: 'gridline', x1: m.l, x2: m.l + pw, y1: gy, y2: gy }, svg);
      var tl = svgEl('text', { x: m.l - 6, y: gy + 3, 'text-anchor': 'end' }, svg);
      tl.textContent = fmt(p);
    }
    svgEl('line', { class: 'axis', x1: m.l, x2: m.l + pw, y1: m.t + ph, y2: m.t + ph }, svg);
    for (var t = 0; t <= 5; t++) {
      var xv = x0 + (x1 - x0) * t / 5, xx = sx(xv);
      var xl = svgEl('text', { x: xx, y: m.t + ph + 14, 'text-anchor': 'middle' }, svg);
      xl.textContent = xv.toFixed(3);
    }

    function bars(arr, cls) {
      for (var i = 0; i < nb; i++) {
        if (arr[i] <= 0) continue;
        var bx = sx(d.edges[i]), bw = Math.max(0.6, sx(d.edges[i + 1]) - bx), bh = sh(arr[i]);
        svgEl('rect', { class: cls, x: bx, y: m.t + ph - bh, width: bw, height: bh }, svg);
      }
    }
    bars(d.inactive, 'bar-inactive');
    bars(d.active, 'bar-active');

    for (var i = 0; i < nb; i++) {
      var bx = sx(d.edges[i]), bw = sx(d.edges[i + 1]) - bx;
      var band = svgEl('rect', { class: 'hoverband', x: bx, y: m.t, width: Math.max(0.6, bw), height: ph }, svg);
      (function (i) {
        band.addEventListener('mousemove', function (e) {
          showTip('<b>[' + d.edges[i].toFixed(4) + ', ' + d.edges[i + 1].toFixed(4) + ']</b><br>' +
            '<span class="tip-g">activating</span> ' + fmt(d.active[i]) + '<br>' +
            '<span class="tip-x">not activating</span> ' + fmt(d.inactive[i]), e);
        });
        band.addEventListener('mouseleave', hideTip);
      })(i);
    }
  }

  function renderUsers(host, users) {
    var n = users.length;
    if (!n) { host.innerHTML = '<svg viewBox="0 0 10 10"></svg>'; return; }
    var rh = 22, m = { t: 8, r: 44, b: 24, l: 150 };
    var W = host.clientWidth, H = m.t + m.b + n * rh;
    var pw = W - m.l - m.r;
    var maxc = users[0].count;
    var sx = function (c) { return c / maxc * pw; };
    var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H }, host);

    for (var t = 0; t <= 4; t++) {
      var cv = Math.round(maxc * t / 4), gx = m.l + sx(cv);
      svgEl('line', { class: 'gridline', x1: gx, x2: gx, y1: m.t, y2: m.t + n * rh }, svg);
      var xl = svgEl('text', { x: gx, y: m.t + n * rh + 15, 'text-anchor': 'middle' }, svg);
      xl.textContent = fmt(cv);
    }
    svgEl('line', { class: 'axis', x1: m.l, x2: m.l, y1: m.t, y2: m.t + n * rh }, svg);

    users.forEach(function (u, i) {
      var y = m.t + i * rh, bw = sx(u.count);
      svgEl('rect', { class: 'bar-user', x: m.l, y: y + 3, width: Math.max(1, bw), height: rh - 6, rx: 2 }, svg);
      var lab = svgEl('text', { x: m.l - 8, y: y + rh / 2 + 3, 'text-anchor': 'end' }, svg);
      lab.textContent = '@' + u.handle;
      var val = svgEl('text', { x: m.l + bw + 6, y: y + rh / 2 + 3 }, svg);
      val.textContent = fmt(u.count);
      var band = svgEl('rect', { class: 'hoverband', x: 0, y: y, width: W, height: rh }, svg);
      band.addEventListener('mousemove', function (e) {
        showTip('<b>@' + u.handle + '</b><br>' + u.count.toLocaleString() + ' activating tweets', e);
      });
      band.addEventListener('mouseleave', hideTip);
    });
  }

  function renderCharts() {
    document.querySelectorAll('.chart').forEach(function (c) {
      var host = c.querySelector('.chart-svg');
      if (!host.clientWidth) return;
      host.innerHTML = '';
      var d = CHART_DATA[c.dataset.fid];
      if (c.dataset.kind === 'preact') renderPreact(host, d.preact);
      else renderUsers(host, d.users);
    });
  }

  document.querySelectorAll('.tab').forEach(function (btn) {
    btn.addEventListener('click', function () {
      document.querySelectorAll('.tab').forEach(function (t) { t.setAttribute('aria-selected', t === btn); });
      document.querySelectorAll('.panel').forEach(function (p) { p.hidden = p.id !== btn.dataset.target; });
      renderCharts();
    });
  });
  document.querySelectorAll('.attrToggle').forEach(function (t) {
    t.addEventListener('change', function () {
      document.querySelectorAll('.attrToggle').forEach(function (o) { o.checked = t.checked; });
      document.body.classList.toggle('attr-off', !t.checked);
    });
  });

  var rt;
  window.addEventListener('resize', function () { clearTimeout(rt); rt = setTimeout(renderCharts, 150); });
  window.addEventListener('load', renderCharts);
  renderCharts();
</script>
</body>
</html>""")


def activation_scale(preact, threshold, ref):
    if ref <= 0:
        return 0.0
    return min(abs(preact - threshold) / ref, 1.0) ** ATTR_ACTIVATION_GAMMA


def cards_for(tweets, attr_map, feature_id, threshold, ref):
    cards = []
    for tweet in tweets:
        text = str(tweet["text"])
        if attr_map is not None:
            scale = activation_scale(tweet.get("preact", 0.0), threshold, ref)
            pairs = attr_map.get((feature_id, tweet.get("tweet_id")))
            text_html = attribution_html(pairs, scale) if (pairs and scale > 0) else plain_text_html(text)
        else:
            text_html = plain_text_html(text)
        cards.append(make_card(tweet, text_html))
    return cards


def render_report(run, features_raw, attr_map):
    features = []
    for f in features_raw:
        active_cards = cards_for(f["active"], attr_map, f["id"], f["threshold"], f["attr_ref"])
        not_active_cards = cards_for(f["not_active"], attr_map, f["id"], f["threshold"], f["attr_ref"])

        groups = {q: [] for q in (1, 2, 3, 4)}
        for tweet, card in zip(f["active"], active_cards):
            groups[tweet["quartile"]].append(card)

        blocks, offset = [], 0
        for q in (1, 2, 3, 4):
            if not groups[q]:
                continue
            right = not_active_cards[offset:offset + len(groups[q])]
            offset += len(groups[q])
            rows = [list(p) for p in zip_longest(groups[q], right)]
            blocks.append({"label": QUARTILE_LABELS[q], "rows": rows})

        features.append({"id": f["id"], "blocks": blocks, "summary": f["summary"],
                         "autointerp": f.get("autointerp")})

    chart_json = json.dumps({
        f["id"]: {"preact": f["summary"]["preact"], "users": f["summary"]["users"]}
        for f in features_raw
    })
    return TEMPLATE.render(run=run, features=features, attributed=attr_map is not None, chart_json=chart_json)


def parse_args():
    parser = argparse.ArgumentParser(description="Render SAE feature activations as a tabbed HTML report.")
    parser.add_argument("--path", type=str, required=True, help="Path to run directory")
    parser.add_argument("--features", nargs="+", type=int, required=True, help="Feature indices (1-indexed)")
    parser.add_argument("--n", type=int, default=5, help="Activating tweets sampled per quartile (4*n total)")
    parser.add_argument("--out", type=str, default=None, help="Output HTML path (default: <path>/report.html)")
    parser.add_argument("--token-attr", action="store_true", help="Color tokens by occlusion attribution")
    parser.add_argument("--autointerp", action="store_true", help="Generate a local-LLM interpretation per feature")
    parser.add_argument("--autointerp-model", type=str, default=AUTOINTERP_MODEL, help="HF model id for autointerp")
    parser.add_argument("--autointerp-examples", type=int, default=AUTOINTERP_N_EXAMPLES,
                        help="Top-K strongest distinct tweets fed to the LLM")
    return parser.parse_args()


def build_summary(act_col, preact_col, handles):
    activated = act_col > 0
    n_act, total = int(activated.sum()), len(act_col)
    return {
        "n_act": f"{n_act:,}",
        "total": f"{total:,}",
        "pct": f"{100 * n_act / total:.3f}%",
        "preact": preact_hist_data(preact_col, activated),
        "users": user_hist_data(handles[activated]),
    }


_RT_PREFIX = re.compile(r"^RT @\w+:\s*", re.IGNORECASE)
_URL = re.compile(r"https?://\S+")
_WS = re.compile(r"\s+")


def normalize_for_dedup(text):
    t = _RT_PREFIX.sub("", text)
    t = _URL.sub("", t)
    return _WS.sub(" ", t.lower()).strip()


def select_interp_examples(texts, k, max_chars):
    seen, out = set(), []
    for text in texts:
        t = str(text).strip()
        norm = normalize_for_dedup(t)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(t[:max_chars])
        if len(out) >= k:
            break
    return out, seen


def _take_distinct(indices, texts, seen, max_chars, k):
    out = []
    for j in indices:
        t = str(texts[j]).strip()
        norm = normalize_for_dedup(t)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(t[:max_chars])
        if len(out) >= k:
            break
    return out


def select_score_pool(act_col, preact_col, texts, exclude_norms, n, max_chars, seed):
    rng = random.Random(seed)
    seen = set(exclude_norms)

    # positives: held-out activating tweets spread across the full activation-strength range
    act_order = np.argsort(-act_col)
    n_act = int((act_col > 0).sum())
    positives = []
    if n_act:
        cand = act_order[np.linspace(0, n_act - 1, min(n_act, n * 8)).round().astype(int)]
        held = _take_distinct(cand, texts, seen, max_chars, len(cand))
        if len(held) > n:
            positives = [held[i] for i in np.linspace(0, len(held) - 1, n).round().astype(int)]
        else:
            positives = held

    # negatives, balanced to #positives: half HARD (non-activating but nearest the threshold),
    # half random non-activating — so accuracy isn't inflated by trivially-different negatives
    target = len(positives)
    nonact = np.where(act_col == 0)[0]
    negatives = []
    if len(nonact) and target:
        n_hard = target // 2
        k = min(len(nonact), n_hard * 4)
        hard_local = np.argpartition(-preact_col[nonact], k - 1)[:k]
        hard = nonact[hard_local[np.argsort(-preact_col[nonact][hard_local])]]
        negatives = _take_distinct(hard, texts, seen, max_chars, n_hard)

        rand_local = rng.sample(range(len(nonact)), min(len(nonact), target * 4))
        negatives += _take_distinct(nonact[rand_local], texts, seen, max_chars, target - len(negatives))
    return positives, negatives


def build_interp_messages(examples):
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(examples))
    user = (f"Here are {len(examples)} tweets that most strongly activate this SAE feature, "
            f"in no particular order:\n\n{numbered}\n\n"
            "What single concept does this feature represent? Reply with only the JSON object.")
    return [{"role": "system", "content": INTERP_SYSTEM}, {"role": "user", "content": user}]


def build_score_messages(explanation, texts):
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    user = (f"Concept: {explanation}\n\nTweets:\n{numbered}\n\n"
            f"Reply with a JSON array of {len(texts)} integers (0 or 1), one per tweet in order.")
    return [{"role": "system", "content": SCORE_SYSTEM}, {"role": "user", "content": user}]


def parse_interp_output(raw):
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            label = str(obj.get("label", "")).strip()
            explanation = str(obj.get("explanation", "")).strip()
            if label:
                return label[:80], explanation
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
    lines = [l for l in text.splitlines() if l.strip()]
    return (lines[0].strip()[:80] if lines else "(uninterpretable)",
            " ".join(l.strip() for l in lines[1:]).strip())


def parse_score_output(raw, n):
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
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


class FeatureInterpreter:
    def __init__(self, model_id):
        import torch
        import transformers

        self.torch = torch
        self.model_id = model_id

        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"Loading {model_id} for autointerp (fp16, device_map=auto across {n_gpu} GPU(s))...")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float16, device_map="auto"
        )
        self.model.eval()
        self.input_device = self.model.get_input_embeddings().weight.device
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _generate(self, messages, max_new_tokens):
        torch = self.torch
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.input_device)
        with torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tokenizer.pad_token_id)
        gen = out[0, enc["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)

    def _score(self, explanation, positives, negatives, feature_id):
        items = [(t, 1) for t in positives] + [(t, 0) for t in negatives]
        random.Random(feature_id).shuffle(items)
        texts = [t for t, _ in items]
        labels = [l for _, l in items]
        raw = self._generate(build_score_messages(explanation, texts), len(texts) * 3 + 32)
        preds = parse_score_output(raw, len(texts))
        if preds is None:
            return None
        return round(sum(int(p == l) for p, l in zip(preds, labels)) / len(labels), 3)

    def interpret(self, feature_id, examples, positives, negatives):
        if not examples:
            return {"label": "(no activating tweets)",
                    "explanation": "This feature did not activate on any tweet in the corpus.",
                    "score": None, "n_examples": 0, "n_pos": 0, "n_neg": 0, "model": self.model_id}

        label, explanation = parse_interp_output(
            self._generate(build_interp_messages(examples), AUTOINTERP_MAX_NEW_TOKENS))
        score = None
        if explanation and positives and negatives:
            score = self._score(explanation, positives, negatives, feature_id)
        return {"label": label, "explanation": explanation, "score": score,
                "n_examples": len(examples), "n_pos": len(positives), "n_neg": len(negatives),
                "model": self.model_id}


def main(args):
    tweets_df = pd.read_csv("data/tweets.csv")
    act_ids, mat = load_activations(args.path)
    tweet_ids = tweets_df["tweet_id"].to_numpy()
    handles = tweets_df["twitter"].astype(str).to_numpy()

    print("Computing pre-activations...")
    pre_ids, preacts = compute_preacts(args.path, args.features)
    threshold = read_threshold(args.path)
    texts = tweets_df["text"].to_numpy()

    features_raw = []
    for feature in args.features:
        full_act = mat[:, feature - 1].toarray().flatten()
        preact_full = preacts[feature]
        attr_ref = float(np.quantile(np.abs(preact_full - threshold), ATTR_REF_QUANTILE))
        summary = build_summary(
            align_to_tweets(full_act, act_ids, pre_ids),
            preact_full,
            align_to_tweets(handles, tweet_ids, pre_ids),
        )

        act_col = align_to_tweets(full_act, act_ids, tweet_ids)
        preact_col = align_to_tweets(preact_full, pre_ids, tweet_ids)
        work_df = tweets_df.assign(preact=preact_col)
        active, not_active = pick_tweets(work_df, act_col, args.n)

        examples, score_pos, score_neg = [], [], []
        if args.autointerp:
            top_texts = []
            for j in np.argsort(-act_col):
                if act_col[j] <= 0:
                    break
                top_texts.append(texts[j])
                if len(top_texts) >= 4 * args.autointerp_examples:
                    break
            examples, ex_norms = select_interp_examples(top_texts, args.autointerp_examples, AUTOINTERP_MAX_TWEET_CHARS)
            score_pos, score_neg = select_score_pool(
                act_col, preact_col, texts, ex_norms, AUTOINTERP_SCORE_PER_CLASS, AUTOINTERP_MAX_TWEET_CHARS, feature)

        features_raw.append({"id": feature, "active": active, "not_active": not_active,
                             "summary": summary, "threshold": threshold, "attr_ref": attr_ref,
                             "interp_examples": examples, "score_pos": score_pos, "score_neg": score_neg})
        print(f"Feature {feature}: {len(active)} activating, {len(not_active)} non-activating")

    attr_map = None
    if args.token_attr:
        import torch
        devices = gpu_pool()
        dtype = torch.float16 if devices[0].type == "cuda" else torch.float32
        attributor = ParallelAttributor(args.path, devices, dtype)
        attr_map = attributor.attribute_all(build_attr_jobs(features_raw))
        del attributor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.autointerp:
        interp = FeatureInterpreter(args.autointerp_model)
        for f in tqdm(features_raw, desc="Auto-interp"):
            f["autointerp"] = interp.interpret(f["id"], f["interp_examples"], f["score_pos"], f["score_neg"])
        del interp
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path = args.out or os.path.join(args.path, "report.html")
    with open(out_path, "w") as f:
        f.write(render_report(args.path, features_raw, attr_map))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    args = parse_args()
    main(args)
