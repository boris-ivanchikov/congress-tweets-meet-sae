"""Render SAE feature activations as a tabbed HTML report.

Reads a trained SAE run directory (read-only input); artifacts are read from and
written to the mirrored interpretation/runs/<run-name>/ directory.
Auto-interpretation labels come from interpretations.csv (see autointerp.py);
token-level attribution is computed here by occlusion (needs the embedding model).
"""

import os
import json
import html
import argparse
from itertools import zip_longest
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import jinja2
from tqdm import tqdm

from common import (MODEL, Activations, align_to_tweets, artifacts_dir, read_threshold,
                    compute_preacts, human_date)
from sae.model import init_sae

MAX_LENGTH = 100
OCCLUSION_BATCH = 64
ATTR_REF_QUANTILE = 0.99
ATTR_ACTIVATION_GAMMA = 1.0
QUARTILE_LABELS = {1: "Q1 — strongest activation", 2: "Q2", 3: "Q3", 4: "Q4 — weakest activation"}


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
        <p class="ai-meta">{{ f.autointerp.meta }}</p>
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


def load_interpretations(csv_path, model=None):
    """feature -> autointerp dict from interpret.py output (preferring `model` if given)."""
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    if model:
        df = df[df["model"] == model]
    out = {}
    for _, row in df.drop_duplicates("feature", keep="last").iterrows():
        meta = []
        if pd.notna(row.get("type")):
            meta.append(str(row["type"]))
        if pd.notna(row.get("score")):
            ci = ""
            if pd.notna(row.get("score_lo")):
                ci = f" [{row['score_lo']:.2f}, {row['score_hi']:.2f}]"
            meta.append(f"detection acc {row['score']}{ci} "
                        f"({int(row['n_pos']) + int(row['n_neg'])} held-out)")
        if pd.notna(row.get("score_weighted")):
            meta.append(f"act-weighted {row['score_weighted']}")
        meta.append(f"{int(row['n_examples'])} examples")
        meta.append(str(row["model"]))
        out[int(row["feature"])] = {
            "label": row["label"],
            "explanation": row["explanation"] if pd.notna(row["explanation"]) else "",
            "meta": " · ".join(meta),
        }
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Render SAE feature activations as a tabbed HTML report.")
    parser.add_argument("--path", type=str, required=True,
                        help="Path to trained SAE run directory (read-only input)")
    parser.add_argument("--features", nargs="+", type=int, required=True, help="Feature indices (1-indexed)")
    parser.add_argument("--n", type=int, default=5, help="Activating tweets sampled per quartile (4*n total)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output HTML path (default: interpretation/runs/<run>/report.html)")
    parser.add_argument("--token-attr", action="store_true", help="Color tokens by occlusion attribution")
    parser.add_argument("--interpretations", type=str, default=None,
                        help="interpretations.csv from autointerp.py "
                             "(default: interpretation/runs/<run>/interpretations.csv)")
    parser.add_argument("--interp-model", type=str, default=None,
                        help="Which model's interpretations to display (default: last per feature)")
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


def main(args):
    out_dir = artifacts_dir(args.path)
    tweets_df = pd.read_csv("data/tweets.csv")
    acts = Activations(args.path)
    tweet_ids = tweets_df["tweet_id"].to_numpy()
    handles = tweets_df["twitter"].astype(str).to_numpy()

    print("Computing pre-activations...")
    pre_ids, preacts = compute_preacts(args.path, args.features)
    threshold = read_threshold(args.path)

    interp_csv = args.interpretations or os.path.join(out_dir, "interpretations.csv")
    interps = load_interpretations(interp_csv, args.interp_model)

    features_raw = []
    for feature in args.features:
        full_act = acts.col(feature)
        preact_full = preacts[feature]
        attr_ref = float(np.quantile(np.abs(preact_full - threshold), ATTR_REF_QUANTILE))
        summary = build_summary(
            align_to_tweets(full_act, acts.ids, pre_ids),
            preact_full,
            align_to_tweets(handles, tweet_ids, pre_ids),
        )

        act_col = align_to_tweets(full_act, acts.ids, tweet_ids)
        preact_col = align_to_tweets(preact_full, pre_ids, tweet_ids)
        work_df = tweets_df.assign(preact=preact_col)
        active, not_active = pick_tweets(work_df, act_col, args.n)

        features_raw.append({"id": feature, "active": active, "not_active": not_active,
                             "summary": summary, "threshold": threshold, "attr_ref": attr_ref,
                             "autointerp": interps.get(feature)})
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

    out_path = args.out or os.path.join(out_dir, "report.html")
    with open(out_path, "w") as f:
        f.write(render_report(args.path, features_raw, attr_map))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main(parse_args())
