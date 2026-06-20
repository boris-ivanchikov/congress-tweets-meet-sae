import os
import sys
import json
import html
import argparse

import h5py
import scipy.sparse
import pandas as pd
import jinja2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sae.model import init_sae

MODEL = "Qwen/Qwen3-Embedding-8B"
MAX_LENGTH = 100
ATTR_STEPS = 24
ATTR_BATCH_SIZE = 4


def load_activations(path):
    with h5py.File(os.path.join(path, "activations.h5"), "r") as f:
        return scipy.sparse.csc_matrix(
            (f["data"][:], f["indices"][:], f["indptr"][:]),
            shape=f.attrs["shape"][:],
        )


def human_date(value):
    return str(value)[:16]


def pick_tweets(tweets_df, activations, args):
    tweets_df = tweets_df.assign(act=activations)

    activating = tweets_df[tweets_df["act"] > 0].sort_values("act", ascending=False)
    top = activating.iloc[: args.num_top]
    rest = activating.iloc[args.num_top :]
    sampled = rest.sample(n=min(args.num_random, len(rest)))
    active = pd.concat([top, sampled])

    not_activating = tweets_df[tweets_df["act"] == 0]
    not_active = not_activating.sample(n=min(len(active), len(not_activating)))

    return active.to_dict("records"), not_active.to_dict("records")


class FeatureAttributor:
    def __init__(self, run_path, device, dtype):
        import torch
        import transformers

        self.torch = torch
        self.device = device

        with open(os.path.join(run_path, "config.json")) as f:
            config = json.load(f)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL, padding_side="left")
        self.base_model = transformers.AutoModel.from_pretrained(MODEL, dtype=dtype).to(device)
        self.base_model.eval()
        self.base_model.requires_grad_(False)

        self.sae = init_sae(config["model"]).to(device)
        self.sae.load_state_dict(torch.load(os.path.join(run_path, "weights.pt"), weights_only=False))
        self.sae.eval()
        self.sae.requires_grad_(False)
        self.sae_dtype = next(self.sae.parameters()).dtype

        from captum.attr import LayerIntegratedGradients
        self.lig = LayerIntegratedGradients(self._forward, self.base_model.get_input_embeddings())

    def _last_token_pool(self, last_hidden_states, attention_mask):
        if attention_mask[:, -1].sum() == attention_mask.shape[0]:
            return last_hidden_states[:, -1]
        lengths = attention_mask.sum(dim=1) - 1
        return last_hidden_states[self.torch.arange(last_hidden_states.shape[0]), lengths]

    def _forward(self, input_ids, attention_mask, feature_idx):
        import torch.nn.functional as F
        out = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._last_token_pool(out.last_hidden_state, attention_mask)
        pooled = F.normalize(pooled, p=2, dim=1).to(self.sae_dtype)
        return self.sae.encode(pooled)[:, feature_idx]

    def attribute(self, text, feature_idx):
        torch = self.torch
        enc = self.tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(self.device)
        input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]
        baseline = torch.full_like(input_ids, self.tokenizer.pad_token_id or 0)

        attributions = self.lig.attribute(
            inputs=input_ids,
            baselines=baseline,
            additional_forward_args=(attention_mask, feature_idx),
            n_steps=ATTR_STEPS,
            internal_batch_size=ATTR_BATCH_SIZE,
        )

        scores = attributions.sum(dim=-1).squeeze(0).float()
        denom = scores.abs().max()
        scores = scores / denom if denom > 0 else scores

        tokens = [self.tokenizer.decode([i]) for i in input_ids[0].tolist()]
        return list(zip(tokens, scores.tolist()))


def plain_text_html(text):
    return html.escape(text)


def attribution_html(token_scores):
    spans = []
    for tok, score in token_scores:
        r, g, b = (46, 160, 67) if score >= 0 else (220, 68, 55)
        alpha = min(abs(score), 1.0) * 0.85
        spans.append(f'<span class="tok" style="background:rgba({r},{g},{b},{alpha:.3f})">{html.escape(tok)}</span>')
    return "".join(spans)


def make_card(tweet, text_html):
    act = tweet["act"]
    return {
        "name": (tweet.get("name") or "").strip(),
        "handle": "@" + str(tweet.get("twitter", "")).strip(),
        "date": human_date(tweet.get("posted_at", "")),
        "act": f"{act:.3f}",
        "zero": act == 0,
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
  .sub { color: #656d76; font-size: 13px; margin-bottom: 28px; }
  .feature { margin-bottom: 44px; }
  .feature > h2 { font-size: 16px; font-weight: 600; margin: 0 0 14px;
                  padding-bottom: 8px; border-bottom: 1px solid #d0d7de; }
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; align-items: start; }
  .col-label { font-size: 11px; font-weight: 600; letter-spacing: .04em; text-transform: uppercase;
               color: #656d76; margin: 0 0 10px; }
  .card { background: #fff; border: 1px solid #d8dee4; border-radius: 10px; padding: 12px 14px;
          margin-bottom: 12px; box-shadow: 0 1px 2px rgba(31,35,40,.04); }
  .card-head { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; margin-bottom: 6px; }
  .who { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .name { font-weight: 600; font-size: 13px; }
  .handle, .date { color: #656d76; font-size: 12px; }
  .badge { font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; background: #eaf3ff; color: #0969da;
           border-radius: 999px; padding: 2px 8px; white-space: nowrap; flex: none; }
  .badge.zero { background: #f0f1f3; color: #656d76; }
  .text { font-size: 14px; white-space: pre-wrap; overflow-wrap: anywhere; }
  .tok { border-radius: 3px; }
  .legend { font-size: 11px; color: #656d76; margin: 2px 0 12px; }
  .chip { display: inline-block; width: 11px; height: 11px; border-radius: 3px; vertical-align: middle; margin: 0 3px 0 10px; }
  .empty { color: #909aa4; font-size: 13px; font-style: italic; }
</style>
</head>
<body>
<div class="wrap">
  <h1>SAE feature visualization</h1>
  <div class="sub">{{ run }} &middot; {{ features|length }} feature(s){% if attributed %} &middot; token attribution via integrated gradients{% endif %}</div>

  {% macro card(c) %}
  <div class="card">
    <div class="card-head">
      <div class="who"><span class="name">{{ c.name }}</span>
        <span class="handle">{{ c.handle }}</span> &middot; <span class="date">{{ c.date }}</span></div>
      <span class="badge {{ 'zero' if c.zero }}">{{ c.act }}</span>
    </div>
    <div class="text">{{ c.text_html|safe }}</div>
  </div>
  {% endmacro %}

  {% for f in features %}
  <section class="feature">
    <h2>Feature {{ f.id }}</h2>
    <div class="cols">
      <div>
        <p class="col-label">Activating</p>
        {% if attributed %}<div class="legend">
          <span class="chip" style="background:rgba(46,160,67,.7)"></span>pushes feature up
          <span class="chip" style="background:rgba(220,68,55,.7)"></span>pushes feature down</div>{% endif %}
        {% for c in f.active %}{{ card(c) }}{% else %}<p class="empty">No activating tweets.</p>{% endfor %}
      </div>
      <div>
        <p class="col-label">Not activating (random)</p>
        {% for c in f.not_active %}{{ card(c) }}{% else %}<p class="empty">No tweets.</p>{% endfor %}
      </div>
    </div>
  </section>
  {% endfor %}
</div>
</body>
</html>""")


def render_report(run, features_raw, attributor):
    features = []
    for f in features_raw:
        active_cards = []
        for tweet in f["active"]:
            text = str(tweet["text"])
            if attributor is not None:
                try:
                    text_html = attribution_html(attributor.attribute(text, f["id"] - 1))
                except Exception as exc:
                    print(f"  attribution failed for feature {f['id']}: {exc}")
                    text_html = plain_text_html(text)
            else:
                text_html = plain_text_html(text)
            active_cards.append(make_card(tweet, text_html))
        not_active_cards = [make_card(t, plain_text_html(str(t["text"]))) for t in f["not_active"]]
        features.append({"id": f["id"], "active": active_cards, "not_active": not_active_cards})
    return TEMPLATE.render(run=run, features=features, attributed=attributor is not None)


def parse_args():
    parser = argparse.ArgumentParser(description="Render SAE feature activations as a side-by-side HTML report.")
    parser.add_argument("--path", type=str, required=True, help="Path to run directory")
    parser.add_argument("--features", nargs="+", type=int, required=True, help="Feature indices (1-indexed)")
    parser.add_argument("--num-top", type=int, default=10, help="Number of top activating tweets")
    parser.add_argument("--num-random", type=int, default=10, help="Number of random activating tweets from the rest")
    parser.add_argument("--out", type=str, default=None, help="Output HTML path (default: <path>/visualization.html)")
    parser.add_argument("--feature-attr", action="store_true", help="Color tokens by integrated-gradients attribution")
    return parser.parse_args()


def main(args):
    tweets_df = pd.read_csv("data/tweets.csv")
    mat = load_activations(args.path)

    features_raw = []
    for feature in args.features:
        active, not_active = pick_tweets(tweets_df, mat[:, feature - 1].toarray().flatten(), args)
        features_raw.append({"id": feature, "active": active, "not_active": not_active})
        print(f"Feature {feature}: {len(active)} activating, {len(not_active)} non-activating")

    attributor = None
    if args.feature_attr:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        print(f"Loading {MODEL} for attribution on {device} ({dtype})...")
        attributor = FeatureAttributor(args.path, device, dtype)

    out_path = args.out or os.path.join(args.path, "visualization.html")
    with open(out_path, "w") as f:
        f.write(render_report(args.path, features_raw, attributor))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    args = parse_args()
    main(args)
