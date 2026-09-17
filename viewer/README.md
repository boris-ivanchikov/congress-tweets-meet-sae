---
title: Congress Tweets SAE Explorer
emoji: 🏛️
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
app_file: index.html
pinned: false
---

# Congress Tweets SAE Explorer

An interactive browser for sparse-autoencoder activations learned from embeddings
of U.S. congressional tweets. Each activation is described at two levels:

- **Core interpretation:** the narrow pattern seen in the strongest activations.
- **Broad interpretation:** the related pattern shared by weaker activations.

The Space uses a small FastAPI backend. It provides in-memory search over the
3.25 million tweets embedded for the SAE, exact multi-activation filters, and
retrieves every nonzero activation for a selected tweet. It does not run the
embedding model or SAE online.

## Build the data locally

Run this command from the project root after placing the completed
`interpretations.csv` in `interpretation/runs/chief-dream-73/`:

```bash
.venv/bin/python viewer/build_data.py \
  --path sae/runs/chief-dream-73 \
  --interpretations interpretation/runs/chief-dream-73/interpretations.csv

.venv/bin/python viewer/build_backend_data.py

.venv/bin/python viewer/build_activation_graph.py

# Optional but recommended: replace the quick spring positions with a settled,
# two-stage ForceAtlas2 LinLog constellation. Install its build dependencies first.
npm --prefix viewer install
NODE_PATH=viewer/node_modules node viewer/precompute_forceatlas2.cjs \
  --input viewer/data/activation_graph.json \
  --output /tmp/activation-forceatlas2.json \
  --graph-output viewer/data/activation_graph.json \
  --iterations 3600 --discovery-iterations 1100 \
  --adaptive-cohesion 0.3 --edge-power 2.2 \
  --repulsion 0.03 --gravity 0.05
```

The graph builder samples up to 1,024 activating tweets per activation and
scores frequency-normalized co-occurrence with cosine, lift, and sampling
confidence. It retains mutual top-30 neighbors and hides nodes outside the
main connected component. ForceAtlas2 first discovers the graph's geometry;
layout-aware communities are then inferred from both bonds and proximity. Two
lightweight, invisible bonds per node tighten each emergent community before a
second ForceAtlas2 equilibrium. The old precomputed Louvain partition never
determines positions.

The browser supports node dragging, static positioning, Ctrl/Cmd-click
multi-selection, and Ctrl/Cmd-drag or Shift-drag rectangle selection. Multiple
selections are summarized by activation frequency in the right-hand detail panel.
An optional Barnes-Hut ForceAtlas2 worker reheats after a drag or graph-setting
change, cools through progressively slower phases, updates the layout-aware
colors, and stops at equilibrium. Its LinLog, attraction, repulsion, gravity,
edge-contrast, community-cohesion, and weak-edge controls update the layout in
real time.

The persistent Display rail switches between cleaned and raw tweet text,
core and broad activation interpretations, and optionally hides high-confidence
lexical/formal activations from every tweet-explorer view and from the graph.
The lexical filter is inferred from interpretation phrases such as word family,
substring, prefix, lexeme, spelling, capitalization, or names beginning with;
it does not affect the individual activation explorer.

Preview the Space locally:

```bash
VIEWER_STATIC_DIR=viewer VIEWER_BACKEND_DATA=viewer/backend_data \
  .venv/bin/uvicorn viewer.server:app --port 7860
```

Then open <http://localhost:7860/?feature=1>.

## Deploy

The deployment helper creates or updates
`ivanchikovboris/congress-tweets-sae-explorer`. Visibility is deliberately
required rather than inferred:

```bash
.venv/bin/python viewer/deploy.py --visibility private --upload-data
```

The immutable backend bundle is uploaded into the private Storage Bucket
`ivanchikovboris/congress-tweets-sae-explorer-data`, then mounted read-only
into the Docker Space. Later frontend/backend code-only deployments can omit
`--upload-data`.

The other supported choices are `protected` and `public`. A protected Space
keeps its source repository private while leaving the application public; the
JSON delivered to visitors is still observable by those visitors.

The helper refuses to deploy a partial activation export. It uploads the
prepared Parquet tweet corpus and compact row-oriented activation arrays,
but never the embedding model, SAE weights, API keys, scorer details, or
reasoning logs.
