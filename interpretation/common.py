import os
import re
import sys
import json
import zipfile

import h5py
import numpy as np
from numpy.lib import format as npy_format
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sae.model import init_sae

MODEL = "Qwen/Qwen3-Embedding-8B"
EMBEDDINGS = "data/embeddings.npz"
PREACT_BATCH = 100_000


def artifacts_dir(sae_path):
    """Output directory for this SAE run's interpretation artifacts.

    Mirrors the run name under interpretation/runs/ so the sae/runs/ input
    directory is never written to.
    """
    name = os.path.basename(os.path.normpath(sae_path))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", name)
    os.makedirs(out, exist_ok=True)
    return out


class Activations:
    """Lazy per-feature access to activations.h5 (CSC layout on disk)."""

    def __init__(self, sae_path):
        self._f = h5py.File(os.path.join(sae_path, "activations.h5"), "r")
        self.indptr = self._f["indptr"][:]
        self.shape = tuple(int(x) for x in self._f.attrs["shape"][:])
        self.ids = self._f["ids"][:]
        self.nnz = np.diff(self.indptr)
        self.pct_active = 100.0 * self.nnz / self.shape[0]

    def col(self, feature):
        """Dense activation column for a 1-indexed feature."""
        s, e = self.indptr[feature - 1], self.indptr[feature]
        out = np.zeros(self.shape[0], dtype=np.float32)
        out[self._f["indices"][s:e]] = self._f["data"][s:e]
        return out


def align_to_tweets(col, act_ids, tweet_ids):
    row_of_id = pd.Series(np.arange(len(act_ids)), index=act_ids)
    rows = row_of_id.reindex(tweet_ids).to_numpy()
    return col[rows]


def load_sae(run_path, device):
    import torch

    with open(os.path.join(run_path, "config.json")) as f:
        config = json.load(f)
    sae = init_sae(config["model"]).to(device)
    sae.load_state_dict(
        torch.load(os.path.join(run_path, "weights.pt"), weights_only=False, map_location="cpu"))
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


def preact_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_preacts(run_path, feature_ids):
    """Pre-activations for the given (1-indexed) features over the whole corpus.

    Returns (ids, {feature_id: float16 column}).
    """
    import torch

    with np.load(EMBEDDINGS) as data:
        pre_ids = data["ids"][:]

    device = preact_device()
    sae = load_sae(run_path, device)
    with torch.no_grad():
        idx = torch.tensor([f - 1 for f in feature_ids], device=device)
        w_sub = sae.w_enc.weight[idx].float()
        b_sub = sae.b_enc[idx].float()
        b_pre = sae.b_pre.float()
        mm = embeddings_memmap()
        n = mm.shape[0]
        out = np.empty((n, len(feature_ids)), dtype=np.float16)
        for s in tqdm(range(0, n, PREACT_BATCH), desc="Pre-activations"):
            e = min(s + PREACT_BATCH, n)
            x = torch.from_numpy(np.array(mm[s:e])).to(device).float()
            out[s:e] = ((x - b_pre) @ w_sub.T + b_sub).cpu().numpy().astype(np.float16)
    del sae

    return pre_ids, {f: out[:, k].astype(np.float32) for k, f in enumerate(feature_ids)}


def human_date(value):
    return str(value)[:16]


_RT_PREFIX = re.compile(r"^RT @\w+:\s*", re.IGNORECASE)
_URL = re.compile(r"https?://\S+")
_WS = re.compile(r"\s+")


def normalize_for_dedup(text):
    t = _RT_PREFIX.sub("", text)
    t = _URL.sub("", t)
    return _WS.sub(" ", t.lower()).strip()
