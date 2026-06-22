import zipfile
import numpy as np
from numpy.lib import format as npy_format
import torch
import torch.nn as nn


def _npz_memmap(path, member):
    zi = zipfile.ZipFile(path).getinfo(member)
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


class EmbeddingLoader:
    def __init__(
            self,
            path,
            batch_size,
            shuffle=True,
            drop_partial_batch=True,
            rank=0,
            world_size=1,
            seed=42,
        ):
        emb = _npz_memmap(path, "embeddings.npy")
        ids = _npz_memmap(path, "ids.npy")
        per = emb.shape[0] // world_size
        start, end = rank * per, rank * per + per
        self.ids = torch.from_numpy(np.array(ids[start:end]))
        self.embeddings = torch.from_numpy(np.array(emb[start:end]))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_partial_batch = drop_partial_batch
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        if self.drop_partial_batch:
            return self.ids.shape[0] // self.batch_size
        return (self.ids.shape[0] + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed + self.epoch)
            idx = torch.randperm(self.ids.shape[0], generator=g)
        else:
            idx = torch.arange(self.ids.shape[0])
        for i in range(len(self)):
            sl = idx[i * self.batch_size:(i + 1) * self.batch_size]
            yield self.ids[sl], self.embeddings[sl]


class TopKSAE(nn.Module):
    def __init__(self, input_dim: int, expansion_factor: int, top_k: int):
        super().__init__()
        self.input_dim = input_dim
        self.expansion_factor = expansion_factor
        self.dict_size = input_dim * expansion_factor
        self.top_k = top_k

        self.w_enc = nn.Linear(input_dim, input_dim * expansion_factor, bias=False)
        self.w_dec = nn.Linear(input_dim * expansion_factor, input_dim, bias=False)
        self.b_pre = nn.Parameter(torch.zeros(input_dim))
        self.b_enc = nn.Parameter(torch.zeros(input_dim * expansion_factor))
    
    def encode(self, x):
        return self.w_enc(x - self.b_pre) + self.b_enc
    
    def topk(self, z):
        _, idx = z.topk(self.top_k, dim=1)
        z_topk = torch.zeros_like(z).scatter_(1, idx, z.gather(1, idx))
        return z_topk
    
    def decode(self, z):
        return self.w_dec(z) + self.b_pre
    
    def forward(self, x):
        return self.topk(self.encode(x))
    
    @torch.no_grad()
    def normalize_decoder_weights(self):
        w = self.w_dec.weight
        w_normed = w / w.norm(dim=0, keepdim=True)
        grad_proj = (w.grad * w_normed).sum(dim=0, keepdim=True) * w_normed
        w.grad -= grad_proj
        w.data = w_normed
        
        
class BatchTopKSAE(TopKSAE):
    def __init__(self, input_dim: int, expansion_factor: int, top_k: int):
        super().__init__(input_dim, expansion_factor, top_k)
        self.register_buffer("threshold", torch.zeros(1))
    
    def topk(self, z):
        values, idx = z.flatten().topk(self.top_k * z.shape[0], dim=0)
        z_topk = torch.zeros_like(z.flatten()) \
            .scatter_(0, idx, z.flatten().gather(0, idx)) \
            .reshape(z.shape)
        self.threshold = self.threshold.detach() * 0.9 + values.min().item() * 0.1
        return z_topk

    def forward(self, x):
        z = self.encode(x)
        z[z < self.threshold] = 0.0
        return z


class MatryoshkaSAE(BatchTopKSAE):
    def __init__(
            self, 
            input_dim: int, 
            expansion_factor: int,
            top_k: int, 
            prefixes: list[int]
        ):
        super().__init__(input_dim, expansion_factor, top_k)
        self.prefixes = sorted(prefixes)
        if self.dict_size not in self.prefixes:
            raise ValueError(f"Dict size ({self.dict_size}) is required in prefixes")
    
    def decode(self, z):
        return torch.stack([
            z[:, :prefix] @ self.w_dec.weight.T[:prefix] + self.b_pre
            for prefix in self.prefixes
        ], dim=1)


SAE_REGISTRY = {
    "TopKSAE": TopKSAE,
    "BatchTopKSAE": BatchTopKSAE,
    "MatryoshkaSAE": MatryoshkaSAE,
}


def init_sae(cfg: dict) -> TopKSAE:
    cls = SAE_REGISTRY[cfg["type"]]
    return cls(**{k: v for k, v in cfg.items() if k != "type"})