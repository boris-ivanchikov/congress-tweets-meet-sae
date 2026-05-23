import numpy as np
import torch
import torch.nn as nn

class EmbeddingLoader:
    def __init__(self, path, batch_size, shuffle=True, rank=0, world_size=1): 
        with np.load(path) as data:
            self.ids = torch.from_numpy(data["ids"])
            self.embeddings = torch.from_numpy(data["embeddings"])
        self.batch_size = batch_size
        if not shuffle:
            self.idx = torch.arange(self.ids.shape[0])
        else:
            self.idx = torch.randperm(self.ids.shape[0])
        self.idx = self.idx[rank::world_size]
    
    def __len__(self):
        return self.idx.shape[0] // self.batch_size
    
    def __iter__(self):
        for i in range(0, self.idx.shape[0], self.batch_size):
            yield self.ids[self.idx[i:i+self.batch_size]], self.embeddings[self.idx[i:i+self.batch_size]]


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
        self.threshold = torch.register_buffer("threshold", torch.zeros(1))
    
    def topk(self, z):
        values, idx = z.flatten().topk(self.top_k * z.shape[0], dim=0)
        z_topk = torch.zeros_like(z.flatten()) \
            .scatter_(0, idx, z.flatten().gather(0, idx)) \
            .reshape(z.shape)
        self.threshold = self.threshold * 0.9 + values.min() * 0.1
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