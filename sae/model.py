import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SAE(nn.Module):
    def __init__(self, input_dim: int, expansion_factor: int, top_k: int, batch_top_k: bool):
        super().__init__()
        self.dict_size = input_dim * expansion_factor
        self.top_k = top_k
        self.batch_top_k = batch_top_k
        self.w_enc = nn.Linear(input_dim, input_dim * expansion_factor, bias=False)
        self.w_dec = nn.Linear(input_dim * expansion_factor, input_dim, bias=False)
        self.b_pre = nn.Parameter(torch.zeros(input_dim))
        self.b_enc = nn.Parameter(torch.zeros(input_dim * expansion_factor))

    def encode(self, x):
        z = F.relu(self.w_enc(x - self.b_pre) + self.b_enc)
        return z
    
    def topk(self, z):
        if not self.config.batch_top_k:
            _, idx = z.topk(self.config.top_k, dim=1)
            z_topk = torch.zeros_like(z).scatter_(1, idx, z.gather(1, idx))
        else:
            _, idx = z.flatten().topk(self.config.top_k * z.shape[0], dim=0)
            z_topk = torch.zeros_like(z.flatten()) \
                .scatter_(0, idx, z.flatten().gather(0, idx)) \
                .reshape(z.shape)
        return z_topk
    
    def decode(self, z, prefix=None):
        if prefix is None:
            prefix = self.dict_size
        return z[:, :prefix] @ self.w_dec.weight.T[:prefix] + self.b_pre
    
    def forward(self, x):
        return self.encode(x)
    
    @torch.no_grad()
    def normalize_decoder_weights(self):
        w = self.w_dec.weight
        w_normed = w / w.norm(dim=0, keepdim=True)
        grad_proj = (w.grad * w_normed).sum(dim=0, keepdim=True) * w_normed
        w.grad -= grad_proj
        w.data = w_normed

    @classmethod
    def load_checkpoint(cls, path):
        config_path = os.path.join(path, "config.json")
        weights_path = os.path.join(path, "weights.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint {path} not found")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"weights.pt not found in {path}")
        with open(config_path, "r") as f:
            config = json.load(f)
        model = cls(config["input_dim"], config["expansion_factor"])
        model.load_state_dict(torch.load(weights_path))
        return model
        