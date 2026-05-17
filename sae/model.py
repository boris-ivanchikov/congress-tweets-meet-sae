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
    def __init__(self, input_dim, expansion_factor):
        super().__init__()
        self.dict_size = input_dim * expansion_factor
        self.w_enc = nn.Linear(input_dim, input_dim * expansion_factor, bias=False)
        self.w_dec = nn.Linear(input_dim * expansion_factor, input_dim, bias=False)
        self.b_pre = nn.Parameter(torch.zeros(input_dim))
        self.b_enc = nn.Parameter(torch.zeros(input_dim * expansion_factor))

    def encode(self, x):
        z = F.relu(self.w_enc(x - self.b_pre) + self.b_enc)
        return z
    
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