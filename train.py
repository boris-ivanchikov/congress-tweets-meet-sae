import os
import argparse
from tqdm import tqdm
from pydantic import BaseModel, model_validator, computed_field
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class Config(BaseModel):
    input_dim: int = 4096
    expansion_factor: int = 8
    top_k: int | None = None
    batch_top_k: bool = False
    dead_num_iters: int | None = None
    aux_top_k: int | None = None
    aux_loss_weight: float = 0.0
    matryoshka_prefixes: list[int] | None = None

    @computed_field
    @property
    def dict_size(self) -> int:
        return self.input_dim * self.expansion_factor

    @model_validator(mode="after")
    def validate_and_infer(self) -> "Config":
        if self.dead_num_iters is None and self.aux_loss_weight > 0:
            raise ValueError("dead_num_iters must be specified if aux_loss_weight > 0")
        if self.top_k is None:
            self.top_k = self.dict_size
        if self.aux_top_k is None:
            self.aux_top_k = min(self.top_k * 2, self.dict_size)
        if self.matryoshka_prefixes is None:
            self.matryoshka_prefixes = [self.dict_size]
        if self.dict_size not in self.matryoshka_prefixes:
            raise ValueError(f"matryoshka_prefixes must include full dict_size ({self.dict_size})")
        self.matryoshka_prefixes.sort()

        return self


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


class TrainingGraph(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.model = SAE(config.input_dim, config.expansion_factor).to(device)
        self.dead_counter = torch.zeros(config.dict_size).to(device)
    

    def forward(self, x):
        z = self.model.encode(x)
       
        if not self.config.batch_top_k:
            _, idx = z.topk(self.config.top_k, dim=1)
            z_topk = torch.zeros_like(z).scatter_(1, idx, z.gather(1, idx))
        else:
            _, idx = z.flatten().topk(self.config.top_k * z.shape[0], dim=0)
            z_topk = torch.zeros_like(z.flatten()) \
                .scatter_(0, idx, z.flatten().gather(0, idx)) \
                .reshape(z.shape)
            
        errors = torch.stack([(x - self.model.decode(z_topk, prefix)) for prefix in self.config.matryoshka_prefixes])
        reconstruction_loss = (errors ** 2).mean()

        self.dead_counter += 1
        self.dead_counter[(z_topk > 0).any(dim=0)] = 0
        dead_mask = self.dead_counter > self.config.dead_num_iters

        if self.config.aux_loss_weight > 0 and dead_mask.any():
            _, idx = z[:, dead_mask].topk(min(self.config.aux_top_k, dead_mask.sum().item()), dim=1)
            dead_topk = z[:, dead_mask].gather(1, idx).unsqueeze(1)
            dead_topk_w_dec = self.model.w_dec.weight[:, dead_mask].T[idx]
            reconstructed_error = (dead_topk @ dead_topk_w_dec).squeeze(1)
            aux_loss = F.mse_loss(reconstructed_error, errors[-1])
        else:
            aux_loss = torch.tensor(0.0, device=self.device)

        loss = reconstruction_loss + self.config.aux_loss_weight * aux_loss

        return {
            "loss": loss, 
            "reconstruction_loss": reconstruction_loss,
            "aux_loss": aux_loss,
        }
    

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
        return self.idx.shape[0]
    
    def __iter__(self):
        for i in range(0, self.idx.shape[0], self.batch_size):
            yield self.ids[self.idx[i:i+self.batch_size]], self.embeddings[self.idx[i:i+self.batch_size]]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--prefetch-factor', type=int, default=2)
    parser.add_argument('--limit', type=int, default=None)
    return parser.parse_args()


def main(args):
    with open(args.config, "r") as f:
        data = yaml.safe_load(f)
    config = Config(**data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.random.manual_seed(42)
    loader = EmbeddingLoader(path="data/embeddings.npz", batch_size=args.batch_size, shuffle=True)
    graph = TrainingGraph(config, device=device)
    optimizer = torch.optim.Adam(graph.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(loader))

    pbar = tqdm(total=args.epochs * len(loader))
    for epoch in range(args.epoch):
        pbar.set_description(f"Epoch {epoch+1}/{args.epochs}")
        for i, (ids, x) in enumerate(loader):
            optimizer.zero_grad()
            outputs = graph(x)
            loss = outputs["loss"]
            loss.backward()
            graph.model.normalize_decoder_weights()
            optimizer.step()
            scheduler.step()
            pbar.update(1)


if __name__ == "__main__":
    args = parse_args()
    main(args)