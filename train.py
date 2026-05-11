import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class EmbeddingLoader:
    def __init__(self, path, batch_size, shuffle=True):
        with np.load(path) as data:
            self.ids = torch.from_numpy(data["ids"])
            self.embeddings = torch.from_numpy(data["embeddings"])
        self.batch_size = batch_size
        if not shuffle:
            self.idx = torch.arange(self.ids.shape[0])
        else:
            self.idx = torch.randperm(self.ids.shape[0])
    
    def __iter__(self):
        for i in range(0, self.idx.shape[0], self.batch_size):
            yield self.ids[self.idx[i:i+self.batch_size]], self.embeddings[self.idx[i:i+self.batch_size]]


class BaseSAE(nn.Module):
    def __init__(self, input_dim, expansion_factor):
        super().__init__()
        self.w_enc = nn.Linear(input_dim, input_dim * expansion_factor)
        self.w_dec = nn.Linear(input_dim * expansion_factor, input_dim)
        self.b_pre = nn.Parameter(torch.zeros(input_dim))
        self.b_enc = nn.Parameter(torch.zeros(input_dim * expansion_factor))

    def encode(self, x):
        return F.relu(self.w_enc(x - self.b_pre) + self.b_enc)
    
    def decode(self, z):
        return self.w_dec(z) + self.b_pre
    
    def forward(self, x):
        raise NotImplementedError()
    

class VanillaSAE(BaseSAE):
    def __init__(self, input_dim, expansion_factor, lmbda):
        super().__init__(input_dim, expansion_factor)
        self.lmbda = lmbda
    
    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        
        mse_loss = F.mse_loss(x_hat, x)
        sparsity_loss = z.norm(p=1, dim=1).mean()
        return {
            "loss": mse_loss + self.lmbda * sparsity_loss,
            "mse_loss": mse_loss,
            "sparsity_loss": sparsity_loss
        }
    

# TODO: transpose init
# TODO: auxillary loss
class TopKSAE(BaseSAE):
    def __init__(self, input_dim, expansion_factor, top_k):
        super().__init__(input_dim, expansion_factor)
        self.top_k = top_k

    def forward(self, x):
        z = self.encode(x)
        _, idx = z.topk(self.top_k, dim=1)
        z = torch.zeros_like(z).scatter_(1, idx, z.gather(1, idx)) # keep top-k only
        x_hat = self.decode(z)
        mse_loss = F.mse_loss(x_hat, x)
        return {"loss": mse_loss, "mse_loss": mse_loss}


class MatryoskaSAE(BaseSAE):
    def __init__(self, input_dim, expansion_factor, top_ks):
        super().__init__(input_dim, expansion_factor)
        self.top_ks = top_ks
    
    def forward(self, x):
        z = self.encode(x)
        losses = []
        for top_k in self.top_ks:
            _, idx = z.topk(top_k, dim=1)
            z = torch.zeros_like(z).scatter_(1, idx, z.gather(1, idx))
            x_hat = self.decode(z)
            mse_loss = F.mse_loss(x_hat, x)
            losses.append(mse_loss)
        return {
            "loss": sum(losses) / len(losses),
            "mse_loss": losses[-1]
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--expansion-factor", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


# TODO
def main(args):
    pass


if __name__ == "__main__":
    args = parse_args()
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(
            backend="nccl", 
            device_id=torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        )
        main(args)
        dist.destroy_process_group()
    else:
        main(args)