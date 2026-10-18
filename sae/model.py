import h5py
import numpy as np
import torch
import torch.nn as nn

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
            load_into_ram=False,
        ):
        self.path = path
        with h5py.File(path, "r") as f:
            per = f["embeddings"].shape[0] // world_size
            self.start, self.end = rank * per, (rank + 1) * per
            self.ids = torch.from_numpy(f["ids"][self.start:self.end])
            shape = f["embeddings"].shape
            dtype = f["embeddings"].dtype
            self.embeddings = None
            if load_into_ram:
                self.embeddings = np.empty((self.end - self.start, shape[1]), dtype=dtype)
                if self.end > self.start:
                    f["embeddings"].read_direct(
                        self.embeddings, source_sel=np.s_[self.start:self.end, :],
                    )
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_partial_batch = drop_partial_batch
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def mean(self):
        with h5py.File(self.path, "r") as f:
            emb = self.embeddings if self.embeddings is not None else f["embeddings"]
            total = np.zeros(emb.shape[1], dtype=np.float64)
            begin, end = (0, len(emb)) if self.embeddings is not None else (self.start, self.end)
            for start in range(begin, end, 8192):
                total += emb[start:min(start + 8192, end)].sum(axis=0, dtype=np.float64)
        if self.end == self.start:
            raise ValueError("Cannot compute the mean of an empty shard")
        return torch.from_numpy((total / (self.end - self.start)).astype(np.float32))

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
        if self.embeddings is not None:
            for i in range(len(self)):
                sl = idx[i * self.batch_size:(i + 1) * self.batch_size]
                batch = self.embeddings[sl.numpy()]
                yield self.ids[sl], torch.from_numpy(batch)
            return
        with h5py.File(self.path, "r") as f:
            emb = f["embeddings"]
            for i in range(len(self)):
                sl = idx[i * self.batch_size:(i + 1) * self.batch_size]
                if self.shuffle:
                    order = np.argsort(sl.numpy())
                    rows = sl.numpy()[order] + self.start
                    batch = np.empty((len(sl), emb.shape[1]), dtype=emb.dtype)
                    for j in range(0, len(rows), 1024):
                        batch[order[j:j + 1024]] = emb[rows[j:j + 1024]]
                else:
                    start = self.start + i * self.batch_size
                    batch = emb[start:min(start + self.batch_size, self.end)]
                yield self.ids[sl], torch.from_numpy(batch)


class BatchTopKSAE(nn.Module):
    def __init__(self, input_dim: int, expansion_factor: int, top_k: int):
        super().__init__()
        self.input_dim = input_dim
        self.expansion_factor = expansion_factor
        self.dict_size = input_dim * expansion_factor
        self.top_k = top_k
        self.register_buffer("threshold", torch.zeros(1))

        self.w_enc = nn.Linear(input_dim, input_dim * expansion_factor, bias=False)
        self.w_dec = nn.Linear(input_dim * expansion_factor, input_dim, bias=False)
        self.b_pre = nn.Parameter(torch.zeros(input_dim))
        self.b_enc = nn.Parameter(torch.zeros(input_dim * expansion_factor))

        self.w_dec.weight.data /= self.w_dec.weight.data.norm(dim=0, keepdim=True)
        self.w_enc.weight.data = self.w_dec.weight.data.T.clone()
    
    def encode(self, x):
        return self.w_enc(x - self.b_pre) + self.b_enc
    
    def apply_activation(self, z):
        if self.training:
            values, idx = z.flatten().topk(self.top_k * z.shape[0], dim=0)
            z_sparse = torch.zeros_like(z.flatten()) \
                .scatter_(0, idx, z.flatten().gather(0, idx)) \
                .reshape(z.shape)
            self.threshold = self.threshold.detach() * 0.9 + values.min().item() * 0.1
        else:
            z_sparse = z * (z > self.threshold)

        return z_sparse
    
    def decode(self, z_sparse):
        return self.w_dec(z_sparse) + self.b_pre
    
    def forward(self, x):
        return self.apply_activation(self.encode(x))
    
    @torch.no_grad()
    def project_decoder_grads(self):
        w = self.w_dec.weight
        w_normed = w / w.norm(dim=0, keepdim=True)
        w.grad -= (w.grad * w_normed).sum(dim=0, keepdim=True) * w_normed

    @torch.no_grad()
    def normalize_decoder_weights(self):
        w = self.w_dec.weight
        w.data /= w.data.norm(dim=0, keepdim=True)

