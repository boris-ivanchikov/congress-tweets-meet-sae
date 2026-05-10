import os
import argparse
import tempfile
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import transformers

MODEL = "Qwen/Qwen3-Embedding-8B"


class TweetsDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_csv(path)
        self.data["tweet_id"] = self.data["tweet_id"].astype(np.int64)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            self.data = self.data.loc[idx].reset_index(drop=True)
            return self
        return self.data.loc[idx, "tweet_id"], self.data.loc[idx, "text"]


class TweetsCollator:
    def __init__(self, tokenizer: transformers.AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        ids, texts = zip(*batch)
        ids = torch.tensor(ids)
        texts = self.tokenizer(texts, padding=True, truncation=True, max_length=100, return_tensors="pt")
        return ids, texts


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main(args):
    if dist.is_initialized():
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = torch.device("cuda")

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL, padding_side='left')
    model = transformers.AutoModel.from_pretrained(MODEL, dtype=torch.float16).to(device)
    model.eval()

    dataset = TweetsDataset("data/tweets.csv")
    collator = TweetsCollator(tokenizer)

    if args.limit is not None:
        dataset = dataset[:args.limit]

    if dist.is_initialized() and dist.get_world_size() > 1:
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler, 
        collate_fn=collator,
        prefetch_factor=args.prefetch_factor,
        num_workers=args.num_workers,
        pin_memory=True
    )

    all_ids = []
    all_embeddings = []
    with torch.inference_mode():
        for ids, seqs in tqdm(dataloader):
            seqs = seqs.to(device)
            outputs = model(**seqs)
            embeddings = last_token_pool(outputs.last_hidden_state, seqs["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)

            all_ids.append(ids.cpu().numpy())
            all_embeddings.append(embeddings.cpu().numpy())
    
    ids = np.concatenate(all_ids)
    embeddings = np.concatenate(all_embeddings)

    if dist.is_initialized():
        rank = dist.get_rank()
        with tempfile.TemporaryDirectory() as tmpdir:
            np.savez(os.path.join(tmpdir, f"embeddings_{rank}.npz"), ids=ids, embeddings=embeddings)
            dist.barrier()
        
            if rank == 0:
                chunks = [np.load(os.path.join(tmpdir, f"embeddings_{r}.npz") for r in range(dist.get_world_size()))]
                np.savez("data/embeddings.npz",
                        ids=np.concatenate([c["ids"] for c in chunks]),
                        embeddings=np.concatenate([c["embeddings"] for c in chunks]))
    else:
        np.savez("data/embeddings.npz", ids=ids, embeddings=embeddings)


if __name__ == "__main__":
    args = parse_args()
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        main(args)
        dist.destroy_process_group()
    else:
        main(args)