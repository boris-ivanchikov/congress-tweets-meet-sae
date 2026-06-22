import os
import random
import argparse
from tqdm import tqdm
from pydantic import BaseModel
import yaml
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
import namer
from model import EmbeddingLoader, init_sae


def normalized_mse(x, x_hat):
    error = x - x_hat
    x_c = x - x.mean(dim=0)
    return (error ** 2).sum() / (x_c ** 2).sum()


class TrainingConfig(BaseModel):
    epochs: int
    batch_size: int
    lr: float
    dead_num_iters: int
    aux_top_k: int
    aux_loss_weight: float


class TrainingGraph(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.config = config
        self.model = model
        self.register_buffer('dead_counter', torch.zeros(model.dict_size))
    
    def forward(self, x):
        z = self.model.encode(x)
        z_topk = self.model.topk(z)
        x_hat = self.model.decode(z_topk)
        if x_hat.dim() == 2:
            reconstruction_loss = normalized_mse(x, x_hat)
            reconstruction_loss_last = reconstruction_loss
        elif x_hat.dim() == 3: # matryoshka
            n = x_hat.shape[1]
            reconstruction_loss = sum(normalized_mse(x, x_hat[:, i, :]) for i in range(n)) / n
            reconstruction_loss_last = normalized_mse(x, x_hat[:, -1, :])

        self.dead_counter += 1
        fired = (z_topk > 0).any(dim=0).float()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(fired, op=dist.ReduceOp.MAX)
        self.dead_counter[fired > 0] = 0
        dead_mask = self.dead_counter > self.config.dead_num_iters

        if self.config.aux_loss_weight > 0 and dead_mask.any():
            error = (x - x_hat) if x_hat.dim() == 2 else (x - x_hat[:, -1, :])
            num_dead = dead_mask.sum().item()
            _, idx = z[:, dead_mask].topk(min(self.config.aux_top_k, num_dead), dim=1)
            dead_vals = z[:, dead_mask].gather(1, idx)
            dead_decoder = self.model.w_dec.weight[:, dead_mask].T
            z_dead_sparse = dead_vals.new_zeros(x.shape[0], num_dead).scatter(1, idx, dead_vals)
            reconstructed_error = z_dead_sparse @ dead_decoder
            aux_loss = normalized_mse(error.detach(), reconstructed_error)
        else:
            aux_loss = torch.tensor(0.0, device=x.device)

        loss = reconstruction_loss + self.config.aux_loss_weight * aux_loss

        return {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss_last,
            "aux_loss": aux_loss,
            "num_dead": dead_mask.sum().item(),
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    return parser.parse_args()


def main(args):
    torch.random.manual_seed(42)

    distributed = dist.is_available() and dist.is_initialized()
    if distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    with open(args.config, "r") as f:
        data = yaml.safe_load(f)
    config = TrainingConfig(**data["training"])

    if is_main:
        logger.info("Creating graph...")
    model = init_sae(data["model"])
    graph = TrainingGraph(model, config).to(device)
    ddp = DDP(graph, device_ids=[local_rank]) if distributed else graph

    name = None
    writer = None
    if is_main:
        name = namer.generate() + "-" + str(random.randint(10, 99))
        logger.info(f"Starting run {name}")
        writer = SummaryWriter(f"sae/runs/{name}")

    if is_main:
        logger.info("Creating loader...")
    loader = EmbeddingLoader(path="data/embeddings.npz", batch_size=config.batch_size,
                             shuffle=True, rank=rank, world_size=world_size)

    optimizer = torch.optim.Adam(graph.parameters(), lr=config.lr)

    pbar = tqdm(total=config.epochs * len(loader)) if is_main else None
    for epoch in range(config.epochs):
        loader.set_epoch(epoch)
        if is_main:
            pbar.set_description(f"Epoch {epoch+1}/{config.epochs}")
        for i, (ids, x) in enumerate(loader):
            outputs = ddp(x.to(device))
            loss = outputs["loss"]
            loss.backward()
            graph.model.normalize_decoder_weights()

            if is_main:
                global_step = epoch * len(loader) + i
                writer.add_scalar("loss", loss, global_step=global_step)
                writer.add_scalar("reconstruction_loss", outputs["reconstruction_loss"], global_step=global_step)
                writer.add_scalar("aux_loss", outputs["aux_loss"], global_step=global_step)
                writer.add_scalar("num_dead", outputs["num_dead"], global_step=global_step)

            optimizer.step()
            optimizer.zero_grad()
            if is_main:
                pbar.update(1)

    if is_main:
        torch.save(graph.model.state_dict(), f"sae/runs/{name}/weights.pt")
        with open(f"sae/runs/{name}/config.json", "w") as f:
            json.dump(data, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    args.config = os.path.abspath(args.config)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
        main(args)
        dist.destroy_process_group()
    else:
        main(args)