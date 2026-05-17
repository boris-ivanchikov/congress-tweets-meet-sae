import random
import argparse
from tqdm import tqdm
from pydantic import BaseModel, model_validator, computed_field
import yaml
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
import namer
from model import EmbeddingLoader, SAE


class Config(BaseModel):
    # model
    input_dim: int = 4096
    expansion_factor: int = 8
    top_k: int | None = None
    batch_top_k: bool = False

    # training
    epochs: int
    batch_size: int
    lr: float
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


class TrainingGraph(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = SAE(
            input_dim=config.input_dim, 
            expansion_factor=config.expansion_factor,
            top_k=config.top_k,
            batch_top_k=config.batch_top_k,
        )
        self.dead_counter = torch.zeros(config.dict_size)
    
    def forward(self, x):
        z = self.model.encode(x)
        z_topk = self.model.topk(z)
            
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
            aux_loss = torch.tensor(0.0, device=x.device)

        loss = reconstruction_loss + self.config.aux_loss_weight * aux_loss

        return {
            "loss": loss, 
            "reconstruction_loss": reconstruction_loss,
            "aux_loss": aux_loss,
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    return parser.parse_args()


def main(args):
    with open(args.config, "r") as f:
        data = yaml.safe_load(f)
    config = Config(**data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.random.manual_seed(42)

    name = namer.generate() + "-" + str(random.randint(10, 99))
    logger.info(f"Starting run {name}")
    writer = SummaryWriter(f"runs/{name}")

    logger.info("Creating loader...")
    loader = EmbeddingLoader(path="../data/embeddings.npz", batch_size=config.batch_size, shuffle=True)

    logger.info("Creating graph...")
    graph = TrainingGraph(config).to(device)

    logger.info("Creating optimizer and sheduler...")
    optimizer = torch.optim.Adam(graph.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs * len(loader))

    pbar = tqdm(total=config.epochs * len(loader))
    for epoch in range(config.epochs):
        pbar.set_description(f"Epoch {epoch+1}/{config.epochs}")
        for i, (ids, x) in enumerate(loader):
            outputs = graph(x.to(device))
            loss = outputs["loss"]
            loss.backward()
            graph.model.normalize_decoder_weights()

            global_step = epoch * len(loader) + i
            writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step=global_step)
            writer.add_scalar("loss", loss, global_step=global_step)
            writer.add_scalar("reconstruction_loss", outputs["reconstruction_loss"], global_step=global_step)
            writer.add_scalar("aux_loss", outputs["aux_loss"], global_step=global_step)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            pbar.update(1)

    torch.save(graph.model.state_dict(), f"runs/{name}/weights.pt")
    with open(f"runs/{name}/config.json", "w") as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)