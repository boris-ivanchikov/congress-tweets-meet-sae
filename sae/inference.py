import os
import json
import argparse
import torch
from tqdm import tqdm
from model import EmbeddingLoader, init_sae
from loguru import logger
from scipy import sparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32768)
    return parser.parse_args()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Creating loader...")
    loader = EmbeddingLoader(path="../data/embeddings.npz", batch_size=args.batch_size, shuffle=False)

    logger.info(f"Loading model from {args.path}...")
    config_path = os.path.join(args.path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    model = init_sae(config["model"]).to(device)

    for i, (ids, x) in tqdm(enumerate(loader)):
        z = model(x.to(device))
        # TODO


if __name__ == '__main__':
    args = parse_args()
    main(args)