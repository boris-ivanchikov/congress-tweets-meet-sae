import os
import argparse
import torch
from tqdm import tqdm
from model import EmbeddingLoader, SAE
from loguru import logger


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
    model = SAE.load_checkpoint(args.path)

    for i, (ids, x) in tqdm(enumerate(loader)):
        z_topk = model.topk(model.encode(x.to(device)))
        # TODO


if __name__ == '__main__':
    args = parse_args()
    main(args)