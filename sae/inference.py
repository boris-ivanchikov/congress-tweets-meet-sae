import os
import json
import argparse
import torch
from tqdm import tqdm
from model import EmbeddingLoader, init_sae
from loguru import logger
import numpy as np
import scipy
import h5py


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32768)
    return parser.parse_args()


@torch.no_grad()
def main(args):
    if not os.path.exists(args.path):
        raise FileNotFoundError(f"Path {args.path} does not exist")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Creating loader...")
    loader = EmbeddingLoader(path="../data/embeddings.npz", batch_size=args.batch_size, shuffle=False, drop_partial_batch=False)

    logger.info(f"Loading model from {args.path}...")
    config_path = os.path.join(args.path, "config.json")
    weights_path = os.path.join(args.path, "weights.pt")
    output_path = os.path.join(args.path, "activations.h5")
    with open(config_path, "r") as f:
        config = json.load(f)
    model = init_sae(config["model"]).to(device)
    model.load_state_dict(torch.load(weights_path, weights_only=False))

    ids_list = []
    rows = []
    cols = []
    vals = []
    pbar = tqdm(total=len(loader))
    pbar.set_description("Inference")
    for i, (ids, x) in enumerate(loader):
        ids_list.append(ids.cpu().numpy())
        z = model(x.to(device))
        nz_rows, nz_cols = torch.nonzero(z, as_tuple=True)
        nz_rows = nz_rows.cpu().float().numpy()
        nz_cols = nz_cols.cpu().float().numpy()
        nz_vals = z[nz_rows, nz_cols].cpu().float().numpy()

        rows.append(nz_rows + i * args.batch_size)
        cols.append(nz_cols)
        vals.append(nz_vals)
        pbar.update(1)

    ids = np.concatenate(ids_list)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)

    logger.info("Creating sparse matrix...")
    mat = scipy.sparse.csc_matrix(
        (vals, (rows, cols)),
        shape=(ids.shape[0], model.dict_size),
    )
    
    logger.info(f"Saving data to {output_path}...")
    with h5py.File(output_path, "w") as f:
        f["ids"] = ids
        f["data"] = mat.data
        f["indices"] = mat.indices
        f["indptr"] = mat.indptr
        f.attrs["shape"] = mat.shape


if __name__ == '__main__':
    args = parse_args()
    main(args)