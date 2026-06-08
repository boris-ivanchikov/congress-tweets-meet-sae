import os
import argparse
from rich.console import Console
from rich.table import Table
import pandas as pd
import h5py
import scipy
from itertools import zip_longest

def display_tweet(t: dict):
    return " - ".join([t["name"], "@" + t["twitter"], t["posted_at"], t["text"]]) + f" ({t["act"]:.3f})"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to activations")
    parser.add_argument("--features", nargs='+', type=int, required=True, help="Feature indices (1-indexed)")
    parser.add_argument("--num-tweets", type=int, default=10, help="Number of tweets")
    parser.add_argument("--threshold", type=float, default=None, help="Threshold for activations")
    return parser.parse_args()

def main(args):
    tweets_df = pd.read_csv("data/tweets.csv")
    
    with h5py.File(args.path, "r") as f:
        data = f["data"][:]
        indices = f["indices"][:]
        indptr = f["indptr"][:]
        shape = f.attrs["shape"][:]
    
    mat = scipy.sparse.csc_matrix(
        (data, indices, indptr),
        shape=shape,
    )
        
    console = Console(width=180)

    for feature in map(int, args.features):
        table = Table(show_lines=True)
        table.add_column("Activated", width=85)
        table.add_column("Not Activated", width=85)

        activations = mat[:, feature - 1].toarray().flatten()
        tweets_df["act"] = activations
        
        if args.threshold is not None:
            active = tweets_df[activations > args.threshold]
            active = active.sample(n=min(args.num_tweets, active.shape[0]))
        else:
            active = tweets_df[activations > 0].sort_values(by="act", ascending=False)[:args.num_tweets]
        not_active = tweets_df[activations == 0].sample(n=args.num_tweets)

        active = active.to_dict(orient='records')
        not_active = not_active.to_dict(orient='records')
        
        for a, b in zip_longest(active, not_active):
            table.add_row(
                display_tweet(a), 
                display_tweet(b)
            )

        console.print(table)


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    
    args = parse_args()
    main(args)