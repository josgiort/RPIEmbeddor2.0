import os
import argparse
import numpy as np
import pandas as pd
import torch
from time import time
from statistics import mean
from tqdm import tqdm

from sliding_window import embed_sequence_with_windows

# from embedders.esm2 import ESM2Embedder
# from embedders.rnafm import RNAFMEmbedder
from embedders.lamar import LamarEmbedder
# from embedders.vesm import VESMEmbedder
# from embedders.lcplm import LCPLMEmbedder
# from embedders.esmc import ESMCEmbedder
# from embedders.rnaelectra import RNAELECTRAEmbedder

import sys
from pathlib import Path
src_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(src_dir))
from utils import divide_dataframe


EMBEDDER_REGISTRY = {

    # "esm2": ESM2Embedder,
    # "rna_fm": RNAFMEmbedder,
    "lamar": LamarEmbedder,
    # "vesm": VESMEmbedder,
    # "lcplm": LCPLMEmbedder,
    # "esmc": ESMCEmbedder,
    # "rnaelectra": RNAELECTRAEmbedder,
}

def load_embedder(args):
    if args.model_type not in EMBEDDER_REGISTRY:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    return EMBEDDER_REGISTRY[args.model_type](enable_cuda=args.enable_cuda, weights=args.weights, checkpoint_path=args.checkpoint_path,)

# with batch size implementation
# def create_embeddings(args):
#     embedder = load_embedder(args)

#     if not args.if_inference:
#         print(f"Running with task id {args.task_id} and max task id {args.max_task_id}")

#     df = pd.read_parquet(args.unique_seq_path, engine='pyarrow')
#     data_batch = divide_dataframe(df, args.max_task_id, args.task_id)
#     if len(data_batch) == 0:
#         print("No data to process.")
#         return

#     if not args.if_inference:
#         print(f"Creating embeddings for {len(data_batch)} sequences out of {df.shape[0]}")

#     os.makedirs(args.emb_dir, exist_ok=True)
#     idx = "1" if embedder.sequence_type == "rna" else "2"
#     rows = list(data_batch)
#     timings = []

#     for i in tqdm(range(0, len(rows), args.batch_size)):
#         batch_rows = rows[i: i + args.batch_size]
#         start = time()

#         ids  = [row[f"Sequence_{idx}_emb_ID"] for row in batch_rows]
#         seqs = [row[f"Sequence_{idx}"]         for row in batch_rows]

#         embeddings = embedder.embed_batch(ids, seqs)

#         for eid, emb in zip(ids, embeddings):
#             np.save(f"{args.emb_dir}/{eid}", emb)

#         timings.append((time() - start) / len(batch_rows))

#     print(f"Average embedding time per sequence: {mean(timings):.4f}")

# Original
def create_embeddings(args):

    embedder = load_embedder(args)
    
    if not args.if_inference:
        print(f"Running with task id {args.task_id} and max task id {args.max_task_id}")

    df = pd.read_parquet(args.unique_seq_path, engine='pyarrow')

    # Split data across multiple tasks
    data_batch = divide_dataframe(df, args.max_task_id, args.task_id)

    if len(data_batch) == 0:
        print("No data to process.")
        return
    
    if not args.if_inference:
        print(f"Creating embeddings for {len(data_batch)} sequences out of {df.shape[0]}")

    os.makedirs(args.emb_dir, exist_ok=True)

    idx = "1" if embedder.sequence_type == "rna" else "2"

    timings = []

    for _, row in tqdm(enumerate(data_batch), total=len(data_batch)):
        start = time()
        embedding_id = row[f"Sequence_{idx}_emb_ID"]
        sequence = row[f"Sequence_{idx}"]
        emb = embed_sequence_with_windows(
            embedding_id=embedding_id,
            sequence=sequence,
            embed_window_fn=embedder.embed_window,
            window_size=embedder.max_len
        )
        np.save(
            f"{args.emb_dir}/{embedding_id}",
            emb
        )
        timings.append(time() - start)

    print(
        f"Average embedding time: "
        f"{mean(timings):.4f}"
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--enable_cuda', action='store_true', default=True, help='Enable CUDA')
    parser.add_argument('--unique_seq_path', type=str, default="data/annotations/eCLIP2/unique_rnas_eCLIP2_10RBPs_fixlen151.parquet", help='Path to parquet file containing sequences')
    parser.add_argument('--max_task_id', type=int, default=1, help='Maximum task ID')
    parser.add_argument('--task_id', type=int, default=1, help='Current task ID')
    parser.add_argument('--working_dir', type=str, default="/gpfs/bwfor/work/ws/fr_jg590-rpiemb_thesis_jan2026/rpi-main_4",help='Working directory path')
    parser.add_argument('--emb_dir', type=str, default="data/embeddings/eCLIP_2/eCLIP2_10RBPs_RNAdistr", help='Directory to save embeddings')
    parser.add_argument('--model_type', type=str, choices=['rna_fm', 'esm2', 'lamar', 'vesm', 'lcplm', 'esmc', 'rnaelectra'], required=True, help='Foundation model to use')
    parser.add_argument('--if_inference', action='store_true', default=False, help='Hide logging information')

    parser.add_argument('--weights', type=str, default='pretrained', choices=['pretrained', 'finetuned', 'random'], help='Which weights to use: pretrained, finetuned, or random init')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to finetuned checkpoint (required when --weights finetuned)')

    parser.add_argument('--batch_size', type=int, default=2,
                    help='Number of sequences to process per batch')


    args = parser.parse_args()
    os.chdir(args.working_dir)


    # Validate:
    if args.weights == 'finetuned' and args.checkpoint_path is None:
        raise ValueError("--checkpoint_path is required when --weights finetuned")


    # automatically create subdirectory by model name
    suffix = args.model_type if args.weights == 'pretrained' else f"{args.model_type}_{args.weights}"
    args.emb_dir = os.path.join(args.emb_dir, suffix)

    if not args.if_inference:
        print(f"Model: {args.model_type}")
        print(f"Embeddings directory: {args.emb_dir}")

    create_embeddings(args)