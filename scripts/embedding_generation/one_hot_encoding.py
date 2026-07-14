import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


RNA_ALPHABET = {
    "A": 0,
    "C": 1,
    "G": 2,
    "U": 3,
}

PROTEIN_ALPHABET = {
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "X": 19,
    "Y": 20,
}


def one_hot_encode(seq, alphabet, unknown_token=None):
    seq = seq.upper()

    dim = len(alphabet)
    arr = np.zeros((len(seq), dim), dtype=np.float32)

    for i, char in enumerate(seq):
        if char in alphabet:
            arr[i, alphabet[char]] = 1.0
        elif unknown_token is not None and unknown_token in alphabet:
            arr[i, alphabet[unknown_token]] = 1.0
        else:
            raise ValueError(f"Unknown character '{char}' in sequence")

    return arr


def create_onehot_individual_files(
    seq_path,
    emb_dir,
    sequence_type,
):
    seq_path = Path(seq_path)
    emb_dir = Path(emb_dir)
    emb_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(seq_path)

    if sequence_type == "rna":
        seq_col = "Sequence_1"
        id_col = "Sequence_1_emb_ID"
        alphabet = RNA_ALPHABET
        unknown_token = None

    elif sequence_type == "protein":
        seq_col = "Sequence_2"
        id_col = "Sequence_2_emb_ID"
        alphabet = PROTEIN_ALPHABET
        unknown_token = "X"

    else:
        raise ValueError("sequence_type must be 'rna' or 'protein'")

    df = df.sort_values(by=id_col)

    print(f"Creating one-hot files for {sequence_type}")
    print(f"Input:  {seq_path}")
    print(f"Output: {emb_dir}")
    print(f"Rows:   {len(df)}")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        seq_id = row[id_col]
        seq = row[seq_col]

        if not isinstance(seq, str):
            raise ValueError(f"Sequence for {seq_id} is not a string")

        emb = one_hot_encode(
            seq=seq,
            alphabet=alphabet,
            unknown_token=unknown_token,
        )

        np.save(emb_dir / f"{seq_id}.npy", emb)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--working_dir",
        type=str,
        default="/gpfs/bwfor/work/ws/fr_jg590-rpiemb_thesis_jan2026/rpi-main_4",
    )

    parser.add_argument(
        "--sequence_type",
        type=str,
        choices=["rna", "protein"],
        required=True,
    )

    parser.add_argument(
        "--seq_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--emb_dir",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    os.chdir(args.working_dir)

    create_onehot_individual_files(
        seq_path=args.seq_path,
        emb_dir=args.emb_dir,
        sequence_type=args.sequence_type,
    )