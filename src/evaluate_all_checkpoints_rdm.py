import argparse
import re
from pathlib import Path

import pandas as pd
import torch
from lightning import Trainer

from model import ModelWrapper
from dataloader import get_dataloader


def parse_epoch(path):
    m = re.search(r"epoch=(\d+)", path.name)
    return int(m.group(1)) if m else -1


def parse_val_loss(path):
    m = re.search(r"valid_loss_epoch=([0-9]+(?:\.[0-9]+)?)", path.name)
    return float(m.group(1)) if m else None


def tensor_to_float(x):
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


def main(args):
    checkpoint_dir = Path(args.checkpoint_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    checkpoints = sorted(
        checkpoint_dir.glob("epoch-epoch=*.ckpt"),
        key=parse_epoch
    )

    print(f"Found {len(checkpoints)} epoch checkpoints.")

    test_dataloader = get_dataloader(
        loader_type=args.loader_type,
        dataset_path=args.test_set_path,
        rna_embeddings_path=args.rna_embeddings_path,
        protein_embeddings_path=args.protein_embeddings_path,
        use_individual_files=args.use_individual_files,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
    )

    rows = []

    for ckpt in checkpoints:
        epoch = parse_epoch(ckpt)
        val_loss = parse_val_loss(ckpt)

        print("\n" + "=" * 80)
        print(f"Testing epoch {epoch}: {ckpt.name}")
        print("=" * 80)

        model = ModelWrapper.load_from_checkpoint(
            checkpoint_path=str(ckpt),
            map_location=torch.device("cpu") if args.device == "cpu" else torch.device("cuda"),
            cpr=False,
        )

        model.eval()

        trainer = Trainer(
            accelerator=args.device,
            devices=1,
            logger=False,
            enable_checkpointing=False,
        )

        results = trainer.test(model, dataloaders=test_dataloader, verbose=False)[0]

        row = {
            "epoch": epoch,
            "valid_loss_epoch_from_filename": val_loss,
            "checkpoint": ckpt.name,
        }

        for key, value in results.items():
            row[key] = tensor_to_float(value)

        rows.append(row)

        pd.DataFrame(rows).sort_values("epoch").to_csv(output_csv, index=False)
        print(f"Saved partial results to {output_csv}")

    print("\nDone.")
    print(f"Final CSV: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output_csv", required=True)

    parser.add_argument("--loader_type", default="RPIDataset")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--use_individual_files", action="store_true")

    parser.add_argument("--protein_embeddings_path", required=True)
    parser.add_argument("--rna_embeddings_path", required=True)
    parser.add_argument("--test_set_path", required=True)

    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()
    main(args)