# test_all.py
import sys
import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from lightning import Trainer
from typing import Optional, Dict, List
src_dir = Path(__file__).parent
sys.path.append(str(src_dir))

from model import RNAProteinInterAct, ModelWrapper
from dataloader import get_dataloader

PROTEIN_FMS = ["esm", "esmc", "lcplm", "vesm"]
RNA_FMS     = ["lamar", "rna_fm", "rnaelectra"]
SEEDS       = [2727, 5924, 8137]

PROTEIN_DIMS = {"esm": 640, "esmc": 1152, "lcplm": 1536, "vesm": 640}
RNA_DIMS     = {"lamar": 768, "rna_fm": 640, "rnaelectra": 512}

PROTEIN_DISPLAY = {"esm": "ESM-2", "esmc": "ESM-C", "lcplm": "LC-PLM", "vesm": "V-ESM"}
RNA_DISPLAY     = {"lamar": "LAMAR", "rna_fm": "RNA-FM", "rnaelectra": "RNAElectra"}

METRICS_OF_INTEREST = [
    "test_BinaryAUROC",
    "test_BinaryAUPR",
    "test_BinaryF1Score",
    "test_BinaryAccuracy",
    "test_BinaryPrecision",
    "test_BinaryRecall",
]


def find_best_checkpoint(ckpt_dir: Path) -> Path:
    best = sorted(ckpt_dir.glob("best-*.ckpt"))
    if best:
        return best[0]
    last = sorted(ckpt_dir.glob("last-*.ckpt"))
    if last:
        return last[0]
    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")


def test_single_seed(protein_fm, rna_fm, seed, args) -> Optional[Dict]:
    pair_seed_name = f"{protein_fm}_x_{rna_fm}_seed{seed}"

    ckpt_dir = Path(args.checkpoints_base) / pair_seed_name

    try:
        ckpt_path = find_best_checkpoint(ckpt_dir)
    except FileNotFoundError as e:
        print(f"[SKIP] {pair_seed_name}: {e}")
        return None

    print(f"\n{'='*60}")
    print(f"Testing: {pair_seed_name}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"{'='*60}")

    # rpi_model = RNAProteinInterAct(
    #     batch_first=True,
    #     protein_embed_dim=PROTEIN_DIMS[protein_fm],
    #     rna_embed_dim=RNA_DIMS[rna_fm],
    #     d_model=args.d_model,
    #     num_encoder_layers=args.num_encoder_layers,
    #     nhead=args.n_head,
    #     dim_feedforward=args.dim_feedforward,
    #     key_padding_mask=args.key_padding_mask,
    #     norm_first=True,
    #     dropout=args.dropout,
    # )

    lightning_module = ModelWrapper.load_from_checkpoint(
        checkpoint_path=str(ckpt_path),
        map_location=torch.device('cpu') if args.device == 'cpu' else torch.device('cuda'),
        # model=rpi_model,
        # lr_init=0.001,
        # weight_decay=0.1,
        # t_max=90,
        # warmup_steps=1000,
        # seed=seed,
        cpr=False,
    )
    lightning_module.eval()

    protein_embeddings_path = Path(args.protein_embeddings_path) / protein_fm
    rna_embeddings_path     = Path(args.rna_embeddings_path)     / rna_fm

    test_dataloader = get_dataloader(
        loader_type="RPIDataset",
        dataset_path=args.test_set_path,
        rna_embeddings_path=str(rna_embeddings_path),
        protein_embeddings_path=str(protein_embeddings_path),
        use_individual_files=True,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
    )

    trainer = Trainer(
        accelerator=args.device,
        devices=1,
        logger=False,
        enable_progress_bar=True,
    )
    trainer.test(lightning_module, dataloaders=test_dataloader)

    metrics = {k: float(v) for k, v in trainer.logged_metrics.items()}
    metrics["protein_fm"] = protein_fm
    metrics["rna_fm"]     = rna_fm
    metrics["seed"]       = seed
    metrics["pair"]       = f"{protein_fm}_x_{rna_fm}"
    metrics["checkpoint"] = str(ckpt_path.name)
    return metrics


def aggregate_seeds(seed_results: List[Dict]) -> Dict:
    """Given a list of per-seed metric dicts, return mean and std for each metric."""
    agg = {
        "protein_fm": seed_results[0]["protein_fm"],
        "rna_fm":     seed_results[0]["rna_fm"],
        "pair":       seed_results[0]["pair"],
        "n_seeds":    len(seed_results),
    }
    for metric in METRICS_OF_INTEREST:
        vals = [r[metric] for r in seed_results if metric in r]
        if vals:
            agg[f"{metric}_mean"] = float(np.mean(vals))
            agg[f"{metric}_std"]  = float(np.std(vals))
    return agg


def build_latex_table(df_agg: pd.DataFrame) -> str:
    rna_order     = ["lamar", "rna_fm", "rnaelectra"]
    protein_order = ["esm", "esmc", "lcplm", "vesm"]

    col_headers = " & ".join(
        [f"\\textbf{{{RNA_DISPLAY[r]}}}" for r in rna_order]
    )

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Test performance across FM embedding pairs (mean $\pm$ std over 3 seeds). "
                 r"Each cell reports AUROC / AUPR / F1.}")
    lines.append(r"\label{tab:fm_comparison}")
    lines.append(r"\begin{tabular}{l" + "c" * len(rna_order) + "}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Protein FM} & " + col_headers + r" \\")
    lines.append(r"\midrule")

    for prot in protein_order:
        row_vals = []
        for rna in rna_order:
            sub = df_agg[(df_agg["protein_fm"] == prot) & (df_agg["rna_fm"] == rna)]
            if sub.empty:
                row_vals.append("--")
            else:
                r = sub.iloc[0]
                def fmt(metric):
                    m = r.get(f"{metric}_mean", float("nan"))
                    s = r.get(f"{metric}_std",  float("nan"))
                    return f"{m:.3f}$\\pm${s:.3f}"
                row_vals.append(
                    f"{fmt('test_BinaryAUROC')} / "
                    f"{fmt('test_BinaryAUPR')} / "
                    f"{fmt('test_BinaryF1Score')}"
                )
        lines.append(f"{PROTEIN_DISPLAY[prot]} & " + " & ".join(row_vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main(args):
    all_seed_results = []  # one row per (pair, seed)
    agg_results      = []  # one row per pair (mean±std)

    for protein_fm in PROTEIN_FMS:
        for rna_fm in RNA_FMS:
            seed_rows = []
            for seed in SEEDS:
                row = test_single_seed(protein_fm, rna_fm, seed, args)
                if row is not None:
                    seed_rows.append(row)
                    all_seed_results.append(row)

            if seed_rows:
                agg_results.append(aggregate_seeds(seed_rows))
            else:
                print(f"[WARN] No valid seeds found for {protein_fm}_x_{rna_fm} — skipping.")

    if not agg_results:
        print("No results collected — check checkpoint paths.")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save per-seed raw results
    df_raw = pd.DataFrame(all_seed_results)
    df_raw.to_csv(out_dir / "fm_comparison_per_seed.csv", index=False)
    print(f"\nPer-seed results saved to {out_dir / 'fm_comparison_per_seed.csv'}")

    # Save aggregated results
    df_agg = pd.DataFrame(agg_results)
    df_agg.to_csv(out_dir / "fm_comparison_aggregated.csv", index=False)
    print(f"Aggregated results saved to {out_dir / 'fm_comparison_aggregated.csv'}")

    # Print summary
    display_cols = ["pair", "n_seeds"] + [
        f"{m}_mean" for m in METRICS_OF_INTEREST if f"{m}_mean" in df_agg.columns
    ]
    print("\n" + df_agg[display_cols].to_string(index=False))

    # Save LaTeX table
    latex = build_latex_table(df_agg)
    latex_path = out_dir / "fm_comparison_table.tex"
    latex_path.write_text(latex)
    print(f"\nLaTeX table saved to {latex_path}")
    print("\n" + latex)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_base", required=True,
                        help="Base dir containing subfolders like esm_x_lamar_seed2727/")
    parser.add_argument("--protein_embeddings_path",
                        default="data/embeddings/eCLIP_2/eCLIP2_ALLrbps_rnalength151")
    parser.add_argument("--rna_embeddings_path",
                        default="data/embeddings/eCLIP_2/eCLIP2_ALLrbps_rnalength151")
    parser.add_argument("--test_set_path",
                        default="data/interactions/eCLIP2/eCLIP2_ALLrbps_fixlen151_test.parquet")
    # parser.add_argument("--d_model",            type=int,   default=256)
    # parser.add_argument("--num_encoder_layers", type=int,   default=1)
    # parser.add_argument("--n_head",             type=int,   default=2)
    # parser.add_argument("--dim_feedforward",    type=int,   default=20)
    # parser.add_argument("--dropout",            type=float, default=0.3)
    # parser.add_argument("--key_padding_mask",   action="store_true", default=False)
    parser.add_argument("--batch_size",         type=int,   default=64)
    parser.add_argument("--num_workers",        type=int,   default=8)
    parser.add_argument("--device",             default="cuda")
    parser.add_argument("--output_dir",         default="16Juni_results_fmcomparison_protsDistributed")
    args = parser.parse_args()
    main(args)