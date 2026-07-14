import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

from tqdm import tqdm

import argparse
import random

# =========================
# Config
# =========================

TRAIN_PATH = "/gpfs/bwfor/work/ws/fr_jg590-rpiemb_thesis_jan2026/rpi-main_4/data/interactions/eCLIP2/train_set_withValSet_randomsplit.parquet"

VAL_PATH = "/gpfs/bwfor/work/ws/fr_jg590-rpiemb_thesis_jan2026/rpi-main_4/data/interactions/eCLIP2/validation_set_withValSet_randomsplit.parquet"

TEST_PATH = "/gpfs/bwfor/work/ws/fr_jg590-rpiemb_thesis_jan2026/rpi-main_4/data/interactions/eCLIP2/test_set_withValSet.parquet"

RNA_EMB_DIR = Path(
    "/gpfs/bwfor/work/ws/fr_jg590-rpiemb_thesis_jan2026/rpi-main_4/data/embeddings/eCLIP_2/RNAInterAct/rna_fm"
)

PROT_EMB_DIR = Path(
    "/gpfs/bwfor/work/ws/fr_jg590-rpiemb_thesis_jan2026/rpi-main_4/data/embeddings/eCLIP_2/RNAInterAct/esm"
)

# =========================
# Feature extraction
# =========================


def prepare_features(df, rna_dir, prot_dir):

    features = []
    labels = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Extracting features"
    ):

        rna_id = row["Sequence_1_emb_ID"]
        prot_id = row["Sequence_2_emb_ID"]

        rna_emb = np.load(
            rna_dir / f"{rna_id}.npy"
        ).astype(np.float32)

        prot_emb = np.load(
            prot_dir / f"{prot_id}.npy"
        ).astype(np.float32)

        # Mean pooling over sequence dimension
        rna_feat = rna_emb.mean(axis=0)      # (768,)
        prot_feat = prot_emb.mean(axis=0)    # (1152,)

        # Concatenate representations directly
        feat = np.concatenate([
            rna_feat,
            prot_feat
        ])   # total dimension = 1920

        features.append(feat)
        labels.append(float(row["interaction"]))

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.float32)
    )

# =========================
# Evaluation
# =========================

def evaluate(clf, X, y, split_name):

    probs = clf.predict_proba(X)[:, 1]

    # Match TorchMetrics threshold=0.5
    preds = (probs >= 0.5).astype(int)

    aupr = average_precision_score(y, probs)
    auroc = roc_auc_score(y, probs)

    accuracy = accuracy_score(y, preds)
    f1 = f1_score(y, preds)

    precision = precision_score(y, preds)
    recall = recall_score(y, preds)

    print("\n" + "-" * 60)
    print(f"{split_name} metrics")
    print("-" * 60)

    print(f"BinaryAUPR       : {aupr:.6f}")
    print(f"BinaryAUROC      : {auroc:.6f}")
    print(f"BinaryAccuracy   : {accuracy:.6f}")
    print(f"BinaryF1Score    : {f1:.6f}")
    print(f"BinaryPrecision  : {precision:.6f}")
    print(f"BinaryRecall     : {recall:.6f}")

    return {
        "BinaryAUPR": aupr,
        "BinaryAUROC": auroc,
        "BinaryAccuracy": accuracy,
        "BinaryF1Score": f1,
        "BinaryPrecision": precision,
        "BinaryRecall": recall,
    }


# =========================
# Main
# =========================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2727)
    args = parser.parse_args()

    SEED = args.seed
    np.random.seed(SEED)

    print("Loading datasets...")

    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    print(
        f"\nTrain: {len(train_df)}"
        f"\nVal:   {len(val_df)}"
        f"\nTest:  {len(test_df)}"
    )

    print("\nExtracting train features...")
    X_train, y_train = prepare_features(
        train_df,
        RNA_EMB_DIR,
        PROT_EMB_DIR
    )

    print("\nExtracting validation features...")
    X_val, y_val = prepare_features(
        val_df,
        RNA_EMB_DIR,
        PROT_EMB_DIR
    )

    print("\nExtracting test features...")
    X_test, y_test = prepare_features(
        test_df,
        RNA_EMB_DIR,
        PROT_EMB_DIR
    )

    print("\nScaling features...")

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print("\nTraining Logistic Regression...")

    clf = LogisticRegression(
        max_iter=2000,
        random_state=SEED,
        C=0.1,
        penalty="l2",
        solver="lbfgs",
        n_jobs=-1
    )

    clf.fit(X_train, y_train)

    print("\nTraining completed.")
    print(f"Converged after {clf.n_iter_[0]} optimization iterations.")

    train_metrics = evaluate(
        clf,
        X_train,
        y_train,
        "Train"
    )

    val_metrics = evaluate(
        clf,
        X_val,
        y_val,
        "Validation"
    )

    test_metrics = evaluate(
        clf,
        X_test,
        y_test,
        "Test"
    )

    print("\n" + "=" * 60)
    print("FINAL TEST METRICS")
    print("=" * 60)

    for k, v in test_metrics.items():
        print(f"{k:<18}: {v:.6f}")