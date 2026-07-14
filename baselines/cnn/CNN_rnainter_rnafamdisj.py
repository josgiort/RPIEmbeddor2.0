import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
import argparse
import random
# ============================================================
# CONFIG
# ============================================================

TRAIN_PATH = "/gpfs/bwfor/work/ws/fr_jg590-rpiemb_thesis_jan2026/rpi-main_4/data/interactions/eCLIP2/train_set_withValSet.parquet"

VAL_PATH = "/gpfs/bwfor/work/ws/fr_jg590-rpiemb_thesis_jan2026/rpi-main_4/data/interactions/eCLIP2/validation_set_withValSet.parquet"

TEST_PATH = "/gpfs/bwfor/work/ws/fr_jg590-rpiemb_thesis_jan2026/rpi-main_4/data/interactions/eCLIP2/test_set_withValSet.parquet"

RNA_EMB_DIR = Path(
    "/gpfs/bwfor/work/ws/fr_jg590-rpiemb_thesis_jan2026/rpi-main_4/data/embeddings/eCLIP_2/RNAInterAct/rna_fm"
)

PROT_EMB_DIR = Path(
    "/gpfs/bwfor/work/ws/fr_jg590-rpiemb_thesis_jan2026/rpi-main_4/data/embeddings/eCLIP_2/RNAInterAct/esm"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 1024
NUM_EPOCHS = 90
LR = 1e-3

CHECKPOINT_DIR = None

BEST_CHECKPOINT_PATH = (
    f"{CHECKPOINT_DIR}/best_cnn_baseline.pt"
)

LAST_CHECKPOINT_PATH = (
    f"{CHECKPOINT_DIR}/last_cnn_baseline.pt"
)


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================
# FEATURE EXTRACTION
# ============================================================

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

        rna_feat = rna_emb.mean(axis=0)      # 768
        prot_feat = prot_emb.mean(axis=0)    # 1152

        feat = np.concatenate([
            rna_feat,
            prot_feat
        ])

        features.append(feat)
        labels.append(float(row["interaction"]))

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.float32)
    )

# ============================================================
# DATASET
# ============================================================

class FeatureDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]

# ============================================================
# CNN
# ============================================================

class CNNBaseline(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=7,
            padding=3
        )

        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(
            32,
            64,
            kernel_size=5,
            padding=2
        )

        self.bn2 = nn.BatchNorm1d(64)

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Linear(64, 128)

        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):

        x = x.unsqueeze(1)

        x = torch.relu(
            self.bn1(
                self.conv1(x)
            )
        )

        x = torch.relu(
            self.bn2(
                self.conv2(x)
            )
        )

        x = self.pool(x)

        x = x.squeeze(-1)

        x = torch.relu(
            self.fc1(x)
        )

        x = self.dropout(x)

        x = self.fc2(x)

        return x.squeeze(1)

# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader, criterion):

    model.eval()

    probs_all = []
    labels_all = []

    total_loss = 0

    with torch.no_grad():

        for X, y in loader:

            X = X.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(X)

            loss = criterion(logits, y)

            total_loss += loss.item()

            probs = torch.sigmoid(logits)

            probs_all.extend(
                probs.cpu().numpy()
            )

            labels_all.extend(
                y.cpu().numpy()
            )

    probs_all = np.array(probs_all)
    labels_all = np.array(labels_all)

    preds = (probs_all >= 0.5).astype(int)

    avg_loss = total_loss / len(loader)

    return {

        "loss": avg_loss,

        "BinaryAUPR":
            average_precision_score(
                labels_all,
                probs_all
            ),

        "BinaryAUROC":
            roc_auc_score(
                labels_all,
                probs_all
            ),

        "BinaryAccuracy":
            accuracy_score(
                labels_all,
                preds
            ),

        "BinaryF1Score":
            f1_score(
                labels_all,
                preds
            ),

        "BinaryPrecision":
            precision_score(
                labels_all,
                preds
            ),

        "BinaryRecall":
            recall_score(
                labels_all,
                preds
            )
    }

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2727)
    args = parser.parse_args()

    SEED = args.seed
    set_seed(SEED)


    CHECKPOINT_DIR = f"CNN_onehot_RNAInterActRNAfamDISJ_ESM2xRNAFM_seed{SEED}"

    Path(CHECKPOINT_DIR).mkdir(
        parents=True,
        exist_ok=True
    )

    BEST_CHECKPOINT_PATH = (
        f"{CHECKPOINT_DIR}/best_cnn_baseline.pt"
    )

    LAST_CHECKPOINT_PATH = (
        f"{CHECKPOINT_DIR}/last_cnn_baseline.pt"
    )

    print(f"Using seed: {SEED}")

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

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        FeatureDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g
    )

    val_loader = DataLoader(
        FeatureDataset(X_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        FeatureDataset(X_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = CNNBaseline().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    best_val_loss = float("inf")

    print("\nStarting training...")

    for epoch in range(NUM_EPOCHS):

        model.train()

        running_loss = 0

        for X, y in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}"
        ):

            X = X.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            logits = model(X)

            loss = criterion(
                logits,
                y
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        val_metrics = evaluate(
            model,
            val_loader,
            criterion
        )

        print(
            f"\nEpoch {epoch+1}"
        )

        print(
            f"Train Loss: "
            f"{running_loss/len(train_loader):.4f}"
        )

        print(
            f"Val Loss: "
            f"{val_metrics['loss']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:

            best_val_loss = val_metrics["loss"]

            torch.save(
                model.state_dict(),
                BEST_CHECKPOINT_PATH
            )

            print(
                f"New best checkpoint saved "
                f"(Val Loss = {best_val_loss:.4f})"
            )

    print("\nLoading best model...")

    # ============================================================
    # SAVE LAST CHECKPOINT
    # ============================================================

    print("\nSaving final checkpoint...")

    torch.save(
        model.state_dict(),
        LAST_CHECKPOINT_PATH
    )

    # ============================================================
    # TEST BEST CHECKPOINT
    # ============================================================

    print("\nLoading BEST checkpoint...")

    model.load_state_dict(
        torch.load(
            BEST_CHECKPOINT_PATH
        )
    )

    best_test_metrics = evaluate(
        model,
        test_loader,
        criterion
    )

    print("\n" + "-" * 60)
    print("BEST CHECKPOINT TEST METRICS")
    print("-" * 60)

    for k, v in best_test_metrics.items():

        print(
            f"{k:<18}: {v:.6f}"
        )

    # ============================================================
    # TEST LAST CHECKPOINT
    # ============================================================

    print("\nLoading LAST checkpoint...")

    model.load_state_dict(
        torch.load(
            LAST_CHECKPOINT_PATH
        )
    )

    last_test_metrics = evaluate(
        model,
        test_loader,
        criterion
    )

    print("\n" + "-" * 60)
    print("LAST CHECKPOINT TEST METRICS")
    print("-" * 60)

    for k, v in last_test_metrics.items():

        print(
            f"{k:<18}: {v:.6f}"
        )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(
        f"Best validation loss observed: "
        f"{best_val_loss:.6f}"
    )