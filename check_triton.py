# import torch, triton
# from transformers import AutoModel, AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
# model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

# x = torch.randint(0, tokenizer.vocab_size, (1, 128)).cuda()
# model = model.cuda()
# with torch.no_grad():
#     y = model(x)





import numpy as np

# emb = np.load("data/embeddings/clip/rna_embeddings_dnabert2.npy")
# print("Shape:", emb.shape)

# # If you have (N, L, D), average across tokens:
# if emb.ndim == 3:
#     emb = emb.mean(axis=1)

# print("NaN count:", np.isnan(emb).sum())
# print("Inf count:", np.isinf(emb).sum())
# print("Mean value:", emb.mean())
# print("Std (overall):", emb.std())
# print("Average embedding norm:", np.linalg.norm(emb, axis=1).mean())





# import numpy as np
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

# # === Load embeddings ===
# emb = np.load("data/embeddings/clip/rna_embeddings_dnabert2.npy")

# # Average across sequence length (axis=1)
# if emb.ndim == 3:
#     emb = emb.mean(axis=1)

# # === PCA ===
# print("Running PCA on embeddings...")
# pca = PCA(n_components=2)
# proj = pca.fit_transform(emb)

# # === Save plot ===
# plt.figure(figsize=(6,6))
# plt.scatter(proj[:,0], proj[:,1], s=5, alpha=0.6)
# plt.title("PCA of DNABERT2 embeddings")
# plt.xlabel("PC1"); plt.ylabel("PC2")

# # Save the figure
# plt.tight_layout()
# plt.savefig("rna_embeddings_dnabert2_pca.png", dpi=300)
# print("✅ PCA plot saved as: rna_embeddings_dnabert2_pca.png")

# # Optional: print variance explained
# print("Explained variance ratios:", pca.explained_variance_ratio_)











# # For synthetic (working) data
# synthetic_rna = np.load("data/embeddings/rna_embeddings_lamar.npy")  # Original embeddings
# print(f"Synthetic RNA embeddings: shape {synthetic_rna.shape}")
# print(f"  Mean: {synthetic_rna.mean():.6f}, Std: {synthetic_rna.std():.6f}")

# # For CLIP (broken) data  
# clip_rna = np.load("data/embeddings/rnaclip8prots/rna_embeddings_lamar.npy")
# print(f"CLIP RNA embeddings: shape {clip_rna.shape}")
# print(f"  Mean: {clip_rna.mean():.6f}, Std: {clip_rna.std():.6f}")

# # Same for protein
# synthetic_prot = np.load("data/embeddings/protein_embeddings.npy")
# clip_prot = np.load("data/embeddings/rnaclip8prots/protein_embeddings.npy")
# print(f"\nSynthetic protein: Mean {synthetic_prot.mean():.6f}")
# print(f"CLIP protein: Mean {clip_prot.mean():.6f}")




#!/usr/bin/env python3
"""
Rebalance CLIP dataset from artificial 50/50 to natural ~12% positive rate
This matches the working synthetic dataset's class distribution
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# === Configuration ===
TARGET_POSITIVE_RATE = 0.12  # Match synthetic dataset (12% positive)
SAMPLES_PER_PROTEIN = 600     # Keep same total per protein

# === Load CLIP data ===
print("Loading CLIP dataset...")
df_clip = pd.read_parquet("data/interactions/rnaclip8prots_train.parquet")
df_clip_val = pd.read_parquet("data/interactions/rnaclip8prots_val.parquet")
df_clip_test = pd.read_parquet("data/interactions/rnaclip8prots_test.parquet")

# Combine all
df_all = pd.concat([df_clip, df_clip_val, df_clip_test], ignore_index=True)
print(f"Total CLIP samples: {len(df_all)}")
print(f"Current positive rate: {(df_all['interaction']==1).sum()/len(df_all)*100:.1f}%")

# === Rebalance per protein ===
print(f"\nRebalancing to {TARGET_POSITIVE_RATE*100:.0f}% positive rate...")

n_pos_target = int(SAMPLES_PER_PROTEIN * TARGET_POSITIVE_RATE)  # ~72 positives
n_neg_target = SAMPLES_PER_PROTEIN - n_pos_target               # ~528 negatives

print(f"Target per protein: {n_pos_target} pos + {n_neg_target} neg = {SAMPLES_PER_PROTEIN} total")

rebalanced_samples = []

for prot_id in df_all['Sequence_2_emb_ID'].unique():
    prot_data = df_all[df_all['Sequence_2_emb_ID'] == prot_id].copy()
    
    # Split by label
    pos_samples = prot_data[prot_data['interaction'] == 1]
    neg_samples = prot_data[prot_data['interaction'] == 0]
    
    print(f"\nProtein {prot_id}:")
    print(f"  Available: {len(pos_samples)} pos, {len(neg_samples)} neg")
    
    # Sample target amounts (or all if not enough)
    n_pos_actual = min(len(pos_samples), n_pos_target)
    n_neg_actual = min(len(neg_samples), n_neg_target)
    
    # If we don't have enough positives, reduce negatives proportionally
    if n_pos_actual < n_pos_target:
        actual_rate = n_pos_actual / SAMPLES_PER_PROTEIN
        n_neg_actual = int(n_pos_actual / actual_rate) - n_pos_actual
        n_neg_actual = min(n_neg_actual, len(neg_samples))
    
    # Sample
    sampled_pos = pos_samples.sample(n=n_pos_actual, random_state=42, replace=False)
    sampled_neg = neg_samples.sample(n=n_neg_actual, random_state=42, replace=False)
    
    prot_subset = pd.concat([sampled_pos, sampled_neg], ignore_index=True)
    rebalanced_samples.append(prot_subset)
    
    actual_pos_rate = len(sampled_pos) / len(prot_subset) * 100
    print(f"  Sampled: {len(sampled_pos)} pos + {len(sampled_neg)} neg = {len(prot_subset)} total ({actual_pos_rate:.1f}% pos)")

# Combine
df_rebalanced = pd.concat(rebalanced_samples, ignore_index=True)

print(f"\n✅ Rebalanced dataset created:")
print(f"  Total samples: {len(df_rebalanced)}")
print(f"  Positive rate: {(df_rebalanced['interaction']==1).sum()/len(df_rebalanced)*100:.1f}%")
print(f"  Positives: {(df_rebalanced['interaction']==1).sum()}")
print(f"  Negatives: {(df_rebalanced['interaction']==0).sum()}")

# === Split using GroupShuffleSplit ===
print("\nSplitting into train/val/test (preserving RNA clusters)...")
groups = df_rebalanced["Sequence_1_cluster"]
train_frac, val_frac, test_frac = 0.8, 0.1, 0.1

# Train vs temp
splitter = GroupShuffleSplit(n_splits=1, train_size=train_frac, random_state=42)
train_idx, temp_idx = next(splitter.split(df_rebalanced, groups=groups))
train_df, temp_df = df_rebalanced.iloc[train_idx], df_rebalanced.iloc[temp_idx]

# Val vs test
splitter = GroupShuffleSplit(n_splits=1, 
                             train_size=val_frac/(val_frac+test_frac),
                             random_state=42)
val_idx, test_idx = next(splitter.split(temp_df, groups=temp_df["Sequence_1_cluster"]))
val_df, test_df = temp_df.iloc[val_idx], temp_df.iloc[test_idx]

# Shuffle
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

# === Save ===
train_df.to_parquet("data/interactions/rnaclip8prots_train_rebalanced.parquet", index=False)
val_df.to_parquet("data/interactions/rnaclip8prots_val_rebalanced.parquet", index=False)
test_df.to_parquet("data/interactions/rnaclip8prots_test_rebalanced.parquet", index=False)

print(f"\n✅ Splits created and saved:")
print(f"  Train: {len(train_df)} samples ({(train_df['interaction']==1).sum()/len(train_df)*100:.1f}% pos)")
print(f"  Val:   {len(val_df)} samples ({(val_df['interaction']==1).sum()/len(val_df)*100:.1f}% pos)")
print(f"  Test:  {len(test_df)} samples ({(test_df['interaction']==1).sum()/len(test_df)*100:.1f}% pos)")

print(f"\n🎯 Now train with:")
print(f"  python src/train.py --seed 824319 --max_epochs 90 \\")
print(f"    --train_set_path data/interactions/rnaclip8prots_train_rebalanced.parquet \\")
print(f"    --rna_embeddings_path data/embeddings/rnaclip8prots/rna_embeddings_lamar.npy \\")
print(f"    --protein_embeddings_path data/embeddings/rnaclip8prots/protein_embeddings.npy")