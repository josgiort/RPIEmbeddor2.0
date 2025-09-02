import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# Load dataframe
df = pd.read_parquet("data/interactions/test_set.parquet")

# Sample a smaller subset of rows without replacement
subset_size = 2000  # adjust depending on memory
df_sub = df.sample(n=subset_size, random_state=42, replace=False).reset_index(drop=True)

# Load embeddings (only once, full arrays)
rna_emb = np.load("data/embeddings/rna_embeddings.npy")
protein_emb = np.load("data/embeddings/protein_embeddings.npy")

# Extract embeddings for the subset
rna_ids = df_sub['Sequence_1_emb_ID'].values
protein_ids = df_sub['Sequence_2_emb_ID'].values

X = np.hstack([
    rna_emb[rna_ids].mean(axis=1),      # mean-pool RNA embeddings
    protein_emb[protein_ids].mean(axis=1)  # mean-pool protein embeddings
])

# Labels
y = df_sub['interaction'].astype(int).values

# 🔹 Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic regression
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
