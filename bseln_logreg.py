import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# Load dataframe
df_tr = pd.read_parquet("data/interactions/train_set.parquet")

# Sample a smaller subset of rows without replacement
# subset_size = 2000  # adjust this depending on memory
# df_sub = df.sample(n=subset_size, random_state=42, replace=False).reset_index(drop=True)

# # Load embeddings (only once, full arrays)
# rna_emb_tr = np.load("data/embeddings/rna_embeddings.npy")
# protein_emb_tr = np.load("data/embeddings/protein_embeddings.npy")

# # Extract the embeddings for the sampled subset
# rna_ids_tr = df_tr['Sequence_1_emb_ID'].values
# protein_ids_tr = df_tr['Sequence_2_emb_ID'].values
# X_train = np.hstack([
#     rna_emb_tr[rna_ids_tr].mean(axis=1),  # mean-pool over sequence
#     protein_emb_tr[protein_ids_tr].mean(axis=1)
# ])

# # Labels
# y_train = df_tr['interaction'].astype(int).values


df_te = pd.read_parquet("data/interactions/test_set.parquet")

# # Load embeddings (only once, full arrays)
# rna_emb_te = np.load("data/embeddings/rna_embeddings.npy")
# protein_emb_te = np.load("data/embeddings/protein_embeddings.npy")

# # Extract the embeddings for the sampled subset
# rna_ids_te = df_te['Sequence_1_emb_ID'].values
# protein_ids_te = df_te['Sequence_2_emb_ID'].values
# X_test = np.hstack([
#     rna_emb_te[rna_ids_te].mean(axis=1),  # mean-pool over sequence
#     protein_emb_te[protein_ids_te].mean(axis=1)
# ])



tr_rna_ids = set(df_tr["Sequence_1_emb_ID"].unique())
te_rna_ids = set(df_te["Sequence_1_emb_ID"].unique())
print("RNA overlap:", len(tr_rna_ids & te_rna_ids) / len(te_rna_ids))

tr_prot_ids = set(df_tr["Sequence_2_emb_ID"].unique())
te_prot_ids = set(df_te["Sequence_2_emb_ID"].unique())
print("Protein overlap:", len(tr_prot_ids & te_prot_ids) / len(te_prot_ids))






# # Labels
# y_test = df_te['interaction'].astype(int).values

# # Logistic regression
# clf = LogisticRegression(max_iter=1000)
# clf.fit(X_train, y_train)

# # Evaluate
# y_pred = clf.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("ROC-AUC:", roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))