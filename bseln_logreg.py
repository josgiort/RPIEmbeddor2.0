import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

# Load dataframe
df = pd.read_parquet("data/interactions/pum2_pairs_train.parquet")

# 🔹 Pick a *very small* subset (e.g. 20 datapoints, balanced if possible)
subset_size = 20
df_sub = df.sample(n=subset_size, random_state=42, replace=False).reset_index(drop=True)
# df_sub = df

# Load embeddings (only once, full arrays)
rna_emb = np.load("data/embeddings/clip/rna_embeddings_rnafm.npy")
protein_emb = np.load("data/embeddings/clip/protein_embeddings.npy")

# Extract embeddings for the subset
rna_ids = df_sub['Sequence_1_emb_ID'].values
protein_ids = df_sub['Sequence_2_emb_ID'].values

X = np.hstack([
    rna_emb[rna_ids].mean(axis=1),         # mean-pool RNA embeddings
    protein_emb[protein_ids].mean(axis=1)  # mean-pool protein embeddings
])

y = df_sub['interaction'].astype(int).values

# 🔹 No split → train and evaluate on the SAME set
clf = LogisticRegression(max_iter=5000)  # more iters for convergence
clf.fit(X, y)

y_pred = clf.predict(X)
y_prob = clf.predict_proba(X)[:, 1]

print("Training Accuracy:", accuracy_score(y, y_pred))
print("Training LogLoss:", log_loss(y, y_prob))




# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.linear_model import SGDClassifier
# from sklearn.metrics import accuracy_score, log_loss

# # Load dataframe
# df = pd.read_parquet("data/interactions/pum2_pairs_train.parquet")

# # 🔹 Pick subset
# subset_size = 20
# df_sub = df.sample(n=subset_size, random_state=42, replace=False).reset_index(drop=True)
# # df_sub = df
# # Load embeddings
# rna_emb = np.load("data/embeddings/clip/rna_embeddings_trust_remote.npy")
# protein_emb = np.load("data/embeddings/clip/protein_embeddings.npy")

# # Extract embeddings for subset
# rna_ids = df_sub['Sequence_1_emb_ID'].values
# protein_ids = df_sub['Sequence_2_emb_ID'].values

# X = np.hstack([
#     rna_emb[rna_ids].mean(axis=1),
#     protein_emb[protein_ids].mean(axis=1)
# ])

# y = df_sub['interaction'].astype(int).values

# # 🔹 SGDClassifier allows partial_fit so we can track losses
# clf = SGDClassifier(loss="log_loss", alpha=1e-5, learning_rate="constant", eta0=0.01, random_state=42, warm_start=True)

# n_epochs = 5000
# losses = []
# accuracies = []

# for epoch in range(n_epochs):
#     clf.partial_fit(X, y, classes=np.array([0, 1]))
#     y_prob = clf.predict_proba(X)[:, 1]
#     y_pred = (y_prob > 0.5).astype(int)
    
#     losses.append(log_loss(y, y_prob))
#     accuracies.append(accuracy_score(y, y_pred))

# # 🔹 Plot training curves
# plt.figure(figsize=(12,5))

# plt.subplot(1,2,1)
# plt.plot(losses, marker="o")
# plt.title("Training LogLoss")
# plt.xlabel("Epoch")
# plt.ylabel("LogLoss")

# plt.subplot(1,2,2)
# plt.plot(accuracies, marker="o")
# plt.title("Training Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")

# plt.tight_layout()
# plt.savefig("training_curves_trust_20_tunned.png")  # 🔹 Save instead of show
# print("✅ Saved training_curves.png")
