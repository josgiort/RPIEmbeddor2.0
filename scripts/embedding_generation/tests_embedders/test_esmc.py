import torch
import torch.nn.functional as F
from embedders.esmc import ESMCEmbedder
from esm.sdk.api import ESMProtein, LogitsConfig

embedder = ESMCEmbedder(enable_cuda=True)
device = embedder.device

# ── Test 1: raw output shape — does it include BOS/EOS? ───────────────────
# from esm.sdk.api import ESMProtein, LogitsConfig

# test_seq = "ACDEFGHIKLMNPQRSTVWYU"
# protein = ESMProtein(sequence=test_seq)
# with torch.no_grad():
#     protein_tensor = embedder.client.encode(protein)
#     output = embedder.client.logits(
#         protein_tensor,
#         LogitsConfig(sequence=True, return_embeddings=True)
#     )
# print(f"Sequence length:    {len(test_seq)}")
# print(f"Raw embedding shape: {output.embeddings.shape}")
# This tells us directly whether BOS/EOS are included




# ── Test 1: raw output shape — does it include BOS/EOS? ───────────────────
test_seq = "ACDEFGHIKLMNPQRSTVWYU"
protein = ESMProtein(sequence=test_seq)
with torch.no_grad():
    protein_tensor = embedder.client.encode(protein)
    output = embedder.client.logits(
        protein_tensor,
        LogitsConfig(sequence=True, return_embeddings=True)
    )
print(f"Sequence length:     {len(test_seq)}")
print(f"Raw embedding shape: {output.embeddings.shape}")
# If shape[1] == len+2 → BOS/EOS included, need to strip
# If shape[1] == len   → already clean


# ── Test 2: cosine similarity I vs L ──────────────────────────────────────
emb_I = embedder.embed_window("test", "I")
emb_L = embedder.embed_window("test", "L")
emb_I_t = torch.tensor(emb_I)
emb_L_t = torch.tensor(emb_L)
sim = F.cosine_similarity(emb_I_t, emb_L_t, dim=-1).mean()
print(f"Cosine similarity I vs L: {sim.item():.4f}")
assert sim.item() > 0.5, "Similarity unexpectedly low"





# ── Test 3: embed_window strips correctly ─────────────────────────────────
for seq in ["MKTLLLTLLVVTIVCLDLGY", "ACDEFGHIKLMNPQRSTVWY"]:
    emb = embedder.embed_window("test", seq)
    print(f"Sequence length: {len(seq)}, Embedding shape: {emb.shape}")
    assert emb.shape[0] == len(seq), f"Shape mismatch for: {seq}"

print("\nAll tests passed ✅")
