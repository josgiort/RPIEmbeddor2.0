import torch
import torch.nn.functional as F
from embedders.lcplm import LCPLMEmbedder

embedder = LCPLMEmbedder(enable_cuda=True)
tokenizer = embedder.tokenizer
model = embedder.model
device = embedder.device

# ── Test 1: tokenizer handles special amino acids ──────────────────────────
test_seq = "ACDEFGHIKLMNPQRSTVWYU"  # U = selenocysteine
inputs = tokenizer(test_seq, return_tensors="pt")
tokens = inputs["input_ids"][0]
decoded = [tokenizer.decode([t]) for t in tokens]
print(f"Tokens:  {tokens.tolist()}")
print(f"Decoded: {decoded}")
assert "<unk>" not in decoded, "Unknown token found — some amino acids are not in vocabulary!"

# ── Test 2: cosine similarity between similar amino acids ──────────────────
inputs = tokenizer("IL", return_tensors="pt").to(device)
with torch.no_grad():
    out = model(**inputs, output_hidden_states=True)
    embs = out.hidden_states[-1][0]  # last layer, first batch

emb_I = embs[1]  # skip CLS
emb_L = embs[2]
sim = F.cosine_similarity(emb_I.unsqueeze(0), emb_L.unsqueeze(0))
print(f"Cosine similarity I vs L: {sim.item():.4f}")
assert sim.item() > 0.5, "Similarity unexpectedly low — check model output"

# ── Test 3: embedding shape matches sequence length ────────────────────────
test_sequences = [
    "MKTLLLTLLVVTIVCLDLGY",
    "ACDEFGHIKLMNPQRSTVWY",
]
for seq in test_sequences:
    emb = embedder.embed_window("test", seq)
    print(f"Sequence length: {len(seq)}, Embedding shape: {emb.shape}")
    assert emb.shape[0] == len(seq), f"Shape mismatch for: {seq}"
    # assert emb.shape[1] == 640, f"Unexpected hidden dim: {emb.shape[1]}"

print("\nAll tests passed ✅")