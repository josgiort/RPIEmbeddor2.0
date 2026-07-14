import torch
import torch.nn.functional as F

from embedders.rnaelectra import RNAELECTRAEmbedder

embedder = RNAELECTRAEmbedder(enable_cuda=True)

tokenizer = embedder.tokenizer
model = embedder.model
device = embedder.device


# ────────────────────────────────────────────────────────────────────────────
# Test 1: tokenizer behavior and RNA normalization
# ────────────────────────────────────────────────────────────────────────────

print("\n=== Test 1: tokenizer behavior ===")

raw_seq = "AUGCAUGCU"

# RNAElectra internally expects DNA alphabet
normalized_seq = raw_seq.upper().replace("U", "T")

inputs = tokenizer(normalized_seq, return_tensors="pt")

tokens = inputs["input_ids"][0]

decoded = [tokenizer.decode([t]) for t in tokens]

print(f"Original RNA sequence:   {raw_seq}")
print(f"Normalized sequence:    {normalized_seq}")

print(f"Generated tokens:       {tokens.tolist()}")
print(f"Decoded tokens:         {decoded}")

assert "[UNK]" not in decoded, \
    "Unknown token found after RNA normalization"

print("\nTokenizer test passed ✅")


# ────────────────────────────────────────────────────────────────────────────
# Test 2: verify tokenizer special tokens
# ────────────────────────────────────────────────────────────────────────────

print("\n=== Test 2: special token structure ===")

print(f"First token: {decoded[0]}")

assert decoded[0] == "[CLS]", \
    "Tokenizer does not prepend CLS token correctly"

# RNAElectra does NOT append EOS/SEP
print("No EOS token detected (expected behavior for RNAElectra)")

print("\nSpecial token test passed ✅")


# ────────────────────────────────────────────────────────────────────────────
# Test 3: embedding similarity between biologically related nucleotides
# ────────────────────────────────────────────────────────────────────────────

print("\n=== Test 3: cosine similarity ===")

seq = "AT"

inputs = tokenizer(seq, return_tensors="pt")

inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():

    outputs = model(**inputs)

embs = outputs.last_hidden_state[0]

# skip CLS
emb_A = embs[1]
emb_T = embs[2]

similarity = F.cosine_similarity(
    emb_A.unsqueeze(0),
    emb_T.unsqueeze(0)
)

print(f"Cosine similarity A vs T: {similarity.item():.4f}")

print("\nCosine similarity test completed ✅")


# ────────────────────────────────────────────────────────────────────────────
# Test 4: embedding length consistency
# ────────────────────────────────────────────────────────────────────────────

print("\n=== Test 4: embedding shape consistency ===")

test_sequences = [
    "AUGCAUGCAUGC",
    "GGGAAAUUUCCC",
    "AUAUAUAUAUAU",
]

for seq in test_sequences:

    normalized_seq = seq.upper().replace("U", "T")

    emb = embedder.embed_window("test", normalized_seq)

    print(f"\nOriginal RNA sequence: {seq}")
    print(f"Normalized sequence:  {normalized_seq}")

    print(f"Sequence length:      {len(normalized_seq)}")
    print(f"Embedding shape:      {emb.shape}")

    assert emb.shape[0] == len(normalized_seq), \
        "Embedding length mismatch"

print("\nEmbedding shape tests passed ✅")


# ────────────────────────────────────────────────────────────────────────────
# Test 5: hidden dimension consistency
# ────────────────────────────────────────────────────────────────────────────

print("\n=== Test 5: hidden dimension ===")

seq = "ATGCATGCATGC"

emb = embedder.embed_window("test", seq)

print(f"Hidden dimension: {emb.shape[1]}")

assert emb.shape[1] == 512, \
    "Unexpected hidden dimension"

print("\nHidden dimension test passed ✅")


# ────────────────────────────────────────────────────────────────────────────
# Test 6: invalid character handling
# ────────────────────────────────────────────────────────────────────────────

print("\n=== Test 6: invalid character handling ===")

bad_seq = "ATGCXATGC"

inputs = tokenizer(bad_seq, return_tensors="pt")

decoded = [
    tokenizer.decode([t])
    for t in inputs["input_ids"][0]
]

print(f"Decoded tokens: {decoded}")

assert "[UNK]" in decoded, \
    "Tokenizer failed to flag invalid nucleotide"

print("\nInvalid character handling test passed ✅")


print("\nAll RNAElectra tests passed successfully ✅")



# ────────────────────────────────────────────────────────────────────────────
# Test A: native RNA alphabet support
# ────────────────────────────────────────────────────────────────────────────

print("\n=== Test A: native RNA support ===")

raw_rna_seq = "AUGCU"

inputs = tokenizer(raw_rna_seq, return_tensors="pt")

tokens = inputs["input_ids"][0]

decoded = [
    tokenizer.decode([t])
    for t in tokens
]

print(f"Raw RNA sequence: {raw_rna_seq}")
print(f"Decoded tokens:   {decoded}")

if "[UNK]" in decoded:

    print("\nRESULT:")
    print("Tokenizer does NOT natively support RNA alphabet.")
    print("U nucleotides are mapped to [UNK].")

else:

    print("\nRESULT:")
    print("Tokenizer supports RNA alphabet directly.")





# ────────────────────────────────────────────────────────────────────────────
# Test B: EOS token behavior
# ────────────────────────────────────────────────────────────────────────────

print("\n=== Test B: EOS token behavior ===")

seq = "ATGC"

inputs = tokenizer(seq, return_tensors="pt")

tokens = inputs["input_ids"][0]

decoded = [
    tokenizer.decode([t])
    for t in tokens
]

print(f"Decoded tokens: {decoded}")

assert decoded[0] == "[CLS]", \
    "CLS token missing"

assert "[EOS]" not in decoded, \
    "Unexpected EOS token detected"

assert "[SEP]" not in decoded, \
    "Unexpected SEP token detected"

print("\nEOS/SEP absence verified ✅")