import torch
import numpy as np

from LAMAR.modeling_nucESM2 import EsmModel
from transformers import AutoConfig, AutoTokenizer
from safetensors.torch import load_file, load_model

from .base import BaseEmbedder

class LamarEmbedder(BaseEmbedder):

    def __init__(self, enable_cuda=True,  weights='pretrained', checkpoint_path = None):

        super().__init__(enable_cuda, weights, checkpoint_path)
        weights_type = weights

        self.sequence_type = "rna"        

        config = AutoConfig.from_pretrained(
            "/gpfs/bwfor/work/ws/fr_jg590-rpiemb_thesis_jan2026/LAMAR/config/config_150M.json", token_dropout=False, positional_embedding_type='rotary', 
            hidden_size=768, intermediate_size=3072, num_attention_heads=12, num_hidden_layers=12
        )

        self.tokenizer = AutoTokenizer.from_pretrained("/gpfs/bwfor/work/ws/fr_jg590-rpiemb_thesis_jan2026/LAMAR/tokenizer/single_nucleotide/", model_max_length=config.max_position_embeddings)

        config.vocab_size = len(self.tokenizer)
        config.pad_token_id = self.tokenizer.pad_token_id
        config.mask_token_id = self.tokenizer.mask_token_id

        if weights_type == "random":
            torch.manual_seed(2727)
        self.model = EsmModel(config)

        self.max_len = config.max_position_embeddings - 2

        if weights_type == 'pretrained':
            print("Loading LAMAR pretrained weights...")

            # Load pretrained weights from safetensors
            weights = load_file('/gpfs/bwfor/work/ws/fr_jg590-rpiemb_thesis_jan2026/LAMAR/model.safetensors')

            # Clean keys (same logic as before)
            weights_dict = {}
            for k, v in weights.items():
                new_k = k.replace('esm.', '') if k.startswith('esm.') else k
                weights_dict[new_k] = v

            # Load into model
            self.model.load_state_dict(weights_dict, strict=False)

        elif weights_type == 'finetuned':
            # /gpfs/bwfor/work/ws/fr_jg590-rpiemb_thesis_jan2026/rpi-main_4/finetuning/foundationmodel_lamar_gridsearch/runs_refined/lr1e-06_wd0.03_layers1_wu0.03_bs8/last_model/pytorch_model.bin
            print("Loading LAMAR finetuned weights...")
            # Load finetuned checkpoint
            fine_tunned_checkpoint = torch.load(checkpoint_path, map_location=self.device)
            # Extract state dict (PyTorch Lightning saves it under 'state_dict' key)
            if 'state_dict' in fine_tunned_checkpoint:
                state_dict = fine_tunned_checkpoint['state_dict']
            else:
                state_dict = fine_tunned_checkpoint

            # CORRECTED: Extract only encoder weights
            encoder_weights = {}
            for k, v in state_dict.items():
                # Skip classification head weights
                if 'classifier' in k:
                    continue
                
                # Clean up key names
                # From: 'esm.encoder.layer.10.attention.self.query.weight'
                # To:   'encoder.layer.10.attention.self.query.weight'
                new_k = k.replace('esm.', '')  # Remove 'esm.' prefix if present
                encoder_weights[new_k] = v

            self.model.load_state_dict(encoder_weights, strict=False)
        elif weights_type == 'random':
            print("Using random weight initialization (no pretrained weights loaded).")
            # Done already since this is the default version

        
        self.model.eval().to(self.device)

    def preprocess_sequence(self, sequence):

        return sequence.upper().replace("U", "T")

    def embed_window(self, seq_id, sequence):

        sequence = self.preprocess_sequence(sequence)

        batch_tokens = self.tokenizer(sequence, return_tensors="pt")

        batch_tokens = {
            k: v.to(self.device)
            for k, v in batch_tokens.items()
        }

        batch_lens = (batch_tokens["attention_mask"] == 1).sum(dim=1)

        with torch.no_grad():

            results = self.model(input_ids=batch_tokens["input_ids"], attention_mask=batch_tokens["attention_mask"])

        token_representations = results.last_hidden_state

        emb = token_representations[0,1: batch_lens - 1].cpu().float().numpy()

        del batch_tokens, results, token_representations, batch_lens

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return emb

    def embed_batch(self, seq_ids: list, sequences: list) -> list:
        """
        Embed a batch of sequences at once.
        Returns list of np.arrays, one per sequence, each shape (L_i, D).
        """
        sequences = [self.preprocess_sequence(s) for s in sequences]
        batch_tokens = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len + 2
        )
        batch_tokens = {k: v.to(self.device) for k, v in batch_tokens.items()}
        batch_lens = (batch_tokens["attention_mask"] == 1).sum(dim=1)

        with torch.no_grad():
            results = self.model(
                input_ids=batch_tokens["input_ids"],
                attention_mask=batch_tokens["attention_mask"]
            )
        token_representations = results.last_hidden_state  # (B, L_max, D)

        embeddings = []
        for i, seq_len in enumerate(batch_lens):
            # strip CLS and EOS tokens
            emb = token_representations[i, 1: seq_len - 1].cpu().float().numpy()
            embeddings.append(emb)

        del batch_tokens, results, token_representations, batch_lens
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return embeddings