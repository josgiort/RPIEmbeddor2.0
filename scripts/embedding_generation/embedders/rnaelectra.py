import torch
import numpy as np
import torch
from transformers import AutoModel
from tokenizer import NucEL_Tokenizer
# from transformers import AutoTokenizer

from .base import BaseEmbedder


class RNAELECTRAEmbedder(BaseEmbedder):

    def __init__(self, enable_cuda=True, weights='pretrained', checkpoint_path=None):

        super().__init__(enable_cuda, weights, checkpoint_path)

        self.model = AutoModel.from_pretrained("FreakingPotato/RNAElectra",trust_remote_code=True, use_safetensors=True)

        self.tokenizer = NucEL_Tokenizer.from_pretrained("FreakingPotato/RNAElectra",trust_remote_code=True)

        # self.tokenizer = AutoTokenizer.from_pretrained("FreakingPotato/RNAElectra", trust_remote_code=True, use_fast=False)

        self.max_len = 1022

        self.sequence_type = "rna"

        self.model.eval().to(self.device)

    def preprocess_sequence(self, sequence):

        return sequence.upper().replace("U", "T")


    def embed_window(self, seq_id, sequence):

        sequence = self.preprocess_sequence(sequence)

        batch_tokens = self.tokenizer(sequence, return_tensors="pt", padding=False)

        batch_tokens = {k: v.to(self.device) for k, v in batch_tokens.items()}

        batch_lens = batch_tokens["attention_mask"].sum(1)

        with torch.no_grad():
            results = self.model(**batch_tokens)

        token_representations = results.last_hidden_state

        emb = token_representations[0, 1: batch_lens[0]].cpu().float().numpy()

        del batch_tokens, token_representations, results, batch_lens

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return emb