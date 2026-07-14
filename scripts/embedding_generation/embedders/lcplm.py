import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np

from .base import BaseEmbedder


class LCPLMEmbedder(BaseEmbedder):

    def __init__(self, enable_cuda=True, weights='pretrained', checkpoint_path=None):

        super().__init__(enable_cuda, weights, checkpoint_path)

        self.model = AutoModelForMaskedLM.from_pretrained("/gpfs/bwfor/work/ws/fr_jg590-rpiemb_thesis_jan2026/LC-PLM", trust_remote_code=True)

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

        self.max_len = 10000

        self.sequence_type = "protein"

        self.model.eval().to(self.device)

    # original

    def embed_window(self, seq_id, sequence):
        sequence = self.preprocess_sequence(sequence)

        batch_tokens = self.tokenizer(sequence, return_tensors="pt")

        batch_tokens = batch_tokens.to(self.device)

        batch_lens = (batch_tokens["input_ids"] != self.tokenizer.pad_token_id).sum(1).cpu()

        with torch.no_grad():
            results = self.model(**batch_tokens, output_hidden_states=True)

        token_representations = results.hidden_states[-1]

        emb = token_representations[0, 1: batch_lens[0] - 1].cpu().float().numpy()

        del batch_tokens, token_representations, results, batch_lens


        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return emb

    #temporal 
    # def embed_window(self, seq_id, sequence):
    #     sequence = self.preprocess_sequence(sequence)
    #     batch_tokens = self.tokenizer(sequence, return_tensors="pt")
    #     batch_tokens = {k: v.to(self.device) for k, v in batch_tokens.items()}
    #     batch_lens = (batch_tokens["input_ids"] != self.tokenizer.pad_token_id).sum(1).cpu()
    #     with torch.no_grad():
    #         results = self.model(**batch_tokens, output_hidden_states=True)
    #         token_representations = results.hidden_states[-1].cpu().float()
    #     emb = token_representations[0, 1: batch_lens[0] - 1].numpy()
    #     del batch_tokens, token_representations, results, batch_lens
    #     torch.cuda.empty_cache()
    #     return emb