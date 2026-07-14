import fm
import torch
import numpy as np

from .base import BaseEmbedder


class RNAFMEmbedder(BaseEmbedder):

    def __init__(self, enable_cuda=True):

        super().__init__(enable_cuda)

        self.model, self.alphabet = fm.pretrained.rna_fm_t12()

        self.batch_converter = self.alphabet.get_batch_converter()

        self.repr_layer = 12

        self.max_len = 1022

        self.sequence_type = "rna"

        self.model.eval().to(self.device)

    def embed_window(self, seq_id, sequence):

        _, _, batch_tokens = self.batch_converter(
            [(seq_id, sequence)]
        )

        batch_lens = (
            batch_tokens != self.alphabet.padding_idx
        ).sum(1)

        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():

            results = self.model(
                batch_tokens,
                repr_layers=[self.repr_layer],
                return_contacts=False
            )

        token_repr = results["representations"][self.repr_layer]

        emb = token_repr[
            0,
            1: batch_lens[0] - 1
        ].cpu().float().numpy()

        del batch_tokens, results

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return emb