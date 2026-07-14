import esm
import torch
import numpy as np

from .base import BaseEmbedder


class ESM2Embedder(BaseEmbedder):

    def __init__(self, enable_cuda=True, weights="pretrained", checkpoint_path=None):

        super().__init__(enable_cuda)

        self.sequence_type = "protein"

        self.model, self.alphabet = esm.pretrained.esm2_t30_150M_UR50D()


        if weights == "pretrained":

            print("Loading ESM2 pretrained weights...")

        elif weights == "random":

            print("Using random ESM2 initialization.")

            torch.manual_seed(2727)

            for module in self.model.modules():

                if hasattr(module, "reset_parameters"):

                    module.reset_parameters()

        else:

            raise ValueError(
                f"Unknown weights option: {weights}. "
                "Use 'pretrained' or 'random'."
            )

        self.batch_converter = self.alphabet.get_batch_converter()

        self.repr_layer = 30

        self.max_len = 1022

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