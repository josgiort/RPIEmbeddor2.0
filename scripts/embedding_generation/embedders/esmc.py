from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import torch
from .base import BaseEmbedder

class ESMCEmbedder(BaseEmbedder):

    def __init__(self, model_name="esmc_600m", enable_cuda=True, weights='pretrained', checkpoint_path=None):

        super().__init__(enable_cuda, weights, checkpoint_path)

        self.client = ESMC.from_pretrained(model_name).to(self.device).eval()

        self.sequence_type = "protein"

        self.model_name = model_name

        self.max_len = 2046

    def embed_window(self, seq_id, sequence):

        protein = ESMProtein(sequence=sequence)

        with torch.no_grad():

            protein_tensor = self.client.encode(protein)

            output = self.client.logits(
                protein_tensor,
                LogitsConfig(
                    sequence=True,
                    return_embeddings=True
                )
            )
        emb = output.embeddings[0, 1:-1].cpu().float().numpy()

        del output, protein_tensor

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return emb