from .base import BaseEmbedder
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, EsmForMaskedLM
import warnings
warnings.simplefilter("ignore", FutureWarning) # ignore future warnings from transformers package
warnings.simplefilter("ignore", UserWarning) # ignore user warnings




class VESMEmbedder(BaseEmbedder):

    def __init__(self, enable_cuda=True, weights='pretrained', checkpoint_path=None): #change

        super().__init__(enable_cuda, weights, checkpoint_path) #change


        self.max_len = 1022

        self.sequence_type = "protein"

        local_dir = '/gpfs/bwfor/work/ws/fr_jg590-rpiemb_thesis_jan2026/vesm' # local directory to store models

        esm_dict = {
        "VESM_3B": 'facebook/esm2_t36_3B_UR50D',
        "VESM_650M": 'facebook/esm2_t33_650M_UR50D',
        "VESM_150M": 'facebook/esm2_t30_150M_UR50D',
        "VESM_35M": 'facebook/esm2_t12_35M_UR50D',
        "VESM3": "esm3_sm_open_v1"
        }

        model_name = "VESM_150M"

        if model_name in esm_dict:
            ckt = esm_dict[model_name]
        else:
            raise ValueError(f"Model not found: {model_name}")
        
        # download weights
        hf_hub_download(repo_id="ntranoslab/vesm", filename=f"{model_name}.pth", local_dir=local_dir)
        
        # load base model
        if model_name == "VESM3":
            from esm.models.esm3 import ESM3
            self.model = ESM3.from_pretrained(ckt).to(torch.float)
            self.tokenizer = self.model.tokenizers.sequence
        else:
            self.model = EsmForMaskedLM.from_pretrained(ckt)
            self.tokenizer = AutoTokenizer.from_pretrained(ckt)
            # load pretrained VESM
            self.model.load_state_dict(torch.load(f'{local_dir}/{model_name}.pth', map_location=self.device), strict=False)
            self.repr_layer = 30

        self.model.eval().to(self.device)

    def embed_window(self, seq_id, sequence):

        sequence = self.preprocess_sequence(sequence)

        # 1. Tokenize using the AutoTokenizer you loaded
        inputs = self.tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len + 2)
        batch_tokens = inputs["input_ids"].to(self.device)
        
        # 2. Get the length (excluding [CLS] and [SEP]/padding)
        # Hugging Face usually adds [CLS] at 0 and [SEP] at the end
        batch_lens = (batch_tokens != self.tokenizer.pad_token_id).sum(1).cpu()

        with torch.no_grad():
            # 3. Tell the model to return hidden states
            results = self.model(batch_tokens, output_hidden_states=True)

            # 4. Extract the representations
            # hidden_states is a tuple: (embeddings, layer1, layer2, ..., layerN)
            # To get the last layer:
            all_layers = results.hidden_states 
            # Layer 30 for ESM-2 150M would be index 30
            token_representations = all_layers[self.repr_layer].cpu()            
    
        emb = token_representations[0, 1: batch_lens[0] - 1].float().numpy()
            
        del batch_tokens, token_representations, results, batch_lens

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return emb