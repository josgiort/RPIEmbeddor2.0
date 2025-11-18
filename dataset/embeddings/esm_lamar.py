import argparse
import sys
import os
import psutil
# import fm
import esm

import pandas as pd
import numpy as np

from tqdm import tqdm
from time import time
from statistics import mean
from pathlib import Path

src_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(src_dir))
from utils import divide_dataframe

# New imports

# from LAMAR.modeling_nucESM2 import EsmModel
# from transformers import AutoConfig, AutoTokenizer
# from safetensors.torch import load_file, load_model
import torch

# working original create embedding function
# def create_embeddings(emb_dir, data_path, model_type, enable_cuda, max_task_id, task_id, if_inference):
#     """
#     Create and save embeddings for sequences using a specified model (RNA-FM or ESM-2).

#     Args:
#     - emb_dir (str): Directory to save the embeddings.
#     - data_path (str): Path to the data file (RNA or protein data).
#     - model_type (str): Type of model ('rna_fm' or 'esm2').
#     - enable_cuda (bool): Whether to use CUDA.
#     - max_task_id (int): Maximum task ID for data splitting.
#     - task_id (int): Current task ID.
#     - if_inference (bool): Whether to print additional information.

#     Returns:
#     - None
#     """
#     if not if_inference:
#         print(f"Running with task id {task_id} and max task id {max_task_id}")
#     df = pd.read_parquet(data_path, engine='pyarrow')

#     # Split data across multiple tasks
#     data_batch = divide_dataframe(df, max_task_id, task_id)
#     if len(data_batch) == 0:
#         print("No data to process.")
#         return
    
#     if not if_inference:
#         print(f"Creating embeddings for {len(data_batch)} sequences out of {df.shape[0]}")

#     if model_type == 'lamar':
#         model_max_length = 1026
#         device = torch.device("cuda:0")
#         # device = torch.device("cpu")
#         # instance tokenizer and config
#         tokenizer = AutoTokenizer.from_pretrained("/gpfs/bwfor/work/ws/fr_jg590-fr_jg590-restored/LAMAR/tokenizer/single_nucleotide/", model_max_length=model_max_length)
#         config = AutoConfig.from_pretrained(
#             "/gpfs/bwfor/work/ws/fr_jg590-fr_jg590-restored/LAMAR/config/config_150M.json", vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id,
#             mask_token_id=tokenizer.mask_token_id, token_dropout=False, positional_embedding_type='rotary', 
#             hidden_size=768, intermediate_size=3072, num_attention_heads=12, num_hidden_layers=12
#         )
#         # instance the model and load pretrained weights
#         model = EsmModel(config)
#         weights = load_file('/gpfs/bwfor/work/ws/fr_jg590-fr_jg590-restored/LAMAR/model.safetensors')
#         weights_dict = {}
#         for k, v in weights.items():
#             new_k = k.replace('esm.', '') if 'esm' in k else k
#             weights_dict[new_k] = v
#         model.load_state_dict(weights_dict, strict=False)
#         model = model.to(device)
#         idx = '1'
#     # We keep ESM2 for proteins        
#     elif model_type == 'esm2':
#         model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
#         batch_converter = alphabet.get_batch_converter()
#         idx = '2'
#         repr_layer = 30
#     else:
#         raise ValueError("Invalid model type. Choose 'rna_fm' or 'esm2'.")

#      # Ensure the directory exists
#     if not os.path.exists(emb_dir):
#         os.makedirs(emb_dir)
    
#     model.eval()  # Disables dropout for deterministic results

#     if enable_cuda:
#         model.cuda()        

#     timings = []

#     for _, row in tqdm(enumerate(data_batch), total=len(data_batch), desc="Embedding sequences"):
#         start = time()

#         embedding_id = row[f'Sequence_{idx}_emb_ID']
#         sequence = row[f'Sequence_{idx}'].upper()
        
#         if model_type == 'lamar':
#             # Preprocess sequences so that the U gets replaced by T, since LAMAR
#             # doesnt know RNA but DNA. LATER WE FINETUNE IT BY RETRAINING WITH RNA SAMPLES            
#             sequence = sequence.replace("U", "T")

#             # Compute embeddings
#             with torch.no_grad():
#                 batch_tokens = tokenizer(sequence, return_tensors="pt")
#                 if enable_cuda: 
#                     batch_tokens = {k: v.cuda() for k, v in batch_tokens.items()} 
                
#                 input_ids = batch_tokens['input_ids'].to(device)
#                 attention_mask = batch_tokens['attention_mask'].to(device)
#                 batch_lens = (attention_mask == 1).sum(dim=1) 

#                 results = model(
#                     input_ids=input_ids, 
#                     attention_mask=attention_mask
#                 )
#                 # token_representations = results.last_hidden_state[0, 1 : -1, :].cpu() # shape: [batch_size, batch_lens, hidden_dim]
#                 token_representations = results.last_hidden_state.cpu()
        
#         elif model_type == 'esm2':
#             # Process sequence
#             _, _, batch_tokens = batch_converter([(embedding_id, sequence)])
#             batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

#             if enable_cuda:
#                 batch_tokens = batch_tokens.to(device='cuda')

#             # Extract per-residue representations (on CPU)
#             with torch.no_grad():
#                 results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=True)
#             token_representations = results["representations"][repr_layer].cpu()
        
#         # Generate per-sequence representations
#         # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
#         for i, tokens_len in enumerate(batch_lens):
#             np.save(f"{emb_dir}/{embedding_id}",token_representations[i, 1: tokens_len - 1].float())
#         del batch_tokens, token_representations, results, batch_lens

#         if enable_cuda:
#             torch.cuda.empty_cache()

#         timings.append(time() - start)

#     if not if_inference:
#         print(f"Average time per sequence embedding extraction: {mean(timings):.4f}")


# Create_embeddings function with mean pooled
def create_embeddings(emb_dir, data_path, model_type, enable_cuda, max_task_id, task_id, if_inference):
    """
    Create and save embeddings for sequences using a specified model (RNA-FM or ESM-2).

    Args:
    - emb_dir (str): Directory to save the embeddings.
    - data_path (str): Path to the data file (RNA or protein data).
    - model_type (str): Type of model ('rna_fm' or 'esm2').
    - enable_cuda (bool): Whether to use CUDA.
    - max_task_id (int): Maximum task ID for data splitting.
    - task_id (int): Current task ID.
    - if_inference (bool): Whether to print additional information.

    Returns:
    - None
    """
    if not if_inference:
        print(f"Running with task id {task_id} and max task id {max_task_id}")
    df = pd.read_parquet(data_path, engine='pyarrow')

    # Split data across multiple tasks
    data_batch = divide_dataframe(df, max_task_id, task_id)
    if len(data_batch) == 0:
        print("No data to process.")
        return
    
    if not if_inference:
        print(f"Creating embeddings for {len(data_batch)} sequences out of {df.shape[0]}")

    if model_type == 'lamar':
        model_max_length = 1026
        # device = torch.device("cuda:0")    
        device = torch.device("cpu")
        # instance tokenizer and config
        tokenizer = AutoTokenizer.from_pretrained("/gpfs/bwfor/work/ws/fr_jg590-rpiemb_thesis_nov2025/LAMAR/tokenizer/single_nucleotide/", model_max_length=model_max_length)
        config = AutoConfig.from_pretrained(
            "/gpfs/bwfor/work/ws/fr_jg590-rpiemb_thesis_nov2025/LAMAR/config/config_150M.json", vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id,
            mask_token_id=tokenizer.mask_token_id, token_dropout=False, positional_embedding_type='rotary', 
            hidden_size=768, intermediate_size=3072, num_attention_heads=12, num_hidden_layers=12
        )
        # instance the model and load pretrained weights
        model = EsmModel(config)
        weights = load_file('/gpfs/bwfor/work/ws/fr_jg590-rpiemb_thesis_nov2025/LAMAR/model.safetensors')
        weights_dict = {}
        for k, v in weights.items():
            new_k = k.replace('esm.', '') if 'esm' in k else k
            weights_dict[new_k] = v
        model.load_state_dict(weights_dict, strict=False)
        model = model.to(device)
        idx = '1'
    # We keep ESM2 for proteins        
    elif model_type == 'esm2':
        model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        idx = '2'
        repr_layer = 30
    else:
        raise ValueError("Invalid model type. Choose 'rna_fm' or 'esm2'.")

     # Ensure the directory exists
    if not os.path.exists(emb_dir):
        os.makedirs(emb_dir)
    
    model.eval()  # Disables dropout for deterministic results

    if enable_cuda:
        model.cuda()        

    timings = []

    for _, row in tqdm(enumerate(data_batch), total=len(data_batch), desc="Embedding sequences"):
        start = time()

        embedding_id = row[f'Sequence_{idx}_emb_ID']
        sequence = row[f'Sequence_{idx}'].upper()
        
        if model_type == 'lamar':
            # Preprocess sequences so that the U gets replaced by T, since LAMAR
            # doesnt know RNA but DNA. LATER WE FINETUNE IT BY RETRAINING WITH RNA SAMPLES            
            sequence = sequence.replace("U", "T")

            # Compute embeddings
            with torch.no_grad():
                batch_tokens = tokenizer(sequence, return_tensors="pt")
                if enable_cuda: 
                    batch_tokens = {k: v.cuda() for k, v in batch_tokens.items()} 
                
                input_ids = batch_tokens['input_ids'].to(device)
                attention_mask = batch_tokens['attention_mask'].to(device)
                batch_lens = (attention_mask == 1).sum(dim=1) 

                results = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask
                )
                # token_representations = results.last_hidden_state[0, 1 : -1, :].cpu() # shape: [batch_size, batch_lens, hidden_dim]
                token_representations = results.last_hidden_state.cpu()
        
        elif model_type == 'esm2':
            # Process sequence
            _, _, batch_tokens = batch_converter([(embedding_id, sequence)])
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

            if enable_cuda:
                batch_tokens = batch_tokens.to(device='cuda')

            # Extract per-residue representations (on CPU)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=True)
            token_representations = results["representations"][repr_layer].cpu()
        
        # Generate per-sequence representations
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        for i, tokens_len in enumerate(batch_lens):
            # Commenting original code temporarily to try the mean pooled and bypassing of transformer with a simple FFN approach
            # np.save(f"{emb_dir}/{embedding_id}",token_representations[i, 1: tokens_len - 1].float())
            
            seq_embeddings = token_representations[i, 1: tokens_len - 1, :].float()
            mean_pooled_vector = seq_embeddings.mean(dim=0)
            np.save(f"{emb_dir}/{embedding_id}", mean_pooled_vector.numpy())

        del batch_tokens, token_representations, results, batch_lens

        if enable_cuda:
            torch.cuda.empty_cache()

        timings.append(time() - start)

    if not if_inference:
        print(f"Average time per sequence embedding extraction: {mean(timings):.4f}")




# working original merge embedding function
# def merge_embeddings(emb_dir, model_type):
    """
    Merge embeddings for all sequences into a single array.

    Args:
    - emb_dir (str): Path to the directory containing the embeddings.
    - model_type (str): Type of model ('rna_fm' or 'esm2').

    Returns:
    - None
    """
    # sequence_type, hid_dim = ('rna', 768) if model_type == 'lamar' else ('protein', 640)

    # # Get all .npy files from the directory
    # file_paths = [os.path.join(emb_dir, f) for f in os.listdir(emb_dir) if f.endswith('.npy')]

    # embeddings = []


    # # Pad embeddings to the same length
    # for embedding_path in tqdm(file_paths, total=len(file_paths), desc="Merging embeddings"):
    #     emb = np.load(embedding_path)
    #     padded_emb = np.zeros((1024, hid_dim))
    #     padded_emb[:emb.shape[0], :] = emb
    #     embeddings.append(padded_emb)

    # # Stack embeddings into a single array
    # embeddings = np.stack(embeddings, axis=0)

    # # Memory usage information (optional)
    # print('RAM Used (GB):', psutil.virtual_memory()[3] / 1e9)
    # print(f"Embeddings shape: {embeddings.shape}")

    # # Save the merged embeddings
    # parent_dir = os.path.dirname(emb_dir)
    # merged_embeddings_path = os.path.join(parent_dir, f"{sequence_type}_embeddings.npy")
    # np.save(merged_embeddings_path, embeddings)
    # print(f"Merged embeddings saved to {merged_embeddings_path}")

# Modified merge_emnbedding
def merge_embeddings(emb_dir, model_type):
    """
    Merge embeddings for all sequences into a single array.

    Args:
    - emb_dir (str): Path to the directory containing the embeddings.
    - model_type (str): Type of model ('rna_fm' or 'esm2').

    Returns:
    - None
    """
    sequence_type, hid_dim = ('rna', 768) if model_type == 'lamar' else ('protein', 640)

    # Get all .npy files from the directory
    file_paths = [os.path.join(emb_dir, f) for f in os.listdir(emb_dir) if f.endswith('.npy')]

    embeddings = []


    # Pad embeddings to the same length
    for embedding_path in tqdm(file_paths, total=len(file_paths), desc="Merging embeddings"):
        emb = np.load(embedding_path)
        # padded_emb = np.zeros((1024, hid_dim))
        # padded_emb[:emb.shape[0], :] = emb
        embeddings.append(emb)

    # Stack embeddings into a single array
    embeddings = np.stack(embeddings, axis=0)

    # Memory usage information (optional)
    print('RAM Used (GB):', psutil.virtual_memory()[3] / 1e9)
    print(f"Embeddings shape: {embeddings.shape}")

    # Save the merged embeddings
    parent_dir = os.path.dirname(emb_dir)
    merged_embeddings_path = os.path.join(parent_dir, f"{sequence_type}_embeddings_MeanPooledRnaclip8prots.npy")
    np.save(merged_embeddings_path, embeddings)
    print(f"Merged embeddings saved to {merged_embeddings_path}")






if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--enable_cuda', type=bool, default=False, help='Enable or disable CUDA')
    parser.add_argument('--unique_seq_path', type=str, default="data/annotations/unique_proteins_rnaclip8prots.parquet", help='Path to the unique sequence data')
    parser.add_argument('--max_task_id', type=int, default=1, help='Maximum task ID')
    parser.add_argument('--task_id', type=int, default=1, help='Task ID')
    parser.add_argument('--working_dir', type=str, default='/gpfs/bwfor/work/ws/fr_jg590-rpiemb_thesis_nov2025/rpi-main_4/', help='Working directory path.')
    parser.add_argument('--emb_dir', type=str, default="data/embeddings/MeanPooledRnaclip8prots", help='Directory to save the results')
    parser.add_argument('--model_type', type=str, choices=['lamar', 'esm2'], required=True, help='Type of model to use (lamar or esm2)')
    parser.add_argument('--if_inference', action='store_true', default=False, help='Hides logging info during inference')
                        
    args = parser.parse_args()

    os.chdir(args.working_dir)




    if args.model_type == 'lamar':
        rna_fm_dir = os.path.join(args.emb_dir, "lamar")

        create_embeddings(rna_fm_dir, args.unique_seq_path, args.model_type, args.enable_cuda, args.max_task_id, args.task_id, args.if_inference)
        merge_embeddings(rna_fm_dir, args.model_type)

    elif args.model_type == 'esm2':
        esm_dir = os.path.join(args.emb_dir, "esm")

        create_embeddings(esm_dir, args.unique_seq_path, args.model_type, args.enable_cuda, args.max_task_id, args.task_id, args.if_inference)
        merge_embeddings(esm_dir, args.model_type)

    else:
        raise ValueError("Invalid model type. Choose 'lamar' or 'esm2'.")
