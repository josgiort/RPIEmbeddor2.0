import random
import os

import torch

import pandas as pd
import numpy as np

from time import time
from torch.utils.data import DataLoader, Dataset, random_split
from typing import Optional


# class RPIDataset(Dataset):
#     """
#     Loads Pandas Dataframe and reads embeddings from storage when needed.
#     Recommended for using on cluster. Be aware that its usage needs a lot of 
#     system memory (RAM) since all embeddings have to be preloaded into memory.
#     """
#     def __init__(self,
#                  rna_embeddings: np.array,
#                  protein_embeddings: np.array,
#                  dataset_path: str,
#                  ):
#         self.dataset = pd.read_parquet(dataset_path, engine='pyarrow')
#         self.dataset = self.dataset.assign(row_number=range(len(self.dataset)))

#         self.protein_embeddings = protein_embeddings
#         self.rna_embeddings = rna_embeddings
#         self.length = self.dataset.shape[0]

#     @staticmethod
#     def pre_load_rna_embeddings(
#             rna_embeddings_path: str
#     ):
#         print("Loading RNA embeddings...")
#         start = time()
#         rna_embeddings = np.load(rna_embeddings_path)
#         print(f"Loaded RNA embeddings into RAM in {round(time() - start, 2)} seconds")
#         return rna_embeddings

#     @staticmethod
#     def pre_load_protein_embeddings(
#             protein_embeddings_path: str
#     ):
#         print("Loading Protein embeddings...")
#         start = time()
#         protein_embeddings = np.load(protein_embeddings_path)
#         print(f"Loaded Protein embeddings into RAM in {round(time() - start, 2)} seconds")
#         return protein_embeddings

#     @staticmethod
#     def pre_load_embeddings(
#             rna_embeddings_path: str,
#             protein_embeddings_path: str, ):
#         return (RPIDataset.pre_load_rna_embeddings(rna_embeddings_path),
#                 RPIDataset.pre_load_protein_embeddings(protein_embeddings_path))

#     def __len__(self) -> int:
#         return self.length

#     def __getitem__(self, index):
#         # import pudb; pudb.set_trace()
#         # Get entry information?????? ERROR???
#         row = self.dataset[self.dataset['row_number'] == index][
#             ['Sequence_1_emb_ID', 'Sequence_2_emb_ID', 'interaction', 'row_number']]
#         assert row.shape[0] == 1

#         seq_1_emb_ID, seq_2_emb_ID, interaction, row_number = row.values.tolist()[0]

#         interaction = float(interaction)

#         # Get already padded embeddings
#         padded_seq_1_embed = self.rna_embeddings[seq_1_emb_ID]
#         padded_seq_2_embed = self.protein_embeddings[seq_2_emb_ID]

#         return padded_seq_1_embed, padded_seq_2_embed, interaction, row_number


# class RPIDatasetRNARand(RPIDataset):
#     """
#     Data Loader Class for ablation experiment.
#     Replaces the RNA input with a random tensor.
#     However, keeps the padded zero values and generates random values within 
#     the range (min, max) of the original tensor.
#     """
#     def __getitem__(self, index):
#         padded_seq_1_embed, padded_seq_2_embed, interacts, rna_inter_id = super().__getitem__(index)
        
#         # Replace sequence by random
#         non_zero_indices = padded_seq_1_embed.nonzero()
#         random_values = (padded_seq_1_embed.min() - padded_seq_1_embed.max()) * torch.rand(
#             non_zero_indices[0].shape[0]) + padded_seq_1_embed.max()
#         padded_random_seq_1 = padded_seq_1_embed.copy()
#         padded_random_seq_1[non_zero_indices] = random_values

#         return padded_random_seq_1, padded_seq_2_embed, interacts, rna_inter_id


# class RPIDatasetProteinRand(RPIDataset):
#     """
#     Data Loader Class for ablation experiment.
#     Replaces the protein input with a random tensor.
#     However, keeps the padded zero values and generates random values within 
#     the range (min, max) of the original tensor.
#     """
#     def __getitem__(self, index):
#         padded_seq_1_embed, padded_seq_2_embed, interacts, rna_inter_id = super().__getitem__(index)
        
#         # Replace sequence by random
#         non_zero_indices = padded_seq_2_embed.nonzero()
#         random_values = (padded_seq_2_embed.min() - padded_seq_2_embed.max()) * torch.rand(
#             non_zero_indices[0].shape[0]) + padded_seq_2_embed.max()
#         padded_random_seq_2 = padded_seq_2_embed.copy()
#         padded_random_seq_2[non_zero_indices] = random_values

#         return padded_seq_1_embed, padded_random_seq_2, interacts, rna_inter_id


# def get_dataloader(
#         loader_type: str,
#         dataset_path: str,
#         rna_embeddings_path: str,
#         protein_embeddings_path: str,
#         seed: Optional[int] = None,
#         shuffle: bool = True,
#         **kwargs
#     ):
#     """
#     Loads interaction dataset from .parquet file and returns two DataLoader objects.
#     When using for training, provide split_set_size and seed to enable random split
#     between training and validation sets.
#     When using for testing, skip the optional arguments.

#     Args:
#     - loader_type (str): Type of dataloader to be used. Can be one of 
#     ['RPIDatasetProteinRand', 'RPIDatasetRNARand', 'PandasInMemory'].
#     - dataset_path (str): Path to the dataset file.
#     - rna_embeddings_path (str): Path to the RNA embeddings file.
#     - protein_embeddings_path (str): Path to the protein embeddings file.
#     - seed (int): Seed for reproducibility of the split.
#     - shuffle (bool): Whether to shuffle the dataset.

#     Returns:
#     dataset_dataloader (DataLoader): DataLoader object for the dataset.
#     """
#     assert loader_type in ['RPIDatasetProteinRand', 'RPIDatasetRNARand', 'RPIDataset',
#                            ], 'Invalid loader_type specified.'
#     if seed:
#         set_seed(seed)
    
#     rna_embeddings, protein_embeddings = RPIDataset.pre_load_embeddings(
#         rna_embeddings_path,
#         protein_embeddings_path
#     )

#     if loader_type == 'RPIDatasetProteinRand':
#         dataset = RPIDatasetProteinRand(
#             rna_embeddings,
#             protein_embeddings,
#             dataset_path,
#         )
#     elif loader_type == 'RPIDatasetRNARand':
#         dataset = RPIDatasetRNARand(
#             rna_embeddings,
#             protein_embeddings,
#             dataset_path,
#         )
#     elif loader_type == 'RPIDataset':
#         dataset = RPIDataset(
#             rna_embeddings,
#             protein_embeddings,
#             dataset_path,
#         )
#     return DataLoader(dataset, shuffle=shuffle, **kwargs)
  

# def set_seed(seed):
#     """
#     Set the seed for reproducibility in random operations.

#     Args:
#     seed (int): The seed value to be set for all relevant libraries.
#     """
#     random.seed(seed)  # Python's built-in random module
#     np.random.seed(seed)  # Numpy library
#     os.environ['PYTHONHASHSEED'] = str(seed)  # Environment variable

#     # For PyTorch 
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False






# dataloader.py - VERSIÓN SIMPLIFICADA

import random
import os
import torch
import pandas as pd
import numpy as np
from time import time
from torch.utils.data import DataLoader, Dataset
from typing import Optional


class RPIDatasetOnTheFly(Dataset):
    """
    Loads embeddings on-the-fly from individual files.
    Zero RAM overhead - only loads what's needed per batch.
    """
    def __init__(self,
                 rna_embeddings_dir: str,
                 protein_embeddings_dir: str,
                 dataset_path: str):
        
        self.dataset = pd.read_parquet(dataset_path, engine='pyarrow')
        self.dataset = self.dataset.assign(row_number=range(len(self.dataset)))
        
        self.rna_embeddings_dir = rna_embeddings_dir
        self.protein_embeddings_dir = protein_embeddings_dir
        
        self.length = self.dataset.shape[0]
        
        print(f"Dataset size: {self.length}")
        print(f"RNA embeddings dir: {rna_embeddings_dir}")
        print(f"Protein embeddings dir: {protein_embeddings_dir}")
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, index):
        # Get entry information
        row = self.dataset[self.dataset['row_number'] == index][
            ['Sequence_1_emb_ID', 'Sequence_2_emb_ID', 'interaction', 'row_number']]
        assert row.shape[0] == 1
        
        seq_1_emb_ID, seq_2_emb_ID, interaction, row_number = row.values.tolist()[0]
        interaction = float(interaction)
        
        # ✅ Load RNA embedding from individual file
        rna_emb_path = os.path.join(self.rna_embeddings_dir, f"{seq_1_emb_ID}.npy")
        rna_emb = np.load(rna_emb_path).astype('float32')
        
        # ✅ Load Protein embedding from individual file
        prot_emb_path = os.path.join(self.protein_embeddings_dir, f"{seq_2_emb_ID}.npy")
        prot_emb = np.load(prot_emb_path).astype('float32')
        
        # Disabled since a data collator now does the padding per batch, as is more memory efficient
        # Pad to 1024 (necessary for batching)
        # max_len = 1024
        
        # RNA padding
        # rna_dim = rna_emb.shape[1] if len(rna_emb.shape) > 1 else 768
        # padded_rna_emb = np.zeros((max_len, rna_dim), dtype='float32')
        # padded_rna_emb[:min(rna_emb.shape[0], max_len), :] = rna_emb[:max_len, :]
        
        # Protein padding
        # prot_dim = prot_emb.shape[1] if len(prot_emb.shape) > 1 else 640
        # padded_prot_emb = np.zeros((max_len, prot_dim), dtype='float32')
        # padded_prot_emb[:min(prot_emb.shape[0], max_len), :] = prot_emb[:max_len, :]
        
        return rna_emb, prot_emb, interaction, row_number
        # return torch.tensor(padded_rna_emb), torch.tensor(padded_prot_emb), interaction, row_number


class RPIDatasetOnTheFlyRNARand(RPIDatasetOnTheFly):
    """On-the-fly loading with random RNA"""
    def __getitem__(self, index):
        padded_seq_1_embed, padded_seq_2_embed, interaction, row_number = super().__getitem__(index)
        
        # Replace RNA with random
        non_zero_mask = padded_seq_1_embed.sum(axis=1) != 0
        if non_zero_mask.any():
            random_values = np.random.uniform(
                padded_seq_1_embed.min(),
                padded_seq_1_embed.max(),
                padded_seq_1_embed.shape
            ).astype('float32')
            padded_seq_1_embed = np.where(non_zero_mask[:, None], random_values, 0)
        
        return padded_seq_1_embed, padded_seq_2_embed, interaction, row_number


class RPIDatasetOnTheFlyProteinRand(RPIDatasetOnTheFly):
    """On-the-fly loading with random Protein"""
    def __getitem__(self, index):
        padded_seq_1_embed, padded_seq_2_embed, interaction, row_number = super().__getitem__(index)
        
        # Replace Protein with random
        non_zero_mask = padded_seq_2_embed.sum(axis=1) != 0
        if non_zero_mask.any():
            random_values = np.random.uniform(
                padded_seq_2_embed.min(),
                padded_seq_2_embed.max(),
                padded_seq_2_embed.shape
            ).astype('float32')
            padded_seq_2_embed = np.where(non_zero_mask[:, None], random_values, 0)
        
        return padded_seq_1_embed, padded_seq_2_embed, interaction, row_number


def get_dataloader(
        loader_type: str,
        dataset_path: str,
        rna_embeddings_path: str,
        protein_embeddings_path: str,
        seed: Optional[int] = None,
        shuffle: bool = True,
        use_individual_files: bool = True,  # Cambiado a True por defecto
        **kwargs
    ):
    """
    Loads interaction dataset from .parquet file and returns DataLoader.
    
    Args:
    - loader_type (str): Type of dataloader ('RPIDataset', 'RPIDatasetRNARand', 'RPIDatasetProteinRand')
    - use_individual_files (bool): MUST be True (concatenated loading not implemented)
    - rna_embeddings_path (str): Directory containing individual RNA .npy files
    - protein_embeddings_path (str): Directory containing individual Protein .npy files
    """

    def collate_fn(batch):
        rna_embs, prot_embs, interactions, row_numbers = zip(*batch)
        
        max_rna_len  = max(e.shape[0] for e in rna_embs)
        max_prot_len = max(e.shape[0] for e in prot_embs)
        
        rna_dim  = rna_embs[0].shape[1]
        prot_dim = prot_embs[0].shape[1]
        
        padded_rna  = np.zeros((len(batch), max_rna_len,  rna_dim),  dtype='float32')
        padded_prot = np.zeros((len(batch), max_prot_len, prot_dim), dtype='float32')
        
        # True = padding position (to be masked), False = real position
        rna_mask  = np.ones((len(batch), max_rna_len),  dtype=bool)
        prot_mask = np.ones((len(batch), max_prot_len), dtype=bool)
        
        for i, (rna, prot) in enumerate(zip(rna_embs, prot_embs)):
            padded_rna[i,  :rna.shape[0],  :] = rna
            padded_prot[i, :prot.shape[0], :] = prot
            rna_mask[i,  :rna.shape[0]]  = False  # real positions
            prot_mask[i, :prot.shape[0]] = False  # real positions
        
        return (
            torch.tensor(padded_rna),
            torch.tensor(padded_prot),
            torch.tensor(interactions, dtype=torch.float32),
            torch.tensor(row_numbers),
            torch.tensor(rna_mask),
            torch.tensor(prot_mask),
        )


    assert loader_type in ['RPIDatasetProteinRand', 'RPIDatasetRNARand', 'RPIDataset'], \
        'Invalid loader_type specified.'
    
    assert use_individual_files, "Only individual files loading is currently supported"
    
    if seed:
        set_seed(seed)
    
    # Load from individual files (on-the-fly)
    if loader_type == 'RPIDataset':
        dataset = RPIDatasetOnTheFly(
            rna_embeddings_path,  # Directory
            protein_embeddings_path,  # Directory
            dataset_path
        )
    elif loader_type == 'RPIDatasetRNARand':
        dataset = RPIDatasetOnTheFlyRNARand(
            rna_embeddings_path,
            protein_embeddings_path,
            dataset_path
        )
    elif loader_type == 'RPIDatasetProteinRand':
        dataset = RPIDatasetOnTheFlyProteinRand(
            rna_embeddings_path,
            protein_embeddings_path,
            dataset_path
        )
    
    return DataLoader(dataset, shuffle=shuffle, collate_fn=collate_fn, **kwargs)
    # return DataLoader(dataset, shuffle=shuffle, **kwargs)


def set_seed(seed):
    """
    Set the seed for reproducibility in random operations.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False