"""
A custom PyTorch Dataset for loading tokenized data from sharded files.
Optimized for small local training on laptops.
Supports distributed training and batching for language modeling tasks.
"""
import os
import numpy as np
import torch


class SmallLLMDataLoader(torch.utils.data.Dataset):
    """
    A lightweight PyTorch Dataset for local training with small corpora.
    Optimized for affordability and fast iteration on laptops.
    """
    
    def __init__(self, B, T, process_rank=0, num_processes=1, split='train', small_corpus=True):
        """
        Initialize the data loader.
        
        Args:
            B (int): Batch size (number of sequences per batch).
            T (int): Sequence length (number of tokens per sequence).
            process_rank (int): Rank of the current process in distributed training.
            num_processes (int): Total number of processes in distributed training.
            split (str): Dataset split ('train' or 'val').
            small_corpus (bool): Whether to use the small corpus directory.
        """
        super().__init__()
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}, "Split must be 'train' or 'val'"
        
        # Use small corpus directory if specified
        script_dir = os.path.dirname(__file__)
        data_root = os.path.join(script_dir, "..", "data", "processed", "polish_sample")
        
        # Get the shard filenames
        shard_filenames = os.listdir(data_root)
        # Filter shard filenames to match the Polish Wikipedia sample format
        shard_filenames = sorted([
            filename for filename in shard_filenames 
            if f'polish_wiki_{split}' in filename
        ])
        
        # If no shards are found for the requested split, try to use the other split
        if len(shard_filenames) == 0:
            other_split = 'val' if split == 'train' else 'train'
            shard_filenames = sorted([
                filename for filename in os.listdir(data_root) 
                if f'polish_wiki_{other_split}' in filename
            ])
            if process_rank == 0:
                print(f'No shards found for split {split}, using split {other_split} instead')
        
        self.shard_filepaths = [os.path.join(data_root, filename) for filename in shard_filenames]
        
        assert len(self.shard_filepaths) > 0, f'No shards found for split {split}'
        
        if process_rank == 0:
            print(f'Found {len(self.shard_filepaths)} shards for split {split}')
        
        self.reset()
    
    def reset(self):
        """Reset the loader to the beginning of the first shard."""
        self.curr_shard = 0
        self.tokens = self.load_tokens(self.shard_filepaths[self.curr_shard])
        self.curr_pos = self.B * self.T * self.process_rank
    
    def load_tokens(self, filepath):
        """
        Load tokens from a shard file.
        
        Args:
            filepath (str): Path to the shard file.
            
        Returns:
            torch.Tensor: Tensor of tokens loaded from the file.
        """
        tokens = torch.tensor(np.load(filepath).astype(np.int32), dtype=torch.long)
        return tokens
    
    def __len__(self):
        """
        Return the total number of batches available.
        
        Returns:
            int: Number of batches.
        """
        total_tokens = sum(len(self.load_tokens(path)) for path in self.shard_filepaths)
        return (total_tokens - 1) // (self.B * self.T * self.num_processes)
    
    def __getitem__(self, idx):
        """
        Fetch a batch of data for training.
        
        Args:
            idx (int): Index of the batch (unused, as the loader manages its own position).
            
        Returns:
            tuple: (x_batch, y_batch), where x_batch is the input and y_batch is the target.
        """
        B, T = self.B, self.T
        
        # Extract batch
        batch = self.tokens[self.curr_pos : self.curr_pos + B * T + 1]
        x_batch = batch[:-1].view(B, T)
        y_batch = batch[1:].view(B, T)
        
        # Update position
        self.curr_pos += B * T * self.num_processes
        
        # Move to the next shard if the current one is exhausted
        if self.curr_pos + (B * T + 1) > len(self.tokens):
            self.curr_shard = (self.curr_shard + 1) % len(self.shard_filepaths)
            self.tokens = self.load_tokens(self.shard_filepaths[self.curr_shard])
            self.curr_pos = self.B * self.T * self.process_rank
        
        return x_batch, y_batch