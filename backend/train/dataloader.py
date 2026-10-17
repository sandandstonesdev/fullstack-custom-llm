"""
A shard-backed token loader for small local training (data/processed/shards).
"""

import os
import numpy as np
import torch

from configs import SHARD_NAME, SHARDS_DIR


class SmallLLMDataLoader(torch.utils.data.Dataset):
    def __init__(
        self, B, T, process_rank=0, num_processes=1, split="train", max_shards=None
    ):
        """
        Initialize the data loader.

        Args:
            B (int): Batch size (sequences per batch).
            T (int): Sequence length (tokens per sequence).
            split (str): Dataset split ('train' or 'val').
            max_shards (int, optional): Use only the first N shards of the
                split - fast smoke tests without regenerating shards.
        """
        super().__init__()
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}, "Split must be 'train' or 'val'"

        # Shards live in data/processed/shards (see backend/configs).
        data_root = str(SHARDS_DIR)

        # Get the shard filenames
        # Filter to the naming convention written by prepare_dataset:
        # {SHARD_NAME}_{split}_{index:06d}.npy. Match prefix + .npy
        # suffix explicitly: a loose substring check would also pick up
        # unrelated files, e.g. the *.npy.tmp shards prepare_dataset
        # writes while swapping a new dataset into place.
        shard_prefix = f"{SHARD_NAME}_{split}_"
        shard_filenames = sorted(
            filename
            for filename in os.listdir(data_root)
            if filename.startswith(shard_prefix) and filename.endswith(".npy")
        )

        # If shards for the requested split are missing, that is a hard
        # error: silently training (or validating) on the other split
        # would corrupt the train/val separation and early stopping.
        assert len(shard_filenames) > 0, (
            f"No shards found for split {split} in {data_root}. "
            "Run: python -m cli --prepare_dataset"
        )

        self.shard_filepaths = [
            os.path.join(data_root, filename) for filename in shard_filenames
        ]
        if max_shards is not None:
            if max_shards <= 0:
                raise ValueError(f"max_shards must be positive, got {max_shards}")
            self.shard_filepaths = self.shard_filepaths[:max_shards]
        # Token counts per shard, filled lazily by __len__ (avoids re-reading
        # every shard from disk on each call).
        self._shard_token_counts = None

        if process_rank == 0:
            print(f"Found {len(self.shard_filepaths)} shards for split {split}")

        self.reset()

    def reset(self):
        """Reset the loader to the beginning of the first shard."""
        self.curr_shard = 0
        self.tokens = self.load_tokens(self.shard_filepaths[self.curr_shard])
        self.curr_pos = self.B * self.T * self.process_rank

    def load_tokens(self, filepath):
        """Load one shard as a long tensor."""
        tokens = torch.tensor(np.load(filepath).astype(np.int32), dtype=torch.long)
        return tokens

    def __len__(self):
        """
        Return the number of full batches per epoch.

        Counts per shard, matching __getitem__'s skip logic: a shard
        holds (n - 1) // (B*T) full batches; shards too small for a
        single batch contribute none.
        """
        if self._shard_token_counts is None:
            self._shard_token_counts = [
                len(self.load_tokens(path)) for path in self.shard_filepaths
            ]
        total_batches = sum(
            (n - 1) // (self.B * self.T)
            for n in self._shard_token_counts
            if n > self.B * self.T
        )
        return total_batches // self.num_processes

    def __getitem__(self, idx):
        """
        Fetch the next batch (idx is unused; the loader manages its own
        position and rotates shards on its own).

        Returns:
            tuple: (x_batch, y_batch) - inputs and shifted targets.
        """
        B, T = self.B, self.T

        # If the current position cannot yield a full batch (end of shard,
        # or a shard too small to hold a single batch, e.g. an empty
        # leftover shard), move on to the next shard until one can.
        # Only the tail tokens of each shard are dropped this way, never
        # mid-shard data.
        if self.curr_pos + (B * T + 1) > len(self.tokens):
            for _ in range(len(self.shard_filepaths)):
                self.curr_shard = (self.curr_shard + 1) % len(self.shard_filepaths)
                self.tokens = self.load_tokens(self.shard_filepaths[self.curr_shard])
                self.curr_pos = self.B * self.T * self.process_rank
                if self.curr_pos + (B * T + 1) <= len(self.tokens):
                    break
            else:
                raise RuntimeError(
                    f"No shard in {self.shard_filepaths} holds enough tokens "
                    f"for one B={B}, T={T} batch"
                )

        # Extract batch
        batch = self.tokens[self.curr_pos : self.curr_pos + B * T + 1]
        x_batch = batch[:-1].view(B, T)
        y_batch = batch[1:].view(B, T)

        # Update position
        self.curr_pos += B * T * self.num_processes

        return x_batch, y_batch
