import math
import os
import random
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from configs import RAW_DATA_PATH, SHARD_NAME, TOKENIZER_PATH
from tokenizer.train_tokenizer import EOT_TOKEN


class DatasetPreparer:
    def __init__(
        self,
        script_dir,
        local_dir="",
        shard_size=int(5e4),
        max_tokens=int(1e7),
        val_ratio=0.1,
        shuffle_seed=42,
        tokenizer_path=None,
        corpus_path=None,
    ):
        self.script_dir = script_dir
        self.local_dir = local_dir
        self.shard_size = shard_size
        self.max_tokens = max_tokens
        self.val_ratio = val_ratio
        self.shuffle_seed = shuffle_seed
        # Shards are written to <script_dir>/<local_dir>; cli.py passes
        # data/processed/shards as script_dir so everything lands there.
        self.DATA_CACHE_DIR = os.path.join(script_dir, local_dir)
        os.makedirs(self.DATA_CACHE_DIR, exist_ok=True)

        # Use provided tokenizer path or default
        if tokenizer_path:
            self.tok = Tokenizer.from_file(tokenizer_path)
        else:
            self.tok = Tokenizer.from_file(str(TOKENIZER_PATH))
        # EOT id as actually emitted by the tokenizer (TemplateProcessing
        # appends it on every encode).
        empty_ids = self.tok.encode("").ids
        self.eot = empty_ids[-1] if empty_ids else self.tok.token_to_id(EOT_TOKEN)
        if self.eot is None:
            self.eot = 0  # [PAD] fallback

        # Corpus contract (see corpus/text.py): one sample per line, one
        # document per sample. tokenize() appends the EOT boundary to
        # each line, so every sample becomes an EOT-delimited document.
        with open(corpus_path or str(RAW_DATA_PATH), "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        self.articles = [{"text": line} for line in lines]

    def tokenize(self, doc):
        tokens = self.tok.encode(doc["text"]).ids
        # The template appends EOT on encode; add it only if missing.
        if not tokens or tokens[-1] != self.eot:
            tokens = tokens + [self.eot]
        tokens_np = np.array(tokens)
        assert (tokens_np >= 0).all() and (tokens_np < 2**16).all()
        return tokens_np.astype(np.uint16)

    def prepare_shards(self):
        if not self.articles:
            raise ValueError("Corpus is empty: no non-blank lines found")

        # Shuffle documents before sharding: the corpus arrives grouped
        # by book, and an unshuffled split would put whole books in val.
        random.Random(self.shuffle_seed).shuffle(self.articles)

        # One document per line: each tokenized doc ends with EOT, so
        # concatenating yields EOT-delimited documents.
        all_tokens = np.concatenate([self.tokenize(doc) for doc in self.articles])

        # Limit tokens to target size
        if len(all_tokens) > self.max_tokens:
            all_tokens = all_tokens[: self.max_tokens]

        # ceil, not // + 1: an exact multiple of shard_size would otherwise
        # produce a trailing EMPTY shard, which would crash the dataloader
        # mid-training when the shard rotation reaches it.
        num_shards = math.ceil(len(all_tokens) / self.shard_size)
        # Held-out val split: the first val_ratio of shards. A fixed
        # one-shard val would swamp small corpora.
        num_val_shards = max(1, round(num_shards * self.val_ratio))

        # Remove leftover .tmp files from a run that crashed mid-write
        # (new shards below are written as .tmp first, so they must not
        # be cleaned up here - only leftovers from older runs).
        cache_dir = Path(self.DATA_CACHE_DIR)
        for leftover in cache_dir.glob(f"{SHARD_NAME}_*.npy.tmp"):
            os.remove(leftover)

        # Write every new shard to <name>.npy.tmp FIRST and only then swap
        # it into place. Until the swap, a crash leaves the OLD dataset
        # fully intact - never an empty directory (which a
        # delete-before-write pass would cause) and never a half-written
        # file in use (the .tmp suffix keeps it out of the dataloader's
        # shard filter).
        new_files = []
        new_filenames = set()
        for i in range(num_shards):
            start = i * self.shard_size
            end = (i + 1) * self.shard_size
            shard = all_tokens[start:end]
            split = "val" if i < num_val_shards else "train"
            filename = f"{SHARD_NAME}_{split}_{i:06d}.npy"
            filepath = os.path.join(self.DATA_CACHE_DIR, filename)
            tmp_filepath = f"{filepath}.tmp"
            # np.save appends ".npy" to names that do not end with it,
            # so write through a file handle to keep the exact .tmp name.
            with open(tmp_filepath, "wb") as tmp_file:
                np.save(tmp_file, shard)
            new_files.append((tmp_filepath, filepath))
            new_filenames.add(filename)

        # Swap new shards into place. os.replace is atomic AND overwrites
        # an old shard of the same name, so there is no separate
        # delete-before-swap pass: each file is either fully old or fully
        # new at any moment.
        for tmp_filepath, filepath in new_files:
            os.replace(tmp_filepath, filepath)

        # Drop shards from previous runs that the new dataset did NOT
        # overwrite: they may come from an older tokenizer and would
        # silently corrupt training if mixed in. After the swap, when
        # the new data is already safely in place.
        for stale in cache_dir.glob(f"{SHARD_NAME}_*.npy"):
            if stale.name not in new_filenames:
                os.remove(stale)

        print(f"Corpus shards saved to {self.DATA_CACHE_DIR}")
        print(f"Total tokens: {len(all_tokens)}")
        print(f"Documents: {len(self.articles)}")
        print(f"Shard size: {self.shard_size}")
        print(
            f"Number of shards: {num_shards} (val: {num_val_shards}, train: {num_shards - num_val_shards})"
        )


if __name__ == "__main__":
    from configs import SHARDS_DIR

    preparer = DatasetPreparer(script_dir=str(SHARDS_DIR))
    preparer.prepare_shards()
