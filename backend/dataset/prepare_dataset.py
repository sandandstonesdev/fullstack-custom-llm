import os
import numpy as np
from tokenizers import Tokenizer

class PolishWikipediaDatasetPreparer:
    def __init__(self, script_dir, local_dir="polish_corpus_dataset", 
                 shard_size=int(5e4), max_tokens=int(1e6), tokenizer_path=None, corpus_path=None):
        self.script_dir = script_dir
        self.local_dir = local_dir
        self.shard_size = shard_size
        self.max_tokens = max_tokens
        self.DATA_CACHE_DIR = os.path.join(script_dir, "polish_sample_dataset")
        os.makedirs(self.DATA_CACHE_DIR, exist_ok=True)
        
        # Use provided tokenizer path or default
        if tokenizer_path:
            self.tok = Tokenizer.from_file(tokenizer_path)
        else:
            # Default path
            tokenizer_path = os.path.join(script_dir, "../../data/processed/models/tokenizers/polish_bpe_32k.json")
            self.tok = Tokenizer.from_file(tokenizer_path)
        self.eot = self.tok.token_to_id("<|endoftext|>")
        
        # Use provided corpus path or default
        if corpus_path:
            with open(corpus_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            # Default path
            corpus_path = os.path.join(script_dir, "../../data/raw/polish_corpus.txt")
            with open(corpus_path, "r", encoding="utf-8") as f:
                text = f.read()
        
        # Use full corpus (removed the 25% limit since we have better content)
        self.articles = [{"text": text}]

    def tokenize(self, doc):
        tokens = [self.eot]
        tokens.extend(self.tok.encode(doc['text']).ids)
        tokens_np = np.array(tokens)
        assert (tokens_np >= 0).all() and (tokens_np < 2**16).all()
        return tokens_np.astype(np.uint16)

    def prepare_shards(self):
        all_tokens = self.tokenize(self.articles[0])
        
        # Limit tokens to target size
        if len(all_tokens) > self.max_tokens:
            all_tokens = all_tokens[:self.max_tokens]
        
        num_shards = (len(all_tokens) // self.shard_size) + 1
        
        for i in range(num_shards):
            start = i * self.shard_size
            end = (i + 1) * self.shard_size
            shard = all_tokens[start:end]
            split = 'val' if i == 0 else 'train'
            filepath = os.path.join(self.DATA_CACHE_DIR, f'polish_wiki_{split}_{i:06d}.npy')
            np.save(filepath, shard)
        
        print(f"Expanded corpus shards saved to {self.DATA_CACHE_DIR}")
        print(f"Total tokens: {len(all_tokens)}")
        print(f"Shard size: {self.shard_size}")
        print(f"Number of shards: {num_shards}")
        print("Using new expanded Polish corpus with diverse content!")

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    preparer = PolishWikipediaDatasetPreparer(script_dir)
    preparer.prepare_shards()