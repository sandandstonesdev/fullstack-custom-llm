"""
Configuration for the GPT model.
"""
from dataclasses import dataclass


@dataclass
class GPTConfig:
    """
    Configuration for the GPT model.
    
    Attributes:
        context_length (int): Maximum context/sequence length.
        vocab_size (int): Number of tokens in the vocabulary.
        num_layers (int): Number of transformer layers.
        embd_size (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
        bias (bool): Whether to use bias in linear layers.
    """
    context_length: int = 1024    # Max context/sequence length
    vocab_size: int = 50257        # Number of tokens: 50000 BPE merges + 256 bytes tokens + 1 <endoftext> token
    num_layers: int = 12           # Number of transformer layers
    embd_size: int = 768          # Embedding dimension
    num_heads: int = 12           # Number of attention heads
    dropout: float = 0.1           # Dropout probability
    bias: bool = True             # Use bias in linear layers
