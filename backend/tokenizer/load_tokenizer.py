from tokenizers import Tokenizer
from pathlib import Path

def load_tokenizer(tokenizer_path: str):
    """
    Load a pre-trained tokenizer from a file.

    Args:
        tokenizer_path: Path to the tokenizer JSON file.

    Returns:
        Tokenizer: Loaded tokenizer.
    """
    if not Path(tokenizer_path).exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return tokenizer