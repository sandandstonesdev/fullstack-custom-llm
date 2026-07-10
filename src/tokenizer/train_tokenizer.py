import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

def train_tokenizer(input_path, output_path, vocab_size=32768, min_frequency=2):
    """Train a Polish BPE tokenizer with the specified vocabulary size."""
    
    print("Training Polish BPE Tokenizer")
    print("=" * 50)
    
    # Load the expanded Polish corpus
    corpus_path = input_path
    
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found at {corpus_path}")
    
    print(f"Loading corpus from: {corpus_path}")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus_text = f.read()
    
    print(f"Corpus size: {len(corpus_text):,} characters")
    
    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    
    # Customize pre-tokenizer for Polish
    pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.pre_tokenizer = pre_tokenizer
    
    # Customize decoder
    decoder = decoders.ByteLevel()
    tokenizer.decoder = decoder
    
    # Customize post-processor
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<|endoftext|> $A",
        pair="<|endoftext|> $A <|endoftext|> $B:1",
        special_tokens=[("<|endoftext|>", 1)]
    )
    
    # Train the tokenizer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["[PAD]", "[UNK]", "<|endoftext|>"]
    )
    
    print("Training tokenizer...")
    tokenizer.train_from_iterator([corpus_text], trainer=trainer)
    
    # Save the tokenizer
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(output_path)
    
    print(f"Tokenizer saved to: {output_path}")
    
    # Test the tokenizer
    test_text = "Polska to piękny kraj z bogatą historią i kulturą."
    encoded = tokenizer.encode(test_text)
    
    print(f"\nTokenizer Test:")
    print(f"Input: '{test_text}'")
    print(f"Tokens: {encoded.tokens[:10]}...")  # Show first 10 tokens
    print(f"Token IDs: {encoded.ids[:10]}...")  # Show first 10 token IDs
    print(f"Total tokens: {len(encoded.ids)}")
    
    print(f"\nTokenizer training complete!")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Special tokens: <|endoftext|> = {tokenizer.get_vocab(with_added_tokens=True)['<|endoftext|>']}")


if __name__ == "__main__":
    # Default paths for standalone execution
    input_path = os.path.join('data', 'raw', 'polish_corpus.txt')
    output_path = os.path.join('data', 'processed', 'models', 'tokenizers', 'polish_bpe_32k.json')
    train_tokenizer(input_path, output_path)
