import os
import tempfile
from src.tokenizer.train_tokenizer import train_tokenizer
from src.tokenizer.load_tokenizer import load_tokenizer

def test_tokenizer_training():
    # Create a temporary corpus
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("To jest testowy korpus. Zawiera kilka zdań po polsku.")
        temp_corpus_path = f.name

    # Create a temporary output path
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_tokenizer_path = f.name

    try:
        # Train a small tokenizer for testing
        train_tokenizer(
            input_path=temp_corpus_path,
            output_path=temp_tokenizer_path,
            vocab_size=1000,
            min_frequency=1,
        )
        assert os.path.exists(temp_tokenizer_path), "Tokenizer file not created."

        # Test loading the tokenizer
        tokenizer = load_tokenizer(temp_tokenizer_path)
        assert tokenizer is not None, "Tokenizer failed to load."

        # Test tokenization
        output = tokenizer.encode("To jest test.")
        assert len(output.ids) > 0, "Tokenization failed."
        print("✅ Tokenizer training and loading test passed.")
    finally:
        # Clean up
        os.unlink(temp_corpus_path)
        os.unlink(temp_tokenizer_path)

if __name__ == "__main__":
    test_tokenizer_training()