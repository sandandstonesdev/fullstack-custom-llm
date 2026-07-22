import argparse
from pathlib import Path
from tokenizer.train_tokenizer import train_tokenizer
from dataset.prepare_dataset import PolishWikipediaDatasetPreparer
from model.inference import generate, load_trained_model  # Import the inference functions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_tokenizer",
        action="store_true",
        help="Train a new tokenizer before preparing the dataset.",
    )
    parser.add_argument(
        "--prepare_dataset",
        action="store_true",
        help="Prepare the dataset (tokenize and shard).",
    )
    parser.add_argument(
        "--train_model",
        action="store_true",
        help="Train the language model.",
    )
    parser.add_argument(
        "--inference",
        type=str,
        default=None,
        help="Run inference on the given prompt.",
    )
    args = parser.parse_args()

    # Paths - Updated to match your actual project structure
    project_root = Path(__file__).parent
    raw_data_path = project_root / "data" / "raw" / "polish_corpus.txt"
    tokenizer_path = project_root / "data" / "processed" / "models" / "tokenizers" / "polish_bpe_32k.json"
    script_dir = project_root / "dataset"  # Updated: no src/ prefix
    model_path = project_root / "data" / "processed" / "models" / "gpt2_polish"

    # Debug: Print the actual paths being used
    print(f"Project root: {project_root}")
    print(f"Tokenizer path: {tokenizer_path}")
    print(f"Tokenizer exists: {tokenizer_path.exists()}")
    print(f"Raw data exists: {raw_data_path.exists()}")

    # Train tokenizer if requested
    if args.train_tokenizer:
        print("Training tokenizer...")
        # Create directories if they don't exist
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if raw data exists
        if not raw_data_path.exists():
            print(f"Error: Raw data file not found at {raw_data_path}")
            return
            
        train_tokenizer(
            input_path=str(raw_data_path),
            output_path=str(tokenizer_path),
            vocab_size=32768,
            min_frequency=2,
        )

    # Prepare dataset if requested
    if args.prepare_dataset:
        print("Preparing dataset...")
        
        # Check if tokenizer exists
        if not tokenizer_path.exists():
            print(f"Error: Tokenizer not found at {tokenizer_path}")
            print("Please run with --train_tokenizer first")
            return
            
        preparer = PolishWikipediaDatasetPreparer(
            script_dir=str(script_dir),
            tokenizer_path=str(tokenizer_path),
            corpus_path=str(raw_data_path),
        )
        preparer.prepare_shards()

    # Train model if requested
    if args.train_model:
        print("Training model...")
        
        # Check if tokenizer exists
        if not tokenizer_path.exists():
            print(f"Error: Tokenizer not found at {tokenizer_path}")
            print("Please run with --train_tokenizer first")
            return
            
        # Create directories if they don't exist
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Import the training script and run it
        from train.train import train_small_model
        train_small_model()

    # Run inference if requested
    if args.inference:
        print(f"Running inference on prompt: '{args.inference}'")
        # Load the model and tokenizer
        model, tokenizer = load_trained_model()
        if model is not None:
            generated_text = generate(
                model, tokenizer, args.inference, max_new_tokens=20,
            )
            print("Generated Text:")
            print(generated_text)
        else:
            print("Failed to load model. Please train a model first.")

if __name__ == "__main__":
    main()