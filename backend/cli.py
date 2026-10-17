import argparse

from configs import (
    MODEL_PATH,
    RAW_DATA_PATH,
    SHARDS_DIR,
    TOKENIZER_PATH,
    TOKENIZER_VOCAB_SIZE,
)


def _require_tokenizer() -> bool:
    """Dataset prep and training both need a trained tokenizer first."""
    if not TOKENIZER_PATH.exists():
        print(f"Error: Tokenizer not found at {TOKENIZER_PATH}")
        print("Please run with --train_tokenizer first")
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build_corpus_lektury",
        action="store_true",
        help="Build the corpus from Wolne Lektury (HF: ArielUW/wolnelektury). "
        "Default selection: the curated mythological-literary collection "
        "(Greek/Roman antiquity, mythographic studies, Baltic legends, "
        "Egypt).",
    )
    parser.add_argument(
        "--corpus_samples",
        type=int,
        default=None,
        help="With --build_corpus_lektury: optional cap on the number of corpus "
        "samples (default: no cap - the whole selection is used).",
    )
    parser.add_argument(
        "--lektury_books",
        type=str,
        default=None,
        help="With --build_corpus_lektury: comma-separated substring filters for "
        "books, e.g. 'faraon,homer-iliada'. Replaces the default "
        "mythology collection.",
    )
    parser.add_argument(
        "--lektury_all",
        action="store_true",
        help="With --build_corpus_lektury: use all ~2400 books from the dataset "
        "instead of the curated mythology collection (combine with "
        "--corpus_samples to cap the size).",
    )
    parser.add_argument(
        "--lektury_list",
        action="store_true",
        help="List the available Wolne Lektury book filenames and exit.",
    )
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
        "--batch_size",
        type=int,
        default=None,
        help="With --train_model: override the training batch size (default: 8).",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="With --train_model: override the max number of epochs (default: 5).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="With --train_model: epochs without val improvement before early stop "
        "(default: 2). Raise it for longer runs, e.g. 3 with --max_epochs 12.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="With --train_model: override the learning rate (default: 5e-4).",
    )
    parser.add_argument(
        "--max_shards",
        type=int,
        default=None,
        help="With --train_model: train/validate on only the first N shards of "
        "each split - fast smoke test (default: all shards).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="With --train_model: checkpoint destination; with --inference: checkpoint "
        "to load (default: configs.MODEL_PATH / MODEL_PATH env). Point smoke runs "
        "elsewhere to keep the real checkpoint intact.",
    )
    parser.add_argument(
        "--inference",
        type=str,
        default=None,
        help="Run inference on the given prompt.",
    )
    args = parser.parse_args()

    print(f"Tokenizer path: {TOKENIZER_PATH}")
    print(f"Tokenizer exists: {TOKENIZER_PATH.exists()}")
    print(f"Raw data exists: {RAW_DATA_PATH.exists()}")

    # Build corpus from Wolne Lektury
    if args.lektury_list:
        from corpus.build_wolnelektury import list_available_books

        for book in list_available_books():
            print(book)
        return
    if args.build_corpus_lektury:
        from corpus.build_wolnelektury import MYTHOLOGY_BOOKS, build_corpus_from_lektury

        if args.lektury_books and args.lektury_all:
            print("Error: --lektury_books and --lektury_all are mutually exclusive")
            return
        if args.lektury_books:
            books = [b.strip() for b in args.lektury_books.split(",") if b.strip()]
        elif args.lektury_all:
            books = None  # all books in dataset order
        else:
            books = list(MYTHOLOGY_BOOKS)  # curated mythology collection
        RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        n = build_corpus_from_lektury(
            output_path=str(RAW_DATA_PATH),
            target_samples=args.corpus_samples,
            books=books,
            overwrite=True,  # explicit at the CLI boundary; library default is guarded
        )
        print(f"Corpus built: {n} samples → {RAW_DATA_PATH}")

    # Train tokenizer if requested
    if args.train_tokenizer:
        from tokenizer.train_tokenizer import train_tokenizer

        print("Training tokenizer...")
        if not RAW_DATA_PATH.exists():
            print(f"Error: Raw data file not found at {RAW_DATA_PATH}")
            return

        TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
        train_tokenizer(
            input_path=str(RAW_DATA_PATH),
            output_path=str(TOKENIZER_PATH),
            vocab_size=TOKENIZER_VOCAB_SIZE,
            min_frequency=2,
        )

    # Prepare dataset if requested
    if args.prepare_dataset:
        from dataset.prepare_dataset import DatasetPreparer

        print("Preparing dataset...")

        if not _require_tokenizer():
            return

        SHARDS_DIR.mkdir(parents=True, exist_ok=True)
        preparer = DatasetPreparer(
            script_dir=str(SHARDS_DIR),
            local_dir="",
            tokenizer_path=str(TOKENIZER_PATH),
            corpus_path=str(RAW_DATA_PATH),
        )
        preparer.prepare_shards()

    # Train model if requested
    if args.train_model:
        print("Training model...")

        if not _require_tokenizer():
            return

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        from train.train import TrainingConfig, train_small_model

        # Only pass explicitly provided overrides; the rest stay at
        # TrainingConfig defaults.
        overrides = {
            field: value
            for field, value in {
                "batch_size": args.batch_size,
                "max_epochs": args.max_epochs,
                "patience": args.patience,
                "learning_rate": args.learning_rate,
                "max_shards": args.max_shards,
                "model_path": args.model_path,
            }.items()
            if value is not None
        }
        train_small_model(TrainingConfig(**overrides))

    # Run inference if requested
    if args.inference:
        from model.inference import generate, load_trained_model

        print(f"Running inference on prompt: '{args.inference}'")
        model, tokenizer = load_trained_model(args.model_path)
        if model is not None:
            generated_text = generate(
                model,
                tokenizer,
                args.inference,
                max_new_tokens=20,
            )
            print("Generated Text:")
            print(generated_text)
        else:
            print("Failed to load model. Please train a model first.")


if __name__ == "__main__":
    main()
