import os

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

from configs import TOKENIZER_VOCAB_SIZE

# Special tokens in id order: [PAD]=0, [UNK]=1, <|eot|>=2.
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "<|eot|>"]
EOT_TOKEN = "<|eot|>"


def train_tokenizer(input_path, output_path, vocab_size=None, min_frequency=2):
    """Train a Polish BPE tokenizer; <|eot|> terminates every encoded document."""
    if vocab_size is None:
        vocab_size = TOKENIZER_VOCAB_SIZE

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Corpus file not found at {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        corpus_text = f.read()
    print(f"Training tokenizer on {input_path} ({len(corpus_text):,} chars, vocab {vocab_size})")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.train_from_iterator([corpus_text], trainer=trainer)

    # Append the EOT boundary on every encode. The id is looked up AFTER
    # training, so it always matches SPECIAL_TOKENS instead of a magic number.
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"$A {EOT_TOKEN}",
        pair=f"$A {EOT_TOKEN} $B:1",
        special_tokens=[(EOT_TOKEN, tokenizer.token_to_id(EOT_TOKEN))],
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tokenizer.save(output_path)

    encoded = tokenizer.encode("Polska to piękny kraj z bogatą historią i kulturą.")
    print(f"Tokenizer saved to {output_path} (vocab {tokenizer.get_vocab_size()}, "
          f"{EOT_TOKEN} id {tokenizer.token_to_id(EOT_TOKEN)}, "
          f"demo: {len(encoded.ids)} tokens)")


if __name__ == "__main__":
    from configs import RAW_DATA_PATH, TOKENIZER_PATH
    train_tokenizer(str(RAW_DATA_PATH), str(TOKENIZER_PATH))
