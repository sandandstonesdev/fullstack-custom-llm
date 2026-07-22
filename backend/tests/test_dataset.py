import os
import tempfile
import shutil
from dataset.prepare_dataset import PolishWikipediaDatasetPreparer
from tokenizer.train_tokenizer import train_tokenizer

def test_dataset_preparation():
    # Create a temporary corpus
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("To jest testowy korpus. Zawiera kilka zdań po polsku.")
        temp_corpus_path = f.name

    # Create a temporary tokenizer
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_tokenizer_path = f.name
    train_tokenizer(
        input_path=temp_corpus_path,
        output_path=temp_tokenizer_path,
        vocab_size=1000,
        min_frequency=1,
    )

    # Create a temporary output directory
    temp_output_dir = tempfile.mkdtemp()

    try:
        # Prepare dataset
        preparer = PolishWikipediaDatasetPreparer(
            script_dir=temp_output_dir,
            local_dir="test_shards",
            shard_size=100,
            max_tokens=500,
            tokenizer_path=temp_tokenizer_path,
            corpus_path=temp_corpus_path,
        )
        preparer.prepare_shards()

        # Check if shards were created
        shard_dir = os.path.join(temp_output_dir, "test_shards")
        assert os.path.exists(shard_dir), "Shard directory not created."
        assert len(os.listdir(shard_dir)) > 0, "No shards created."
        print("Dataset preparation test passed.")
    finally:
        # Clean up
        os.unlink(temp_corpus_path)
        os.unlink(temp_tokenizer_path)
        shutil.rmtree(temp_output_dir)

if __name__ == "__main__":
    
    test_dataset_preparation()