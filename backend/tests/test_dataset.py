import os
import tempfile
import shutil
import numpy as np
from configs import SHARD_NAME
from dataset.prepare_dataset import DatasetPreparer
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
        preparer = DatasetPreparer(
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

def test_stale_shards_and_crash_leftovers_are_cleaned():
    # Same setup as the happy-path test: temp corpus + temp tokenizer.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("To jest testowy korpus. Zawiera kilka zdań po polsku.")
        temp_corpus_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_tokenizer_path = f.name
    train_tokenizer(
        input_path=temp_corpus_path,
        output_path=temp_tokenizer_path,
        vocab_size=1000,
        min_frequency=1,
    )

    temp_output_dir = tempfile.mkdtemp()
    shard_dir = os.path.join(temp_output_dir, "test_shards")
    os.makedirs(shard_dir, exist_ok=True)

    try:
        # A shard from a previous, larger run: must be deleted (it could
        # come from an older tokenizer and would corrupt training).
        stale = os.path.join(shard_dir, f"{SHARD_NAME}_train_099999.npy")
        with open(stale, "wb") as f:
            f.write(b"stale")
        # A half-written shard left by a crashed run: must be cleaned up.
        leftover_tmp = os.path.join(shard_dir, f"{SHARD_NAME}_train_000000.npy.tmp")
        with open(leftover_tmp, "wb") as f:
            f.write(b"half written")

        preparer = DatasetPreparer(
            script_dir=temp_output_dir,
            local_dir="test_shards",
            shard_size=100,
            max_tokens=500,
            tokenizer_path=temp_tokenizer_path,
            corpus_path=temp_corpus_path,
        )
        preparer.prepare_shards()

        assert not os.path.exists(stale), "stale shard from a previous run survived"
        assert not os.path.exists(leftover_tmp), "crash leftover .tmp survived"
        # New shards exist as .npy and no .tmp files remain after the swap.
        npy_files = [f for f in os.listdir(shard_dir) if f.endswith(".npy")]
        tmp_files = [f for f in os.listdir(shard_dir) if f.endswith(".npy.tmp")]
        assert npy_files, "no new shards written"
        assert not tmp_files, f"unswapped .tmp files left behind: {tmp_files}"
        assert np.load(os.path.join(shard_dir, npy_files[0])).dtype == np.uint16
        print("Stale cleanup / atomic swap test passed.")
    finally:
        os.unlink(temp_corpus_path)
        os.unlink(temp_tokenizer_path)
        shutil.rmtree(temp_output_dir)


if __name__ == "__main__":
    test_dataset_preparation()
    test_stale_shards_and_crash_leftovers_are_cleaned()