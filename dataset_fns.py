"""
This module contains functions and classes that handle and manipulate datasets.
"""

from pathlib import Path

import dill
from datasets import DatasetDict, load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from logger import logger_factory

logger = logger_factory()


def split_dataset(dataset_dict, test_size=0.05, dev_size=0.05, seed=42):
    """
    Split a HuggingFace DatasetDict with only a train split into train, test, and dev splits.

    Args:
        dataset_dict (DatasetDict): Input dataset with only a 'train' split
        test_size (float): Proportion of data for test split (default: 0.05)
        dev_size (float): Proportion of data for dev split (default: 0.05)
        seed (int): Random seed for reproducibility (default: 42)

    Returns:
        DatasetDict: Dataset with train, test, and dev splits
    """
    if not isinstance(dataset_dict, DatasetDict):
        raise ValueError("Input must be a DatasetDict")

    if "train" not in dataset_dict:
        raise ValueError("Input dataset must contain a 'train' split")

    if test_size + dev_size >= 1:
        raise ValueError("Sum of test_size and dev_size must be less than 1")

    # Get the full training dataset
    full_train = dataset_dict["train"]

    # Create splits
    splits = full_train.train_test_split(
        test_size=(test_size + dev_size), shuffle=True, seed=seed
    )

    # Further split the test portion into test and dev
    test_dev_splits = splits["test"].train_test_split(
        test_size=test_size / (test_size + dev_size), shuffle=True, seed=seed
    )

    # Create new DatasetDict with all splits
    return DatasetDict(
        {
            "train": splits["train"],
            "test": test_dev_splits["test"],
            "validation": test_dev_splits["train"],
        }
    )


def batch_tokenization(example) -> str:
    batch = tokenizer.encode_batch(example["text"])
    return {
        "tokens": [b.tokens for b in batch],
        "ids": [b.ids for b in batch],
        "attention_mask": [b.attention_mask for b in batch],
    }


def tokenizer_batch_iterator(batch_size=2048):
    tok_dataset = ds["train"].select_columns("text")
    for batch in tok_dataset.iter(batch_size):
        yield batch["text"]


if __name__ == "__main__":
    logger.info("Executing dataset functions")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en")

    logger.info("Creating and training tokenizer...")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    )
    tokenizer.train_from_iterator(
        tokenizer_batch_iterator(), trainer=trainer, length=len(ds["train"])
    )

    test_string = "This is a good tokenizer for processing words!"
    e = tokenizer.encode(test_string)
    print(e.tokens)
    logger.info("Saving tokenizer...")
    tokenizer.save("models/tokenizer/bpe_tokenizer.json")

    with open("./tokenizer.pkl", "wb") as f:
        dill.dump(tokenizer, f)

    logger.info("Adding extra features to dataset...")
    ds["train"] = ds["train"].map(batch_tokenization, batched=True, num_proc=24)

    logger.info("Creating splits...")
    ds = split_dataset(ds)

    logger.info("Saving data...")
    data_path = Path("./data").absolute()
    ds.save_to_disk(str(data_path))
