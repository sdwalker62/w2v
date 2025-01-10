"""Main training logic. Does not concern itself with data manipulation."""

from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import TypeAlias

import torch
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from rich import box
from rich.console import Console
from rich.table import Table
from tokenizers import Tokenizer
from torch import nn, optim
from torch.nn import Module
from torch.utils.data import DataLoader

from model import Word2VecModel

dataset: TypeAlias = Dataset | DatasetDict | IterableDataset | IterableDatasetDict


class HyperParams:
    def __init__(
        self,
        n_epochs=1,
        lr=1e-1,
        optimizer="adam",
        loss_fn="ce",
        embed_dim=128,
        is_skipgram=False,
        batch_size=16,
        window_size=2,
    ) -> None:
        self.n_epochs = n_epochs
        self.lr = lr
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.embed_dim = embed_dim
        self.is_skipgram = is_skipgram
        self.batch_size = batch_size
        self.window_size = window_size


class Trainer:
    def __init__(
        self,
        dd: dataset,
        model: Module,
        hparams: HyperParams,
        tokenizer: Tokenizer | None,
        logger: Logger,
        chkpt_dir: Path,
    ) -> None:
        self.model = model
        if tokenizer:
            self.tokenizer = tokenizer
        self.chkpt_iter = 5  # Number of training passes between each checkpoint (includes validation passes)
        self.hyperparams = hparams
        self.dd = dd
        if isinstance(self.dd, DatasetDict) or isinstance(self.dd, IterableDatasetDict):
            self.td = self.dd["train"]
            # self.train_dl = DataLoader(
            #     self.td.with_format("torch"),
            #     num_workers=4,
            #     batch_size=self.hyperparams.batch_size,
            #     collate_fn=self._generate_pairs,
            # )
            print(self.td)
            self.vd = self.dd["validation"]
            self.testd = self.dd["test"]

        if isinstance(dd, IterableDatasetDict) or isinstance(dd, IterableDataset):
            self.streaming = True
        self.log = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.console = Console()
        self.chkpt_dir = chkpt_dir
        self.current_time = datetime.now()
        self.chkpt_n = 0

        self.create_dirs()

    def pre_check(self):
        """Performs pre-training actions to ensure functionality."""
        self.log.info(len(self.td))
        self.log.info(len(self.vd))
        self.log.info(len(self.testd))

    def prepare_data(self, list) -> None:
        """Collate data before the dataloader"""

    def start(self):
        """Training entrypoint."""

    def train_epoch(
        self,
        output_path: str | None = None,
    ) -> tuple[Word2VecModel, dict[str, torch.Tensor]]:
        """
        Train Word2Vec model

        Args:
            texts: List of tokenized texts
            output_path: Optional path to save trained model
            device: Device to train on ('cuda' or 'cpu')

        Returns:
            Trained model and word embeddings dictionary
        """
        # Create dataset
        # dataset = Word2VecDataset(
        #     texts, self.window_size, self.min_count, self.is_skip_gram
        # )

        # dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize model
        vocab_size = self.tokenizer.get_vocab_size()
        self.log.info(f"Vocab size: {vocab_size}")
        model = Word2VecModel(
            vocab_size, self.hyperparams.embed_dim, self.hyperparams.is_skipgram
        ).to(self.device)

        # Initialize optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=self.hyperparams.lr)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        passed_samples = 0
        model.train()
        for epoch in range(self.hyperparams.n_epochs):
            self.log.info(f"Running epoch: {epoch}")
            total_loss = 0

            # The collation function runs after the batching process if we use
            # `torch.data.DataLoader` so we will instead loop over the training
            # data, collate each item, and then batch the results.
            # Remote Data -> Get Row -> Collate -> Batch -> Train
            for example in self.td:
                """Each example is a dictionary with features such as ids, text, etc."""
                batches = self.prepare_batches(example)  # list[tuple[list[int], int]]
                for batch_idx, X in enumerate(batches):
                    X, y = self.split_to_tensors(X)
                    passed_samples += self.hyperparams.batch_size
                    # Forward pass
                    optimizer.zero_grad()
                    output = model(X)
                    loss = criterion(output, y)

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                    if (batch_idx + 1) % 100 == 0:
                        print(
                            f"Epoch {epoch + 1}/{self.hyperparams.n_epochs}, "
                            f"Loss: {total_loss / (batch_idx + 1):.4f}"
                        )

            print(
                f"Epoch {epoch + 1} completed, "
                f"Average Loss: {total_loss / passed_samples:.4f}"
            )

        # Save model if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": dataset.vocab,
                    "embedding_dim": self.embedding_dim,
                    "is_skip_gram": self.is_skip_gram,
                },
                output_path,
            )
            print(f"Model saved to {output_path}")

        # Create word embeddings dictionary
        embeddings = {
            word: model.embedding.weight.data[idx].cpu()
            for word, idx in dataset.vocab.items()
        }

        return model, embeddings

    def validate(self):
        """Completes a single pass over the validation set."""

    def test(self):
        """Completes a single pass over the test set."""

    def create_dirs(self) -> None:
        """Create any necessary directories for the training script"""
        self.log.info(f"Creating checkpoint directory within: {self.chkpt_dir}")
        if self.hyperparams.is_skipgram:
            dir_name = "skipgram--"
        else:
            dir_name = "cbow--"
        dir_name += self.current_time.strftime("%Y-%m-%d-%H-%M-%S")
        specific_chkpt_dir = self.chkpt_dir / Path(dir_name)
        if specific_chkpt_dir.absolute().exists():
            specific_chkpt_dir.absolute().mkdir(exist_ok=False)

    def checkpoint(self):
        """Perform checkpointing duties."""
        # Create unique directory within the checkpoint directory
        assert self.chkpt_dir.exists(), self.log.error(
            f"Cannot find checkpoint directory: {self.chkpt_dir}"
        )
        self.chkpt_n += 1
        self.log.info(f"Performing checkpoint: {self.chkpt_n}")

        self.save_model()

    def save_model(self):
        """Dumps the model to disk."""
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "vocab": dataset.vocab,
                    "embedding_dim": self.embedding_dim,
                    "is_skip_gram": self.is_skip_gram,
                },
                output_path,
            )
            self.log.info(f"Model saved to {output_path}")

    def load_model(
        self, path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> tuple[Word2VecModel, dict[str, int]]:
        """
        Load saved model

        Args:
            path: Path to saved model
            device: Device to load model on

        Returns:
            Loaded model and vocabulary
        """
        checkpoint = torch.load(path, map_location=device)

        model = Word2VecModel(
            len(checkpoint["vocab"]),
            checkpoint["embedding_dim"],
            checkpoint["is_skip_gram"],
        ).to(device)

        model.load_state_dict(checkpoint["model_state_dict"])
        return model, checkpoint["vocab"]

    def prepare_batches(self, row_dict) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Generate input-target pairs for training"""
        pairs = []
        # word_indices = [self.vocab.get(word, 0) for word in text]
        word_indices = list(row_dict["ids"])
        for i in range(len(word_indices)):
            # Generate context window
            context_start = max(0, i - self.hyperparams.window_size)
            context_end = min(len(word_indices), i + self.hyperparams.window_size + 1)
            context = word_indices[context_start:i] + word_indices[i + 1 : context_end]

            if len(context) == 0:
                continue

            if self.hyperparams.is_skipgram:
                # Skip-gram: predict context words from center word
                center = word_indices[i]
                for ctx in context:
                    if ctx != 0:  # Skip padding
                        pairs.append((center, ctx))
            else:
                # CBOW: predict center word from context words
                if word_indices[i] == 0:  # Skip padding
                    continue

                # Pad context to fixed size
                ctx_size = 2 * self.hyperparams.window_size
                ctx_padded = context + [0] * (ctx_size - len(context))
                ctx_padded = ctx_padded[:ctx_size]

                pairs.append((ctx_padded, word_indices[i]))

        # Create batches
        pairs = [
            pairs[i : i + self.hyperparams.batch_size]
            for i in range(0, len(pairs), self.hyperparams.batch_size)
        ]
        return pairs

    def split_to_tensors(
        self,
        data: list[tuple[list[int], int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert a list[tuple[list[int], int]] into two PyTorch tensors.

        Args:
            data: List of tuples where each tuple contains (list[int], int)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: First tensor contains stacked lists,
                                            second tensor contains stacked integers
        """
        # Unzip the tuples into two separate lists
        lists, values = zip(*data)

        # Convert lists to tensor - needs to be padded if different lengths
        max_len = max(len(lst) for lst in lists)
        padded_lists = [lst + [0] * (max_len - len(lst)) for lst in lists]
        lists_tensor = torch.tensor(padded_lists)

        # Convert values to tensor
        values_tensor = torch.tensor(values)

        return lists_tensor.to(self.device), values_tensor.to(self.device)

    def display_config_table(self) -> None:
        """Display a nicely formatted table of the hyperparameters."""
        table = Table(
            title="Hyperparameters",
            box=box.ROUNDED,
            highlight=True,
            show_header=True,
            header_style="dim white",
            title_style="bold white",
            caption="* Default values may vary based on specific use cases",
            caption_style="italic",
        )

        table.add_column("Hyperparameter", style="dim green")
        table.add_column("Value", justify="center", style="dim yellow")
        table.add_column("Description", style="dim")
        table.add_column("Range", justify="center", style="yellow")

        table.add_row(
            "Optimization Parameters",
            style="bold black on light_goldenrod2",
            end_section=True,
        )
        table.add_row(
            "Learning Rate",
            str(self.hyperparams.lr),
            "Step size for gradient descent updates",
            "1e-5 to 1e-1",
        )

        table.add_row(
            "Hidden Units", "256", "Number of neurons in hidden layers", "32 to 1024"
        )
        table.add_row(
            "Optimizer",
            "Adam",
            "Algorithm for updating network weights",
            "SGD/Adam/RMSprop",
        )
        table.add_row(
            "Activation",
            "ReLU",
            "Non-linear function applied to layer outputs",
            "ReLU/Tanh/GELU",
        )

        table.add_row(
            "Loop Parameters",
            style="bold black on yellow",
            end_section=True,
        )

        table.add_row(
            "Batch Size",
            "32",
            "Number of samples processed before model update",
            "8 to 512",
        )
        table.add_row(
            "Epochs",
            str(self.hyperparams.n_epochs),
            "Number of complete passes through the dataset",
            "10 to 1000",
        )
        table.add_row(
            "Learning Rate Schedule",
            "CosineAnnealing",
            "Strategy for adjusting learning rate during training",
            "Step/Cosine/Linear",
        )
        table.add_row(
            "Early Stopping Patience",
            "10",
            "Epochs to wait before stopping if no improvement",
            "5 to 50",
        )
        print("\n")
        self.console.print(table)
