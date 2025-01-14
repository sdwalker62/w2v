"""Main training logic. Does not concern itself with data manipulation."""

import json
import re
import statistics
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
import matplotlib.pyplot as plt

from model import Word2VecModel

dataset: TypeAlias = Dataset | DatasetDict | IterableDataset | IterableDatasetDict

# TODO: Add wandb integration
# TODO: Add graphs to checkpoint dir
# TODO: Add load model
# TODO: Auto determine best checkpoint from testing loss
# TODO: Finish rich table


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
        window_size=4,
        iter_report=1_000,
        chkpt_iter=100_000,
    ) -> None:
        self.n_epochs = n_epochs
        self.lr = lr
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.embed_dim = embed_dim
        self.is_skipgram = is_skipgram
        self.batch_size = batch_size
        self.window_size = window_size
        self.iter_report = iter_report
        self.chkpt_iter = chkpt_iter


class Trainer:
    def __init__(
        self,
        dd: dataset,
        model: Word2VecModel,
        hparams: HyperParams,
        tokenizer: Tokenizer | None,
        logger: Logger,
        chkpt_dir: Path,
    ) -> None:
        self.log = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hyperparams = hparams
        self.tokenizer = tokenizer
        self.vocab = self.tokenizer.get_vocab()
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.log.info(f"Vocab size: {self.vocab_size}")
        self.model = model(
            self.vocab_size,
            self.hyperparams.embed_dim,
            self.hyperparams.is_skipgram,
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.hyperparams.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.dd = dd
        if isinstance(self.dd, DatasetDict) or isinstance(self.dd, IterableDatasetDict):
            self.td = self.dd["train"]
            self.vd = self.dd["validation"]
            self.testd = self.dd["test"]
        self.console = Console()
        self.chkpt_dir = chkpt_dir
        self.specific_chkpt_dir = None
        self.current_time = datetime.now()
        self.chkpt_n = 0
        self.training_losses = []
        self.validation_losses = []
        self.testing_loss = 0.0
        self.phrases_txt, self.phrases_list = None, None
        self.words_txt, self.words_list = None, None
        self.validation_analogy_results = []
        self.test_analogy_results = None
        self.embedding_history = []
        self.embeddings = None
        self.debug_mode = True
        self.debug_iter = 10
        # init functions
        self.create_dirs()
        self.load_analogies()
        self.clean_analogies()
        self.transform_analogies()
        self.display_config_table()

    def pre_check(self) -> None:
        """Performs pre-training actions to ensure functionality."""
        self.log.info("Executing pre-checks")
        # TODO: Finish this function!
        # Set to debug mode
        # delete the created pre-check directory
        # set debug mode off
        self.log.info("Pre-checks complete, artifacts removed")

    def dump_stats(self) -> None:
        """Dump training statistics to JSON"""
        print("\n")
        self.log.info("Gathering training statistics")
        training_stats = {
            "training_losses": self.training_losses,
            "validation_losses": self.validation_losses,
            "test_loss": self.testing_loss,
            "mean_analogy_norm": self.test_analogy_results[0],
            "median_analogy_norm": self.test_analogy_results[1],
            "std_analogy_norm": self.test_analogy_results[2],
        }
        self.log.info(
            f"Saving training statistics at {str(self.specific_chkpt_dir)}/training_stats.json"
        )
        with open(self.specific_chkpt_dir / "training_stats.json", "w") as f:
            json.dump(training_stats, f)  # noqa

    def start(self) -> None:
        """Training entrypoint."""
        self.log.info("Training started")
        self.pre_check()
        self.train()

    def debug_boundary(self, idx: int) -> bool:
        """Stop iteration for debugging.

        Parameters
        ----------
        idx : int
            Checked index for stop condition

        Returns
        -------
        None
        """
        if self.debug_mode and idx == self.debug_iter:
            return True
        return False

    def update_embedding_table(self) -> None:
        """Updates the embedding table."""
        self.embeddings = {
            word: self.model.embedding.weight.data[idx].cpu()
            for word, idx in self.vocab.items()
        }
        self.embedding_history.append(self.embeddings)

    def train(self) -> None:
        """Training loop logic."""
        print("\n")
        # Training loop
        training_passes = 0
        for epoch in range(self.hyperparams.n_epochs):
            print("\n")
            self.log.info(f"Running epoch: {epoch + 1}")
            self.model.train()
            total_loss = 0

            # The collation function runs after the batching process if we use
            # `torch.data.DataLoader` so we will instead loop over the training
            # data, collate each item, and then batch the results.
            # Remote Data -> Get Row -> Collate -> Batch -> Train
            for example_idx, example in enumerate(self.td):
                """Each example is a dictionary with features such as ids, text, etc."""
                batches = self.prepare_batches(example)  # list[tuple[list[int], int]]
                for batch_idx, _X in enumerate(batches):
                    _X, _y = self.split_to_tensors(_X)
                    training_passes += 1
                    # Forward pass
                    self.optimizer.zero_grad()
                    output = self.model(_X)
                    loss = self.criterion(output, _y)

                    # Backward pass
                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()

                    if (batch_idx + 1) % self.hyperparams.iter_report == 0:
                        self.log.info(
                            f"Epoch {epoch + 1}/{self.hyperparams.n_epochs}, "
                            f"Loss: {total_loss / (batch_idx + 1):.4f}"
                        )
                    if self.debug_boundary(batch_idx):
                        break
                if self.debug_boundary(example_idx):
                    break
                if (example_idx + 1) % self.hyperparams.chkpt_iter == 0:
                    self.checkpoint(f"iter-{epoch + 1}_{example_idx + 1}")

            self.log.info(
                f"Epoch {epoch + 1} completed, "
                f"Average Loss: {total_loss / training_passes:.4f}"
            )
            self.checkpoint(f"epoch-{epoch}")
            self.test(validation=True)
        self.test()
        self.dump_stats()

    def test(self, validation: bool = False) -> None:
        """Completes a single pass over the test/validation set."""
        print("\n")
        if validation:
            self.log.info("Running validation loop")
            data = self.vd
        else:
            self.log.info("Running test loop")
            data = self.testd
        self.model.eval()
        running_loss = 0.0
        total_batches = 0
        with torch.no_grad():
            for example_idx, example in enumerate(data):
                """Each example is a dictionary with features such as ids, text, etc."""
                batches = self.prepare_batches(example)  # list[tuple[list[int], int]]
                for batch_idx, x_batch in enumerate(batches):
                    x_batch, y_true = self.split_to_tensors(x_batch)
                    output = self.model(x_batch)
                    loss = self.criterion(output, y_true)
                    running_loss += loss
                    total_batches += 1
                    if self.debug_boundary(batch_idx):
                        break
                if self.debug_boundary(example_idx):
                    break
        avg_loss = running_loss / total_batches
        self.testing_loss = avg_loss.item()

        self.test_model_on_analogies(False)

    def create_dirs(self) -> None:
        """Create any necessary directories for the training script"""
        self.log.info(f"Creating checkpoint directory within: {self.chkpt_dir}")
        if self.hyperparams.is_skipgram:
            dir_name = "skipgram--"
        else:
            dir_name = "cbow--"
        dir_name += self.current_time.strftime("%Y-%m-%d-%H-%M-%S")
        self.specific_chkpt_dir = self.chkpt_dir / Path(dir_name)
        if not self.specific_chkpt_dir.absolute().exists():
            self.specific_chkpt_dir.absolute().mkdir(exist_ok=False, parents=True)

    def checkpoint(self, desc: str) -> None:
        """Perform checkpointing duties.

        Parameters
        ----------
        desc : str
            Description to be added as a suffix to the save file
        """
        # Create unique directory within the checkpoint directory
        assert self.chkpt_dir.exists(), self.log.error(
            f"Cannot find checkpoint directory: {self.chkpt_dir}"
        )
        self.chkpt_n += 1
        self.log.info(f"Performing checkpoint: {self.chkpt_n}")

        self.save_model(desc)

        # Grab new embeddings
        self.update_embedding_table()

    def save_model(self, desc: str) -> None:
        """Dumps the model to disk.

        Parameters
        ----------
        desc : str
            Description to be added as a suffix to the save file
        """
        output_name = str(self.chkpt_n) + "-" + desc + ".pt"
        output_path = self.specific_chkpt_dir / Path(output_name)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "tokenizer": self.tokenizer,
                "embedding_history": self.embedding_history,
                "validation_analogy_results": self.validation_analogy_results,
                "embedding_dim": self.hyperparams.embed_dim,
                "is_skip_gram": self.hyperparams.is_skipgram,
            },
            output_path,
        )
        self.log.info(f"Model saved to {output_path}")

    def load_model(
        self,
        path: str,
    ) -> Word2VecModel:
        """Load the model from disk, includes extras such as the tokenizer.

        Parameters
        ----------
        path : str
            Path to the directory containing the `*.pt` file

        Returns
        -------
        Word2VecModel
            The PyTorch `module` object
        """
        # TODO: Fix the load logic
        checkpoint = torch.load(path, map_location=self.device)

        model = Word2VecModel(
            len(checkpoint["vocab"]),
            checkpoint["embedding_dim"],
            checkpoint["is_skip_gram"],
        ).to(self.device)

        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def prepare_batches(self, row_dict) -> list[list[tuple]]:
        """Generate input-target pairs for training.

        Parameters
        ----------
        row_dict : dict
            A dictionary of columns from a single row of the training table

        Returns
        -------
        list[tuple[torch.Tensor, torch.Tensor]]
            Returns a list of tuples of the form [(X1, y1), (X2, y2)]
        """
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

            center = word_indices[i]
            pad_id = self.vocab["[PAD]"]

            # Skipgram
            if self.hyperparams.is_skipgram:
                # Skip-gram: predict context words from center word
                for ctx in context:
                    if ctx != pad_id:  # Skip padding
                        pairs.append(([center], ctx))
            # CBOW
            else:
                if word_indices[i] == pad_id:  # Skip padding
                    continue

                # Pad context to fixed size
                ctx_size = 2 * self.hyperparams.window_size
                ctx_padded = context + [pad_id] * (ctx_size - len(context))
                ctx_padded = ctx_padded[:ctx_size]
                # CBOW: predict center word from context words
                pairs.append((ctx_padded, center))

        # Create batches
        pairs = [
            pairs[i : i + self.hyperparams.batch_size]
            for i in range(0, len(pairs), self.hyperparams.batch_size)
        ]
        return pairs

    def split_to_tensors(
        self,
        data: list[tuple],
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

    def load_analogies(self) -> None:
        """Loads the analogy data for testing phase 2."""
        phrases_path = Path("./testing_data/question-phrases.txt").absolute()
        words_path = Path("./testing_data/question-words.txt").absolute()

        assert phrases_path.exists(), self.log.error(
            f"Cannot find phrases file at {str(phrases_path.absolute())}"
        )
        assert words_path.exists(), self.log.error(
            f"Cannot find words file at {str(words_path.absolute())}"
        )

        with open(phrases_path, "r") as f:
            self.phrases_txt = f.readlines()

        with open(words_path, "r") as f:
            self.words_txt = f.readlines()

    def clean_analogies(self) -> None:
        """Removes headers and `\n` characters from the lists."""
        header_pattern = r"^: .+"
        clean_header_matches_phrases = [
            s for s in self.phrases_txt if not re.match(header_pattern, s)
        ]
        clean_header_matches_phrases = [
            s.replace("\n", "") for s in clean_header_matches_phrases
        ]
        clean_header_matches_words = [
            s for s in self.words_txt if not re.match(header_pattern, s)
        ]
        clean_header_matches_words = [
            s.replace("\n", "") for s in clean_header_matches_words
        ]

        self.phrases_txt = clean_header_matches_phrases
        self.words_txt = clean_header_matches_words

    def transform_analogies(self) -> None:
        """Transform the analogy list to a form useful for processing."""
        self.phrases_list = [(*s.split(" "),) for s in self.phrases_txt]
        self.words_list = [(*s.split(" "),) for s in self.words_txt]

    def test_model_on_analogies(self, validation_mode: bool = True) -> None:
        """Run analogy tests"""
        self.log.info("Processing analogies")

        # Grab new embeddings
        self.update_embedding_table()

        # phrases tests
        norms = []
        for _list in [self.phrases_list, self.words_list]:
            for p in _list:
                # p := (word1, word2, word3, word4)
                embeddings = []
                encodings = self.tokenizer.encode_batch(p)
                for e in encodings:
                    # e := Encoding Object (contains ids, tokens, etc.)
                    aggregate_emb = torch.zeros(self.hyperparams.embed_dim)
                    for sub_token in e.tokens:
                        if sub_token in self.embeddings.keys():
                            emb = self.embeddings[sub_token]
                        else:
                            emb = self.embeddings["[UNK]"]
                        aggregate_emb += emb
                        aggregate_emb /= len(e.tokens)
                    embeddings.append(aggregate_emb)

                first_diff = embeddings[0] - embeddings[1]
                second_diff = embeddings[2] - embeddings[3]
                diff = first_diff - second_diff
                norm = diff.norm()
                norms.append(norm.item())

        mean = statistics.mean(norms)
        median = statistics.median(norms)
        stdev = statistics.stdev(norms)

        results = (mean, median, stdev)
        if validation_mode:
            self.validation_analogy_results.append(results)
        else:
            self.test_analogy_results = results

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
