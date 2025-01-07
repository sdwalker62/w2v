"""Main training logic. Does not concern itself with data manipulation."""

from logger import logger_factory

from rich.console import Console
from rich.table import Table
from rich import box
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from typing import TypeAlias
from torch.nn import Module

dataset: TypeAlias = Dataset | DatasetDict | IterableDataset | IterableDatasetDict


class HyperParams:
    def __init__(self, n_epochs=1, lr=1e-1, optimizer="adam", loss_fn="ce") -> None:
        self.n_epochs = n_epochs
        self.lr = lr
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.console = Console()

    def hypertable(self) -> None:
        """Display a nicely formatted table of the hyperparameters."""
        table = Table(
            title="Machine Learning Training Hyperparameters",
            box=box.ROUNDED,
            highlight=True,
            show_header=True,
            header_style="bold cyan",
            title_style="bold magenta",
            caption="* Default values may vary based on specific use cases",
            caption_style="italic",
        )

        table.add_column("Hyperparameter", style="bold green")
        table.add_column("Value", justify="center")
        table.add_column("Description", style="dim")
        table.add_column("Typical Range", justify="center", style="yellow")

        table.add_row(
            "Learning Rate",
            "0.001",
            "Step size for gradient descent updates",
            "1e-5 to 1e-1",
        )

        table.add_row(
            "Batch Size",
            "32",
            "Number of samples processed before model update",
            "8 to 512",
        )
        table.add_row(
            "Epochs",
            "100",
            "Number of complete passes through the dataset",
            "10 to 1000",
        )
        table.add_row(
            "Hidden Units", "256", "Number of neurons in hidden layers", "32 to 1024"
        )
        table.add_row(
            "Dropout Rate",
            "0.2",
            "Fraction of neurons randomly deactivated during training",
            "0.1 to 0.5",
        )
        table.add_row(
            "Weight Decay", "1e-4", "L2 regularization parameter", "1e-6 to 1e-3"
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

        self.console.print(table)


class Trainer:
    def __init__(self, dd: dataset, model: Module, hparams: HyperParams) -> None:
        self.model = model
        self.chkpt_iter = 5  # Number of training passes between each checkpoint (includes validation passes)
        self.hyperparams = hparams
        self.dd = dd
        if isinstance(self.dd, DatasetDict) or isinstance(self.dd, IterableDatasetDict):
            self.td = self.dd["train"]
            self.vd = self.dd["validation"]
            self.testd = self.dd["test"]

        if isinstance(dd, IterableDatasetDict) or isinstance(dd, IterableDataset):
            self.streaming = True
        self.log = logger_factory()

    def pre_check(self):
        """Performs pre-training actions to ensure functionality."""
        self.log.info(len(self.td))
        self.log.info(len(self.vd))
        self.log.info(len(self.testd))

    def start(self):
        """Training entrypoint."""

    def train_epoch(self):
        """Completes a single pass over the training set."""

    def validate(self):
        """Completes a single pass over the validation set."""

    def test(self):
        """Completes a single pass over the test set."""

    def checkpoint(self):
        """Perform checkpointing duties."""

    def save_model(self):
        """Dumps the model to disk."""
