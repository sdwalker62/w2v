"""Main training logic. Does not concern itself with data manipulation."""

from logger import logger_factory

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from typing import TypeAlias
from dataclasses import dataclass
from torch.nn import Module

dataset: TypeAlias = Dataset | DatasetDict | IterableDataset | IterableDatasetDict


@dataclass(frozen=True)
class HyperParams:
    n_epochs: int = 1
    lr: float = 1e-1
    optimizer: str = "adam"
    loss_fn: str = "ce"


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
