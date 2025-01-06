"""Main training logic. Does not concern itself with data manipulation."""

from logger import logger_factory

from datasets import DatasetDict


class Trainer:
    def __init__(self, dd: DatasetDict) -> None:
        self.n_epochs = 1
        self.model = None
        self.chkpt_iter = 5  # Number of training passes between each checkpoint (includes validation passes)
        self.hyperparams = dict()
        self.dd = dd
        self.td = dd["training"]
        self.vd = dd["validation"]
        self.testd = dd["test"]
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
