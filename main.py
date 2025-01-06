from trainer import Trainer
from datasets import DatasetDict, load_dataset
from logger import logger_factory

if __name__ == "__main__":
    logger = logger_factory()
    logger.info("W2V")

    dd: DatasetDict = load_dataset("sigil-ml/PreTokenizedWikiEn", streaming=True)
    trainer = Trainer(dd)

    trainer.pre_check()
