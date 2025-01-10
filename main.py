from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer

from logger import logger_factory
from model import Word2VecModel
from trainer import HyperParams, Trainer
from utility_fns import horizontal_rule

if __name__ == "__main__":
    logger = logger_factory()
    logger.info("W2V")
    horizontal_rule()

    hparams = HyperParams()

    # Tokenizer
    tokenizer = Tokenizer.from_pretrained("sigil-ml/WikipediaBPETokenizer")

    model = Word2VecModel

    dd = load_dataset("sigil-ml/PreTokenizedWikiEn", streaming=True)

    chkpt_dir = Path("./checkpoints")
    trainer = Trainer(dd, model, hparams, tokenizer, logger, chkpt_dir)
    trainer.start()
    # trainer.display_config_table()

    # trainer.pre_check()
