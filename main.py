from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE

from logger import logger_factory
from model import Word2VecModel
from trainer import HyperParams, Trainer
from utility_fns import horizontal_rule

if __name__ == "__main__":
    logger = logger_factory()
    logger.info("W2V")
    horizontal_rule()

    hparams = HyperParams()
    vocab_size = 10_000
    embedding_dim = 64
    is_skipgram = True

    # Tokenizer
    tokenizer = Tokenizer.from_pretrained("sigil-ml/WikipediaBPETokenizer")

    model = Word2VecModel(vocab_size, embedding_dim, is_skipgram)

    dd = load_dataset("sigil-ml/PreTokenizedWikiEn", streaming=True)

    chkpt_dir = Path("./checkpoints")
    trainer = Trainer(dd, model, hparams, tokenizer, logger, chkpt_dir)
    trainer.start()
    # trainer.display_config_table()

    # trainer.pre_check()
