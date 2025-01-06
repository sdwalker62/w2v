from trainer import Trainer, HyperParams
from datasets import DatasetDict, load_dataset
from logger import logger_factory
from model import Word2VecModel
import shutil


terminal_width = shutil.get_terminal_size().columns


if __name__ == "__main__":
    logger = logger_factory()
    logger.info("W2V")
    logger.info(terminal_width * "=")

    hparams = HyperParams()
    vocab_size = 10_000
    embedding_dim = 64
    is_skipgram = True
    model = Word2VecModel(vocab_size, embedding_dim, is_skipgram)


    dd = load_dataset("sigil-ml/PreTokenizedWikiEn", streaming=True)
    trainer = Trainer(dd, model, hparams)

    trainer.pre_check()
