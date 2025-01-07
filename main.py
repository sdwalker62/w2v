from trainer import Trainer, HyperParams
from datasets import load_dataset
from logger import logger_factory
from model import Word2VecModel
from utility_fns import horizontal_rule


if __name__ == "__main__":
    logger = logger_factory()
    logger.info("W2V")
    horizontal_rule()

    hparams = HyperParams()
    vocab_size = 10_000
    embedding_dim = 64
    is_skipgram = True
    hparams.hypertable()
    model = Word2VecModel(vocab_size, embedding_dim, is_skipgram)

    dd = load_dataset("sigil-ml/PreTokenizedWikiEn", streaming=True)
    trainer = Trainer(dd, model, hparams)
    
    trainer.pre_check()
