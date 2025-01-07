"""W2V models for both CBOW and Skipgram"""

from torch import nn
import torch


class Word2VecModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, is_skip_gram: bool = True):
        """
        Word2Vec model implementation in PyTorch

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            is_skip_gram: If True, use Skip-gram model. If False, use CBOW
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.is_skip_gram = is_skip_gram

        # Input embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Output layer
        self.output = nn.Linear(embedding_dim, vocab_size)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize embeddings and linear layer weights"""
        initrange = 0.5 / self.embedding_dim
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output.weight.data.uniform_(-initrange, initrange)
        self.output.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model

        Args:
            x: Input tensor of word indices
               For Skip-gram: shape (batch_size, 1)
               For CBOW: shape (batch_size, context_size)

        Returns:
            Output logits of shape (batch_size, vocab_size)
        """
        if self.is_skip_gram:
            embeds = self.embedding(x).squeeze(1)
        else:
            embeds = self.embedding(x).mean(dim=1)

        return self.output(embeds)
