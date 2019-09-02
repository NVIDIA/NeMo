import torch
from torch import nn


class TransformerLogSoftmax(nn.Module):
    """
    Output layer of Transformer architecture which approximates probability
    distribution over *vocab_size* output tokens.
    """

    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states):
        output_states = self.dense(hidden_states).float()
        log_probs = torch.log_softmax(
            output_states, dim=-1).to(hidden_states.dtype)
        return log_probs


class ClassificationLogSoftmax(nn.Module):
    """
    Classifier on top of the hidden representation of the first token, which
    is usually [CLS] token in BERT-like architectures.
    """

    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states):
        output_states = self.dense1(hidden_states[:, 0])
        output_states = torch.tanh(output_states)
        output_states = self.dense2(output_states).float()
        log_probs = torch.log_softmax(
            output_states, dim=-1).to(hidden_states.dtype)
        return log_probs
