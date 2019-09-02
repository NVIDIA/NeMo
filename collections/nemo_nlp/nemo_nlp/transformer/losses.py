import torch
from torch import nn


class SmoothedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing for a batch of sequences.

    Args:
        label_smoothing: label smoothing coefficient, usually set between 0.0
            and 0.1 in language modeling and translation pipelines
        predict_last_k: int parameter which sets the number of last tokens to
            calculate the loss for, for example
            0: (default) calculate loss on the entire sequence (e.g., NMT)
            1: calculate loss on the last token only (e.g., LM evaluation)
            Intermediate values allow to control the trade-off between eval
            time (proportional to the number of batches) and eval performance
            (proportional to the number of context tokens).
    """

    def __init__(self, label_smoothing=0.0, predict_last_k=0):
        super().__init__()
        self._smoothing = label_smoothing
        self._predict_last_k = predict_last_k

    def forward(self, log_probs, output_ids, output_mask):
        """
        Args:
            log_probs: float tensor of shape batch_size x seq_len x vocab_size
            output_ids: int tensor of shape batch_size x seq_len
            output_mask: binary tensor of shape batch_size x seq_len
        """
        batch_size, seq_len, vocab_size = log_probs.size()
        smoothing = vocab_size * self._smoothing / (vocab_size - 1)
        target_log_probs = log_probs.gather(
            2, output_ids.unsqueeze(2)).squeeze(2)
        smoothing_log_probs = log_probs.mean(dim=-1)
        neg_log_likelihood = (1.0 - smoothing) * target_log_probs + \
            smoothing * smoothing_log_probs
        neg_log_likelihood = neg_log_likelihood[:, -self._predict_last_k:]
        output_mask = output_mask[:, -self._predict_last_k:]
        neg_log_likelihood = -torch.sum(neg_log_likelihood * output_mask)
        neg_log_likelihood = neg_log_likelihood / (output_mask.sum() + 1e-6)
        return neg_log_likelihood


class SequenceClassificationLoss(nn.Module):
    """
    Sequence classification loss.
    """

    def __init__(self):
        super().__init__()

    def forward(self, log_probs, labels):
        """
        Args:
            log_probs: float tensor of shape batch_size x num_classes
            labels: int tensor of shape batch_size
        """
        log_probs_target = log_probs.gather(1, labels.unsqueeze(1))
        neg_log_likelihood = -torch.mean(log_probs_target)
        return neg_log_likelihood
