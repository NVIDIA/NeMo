from __future__ import annotations
import torch
import torch.nn as nn


class FusedBatchNorm1d(nn.Module):
    """
    Fused BatchNorm to use in Conformer to improve accuracy in finetuning with TTS scenario
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    @classmethod
    def from_batchnorm(cls, bn: nn.BatchNorm1d) -> FusedBatchNorm1d:
        assert isinstance(bn, nn.BatchNorm1d)
        fused_bn = FusedBatchNorm1d(bn.num_features)
        # init projection params from original batch norm
        # so, for inference mode output is the same
        std = torch.sqrt(bn.running_var.data + bn.eps)
        fused_bn.weight.data = bn.weight.data / std
        fused_bn.bias.data = bn.bias.data - bn.running_mean.data * fused_bn.weight.data
        return fused_bn

    def forward(self, x: torch.Tensor):
        return x * self.weight.unsqueeze(-1) + self.bias.unsqueeze(-1)
