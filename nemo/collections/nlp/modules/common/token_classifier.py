from torch import nn as nn

from nemo.collections.common.parts import MultiLayerPerceptron, transformer_weights_init

__all__ = ['TokenClassifier']

ACT2FN = {"gelu": nn.functional.gelu, "relu": nn.functional.relu}


class TokenClassifier(nn.Module):
    def __init__(
        self,
        hidden_size: object,
        num_classes: object,
        activation: object = 'relu',
        log_softmax: object = True,
        dropout: object = 0.0,
        use_transformer_pretrained: object = True,
    ) -> object:
        super().__init__()
        if activation not in ACT2FN:
            raise ValueError(f'activation "{activation}" not found')
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = ACT2FN[activation]
        self.norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.mlp = MultiLayerPerceptron(
            hidden_size, num_classes, num_layers=1, activation=activation, log_softmax=log_softmax
        )
        self.dropout = nn.Dropout(dropout)
        if use_transformer_pretrained:
            self.apply(lambda module: transformer_weights_init(module, xavier=False))

    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act(hidden_states)
        transform = self.norm(hidden_states)
        logits = self.mlp(transform)
        return logits
