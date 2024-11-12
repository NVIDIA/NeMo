import torch
from torch import Tensor, nn


class TransformersProjector(nn.Module):
    def __init__(self, in_features, out_features, num_query_token, **kwargs):
        super().__init__()
        hidden_dim = 512
        self.in_fc = nn.Linear(in_features, hidden_dim)
        self.tfm = nn.Transformer(
            batch_first=True,
            norm_first=True,
            d_model=hidden_dim,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            nhead=4,
        )
        self.out_fc = nn.Linear(hidden_dim, out_features)

        self.query_embs = nn.Parameter(torch.randn(1, num_query_token, hidden_dim))
        self.query_embs.data.normal_(mean=0.0, std=0.0)

    def forward(self, x):
        # x = x + input_embs # Yash TODO: pass in input embeddings
        x = self.in_fc(x)
        x = self.tfm(x, self.query_embs.repeat(x.shape[0], 1, 1))
        outputs = self.out_fc(x)
        return outputs
