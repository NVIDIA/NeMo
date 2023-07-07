import torch

class SimplestModule(torch.nn.Module):
    def __init__(self, dec_hid_size, token_input_size=1, kernel_size=15, dropout=0.5):
        super().__init__()
        self.conv = torch.nn.Conv1d(dec_hid_size+token_input_size+1, dec_hid_size, kernel_size=kernel_size, padding=(kernel_size // 2))
        self.norm = torch.nn.LayerNorm(dec_hid_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, dec_hidden, dec_logits, layer_i, mask):
        layer_index_tensor = torch.tile(torch.tensor([layer_i], requires_grad=True, dtype=dec_hidden.dtype, device=dec_hidden.device), [*dec_hidden.shape[:-1],1])
        # dec_prediction = torch.argmax(dec_logits, dim=-1, keepdim=True)
        out = torch.cat([dec_hidden, dec_logits, layer_index_tensor], dim=-1) * mask.T.unsqueeze(-1)
        # import ipdb; ipdb.set_trace()
        out = torch.nn.functional.relu(self.conv(out.transpose(1, 2)))
        out = self.norm(out.transpose(1, 2))
        out = self.dropout(out) * mask.T.unsqueeze(-1)

        return out