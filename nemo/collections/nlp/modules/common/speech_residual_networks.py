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
        out = torch.cat([dec_hidden, dec_logits, layer_index_tensor], dim=-1)
        if mask is not None:
            out = out * mask.T.unsqueeze(-1)
        out = torch.nn.functional.relu(self.conv(out.transpose(1, 2)))
        out = self.norm(out.transpose(1, 2))
        out = self.dropout(out)
        if mask is not None:
            out = out * mask.T.unsqueeze(-1)

        return out

class LinearModule(torch.nn.Module):
    def __init__(self, dec_hid_size, token_input_size, token_output_size=None, n_quantizer_layers=8, dropout=0.25):
        super().__init__()
        if token_output_size is None:
            token_output_size = token_input_size
        self.layers = torch.nn.ModuleList([torch.nn.Linear(dec_hid_size+token_input_size, token_output_size) for _ in range(n_quantizer_layers)])
        # self.dropout = torch.nn.Dropout(dropout)

    def forward(self, dec_hidden, dec_logits, layer_i, mask):
        out = torch.cat([dec_hidden, dec_logits], dim=-1)
        # if mask is not None:
        #     out = out * mask.T.unsqueeze(-1)
        out = self.layers[layer_i](out)
        # out = self.norm(out.transpose(1, 2))
        # out = self.dropout(out)
        if mask is not None:
            out = out * mask.T.unsqueeze(-1)

        return out