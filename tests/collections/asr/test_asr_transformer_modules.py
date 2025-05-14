import pytest
import torch
from nemo.collections.asr.modules.transformer.transformer_modules import MultiHeadAttention


class TestTransformerMultiHeadAttention:
    @pytest.mark.unit
    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    @pytest.mark.parametrize("seq_len", [50, 200, 1000])
    @pytest.mark.parametrize("masked", [False, True])
    def test_mha_torch_sdpa(self, batch_size, seq_len, masked):
        torch.random.manual_seed(0)
        num_heads = 8
        hidden_dim = 192
        mha = MultiHeadAttention(hidden_dim, num_heads, 0.1, 0.1)
        mha_sdpa = MultiHeadAttention(hidden_dim, num_heads, 0.1, 0.1, use_pytorch_sdpa=True)
        mha_sdpa.load_state_dict(mha.state_dict())

        mha.eval()
        mha_sdpa.eval()

        input_tensor = torch.randn(batch_size, seq_len, 192)
        if masked:
            mask = torch.randint(0, 2, (batch_size, num_heads, seq_len, seq_len)).bool()
            mask = torch.where(mask, torch.tensor(float('-inf')), torch.tensor(0.0))
        else:
            mask = None

        with torch.no_grad():
            output_tensor = mha(input_tensor, input_tensor, input_tensor, mask)
            output_tensor_sdpa = mha_sdpa(input_tensor, input_tensor, input_tensor, mask)
            assert torch.allclose(output_tensor, output_tensor_sdpa, atol=1e-5)
