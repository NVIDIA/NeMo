import os

import autoconfig.utils as ut
import pytest
from omegaconf import OmegaConf


class TestCalculateModelSize:

    margin = 0.05

    @pytest.mark.parametrize(
        "vocab,seq_len,hs,layers,ffn,kv,att,model_name,expected",
        [
            # GPT-3 tests
            (51200, 2048, 768, 12, 768 * 4, None, 12, "gpt3", 0.126),
            (51200, 4096, 1536, 26, 1536 * 4, None, 16, "gpt3", 0.843),
            (51200, 4096, 2560, 24, 2560 * 4, None, 32, "gpt3", 2.0),
            (51200, 2048, 4096, 24, 4096 * 4, None, 32, "gpt3", 5.0),
            (51200, 2048, 5120, 24, 5120 * 4, None, 40, "gpt3", 8.0),
            (51200, 2048, 6144, 44, 6144 * 4, None, 48, "gpt3", 20.0),
            (51200, 8192, 8192, 52, 8192 * 4, None, 64, "gpt3", 43.0),
            (51200, 2048, 12288, 96, 12288 * 4, None, 96, "gpt3", 175.0),
            # T5 tests
            (29000, 512, 768, 12, 2048, 64, 12, "t5", 0.22),
            (29000, 512, 2048, 24, 5120, 64, 32, "t5", 2.8),
            (29000, 512, 4096, 24, 10240, 64, 64, "t5", 11.0),
            (29000, 512, 5120, 36, 10880, 64, 80, "t5", 23.5),
            (29000, 512, 6144, 48, 10880, 64, 96, "t5", 41.2),
            # mT5 tests
            (250000, 512, 512, 8, 1024, 64, 6, "mt5", 0.17),
            (250000, 512, 768, 12, 2048, 64, 12, "mt5", 0.39),
            (250000, 512, 2048, 24, 5120, 64, 32, "mt5", 3.2),
            (250000, 512, 4096, 24, 10240, 64, 64, "mt5", 11.9),
            (250000, 512, 5120, 36, 10880, 64, 80, "mt5", 24.65),
            (250000, 512, 6144, 48, 10880, 64, 96, "mt5", 42.65),
            # BERT tests
            (30522, 512, 768, 12, 768 * 4, None, 12, "bert", 0.11),
            (30522, 512, 2560, 48, 2560 * 4, None, 40, "bert", 4.0),
            (30522, 512, 6144, 44, 6144 * 4, None, 96, "bert", 20.0),
            (30522, 512, 9216, 96, 9216 * 4, None, 96, "bert", 100.0),
        ],
    )
    def test_calculate_model_size(
        self, vocab, seq_len, hs, layers, ffn, kv, att, model_name, expected
    ):
        params = {
            "vocab_size": vocab,
            "seq_length": seq_len,
            "hidden_size": hs,
            "num_layers": layers,
            "ffn_size": ffn,
            "kv_channels": kv,
            "att_heads": att,
            "model_name": model_name,
        }
        output_size = ut._calculate_model_size(**params)
        assert output_size == pytest.approx(
            expected=expected, rel=self.margin
        ), f"Output of _calculate_model_size should be approximately {expected}, "
        f"but it is {output_size}. Inputs: vocab_size={vocab}, seq_length={seq_len}, "
        f"hidden_size={hs}, num_layers={layers}, ffn_size={ffn}, kv_channels={kv}, "
        f"att_heads={att}, model_name={model_name}. "

    def test_calculate_model_size_not_implemented_error(self):
        params = {
            "vocab_size": 100,
            "seq_length": 100,
            "hidden_size": 100,
            "num_layers": 10,
            "ffn_size": 100,
            "kv_channels": 10,
            "att_heads": 10,
            "model_name": "incorrect_model",
        }
        with pytest.raises(NotImplementedError):
            output_size = ut._calculate_model_size(**params)


class TestCalculatemodelSizeParams:

    margin = 0.05

    @pytest.mark.parametrize(
        "model_size,vocab,seq_len,model_name,expected",
        [
            # GPT-3 tests
            (
                0.126,
                51200,
                2048,
                "gpt3",
                {
                    "layers": 12,
                    "hs": 768,
                    "att": 12,
                    "ffn": None,
                    "kv": None,
                    "lr": 6e-4,
                },
            ),
            (
                0.843,
                51200,
                4096,
                "gpt3",
                {
                    "layers": 26,
                    "hs": 1536,
                    "att": 16,
                    "ffn": None,
                    "kv": None,
                    "lr": 2.5e-4,
                },
            ),
            (
                2.0,
                51200,
                4096,
                "gpt3",
                {
                    "layers": 24,
                    "hs": 2560,
                    "att": 32,
                    "ffn": None,
                    "kv": None,
                    "lr": 1.6e-4,
                },
            ),
            (
                5.0,
                51200,
                2048,
                "gpt3",
                {
                    "layers": 24,
                    "hs": 4096,
                    "att": 32,
                    "ffn": None,
                    "kv": None,
                    "lr": 1.2e-4,
                },
            ),
            (
                8.0,
                51200,
                2048,
                "gpt3",
                {
                    "layers": 24,
                    "hs": 5120,
                    "att": 40,
                    "ffn": None,
                    "kv": None,
                    "lr": 1e-4,
                },
            ),
            (
                20.0,
                51200,
                2048,
                "gpt3",
                {
                    "layers": 44,
                    "hs": 6144,
                    "att": 48,
                    "ffn": None,
                    "kv": None,
                    "lr": 1e-4,
                },
            ),
            (
                43.0,
                51200,
                2048,
                "gpt3",
                {
                    "layers": 52,
                    "hs": 8192,
                    "att": 64,
                    "ffn": None,
                    "kv": None,
                    "lr": 0.8e-4,
                },
            ),
            (
                175.0,
                51200,
                2048,
                "gpt3",
                {
                    "layers": 96,
                    "hs": 12288,
                    "att": 96,
                    "ffn": None,
                    "kv": None,
                    "lr": 0.6e-4,
                },
            ),
            # T5 tests
            (
                0.22,
                29000,
                512,
                "t5",
                {
                    "layers": 12,
                    "hs": 768,
                    "att": 12,
                    "ffn": 2048,
                    "kv": 64,
                    "lr": 0.0001,
                },
            ),
            (
                2.8,
                29000,
                512,
                "t5",
                {
                    "layers": 24,
                    "hs": 2048,
                    "att": 32,
                    "ffn": 5120,
                    "kv": 64,
                    "lr": 0.0001,
                },
            ),
            (
                11.0,
                29000,
                512,
                "t5",
                {
                    "layers": 24,
                    "hs": 4096,
                    "att": 64,
                    "ffn": 10240,
                    "kv": 64,
                    "lr": 0.0001,
                },
            ),
            (
                23.5,
                29000,
                512,
                "t5",
                {
                    "layers": 36,
                    "hs": 5120,
                    "att": 80,
                    "ffn": 10880,
                    "kv": 64,
                    "lr": 0.0001,
                },
            ),
            (
                41.2,
                29000,
                512,
                "t5",
                {
                    "layers": 48,
                    "hs": 6144,
                    "att": 96,
                    "ffn": 10880,
                    "kv": 64,
                    "lr": 0.0001,
                },
            ),
            # mT5 tests
            (
                0.17,
                250000,
                512,
                "mt5",
                {"layers": 8, "hs": 512, "att": 6, "ffn": 1024, "kv": 64, "lr": 0.0001},
            ),
            (
                0.39,
                250000,
                512,
                "mt5",
                {
                    "layers": 12,
                    "hs": 768,
                    "att": 12,
                    "ffn": 2048,
                    "kv": 64,
                    "lr": 0.0001,
                },
            ),
            (
                3.2,
                250000,
                512,
                "mt5",
                {
                    "layers": 24,
                    "hs": 2048,
                    "att": 32,
                    "ffn": 5120,
                    "kv": 64,
                    "lr": 0.0001,
                },
            ),
            (
                11.9,
                250000,
                512,
                "mt5",
                {
                    "layers": 24,
                    "hs": 4096,
                    "att": 64,
                    "ffn": 10240,
                    "kv": 64,
                    "lr": 0.0001,
                },
            ),
            (
                24.65,
                250000,
                512,
                "mt5",
                {
                    "layers": 36,
                    "hs": 5120,
                    "att": 80,
                    "ffn": 10880,
                    "kv": 64,
                    "lr": 0.0001,
                },
            ),
            (
                42.65,
                250000,
                512,
                "mt5",
                {
                    "layers": 48,
                    "hs": 6144,
                    "att": 96,
                    "ffn": 10880,
                    "kv": 64,
                    "lr": 0.0001,
                },
            ),
            # BERT tests
            (
                0.11,
                30522,
                512,
                "bert",
                {
                    "layers": 12,
                    "hs": 768,
                    "att": 12,
                    "ffn": 4 * 768,
                    "kv": None,
                    "lr": 2e-4,
                },
            ),
            (
                4.0,
                30522,
                512,
                "bert",
                {
                    "layers": 48,
                    "hs": 2560,
                    "att": 32,
                    "ffn": 4 * 2560,
                    "kv": None,
                    "lr": 1e-4,
                },
            ),
            (
                20.0,
                30522,
                512,
                "bert",
                {
                    "layers": 44,
                    "hs": 6144,
                    "att": 48,
                    "ffn": 4 * 6144,
                    "kv": None,
                    "lr": 1e-4,
                },
            ),
            (
                100.0,
                30522,
                512,
                "bert",
                {
                    "layers": 96,
                    "hs": 9216,
                    "att": 96,
                    "ffn": 4 * 9216,
                    "kv": None,
                    "lr": 1e-4,
                },
            ),
        ],
    )
    def test_calculate_model_size_params(
        self, model_size, vocab, seq_len, model_name, expected
    ):
        params = {
            "model_size_in_b": model_size,
            "vocab_size": vocab,
            "seq_length": seq_len,
            "model_name": model_name,
        }
        layers, hs, att, ffn, kv, lr = ut.calculate_model_size_params(**params)
        assert (
            layers == expected["layers"]
        ), f"utils.calculate_model_size_params returned layers={layers} but layers={expected['layers']} is expected."
        assert (
            hs == expected["hs"]
        ), f"utils.calculate_model_size_params returned hidden_size={hs} but hidden_size{expected['hs']} is expected."
        assert (
            att == expected["att"]
        ), f"utils.calculate_model_size_params returned attention_heads={att} but attention_heads{expected['att']} is expected."
        assert (
            ffn == expected["ffn"]
        ), f"utils.calculate_model_size_params returned ffn_hidden_size={ffn} but ffn_hidden_size={expected['ffn']} is expected."
        assert (
            kv == expected["kv"]
        ), f"utils.calculate_model_size_params returned kv_channels={kv} but kv_channels={expected['kv']} is expected."
        assert (
            lr == expected["lr"]
        ), f"utils.calculate_model_size_params returned lr={lr} but lr={expected['lr']} is expected."

    def test_calculate_model_size_params_not_implemented_error(self):
        params = {
            "model_size_in_b": 2.0,
            "vocab_size": 100,
            "seq_length": 100,
            "model_name": "incorrect",
        }
        with pytest.raises(NotImplementedError):
            out = ut.calculate_model_size_params(**params)
