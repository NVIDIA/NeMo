# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from examples.asr.export.build_torch_trt_encoder import capture_encoder
from nemo.collections.asr.modules.transformer_encoder import TransformerEncoder


@pytest.mark.unit
def test_dynamic_transformer_torch_export():
    torch.manual_seed(42)
    model = TransformerEncoder(
        feat_in=8,
        d_model=32,
        n_heads=2,
        n_layers=2,
        feat_out=24,
        subsampling='feature_stacking',
        subsampling_factor=4,
        self_attention_model='rope',
        rotary_fraction=0.5,
        qk_norm=True,
        qkv_bias=True,
        drop_rate=0.0,
        pos_emb_max_len=128,
        sync_max_audio_length=False,
    ).eval()
    program = capture_encoder(model, min_frames=8, opt_frames=16, max_frames=128, opt_batch=2, max_batch=4)
    restored = program.module()
    for batch, time, lengths in [(2, 16, [16, 9]), (1, 9, [9]), (3, 23, [23, 13, 0]), (2, 80, [65, 0])]:
        features = torch.randn(batch, 8, time)
        lengths = torch.tensor(lengths)
        with torch.no_grad():
            expected = model(features, lengths)
            actual = restored(features, lengths)
        torch.testing.assert_close(actual[0], expected[0], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)


@pytest.mark.unit
@pytest.mark.run_only_on("GPU")
def test_dynamic_transformer_tensorrt(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("TensorRT execution requires CUDA.")
    trt = pytest.importorskip("tensorrt")
    pytest.importorskip("torch_tensorrt")
    from examples.asr.export.build_torch_trt_encoder import build_encoder

    torch.manual_seed(42)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    model = (
        TransformerEncoder(
            feat_in=8,
            d_model=32,
            n_heads=2,
            n_layers=2,
            feat_out=24,
            subsampling='feature_stacking',
            subsampling_factor=4,
            self_attention_model='rope',
            rotary_fraction=0.5,
            qk_norm=True,
            qkv_bias=True,
            drop_rate=0.0,
            pos_emb_max_len=128,
            sync_max_audio_length=False,
        )
        .eval()
        .cuda()
    )
    program = capture_encoder(model, min_frames=8, opt_frames=16, max_frames=128, opt_batch=2, max_batch=4)
    plan = build_encoder(program, 8, 16, 128, 2, 4, 8, workspace_bytes=128 * 1024**2)
    saved = tmp_path / "encoder.plan"
    saved.write_bytes(plan)
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(saved.read_bytes())
    assert engine is not None
    context = engine.create_execution_context()
    assert context is not None
    assert {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)} == {
        "audio_signal",
        "length",
        "outputs",
        "encoded_lengths",
    }
    dtypes = {trt.float32: torch.float32, trt.int64: torch.int64}
    with torch.inference_mode():
        for batch, frames, lengths in [(2, 16, [16, 9]), (1, 9, [9]), (4, 23, [23, 13, 1, 0]), (2, 128, [128, 0])]:
            features = torch.randn(batch, 8, frames, device="cuda")
            lengths = torch.tensor(lengths, device="cuda")
            expected = model(features, lengths)
            buffers = {"audio_signal": features, "length": lengths}
            for name, value in buffers.items():
                assert context.set_input_shape(name, tuple(value.shape))
                assert context.set_tensor_address(name, value.data_ptr())
            assert not context.infer_shapes()
            for name in ("outputs", "encoded_lengths"):
                buffers[name] = torch.empty(
                    tuple(context.get_tensor_shape(name)), device="cuda", dtype=dtypes[engine.get_tensor_dtype(name)]
                )
                assert context.set_tensor_address(name, buffers[name].data_ptr())
            assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            torch.cuda.current_stream().synchronize()
            torch.testing.assert_close(buffers["outputs"], expected[0], atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(buffers["encoded_lengths"], expected[1], atol=0, rtol=0)
