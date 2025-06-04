import pytest
import torch.testing
from lhotse.testing.random import deterministic_rng

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.modules import ConformerEncoder
from nemo.collections.asr.parts.preprocessing import FilterbankFeatures


@pytest.mark.parametrize("length", list(range(15950, 16050)))
def test_preprocessor_invariant_to_padding(deterministic_rng, length):
    # Settings corresponding to Canary-1B features
    f = FilterbankFeatures(n_window_size=400, nfilt=128, pad_to=0).eval()

    # Test data:
    # * a1: 1s "audio"
    # * a2: 1s "audio" + 1s padding, keep length tensor unchanged
    a1 = torch.arange(0, length).unsqueeze(0) / 16000
    a1l = torch.tensor([length])

    a2 = torch.cat([a1, torch.zeros(1, 16000)], dim=1)
    a2l = a1l.clone()

    mels1, mels1l = f(a1, a1l)
    mels2, mels2l = f(a2, a2l)

    torch.testing.assert_close(mels1, mels2[..., :mels1l])


@pytest.mark.skip(reason="Used only for debugging.")
@pytest.mark.parametrize("length", [16000])
def test_canary_invariant_to_padding(deterministic_rng, length):
    model = ASRModel.from_pretrained("nvidia/canary-180m-flash").eval()

    # Test data:
    # * a1: 1s "audio"
    # * a2: 1s "audio" + 1s padding, keep length tensor unchanged
    a1 = torch.arange(0, length).unsqueeze(0) / 16000
    a1l = torch.tensor([length])

    a2 = torch.cat([a1, torch.zeros(1, 16000)], dim=1)
    a2l = a1l.clone()

    mels1, mels1l = model.preprocessor(input_signal=a1, length=a1l)
    mels2, mels2l = model.preprocessor(input_signal=a2, length=a2l)

    torch.testing.assert_close(mels1, mels2[..., :mels1l])

    h1, h1l = model.encoder(audio_signal=mels1, length=mels1l)
    h2, h2l = model.encoder(audio_signal=mels2, length=mels2l)

    torch.testing.assert_close(h1, h2[..., :h1l])


@pytest.mark.xfail(reason="Fixme")
@pytest.mark.parametrize("length", [16000])
def test_conformer_inference_invariant_to_padding(deterministic_rng, length):
    f = FilterbankFeatures(n_window_size=400, nfilt=128, pad_to=0).eval()
    model = ConformerEncoder(feat_in=128, n_layers=2, d_model=128, feat_out=128, causal_downsampling=True)

    # Test data:
    # * a1: 1s "audio"
    # * a2: 1s "audio" + 1s padding, keep length tensor unchanged
    a1 = torch.arange(0, length).unsqueeze(0) / 16000
    a1l = torch.tensor([length])

    a2 = torch.cat([a1, torch.zeros(1, 16000)], dim=1)
    a2l = a1l.clone()

    mels1, mels1l = f(a1, a1l)
    mels2, mels2l = f(a2, a2l)

    torch.testing.assert_close(mels1, mels2[..., :mels1l])

    h1, h1l = model(audio_signal=mels1, length=mels1l)
    h2, h2l = model(audio_signal=mels2, length=mels2l)

    torch.testing.assert_close(h1, h2[..., :h1l])


def test_conformer_inference_invariant_to_batch_size(deterministic_rng):
    model = ConformerEncoder(feat_in=128, n_layers=2, d_model=128, feat_out=128)
    model = model.eval()

    audio_signal_bs1, length_bs1 = model.input_example()
    h_bs1, h_length_bs1 = model(audio_signal=audio_signal_bs1, length=length_bs1)

    audio_signal_bs2 = audio_signal_bs1.repeat(2, 1, 1)
    length_bs2 = length_bs1.repeat(2)
    h_bs2, h_length_bs2 = model(audio_signal=audio_signal_bs2, length=length_bs2)

    torch.testing.assert_close(h_bs1, h_bs2[:1])
    torch.testing.assert_close(h_bs1, h_bs2[1:])
