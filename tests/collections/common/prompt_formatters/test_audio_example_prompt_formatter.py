import pytest
import torch

from nemo.collections.common.prompts.audio_example import AudioExamplePromptFormatter
from nemo.collections.common.prompts.formatter import Modality


def test_example_prompt_formatter_with_audio_requires_modality_encoder(bpe_tokenizer):
    formatter = AudioExamplePromptFormatter(bpe_tokenizer)
    audio = torch.rand(1, 16000)
    with pytest.raises(AssertionError, match="provide modality_encoders argument"):
        formatter.encode_dialog(
            [
                {"role": "user", "slots": {"message": ["TEST", audio]}},
                {"role": "assistant", "slots": {"message": "TEST"}},
            ]
        )


def test_example_prompt_formatter_training(bpe_tokenizer):
    # TODO: remove modality_encoders API?
    formatter = AudioExamplePromptFormatter(
        bpe_tokenizer,
        modality_encoders={
            Modality.Text: torch.nn.Embedding(8, 8),
            Modality.Audio: torch.nn.Conv1d(
                in_channels=1,
                out_channels=80,
                kernel_size=100,
                stride=100,
                padding=0,
            ),
        },
    )
    audio = torch.rand(1, 16000)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"message": ["TEST", audio]}},
            {"role": "assistant", "slots": {"message": "TEST"}},
        ]
    )
    # TODO: replace IDs with float tensors; check only dtype and shape
    assert set(ans) == {"input_ids", "context_ids", "answer_ids", "mask"}
    # fmt: off
    assert ans["input_ids"].tolist() == [21, 8, 7, 54, 42, 49, 30, 50, 1, 81, 20, 30, 54, 72, 42, 49, 30, 50, 1, 81, 20, 30, 66, 8, 7]
    assert ans["context_ids"].tolist() == [21, 8, 7, 54, 42, 49, 30, 50, 1, 81, 20, 30, 54, 72, 42, 49, 30, 50]
    assert ans["answer_ids"].tolist() == [1, 81, 20, 30, 66, 8, 7]
    assert ans["mask"].tolist() == [False] * 18 + [True] * 7
    # fmt: on


def test_example_prompt_formatter_inference(bpe_tokenizer):
    formatter = AudioExamplePromptFormatter(bpe_tokenizer)
    audio = torch.rand(1, 16000)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"message": ["TEST", audio]}},
        ]
    )
    # TODO: replace IDs with float tensors; check only dtype and shape
    assert set(ans) == {"input_ids", "context_ids"}
    # fmt: off
    assert ans["input_ids"].tolist() == ans["context_ids"].tolist()
    assert ans["input_ids"].tolist() == [21, 8, 7, 54, 42, 49, 30, 50, 1, 81, 20, 30, 54, 72, 42, 49, 30, 50]
    # fmt: on


def test_multimodal_prompt_formatter_audio_representation_to_token_ids(bpe_tokenizer):
    formatter = AudioExamplePromptFormatter(bpe_tokenizer)
    B = 4
    ATi = 100  # audio num frames
    TTo = 16  # text num frames
    H = 4  # embedding dim
    audio_input = torch.randn(B, ATi, H)
    text_target = torch.randint(0, 1024, size=(B, TTo))

    for b in range(B):
        ans = formatter.encode_dialog(
            [
                {"role": "user", "slots": {"message": audio_input[b]}},
                {"role": "assistant", "slots": {"message": text_target[b]}},
            ]
        )
        # TODO: replace IDs with float tensors; check only dtype and shape
        assert set(ans) == {"input_ids", "context_ids", "answer_ids", "mask"}
        # fmt: off
        assert ans["input_ids"].tolist() == [21, 8, 7, 54, 42, 49, 30, 50, 1, 81, 20, 30, 54, 72, 42, 49, 30, 50, 1, 81, 20, 30, 66, 8, 7]
        assert ans["context_ids"].tolist() == [21, 8, 7, 54, 42, 49, 30, 50, 1, 81, 20, 30, 54, 72, 42, 49, 30, 50]
        assert ans["answer_ids"].tolist() == [1, 81, 20, 30, 66, 8, 7]
        assert ans["mask"].tolist() == [False] * 18 + [True] * 7
        # fmt: on


def test_multimodal_prompt_formatter_audio_text_representation_to_token_ids(bpe_tokenizer):
    formatter = AudioExamplePromptFormatter(bpe_tokenizer)
    B = 4
    ATi = 100  # audio num frames
    TT0i = 5  # text num frames: text input 0
    TT1i = 21  # text num frames: text input 1
    TTo = 16  # text num frames: text output
    H = 4  # embedding dim
    audio_input = torch.randn(B, ATi, H)
    text_input_0 = torch.randint(0, 1024, size=(B, TT0i))
    text_input_1 = torch.randint(0, 1024, size=(B, TT1i))
    text_target = torch.randint(0, 1024, size=(B, TTo))
    for b in range(B):
        ans = formatter.encode_dialog(
            [
                {"role": "user", "slots": {"message": [text_input_0[b], audio_input[b], text_input_1[b]]}},
                {"role": "assistant", "slots": {"message": text_target[b]}},
            ]
        )
        # TODO: replace IDs with float tensors; check only dtype and shape
        assert set(ans) == {"input_ids", "context_ids", "answer_ids", "mask"}
        # fmt: off
        assert ans["input_ids"].tolist() == [21, 8, 7, 54, 42, 49, 30, 50, 1, 81, 20, 30, 54, 72, 42, 49, 30, 50, 1, 81, 20, 30, 66, 8, 7]
        assert ans["context_ids"].tolist() == [21, 8, 7, 54, 42, 49, 30, 50, 1, 81, 20, 30, 54, 72, 42, 49, 30, 50]
        assert ans["answer_ids"].tolist() == [1, 81, 20, 30, 66, 8, 7]
        assert ans["mask"].tolist() == [False] * 18 + [True] * 7
        # fmt: on


# TODO: considered use-cases
#   * text input/output as str
#   * text input/output as token ids [1D int tensor (T,)]
#   * text input as embeddings [2D float tensor (T, H)]
#   * audio input as embeddings [2D float tensor (T, H)]
#   * audio input as audio codecs [3D float tensor (T, H, Codebook)] -> may be supported as regular 2D embedding instead (user responsibility to apply proper partial audio decoder)
#   * should we support text/audio output representation as embedding?
