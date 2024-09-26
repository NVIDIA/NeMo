import pytest
import torch
from lhotse import CutSet, MonoCut, SupervisionSegment
from lhotse.testing.dummies import DummyManifest, dummy_recording
from torch import tensor

from nemo.collections.common.tokenizers import SentencePieceTokenizer
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import create_spt_model
from nemo.collections.multimodal.speech_llm.data.lhotse_dataset import LhotseAudioQuestionAnswerDataset
from nemo.collections.multimodal.speech_llm.parts.utils.data_utils import TextProcessing


@pytest.fixture
def tokenizer(capsys, tmp_path_factory):
    TOKENIZER_TRAIN_TEXT = """
    Example system message.
    Example user message.
    Example assistant message.
    TEST
    [INST]
    [/INST]
    <s>
    </s>
    <<SYS>>
    <</SYS>>
    User: Assistant:
    user model
    Instruct Output 
    \n\n
    <start_of_turn> <end_of_turn>
    <|
    |>
    <|en|> <|de|> <|fr|> <|es|> <|transcribe|> <|translate|> <|pnc|> <|nopnc|> <|startoftranscript|> <|endoftext|>
    Feel free to add new tokens for your own tests!?
    But know that if you do so, you may need to update the token IDs in the existing tests! 
    So, it might be a good idea to create a new tokenizer instead when adding new prompt formats.
    """
    tmpdir = tmp_path_factory.mktemp("bpe_tokenizer")
    text_path = tmpdir / "text.txt"
    text_path.write_text(TOKENIZER_TRAIN_TEXT)
    with capsys.disabled():
        create_spt_model(str(text_path), vocab_size=512, sample_size=-1, do_lower_case=False, output_dir=str(tmpdir))
    return SentencePieceTokenizer(str(tmpdir / "tokenizer.model"))


@pytest.fixture
def cuts():
    return CutSet(
        [
            MonoCut(
                id="ex0",
                start=0,
                duration=5.0,
                channel=0,
                supervisions=[
                    SupervisionSegment(
                        id="ex0",
                        recording_id="dummy-recording-0000",
                        start=0,
                        duration=5.0,
                        text="some transcription",
                        language="en",
                    )
                ],
                recording=dummy_recording(0, duration=5.0, with_data=True),
                custom={"context": "non default prompt context", "answer": "some desired answer"},
            ),
        ]
    )


def test_speechllm_dataset(tokenizer, cuts):
    text_processor = TextProcessing(
        tokenizer=tokenizer,
    )
    dataset = LhotseAudioQuestionAnswerDataset(
        text_processor=text_processor,
        default_context="do this task",
        tokens_to_generate=128,
        pad_to_max_length=False,
        max_seq_length=64,
    )

    batch = dataset[cuts]
    print(batch)

    expected_keys = {
        "sample_ids",
        "audio_signal",
        "audio_signal_length",
        "audio_ratio",
        "metadata",
        "tokens",
        "tokens_length",
        "labels",
        "loss_mask",
        "position_ids",
        "contexts",
        "context_lengths",
        "max_length",
        "answers",
    }
    missing_keys = expected_keys - set(batch)
    unexpected_keys = set(batch) - expected_keys
    assert not missing_keys and not unexpected_keys, f"{missing_keys=} {unexpected_keys=}"

    assert batch["sample_ids"] == ["ex0"]
    assert batch["metadata"] == [{'audio_filepath': 'ex0.wav'}]
    torch.testing.assert_close(batch["audio_ratio"], tensor([1.0]))
    torch.testing.assert_close(batch["max_length"], tensor([64]))

    assert torch.is_tensor(batch["audio_signal"])
    assert torch.is_floating_point(batch["audio_signal"])
    assert batch["audio_signal"].shape == (1, 80000)
    torch.testing.assert_close(batch["audio_signal_length"], tensor([80000], dtype=torch.int32))

    # fmt: off
    expected = tensor([[  1,  78,   9,   1,  64,  80,   5,  75,  15,   6,   1,  12,  24,  14,
               23,   6,   1,  27,  14,   9,   6,  63,   6,  76,  14,  73,   2,   1,
               56, 100,  41,  14,   9,  -1,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   0,   0,   0,   0,   0]])
    torch.testing.assert_close(batch["tokens"], expected)
    torch.testing.assert_close(batch["tokens_length"], tensor([33]))
    assert tokenizer.ids_to_text(expected[0, :33].tolist()) == "non default prompt context some transcription"

    expected = tensor([[1, 78, 9, 1, 64, 80, 5, 75, 15, 6, 1, 12, 24, 14, 23, 6, 1, 27,
             14, 9, 6, 63, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    torch.testing.assert_close(batch["contexts"], expected)
    torch.testing.assert_close(batch["context_lengths"], tensor([23]))
    assert tokenizer.ids_to_text(expected[0, :23].tolist()) == "non default prompt context"

    expected = tensor([[76, 14, 73, 2, 1, 56, 100, 41, 14, 9, -1, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0]])
    torch.testing.assert_close(batch["answers"], expected)
    assert tokenizer.ids_to_text(expected[0, :10].tolist()) == "some transcription"

    expected = tensor([[78, 9, 1, 64, 80, 5, 75, 15, 6, 1, 12, 24, 14, 23,
             6, 1, 27, 14, 9, 6, 63, 6, 76, 14, 73, 2, 1, 56,
             100, 41, 14, 9, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0]])
    torch.testing.assert_close(batch["labels"], expected)
    assert tokenizer.ids_to_text(expected[0, :32].tolist()) == "non default prompt context some transcription"

    torch.testing.assert_close(
        batch["position_ids"],
        tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]])
    )

    torch.testing.assert_close(
        batch["loss_mask"],
        tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    )
    # fmt: on
