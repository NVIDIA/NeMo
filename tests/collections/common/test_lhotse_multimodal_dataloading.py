import json
from itertools import islice

import lhotse
import pytest
import torch
from lhotse.testing.dummies import dummy_cut, dummy_recording
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.data.lhotse.text_adapters import AudioTurn, NeMoMultimodalConversation, TextTurn
from nemo.collections.common.tokenizers.aggregate_tokenizer import TokenizerWrapper
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer, create_spt_model


class Identity(torch.utils.data.Dataset):
    def __getitem__(self, cuts: lhotse.CutSet) -> lhotse.CutSet:
        return cuts


@pytest.fixture(scope="session")
def multimodal_conversations_path(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("text_data")
    en_path = tmp_path / "manifest.json"
    data = [
        {
            "id": "convo_1",
            "conversations": [
                {
                    "value": "Can you help summarize the following?",
                    "from": "User",
                    "type": "text",
                },
                {
                    "value": "123.wav",
                    "from": "User",
                    "type": "audio",
                    "duration": 5.73,
                },
                {
                    "value": "I'm glad to assist you with your request. Here's a summary:",
                    "from": "Assistant",
                    "type": "text",
                },
                {
                    "value": "123_answer.wav",
                    "from": "Assistant",
                    "type": "audio",
                    "duration": 7.11,
                },
                {
                    "value": "Can you further shorten it?",
                    "from": "User",
                    "type": "text",
                },
                {
                    "value": "Of course!",
                    "from": "Assistant",
                    "type": "text",
                },
            ],
        }
    ]
    lhotse.serialization.save_to_jsonl(data, en_path)
    dummy_recording(0, 5.73, with_data=True).to_cut().save_audio(tmp_path / "123.wav")
    dummy_recording(0, 7.11, with_data=True).to_cut().save_audio(tmp_path / "123_answer.wav")
    return en_path


def test_multimodal_conversation_input(multimodal_conversations_path):

    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "multimodal_conversation",
                    "manifest_filepath": multimodal_conversations_path,
                    "audio_locator_tag": "[audio]",
                },
            ],
            "force_finite": True,
            "shuffle": True,
            "num_workers": 0,
            "batch_size": 1,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    # Note: this test does not need to pass a tokenizer because we use static batch sizes
    dl = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=Identity())
    batches = [batch for batch in dl]
    assert len(batches) == 1

    b = batches[0]
    assert isinstance(b, lhotse.CutSet)
    assert len(b) == 1
    ex = b[0]
    assert isinstance(ex, NeMoMultimodalConversation)
    assert ex.id == "convo_1"
    assert len(ex.turns) == 6
    t = ex.turns[0]
    assert isinstance(t, TextTurn)
    assert t.role == "user"
    assert t.value == "Can you help summarize the following?"
    t = ex.turns[1]
    assert isinstance(t, AudioTurn)
    assert t.role == "user"
    assert t.audio_locator_tag == "[audio]"
    assert t.cut.duration == 5.73
    assert t.cut.load_audio().shape == (1, 91680)
    t = ex.turns[2]
    assert isinstance(t, TextTurn)
    assert t.role == "assistant"
    assert t.value == "I'm glad to assist you with your request. Here's a summary:"
    t = ex.turns[3]
    assert isinstance(t, AudioTurn)
    assert t.role == "assistant"
    assert t.audio_locator_tag == "[audio]"
    assert t.cut.duration == 7.11
    assert t.cut.load_audio().shape == (1, 113760)
    t = ex.turns[4]
    assert isinstance(t, TextTurn)
    assert t.role == "user"
    assert t.value == "Can you further shorten it?"
    t = ex.turns[5]
    assert isinstance(t, TextTurn)
    assert t.role == "assistant"
    assert t.value == "Of course!"
    for key in ("input_ids", "context_ids", "answer_ids", "mask"):
        assert getattr(ex, key) is None  # not tokenized/prompted


@pytest.fixture
def tokenizer(tmp_path_factory, multimodal_conversations_path):
    tmpdir = tmp_path_factory.mktemp("multi_convo_tokenizer")
    text_path = tmpdir / "text.txt"
    text_path.write_text(
        "\n".join(
            turn["value"]
            for item in lhotse.serialization.load_jsonl(multimodal_conversations_path)
            for turn in item["conversations"]
        )
    )
    create_spt_model(
        text_path,
        vocab_size=128,
        sample_size=-1,
        do_lower_case=False,
        output_dir=str(tmpdir),
        bos=True,
        eos=True,
        user_defined_symbols=["[INST]", "[/INST]", "<<SYS>>", "<</SYS>>", "[audio]"],
    )
    return SentencePieceTokenizer(str(tmpdir / "tokenizer.model"))


def test_multimodal_conversation_input_with_prompt(multimodal_conversations_path, tokenizer):

    config = OmegaConf.create(
        {
            "input_cfg": [
                {
                    "type": "multimodal_conversation",
                    "manifest_filepath": multimodal_conversations_path,
                    "audio_locator_tag": "[audio]",
                },
            ],
            "prompt_format": "llama2",
            "force_finite": True,
            "shuffle": True,
            "num_workers": 0,
            "batch_size": 1,
            "seed": 0,
            "shard_seed": 0,
        }
    )

    dl = get_lhotse_dataloader_from_config(
        config=config, global_rank=0, world_size=1, dataset=Identity(), tokenizer=tokenizer
    )
    batches = [batch for batch in dl]
    assert len(batches) == 1

    b = batches[0]
    assert isinstance(b, lhotse.CutSet)
    assert len(b) == 1
    ex = b[0]
    assert isinstance(ex, NeMoMultimodalConversation)

    assert torch.is_tensor(ex.input_ids)
    assert ex.input_ids.shape == (105,)
    assert (
        tokenizer.ids_to_text(ex.input_ids)
        == "[INST] Can you help summarize the following? [audio] [/INST] I'm glad to assist you with your request. Here's a summary: [audio] [INST] Can you further shorten it? [/INST] Of course!"
    )

    assert torch.is_tensor(ex.context_ids)
    assert ex.context_ids.shape == (95,)
    assert (
        tokenizer.ids_to_text(ex.context_ids)
        == "[INST] Can you help summarize the following? [audio] [/INST] I'm glad to assist you with your request. Here's a summary: [audio] [INST] Can you further shorten it? [/INST]"
    )

    assert torch.is_tensor(ex.answer_ids)
    assert ex.answer_ids.shape == (10,)
    assert tokenizer.ids_to_text(ex.answer_ids) == "Of course!"

    assert torch.is_tensor(ex.mask)
    assert ex.mask.shape == (105,)
    assert (ex.mask[:30] == False).all()  # user turn
    assert (ex.mask[30:72] == True).all()  # assistant turn
    assert (ex.mask[72:95] == False).all()  # user turn
    assert (ex.mask[95:] == True).all()  # assistant turn
