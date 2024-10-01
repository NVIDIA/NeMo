import json
from itertools import islice

import lhotse
import pytest
import torch
from lhotse.testing.dummies import dummy_cut, dummy_recording
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.data.lhotse.text_adapters import AudioTurn, NeMoMultimodalConversation, TextTurn


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
    assert len(ex.turns) == 5
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
