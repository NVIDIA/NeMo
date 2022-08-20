# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from nemo.collections.nlp.data.dialogue.data_processor.assistant_data_processor import DialogueAssistantDataProcessor
from nemo.collections.nlp.data.dialogue.data_processor.data_processor import DialogueDataProcessor
from nemo.collections.nlp.data.dialogue.data_processor.sgd_data_processor import DialogueSGDDataProcessor
from nemo.collections.nlp.data.dialogue.dataset.dialogue_gpt_classification_dataset import (
    DialogueGPTClassificationDataset,
)
from nemo.collections.nlp.data.dialogue.dataset.dialogue_s2s_generation_dataset import DialogueS2SGenerationDataset
from nemo.collections.nlp.data.dialogue.dataset.dialogue_sgd_bert_dataset import DialogueSGDBERTDataset
from nemo.collections.nlp.data.dialogue.dataset.dialogue_zero_shot_slot_filling_dataset import (
    DialogueZeroShotSlotFillingDataset,
)
from nemo.collections.nlp.metrics.dialogue_metrics import DialogueClassificationMetrics, DialogueGenerationMetrics
from nemo.collections.nlp.models.dialogue.dialogue_nearest_neighbour_model import DialogueNearestNeighbourModel
from nemo.collections.nlp.models.dialogue.dialogue_zero_shot_slot_filling_model import DialogueZeroShotSlotFillingModel


@pytest.mark.unit
def test_dialogue_metric_generation_f1():

    generated_field = 'That is so good'
    ground_truth_field = 'That is so awesome'

    precision, recall, f1 = DialogueGenerationMetrics._get_one_f1(generated_field, ground_truth_field)
    assert precision == 75
    assert recall == 75
    assert f1 == 75


@pytest.mark.unit
def test_dialogue_metric_split_label_and_slots():
    fields = ["reserve_restaurant\nslots: time_of_day(7pm), number_of_people(3)", "time_of_day(7pm)"]
    labels, slots_list = DialogueClassificationMetrics.split_label_and_slots(fields, with_slots=True)
    assert labels == ["reserve_restaurant", 'none']
    assert slots_list == [["time_of_day(7pm)", "number_of_people(3)"], ["time_of_day(7pm)"]]


@pytest.mark.unit
def test_dialogue_metric_slot_filling_metrics():
    generated_slots = [["time_of_day(7pm)", "number_of_people(3)"], ["time_of_day(7pm)"]]
    ground_truth_slots = [["time_of_day(7pm)"], ["time_of_day(7pm)", "number_of_people(3)"]]

    (
        avg_precision,
        avg_recall,
        avg_f1,
        avg_joint_goal_accuracy,
    ) = DialogueClassificationMetrics.get_slot_filling_metrics(generated_slots, ground_truth_slots)

    assert avg_precision == 75
    assert avg_recall == 75
    assert avg_f1 == 75
    assert avg_joint_goal_accuracy == 0


@pytest.mark.unit
def test_dialogue_assistant_data_processor_normalize_zero_shot_intent():
    label0 = 'food_ordering.contextual_query'
    normalized_label0 = 'contextual query'

    label1 = 'food_ordering.nomatch'
    normalized_label1 = 'no match'

    label2 = 'food_ordering.no'
    normalized_label2 = 'no'

    assert normalized_label0 == DialogueAssistantDataProcessor.normalize_zero_shot_intent(label0)
    assert normalized_label1 == DialogueAssistantDataProcessor.normalize_zero_shot_intent(label1)
    assert normalized_label2 == DialogueAssistantDataProcessor.normalize_zero_shot_intent(label2)


@pytest.mark.unit
def test_dialogue_assistant_data_processor_get_continuous_slots():
    slot_ids = [54, 54, 54, 19, 19, 18, 54, 54, 54]
    empty_slot_id = 54
    bio_slot_ids_to_unified_slot_ids = {18: 18, 19: 19, 54: 54}
    continuous_slots = DialogueAssistantDataProcessor.get_continuous_slots(
        slot_ids, empty_slot_id, bio_slot_ids_to_unified_slot_ids
    )
    assert continuous_slots == {19: [3, 5], 18: [5, 6]}

    # here 18 and 19 maps to the same slot (originally variants of B-slot and I-slot)
    slot_ids = [54, 54, 54, 19, 19, 18, 54, 54, 54]
    empty_slot_id = 54
    bio_slot_ids_to_unified_slot_ids = {18: 18, 19: 18, 54: 54}
    continuous_slots = DialogueAssistantDataProcessor.get_continuous_slots(
        slot_ids, empty_slot_id, bio_slot_ids_to_unified_slot_ids
    )
    assert continuous_slots == {18: [3, 6]}

    # test if function works when non-empty slots are at boundary
    slot_ids = [18, 54, 54, 19, 19]
    empty_slot_id = 54
    bio_slot_ids_to_unified_slot_ids = {18: 18, 19: 19, 54: 54}
    continuous_slots = DialogueAssistantDataProcessor.get_continuous_slots(
        slot_ids, empty_slot_id, bio_slot_ids_to_unified_slot_ids
    )
    assert continuous_slots == {18: [0, 1], 19: [3, 5]}


@pytest.mark.unit
def test_dialogue_assistant_map_bio_format_slots_to_unified_slots():

    slots = ['B-time', 'I-time', 'B-alarm', 'I-alarm', 'O']
    gt_bio_slot_ids_to_unified_slot_ids = {'0': '0', '1': '0', '2': '1', '3': '1', '4': '2'}
    gt_unified_slots = ['time', 'alarm', 'O']
    (
        bio_slot_ids_to_unified_slot_ids,
        unified_slots,
    ) = DialogueAssistantDataProcessor.map_bio_format_slots_to_unified_slots(slots)
    assert gt_bio_slot_ids_to_unified_slot_ids == bio_slot_ids_to_unified_slot_ids
    assert gt_unified_slots == unified_slots

    # case in which BIOS scheme was not used in annotation
    slots = ['time', 'alarm', 'O']
    gt_bio_slot_ids_to_unified_slot_ids = {'0': '0', '1': '1', '2': '2'}
    gt_unified_slots = ['time', 'alarm', 'O']
    (
        bio_slot_ids_to_unified_slot_ids,
        unified_slots,
    ) = DialogueAssistantDataProcessor.map_bio_format_slots_to_unified_slots(slots)

    assert gt_bio_slot_ids_to_unified_slot_ids == bio_slot_ids_to_unified_slot_ids
    assert gt_unified_slots == unified_slots


@pytest.mark.unit
def test_dialogue_data_processor_get_relevant_idxs():

    dataset_split = 'train'
    dev_proportion = 10
    n_samples = 1000
    idxs = DialogueDataProcessor.get_relevant_idxs(dataset_split, n_samples, dev_proportion)

    assert len(idxs) == 900
    assert idxs != list(range(900))

    dataset_split = 'dev'
    dev_proportion = 40
    n_samples = 1000
    idxs = DialogueDataProcessor.get_relevant_idxs(dataset_split, n_samples, dev_proportion)

    assert len(idxs) == 400
    assert idxs != list(range(400))

    dataset_split = 'test'
    dev_proportion = 40
    n_samples = 1000
    idxs = DialogueDataProcessor.get_relevant_idxs(dataset_split, n_samples, dev_proportion)

    assert len(idxs) == 1000
    assert idxs == list(range(1000))


@pytest.mark.unit
def test_dialogue_sgd_data_processor_convert_camelcase_to_lower():
    label = 'none'
    gt_converted_label = 'none'

    assert gt_converted_label == DialogueSGDDataProcessor.convert_camelcase_to_lower(label)

    label = 'ReserveRestaurant'
    gt_converted_label = 'reserve restaurant'

    assert gt_converted_label == DialogueSGDDataProcessor.convert_camelcase_to_lower(label)

    label = 'Alarm'
    gt_converted_label = 'alarm'

    assert gt_converted_label == DialogueSGDDataProcessor.convert_camelcase_to_lower(label)


@pytest.mark.unit
def test_dialogue_gpt_classification_dataset_linearize_slots():

    slots = []
    linearized_slots = 'None'
    assert linearized_slots == DialogueGPTClassificationDataset.linearize_slots(slots)

    slots = {'time': '7pm', 'place': 'field'}
    linearized_slots = 'time(7pm), place(field)'
    assert linearized_slots == DialogueGPTClassificationDataset.linearize_slots(slots)

    slots = {'time': ['7pm', '1900'], 'place': 'field'}
    linearized_slots = 'time(7pm), place(field)'
    assert linearized_slots == DialogueGPTClassificationDataset.linearize_slots(slots)


@pytest.mark.unit
def test_dialogue_gpt_classification_dataset_linearize_slots():

    actions = [
        {'act': 'inform', 'slot': 'time', 'values': ['7pm', '1900']},
        {'act': 'confirm', 'slot': 'place', 'values': ['hall']},
    ]

    prompt_template = 'values'
    formatted_actions = '7pm hall'
    assert formatted_actions == DialogueS2SGenerationDataset.format_actions(prompt_template, actions)

    prompt_template = 'slots_values'
    formatted_actions = 'time (7pm) place (hall)'
    assert formatted_actions == DialogueS2SGenerationDataset.format_actions(prompt_template, actions)

    prompt_template = 'acts_slots_values'
    formatted_actions = 'inform time (7pm) confirm place (hall)'
    assert formatted_actions == DialogueS2SGenerationDataset.format_actions(prompt_template, actions)


@pytest.mark.unit
def test_dialogue_sgd_dataset_naive_tokenize():

    utterance = 'I am feeling hungry so I would like to find a place to eat.'
    tokens = [
        'I',
        ' ',
        'am',
        ' ',
        'feeling',
        ' ',
        'hungry',
        ' ',
        'so',
        ' ',
        'I',
        ' ',
        'would',
        ' ',
        'like',
        ' ',
        'to',
        ' ',
        'find',
        ' ',
        'a',
        ' ',
        'place',
        ' ',
        'to',
        ' ',
        'eat',
        '.',
    ]
    assert tokens == DialogueSGDBERTDataset._naive_tokenize(utterance)


@pytest.mark.unit
def test_dialogue_nearest_neighbour_mean_pooling():

    model_output = [torch.ones(8, 512, 768)]
    attention_mask = torch.ones(8, 512)
    assert torch.equal(
        torch.ones(8, 768).float(), DialogueNearestNeighbourModel.mean_pooling(model_output, attention_mask)
    )

    model_output = [torch.zeros(8, 512, 768)]
    attention_mask = torch.ones(8, 512)
    assert torch.equal(
        torch.zeros(8, 768).float(), DialogueNearestNeighbourModel.mean_pooling(model_output, attention_mask)
    )

    model_output = [torch.cat([torch.zeros(8, 256, 768), torch.ones(8, 256, 768)], axis=1)]
    attention_mask = torch.ones(8, 512)
    assert torch.equal(
        torch.ones(8, 768).float() * 0.5, DialogueNearestNeighbourModel.mean_pooling(model_output, attention_mask)
    )


@pytest.mark.unit
def test_dialogue_get_entity_embedding_from_hidden_states():
    # sentence start with an entity [1, ...]
    # entity follow by a different entity [1, 1, 1]
    # an empty entity follow by a B-entity and I-entity [0, 1, 2]
    # sentence end with an entity [..., 1]
    mention_mask = torch.FloatTensor([[1, 1, 1, 0, 1, 2, 0, 1, 1]])
    hidden_states = torch.ones((1, 9, 3))
    expected_output = torch.FloatTensor(
        [[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]]
    )
    actual_output = DialogueZeroShotSlotFillingModel.get_entity_embedding_from_hidden_states(
        mention_mask, hidden_states
    )
    assert torch.equal(expected_output, actual_output)

    # sentence start with an empty entity [0, ...]
    # an (B-entity, I-entity) follow by a (B-entity, I-entity) [1, 2, 1, 2]
    # an entity has loger I-entity [1, 2, 2, 2]
    # sentence end with an empty entity [..., 0]
    mention_mask = torch.FloatTensor([[0, 1, 2, 1, 2, 2, 0, 0]])
    hidden_states = torch.ones((1, 8, 3))
    expected_output = torch.FloatTensor(
        [[[1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]]
    )
    actual_output = DialogueZeroShotSlotFillingModel.get_entity_embedding_from_hidden_states(
        mention_mask, hidden_states
    )
    assert torch.equal(expected_output, actual_output)

    # longer empty entity [0, 0, 0]
    # sentence end with an I-entity [..., 2]
    mention_mask = torch.FloatTensor([[1, 0, 0, 0, 1, 1, 2, 2, 2]])
    hidden_states = torch.ones((1, 9, 3))
    expected_output = torch.FloatTensor(
        [[[1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]]
    )
    actual_output = DialogueZeroShotSlotFillingModel.get_entity_embedding_from_hidden_states(
        mention_mask, hidden_states
    )
    assert torch.equal(expected_output, actual_output)


@pytest.mark.unit
def test_dialogue_bert_dataset_get_bio_slot_label_from_sequence():
    # Test label_id_for_empty_slot to be 54, and 0 is an entity class, with PAD
    slot_label_list = [54, 54, 54, 0, 0, 54, 54, 46, 46, 12, 48, -1, -1]
    label_id_for_empty_slot = 54
    expected_output = [0, 0, 0, 1, 2, 0, 0, 1, 2, 1, 1, 0, 0]
    actual_output = DialogueZeroShotSlotFillingDataset.get_bio_slot_label_from_sequence(
        slot_label_list, label_id_for_empty_slot
    )
    assert expected_output == actual_output

    # Test label_id_for_empty_slot to be 0, with PAD
    slot_label_list = [0, 0, 0, 0, 0, 6, 3, 3, 0, 2, 2, -1, -1]
    label_id_for_empty_slot = 0
    expected_output = [0, 0, 0, 0, 0, 1, 1, 2, 0, 1, 2, 0, 0]
    actual_output = DialogueZeroShotSlotFillingDataset.get_bio_slot_label_from_sequence(
        slot_label_list, label_id_for_empty_slot
    )
    assert expected_output == actual_output

    # Test label_id_for_empty_slot to be 54,without PAD
    # sentence start with an entity [0, ...]
    # entity follow by a different entity [0, 1, 2]
    # an empty entity follow by a B-entity and I-entity [54, 46, 46]
    # sentence end with an entity [..., 15]
    slot_label_list = [0, 1, 2, 54, 46, 46, 54, 12, 15]
    label_id_for_empty_slot = 54
    expected_output = [1, 1, 1, 0, 1, 2, 0, 1, 1]
    actual_output = DialogueZeroShotSlotFillingDataset.get_bio_slot_label_from_sequence(
        slot_label_list, label_id_for_empty_slot
    )
    assert expected_output == actual_output

    # Test label_id_for_empty_slot to be 54, with PAD
    # sentence start with an empty entity [54, ...]
    # an (B-entity, I-entity) follow by a (B-entity, I-entity) [0, 0, 46, 46]
    # an entity has loger I-entity [46, 46, 46]
    # sentence end with an empty entity [..., 54]
    slot_label_list = [54, 0, 0, 46, 46, 46, 54, 54, -1, -1]
    label_id_for_empty_slot = 54
    expected_output = [0, 1, 2, 1, 2, 2, 0, 0, 0, 0]
    actual_output = DialogueZeroShotSlotFillingDataset.get_bio_slot_label_from_sequence(
        slot_label_list, label_id_for_empty_slot
    )
    assert expected_output == actual_output

    # Test label_id_for_empty_slot to be 0, without PAD
    # longer empty entity [0, 0, 0]
    # sentence end with an I-entity [..., 10]
    slot_label_list = [6, 0, 0, 0, 9, 10, 10, 10, 10]
    label_id_for_empty_slot = 0
    expected_output = [1, 0, 0, 0, 1, 1, 2, 2, 2]
    actual_output = DialogueZeroShotSlotFillingDataset.get_bio_slot_label_from_sequence(
        slot_label_list, label_id_for_empty_slot
    )
    assert expected_output == actual_output


@pytest.mark.unit
def test_dialogue_bert_dataset_get_bio_mention_labels():
    # Test label_id_for_empty_slot to be 54, without PAD
    slot_list = [54, 54, 54, 0, 0, 54, 54, 46, 46, 12, 48]
    bio_list = [0, 0, 0, 1, 2, 0, 0, 1, 2, 1, 1]
    expected_output = [0, 46, 12, 48, -1, -1, -1, -1, -1, -1, -1]
    actual_output = DialogueZeroShotSlotFillingDataset.get_bio_mention_labels(slot_list, bio_list)
    assert expected_output == actual_output

    # Test label_id_for_empty_slot to be 0, with PAD
    slot_list = [0, 0, 0, 0, 0, 6, 3, 3, 0, 2, 2, -1, -1]
    bio_list = [0, 0, 0, 0, 0, 1, 1, 2, 0, 1, 2, 0, 0]
    expected_output = [6, 3, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    actual_output = DialogueZeroShotSlotFillingDataset.get_bio_mention_labels(slot_list, bio_list)
    assert expected_output == actual_output

    # Test label_id_for_empty_slot to be 54,without PAD
    # sentence start with an entity [0, ...]
    # entity follow by a different entity [0, 1, 2]
    # an empty entity follow by a B-entity and I-entity [54, 46, 46]
    # sentence end with an entity [..., 15]
    slot_list = [0, 1, 2, 54, 46, 46, 54, 12, 15]
    bio_list = [1, 1, 1, 0, 1, 2, 0, 1, 1]
    expected_output = [0, 1, 2, 46, 12, 15, -1, -1, -1]
    actual_output = DialogueZeroShotSlotFillingDataset.get_bio_mention_labels(slot_list, bio_list)
    assert expected_output == actual_output

    # Test label_id_for_empty_slot to be 54, with PAD
    # sentence start with an empty entity [54, ...]
    # an (B-entity, I-entity) follow by a (B-entity, I-entity) [0, 0, 46, 46]
    # an entity has loger I-entity [46, 46, 46]
    # sentence end with an empty entity [..., 54]
    slot_list = [54, 0, 0, 46, 46, 46, 54, 54, -1, -1]
    bio_list = [0, 1, 2, 1, 2, 2, 0, 0, 0, 0]
    expected_output = [0, 46, -1, -1, -1, -1, -1, -1, -1, -1]
    actual_output = DialogueZeroShotSlotFillingDataset.get_bio_mention_labels(slot_list, bio_list)
    assert expected_output == actual_output

    # Test label_id_for_empty_slot to be 0, without PAD
    # longer empty entity [0, 0, 0]
    # sentence end with an I-entity [..., 10]
    slot_list = [6, 0, 0, 0, 9, 10, 10, 10, 10]
    bio_list = [1, 0, 0, 0, 1, 1, 2, 2, 2]
    expected_output = [6, 9, 10, -1, -1, -1, -1, -1, -1]
    actual_output = DialogueZeroShotSlotFillingDataset.get_bio_mention_labels(slot_list, bio_list)
    assert expected_output == actual_output


@pytest.mark.unit
def test_align_mention_to_tokens():
    bio_labels = torch.FloatTensor([[1, 0, 0, 1, 2]])
    mention_labels = torch.FloatTensor([[2, 4, 0, 0, 0]])
    expected_output = torch.FloatTensor([[2, 0, 0, 4, 4]])
    actual_output = DialogueZeroShotSlotFillingModel.align_mention_to_tokens(bio_labels, mention_labels)
    assert torch.equal(expected_output, actual_output)

    bio_labels = torch.FloatTensor([[1, 2, 2, 0, 0]])
    mention_labels = torch.FloatTensor([[52, 0, 0, 0, 0]])
    expected_output = torch.FloatTensor([[52, 52, 52, 0, 0]])
    actual_output = DialogueZeroShotSlotFillingModel.align_mention_to_tokens(bio_labels, mention_labels)
    assert torch.equal(expected_output, actual_output)

    bio_labels = torch.FloatTensor([[2, 2, 1]])
    mention_labels = torch.FloatTensor([[3, 0, 0]])
    expected_output = torch.FloatTensor([[0, 0, 3]])
    actual_output = DialogueZeroShotSlotFillingModel.align_mention_to_tokens(bio_labels, mention_labels)
    assert torch.equal(expected_output, actual_output)

    bio_labels = torch.FloatTensor([[1, 0, 2, 2, 1]])
    mention_labels = torch.FloatTensor([[2, 4, 0, 0, 0]])
    expected_output = torch.FloatTensor([[2, 0, 0, 0, 4]])
    actual_output = DialogueZeroShotSlotFillingModel.align_mention_to_tokens(bio_labels, mention_labels)
    assert torch.equal(expected_output, actual_output)

    bio_labels = torch.FloatTensor([[0, 0, 2, 2, 1]])
    mention_labels = torch.FloatTensor([[3, 0, 0, 0, 0]])
    expected_output = torch.FloatTensor([[0, 0, 0, 0, 3]])
    actual_output = DialogueZeroShotSlotFillingModel.align_mention_to_tokens(bio_labels, mention_labels)
    assert torch.equal(expected_output, actual_output)

    bio_labels = torch.FloatTensor([[1, 1, 2, 2, 0]])
    mention_labels = torch.FloatTensor([[0, 8, 0, 0, 0]])
    expected_output = torch.FloatTensor([[0, 8, 8, 8, 0]])
    actual_output = DialogueZeroShotSlotFillingModel.align_mention_to_tokens(bio_labels, mention_labels)
    assert torch.equal(expected_output, actual_output)


@pytest.mark.unit
def test_get_start_and_end_for_bio():
    bio_labels = torch.FloatTensor([1, 0, 0, 1, 2])
    expected_output = [[0, 1], [3, 5]]
    actual_output = DialogueZeroShotSlotFillingModel.get_start_and_end_for_bio(bio_labels)
    assert expected_output == actual_output

    bio_labels = torch.FloatTensor([1, 2, 2, 0, 0])
    expected_output = [[0, 3]]
    actual_output = DialogueZeroShotSlotFillingModel.get_start_and_end_for_bio(bio_labels)
    assert expected_output == actual_output

    bio_labels = torch.FloatTensor([2, 2, 1])
    expected_output = [[2, 3]]
    actual_output = DialogueZeroShotSlotFillingModel.get_start_and_end_for_bio(bio_labels)
    assert expected_output == actual_output

    bio_labels = torch.FloatTensor([1, 0, 2, 2, 1])
    expected_output = [[0, 1], [4, 5]]
    actual_output = DialogueZeroShotSlotFillingModel.get_start_and_end_for_bio(bio_labels)
    assert expected_output == actual_output

    bio_labels = torch.FloatTensor([0, 0, 2, 2, 1])
    expected_output = [[4, 5]]
    actual_output = DialogueZeroShotSlotFillingModel.get_start_and_end_for_bio(bio_labels)
    assert expected_output == actual_output

    bio_labels = torch.FloatTensor([1, 1, 2, 2, 0])
    expected_output = [[0, 1], [1, 4]]
    actual_output = DialogueZeroShotSlotFillingModel.get_start_and_end_for_bio(bio_labels)
    assert expected_output == actual_output
