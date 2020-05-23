# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
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
# =============================================================================

import argparse
import copy
import json
import os
import random
import re
from collections import defaultdict
from pprint import pprint

import inflect
import numpy as np
from tqdm import tqdm

p = inflect.engine()


def get_ontology(dialogues, schemas):
    """
    creates ontology: 
        (service_name, slot_name) -> 
            -> is_categorical -> True/False
            -> possible_values -> set of values
    """
    ontology = defaultdict(defaultdict)
    for schema in schemas:
        service_name = schema['service_name']
        for slot in schema['slots']:
            slot_name = slot['name']
            ontology[(service_name, slot_name)]["is_categorical"] = slot['is_categorical']
            ontology[(service_name, slot_name)]["possible_values"] = set(slot['possible_values'])

    for dialogue in dialogues:
        for turn in dialogue["turns"]:
            for frame in turn["frames"]:
                service_name = frame["service"]
                if "state" in frame:
                    for k, vs in frame["state"]["slot_values"].items():
                        for v in vs:
                            ontology[(service_name, k)]["possible_values"].add(v)
                if "actions" in frame:
                    for action in frame["actions"]:
                        k = action["slot"]
                        for v in action["values"]:
                            if (service_name, k) in ontology:
                                # some slots like 'count' are not in schema
                                ontology[(service_name, k)]["possible_values"].add(v)
    return ontology


def get_affected_future_frames(dialogue, from_turn_id, slot_name, slot_value, service):
    """
    determine for all turns starting from from_turn_id if they contain the given combination of slot_name, slot_value, service
    if so, return affected List[(turn_id, frame_id, slot_name)]
    """
    assert isinstance(from_turn_id, int)
    assert isinstance(slot_name, str)
    assert isinstance(slot_value, str)
    assert isinstance(service, str)
    res = []
    for turn_id, turn in enumerate(dialogue["turns"][from_turn_id:], start=from_turn_id):
        for frame_id, frame in enumerate(turn["frames"]):
            if turn["speaker"] == "SYSTEM":
                if frame["service"] == service:
                    for action in frame["actions"]:
                        if action["slot"] == slot_name and slot_value in action["values"]:
                            res.append((turn_id, frame_id, slot_name))
                            continue
            else:
                if frame["service"] == service and slot_value in frame["state"]["slot_values"].get(slot_name, []):
                    res.append((turn_id, frame_id, slot_name))
                    continue
    return res


def augment_dialog_by_auxiliary_entries(dialogue):
    """
    augments dialogue by slot_to_span and state_update.
    slot_to_span (dict): slotname-> value-> [start_idx, end_idx] for all values in turn that appear exactly once in utterance. 
    state_update (dict): slotname-> [(turn_id, frame_id, slot_name)] only contains newly introduced slotnames. 
        New for system are all slots in "actions".
        New for user are all slots who did not appear in previous turn or whose (list of) value has changed.
    Returns list of following affected turns/frames.  

    """
    prev_service_user = ""
    prev_state_slots_user = {}  # key, value
    for turn_id, turn in enumerate(dialogue["turns"]):
        for frame in turn["frames"]:
            slot_to_spans = defaultdict(dict)
            for slot in frame["slots"]:
                k = slot["slot"]
                start_idx, end_idx = slot["start"], slot["exclusive_end"]
                slot_to_spans[k][turn["utterance"][start_idx:end_idx]] = [start_idx, end_idx]
            frame["slot_to_span"] = slot_to_spans

        if turn["speaker"] == "SYSTEM":
            for frame in turn["frames"]:
                new_slots = defaultdict(list)
                for action in frame["actions"]:
                    slot = action["slot"]
                    slot_values = action["values"]
                    for v in slot_values:
                        new_slots[slot] = get_affected_future_frames(
                            dialogue, turn_id + 1, slot_name=slot, slot_value=v, service=frame["service"]
                        )
                        if v in turn["utterance"]:
                            if slot not in frame["slot_to_span"] or v not in frame["slot_to_span"][slot]:
                                if len(turn["utterance"].split(v)) == 2:
                                    start_idx = turn["utterance"].index(v)
                                    end_idx = start_idx + len(v)
                                    frame["slot_to_span"][slot][v] = [start_idx, end_idx]
                frame["state_update"] = new_slots
        else:
            for frame in turn["frames"]:
                new_slots = defaultdict(list)  # map from slot_value -> List[frames] in future
                for k, vs in frame["state"]["slot_values"].items():
                    for v_id, v in enumerate(vs):
                        if v in turn["utterance"]:
                            if k not in frame["slot_to_span"] or v not in frame["slot_to_span"][k]:
                                if len(turn["utterance"].split(v)) == 2:
                                    start_idx = turn["utterance"].index(v)
                                    end_idx = start_idx + len(v)
                                    frame["slot_to_span"][k][v] = [start_idx, end_idx]
                        if k not in prev_state_slots_user or v not in prev_state_slots_user[k]:
                            new_slots[k] = get_affected_future_frames(
                                dialogue, turn_id + 1, slot_name=k, slot_value=v, service=frame["service"]
                            )
                frame["state_update"] = new_slots

            if len(turn["frames"]) == 1:
                use_frame = turn["frames"][0]
            else:
                use_frame = [frame for frame in turn["frames"] if frame["service"] != prev_service_user][0]
            prev_service_user = use_frame["service"]
            prev_state_slots_user = use_frame["state"]["slot_values"]


def validate(dialogue):
    """
    check if dialogue is valid wrt to non categorical slots:
        -check if span indices are within utterance length
        -check if utterance substring (by span) is found among values in system action
        -check if utterance substring (by span) is found among values in user state->slot_values->key
    Otherwise raise error with turn id and frame id
    """
    for turn_id, turn in enumerate(dialogue["turns"]):
        for frame_id, frame in enumerate(turn["frames"]):
            for slot in frame["slots"]:
                try:
                    st_idx, end_idx, key = slot["start"], slot["exclusive_end"], slot["slot"]
                    word = turn["utterance"][st_idx:end_idx]
                    assert 0 <= st_idx < end_idx <= len(turn["utterance"])
                    if turn["speaker"] == "SYSTEM":
                        found_key = False
                        for action in frame["actions"]:
                            if action["slot"] == key:
                                if word in action["values"]:
                                    found_key = True
                        assert found_key
                    else:
                        if key in frame["state"]["slot_values"]:
                            assert word in frame["state"]["slot_values"][key]
                except Exception:
                    raise ValueError(f"Turn {turn_id}, frame {frame_id}")


def process_dialogues(final_dialogues, dialogue_count, dialogues, replace_turn_prob, replace_word_prob, new_val_func):
    """
    iterates through all dialogues and does replacement according to new_val_func
    writes out into final_dialogues.
    """
    replace_success = 0
    replace_failed = 0
    for dialogue_id, dialogue in tqdm(enumerate(dialogues)):
        d_id, d_count = dialogue["dialogue_id"].split("_")
        d_id = int(d_id)
        dialogue["dialogue_id"] = f"{d_id}_{dialogue_count[d_id]:05d}"
        dialogue_count[d_id] += 1
        for turn_id, turn in enumerate(dialogue["turns"]):
            if random.random() < replace_turn_prob:
                spans = get_sentence_components(turn=turn)
                for span in reversed(spans):
                    if random.random() < replace_word_prob:
                        old_value = dialogue["turns"][turn_id]["utterance"][span[0] : span[1]]
                        new_value = new_val_func(dialogue, turn_id, old_value, span[0], span[1])
                        if new_value:
                            tmp_dialogue = copy.deepcopy(dialogue)
                            try:
                                replace(tmp_dialogue, turn_id, span[0], span[1], new_value)
                                validate(tmp_dialogue)
                                for k, v in tmp_dialogue.items():
                                    dialogue[k] = v
                                replace_success += 1
                            except Exception:
                                replace_failed += 1

        for turn in dialogue["turns"]:
            for frame in turn["frames"]:
                if 'state_update' in frame:
                    frame.pop("state_update")
                if 'slot_to_span' in frame:
                    frame.pop("slot_to_span")
        final_dialogues[d_id].append(dialogue)
    print(f"Replacement success {replace_success}, failed {replace_failed}\n")


def update_spans(dialogue, turn_id, frame_id, start_idx, end_idx, old_value, new_value):
    """
    update slot spans and slot_to_span
    """
    frame = dialogue["turns"][turn_id]["frames"][frame_id]
    offset = len(new_value) - len(old_value)

    for slot in frame['slots']:
        if start_idx < slot['start']:
            slot['start'] += offset
        if start_idx < slot['exclusive_end']:
            slot['exclusive_end'] += offset

    for k, vs in frame['slot_to_span'].items():
        for v, spans in vs.items():
            if start_idx < spans[0]:
                spans[0] += offset
            if start_idx < spans[1]:
                spans[1] += offset


def update_values(dialogue, turn_id, frame_id, key, old_value, new_value):
    """
    only update values: actions, state, slot_to_span
    """
    frame = dialogue["turns"][turn_id]["frames"][frame_id]
    if "actions" in frame:
        for action in frame["actions"]:
            if key == action["slot"] and old_value in action["values"]:
                action["values"].remove(old_value)
                action["values"].append(new_value)
    if "state" in frame:
        for k, vs in frame["state"]["slot_values"].items():
            for v_id, v in enumerate(vs):
                if k == key and v == old_value:
                    vs[v_id] = new_value

    for k, vs in frame["slot_to_span"].items():
        for v, spans in list(vs.items()):
            if k == key and v == old_value:
                vs.pop(v)
                vs[new_value] = spans


def get_sentence_components(turn):
    """
    return list of start and end indices of slot values/ words that appear in utterance
    """
    sentence = turn["utterance"]
    word_indices = np.asarray([False for _ in range(len(sentence) + 1)])
    for frame in turn["frames"]:
        if "state" in frame:
            for k, vs in frame["state"]["slot_values"].items():
                for v in vs:
                    if v in sentence:
                        start_idx = sentence.index(v)
                        end_idx = start_idx + len(v)
                        word_indices[start_idx:end_idx] = True
        if "actions" in frame:
            for action in frame["actions"]:
                k = action["slot"]
                for v in action["values"]:
                    if v in sentence:
                        start_idx = sentence.index(v)
                        end_idx = start_idx + len(v)
                        word_indices[start_idx:end_idx] = True

    for i in range(len(sentence)):
        if sentence[i].isalnum():
            word_indices[i] = True
    res = []
    idx = 0
    while idx < len(word_indices):
        if word_indices[idx]:
            start_idx = idx
            while word_indices[idx]:
                idx += 1
            end_idx = idx
            res.append((start_idx, end_idx))
        idx += 1
    return res


def find_word_in_turn(dialogue, turn_id, value, start_idx, end_idx):
    """
    find non-cat slot value in turn.
    return  List[(turn_id, frame_id, key)]
    """
    assert isinstance(value, str)
    frames = dialogue["turns"][turn_id]["frames"]
    res = []
    for frame_id, frame in enumerate(frames):
        for slot in frame["slots"]:
            if start_idx == slot["start"] and end_idx == slot["exclusive_end"]:
                res.append((turn_id, frame_id, slot["slot"]))
    return res


def get_new_value(dialogue, turn_id, value, start_idx, end_idx):
    """
    replace span with another value from ontology if this belongs non-cat slot
    return new value
    """
    candidates = find_word_in_turn(dialogue, turn_id, value, start_idx, end_idx)
    possible_values = set()
    for _, frame_id, k in candidates:
        frame = dialogue["turns"][turn_id]["frames"][frame_id]
        service = frame["service"]
        if "possible_values" in ontology[(service, k)]:
            possible_values.update(ontology[(service, k)]["possible_values"])
    return random.choice(list(possible_values)) if possible_values else None


def replace(dialogue, turn_id, start_idx, end_idx, new_value):
    """
    replace utterance at turn_id around start_idx:end_idx with new_value.
    If old value is found in turn (non-categorical slot), change all affected frames with new_value:
        -update_values
        -update_spans
    """
    assert isinstance(turn_id, int)
    assert isinstance(start_idx, int)
    assert isinstance(end_idx, int)
    turn = dialogue["turns"][turn_id]
    sentence = turn["utterance"]
    old_value = sentence[start_idx:end_idx]
    affected_values = find_word_in_turn(
        dialogue=dialogue, turn_id=turn_id, value=old_value, start_idx=start_idx, end_idx=end_idx
    )
    affected_spans = [(turn_id, start_idx, end_idx)]
    for _, frame_id, key in affected_values.copy():
        frame = dialogue["turns"][turn_id]["frames"][frame_id]
        new_affected_values = frame["state_update"][key]
        affected_values += new_affected_values
        for a_turn_id, a_frame_id, a_key in new_affected_values:
            assert key == a_key
            spans = (
                dialogue["turns"][a_turn_id]["frames"][a_frame_id]["slot_to_span"].get(a_key, {}).get(old_value, None)
            )
            if spans:
                affected_spans += [(a_turn_id, spans[0], spans[1])]

    for a_turn_id, a_frame_id, a_key in affected_values:
        update_values(dialogue, a_turn_id, a_frame_id, a_key, old_value, new_value)
    for a_turn_id, start_idx, end_idx in affected_spans:
        turn = dialogue["turns"][a_turn_id]
        assert old_value == turn["utterance"][start_idx:end_idx]
        for a_frame_id in range(len(turn["frames"])):
            update_spans(dialogue, a_turn_id, a_frame_id, start_idx, end_idx, old_value, new_value)
        turn["utterance"] = turn["utterance"][:start_idx] + new_value + turn["utterance"][end_idx:]


def num2str(dialogue, turn_id, old_value, start_idx, end_idx):
    """
    gets old_value and returns stringified version if old_value was number and does not belong to non-cat span value
    """
    res = find_word_in_turn(dialogue, turn_id, old_value, start_idx, end_idx)
    if not res and old_value.isnumeric():
        return p.number_to_words(int(old_value)) + " " + old_value
    return None


def test_helper(dialogue, dialogue_id, turn_id, start_idx, end_idx, new_value):
    replace(dialogue, turn_id=turn_id, start_idx=start_idx, end_idx=end_idx, new_value=new_value)
    for turn in dialogue["turns"]:
        for frame in turn["frames"]:
            if "state_update" in frame:
                frame.pop("state_update")


def test(dialogues, dialogue_id, turn_id, old_value, new_value):
    dialogue = copy.deepcopy(dialogues[dialogue_id])
    augment_dialog_by_auxiliary_entries(dialogue)
    m = re.search(old_value, dialogue["turns"][turn_id]["utterance"])
    test_helper(dialogue, dialogue_id, turn_id, start_idx=m.start(), end_idx=m.end(), new_value=new_value)
    pprint(dialogue)
    validate(dialogue)
    d_str_new = json.dumps(dialogue, sort_keys=True, indent=2)
    d_str_old = json.dumps(dialogues[dialogue_id], sort_keys=True, indent=2)
    print(d_str_new == d_str_old)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--concat_orig_dialogue", action="store_true", help="contenate original dialogue to the augmented one"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="",
        help="data directory. contains one schema.json and multiple dialogue*.json files",
    )
    parser.add_argument("--output_dir", type=str, help="output data directory", default=None)
    parser.add_argument("--num2string", action="store_true", help="convert digits to string")
    parser.add_argument("--repeat", type=int, default=5, help="number of augmentation sweeps over input data")
    parser.add_argument("--replace_turn_prob", type=float, default=1.0, help="likelihood to modify an utterance turn")
    parser.add_argument(
        "--replace_word_prob", type=float, default=1.0, help="likelihood to modify a word in an utterance"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    print(vars(args))
    random.seed(args.seed)

    if not os.path.exists(args.input_dir):
        raise ValueError(
            "SGD dataset not found. Dataset can be downloaded from https://github.com/google-research-datasets/dstc8-schema-guided-dialogue"
        )

    in_file_path = args.input_dir
    schema_path = os.path.join(in_file_path, 'schema.json')
    dialogue_files = [
        os.path.join(in_file_path, f)
        for f in os.listdir(in_file_path)
        if os.path.isfile(os.path.join(in_file_path, f))
        if "dialogue" in f
    ]
    dialogue_files.sort()
    orig_dialog = []
    for d_file in dialogue_files:
        orig_dialog.extend(json.load(open(d_file, 'r')))
    print(f"len(orig_dialog) = {len(orig_dialog)}")
    orig_schema = json.load(open(schema_path, 'r'))

    dialogue_count = defaultdict(int)
    final_dialogues = defaultdict(list)

    ontology = get_ontology(dialogues=orig_dialog, schemas=orig_schema)

    for dialogue_id, dialogue in tqdm(enumerate(orig_dialog)):
        validate(dialogue)  # for test purposes
        augment_dialog_by_auxiliary_entries(dialogue)
        validate(dialogue)  # for test purposes

    if args.num2string:
        if args.concat_orig_dialogue:
            process_dialogues(
                final_dialogues=final_dialogues,
                dialogue_count=dialogue_count,
                dialogues=orig_dialog,
                replace_turn_prob=1.0,
                replace_word_prob=1.0,
                new_val_func=num2str,
            )
        else:
            process_dialogues(
                final_dialogues=defaultdict(list),
                dialogue_count=defaultdict(int),
                dialogues=orig_dialog,
                replace_turn_prob=1.0,
                replace_word_prob=1.0,
                new_val_func=num2str,
            )

    for _ in range(args.repeat):
        dialogues = copy.deepcopy(orig_dialog)
        process_dialogues(
            final_dialogues=final_dialogues,
            dialogue_count=dialogue_count,
            dialogues=dialogues,
            replace_turn_prob=args.replace_turn_prob,
            replace_word_prob=args.replace_word_prob,
            new_val_func=get_new_value,
        )

    if args.concat_orig_dialogue and not args.num2string:
        for dialogue_id, dialogue in tqdm(enumerate(orig_dialog)):
            d_id, d_count = dialogue["dialogue_id"].split("_")
            d_id = int(d_id)
            dialogue["dialogue_id"] = f"{d_id}_{dialogue_count[d_id]:05d}"
            dialogue_count[d_id] += 1
            final_dialogues[d_id].append(dialogue)

    for dir_id, dialogues in final_dialogues.items():
        for dialogue in dialogues:
            for turn in dialogue["turns"]:
                for frame in turn["frames"]:
                    if 'state_update' in frame:
                        frame.pop("state_update")
                    if 'slot_to_span' in frame:
                        frame.pop("slot_to_span")
    if args.output_dir is None:
        output_dir = f"augmented_repeat{args.repeat}_replace_turn_prob{args.replace_turn_prob}_replace_word_prob{args.replace_word_prob}_concatorig{args.concat_orig_dialogue}_num2string{args.num2string}"
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    for dir_id, dialogues in final_dialogues.items():
        with open(os.path.join(output_dir, f"dialogues_{dir_id:03d}.json"), 'w') as outfile:
            json.dump(dialogues, outfile, indent=2)

    with open(os.path.join(output_dir, f"schema.json"), 'w') as outfile:
        json.dump(orig_schema, outfile, indent=2)
