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

"""
This file contains code artifacts adapted from the original implementation:
https://github.com/thu-coai/ConvLab-2

Tutorial on how to run this script, preprocess MultiWOZ dataset, 
train Dialogue State Tracker TRADE model, 
and download the TRADE pre-trained checkpoints, could be found here: 
https://nvidia.github.io/NeMo/nlp/intro.html#dialogue-state-tracking

To Run this script download pre-trained TRADE model checkpoint following the above tutorial.

python rule_based_policy_multiwoz.py \
    --data_dir PATH_TO_NEMO_PROCESSED_DATA/multiwoz2.1 \
    --encoder_ckpt PATH_TO_TRADE_EncoderRNN-EPOCH-10.pt \
    --decoder_ckpt PATH_TO_TRADEGenerator-EPOCH-10.pt \
    --mode example \

Use "--mode interactive" to chat with the system and "--hide_output" - to hide intermediate output of the dialogue modules
"""

import argparse
import os

from nemo import core as nemo_core
from nemo.backends.pytorch.common import EncoderRNN
from nemo.collections.nlp.data.datasets.multiwoz_dataset import MultiWOZDataDesc
from nemo.collections.nlp.data.datasets.multiwoz_dataset.state import default_state
from nemo.collections.nlp.nm.non_trainables import (
    RuleBasedMultiwozBotNM,
    TemplateNLGMultiWOZNM,
    TradeStateUpdateNM,
    UtteranceEncoderNM,
)
from nemo.collections.nlp.nm.trainables import TRADEGenerator
from nemo.utils import logging

parser = argparse.ArgumentParser(description="Complete dialogue pipeline examply with TRADE model on MultiWOZ dataset")
parser.add_argument(
    "--data_dir", default="data/multiwoz2.1", type=str, help="path to NeMo processed MultiWOZ data", required=True
)
parser.add_argument(
    "--encoder_ckpt", default=None, type=str, help="Path to pretrained encoder checkpoint", required=True
)
parser.add_argument(
    "--decoder_ckpt", default=None, type=str, help="Path to pretrained decoder checkpoint", required=True
)
parser.add_argument("--emb_dim", default=400, type=int, help="Should match pre-trained TRADE model")
parser.add_argument("--hid_dim", default=400, type=int, help="Should match pre-trained TRADE model")
parser.add_argument("--n_layers", default=1, type=int, help="Should match pre-trained TRADE model")
parser.add_argument(
    "--mode",
    default="example",
    choices=["example", "interactive"],
    help="Examples - pipeline example with the predified user queries, set to to interactive to chat with the system",
)
parser.add_argument(
    "--show_all_output", action="store_true", help="Set to True to show output of all dialogue modules"
)
parser.add_argument("--work_dir", default='outputs', type=str, help='Path to where to store logs')

args = parser.parse_args()

# Check if data dir exists
if not os.path.exists(args.data_dir):
    raise ValueError(f"Data folder `{args.data_dir}` not found")

if args.show_all_output:
    logging.setLevel('DEBUG')

nf = nemo_core.NeuralModuleFactory(
    backend=nemo_core.Backend.PyTorch, local_rank=None, log_dir=args.work_dir, checkpoint_dir=None
)

# List of the domains to be considered
domains = {"attraction": 0, "restaurant": 1, "train": 2, "hotel": 3, "taxi": 5}

# create DataDescriptor that contains information about domains, slots, and associated vocabulary
data_desc = MultiWOZDataDesc(args.data_dir, domains)
vocab_size = len(data_desc.vocab)

utterance_encoder = UtteranceEncoderNM(data_desc=data_desc)

trade_encoder = EncoderRNN(
    input_dim=vocab_size, emb_dim=args.emb_dim, hid_dim=args.hid_dim, dropout=0, n_layers=args.n_layers
)
trade_decoder = TRADEGenerator(
    vocab=data_desc.vocab,
    embeddings=trade_encoder.embedding,
    hid_size=args.hid_dim,
    dropout=0,
    slots=data_desc.slots,
    nb_gate=len(data_desc.gating_dict),
    teacher_forcing=0,
    max_res_len=4,
)

if os.path.exists(args.encoder_ckpt) and os.path.exists(args.decoder_ckpt):
    trade_encoder.restore_from(args.encoder_ckpt)
    trade_decoder.restore_from(args.decoder_ckpt)
else:
    logging.info("Please refer to the NeMo docs for steps on how to obtain TRADE checkpoints")

trade_output_decoder = TradeStateUpdateNM(data_desc=data_desc)
rule_based_policy = RuleBasedMultiwozBotNM(data_dir=args.data_dir)
template_nlg = TemplateNLGMultiWOZNM()


def init_session():
    """
    Restarts dialogue session
    Returns:
        empty system utterance, empty dialogue history and default empty dialogue state
    """
    system_action, belief_state, dial_history = default_state()
    return '', system_action, belief_state, dial_history


def update_state(system_uttr, user_uttr, dial_history, belief_state):
    """
    Returns system reply and updates dialogue belief 
        by passing system and user utterances (dialogue history) through the TRADE Dialogue State Tracker,
        then the output of the TRADE model goes to the Rule-base Dialogue Policy Magager
        and the output of the Policy Manager goes to the Rule-based Natural language generation module
    Args:
        user_uttr (str): User utterance
        system_uttr (str): Previous system utterance
        dialog_history (str): Dialogue history contains all previous system and user utterances
        belief_state (dict): dialogue state
    Returns:
        system_uttr (str): system response
        belief_state (dict): updated dialogue state
        dialog_history (str): Dialogue history contains all previous system and user utterances
    """
    src_ids, src_lens, dial_history = utterance_encoder.forward(
        dial_history=dial_history, user_uttr=user_uttr, sys_uttr=system_uttr
    )
    outputs, hidden = trade_encoder.forward(inputs=src_ids, input_lens=src_lens)
    point_outputs, gate_outputs = trade_decoder.forward(
        encoder_hidden=hidden, encoder_outputs=outputs, input_lens=src_lens, src_ids=src_ids
    )
    belief_state, request_state = trade_output_decoder.forward(
        gating_preds=gate_outputs, point_outputs_pred=point_outputs, belief_state=belief_state, user_uttr=user_uttr
    )
    system_acts, belief_state = rule_based_policy.forward(belief_state=belief_state, request_state=request_state)
    system_uttr = template_nlg.forward(system_acts=system_acts)
    return system_uttr, belief_state, dial_history


examples = [
    ["I want to find a moderate hotel with internet and parking in the east"],
    [
        "Is there a train from Ely to Cambridge on Tuesday ?",
        "I need to arrive by 11 am .",
        "What is the trip duration ?",
    ],
]

trade_encoder.eval()
trade_decoder.eval()

if args.mode == 'interactive':
    # for user_uttr in user_uttrs:
    while True:
        logging.info("Type your text, use STOP to exit and RESTART to start a new dialogue.")
        system_uttr, system_action, belief_state, dial_history = init_session()
        user_uttr = input()

        if user_uttr == "STOP":
            logging.info("===================== Exiting ===================")
            break
        elif user_uttr == "RESTART":
            system_uttr, system_action, belief_state, dial_history = init_session()
            logging.info("============ Starting a new dialogue ============")
        else:
            system_uttr, belief_state, dial_history = update_state(system_uttr, user_uttr, dial_history, belief_state)

elif args.mode == 'example':
    for example in examples:
        logging.info("============ Starting a new dialogue ============")
        system_uttr, system_action, belief_state, dial_history = init_session()
        system_uttr, dialog_history = "", ""
        for user_uttr in example:
            logging.info("User utterance: %s", user_uttr)
            system_uttr, belief_state, dial_history = update_state(system_uttr, user_uttr, dial_history, belief_state)
