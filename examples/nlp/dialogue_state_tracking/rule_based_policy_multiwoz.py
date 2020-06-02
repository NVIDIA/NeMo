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
from nemo.core import NmTensor
from nemo.backends.pytorch.common import EncoderRNN
from nemo.collections.nlp.data.datasets.multiwoz_dataset import MultiWOZDataDesc, dst_update, init_session
from nemo.collections.nlp.nm.non_trainables import RuleBasedMultiwozBotNM, TemplateNLGMultiWOZNM, UtteranceEncoderNM, TradeOutputNM
from nemo.collections.nlp.nm.trainables import TRADEGenerator
from nemo.utils import logging

parser = argparse.ArgumentParser(description="Complete dialogue pipeline examply with TRADE model on MultiWOZ dataset")
parser.add_argument("--data_dir", default="data/multiwoz2.1", type=str, help="path to NeMo processed MultiWOZ data")
parser.add_argument("--emb_dim", default=400, type=int, help="Should match pre-trained TRADE model")
parser.add_argument("--hid_dim", default=400, type=int, help="Should match pre-trained TRADE model")
parser.add_argument("--n_layers", default=1, type=int, help="Should match pre-trained TRADE model")
parser.add_argument(
    "--mode",
    default="example",
    choices=["example", "interactive"],
    help="Examples - pipeline example with the predified user queries, set to to interactive to chat with the system",
)
parser.add_argument("--hide_output", action="store_true", help="Set to True to hide output of the dialogue modules")
parser.add_argument("--encoder_ckpt", default=None, type=str, help="Path to pretrained encoder checkpoint")
parser.add_argument("--decoder_ckpt", default=None, type=str, help="Path to pretrained decoder checkpoint")
parser.add_argument("--work_dir", default='outputs', type=str, help='Path to where to store logs')

args = parser.parse_args()

# Check if data dir exists
if not os.path.exists(args.data_dir):
    raise ValueError(f"Data folder `{args.data_dir}` not found")

if not args.hide_output:
    logging.setLevel('DEBUG')

nf = nemo_core.NeuralModuleFactory(
    backend=nemo_core.Backend.PyTorch, local_rank=None, log_dir=args.work_dir, checkpoint_dir=None
)

# List of the domains to be considered
domains = {"attraction": 0, "restaurant": 1, "train": 2, "hotel": 3, "taxi": 5}

# create DataDescriptor that contains information about domains, slots, and associated vocabulary
data_desc = MultiWOZDataDesc(args.data_dir, domains)
vocab_size = len(data_desc.vocab)
encoder = EncoderRNN(input_dim=vocab_size, emb_dim=args.emb_dim, hid_dim=args.hid_dim, dropout=0, n_layers=args.n_layers)
decoder = TRADEGenerator(
    vocab=data_desc.vocab,
    embeddings=encoder.embedding,
    hid_size=args.hid_dim,
    dropout=0,
    slots=data_desc.slots,
    nb_gate=len(data_desc.gating_dict),
    teacher_forcing=0,
)

if args.encoder_ckpt and args.decoder_ckpt:
    encoder.restore_from(args.encoder_ckpt)
    decoder.restore_from(args.decoder_ckpt)



rule_based_policy = RuleBasedMultiwozBotNM(args.data_dir)
template_nlg = TemplateNLGMultiWOZNM()

user_uttr = "I want to find a moderate hotel"
encoder.eval()
decoder.eval()
system_uttr, dialog_history, state = init_session()
state["history"].append(["sys", system_uttr])
state["history"].append(["user", user_uttr])
state["user_action"] = user_uttr
logging.debug("Dialogue state: %s", state)


# remove history
utterance_encoder = UtteranceEncoderNM(data_desc=data_desc, history=state['history'])
# with NeuralGraph(operation_mode=OperationMode.both) as dialogue_pipeline:
utterance_encoded = utterance_encoder()

outputs, hidden = encoder(inputs=utterance_encoded.src_ids, input_lens=utterance_encoded.src_lens)
point_outputs, gate_outputs = decoder(
    encoder_hidden=hidden,
    encoder_outputs=outputs,
    input_lens=utterance_encoded.src_lens,
    src_ids=utterance_encoded.src_ids
)
trade_output_decoder = TradeOutputNM(data_desc)
trade_output = trade_output_decoder(gating_preds=gate_outputs, point_outputs_pred=point_outputs)





# def get_system_responce(user_uttr, system_uttr, dialog_history, state):
#     """
#     Returns system reply by passing user utterance through TRADE Dialogue State Tracker, then the output of the TRADE model to the
#     Rule-base Dialogue Policy Magager and the output of the Policy Manager to the Rule-based Natural language generation module
#     Args:
#         user_uttr(str): User utterance
#         system_uttr(str): Previous system utterance
#         dialog_history(str): Diaglogue history contains all previous system and user utterances
#         state (dict): dialogue state
#     Returns:
#         system_utterance(str): system response
#         state (dict): updated dialogue state 
#     """
#     state["history"].append(["sys", system_uttr])
#     state["history"].append(["user", user_uttr])
#     state["user_action"] = user_uttr
#     logging.debug("Dialogue state: %s", state)

#     state = dst_update(state, data_desc, user_uttr, encoder, decoder)
#     logging.debug("State after TRADE = Input to DPM: %s", state)

#     dpm_output = rule_based_policy.predict(state)
#     logging.debug("DPM output: %s", dpm_output)
#     logging.debug("State after DPM: %s", state)

#     system_uttr = template_nlg.generate(dpm_output)
#     logging.info("NLG output = System reply: %s", system_uttr)
#     return system_uttr, state


# example_user_uttrs = ["I want to find a moderate hotel", "What is the address ?"]

# encoder.eval()
# decoder.eval()
# logging.info("============ Starting a new dialogue ============")
# system_uttr, dialog_history, state = init_session()

# if args.mode == 'interactive':
#     # for user_uttr in user_uttrs:
#     while True:
#         logging.info("Type your text, use STOP to exit and RESTART to start a new dialogue.")
#         user_uttr = input()

#         if user_uttr == "STOP":
#             logging.info("===================== Exiting ===================")
#             break
#         elif user_uttr == "RESTART":
#             system_uttr, dialog_history, state = init_session()
#             logging.info("============ Starting a new dialogue ============")
#         get_system_responce(user_uttr, system_uttr, dialog_history, state)

# elif args.mode == 'example':
#     for user_uttr in example_user_uttrs:
#         logging.info("User utterance: %s", user_uttr)
#         get_system_responce(user_uttr, system_uttr, dialog_history, state)
