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

'''
Tutorial on how to run this script, preprocess MultiWOZ dataset, 
train Dialogue State Tracker TRADE model, 
and download the TRADE pre-trained checkpoints, could be found here: 
https://nvidia.github.io/NeMo/nlp/intro.html#dialogue-state-tracking

This file contains code artifacts adapted from the original implementation:
https://github.com/thu-coai/ConvLab-2
'''

import argparse
import copy
import os

from nemo import core as nemo_core
from nemo.backends.pytorch.common import EncoderRNN
from nemo.collections.nlp.data.datasets.multiwoz import dst_update, init_session
from nemo.collections.nlp.data.datasets.multiwoz_dataset import MultiWOZDataDesc
from nemo.collections.nlp.nm.non_trainables import RuleBasedMultiwozBotNM, TemplateNLGMultiWOZNM
from nemo.collections.nlp.nm.trainables import TRADEGenerator
from nemo.utils import logging

parser = argparse.ArgumentParser(description='Dialogue state tracking with TRADE model on MultiWOZ dataset')
parser.add_argument("--data_dir", default='data/multiwoz2.1', type=str, help='path to NeMo processed MultiWOZ data')
parser.add_argument("--emb_dim", default=400, type=int)
parser.add_argument("--hid_dim", default=400, type=int)
parser.add_argument("--n_layers", default=1, type=int)
parser.add_argument("--interactive", action="store_true")
parser.add_argument("--encoder_ckpt", default=None, type=str, help='Path to pretrained encoder checkpoint')
parser.add_argument("--decoder_ckpt", default=None, type=str, help='Path to pretrained decoder checkpoint')
args = parser.parse_args()


# Check if data dir exists
if not os.path.exists(args.data_dir):
    raise ValueError(f"Data folder `{args.data_dir}` not found")

nf = nemo_core.NeuralModuleFactory(backend=nemo_core.Backend.PyTorch, local_rank=None,)

# List of the domains to be considered
domains = {"attraction": 0, "restaurant": 1, "train": 2, "hotel": 3, "taxi": 5}

# create DataDescriptor that contains information about domains, slots, and associated vocabulary
data_desc = MultiWOZDataDesc(args.data_dir, domains)
vocab_size = len(data_desc.vocab)
encoder = EncoderRNN(vocab_size, args.emb_dim, args.hid_dim, 0, args.n_layers)
decoder = TRADEGenerator(
    data_desc.vocab,
    encoder.embedding,
    args.hid_dim,
    0,
    data_desc.slots,
    len(data_desc.gating_dict),
    teacher_forcing=0,
)

if args.encoder_ckpt and args.decoder_ckpt:
    encoder.restore_from(args.encoder_ckpt)
    decoder.restore_from(args.decoder_ckpt)

rule_based_policy = RuleBasedMultiwozBotNM(args.data_dir)
template_nlg = TemplateNLGMultiWOZNM()

if args.interactive:
    encoder.eval()
    decoder.eval()
    system_uttr, dialog_history, state = init_session()
    # # example #1
    # user_uttrs = ['I want to find a moderate hotel',
    #               'What is the address ?']
    # example #2
    user_uttrs = ['i need to book a hotel in the east that has 4 stars .', 'Which type of hotel is it ?']

    # TODO make sure punct is surrounded with spaces
    for user_uttr in user_uttrs:
        # print("Type your text, use STOP to exit and RESTART to start a new dialogue.", "\n")
        # # TODO replace with input()
        print('\nUser utterance:', user_uttr)
        # if user_uttr == "STOP":
        #     print("============ Exiting ============")
        #     break
        # elif user_uttr == "RESTART":
        #     print("============ Starting a new dialogue ============")
        # system_uttr, dialog_history, state = init_session()

        state['history'].append(['sys', system_uttr])
        state['history'].append(['user', user_uttr])
        state['user_action'] = user_uttr
        print('\nDialogue state:', state)

        state = dst_update(state, data_desc, user_uttr, encoder, decoder)
        print('\nState after TRADE = Input to DPM:', state)

        dpm_output = rule_based_policy.predict(state)
        print('\nDPM output:', dpm_output)
        print('\nState after DPM:', state)

        system_uttr = template_nlg.generate(dpm_output)
        print('\nNLG output:', system_uttr)
