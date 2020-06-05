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
from os.path import exists, expanduser

from nemo.collections.nlp.data.datasets.multiwoz_dataset import MultiWOZDataDesc
from nemo.collections.nlp.data.datasets.multiwoz_dataset.state import init_state
from nemo.collections.nlp.nm.non_trainables import (
    RuleBasedDPMMultiWOZ,
    SystemUtteranceHistoryUpdate,
    TemplateNLGMultiWOZ,
    TradeStateUpdateNM,
    UserUtteranceEncoder,
)
from nemo.collections.nlp.nm.trainables import EncoderRNN, TRADEGenerator
from nemo.core import DeviceType, NeuralGraph, NeuralModuleFactory, OperationMode
from nemo.utils import logging

# Examples: two "separate" dialogs (one single-turn, one multiple-turn).
examples = [
    # ["I want to find a moderate hotel with internet and parking in the east"],
    [
        "Is there a train from Ely to Cambridge on Tuesday ?",
        "I need to arrive by 11 am .",
        "What is the trip duration ?",
        "Yes, please book it",
    ],
    # ["I want to find a moderate hotel", "What is the address ?"],
]


def forward(dialog_pipeline, user_uttr, dial_history, belief_state):
    """
    Forward pass of the "Complete Dialog Pipeline".

    Returns system reply and updates dialogue belief 
    by passing system and user utterances (dialogue history) through the TRADE Dialogue State Tracker,
    then the output of the TRADE model goes to the Rule-base Dialogue Policy Magager
    and the output of the Dialog Policy Manager goes to the Template-based Natural Language Generation module.

    ..note: NeuralGraph is now lacking a generic forward() pass. Once implemented, this function will become obsolete.
    
    Args:
        user_uttr (str): User utterance
        dialog_history (str): Dialogue history contains all previous system and user utterances
        belief_state (dict): dialogue state
    Returns:
        system_uttr (str): system response
        belief_state (dict): updated dialogue state
        dialog_history (str): Dialogue history contains all previous system and user utterances
    """
    # Manually execute modules in the graph, following the order defined in steps.
    # 1. Forward pass throught Word-Level Dialog State Tracking modules (TRADE).
    # 1.1. User utterance encoder.
    dialog_ids, dialog_lens, dial_history = dialog_pipeline.modules[dialog_pipeline.steps[0]].forward(
        user_uttr=user_uttr, dialog_history=dial_history,
    )
    # 1.2. TRADE encoder.
    outputs, hidden = dialog_pipeline.modules[dialog_pipeline.steps[1]].forward(
        inputs=dialog_ids, input_lens=dialog_lens
    )
    # 1.3. TRADE generator.
    point_outputs, gate_outputs = dialog_pipeline.modules[dialog_pipeline.steps[2]].forward(
        encoder_hidden=hidden, encoder_outputs=outputs, dialog_ids=dialog_ids, dialog_lens=dialog_lens,
    )

    # 1.4. The module "decoding" the TRADE output into belief and request states.
    belief_state, request_state = dialog_pipeline.modules[dialog_pipeline.steps[3]].forward(
        gating_preds=gate_outputs, point_outputs_pred=point_outputs, belief_state=belief_state, user_uttr=user_uttr
    )

    # 2. Forward pass throught Dialog Policy Manager module (Rule-Based, queries a "simple DB" to get required data).
    belief_state, system_acts = dialog_pipeline.modules[dialog_pipeline.steps[4]].forward(
        belief_state=belief_state, request_state=request_state
    )

    # 3. Forward pass throught Natural Language Generator module (Template-Based).
    system_uttr = dialog_pipeline.modules[dialog_pipeline.steps[5]].forward(system_acts=system_acts)

    # 4. Update dialog  history with system utterance
    dial_history = dialog_pipeline.modules[dialog_pipeline.steps[6]].forward(
        sys_uttr=system_uttr, dialog_history=dial_history
    )

    # Return the updated states and dialog history.
    return system_uttr, belief_state, dial_history


if __name__ == "__main__":
    # Parse the command-line arguments.
    parser = argparse.ArgumentParser(
        description="Complete dialogue pipeline examply with TRADE model on MultiWOZ dataset"
    )
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

    # Get the absolute path.
    abs_data_dir = expanduser(args.data_dir)

    # Check if data dir exists
    if not exists(abs_data_dir):
        raise ValueError(f"Data folder `{abs_data_dir}` not found")

    if args.show_all_output:
        logging.setLevel('DEBUG')

    # Initialize NF.
    nf = NeuralModuleFactory(placement=DeviceType.CPU, local_rank=None, log_dir=args.work_dir, checkpoint_dir=None)

    # Initialize the modules.

    # List of the domains to be considered.
    domains = {"attraction": 0, "restaurant": 1, "train": 2, "hotel": 3, "taxi": 5}

    # Create DataDescriptor that contains information about domains, slots, and associated vocabulary
    data_desc = MultiWOZDataDesc(abs_data_dir, domains)
    vocab_size = len(data_desc.vocab)

    # Encoder changing the "user utterance" into format accepted by TRADE encoderRNN.
    user_utterance_encoder = UserUtteranceEncoder(data_desc=data_desc)

    # TRADE modules.
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

    if exists(expanduser(args.encoder_ckpt)) and exists(expanduser(args.decoder_ckpt)):
        trade_encoder.restore_from(args.encoder_ckpt)
        trade_decoder.restore_from(args.decoder_ckpt)
    else:
        logging.info("Please refer to the NeMo docs for steps on how to obtain TRADE checkpoints")

    # Output decoder.
    trade_output_decoder = TradeStateUpdateNM(data_desc=data_desc)

    # DPM module.
    rule_based_policy = RuleBasedDPMMultiWOZ(data_dir=abs_data_dir)
    # NLG module.
    template_nlg = TemplateNLGMultiWOZ()

    # Updates dialog history with system utterance.
    sys_utter_history_update = SystemUtteranceHistoryUpdate()

    # Construct the "evaluation" (inference) neural graph by connecting the modules using nmTensors.
    # Note: Using the same names for passed nmTensor as in the actual forward pass.
    with NeuralGraph(operation_mode=OperationMode.evaluation) as dialog_pipeline:
        # 1.1. User utterance encoder.
        # Bind all the input ports of this module.
        dialog_ids, dialog_lens, dial_history = user_utterance_encoder(
            user_uttr=dialog_pipeline, dialog_history=dialog_pipeline,
        )
        # Fire step 1: 1.2. TRADE encoder.
        outputs, hidden = trade_encoder(inputs=dialog_ids, input_lens=dialog_lens)
        # 1.3. TRADE generator.
        point_outputs, gate_outputs = trade_decoder(
            encoder_hidden=hidden, encoder_outputs=outputs, dialog_ids=dialog_ids, dialog_lens=dialog_lens,
        )

        # 1.4. The module "decoding" the TRADE output into belief and request states.
        # Bind the "belief_state" input port.
        belief_state, request_state = trade_output_decoder(
            gating_preds=gate_outputs,
            point_outputs_pred=point_outputs,
            belief_state=dialog_pipeline,
            user_uttr=dialog_pipeline.inputs["user_uttr"],
        )

        # 2. Forward pass throught Dialog Policy Manager module (Rule-Based, queries a "simple DB" to get required data).
        belief_state, system_acts = rule_based_policy(belief_state=belief_state, request_state=request_state)

        # 3. Forward pass throught Natural Language Generator module (Template-Based).
        system_uttr = template_nlg(system_acts=system_acts)

        # 4. Update dialog  history with system utterance
        dial_history = sys_utter_history_update(sys_uttr=system_uttr, dialog_history=dial_history)

    # Show the graph summary.
    logging.info(dialog_pipeline.summary())

    # "Execute" the graph - depending on the mode.
    if args.mode == 'interactive':
        # for user_uttr in user_uttrs:
        logging.info("============ Starting a new dialogue ============")
        system_uttr, system_action, belief_state, dial_history = init_state()
        while True:
            logging.info("Type your text, use STOP to exit and RESTART to start a new dialogue.")
            user_uttr = input()

            if user_uttr == "STOP":
                logging.info("===================== Exiting ===================")
                break
            elif user_uttr == "RESTART":
                system_uttr, system_action, belief_state, dial_history = init_state()
                logging.info("============ Starting a new dialogue ============")
            else:
                # Pass the "user uterance" as inputs to the dialog pipeline.
                system_uttr, belief_state, dial_history = forward(
                    dialog_pipeline, user_uttr, dial_history, belief_state
                )

    elif args.mode == 'example':
        for example in examples:
            logging.info("============ Starting a new dialogue ============")
            system_uttr, system_action, belief_state, dial_history = init_state()
            # system_uttr, dialog_history = "", ""
            # Execute the dialog by passing the consecutive "user uterances" as inputs to the dialog pipeline.
            for user_uttr in example:
                logging.info("User utterance: %s", user_uttr)
                system_uttr, belief_state, dial_history = forward(
                    dialog_pipeline, user_uttr, dial_history, belief_state
                )
