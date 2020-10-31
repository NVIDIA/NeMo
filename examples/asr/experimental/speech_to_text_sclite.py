# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""
This script serves three goals:
    (1) Demonstrate how to use NeMo Models outside of PytorchLightning
    (2) Shows example of batch ASR inference
    (3) Serves as CI test for pre-trained checkpoint
"""

from argparse import ArgumentParser

import torch
import os
import json
import numpy as np

from nemo.collections.asr.metrics.wer import WER, word_error_rate
from nemo.collections.asr.models import EncDecCTCModel
from nemo.utils import logging

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


can_gpu = torch.cuda.is_available()
def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum(axis=-1).reshape([logits.shape[0], 1])

def get_probs(all_logits):
    probs = []
    for logits in all_logits:
        probs.append(softmax(logits))
    return probs

def get_utt_info(manifest_path):
    info_list = []
    with open(manifest_path, "r") as utt_f:
        for line in utt_f:
            utt = json.loads(line)
            info_list.append(utt)

    return info_list

def get_space_index(all_probs):
    # get timestamps for space symbols
    all_spaces = []
    for i in range(len(all_probs)):
        probs = all_probs[i][:]
        spaces = []

        state = ''
        idx_state = 0

        if np.argmax(probs[0]) == 0:
            state = 'space'

        for idx in range(1, probs.shape[0]):
            current_char_idx = np.argmax(probs[idx])
            if state == 'space' and current_char_idx != 0:
                spaces.append([idx_state, idx - 1])
                state = ''
            if state == '':
                if current_char_idx == 0:
                    state = 'space'
                    idx_state = idx

        if state == 'space':
            spaces.append([idx_state, len(pred) - 1])
        all_spaces.append(spaces)
    return all_spaces

def get_word_index(all_transcripts, all_spaces,info_list, offset, time_stride=0.02):
    # cut words
    all_durs = []
    for i in range(len(all_transcripts)):
        durs = []
        transcripts = all_transcripts[i]
        spaces = all_spaces[i]
        # split the transcript into words
        words = transcripts.split()
        # cut words
        pos_prev = 0
        if len(words) < len(spaces):
            print("Error " + str(i) )
        for j, spot in enumerate(spaces):
            pos_end = (spot[0] + spot[1]) / 2 * time_stride - offset
            durs.append([words[j], pos_prev, (pos_end - pos_prev)])
            pos_prev = pos_end
        durs.append([words[-1], pos_prev, info_list[i]['duration'] - pos_prev])
        all_durs.append(durs)
    return all_durs

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--asr_model", type=str, default="QuartzNet15x5Base-En", required=False, help="Pass: 'QuartzNet15x5Base-En'",
    )
    parser.add_argument("--dataset", type=str, required=True, help="path to evaluation data")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--normalize_text", default=True, type=bool, help="Normalize transcripts or not. Set to False for non-English.")
    parser.add_argument(
        "--sclite_fmt", default="trn", type=str, help="sclite output format. Only trn and ctm are supported"
    )
    parser.add_argument("--out_dir", type=str, required=True, help="Destination dir for output files")
    parser.add_argument("--model_delay", type=float, default=0.18)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.asr_model.endswith('.nemo'):
        logging.info(f"Using local ASR model from {args.asr_model}")
        asr_model = EncDecCTCModel.restore_from(restore_path=args.asr_model)
    else:
        logging.info(f"Using NGC cloud ASR model {args.asr_model}")
        asr_model = EncDecCTCModel.from_pretrained(model_name=args.asr_model)
    asr_model.setup_test_data(
        test_data_config={
            'sample_rate': 16000,
            'manifest_filepath': args.dataset,
            'labels': asr_model.decoder.vocabulary,
            'batch_size': args.batch_size,
            'normalize_transcripts': args.normalize_text,
        }
    )
    if can_gpu:
        asr_model = asr_model.cuda()
    asr_model.eval()
    labels_map = dict([(i, asr_model.decoder.vocabulary[i]) for i in range(len(asr_model.decoder.vocabulary))])

    wer = WER(vocabulary=asr_model.decoder.vocabulary)
    hypotheses = []
    references = []
    all_log_probs = []
    utt_index = 1
    for test_batch in asr_model.test_dataloader():
        if can_gpu:
            test_batch = [x.cuda() for x in test_batch]
        with autocast():
            log_probs, encoded_len, greedy_predictions = asr_model(
                input_signal=test_batch[0], input_signal_length=test_batch[1]
            )
        for r in log_probs.cpu().numpy():
            all_log_probs.append(r)
        hypotheses += wer.ctc_decoder_predictions_tensor(greedy_predictions)
        for batch_ind in range(greedy_predictions.shape[0]):
            reference = ''.join([labels_map[c] for c in test_batch[2][batch_ind].cpu().detach().numpy()])
            references.append(reference)
        del test_batch

    wer_value = word_error_rate(hypotheses=hypotheses, references=references)
    info_list = get_utt_info(args.dataset)
    if args.sclite_fmt == "trn":
        hypfile = os.path.join(args.out_dir, "hyp.trn")
        reffile = os.path.join(args.out_dir, "ref.trn")
        with open(hypfile, "w") as hyp_f, open(reffile, "w") as ref_f:
            for i in range(len(hypotheses)):
                utt_id = os.path.splitext(os.path.basename(info_list[i]['audio_filepath']))[0]
                # rfilter in sctk likes having each utt to have a space at the beginning
                hyp_f.write(" " + hypotheses[i] + " (" + utt_id + ")" + "\n")
                ref_f.write(" " + references[i] + " (" + utt_id + ")" + "\n")

    elif args.sclite_fmt == "ctm":
        ctm_file = os.path.join(args.out_dir, "hyp.ctm")
        labels = list(asr_model.cfg.decoder.params.vocabulary) + ['blank']
        labels[0] = 'space'
        probs = get_probs(all_log_probs)
        spaces = get_space_index(probs)
        word_durs = get_word_index(hypotheses, spaces, info_list, args.model_delay)
        with open(ctm_file, "w") as ctm_f:
            for i, info in enumerate(info_list):
                for word_info in word_durs[i]:
                    ctm_f.write(info['filename'] + " " +info['channel'] + " " + '{:.3f}'.format(info['begin_offset'] + float(word_info[1])) + " " +
                                '{:.3f}'.format(float(word_info[2]) - float(word_info[1])) + " " + word_info[0] + "\n")



if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
