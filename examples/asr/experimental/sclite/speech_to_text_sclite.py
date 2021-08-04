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
This script is based on speech_to_text_infer.py and allows you to score the hypotheses
with sclite. A local installation from https://github.com/usnistgov/SCTK is required.
Hypotheses and references are first saved in trn format and are scored after applying a glm
file (if provided).
"""

import errno
import json
import os
import subprocess
from argparse import ArgumentParser

import torch

from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.models import EncDecCTCModel
from nemo.utils import logging

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


def score_with_sctk(sctk_dir, ref_fname, hyp_fname, out_dir, glm=""):
    sclite_path = os.path.join(sctk_dir, "bin", "sclite")
    if not os.path.exists(sclite_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), sclite_path)
    # apply glm
    if os.path.exists(glm):
        rfilter_path = os.path.join(sctk_dir, "bin", "rfilter1")
        if not os.path.exists(rfilter_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), rfilter_path)
        hypglm = os.path.join(out_dir, os.path.basename(hyp_fname)) + ".glm"
        rfilt_cmd = [rfilter_path] + [glm]
        with open(hypglm, "w") as hypf, open(hyp_fname, "r") as hyp_in:
            subprocess.run(rfilt_cmd, stdin=hyp_in, stdout=hypf)
        refglm = os.path.join(out_dir, os.path.basename(ref_fname)) + ".glm"
        with open(refglm, "w") as reff, open(ref_fname, "r") as ref_in:
            subprocess.run(rfilt_cmd, stdin=ref_in, stdout=reff)
    else:
        refglm = ref_fname
        hypglm = hyp_fname

    _ = subprocess.check_output(f"{sclite_path} -h {hypglm}  -r {refglm} -i wsj -o all", shell=True)


can_gpu = torch.cuda.is_available()


def get_utt_info(manifest_path):
    info_list = []
    with open(manifest_path, "r") as utt_f:
        for line in utt_f:
            utt = json.loads(line)
            info_list.append(utt)

    return info_list


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--asr_model", type=str, default="QuartzNet15x5Base-En", required=False, help="Pass: 'QuartzNet15x5Base-En'",
    )
    parser.add_argument("--dataset", type=str, required=True, help="path to evaluation data")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--dont_normalize_text",
        default=False,
        action='store_true',
        help="Turn off trasnscript normalization. Recommended for non-English.",
    )
    parser.add_argument("--out_dir", type=str, required=True, help="Destination dir for output files")
    parser.add_argument("--sctk_dir", type=str, required=False, default="", help="Path to sctk root dir")
    parser.add_argument("--glm", type=str, required=False, default="", help="Path to glm file")
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    use_sctk = os.path.exists(args.sctk_dir)

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
            'normalize_transcripts': not args.dont_normalize_text,
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

    info_list = get_utt_info(args.dataset)
    hypfile = os.path.join(args.out_dir, "hyp.trn")
    reffile = os.path.join(args.out_dir, "ref.trn")
    with open(hypfile, "w") as hyp_f, open(reffile, "w") as ref_f:
        for i in range(len(hypotheses)):
            utt_id = os.path.splitext(os.path.basename(info_list[i]['audio_filepath']))[0]
            # rfilter in sctk likes each transcript to have a space at the beginning
            hyp_f.write(" " + hypotheses[i] + " (" + utt_id + ")" + "\n")
            ref_f.write(" " + references[i] + " (" + utt_id + ")" + "\n")

    if use_sctk:
        score_with_sctk(args.sctk_dir, reffile, hypfile, args.out_dir, glm=args.glm)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
