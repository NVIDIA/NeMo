# Copyright (C) NVIDIA CORPORATION. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.**
"""
some of the code taken from: https://github.com/NVIDIA/OpenSeq2Seq/blob/master/scripts/decode.py
This file is intended for use after jasper_eval.py and jasper_eval.py saved the log_probabilities. This is useful
if jasper_eval.py runs into memory issues.
"""
import argparse
import copy
import os
import pickle

import numpy as np
from ruamel.yaml import YAML

import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.helpers import post_process_predictions, post_process_transcripts, word_error_rate

logging = nemo.logging


def main():
    parser = argparse.ArgumentParser(description='Jasper')
    # model params
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--eval_datasets", type=str, required=True)
    parser.add_argument("--local_rank", default=None, type=int)
    # load params
    parser.add_argument("--logprob_path", default=None, type=str, required=True)

    # lm inference parameters
    parser.add_argument("--lm_path", default=None, type=str, required=True)
    parser.add_argument('--alpha', default=2.0, type=float, help='value of LM weight', required=False)
    parser.add_argument(
        '--alpha_max',
        type=float,
        help='maximum value of LM weight (for a grid search in \'eval\' mode)',
        required=False,
    )
    parser.add_argument(
        '--alpha_step', type=float, help='step for LM weight\'s tuning in \'eval\' mode', required=False, default=0.1
    )
    parser.add_argument('--beta', default=1.5, type=float, help='value of word count weight', required=False)
    parser.add_argument(
        '--beta_max',
        type=float,
        help='maximum value of word count weight (for a grid search in \
          \'eval\' mode',
        required=False,
    )
    parser.add_argument(
        '--beta_step',
        type=float,
        help='step for word count weight\'s tuning in \'eval\' mode',
        required=False,
        default=0.1,
    )
    parser.add_argument("--beam_width", default=128, type=int)

    args = parser.parse_args()
    logprob_path = args.logprob_path

    if args.local_rank is not None:
        raise NotImplementedError("Beam search decoder with LM does not currently support evaluation on multi-gpu.")

    # Instantiate Neural Factory with supported backend
    neural_factory = nemo.core.NeuralModuleFactory(
        backend=nemo.core.Backend.PyTorch, local_rank=args.local_rank, placement=nemo.core.DeviceType.GPU,
    )

    yaml = YAML(typ="safe")
    with open(args.model_config) as f:
        jasper_params = yaml.load(f)
    vocab = jasper_params['labels']
    eval_datasets = args.eval_datasets

    eval_dl_params = copy.deepcopy(jasper_params["AudioToTextDataLayer"])
    eval_dl_params.update(jasper_params["AudioToTextDataLayer"]["eval"])
    del eval_dl_params["train"]
    del eval_dl_params["eval"]

    data_layer = nemo_asr.AudioToTextDataLayer(
        manifest_filepath=eval_datasets,
        sample_rate=jasper_params['sample_rate'],
        labels=vocab,
        batch_size=-1,
        load_audio=False,
        **eval_dl_params,
    )

    if args.alpha_max is None:
        args.alpha_max = args.alpha
    # include alpha_max in tuning range
    args.alpha_max += args.alpha_step / 10.0

    if args.beta_max is None:
        args.beta_max = args.beta
    # include beta_max in tuning range
    args.beta_max += args.beta_step / 10.0

    beam_wers = []

    references = [data for data in data_layer.data_iterator]
    references = post_process_transcripts([references[0][2]], [references[0][3]], vocab)
    with open(logprob_path, "rb") as file:
        log_probs = pickle.load(file)

    for alpha in np.arange(args.alpha, args.alpha_max, args.alpha_step):
        for beta in np.arange(args.beta, args.beta_max, args.beta_step):
            logging.info('================================')
            logging.info(f'Infering with (alpha, beta): ({alpha}, {beta})')
            beam_search_with_lm = nemo_asr.BeamSearchDecoderWithLM(
                vocab=vocab,
                beam_width=args.beam_width,
                alpha=alpha,
                beta=beta,
                lm_path=args.lm_path,
                num_cpus=max(os.cpu_count(), 1),
                input_tensor=False,
            )
            log_probs = [np.exp(p) for p in log_probs]
            # log_probs = [np.exp(p) for np.asarray(p) in log_probs]
            beam_predictions = beam_search_with_lm(log_probs=log_probs, log_probs_length=None, force_pt=True)

            # Grab top hypothesis
            beam_predictions = [b[0][1] for b in beam_predictions[0]]
            lm_wer = word_error_rate(hypotheses=beam_predictions, references=references)
            logging.info("Beam WER {:.2f}%".format(lm_wer * 100))
            beam_wers.append(((alpha, beta), lm_wer * 100))

    logging.info('Beam WER for (alpha, beta)')
    logging.info('================================')
    logging.info('\n' + '\n'.join([str(e) for e in beam_wers]))
    logging.info('================================')
    best_beam_wer = min(beam_wers, key=lambda x: x[1])
    logging.info('Best (alpha, beta): ' f'{best_beam_wer[0]}, ' f'WER: {best_beam_wer[1]:.2f}%')


if __name__ == "__main__":
    main()
