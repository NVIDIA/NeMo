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
Given NMT model's .nemo file, this script can be used to translate text.
USAGE Example:
1. Obtain text file in src language. You can use sacrebleu to obtain standard test sets like so:
    sacrebleu -t wmt14 -l de-en --echo src > wmt14-de-en.src
2. Translate:
    python nmt_transformer_infer.py --model=[Path to .nemo file] --srctext=wmt14-de-en.src --tgtout=wmt14-de-en.pre
"""


from argparse import ArgumentParser

import numpy as np
import torch

import nemo.collections.nlp as nemo_nlp
from nemo.utils import logging


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to .nemo model file(s). If ensembling, provide comma separated paths to multiple models.",
    )
    parser.add_argument(
        "--srctext",
        type=str,
        required=True,
        help="Path to a TSV file that contains source sentence \t target sentence \t forward score for each beam candidate",
    )
    parser.add_argument(
        "--tgtout", type=str, required=True, help="Path to the file where re-ranked translations are to be written."
    )
    parser.add_argument("--beam_size", type=int, default=4, help="Number of beams to re-rank.")
    parser.add_argument(
        "--target_lang", type=str, default=None, help="Target language identifier ex: en,de,fr,es etc."
    )
    parser.add_argument(
        "--source_lang", type=str, default=None, help="Source language identifier ex: en,de,fr,es etc."
    )

    # shallow fusion specific parameters
    parser.add_argument(
        "--lm_model",
        type=str,
        default=None,
        help="Optional path to an LM model that has the same tokenizer as NMT models.",
    )
    parser.add_argument(
        "--noisy_channel_coef",
        type=float,
        default=0.05,
        help="Weight assigned to reverse NMT model + LM scores for re-ranking.",
    )

    args = parser.parse_args()
    torch.set_grad_enabled(False)
    models = []
    for model_path in args.model.split(','):
        if not model_path.endswith('.nemo'):
            raise NotImplementedError(f"Only support .nemo files, but got: {model_path}")
        model = nemo_nlp.models.machine_translation.MTEncDecModel.restore_from(restore_path=model_path).eval()
        model.eval_loss_fn.reduction = 'none'
        models.append(model)

    src_text = []
    tgt_text = []

    if torch.cuda.is_available():
        models = [model.cuda() for model in models]

    if args.lm_model is not None:
        lm_model = nemo_nlp.models.language_modeling.TransformerLMModel.restore_from(restore_path=args.lm_model).eval()
        logging.info(f"Scoring: {args.srctext}")
    else:
        lm_model = None

    with open(args.srctext, 'r') as src_f:
        count = 0
        for line in src_f:
            src_text.append(line.strip().split('\t'))
            if len(src_text) == args.beam_size:
                # Source and target sequences for the reverse direction model.
                src_texts = [item[1] for item in src_text]
                tgt_texts = [item[0] for item in src_text]
                src, src_mask = models[0].prepare_inference_batch(src_texts)
                tgt, tgt_mask = models[0].prepare_inference_batch(tgt_texts, target=True)
                forward_scores = [float(item[2]) for item in src_text]

                # Ensemble of reverse model scores.
                nmt_lls = []
                for model in models:
                    nmt_log_probs = model(src, src_mask, tgt[:, :-1], tgt_mask[:, :-1])
                    nmt_nll = model.eval_loss_fn(log_probs=nmt_log_probs, labels=tgt[:, 1:])
                    nmt_ll = nmt_nll.view(nmt_log_probs.size(0), nmt_log_probs.size(1)).sum(1) * -1.0
                    nmt_ll = nmt_ll.data.cpu().numpy().tolist()
                    nmt_lls.append(nmt_ll)
                rev_nmt_scores = np.stack(nmt_lls).mean(0)

                # LM scores.
                if lm_model is not None:
                    lm_log_probs = lm_model(src[:, :-1], src_mask[:, :-1])
                    lm_nll = model.eval_loss_fn(log_probs=lm_log_probs, labels=src[:, 1:])
                    lm_ll = lm_nll.view(lm_log_probs.size(0), lm_log_probs.size(1)).sum(1) * -1.0
                    lm_ll = lm_ll.data.cpu().numpy().tolist()
                else:
                    lm_ll = None
                lm_scores = [None] * len(rev_nmt_scores) if lm_ll is None else lm_ll

                # Score fusion.
                fused_scores = []
                for forward_score, rev_score, lm_score in zip(forward_scores, rev_nmt_scores, lm_scores):
                    lm_score = 0 if lm_score is None else lm_score
                    score = forward_score + args.noisy_channel_coef * (rev_score + lm_score)
                    fused_scores.append(score)
                tgt_text.append(src_texts[np.argmax(fused_scores)])
                src_text = []
                count += 1
                print(f'Reranked {count} sentences')

    with open(args.tgtout, 'w') as tgt_f:
        for line in tgt_text:
            tgt_f.write(line + "\n")


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
