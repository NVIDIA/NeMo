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

import torch
import numpy as np

import nemo.collections.nlp as nemo_nlp
from nemo.collections.nlp.modules.common.transformer import (
    BeamSearchSequenceGenerator,
    BeamSearchSequenceGeneratorWithLanguageModel,
)
from nemo.utils import logging

def get_lm_and_nmt_score(src_texts, tgt_texts, models, lm_model):
    inputs = []
    src_lengths = []
    tgt_lengths = []
    model = models[0]
    for txt in src_texts:
        if model.source_processor is not None:
            txt = model.source_processor.normalize(txt)
            txt = model.source_processor.tokenize(txt)
        ids = model.encoder_tokenizer.text_to_ids(txt)
        ids = [model.encoder_tokenizer.bos_id] + ids + [model.encoder_tokenizer.eos_id]
        inputs.append(ids)
        src_lengths.append(len(ids) - 1)
    max_len = max(len(txt) for txt in inputs)
    src_ids_ = np.ones((len(inputs), max_len)) * model.encoder_tokenizer.pad_id
    for i, txt in enumerate(inputs):
        src_ids_[i][: len(txt)] = txt

    src_mask = torch.FloatTensor((src_ids_ != model.encoder_tokenizer.pad_id)).to(model.device)
    src = torch.LongTensor(src_ids_).to(model.device)

    inputs = []
    for txt in tgt_texts:
        if model.target_processor is not None:
            txt = model.target_processor.normalize(txt)
            txt = model.target_processor.tokenize(txt)
        ids = model.decoder_tokenizer.text_to_ids(txt)
        ids = [model.decoder_tokenizer.bos_id] + ids + [model.decoder_tokenizer.eos_id]
        inputs.append(ids)
        tgt_lengths.append(len(ids) - 1)
    max_len = max(len(txt) for txt in inputs)
    tgt_ids_ = np.ones((len(inputs), max_len)) * model.decoder_tokenizer.pad_id
    for i, txt in enumerate(inputs):
        tgt_ids_[i][: len(txt)] = txt

    tgt_mask = torch.FloatTensor((tgt_ids_ != model.decoder_tokenizer.pad_id)).to(model.device)
    tgt = torch.LongTensor(tgt_ids_).to(model.device)
    tgt_inp = tgt[:, :-1]
    tgt_mask = tgt_mask[:, :-1]
    tgt_labels = tgt[:, 1:]

    nmt_lls = []
    for model in models:
        nmt_log_probs = model(src, src_mask, tgt_inp, tgt_mask)
        nmt_nll = model.eval_loss_fn(log_probs=nmt_log_probs, labels=tgt_labels)
        nmt_ll = nmt_nll.view(nmt_log_probs.size(0), nmt_log_probs.size(1)).sum(1) * -1.0
        nmt_ll = nmt_ll.data.cpu().numpy().tolist()
        nmt_lls.append(nmt_ll)
    nmt_ll = np.mean(nmt_lls)

    if lm_model is not None:
        lm_log_probs = lm_model(src[:, :-1], src_mask[:, :-1])
        lm_nll = model.eval_loss_fn(log_probs=lm_log_probs, labels=src[:, 1:])
        lm_ll = lm_nll.view(lm_log_probs.size(0), lm_log_probs.size(1)).sum(1) * -1.0
        lm_ll = lm_ll.data.cpu().numpy().tolist()
    else:
        lm_ll = None
    return nmt_ll, lm_ll, src_lengths, tgt_lengths

def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="")
    parser.add_argument("--srctext", type=str, required=True, help="")
    parser.add_argument("--tgtout", type=str, required=True, help="")
    parser.add_argument("--beam_size", type=int, required=True, help="")
    parser.add_argument("--source_lang", type=str, required=True, help="")
    parser.add_argument("--target_lang", type=str, required=True, help="")

    # shallow fusion specific parameters
    parser.add_argument("--lm_model", type=str, default=None, help="")
    parser.add_argument("--noisy_channel_coef", type=float, default=0.1, help="")

    args = parser.parse_args()
    torch.set_grad_enabled(False)
    if args.model.endswith(".nemo"):
        logging.info("Attempting to initialize from .nemo file")
        models = [
            nemo_nlp.models.machine_translation.MTEncDecModel.restore_from(restore_path=model_path)
            for model_path in args.model.split(',')
        ]
        for model in models:
            model.eval_loss_fn.reduction = 'none'
        if args.source_lang or args.target_lang:
            for model in models:
                model.setup_pre_and_post_processing_utils(args.target_lang, args.source_lang)    
        src_text = []
        tgt_text = []
    else:
        raise NotImplemented(f"Only support .nemo files, but got: {args.model}")

    if torch.cuda.is_available():
        models = [model.cuda() for model in models]

    if args.lm_model is not None:
        lm_model = nemo_nlp.models.language_modeling.TransformerLMModel.restore_from(restore_path=args.lm_model).eval()
        logging.info(f"Scoring: {args.srctext}")
    else:
        lm_model = None

    with open(args.srctext, 'r') as src_f:
        for line in src_f:
            src_text.append(line.strip().split('\t'))
            if len(src_text) == args.beam_size:
                src_texts = [item[1] for item in src_text]
                tgt_texts = [item[0] for item in src_text]
                scores = [float(item[2]) for item in src_text]
                rev_nmt_scores, lm_scores, src_lengths, tgt_lengths = get_lm_and_nmt_score(src_texts, tgt_texts, models, lm_model)
                fused_scores = []
                #mean_source_length = np.mean(src_lengths)
                #stddev_source_length = np.std(src_lengths)
                lm_scores = [None] * len(rev_nmt_scores) if lm_scores is None else lm_scores
                for s, r, l, sl, tl in zip(scores, rev_nmt_scores, lm_scores, src_lengths, tgt_lengths):
                    # len_pen = ((5 + sl) / 6) ** 0.6
                    # len_pen = 1 + ((sl - mean_source_length) / stddev_source_length)
                    l = 0 if l is None else l
                    score = s + args.noisy_channel_coef * (r + l)
                    # score = score * len_pen
                    fused_scores.append(score)
                tgt_text.append(src_texts[np.argmax(fused_scores)])
                src_text = []

        if len(src_text) > 0:
            src_texts = [item[1] for item in src_text]
            tgt_texts = [item[0] for item in src_text]
            scores = [float(item[2]) for item in src_text]
            rev_nmt_scores, lm_scores, src_lengths, tgt_lengths = get_lm_and_nmt_score(src_texts, tgt_texts, model, lm_model)
            fused_scores = []
            for s, r, l, sl, tl in zip(scores, rev_nmt_scores, lm_scores, src_lengths, tgt_lengths):
                score = s + args.noisy_channel_coef * (r + l)
                fused_scores.append(score)
            tgt_text.append(src_texts[np.argmax(fused_scores)])
            src_text = []

    with open(args.tgtout, 'w') as tgt_f:
        for line in tgt_text:
            tgt_f.write(line + "\n")


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter