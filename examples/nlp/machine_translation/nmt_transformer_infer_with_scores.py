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
from nemo.collections.nlp.modules.common.transformer.transformer_generators import (
    EnsembleBeamSearchSequenceGenerator,
    BeamSearchSequenceGenerator,
    BeamSearchSequenceGeneratorWithLanguageModel
)
from nemo.utils import logging

def filter_predicted_ids(tokenizer, ids):
    ids[ids >= tokenizer.vocab_size] = tokenizer.unk_id
    return ids


def nmt_postprocess(beam_results, model):
    beam_results = filter_predicted_ids(model.decoder_tokenizer, beam_results)
    translations = [model.decoder_tokenizer.ids_to_text(tr) for tr in beam_results.cpu().numpy()]
    if model.target_processor is not None:
        translations = [model.target_processor.detokenize(translation.split(' ')) for translation in translations]

    return translations


def input_preprocess(text, model):
    inputs = []
    for txt in text:
        if model.source_processor is not None:
            txt = model.source_processor.normalize(txt)
            txt = model.source_processor.tokenize(txt)
        ids = model.encoder_tokenizer.text_to_ids(txt)
        ids = [model.encoder_tokenizer.bos_id] + ids + [model.encoder_tokenizer.eos_id]
        inputs.append(ids)
    max_len = max(len(txt) for txt in inputs)
    src_ids_ = np.ones((len(inputs), max_len)) * model.encoder_tokenizer.pad_id
    for i, txt in enumerate(inputs):
        src_ids_[i][: len(txt)] = txt

    src_mask = torch.FloatTensor((src_ids_ != model.encoder_tokenizer.pad_id))
    src = torch.LongTensor(src_ids_)
    return src, src_mask

def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="")
    parser.add_argument("--srctext", type=str, required=True, help="")
    parser.add_argument("--tgtout", type=str, required=True, help="")
    parser.add_argument("--batch_size", type=int, default=256, help="")
    parser.add_argument("--beam_size", type=int, default=4, help="")
    parser.add_argument("--len_pen", type=float, default=0.6, help="")
    parser.add_argument("--max_delta_length", type=int, default=5, help="")
    parser.add_argument("--target_lang", type=str, default=None, help="")
    parser.add_argument("--source_lang", type=str, default=None, help="")
    # shallow fusion specific parameters
    parser.add_argument("--lm_model", type=str, default=None, help="")
    parser.add_argument("--fusion_coef", type=float, default=0.0, help="")

    args = parser.parse_args()
    torch.set_grad_enabled(False)
    if args.model.endswith(".nemo"):
        logging.info("Attempting to initialize from .nemo file")
        models = [
            nemo_nlp.models.machine_translation.MTEncDecModel.restore_from(restore_path=model_path)
            for model_path in args.model.split(',')
        ]
        src_text = []
        src_texts = []
        tgt_text = []
        all_scores = []
    else:
        raise NotImplementedError(f"Only support .nemo files, but got: {args.model}")

    for model in models:
        model.beam_search.beam_size = args.beam_size
        model.beam_search.len_pen = args.len_pen
        model.beam_search.max_delta_length = args.max_delta_length
        model.eval()

    if torch.cuda.is_available():
        models = [model.cuda() for model in models]

    logging.info(f"Translating: {args.srctext}")

    if len(models) > 1:
        if args.lm_model is not None:
            lm_model = nemo_nlp.models.language_modeling.TransformerLMModel.restore_from(restore_path=args.lm_model).eval()
        ensemble_generator = EnsembleBeamSearchSequenceGenerator(
            encoders=[model.encoder for model in models],
            embeddings=[model.decoder.embedding for model in models],
            decoders=[model.decoder.decoder for model in models],
            log_softmaxes=[model.log_softmax for model in models],
            max_sequence_length=512,
            beam_size=args.beam_size,
            bos=models[0].decoder_tokenizer.bos_id,
            pad=models[0].decoder_tokenizer.pad_id,
            eos=models[0].decoder_tokenizer.eos_id,
            len_pen=args.len_pen,
            max_delta_length=args.max_delta_length,
            language_model=lm_model,
            fusion_coef=args.fusion_coef
        )
    else:
        if args.lm_model is not None:
            lm_model = nemo_nlp.models.language_modeling.TransformerLMModel.restore_from(restore_path=args.lm_model).eval()
            model.beam_search = BeamSearchSequenceGeneratorWithLanguageModel(
                embedding=model.decoder.embedding,
                decoder=model.decoder.decoder,
                log_softmax=model.log_softmax,
                bos=model.decoder_tokenizer.bos_id,
                pad=model.decoder_tokenizer.pad_id,
                eos=model.decoder_tokenizer.eos_id,
                language_model=lm_model,
                fusion_coef=args.fusion_coef,
                max_sequence_length=model.decoder.max_sequence_length,
                beam_size=args.beam_size,
                len_pen=args.len_pen,
                max_delta_length=args.max_delta_length,
            )
        else:
            model.beam_search = BeamSearchSequenceGenerator(
                embedding=model.decoder.embedding,
                decoder=model.decoder.decoder,
                log_softmax=model.log_softmax,
                bos=model.decoder_tokenizer.bos_id,
                pad=model.decoder_tokenizer.pad_id,
                eos=model.decoder_tokenizer.eos_id,
                max_sequence_length=model.decoder.max_sequence_length,
                beam_size=args.beam_size,
                len_pen=args.len_pen,
                max_delta_length=args.max_delta_length,
            )

    count = 0
    with open(args.srctext, 'r') as src_f:
        for line in src_f:
            src_text.append(line.strip())
            if len(src_text) == args.batch_size:
                if len(models) > 1:
                    src_ids, src_mask = input_preprocess(src_text, models[0])
                    src_ids = src_ids.to(models[0].device)
                    src_mask = src_mask.to(models[0].device)
                    res, scores = ensemble_generator(src_ids, src_mask, return_beam_scores=True)
                    scores = scores.view(-1).data.cpu().numpy().tolist()
                    res = nmt_postprocess(res, models[0])
                else:
                    res, scores = model.translate(
                        text=src_text,
                        source_lang=args.source_lang,
                        target_lang=args.target_lang,
                        return_beam_scores=True
                    )
                assert len(res) == len(scores) == len(src_text) * args.beam_size
                tgt_text += res
                all_scores += scores
                src_texts += [item for item in src_text for i in range(args.beam_size)]
                src_text = []
            count += 1
            # if count % 300 == 0:
            #    print(f"Translated {count} sentences")
        if len(src_text) > 0:
            if len(models) > 1:
                src_ids, src_mask = input_preprocess(src_text, models[0])
                src_ids = src_ids.to(models[0].device)
                src_mask = src_mask.to(models[0].device)
                res, scores = ensemble_generator(src_ids, src_mask, return_beam_scores=True)
                scores = scores.view(-1).data.cpu().numpy().tolist()
                res = nmt_postprocess(res, models[0])
            else:
                res, scores = model.translate(
                    text=src_text,
                    source_lang=args.source_lang,
                    target_lang=args.target_lang,
                    return_beam_scores=True
                )
            assert len(res) == len(scores) == len(src_text) * args.beam_size
            tgt_text += res
            all_scores += scores
            src_texts += [item for item in src_text for i in range(args.beam_size)]

    with open(args.tgtout, 'w') as tgt_f:
        for line, score, inp in zip(tgt_text, all_scores, src_texts):
            tgt_f.write(inp + "\t" + line + "\t" + str(score) + "\n")

if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
