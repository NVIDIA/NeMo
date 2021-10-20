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
Given NMT model's .nemo file(s), this script can be used to translate text.
USAGE Example:
1. Obtain text file in src language. You can use sacrebleu to obtain standard test sets like so:
    sacrebleu -t wmt14 -l de-en --echo src > wmt14-de-en.src
2. Translate:
    python nmt_transformer_infer.py --model=[Path to .nemo file(s)] --srctext=wmt14-de-en.src --tgtout=wmt14-de-en.pre
"""


import json
from argparse import ArgumentParser

import torch

import nemo.collections.nlp as nemo_nlp
from nemo.collections.nlp.modules.common.transformer import (
    BeamSearchSequenceGenerator,
    BeamSearchSequenceGeneratorWithLanguageModel,
    EnsembleBeamSearchSequenceGenerator,
)
from nemo.utils import logging


def translate_text(
    models, args, src_text, tgt_text, tgt_text_all, src_texts, all_scores, all_timing, ensemble_generator
):
    if len(models) > 1:
        src_ids, src_mask = models[0].prepare_inference_batch(src_text)
        best_translations = ensemble_generator(src_ids, src_mask, return_beam_scores=args.write_scores)
        if args.write_scores:
            all_results, scores, best_translations = (
                best_translations[0],
                best_translations[1],
                best_translations[2],
            )
            scores = scores.view(-1).data.cpu().numpy().tolist()
            all_scores += scores
            src_texts += [item for item in src_text for i in range(args.beam_size)]
            all_results = models[0].ids_to_postprocessed_text(
                all_results, models[0].decoder_tokenizer, models[0].target_processor
            )
            tgt_text_all += all_results
        best_translations = models[0].ids_to_postprocessed_text(
            best_translations, models[0].decoder_tokenizer, models[0].target_processor
        )
        tgt_text += best_translations
    else:
        model = models[0]
        best_translations = model.translate(
            text=src_text,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            return_beam_scores=args.write_scores,
            log_timing=args.write_timing,
        )

        if args.write_timing:
            *best_translations, timing_dict = best_translations
            all_timing.append(timing_dict)
        else:
            best_translations = (best_translations,)

        if args.write_scores:
            all_results, scores, best_translations = (
                best_translations[0],
                best_translations[1],
                best_translations[2],
            )
            all_scores += scores
            src_texts += [item for item in src_text for i in range(args.beam_size)]
            tgt_text_all += all_results
        else:
            best_translations = best_translations[0]

        tgt_text += best_translations

    print(f"Translated {len(tgt_text)} sentences")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to .nemo model file(s). If ensembling, provide comma separated paths to multiple models.",
    )
    parser.add_argument("--srctext", type=str, required=True, help="Path to the file to translate.")
    parser.add_argument(
        "--tgtout", type=str, required=True, help="Path to the file where translations are to be written."
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Number of sentences to batch together while translatiing."
    )
    parser.add_argument("--beam_size", type=int, default=4, help="Beam size.")
    parser.add_argument(
        "--len_pen", type=float, default=0.6, help="Length Penalty. Ref: https://arxiv.org/abs/1609.08144"
    )
    parser.add_argument(
        "--max_delta_length",
        type=int,
        default=5,
        help="Stop generating if target sequence length exceeds source length by this number.",
    )
    parser.add_argument(
        "--target_lang", type=str, default=None, help="Target language identifier ex: en,de,fr,es etc."
    )
    parser.add_argument(
        "--source_lang", type=str, default=None, help="Source language identifier ex: en,de,fr,es etc."
    )
    parser.add_argument(
        "--write_scores",
        action="store_true",
        help="Whether to write a separate file with scores not including length penalties corresponding to each beam hypothesis (.score suffix)",
    )
    parser.add_argument(
        "--write_timing",
        action="store_true",
        help="Whether to write a separate file with detailed timing info (.timing.json suffix)",
    )
    # shallow fusion specific parameters
    parser.add_argument(
        "--lm_model",
        type=str,
        default=None,
        help="Optional path to an LM model that has the same tokenizer as NMT models for shallow fuison. Note: If using --write_scores, it will add LM scores as well.",
    )
    parser.add_argument(
        "--fusion_coef", type=float, default=0.07, help="Weight assigned to LM scores during shallow fusion."
    )

    args = parser.parse_args()
    torch.set_grad_enabled(False)
    logging.info("Attempting to initialize from .nemo file")
    models = []
    for model_path in args.model.split(','):
        if not model_path.endswith('.nemo'):
            raise NotImplementedError(f"Only support .nemo files, but got: {model_path}")
        model = nemo_nlp.models.machine_translation.MTEncDecModel.restore_from(restore_path=model_path).eval()
        models.append(model)

    if (len(models) > 1) and (args.write_timing):
        raise RuntimeError("Cannot measure timing when more than 1 model is used")

    src_text = []
    tgt_text = []
    tgt_text_all = []
    src_texts = []
    all_scores = []
    all_timing = []

    if torch.cuda.is_available():
        models = [model.cuda() for model in models]

    if args.lm_model is not None:
        lm_model = nemo_nlp.models.language_modeling.TransformerLMModel.restore_from(restore_path=args.lm_model).eval()
    else:
        lm_model = None

    if len(models) > 1:
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
            fusion_coef=args.fusion_coef,
        )
    else:
        model = models[0]
        ensemble_generator = None
        if lm_model is not None:
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

    logging.info(f"Translating: {args.srctext}")

    with open(args.srctext, 'r') as src_f:
        for line in src_f:
            src_text.append(line.strip())
            if len(src_text) == args.batch_size:
                # warmup when measuring timing
                if not all_timing:
                    print("running a warmup batch")
                    translate_text(
                        models=models,
                        args=args,
                        src_text=src_text,
                        tgt_text=[],
                        tgt_text_all=[],
                        src_texts=[],
                        all_scores=[],
                        all_timing=[],
                        ensemble_generator=ensemble_generator,
                    )
                translate_text(
                    models=models,
                    args=args,
                    src_text=src_text,
                    tgt_text=tgt_text,
                    tgt_text_all=tgt_text_all,
                    src_texts=src_texts,
                    all_scores=all_scores,
                    all_timing=all_timing,
                    ensemble_generator=ensemble_generator,
                )
                src_text = []

        if len(src_text) > 0:
            translate_text(
                models=models,
                args=args,
                src_text=src_text,
                tgt_text=tgt_text,
                tgt_text_all=tgt_text_all,
                src_texts=src_texts,
                all_scores=all_scores,
                all_timing=all_timing,
                ensemble_generator=ensemble_generator,
            )

    with open(args.tgtout, 'w') as tgt_f:
        for line in tgt_text:
            tgt_f.write(line + "\n")

    if args.write_scores:
        with open(args.tgtout + '.score', 'w') as tgt_f_scores:
            for line, score, inp in zip(tgt_text_all, all_scores, src_texts):
                tgt_f_scores.write(inp + "\t" + line + "\t" + str(score) + "\n")

    if args.write_timing:
        # collect list of dicts to a dict of lists
        timing_dict = {}
        if len(all_timing):
            for k in all_timing[0].keys():
                timing_dict[k] = [t[k] for t in all_timing]

        with open(args.tgtout + '.timing.json', 'w') as timing_fh:
            json.dump(timing_dict, timing_fh)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
