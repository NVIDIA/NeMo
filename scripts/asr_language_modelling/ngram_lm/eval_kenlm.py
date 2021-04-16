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
#

TOKEN_OFFSET = 100

import argparse
import json
import os
import pickle

import editdistance
import numpy as np
import torch
from tqdm.auto import tqdm

import nemo
import nemo.collections.asr as nemo_asr
from nemo.utils import logging


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1).reshape([x.shape[0], 1])


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate an ASR model with beam search decoding and an n-gram KenLM language model.'
    )
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--kenlm_model_path", required=False, default=None, type=str)
    parser.add_argument("--input_manifest", required=True, type=str)
    parser.add_argument("--preds_output_path", required=True, type=str)
    parser.add_argument("--probs_cache_path", default=None, type=str)
    parser.add_argument("--use_probs_cache", action="store_true")
    parser.add_argument("--beam_width", required=True, type=int)
    parser.add_argument("--beam_alpha", required=True, type=float)
    parser.add_argument("--beam_beta", required=True, type=float)
    parser.add_argument("--acoustic_batch_size", default=16, type=int)
    parser.add_argument("--beam_batch_size", default=128, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument(
        "--decoding_mode", choices=["greedy", "beamsearch", "beamsearch_ngram"], default="beamsearch_ngram", type=str
    )
    args = parser.parse_args()

    logging.info(f"BEAM WIDTH : {args.beam_width}")
    logging.info(f"BEAM ALPHA : {args.beam_alpha}")
    logging.info(f"BEAM BETA : {args.beam_beta}")

    asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(args.model_path, map_location=torch.device(args.device))
    asr_model.preprocessor.featurizer.dither = 0
    asr_model.preprocessor.featurizer.pad_to = 0
    # Set model to inference mode
    asr_model.eval()

    model_tokenizer = asr_model.tokenizer
    vocabs = list(model_tokenizer.tokenizer.get_vocab().keys())

    target_transcripts = []
    with open(args.input_manifest, 'r') as manifest_file:
        audio_file_paths = []
        for line in tqdm(manifest_file, desc=f"Reading Manifest {args.input_manifest} ..."):
            data = json.loads(line)
            target_transcripts.append(data['text'])
            audio_file_paths.append(data['audio_filepath'])

    # drop it later
    audio_file_paths = audio_file_paths[0:10]

    if args.use_probs_cache and os.path.exists(args.probs_cache_path):
        logging.info(f"Found a pickle file of probabilities at {args.probs_cache_path}.")
        logging.info(f"Loading the cached pickle file of probabilities from {args.probs_cache_path}...")
        with open(args.probs_cache_path, 'rb') as probs_file:
            all_probs = pickle.load(probs_file)

        if len(all_probs) != len(audio_file_paths):
            raise ValueError(
                f"The number of samples in the probabilities file '{args.probs_cache_path}' is not "
                f"the same as the manifest file. "
                f"You may need to delete the probabilities cached file."
            )
    else:
        with torch.no_grad():
            all_logits = asr_model.transcribe(audio_file_paths, batch_size=args.acoustic_batch_size, logprobs=True)
        all_probs = [softmax(logits) for logits in all_logits]
        logging.info(f"Writing pickle files of probabilities at {args.probs_cache_path}")
        with open(args.probs_cache_path, 'wb') as f_dump:
            pickle.dump(all_probs, f_dump)

    wer_dist_greedy = 0
    words_count = 0
    for batch_idx, probs in enumerate(all_probs):
        preds = np.argmax(probs, axis=1)
        preds_tensor = torch.tensor(preds, device='cpu').unsqueeze(0)
        pred_text = asr_model._wer.ctc_decoder_predictions_tensor(preds_tensor)[0]
        pred_split = pred_text.split()
        target_split = target_transcripts[batch_idx].split()
        dist = editdistance.eval(target_split, pred_split)
        wer_dist_greedy += dist
        words_count += len(target_split)

    # delete the model to free the memory
    del asr_model

    if args.decoding_mode == "beamsearch_ngram":
        if not os.path.exists(args.kenlm_model_path):
            raise FileNotFoundError(f"Could not find the KenLM model file '{args.kenlm_model_path}'.")
        lm_path = args.kenlm_model_path
    else:
        lm_path = None

    if args.decoding_mode in ["beamsearch_ngram", "beamsearch"]:
        # creating the beam search decoder
        beam_search_lm = nemo_asr.modules.BeamSearchDecoderWithLM(
            vocab=[chr(idx + TOKEN_OFFSET) for idx in range(len(vocabs))],
            beam_width=args.beam_width,
            alpha=args.beam_alpha,
            beta=args.beam_beta,
            lm_path=lm_path,
            num_cpus=max(os.cpu_count(), 1),
            input_tensor=False,
        )

        wer_dist_best = 0
        wer_dist_min = 0
        wer_dist_max = 0
        sample_idx = 0
        with open(args.preds_output_path, 'w') as f_out:
            for batch_idx in tqdm(range(int(np.ceil(len(all_probs) / args.beam_batch_size)))):
                # disabling type checking
                with nemo.core.typecheck.disable_checks():
                    probs_batch = all_probs[batch_idx * args.beam_batch_size : (batch_idx + 1) * args.beam_batch_size]
                    beams_batch = beam_search_lm.forward(log_probs=probs_batch, log_probs_length=None,)

                for beams_idx, beams in enumerate(beams_batch):
                    for candidate_idx, candidate in enumerate(beams):
                        target_split = target_transcripts[sample_idx + beams_idx].split()
                        pred_text = model_tokenizer.ids_to_text([ord(c) - TOKEN_OFFSET for c in candidate[1]])
                        pred_split = pred_text.split()
                        if candidate_idx == 0:
                            # first candidate
                            dist = editdistance.eval(target_split, pred_split)
                            dist_min = dist_max = dist
                            wer_dist_best += dist
                        else:
                            dist = editdistance.eval(target_split, pred_split)
                            dist_min = min(dist_min, dist)
                            dist_max = max(dist_max, dist)

                        if candidate_idx == args.beam_width - 1:
                            # last candidate
                            wer_dist_min += dist_min
                            wer_dist_max += dist_max

                        # pred = model_tokenizer.ids_to_text([ord(c) - TOKEN_OFFSET for c in candidate[1]])
                        score = candidate[0]
                        f_out.write('{}\t{}\n'.format(pred_text, score))
                sample_idx += len(probs_batch)

        logging.info('Greedy WER = {:.2%}'.format(wer_dist_greedy / words_count))
        logging.info('WER with beam search decoding and N-gram model = {:.2%}'.format(wer_dist_best / words_count))
        logging.info('Best WER = {:.2%}'.format(wer_dist_min / words_count))
        logging.info('Worst WER = {:.2%}'.format(wer_dist_max / words_count))
    elif args.decoding_mode == "greedy":
        logging.info('Greedy WER = {:.2%}'.format(wer_dist_greedy / words_count))
    else:
        raise ValueError(f"'{args.decoding_mode}' is not a supported decoding approach.")


if __name__ == '__main__':
    main()
