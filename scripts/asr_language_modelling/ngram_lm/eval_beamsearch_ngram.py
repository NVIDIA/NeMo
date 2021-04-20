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
import logging
import os
import pickle

import contextlib
import editdistance
import numpy as np
import torch
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm

import nemo
import nemo.collections.asr as nemo_asr
from nemo.utils import logging


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1).reshape([x.shape[0], 1])


def beam_search_eval(
    all_probs,
    target_transcripts,
    model_tokenizer,
    preds_output_file,
    lm_path=None,
    beam_alpha=1.0,
    beam_beta=0.0,
    beam_width=128,
    beam_batch_size=16,
    progress_bar=True,
):
    vocabs = list(model_tokenizer.tokenizer.get_vocab().keys())
    # creating the beam search decoder
    beam_search_lm = nemo_asr.modules.BeamSearchDecoderWithLM(
        vocab=[chr(idx + TOKEN_OFFSET) for idx in range(len(vocabs))],
        beam_width=beam_width,
        alpha=beam_alpha,
        beta=beam_beta,
        lm_path=lm_path,
        num_cpus=max(os.cpu_count(), 1),
        input_tensor=False,
    )

    wer_dist_first = cer_dist_first = 0
    wer_dist_best = cer_dist_best = 0

    words_count = 0
    chars_count = 0

    sample_idx = 0
    with open(preds_output_file, 'w') as f_out:
        if progress_bar:
            it = tqdm(
                range(int(np.ceil(len(all_probs) / beam_batch_size))),
                desc=f"Beam search decoding with width={beam_width}, alpha={beam_alpha}, beta={beam_beta}",
                ncols=120,
            )
        else:
            it = range(int(np.ceil(len(all_probs) / beam_batch_size)))
        for batch_idx in it:
            # disabling type checking
            with nemo.core.typecheck.disable_checks():
                probs_batch = all_probs[batch_idx * beam_batch_size: (batch_idx + 1) * beam_batch_size]
                beams_batch = beam_search_lm.forward(log_probs=probs_batch, log_probs_length=None,)

            for beams_idx, beams in enumerate(beams_batch):
                target = target_transcripts[sample_idx + beams_idx]
                target_split_w = target.split()
                target_split_c = list(target)
                words_count += len(target_split_w)
                chars_count += len(target_split_c)
                wer_dist_min = cer_dist_min = 10000
                for candidate_idx, candidate in enumerate(beams):
                    pred_text = model_tokenizer.ids_to_text([ord(c) - TOKEN_OFFSET for c in candidate[1]])
                    pred_split_w = pred_text.split()
                    wer_dist = editdistance.eval(target_split_w, pred_split_w)
                    pred_split_c = list(pred_text)
                    cer_dist = editdistance.eval(target_split_c, pred_split_c)

                    wer_dist_min = min(wer_dist_min, wer_dist)
                    cer_dist_min = min(cer_dist_min, cer_dist)

                    if candidate_idx == 0:
                        # first candidate
                        wer_dist_first += wer_dist
                        cer_dist_first += cer_dist

                    score = candidate[0]
                    f_out.write('{}\t{}\n'.format(pred_text, score))
                wer_dist_best += wer_dist_min
                cer_dist_best += cer_dist_min
            sample_idx += len(probs_batch)

    logging.info(f"Stored the predictions of beam search decoding at '{preds_output_file}'.")
    if lm_path:
        logging.info('WER/CER with beam search decoding and N-gram model = {:.2%}/{:.2%}'.format(wer_dist_first / words_count, cer_dist_first / chars_count))
    else:
        logging.info('WER/CER with beam search decoding = {:.2%}/{:.2%}'.format(wer_dist_first / words_count, cer_dist_first / chars_count))
    logging.info('Best WER/CER in candidates = {:.2%}/{:.2%}'.format(wer_dist_best / words_count, cer_dist_best / chars_count))
    logging.info(f"=================================================================================")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate an ASR model with beam search decoding and an n-gram KenLM language model.'
    )
    parser.add_argument("--nemo_model_file", required=True, type=str)
    parser.add_argument("--kenlm_model_file", required=False, default=None, type=str)
    parser.add_argument("--input_manifest", required=True, type=str)
    parser.add_argument("--preds_output_folder", required=True, type=str)
    parser.add_argument("--probs_cache_file", default=None, type=str)
    parser.add_argument("--beam_width", required=True, type=int, nargs="+")
    parser.add_argument("--beam_alpha", required=True, type=float, nargs="+")
    parser.add_argument("--beam_beta", required=True, type=float, nargs="+")
    parser.add_argument("--acoustic_batch_size", default=16, type=int)
    parser.add_argument("--beam_batch_size", default=128, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument(
        "--decoding_mode", choices=["greedy", "beamsearch", "beamsearch_ngram"], default="beamsearch_ngram", type=str
    )
    # parser.add_argument("--parallel_runs", default=1, type=int)

    args = parser.parse_args()

    asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
        args.nemo_model_file, map_location=torch.device(args.device)
    )

    model_tokenizer = asr_model.tokenizer

    target_transcripts = []
    with open(args.input_manifest, 'r') as manifest_file:
        audio_file_paths = []
        for line in tqdm(manifest_file, desc=f"Reading Manifest {args.input_manifest} ...", ncols=120):
            data = json.loads(line)
            target_transcripts.append(data['text'])
            audio_file_paths.append(data['audio_filepath'])

    # drop it later
    #audio_file_paths = audio_file_paths[0:100]

    if args.probs_cache_file and os.path.exists(args.probs_cache_file):
        logging.info(f"Found a pickle file of probabilities at '{args.probs_cache_file}'.")
        logging.info(f"Loading the cached pickle file of probabilities from '{args.probs_cache_file}' ...")
        with open(args.probs_cache_file, 'rb') as probs_file:
            all_probs = pickle.load(probs_file)

        if len(all_probs) != len(audio_file_paths):
            raise ValueError(
                f"The number of samples in the probabilities file '{args.probs_cache_file}' does not "
                f"match the manifest file. You may need to delete the probabilities cached file."
            )
    else:
        if args.use_amp:
            if torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
                logging.info("AMP is enabled!\n")
                autocast = torch.cuda.amp.autocast
        else:
            @contextlib.contextmanager
            def autocast():
                yield
        with autocast():
            with torch.no_grad():
                all_logits = asr_model.transcribe(audio_file_paths, batch_size=args.acoustic_batch_size, logprobs=True)
        all_probs = [softmax(logits) for logits in all_logits]
        logging.info(f"Writing pickle files of probabilities at '{args.probs_cache_file}'...")
        if args.probs_cache_file:
            with open(args.probs_cache_file, 'wb') as f_dump:
                pickle.dump(all_probs, f_dump)

    wer_dist_greedy = 0
    cer_dist_greedy = 0
    words_count = 0
    chars_count = 0
    for batch_idx, probs in enumerate(all_probs):
        preds = np.argmax(probs, axis=1)
        preds_tensor = torch.tensor(preds, device='cpu').unsqueeze(0)
        pred_text = asr_model._wer.ctc_decoder_predictions_tensor(preds_tensor)[0]
        pred_split_w = pred_text.split()
        target_split_w = target_transcripts[batch_idx].split()
        pred_split_c = list(pred_text)
        target_split_c = list(target_transcripts[batch_idx])
        wer_dist = editdistance.eval(target_split_w, pred_split_w)
        cer_dist = editdistance.eval(target_split_w, pred_split_w)
        wer_dist_greedy += wer_dist
        cer_dist_greedy += cer_dist
        words_count += len(target_split_w)
        chars_count += len(target_split_c)

    logging.info('Greedy WER/CER = {:.2%}/{:.2%}'.format(wer_dist_greedy / words_count, cer_dist_greedy / chars_count))

    # delete the model to free the memory
    del asr_model

    if args.decoding_mode == "beamsearch_ngram":
        if not os.path.exists(args.kenlm_model_file):
            raise FileNotFoundError(f"Could not find the KenLM model file '{args.kenlm_model_file}'.")
        lm_path = args.kenlm_model_file
    else:
        lm_path = None

    if args.decoding_mode in ["beamsearch_ngram", "beamsearch"]:
        params = {'beam_width': args.beam_width, 'beam_alpha': args.beam_alpha, 'beam_beta': args.beam_beta}
        hp_grid = ParameterGrid(params)
        hp_grid = list(hp_grid)
        if not os.path.exists(args.preds_output_folder):
            os.mkdir(args.preds_output_folder)

        for hp in hp_grid:
            hp["preds_output_file"] = os.path.join(
                args.preds_output_folder,
                f"preds_out_width{hp['beam_width']}_alpha{hp['beam_alpha']}_beta{hp['beam_beta']}.tsv",
            )

        logging.info(f"==============================Starting the beam search decoding===============================")
        logging.info(f"Grid search size: {len(hp_grid)}")
        logging.info(f"It may take some time...")
        logging.info(f"==============================================================================================")
        for hp in hp_grid:
            preds_output_file = os.path.join(
                args.preds_output_folder,
                f"preds_out_width{hp['beam_width']}_alpha{hp['beam_alpha']}_beta{hp['beam_beta']}.tsv",
            )
            beam_search_eval(
                all_probs=all_probs,
                target_transcripts=target_transcripts,
                model_tokenizer=model_tokenizer,
                preds_output_file=preds_output_file,
                lm_path=lm_path,
                beam_width=hp["beam_width"],
                beam_alpha=hp["beam_alpha"],
                beam_beta=hp["beam_beta"],
                beam_batch_size=args.beam_batch_size,
                progress_bar=True,
            )


if __name__ == '__main__':
    main()
