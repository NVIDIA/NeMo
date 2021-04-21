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

# This script would evaluate an N-gram language model trained with KenLM library (https://github.com/kpu/kenlm) in
# fusion with beam search decoders on top of a trained ASR model. NeMo's beam search decoders are capable of using the
# KenLM's N-gram models to find the best candidates. Currently this script supports BPE-based encodings and models.
# You may train the LM model with 'scripts/ngram_lm/train_kenlm.py'.
#
# USAGE: python eval_beamsearch_ngram.py --nemo_model_file <path to the .nemo file of the model> \
#                                         --input_manifest <path to the evaluation Json manifest file \
#                                         --kenlm_model_file <path to the binary KenLM model> \
#                                         --beam_width <list of the beam widths> \
#                                         --beam_alpha <list of the beam alphas> \
#                                         --beam_width <list of the beam betas> \
#                                         --preds_output_folder <optional folder to store the predictions> \
#                                         ...
#
# The script would initially load the ASR model and predict the outputs of the model's encoder as log probabilities.
# This part would be computed in batches on a device selected by '--device', which can be CPU ('--device='cpu') or
# a single GPU ('--device=cuda:0'). The batch size of this part can get specified by '--acoustic_batch_size'.
# Then greedy decoding is done on the outputs and greedy WER and CER are calculated.
# If decoding_mode is set to 'beamsearch' or 'beamsearch_ngram', beam search decoding is also done on the outputs.
#
# The beam search decoder would calculated the scores as:
#
#   final_score = acoustic_score + beam_alpha*lm_score - beam_beta*seq_length
#   beam_alpha: Specifies the The amount of importance to place on the N-gram language model.
#               Larger alpha means more importance on the LM and less importance on the acoustic model.
#   beam_beta: A penalty term given to longer word sequences. Larger beta will result in shorter sequences.
#
# The results would be reported in Word Error Rate (WER) and Character Error Rate (CER). The results if the best
# candidate is selected from the candidates is also reported as the best WER/CER. It can show how good the predicted
# candidates are.
#
# Hyperparameter Grid Search:
# Beam search decoding with N-gram LM has three main hyperparameters: beam_width, beam_alpha, and beam_beta.
# The accuracy of the model is dependent to the values of these parameters, specially beam_alpha and beam_beta.
# You may specify a single or list of values for each of these parameters to perform grid search. It would perform the
# beam search decoding on all the combinations of the these three hyperparameters. For instance, the following set of
# parameters would results in 2*1*2=4 beam search decodings:
#
# python eval_beamsearch_ngram.py ... \
#                     --beam_width 64 128 \
#                     --beam_alpha 1.0 \
#                     --beam_beta 1.0 0.5 \
#
# The following is the list of the arguments for this script:
#
# Args:
#   --nemo_model_file: The path of the '.nemo' file of the ASR model to get evaluated
#
#   --input_manifest: The manifest file of the evaluation set.
#       The manifest Json file need to contain json formatted samples per each line like this
#       {"audio_filepath": "/data_path/file1.wav", "text": "The transcript of the audio file."}
#
#   --kenlm_model_file: The path to of KenLM binary model file.
#       The N-gram LM model can get trained by 'scripts/ngram_lm/train_kenlm.py'.
#
#   --preds_output_folder: The path to an optional folder where the predictions are stored.
#       The top K (K=beam width) candidates predicted by the beam search decoder would be stored in
#       tab separated files ('.tsv'). The results for each combination of beam_width, beam_alpha, and beam_beta are
#       stored in a file named 'preds_out_width{beam_width}_alpha{beam_alpha}_beta{beam_beta}.tsv' under the folder
#       specified by '--preds_output_folder'. Each line contains a candidate with its corresponding score. For example,
#       a prediction file would have 25*4=100 lines if beam_width is 25 and there are 4 samples in the manifest file.
#
#   --probs_cache_file: The cache file for storing the outputs of the model
#       The log probabilities produced from the model's encoder can get stored in a pickle file so that next time, the
#       first step which ic calculating the log probabilities can get skipped.
#
#    --acoustic_batch_size: The batch size to calculate log probabilities
#       You may use the largest batch size feasible to speed up the step of calculating the log probabilities
#
#    --use_amp: Whether to use AMP if available to calculate log probabilities
#       Using AMP to calculate the log probabilities can speed up this step and also makes it possible to use
#       larger batch sizes for '--acoustic_batch_size'
#
#    --device: The device to load the model onto to calculate log probabilities, defaults to 'cuda'
#       It can 'cpu', 'cuda', 'cuda:0', 'cuda:1', ...
#       Currently multi-GPU is not supported but '--probs_cache_file' can help to avoid repeated calculations
#
#    --decoding_mode: The decoding scheme to be used for evaluation, defaults to 'beamsearch_ngram'
#       "greedy": Just greedy decoding is done, and no beam search decoding is performed
#       "beamsearch": The beam search decoding is done but without using the N-gram language model, final results
#           would be equivalent to setting the weight of LM (beam_beta) to zero
#       "beamsearch_ngram": The beam search decoding is done with N-gram LM
#
#    --beam_width: The width or list of the widths of the beam search decoding
#       Width of the beam search specifies the number of top candidates/predictions it would search for
#       You may pass a single value or list of values to perform grid search.
#       Larger beams result in more accurate but slower predictions
#
#    --beam_alpha: The alpha parameter or list of the alphas for the beam search decoding
#       The amount of importance to place on the N-gram language model.
#
#    --beam_beta: The beta parameter or list of the betas for the beam search decoding
#       A penalty term given to longer word sequences. Larger beta will result in shorter sequences.
#
#    --beam_batch_size: The batch size to be used for beam search decoding
#       Larger batch size may use larger memory but may be a little faster, not significantly.


# Please check train_kenlm.py to find out why we need TOKEN_OFFSET
TOKEN_OFFSET = 100

import argparse
import contextlib
import json
import os
import pickle

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
    preds_output_file=None,
    lm_path=None,
    beam_alpha=1.0,
    beam_beta=0.0,
    beam_width=128,
    beam_batch_size=128,
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
    if preds_output_file:
        out_file = open(preds_output_file, 'w')

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
            probs_batch = all_probs[batch_idx * beam_batch_size : (batch_idx + 1) * beam_batch_size]
            beams_batch = beam_search_lm.forward(log_probs=probs_batch, log_probs_length=None,)

        for beams_idx, beams in enumerate(beams_batch):
            target = target_transcripts[sample_idx + beams_idx]
            target_split_w = target.split()
            target_split_c = list(target)
            words_count += len(target_split_w)
            chars_count += len(target_split_c)
            wer_dist_min = cer_dist_min = 10000
            for candidate_idx, candidate in enumerate(beams):
                # Need to shift by TOKEN_OFFSET to retrieve the original sub-word id
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
                if preds_output_file:
                    out_file.write('{}\t{}\n'.format(pred_text, score))
            wer_dist_best += wer_dist_min
            cer_dist_best += cer_dist_min
        sample_idx += len(probs_batch)

    if preds_output_file:
        out_file.close()
        logging.info(f"Stored the predictions of beam search decoding at '{preds_output_file}'.")

    if lm_path:
        logging.info(
            'WER/CER with beam search decoding and N-gram model = {:.2%}/{:.2%}'.format(
                wer_dist_first / words_count, cer_dist_first / chars_count
            )
        )
    else:
        logging.info(
            'WER/CER with beam search decoding = {:.2%}/{:.2%}'.format(
                wer_dist_first / words_count, cer_dist_first / chars_count
            )
        )
    logging.info(
        'Best WER/CER in candidates = {:.2%}/{:.2%}'.format(wer_dist_best / words_count, cer_dist_best / chars_count)
    )
    logging.info(f"=================================================================================")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate an ASR model with beam search decoding and n-gram KenLM language model.'
    )
    parser.add_argument(
        "--nemo_model_file", required=True, type=str, help="The path of the '.nemo' file of the ASR model"
    )
    parser.add_argument(
        "--kenlm_model_file", required=False, default=None, type=str, help="The path of the KenLM binary model file"
    )
    parser.add_argument("--input_manifest", required=True, type=str, help="The manifest file of the evaluation set")
    parser.add_argument(
        "--preds_output_folder", default=None, type=str, help="The optional folder where the predictions are stored"
    )
    parser.add_argument(
        "--probs_cache_file", default=None, type=str, help="The cache file for storing the outputs of the model"
    )
    parser.add_argument(
        "--acoustic_batch_size", default=16, type=int, help="The batch size to calculate log probabilities"
    )
    parser.add_argument(
        "--device", default="cuda", type=str, help="The device to load the model onto to calculate log probabilities"
    )
    parser.add_argument(
        "--use_amp", action="store_true", help="Whether to use AMP if available to calculate log probabilities"
    )
    parser.add_argument(
        "--decoding_mode",
        choices=["greedy", "beamsearch", "beamsearch_ngram"],
        default="beamsearch_ngram",
        type=str,
        help="The decoding scheme to be used for evaluation.",
    )
    parser.add_argument(
        "--beam_width",
        required=True,
        type=int,
        nargs="+",
        help="The width or list of the widths for the beam search decoding",
    )
    parser.add_argument(
        "--beam_alpha",
        required=True,
        type=float,
        nargs="+",
        help="The alpha parameter or list of the alphas for the beam search decoding",
    )
    parser.add_argument(
        "--beam_beta",
        required=True,
        type=float,
        nargs="+",
        help="The beta parameter or list of the betas for the beam search decoding",
    )
    parser.add_argument(
        "--beam_batch_size", default=128, type=int, help="The batch size to be used for beam search decoding"
    )
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
    # audio_file_paths = audio_file_paths[0:100]

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
        if args.probs_cache_file:
            logging.info(f"Writing pickle files of probabilities at '{args.probs_cache_file}'...")
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
        cer_dist = editdistance.eval(target_split_c, pred_split_c)

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

    # 'greedy' decoding_mode would skip the beam search decoding
    if args.decoding_mode in ["beamsearch_ngram", "beamsearch"]:
        params = {'beam_width': args.beam_width, 'beam_alpha': args.beam_alpha, 'beam_beta': args.beam_beta}
        hp_grid = ParameterGrid(params)
        hp_grid = list(hp_grid)

        logging.info(f"==============================Starting the beam search decoding===============================")
        logging.info(f"Grid search size: {len(hp_grid)}")
        logging.info(f"It may take some time...")
        logging.info(f"==============================================================================================")

        if args.preds_output_folder and not os.path.exists(args.preds_output_folder):
            os.mkdir(args.preds_output_folder)
        for hp in hp_grid:
            if args.preds_output_folder:
                preds_output_file = os.path.join(
                    args.preds_output_folder,
                    f"preds_out_width{hp['beam_width']}_alpha{hp['beam_alpha']}_beta{hp['beam_beta']}.tsv",
                )
            else:
                preds_output_file = None
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
