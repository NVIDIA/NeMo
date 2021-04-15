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


def greedy_decoder(logits):
    indices = np.argmax(logits, axis=1)
    res = []
    cur = -1
    for idx in indices:
        if idx != cur:
            cur = idx
            if cur != logits.shape[1] - 1:
                res.append(idx)
    return res


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate an ASR model with beam search decoding and an n-gram KenLM language model.'
    )
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--kenlm_model_path", required=True, type=str)
    parser.add_argument("--input_manifest", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--beam_width", required=True, type=int)
    parser.add_argument("--beam_alpha", required=True, type=float)
    parser.add_argument("--beam_beta", required=True, type=float)
    parser.add_argument("--acoustic_batch_size", default=16, required=True, type=int)
    parser.add_argument("--beam_batch_size", default=16, required=True, type=int)
    parser.add_argument("--device", default="cuda:0", required=True, type=str)
    args = parser.parse_args()

    logging.info(f"BEAM WIDTH : {args.beam_width}")
    logging.info(f"BEAM ALPHA : {args.beam_alpha}")
    logging.info(f"BEAM BETA : {args.beam_beta}")

    asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(args.model_path, map_location=torch.device(args.device))
    asr_model.preprocessor.featurizer.dither = 0
    asr_model.preprocessor.featurizer.pad_to = 0
    # Set model to inference mode
    asr_model.eval()

    vocab = asr_model.tokenizer.tokenizer.get_vocab()
    vocab = list(vocab.keys())

    beam_search_lm = nemo_asr.modules.BeamSearchDecoderWithLM(
        vocab=[chr(idx + TOKEN_OFFSET) for idx in range(len(vocab))],
        beam_width=args.beam_width,
        alpha=args.beam_alpha,
        beta=args.beam_beta,
        lm_path=args.kenlm_model_path,
        num_cpus=max(os.cpu_count(), 1),
        input_tensor=False,
    )


    output_path_pkl = f"{args.output_path}.pkl"
    origs = []
    if not os.path.exists(output_path_pkl):
        probs = []
        with open(args.input_manifest, 'r') as f:
            B = args.acoustic_batch_size
            n = 0
            lines = []
            for line in tqdm(f, desc=f"Reading Manifest {args.input_manifest} ..."):
                data = json.loads(line)
                origs.append(data['text'])
                lines.append(data)
                n = n + 1
                if n < B:
                    continue
                with torch.no_grad():
                    files = [data['audio_filepath'] for data in lines]
                    logprobs = asr_model.transcribe(files, batch_size=B, logprobs=True)
                    for logprob in logprobs:
                        logits = logprob.cpu().numpy()
                        probs.append(softmax(logits))
                n = 0
                lines = []
            if n > 0:
                with torch.no_grad():
                    files = [data['audio_filepath'] for data in lines]
                    logprobs = asr_model.transcribe(files, batch_size=n, logprobs=True)
                    for logprob in logprobs:
                        logits = logprob.cpu().numpy()
                        probs.append(softmax(logits))

        logging.info(f"Writing pickle files of probabilities at {output_path_pkl}")
        with open(output_path_pkl, 'wb') as f_dump:
            pickle.dump(probs, f_dump)
    else:
        logging.info(f"Loading pickle file of probs from {output_path_pkl}")
        with open(output_path_pkl, 'r') as f:
            for line in tqdm(f, desc=f"Reading Manifest {args.input_manifest} ..."):
                data = json.loads(line)
                origs.append(data['text'])

        with open(output_path_pkl, 'rb') as f_dump:
            probs = pickle.load(f_dump)

    with open(args.output_path, 'w') as f_out:
        for idx in tqdm(range(int(np.ceil(len(probs) / args.beam_batch_size)))):
            # DISABLE TYPE CHECKING
            with nemo.core.typecheck.disable_checks():
                beams = beam_search_lm.forward(
                    log_probs=probs[idx * args.beam_batch_size: (idx + 1) * args.beam_batch_size],
                    log_probs_length=None,
                )

            for beam in beams:
                for candidate in beam:
                    pred = asr_model.tokenizer.ids_to_text([ord(c) - TOKEN_OFFSET for c in candidate[1]])
                    score = candidate[0]
                    f_out.write('{}\t{}\n'.format(pred, score))

    # WER calculations
    wer_dist = 0
    wer_dist_min = 0
    wer_words = 0
    with open(args.output_path, 'r') as f_pred:
        for idx, line in enumerate(tqdm(f_pred)):
            pred_text = line.split('\t')[0]
            pred = pred_text.split()
            score = float(line.split('\t')[1])
            if idx % args.beam_width == 0:
                # first candidate
                orig = origs[int(idx / args.beam_width)].split()
                dist = editdistance.eval(orig, pred)
                dist_min = dist
                wer_dist += dist
                wer_words += len(orig)
            else:
                dist_min = min(dist_min, editdistance.eval(orig, pred))
            if idx % args.beam_width == args.beam_width - 1:
                # last candidate
                wer_dist_min += dist_min

    print('WER = {:.2%}'.format(wer_dist / wer_words))
    print('best WER = {:.2%}'.format(wer_dist_min / wer_words))
    print()


if __name__ == '__main__':
    main()
