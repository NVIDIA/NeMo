# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
G2P disambiguation using an Aligner model's input embedding distances.

Does not handle OOV and leaves them as graphemes.

Example:
python aligner_g2p.py \
    --model=<model_path> \
    --manifest=<manifest_path> \
    --out=<output_json_path> \
    --verbose
"""

import argparse
import json
import os

import librosa
import soundfile as sf
import torch

from nemo.collections.tts.models import AlignerModel
from nemo.collections.tts.torch.helpers import general_padding


def get_args():
    """Retrieve arguments for disambiguation.
    """
    parser = argparse.ArgumentParser("G2P disambiguation using Aligner input embedding distances.")
    # TODO(jocelynh): Make this required=False with default download from NGC once ckpt uploaded
    parser.add_argument('--model', required=True, type=str, help="Path to Aligner model checkpoint (.nemo file).")
    parser.add_argument(
        '--manifest',
        required=True,
        type=str,
        help="Path to data manifest. Each entry should contain the path to the audio file as well as the text in graphemes.",
    )
    parser.add_argument(
        '--out', required=True, type=str, help="Path to output file where disambiguations will be written."
    )
    parser.add_argument(
        '--sr',
        required=False,
        default=22050,
        type=int,
        help="Target sample rate to load the dataset. Should match what the model was trained on.",
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="If set to True, logs scores for each disambiguated word in disambiguation_logs.txt.",
    )
    args = parser.parse_args()
    return args


def load_and_prepare_audio(aligner, audio_path, target_sr, device):
    """Loads and resamples audio to target sample rate (if necessary), and preprocesses for Aligner input.
    """
    # Load audio and get length for preprocessing
    audio_data, orig_sr = sf.read(audio_path)
    if orig_sr != target_sr:
        audio_data = librosa.core.resample(audio_data, orig_sr=orig_sr, target_sr=target_sr)

    audio = torch.tensor(audio_data, dtype=torch.float, device=device).unsqueeze(0)
    audio_len = torch.tensor(audio_data.shape[0], device=device).long().unsqueeze(0)

    # Generate spectrogram
    spec, spec_len = aligner.preprocessor(input_signal=audio, length=audio_len)

    return spec, spec_len


def get_mean_distance_for_word(l2_dists, durs, start_token, num_tokens):
    """Calculates the mean distance between text and audio embeddings given a range of text tokens.

    This is the same function as in the Aligner Inference tutorial notebook.
    """
    # Need to calculate which audio frame we start on by summing all durations up to the start token's duration
    start_frame = torch.sum(durs[:start_token]).data

    total_frames = 0
    dist_sum = 0

    # Loop through each text token
    for token_ind in range(start_token, start_token + num_tokens):
        # Loop through each frame for the given text token
        for frame_ind in range(start_frame, start_frame + durs[token_ind]):
            # Recall that the L2 distance matrix is shape [spec_len, text_len]
            dist_sum += l2_dists[frame_ind, token_ind]

        # Update total frames so far & the starting frame for the next token
        total_frames += durs[token_ind]
        start_frame += durs[token_ind]

    return dist_sum / total_frames


def disambiguate_candidates(aligner, text, spec, spec_len, device, log_file=None):
    """Retrieves and disambiguate all candidate sentences for disambiguation of a given text string.

    Assumes the text has no punctuation and has been normalized.
    Also assumes that the max number of candidates per word is a reasonable batch size.

    Note: This could be sped up if multiple words' candidates were batched, but this is conceptually easier.
    """
    # Grab original G2P result
    aligner_g2p = aligner.tokenizer.g2p
    base_g2p = aligner_g2p(text)

    ### Loop Through Words ###
    result_g2p = []
    word_start_idx = 0

    for word in text.split():
        if log_file:
            log_file.write(f"----------------\n{word}\n")

        # Retrieve the length of the word in the default G2P conversion
        g2p_default_len = len(aligner_g2p(word))

        # Check if word needs to be disambiguated
        if (word in aligner_g2p.phoneme_dict) and (not aligner_g2p.is_unique_in_phoneme_dict(word)):
            # Add candidate for each ambiguous pronunciation
            word_candidates = []
            candidate_prons_and_lengths = []

            for pron in aligner_g2p.phoneme_dict[word]:
                # Replace graphemes in the base G2P result with the current variant
                candidate = base_g2p[:word_start_idx] + pron + base_g2p[word_start_idx + g2p_default_len :]
                candidate_tokens = aligner.tokenizer.encode_from_g2p(candidate)

                word_candidates.append(candidate_tokens)
                candidate_prons_and_lengths.append((pron, len(pron)))

            ### Inference ###
            num_candidates = len(word_candidates)

            text_len = [len(toks) for toks in word_candidates]
            text_len_in = torch.tensor(text_len, device=device).long()

            # Have to pad text tokens in case different pronunciations have different lengths
            max_text_len = max(text_len)
            text_stack = []
            for i in range(num_candidates):
                padded_tokens = general_padding(
                    torch.tensor(word_candidates[i], device=device).long(), text_len[i], max_text_len
                )
                text_stack.append(padded_tokens)
            text_in = torch.stack(text_stack)

            # Repeat spectrogram and spec_len tensors to match batch size
            spec_in = spec.repeat([num_candidates, 1, 1])
            spec_len_in = spec_len.repeat([num_candidates])

            with torch.no_grad():
                soft_attn, _ = aligner(spec=spec_in, spec_len=spec_len_in, text=text_in, text_len=text_len_in)

            # Need embedding distances and duration preds to calculate mean distance for just the one word
            text_embeddings = aligner.embed(text_in).transpose(1, 2)
            l2_dists = aligner.alignment_encoder.get_dist(keys=text_embeddings, queries=spec_in).sqrt()

            durations = aligner.alignment_encoder.get_durations(soft_attn, text_len_in, spec_len_in).int()

            # Retrieve average embedding distances
            min_dist = float('inf')
            best_candidate = None
            for i in range(num_candidates):
                candidate_mean_dist = get_mean_distance_for_word(
                    l2_dists=l2_dists[i],
                    durs=durations[i],
                    start_token=word_start_idx + (1 if aligner.tokenizer.pad_with_space else 0),
                    num_tokens=candidate_prons_and_lengths[i][1],
                )
                if log_file:
                    log_file.write(f"{candidate_prons_and_lengths[i][0]} -- {candidate_mean_dist}\n")

                if candidate_mean_dist < min_dist:
                    min_dist = candidate_mean_dist
                    best_candidate = candidate_prons_and_lengths[i][0]

            if log_file:
                log_file.write(f"best candidate: {best_candidate}\n")

            result_g2p.append(' '.join(best_candidate))
        else:
            result_g2p.append(' '.join(aligner_g2p(word)))

        # Advance to phoneme index of next word
        word_start_idx += g2p_default_len + 1  # +1 for space

    if log_file:
        log_file.write(f"===\n{' '.join(result_g2p)}\n===\n")

    return result_g2p


def disambiguate_dataset(aligner, manifest_path, out_path, sr, device, verbose):
    """Disambiguates the phonemes for all words with ambiguous pronunciations in the given manifest.
    """
    log_file = open('disambiguation_logs.txt', 'w') if verbose else None

    with open(out_path, 'w') as f_out:
        with open(manifest_path, 'r') as f_in:
            count = 0

            for line in f_in:
                # Retrieve entry and base G2P conversion for full text
                entry = json.loads(line)
                text = aligner.normalizer.normalize(entry['text'])
                audio_path = entry['audio_filepath']

                # Load and preprocess audio
                spec, spec_len = load_and_prepare_audio(aligner, audio_path, sr, device)

                # Get pronunciation candidates and disambiguate
                disambiguated_text = disambiguate_candidates(aligner, text, spec, spec_len, device, log_file)

                entry['disambiguated_text'] = ' '.join(disambiguated_text)
                f_out.write(f"{json.dumps(entry)}\n")

                count += 1
                if count % 100 == 0:
                    print(f"Finished {count} entries.")

    print(f"Finished all entries, with a total of {count}.")
    log_file.close()


def main():
    args = get_args()

    # Check file paths from arguments
    if not os.path.exists(args.model):
        print("Could not find model checkpoint file: ", args.model)
    if not os.path.exists(args.manifest):
        print("Could not find data manifest file: ", args.manifest)
    if os.path.exists(args.out):
        print("Output file already exists: ", args.out)
        overwrite = input("Is it okay to overwrite it? (Y/N): ")
        if overwrite.lower() != 'y':
            print("Not overwriting output file, quitting.")
            quit()

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aligner = AlignerModel.restore_from(args.model, map_location=device)

    # Disambiguation
    print("Beginning disambiguation...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    disambiguate_dataset(aligner, args.manifest, args.out, args.sr, device, args.verbose)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
