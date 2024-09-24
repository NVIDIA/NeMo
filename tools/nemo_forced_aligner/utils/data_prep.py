# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import json
from dataclasses import dataclass, field
from pathlib import Path
import soundfile as sf
import torch
import utils.constants as constants
from tqdm.auto import tqdm
from utils.tokenization.char_based import CharBasedAligner
from utils.tokenization.token_based import TokenBasedAligner
from utils.units import Batch

from nemo.utils import logging


def is_entry_in_any_lines(manifest_filepath, entry):
    """
    Returns True if entry is a key in any of the JSON lines in manifest_filepath
    """

    entry_in_manifest = False

    with open(manifest_filepath, 'r') as f:
        for line in f:
            data = json.loads(line)

            if entry in data:
                entry_in_manifest = True

    return entry_in_manifest


def is_entry_in_all_lines(manifest_filepath, entry):
    """
    Returns True is entry is a key in all of the JSON lines in manifest_filepath.
    """
    with open(manifest_filepath, 'r') as f:
        for line in f:
            data = json.loads(line)

            if entry not in data:
                return False

    return True


def create_utt_batch(
    manifest_lines_batch,
    model,
    separator,
    align_using_text,
    align_using_pred_text,
    audio_filepath_parts_in_utt_id,
    output_timestep_duration,
    simulate_cache_aware_streaming=False,
    use_buffered_chunked_streaming=False,
    buffered_chunk_params={},
    normalizer=None,
    normalization_params={},
):
    """
    Returns:
        log_probs, y, T, U (y and U are s.t. every other token is a blank) - these are the tensors we will need
            during Viterbi decoding.
        utt_obj_batch: a list of Utterance objects for every utterance in the batch.
        output_timestep_duration: a float indicating the duration of a single output timestep from
            the ASR Model.
    """

    batch = Batch.get_batch(manifest_lines_batch)

    if not use_buffered_chunked_streaming:
        if not simulate_cache_aware_streaming:
            with torch.no_grad():
                hypotheses = model.transcribe(batch.audio_filepaths, return_hypotheses=True, batch_size=batch.B)
        else:
            with torch.no_grad():
                hypotheses = model.transcribe_simulate_cache_aware_streaming(
                    batch.audio_filepaths, return_hypotheses=True, batch_size=batch.B
                )

        # if hypotheses form a tuple (from Hybrid model), extract just "best" hypothesis
        if type(hypotheses) == tuple and len(hypotheses) == 2:
            hypotheses = hypotheses[0]

        for hypothesis in hypotheses:
            batch.log_probs_list.append(hypothesis.y_sequence)
            batch.T_list.append(hypothesis.y_sequence.shape[0])
            batch.pred_texts.append(hypothesis.text)
    else:
        delay = buffered_chunk_params["delay"]
        model_stride_in_secs = buffered_chunk_params["model_stride_in_secs"]
        tokens_per_chunk = buffered_chunk_params["tokens_per_chunk"]
        for l in tqdm(batch.audio_filepaths, desc="Sample:"):
            model.reset()
            model.read_audio_file(l, delay, model_stride_in_secs)
            hyp, logits = model.transcribe(tokens_per_chunk, delay, keep_logits=True)
            batch.log_probs_list.append(logits)
            batch.T_list.append(logits.shape[0])
            batch.pred_texts.append(hyp)

    # we loop over every line in the manifest that is in our current batch,
    # and record the y (list of tokens, including blanks), U (list of lengths of y) and
    # token_info_batch, word_info_batch, segment_info_batch

    batch.set_utterances(
        audio_filepath_parts_in_utt_id=audio_filepath_parts_in_utt_id, align_using_text=align_using_text
    )

    if hasattr(model, 'tokenizer'):
        if hasattr(model, 'blank_id'):
            constants.BLANK_ID = model.blank_id
        else:
            constants.BLANK_ID = len(model.tokenizer.vocab)

        aligner = TokenBasedAligner(tokenizer=model.tokenizer)
        V = len(model.tokenizer.vocab) + 1

    elif hasattr(model.decoder, "vocabulary"):  # i.e. tokenization is simply character-based
        constants.BLANK_ID = len(model.decoder.vocabulary)  # TODO: check this is correct
        constants.SPACE_ID = model.decoder.vocabulary.index(" ")
        aligner = CharBasedAligner(vocabulary=model.decoder.vocabulary)
        V = len(model.decoder.vocabulary) + 1

    else:
        raise RuntimeError("Cannot get tokens of this model.")

    for utt, T in zip(batch.utterances, batch.T_list):
        if align_using_text:
            if normalizer is not None:
                utt.text.text = normalizer.normalize(
                    text=utt.text.text, pred_text=utt.pred_text.text, **normalization_params
                )
            aligner.align(utt.text, T, separator=separator)
        if align_using_pred_text:
            aligner.align(utt.pred_text, T)

    batch.to_tensor(V=V)

    # calculate output_timestep_duration if it is None
    if output_timestep_duration is None:
        if not 'window_stride' in model.cfg.preprocessor:
            raise ValueError(
                "Don't have attribute 'window_stride' in 'model.cfg.preprocessor' => cannot calculate "
                " model_downsample_factor => stopping process"
            )

        if not 'sample_rate' in model.cfg.preprocessor:
            raise ValueError(
                "Don't have attribute 'sample_rate' in 'model.cfg.preprocessor' => cannot calculate start "
                " and end time of segments => stopping process"
            )

        with sf.SoundFile(batch.audio_filepaths[0]) as f:
            audio_dur = f.frames / f.samplerate
        n_input_frames = audio_dur / model.cfg.preprocessor.window_stride
        model_downsample_factor = round(n_input_frames / int(batch.T[0]))

        output_timestep_duration = (
            model.preprocessor.featurizer.hop_length * model_downsample_factor / model.cfg.preprocessor.sample_rate
        )

        logging.info(
            f"Calculated that the model downsample factor is {model_downsample_factor}"
            f" and therefore the ASR model output timestep duration is {output_timestep_duration}"
            " -- will use this for all batches"
        )

        batch.output_timestep_duration = output_timestep_duration

    return batch
