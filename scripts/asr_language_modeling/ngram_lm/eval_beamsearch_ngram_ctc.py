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

"""
# This script would evaluate an N-gram language model trained with KenLM library (https://github.com/kpu/kenlm) in
# fusion with beam search decoders on top of a trained ASR model with CTC decoder. To evaluate a model with 
# Transducer (RNN-T) decoder use another script 'scripts/asr_language_modeling/ngram_lm/eval_beamsearch_ngram_transducer.py'. 
# NeMo's beam search decoders are capable of using the KenLM's N-gram models
# to find the best candidates. This script supports both character level and BPE level
# encodings and models which is detected automatically from the type of the model.
# You may train the LM model with 'scripts/asr_language_modeling/ngram_lm/train_kenlm.py'.

# Config Help

To discover all arguments of the script, please run :
python eval_beamsearch_ngram_ctc.py --help
python eval_beamsearch_ngram_ctc.py --cfg job

# USAGE

python eval_beamsearch_ngram_ctc.py model_path=<path to the .nemo file of the model> \
           dataset_manifest=<path to the input evaluation JSON manifest file> \
           ctc_decoding.beam.kenlm_path=<path to the binary KenLM model> \
           ctc_decoding.beam.kenlm_type=<type of kenlm, must be word or subword > \
           ctc_decoding.beam.beam_size=[<list of the beam widths, separated with commas>] \
           ctc_decoding.beam.beam_alpha=[<list of the beam alphas, separated with commas>] \
           ctc_decoding.beam.beam_beta=[<list of the beam betas, separated with commas>] \
           preds_output_folder=<optional folder to store the predictions> \
           cache_file=null \
           decoding_strategy=<must be one of greedy_batch, greedy, pyctcdecode, beam = pyctcdecode, flashlight> \
           ...


# Grid Search for Hyper parameters

For grid search, you can provide a list of arguments as follows -

           beam_size=[4,8,16,....] \
           beam_alpha=[-2.0,-1.0,...,1.0,2.0] \
           beam_beta=[-1.0,-0.5,0.0,...,1.0] \

# Use ctc_decoding.beam.kenlm_type=subword for LM created by scripts/asr_language_modeling/ngram_lm/train_kenlm.py for subword acoustic model
# and ctc_decoding.beam.kenlm_type=word for LM created by pure KemLM binary or by scripts/asr_language_modeling/ngram_lm/train_kenlm.py for char acoustic model.

# You may find more info on how to use this script at:
# https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_language_modeling.html

"""


import contextlib
import json
import os
import pickle
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import List, Optional

import editdistance
import numpy as np
import torch
from omegaconf import MISSING, OmegaConf
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecHybridRNNTCTCModel
from nemo.collections.asr.parts.submodules import ctc_beam_decoding
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
from nemo.collections.asr.parts.utils.transcribe_utils import PunctuationCapitalization, TextProcessingConfig
from nemo.core.config import hydra_runner
from nemo.utils import logging

# fmt: off


@dataclass
class EvalBeamSearchNGramConfig:
    """
    Evaluate an ASR model with beam search decoding and n-gram KenLM language model.
    """
    # # The path of the '.nemo' file of the ASR model or the name of a pretrained model (ngc / huggingface)
    model_path: str = MISSING

    # File paths
    dataset_manifest: str = MISSING  # The manifest file of the evaluation set
    preds_output_folder: Optional[str] = None  # The optional folder where the predictions are stored
    cache_file: Optional[str] = None  # The cache file for storing the logprobs of the model

    # Parameters for inference
    batch_size: int = 16  # The batch size to calculate log probabilities
    beam_batch_size: int = 1  # The batch size to be used for beam search decoding
    
    # Set `cuda` to int to define CUDA device. If 'None', will look for CUDA
    # device anyway, and do inference on CPU only if CUDA device is not found.
    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    allow_mps: bool = False  # allow to select MPS device (Apple Silicon M-series GPU)
    amp: bool = False
    matmul_precision: str = "highest"  # Literal["highest", "high", "medium"]

    # Beam Search hyperparameters
    ctc_decoding: CTCDecodingConfig = field(default_factory=lambda: CTCDecodingConfig(
        strategy="beam", # gready, beam = pyctcdecode, flashlight
        beam = ctc_beam_decoding.BeamCTCInferConfigList(
            beam_size=[4],
            beam_alpha=[0.5], # LM weight
            beam_beta=[0.5], # length weight
            return_best_hypothesis = False,
            flashlight_cfg=ctc_beam_decoding.FlashlightConfig(lexicon_path = None),
            pyctcdecode_cfg=ctc_beam_decoding.PyCTCDecodeConfig(),
            ),
        ))
    
    text_processing: Optional[TextProcessingConfig] = field(default_factory=lambda: TextProcessingConfig(
        punctuation_marks = ".,?",
        separate_punctuation = False,
        do_lowercase = False,
        rm_punctuation = False,
    ))


def beam_search_eval(
    model: nemo_asr.models.ASRModel,
    cfg: EvalBeamSearchNGramConfig,
    all_hypotheses: List[torch.Tensor],
    target_transcripts: List[str],
    preds_output_file: str = None,
    beam_alpha: float = 1.0,
    beam_beta: float = 0.0,
    beam_size: int = 4,
    beam_batch_size: int = 1,
    progress_bar: bool = True,
    punctuation_capitalization: PunctuationCapitalization = None,
):
    level = logging.getEffectiveLevel()
    logging.setLevel(logging.CRITICAL)
    # Reset config
    if isinstance(model, EncDecHybridRNNTCTCModel):
        model.change_decoding_strategy(decoding_cfg=None, decoder_type="ctc")
    else:
        model.change_decoding_strategy(None)

    # Override the beam search config with current search candidate configuration
    model.cfg.decoding = CTCDecodingConfig(
        strategy=cfg.ctc_decoding.strategy,
        preserve_alignments=cfg.ctc_decoding.preserve_alignments,
        compute_timestamps=cfg.ctc_decoding.compute_timestamps,
        word_seperator=cfg.ctc_decoding.word_seperator,
        ctc_timestamp_type=cfg.ctc_decoding.ctc_timestamp_type,
        batch_dim_index=cfg.ctc_decoding.batch_dim_index,
        greedy=cfg.ctc_decoding.greedy,
        confidence_cfg=cfg.ctc_decoding.confidence_cfg,
        temperature=cfg.ctc_decoding.temperature,
        beam = ctc_beam_decoding.BeamCTCInferConfig(beam_size=beam_size,
                                                    beam_alpha=beam_alpha,
                                                    beam_beta=beam_beta,
                                                    kenlm_path=cfg.ctc_decoding.beam.kenlm_path,
                                                    preserve_alignments=cfg.ctc_decoding.beam.preserve_alignments,
                                                    compute_timestamps=cfg.ctc_decoding.beam.compute_timestamps,
                                                    flashlight_cfg=cfg.ctc_decoding.beam.flashlight_cfg,
                                                    pyctcdecode_cfg=cfg.ctc_decoding.beam.pyctcdecode_cfg,
                                                    return_best_hypothesis=cfg.ctc_decoding.beam.return_best_hypothesis),
        )

    # Update model's decoding strategy
    if isinstance(model, EncDecHybridRNNTCTCModel):
        model.change_decoding_strategy(model.cfg.decoding, decoder_type='ctc')
        decoding = model.ctc_decoding
    else:
        model.change_decoding_strategy(model.cfg.decoding)
        decoding = model.decoding
    logging.setLevel(level)

    wer_dist_first = cer_dist_first = 0
    wer_dist_best = cer_dist_best = 0
    words_count = 0
    chars_count = 0
    sample_idx = 0
    if preds_output_file:
        out_file = open(preds_output_file, 'w', encoding='utf_8', newline='\n')
    
    if progress_bar:
        it = tqdm(
            range(int(np.ceil(len(all_hypotheses) / beam_batch_size))),
            desc=f"Beam search decoding with width={beam_size}, alpha={beam_alpha}, beta={beam_beta}",
            ncols=120,
        )
    else:
        it = range(int(np.ceil(len(all_hypotheses) / beam_batch_size)))
    for batch_idx in it:
        # disabling type checking
        probs_batch = [x.alignments for x in all_hypotheses[batch_idx * beam_batch_size : (batch_idx + 1) * beam_batch_size]]
        probs_lens = torch.tensor([prob.shape[0] for prob in probs_batch])
        with torch.no_grad():
            packed_batch = torch.zeros(len(probs_batch), max(probs_lens), probs_batch[0].shape[-1], device='cpu')

            for prob_index in range(len(probs_batch)):
                packed_batch[prob_index, : probs_lens[prob_index], :] = torch.tensor(
                    probs_batch[prob_index], device=packed_batch.device, dtype=packed_batch.dtype
                )

            _, beams_batch = decoding.ctc_decoder_predictions_tensor(
                packed_batch, decoder_lengths=probs_lens, return_hypotheses=True,
            )

        for beams_idx, beams in enumerate(beams_batch):
            target = target_transcripts[sample_idx + beams_idx]
            target_split_w = target.split()
            target_split_c = list(target)
            words_count += len(target_split_w)
            chars_count += len(target_split_c)
            wer_dist_min = cer_dist_min = 10000
            for candidate_idx, candidate in enumerate(beams):  # type: (int, ctc_beam_decoding.rnnt_utils.Hypothesis)
                pred_text = candidate.text
                if cfg.text_processing.do_lowercase:
                    pred_text = punctuation_capitalization.do_lowercase([pred_text])[0]
                if cfg.text_processing.rm_punctuation:
                    pred_text = punctuation_capitalization.rm_punctuation([pred_text])[0]
                if cfg.text_processing.separate_punctuation:
                    pred_text = punctuation_capitalization.separate_punctuation([pred_text])[0]
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

                score = candidate.score
                if preds_output_file:
                    out_file.write('{}\t{}\n'.format(pred_text, score))
            wer_dist_best += wer_dist_min
            cer_dist_best += cer_dist_min
        sample_idx += len(probs_batch)

    if preds_output_file:
        out_file.close()
        logging.info(f"Stored the predictions of beam search decoding at '{preds_output_file}'.")

    if cfg.ctc_decoding.beam.kenlm_path:
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
        'Oracle WER/CER in candidates with perfect LM= {:.2%}/{:.2%}'.format(
            wer_dist_best / words_count, cer_dist_best / chars_count
        )
    )
    logging.info(f"=================================================================================")

    return wer_dist_first / words_count, cer_dist_first / chars_count


@hydra_runner(config_path=None, config_name='EvalBeamSearchNGramConfig', schema=EvalBeamSearchNGramConfig)
def main(cfg: EvalBeamSearchNGramConfig):
    assert cfg.beam_batch_size==1 # TODO fix bug for Flashlight beamsearch with beam_batch_size>1 and remove this assert

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)  # type: EvalBeamSearchNGramConfig

    # setup GPU
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    if cfg.cuda is None:
        if torch.cuda.is_available():
            map_location = torch.device('cuda:0')
        elif cfg.allow_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logging.warning(
                "MPS device (Apple Silicon M-series GPU) support is experimental."
                " Env variable `PYTORCH_ENABLE_MPS_FALLBACK=1` should be set in most cases to avoid failures."
            )
            map_location = torch.device('mps')
        else:
            map_location = torch.device('cpu')
    elif cfg.cuda == -1:
        map_location = torch.device('cpu')
    else:
        map_location = torch.device(f'cuda:{cfg.cuda}')

    logging.info(f"Inference will be done on device: {map_location}")

    if cfg.model_path.endswith('.nemo'):
        asr_model = nemo_asr.models.ASRModel.restore_from(cfg.model_path, map_location=torch.device(map_location))
    else:
        logging.warning(
            "model_path does not end with .nemo, therefore trying to load a pretrained model with this name."
        )
        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            cfg.model_path, map_location=torch.device(map_location)
        )

    target_transcripts = []
    manifest_dir = Path(cfg.dataset_manifest).parent
    with open(cfg.dataset_manifest, 'r', encoding='utf_8') as manifest_file:
        audio_file_paths = []
        for line in tqdm(manifest_file, desc=f"Reading Manifest {cfg.dataset_manifest} ...", ncols=120):
            data = json.loads(line)
            audio_file = Path(data['audio_filepath'])
            if not audio_file.is_file() and not audio_file.is_absolute():
                audio_file = manifest_dir / audio_file
            target_transcripts.append(data['text'])
            audio_file_paths.append(str(audio_file.absolute()))

    punctuation_capitalization = PunctuationCapitalization(cfg.text_processing.punctuation_marks)
    if cfg.text_processing.do_lowercase:
        target_transcripts = punctuation_capitalization.do_lowercase(target_transcripts)
    if cfg.text_processing.rm_punctuation:
        target_transcripts = punctuation_capitalization.rm_punctuation(target_transcripts)
    if cfg.text_processing.separate_punctuation:
        target_transcripts = punctuation_capitalization.separate_punctuation(target_transcripts)

    if cfg.cache_file and os.path.exists(cfg.cache_file):
        logging.info(f"Found a pickle file of probabilities at '{cfg.cache_file}'.")
        logging.info(f"Loading the cached pickle file of probabilities from '{cfg.cache_file}' ...")
        with open(cfg.cache_file, 'rb') as probs_file:
            all_hypotheses = pickle.load(probs_file)

        if len(all_hypotheses) != len(audio_file_paths):
            raise ValueError(
                f"The number of samples in the probabilities file '{cfg.cache_file}' does not "
                f"match the manifest file. You may need to delete the probabilities cached file."
            )
    else:

        @contextlib.contextmanager
        def default_autocast():
            yield

        if cfg.amp:
            if torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
                logging.info("AMP is enabled!\n")
                autocast = torch.cuda.amp.autocast

            else:
                autocast = default_autocast
        else:

            autocast = default_autocast

        with autocast():
            with torch.no_grad():
                if isinstance(asr_model, EncDecHybridRNNTCTCModel):
                    asr_model.cur_decoder = 'ctc'
                all_hypotheses = asr_model.transcribe(audio_file_paths, batch_size=cfg.batch_size, return_hypotheses=True)
                if type(all_hypotheses) == tuple and len(all_hypotheses) == 2: # if transcriptions form a tuple of (best_hypotheses, all_hypotheses)
                    all_hypotheses = all_hypotheses[1]

        if cfg.cache_file:
            os.makedirs(os.path.split(cfg.cache_file)[0], exist_ok=True)
            logging.info(f"Writing pickle files of probabilities at '{cfg.cache_file}'...")
            with open(cfg.cache_file, 'wb') as f_dump:
                pickle.dump(all_hypotheses, f_dump)

    wer_dist_greedy = 0
    cer_dist_greedy = 0
    words_count = 0
    chars_count = 0
    for batch_idx, hypotheses in enumerate(all_hypotheses):
        preds = np.argmax(hypotheses.alignments, axis=1)
        preds_tensor = torch.tensor(preds, device='cpu').unsqueeze(0)
        if isinstance(asr_model, EncDecHybridRNNTCTCModel):
            pred_text = asr_model.ctc_decoding.ctc_decoder_predictions_tensor(preds_tensor)[0][0]
        else:
            pred_text = asr_model.wer.decoding.ctc_decoder_predictions_tensor(preds_tensor)[0][0]

        if cfg.text_processing.do_lowercase:
            pred_text = punctuation_capitalization.do_lowercase([pred_text])[0]
        if cfg.text_processing.rm_punctuation:
            pred_text = punctuation_capitalization.rm_punctuation([pred_text])[0]
        if cfg.text_processing.separate_punctuation:
            pred_text = punctuation_capitalization.separate_punctuation([pred_text])[0]

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

    asr_model = asr_model.to('cpu')

    if cfg.ctc_decoding.beam.return_best_hypothesis==True:
        raise ValueError("Works only with cfg.ctc_decoding.beam.return_best_hypothesis=False")

    # 'greedy' decoding_mode would skip the beam search decoding
    if cfg.ctc_decoding.strategy in ["beam", "pyctcdecode", "flashlight"]:
        if cfg.ctc_decoding.beam.beam_size is None or cfg.ctc_decoding.beam.beam_alpha is None or cfg.ctc_decoding.beam.beam_beta is None:
            raise ValueError("beam_size, beam_alpha and beam_beta are needed to perform beam search decoding.")
        params = {'beam_size': cfg.ctc_decoding.beam.beam_size, 'beam_alpha': cfg.ctc_decoding.beam.beam_alpha, 'beam_beta': cfg.ctc_decoding.beam.beam_beta}
        hp_grid = ParameterGrid(params)
        hp_grid = list(hp_grid)

        best_wer_beam_size, best_cer_beam_size = None, None
        best_wer_alpha, best_cer_alpha = None, None
        best_wer_beta, best_cer_beta = None, None
        best_wer, best_cer = 1e6, 1e6

        logging.info(f"==============================Starting the beam search decoding===============================")
        logging.info(f"Grid search size: {len(hp_grid)}")
        logging.info(f"It may take some time...")
        logging.info(f"==============================================================================================")

        if cfg.preds_output_folder and not os.path.exists(cfg.preds_output_folder):
            os.mkdir(cfg.preds_output_folder)
        for hp in hp_grid:
            if cfg.preds_output_folder:
                preds_output_file = os.path.join(
                    cfg.preds_output_folder,
                    f"preds_out_size{hp['beam_size']}_alpha{hp['beam_alpha']}_beta{hp['beam_beta']}.tsv",
                )
            else:
                preds_output_file = None

            candidate_wer, candidate_cer = beam_search_eval(
                asr_model,
                cfg,
                all_hypotheses=all_hypotheses,
                target_transcripts=target_transcripts,
                preds_output_file=preds_output_file,
                beam_size=hp["beam_size"],
                beam_alpha=hp["beam_alpha"],
                beam_beta=hp["beam_beta"],
                beam_batch_size=cfg.beam_batch_size,
                progress_bar=True,
                punctuation_capitalization=punctuation_capitalization,
            )

            if candidate_cer < best_cer:
                best_cer_beam_size = hp["beam_size"]
                best_cer_alpha = hp["beam_alpha"]
                best_cer_beta = hp["beam_beta"]
                best_cer = candidate_cer

            if candidate_wer < best_wer:
                best_wer_beam_size = hp["beam_size"]
                best_wer_alpha = hp["beam_alpha"]
                best_wer_beta = hp["beam_beta"]
                best_wer = candidate_wer

        logging.info(
            f'Best WER Candidate = {best_wer:.2%} :: Beam size = {best_wer_beam_size}, '
            f'Beam alpha = {best_wer_alpha}, Beam beta = {best_wer_beta}'
        )

        logging.info(
            f'Best CER Candidate = {best_cer:.2%} :: Beam size = {best_cer_beam_size}, '
            f'Beam alpha = {best_cer_alpha}, Beam beta = {best_cer_beta}'
        )
        logging.info(f"=================================================================================")


if __name__ == '__main__':
    main()
