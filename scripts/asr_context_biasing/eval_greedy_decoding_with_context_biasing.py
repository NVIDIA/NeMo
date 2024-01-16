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
#

"""
# This script would evaluate CTC and Transducer (only hybrid model) models in context biasing mode
# by applying CTC-based Word Spotter (paper link) 

# Config Help

To discover all arguments of the script, please run :
python eval_greedy_decoding_with_context_biasing.py --help
python eval_greedy_decoding_with_context_biasing.py --cfg job

# USAGE

python eval_greedy_decoding_with_context_biasing.py nemo_model_file=<path to the .nemo file of the model> \
           input_manifest=<path to the evaluation JSON manifest file \
           kenlm_model_file=<path to the binary KenLM model> \
           beam_width=[<list of the beam widths, separated with commas>] \
           beam_alpha=[<list of the beam alphas, separated with commas>] \
           preds_output_folder=<optional folder to store the predictions> \
           probs_cache_file=null \
           decoding_strategy=<greedy_batch or maes decoding>
           ...


# Grid Search for Hyper parameters

For grid search, you can provide a list of arguments as follows -

           beam_width=[4,8,16,....] \
           beam_alpha=[-2.0,-1.0,...,1.0,2.0] \

# You may find more info on how to use this script at:
# https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_language_modeling.html

"""


import contextlib
import json
import os
import pickle
import tempfile
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import List, Optional, Dict
import time
import subprocess

import editdistance
import numpy as np
import torch
from omegaconf import MISSING, OmegaConf
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.submodules import rnnt_beam_decoding
from nemo.core.config import hydra_runner
from nemo.utils import logging
# from nemo.collections.asr.parts.k2.context_graph import ContextGraph
from context_graph_ctc import ContextGraphCTC
from nemo.collections.asr.models import EncDecHybridRNNTCTCModel

from ctc_based_word_spotter import recognize_wb
from context_biasing_utils import merge_alignment_with_wb_hyps
from compute_key_words_fscore import compute_fscore


@dataclass
class EvalContextBiasingConfig:
    """
    Evaluate an ASR model with greedy decoding and context biasing.
    """
    # # The path of the '.nemo' file of the ASR model or the name of a pretrained model (ngc / huggingface)
    nemo_model_file: str = MISSING

    # File paths
    input_manifest: str = MISSING  # The manifest file of the evaluation set
    preds_output_folder: Optional[str] = None  # The optional folder where the predictions are stored
    probs_cache_file: Optional[str] = None  # The cache file for storing the logprobs of the model

    # Parameters for inference
    acoustic_batch_size: int = 128  # The batch size to calculate log probabilities
    beam_batch_size: int = 128  # The batch size to be used for beam search decoding
    device: str = "cuda"  # The device to load the model onto to calculate log probabilities
    use_amp: bool = False  # Whether to use AMP if available to calculate log probabilities
    num_workers: int = 1  # Number of workers for DataLoader
    
    # for hybrid model
    decoder_type: Optional[str] = None # [ctc, rnnt] Decoder type for hybrid ctc-rnnt model

    # The decoding scheme to be used for evaluation
    decoding_strategy: str = "greedy_batch" # ["greedy", "greedy_batch"]


    ### Context Biasing ###:
    apply_context_biasing: bool = True
    context_file: Optional[str] = None  # string with context biasing words (words splitted by space)
    beam_threshold: List[float] = field(default_factory=lambda: [5.0]) # beam pruning threshold for ctc-ws decoding
    context_score: List[float] = field(default_factory=lambda: [3.0]) # per token weight for context biasing words
    ctc_ali_token_weight: List[float] = field(default_factory=lambda: [0.6]) # weight of greedy CTC token to prevent false accept errors

    sort_logits: bool = True # do logits sorting before decoding - it reduces computation on puddings
    softmax_temperature: float = 1.00
    preserve_alignments: bool = False

    decoding: rnnt_beam_decoding.BeamRNNTInferConfig = rnnt_beam_decoding.BeamRNNTInferConfig(beam_size=128)


def decoding_step(
    asr_model: nemo_asr.models.ASRModel,
    cfg: EvalContextBiasingConfig,
    all_probs: List[torch.Tensor],
    ctc_logprobs: List[torch.Tensor],
    target_transcripts: List[str],
    audio_file_paths: List[str],
    durations: List[str],
    # preds_output_file: str = None,
    preds_output_manifest: str = None,
    beam_batch_size: int = 128,
    progress_bar: bool = True,
    context_graph: ContextGraphCTC = None,
    hp: Dict = None,
):
    
    # run CTC based WB search:
    cb_results = {}
    for idx, logits in tqdm(enumerate(ctc_logprobs), desc=f"CTC based word boosting...", ncols=120, total=len(ctc_logprobs)):
        # try:
        wb_result = recognize_wb(
            logits.numpy(),
            context_graph,
            asr_model,
            beam_threshold=hp['beam_threshold'],
            context_score=hp['context_score'],      
            keyword_thr=-5, 
            ctc_ali_token_weight=hp['ctc_ali_token_weight']
        )
        cb_results[audio_file_paths[idx]] = wb_result
        # print(f"ref: {target_transcripts[idx]}")
        # print(audio_file_paths[idx] + "\n")
    
    
    level = logging.getEffectiveLevel()
    logging.setLevel(logging.CRITICAL)
    # Reset config
    asr_model.change_decoding_strategy(None)

    # cfg.decoding.hat_ilm_weight = cfg.decoding.hat_ilm_weight * cfg.hat_subtract_ilm
    # Override the beam search config with current search candidate configuration
    cfg.decoding.return_best_hypothesis = False

    # preserve aligmnet:
    asr_model.cfg.decoding.preserve_alignments = cfg.preserve_alignments

    # Update model's decoding strategy config
    asr_model.cfg.decoding.strategy = cfg.decoding_strategy
    asr_model.cfg.decoding.beam = cfg.decoding

    # Update model's decoding strategy
    asr_model.change_decoding_strategy(asr_model.cfg.decoding)
    logging.setLevel(level)

    wer_dist_first = cer_dist_first = 0
    words_count = chars_count = sample_idx = 0

    if preds_output_manifest:
        out_manifest = open(preds_output_manifest, 'w', encoding='utf_8', newline='\n')

    # ctc part
    if cfg.decoder_type == "ctc":
        wer_dist_greedy = 0
        cer_dist_greedy = 0
        words_count = 0
        chars_count = 0
        for batch_idx, probs in enumerate(ctc_logprobs):
            preds = np.argmax(probs, axis=1)
            # do CTC based word boosting
            if cfg.apply_context_biasing and cb_results[audio_file_paths[batch_idx]]:
                # make new text by mearging alignment with ctc-wb predictions:
                boosted_text = merge_alignment_with_wb_hyps(
                    preds.numpy(),
                    asr_model,
                    cb_results[audio_file_paths[batch_idx]],
                    decoder_type="ctc",
                )
                print(f"ref   : {target_transcripts[batch_idx]}")
                print("\n" + audio_file_paths[batch_idx])
                pred_text = boosted_text
            else:
                preds_tensor = torch.tensor(preds, device='cpu').unsqueeze(0)
                if isinstance(asr_model, EncDecHybridRNNTCTCModel):
                    pred_text = asr_model.ctc_decoding.ctc_decoder_predictions_tensor(preds_tensor)[0][0]
                else:
                    pred_text = asr_model._wer.decoding.ctc_decoder_predictions_tensor(preds_tensor)[0][0]

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

            # write manifest with prediction results
            for idx, token_id in enumerate(preds):
                # # token_id = x[0][-1].item()
                if token_id == asr_model.decoder.blank_idx:
                    token_text = "-"
                else:
                    token_text = asr_model.tokenizer.ids_to_text(int(token_id))
                # alignment.append(f"{token_id}: {idx}")

            if preds_output_manifest:
                item = {'audio_filepath': audio_file_paths[batch_idx],
                        'duration': durations[batch_idx],
                        'text': target_transcripts[batch_idx],
                        'pred_text': pred_text,
                        'wer': f"{wer_dist/len(target_split_w):.4f}",}
                out_manifest.write(json.dumps(item) + "\n")

        logging.info('Greedy WER/CER = {:.2%}/{:.2%}'.format(wer_dist_greedy / words_count, cer_dist_greedy / chars_count))
        return wer_dist_greedy / words_count, cer_dist_greedy / chars_count

    # rnnt part
    elif cfg.decoder_type == "rnnt":
        if progress_bar:
            description = "Greedy_batch decoding.."
            it = tqdm(range(int(np.ceil(len(all_probs) / beam_batch_size))), desc=description, ncols=120)
        else:
            it = range(int(np.ceil(len(all_probs) / beam_batch_size)))
        for batch_idx in it:
            probs_batch = all_probs[batch_idx * beam_batch_size : (batch_idx + 1) * beam_batch_size]
            probs_lens = torch.tensor([prob.shape[-1] for prob in probs_batch])
            with torch.no_grad():
                packed_batch = torch.zeros(len(probs_batch), probs_batch[0].shape[0], max(probs_lens), device='cpu')

                for prob_index in range(len(probs_batch)):
                    packed_batch[prob_index, :, : probs_lens[prob_index]] = torch.tensor(
                        probs_batch[prob_index].unsqueeze(0), device=packed_batch.device, dtype=packed_batch.dtype
                    )
                best_hyp_batch, beams_batch = asr_model.decoding.rnnt_decoder_predictions_tensor(
                    packed_batch, probs_lens, return_hypotheses=True,
                )
            beams_batch = [[x] for x in best_hyp_batch]

            for beams_idx, beams in enumerate(beams_batch):
                target = target_transcripts[sample_idx + beams_idx]
                target_split_w = target.split()
                target_split_c = list(target)
                words_count += len(target_split_w)
                chars_count += len(target_split_c)
                # audio_id = os.path.basename(audio_file_paths[sample_idx + beams_idx])
                for candidate_idx, candidate in enumerate(beams):  # type: (int, rnnt_beam_decoding.rnnt_utils.Hypothesis)
                    
                    ###################################
                    if cfg.apply_context_biasing and cb_results[audio_file_paths[sample_idx + beams_idx]]:
                        
                        # make new text by mearging alignment with ctc-wb predictions:
                        # print("----")
                        boosted_text = merge_alignment_with_wb_hyps(
                            candidate,
                            asr_model,
                            cb_results[audio_file_paths[sample_idx + beams_idx]],
                            decoder_type="rnnt",
                        )
                        pred_text = boosted_text
                        beams[0].text = pred_text
                        # print(f"ref   : {target}")
                        # print("\n" + audio_file_paths[sample_idx + beams_idx])
                    else:
                        pred_text = candidate.text
                    
                    #######################################
                    pred_split_w = pred_text.split()
                    wer_dist = editdistance.eval(target_split_w, pred_split_w)
                    pred_split_c = list(pred_text)
                    cer_dist = editdistance.eval(target_split_c, pred_split_c)

                    if candidate_idx == 0:
                        # first candidate
                        wer_dist_tosave = wer_dist
                        wer_dist_first += wer_dist
                        cer_dist_first += cer_dist

                    # score = candidate.score
                    # if preds_output_file:       
                    #     out_file.write('{}\t{}\t{}\n'.format(audio_id, pred_text, score))

                # write manifest with prediction results
                alignment = []

                if preds_output_manifest:
                    item = {'audio_filepath': audio_file_paths[sample_idx + beams_idx],
                            'duration': durations[sample_idx + beams_idx],
                            'text': target_transcripts[sample_idx + beams_idx],
                            'pred_text': beams[0].text,
                            'wer': f"{wer_dist_tosave/len(target_split_w):.3f}",
                            'alignment': f"{alignment}"}
                    out_manifest.write(json.dumps(item) + "\n")
            
            sample_idx += len(probs_batch)

        return wer_dist_first / words_count, cer_dist_first / chars_count




@hydra_runner(config_path=None, config_name='EvalContextBiasingConfig', schema=EvalContextBiasingConfig)
def main(cfg: EvalContextBiasingConfig):
    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)  # type: EvalContextBiasingConfig

    valid_decoding_strategis = ["greedy", "greedy_batch"]
    if cfg.decoding_strategy not in valid_decoding_strategis:
        raise ValueError(
            f"Given decoding_strategy={cfg.decoding_strategy} is invalid. Available options are :\n"
            f"{valid_decoding_strategis}"
        )

    if cfg.nemo_model_file.endswith('.nemo'):
        asr_model = nemo_asr.models.ASRModel.restore_from(cfg.nemo_model_file, map_location=torch.device(cfg.device))
        # asr_model.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=[128, 128])
    else:
        logging.warning(
            "nemo_model_file does not end with .nemo, therefore trying to load a pretrained model with this name."
        )
        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            cfg.nemo_model_file, map_location=torch.device(cfg.device)
        )


    target_transcripts = []
    durations = []
    manifest_dir = Path(cfg.input_manifest).parent
    with open(cfg.input_manifest, 'r', encoding='utf_8') as manifest_file:
        audio_file_paths = []
        for line in tqdm(manifest_file, desc=f"Reading Manifest {cfg.input_manifest} ...", ncols=120):
            data = json.loads(line)
            audio_file = Path(data['audio_filepath'])
            if not audio_file.is_file() and not audio_file.is_absolute():
                audio_file = manifest_dir / audio_file
            target_transcripts.append(data['text'])
            durations.append(data['duration'])
            audio_file_paths.append(str(audio_file.absolute()))

    if cfg.probs_cache_file and os.path.exists(cfg.probs_cache_file):
        logging.info(f"Found a pickle file of probabilities at '{cfg.probs_cache_file}'.")
        logging.info(f"Loading the cached pickle file of probabilities from '{cfg.probs_cache_file}' ...")
        with open(cfg.probs_cache_file, 'rb') as probs_file:
            all_probs = pickle.load(probs_file)

        if len(all_probs) != len(audio_file_paths):
            raise ValueError(
                f"The number of samples in the probabilities file '{cfg.probs_cache_file}' does not "
                f"match the manifest file. You may need to delete the probabilities cached file."
            )
    else:

        @contextlib.contextmanager
        def default_autocast():
            yield

        if cfg.use_amp:
            if torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
                logging.info("AMP is enabled!\n")
                autocast = torch.cuda.amp.autocast

            else:
                autocast = default_autocast
        else:

            autocast = default_autocast

        # manual calculation of encoder_embeddings
        with autocast():
            with torch.no_grad():
                asr_model.eval()
                asr_model.encoder.freeze()
                device = next(asr_model.parameters()).device
                all_probs = []
                ctc_logprobs = []
                with tempfile.TemporaryDirectory() as tmpdir:
                    with open(os.path.join(tmpdir, 'manifest.json'), 'w', encoding='utf-8') as fp:
                        for audio_file in audio_file_paths:
                            entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': ''}
                            fp.write(json.dumps(entry) + '\n')
                    config = {
                        'paths2audio_files': audio_file_paths,
                        'batch_size': cfg.acoustic_batch_size,
                        'temp_dir': tmpdir,
                        'num_workers': cfg.num_workers,
                        'channel_selector': None,
                        'augmentor': None,
                    }
                    temporary_datalayer = asr_model._setup_transcribe_dataloader(config)

                    for test_batch in tqdm(temporary_datalayer, desc="Getting encoder and CTC decoder outputs...", disable=False):
                        # if cfg.decoder_type == "ctc":
                        #     processed_signal, processed_signal_length = asr_model.preprocessor(
                        #         input_signal=test_batch[0].to(device), length=test_batch[1].to(device),
                        #     )
                        #     # logging.warning(f"[DEBUG]: processed_signal.shape is: {processed_signal.shape}")
                        #     encoder_output = asr_model.encoder(audio_signal=processed_signal, length=processed_signal_length)
                        #     encoded = encoder_output[0]
                        #     encoded_len = encoder_output[1]
                        
                        # if cfg.decoder_type == "rnnt":
                        encoded, encoded_len = asr_model.forward(
                            input_signal=test_batch[0].to(device), input_signal_length=test_batch[1].to(device)
                        )
                        
                        start_dec_time = time.time()

                        ctc_dec_outputs = asr_model.ctc_decoder(encoder_output=encoded).cpu()
                        # dump encoder embeddings per file
                        for idx in range(encoded.shape[0]):
                            encoded_no_pad = encoded[idx, :, : encoded_len[idx]]
                            ctc_dec_outputs_no_pad = ctc_dec_outputs[idx, : encoded_len[idx]]
                            all_probs.append(encoded_no_pad)
                            ctc_logprobs.append(ctc_dec_outputs_no_pad)

        if cfg.probs_cache_file:
            logging.info(f"Writing pickle files of probabilities at '{cfg.probs_cache_file}'...")
            with open(cfg.probs_cache_file, 'wb') as f_dump:
                pickle.dump(all_probs, f_dump)

################################_WB_PART_#########################

    if cfg.apply_context_biasing:
        # load context graph:
        context_transcripts = []
        for line in open(cfg.context_file).readlines():
            item = line.strip().lower().split("-")
            word = item[0]
            word_tokenization = [asr_model.tokenizer.text_to_ids(x) for x in item[1:]]
            context_transcripts.append([word, word_tokenization])
        context_words = [item[0] for item in context_transcripts] 



        context_graph = ContextGraphCTC(blank_id=asr_model.decoder.blank_idx)
        # logging.warning(context_transcripts)
        context_graph.build(context_transcripts)
        
################################_WB_PART_#########################

    # sort all_probs according to length:
    if cfg.sort_logits:
        all_probs_with_indeces = (sorted(enumerate(all_probs), key=lambda x: x[1].size()[1], reverse=True))
        all_probs_sorted = []
        target_transcripts_sorted = []
        audio_file_paths_sorted = []
        durations_sorted = []
        ctc_logprobs_sorted = []
        for pair in all_probs_with_indeces:
            all_probs_sorted.append(pair[1])
            target_transcripts_sorted.append(target_transcripts[pair[0]])
            audio_file_paths_sorted.append(audio_file_paths[pair[0]])
            durations_sorted.append(durations[pair[0]])
            ctc_logprobs_sorted.append(ctc_logprobs[pair[0]])
        all_probs = all_probs_sorted
        target_transcripts = target_transcripts_sorted
        audio_file_paths = audio_file_paths_sorted
        durations = durations_sorted
        ctc_logprobs = ctc_logprobs_sorted

    # setup search parameters grid 
    params = {
        'beam_threshold': cfg.beam_threshold,
        'context_score': cfg.context_score,
        'ctc_ali_token_weight': cfg.ctc_ali_token_weight,
    }
    hp_grid = ParameterGrid(params)
    hp_grid = list(hp_grid)

    logging.info(f"=========================Starting the {cfg.decoding_strategy} decoding========================")
    logging.info(f"Grid search size: {len(hp_grid)}")
    logging.info(f"It may take some time...")
    logging.info(f"==============================================================================================")

    asr_model = asr_model.to('cpu')
    best_wer = 1e6

    for hp in hp_grid:
        results_file = f"preds_out_{cfg.decoding_strategy}_bthr-{hp['beam_threshold']}_cs-{hp['context_score']}ctcw-{hp['ctc_ali_token_weight']}"

        # preds_output_file = os.path.join(cfg.preds_output_folder, f"recognition_results.tsv")
        # preds_output_manifest = os.path.join(cfg.preds_output_folder, f"recognition_results.json")
        preds_output_manifest = os.path.join(cfg.preds_output_folder, results_file)
        candidate_wer, candidate_cer = decoding_step(
            asr_model,
            cfg,
            all_probs=all_probs,
            target_transcripts=target_transcripts,
            audio_file_paths=audio_file_paths,
            durations=durations,
            beam_batch_size=cfg.beam_batch_size,
            progress_bar=True,
            # preds_output_file=preds_output_file,
            preds_output_manifest=preds_output_manifest,
            context_graph=context_graph,
            ctc_logprobs=ctc_logprobs,
            hp = hp,
        )

        # compute fscore
        fscore_stats = compute_fscore(preds_output_manifest, context_words, return_scores=True)
        
        if candidate_wer < best_wer:
            best_beam_threshold = hp["beam_threshold"]
            best_context_score = hp["context_score"]
            best_ctc_ali_token_weight = hp["ctc_ali_token_weight"]
            best_wer = candidate_wer
            best_fscore_stats = fscore_stats

        print("***"*15)
        print(f"[INFO]: Greedy batch WER/CER = {candidate_wer:.2%}/{candidate_cer:.2%}")
        # print(f"[INFO]: Precision, Recall, Fscore = {fscore_stats[0]:.4f}, {fscore_stats[1]:.4f}, {fscore_stats[2]:.4f}")
        print(f"[INFO]: Params: beam_threshold = {hp['beam_threshold']}, \
              context_score = {hp['context_score']}, \
              ctc_ali_token_weight = {hp['ctc_ali_token_weight']}")
        print(f"[INFO]: Decoding only time (without encoder) is: {int(time.time() - start_dec_time)} sec")
    
    logging.info(
        f'Best WER = {best_wer:.2%}, '
        f'Precision/Recall/Fscore = {best_fscore_stats[0]:.4f}/{best_fscore_stats[1]:.4f}/{best_fscore_stats[2]:.4f}, '
        f'beam_threshold = {best_beam_threshold}, context_score = {best_context_score}, ctc_ali_token_weight = {best_ctc_ali_token_weight}'
    )

    # compute f-score for the best candidate
    context_words = [item[0] for item in context_transcripts]


if __name__ == '__main__':
    main()