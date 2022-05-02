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

from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from nemo.collections.asr.models import ClusteringDiarizer
from nemo.core.config import hydra_runner
from nemo.utils import logging

"""
This is a helper script to extract speaker embeddings based on manifest file
Usage:
python extract_speaker_embeddings.py --manifest=/path/to/manifest/file' 
--model_path='/path/to/.nemo/file'(optional)
--embedding_dir='/path/to/embedding/directory'

Args:
--manifest: path to manifest file containing audio_file paths for which embeddings need to be extracted
--model_path(optional): path to .nemo speaker verification model file to extract embeddings, if not passed SpeakerNet-M model would 
    be downloaded from NGC and used to extract embeddings
--embeddings_dir(optional): path to directory where embeddings need to stored default:'./'


"""

import json
import os
import pickle as pkl
from argparse import ArgumentParser

import numpy as np
import torch
from omegaconf import OmegaConf
from omegaconf import DictConfig
from tqdm import tqdm
from copy import deepcopy

from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
from nemo.collections.asr.parts.utils.speaker_utils import embedding_normalize
from nemo.collections.asr.models import ClusteringDiarizer
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.collections.asr.models import EncDecDiarLabelModel
from nemo.collections.asr.models.msdd_models import ClusterEmbedding
from nemo.utils import logging

from nemo.collections.asr.parts.utils.speaker_utils import (
    labels_to_pyannote_object,
    labels_to_rttmfile,
    rttm_to_labels,
    get_contiguous_stamps,
    merge_stamps,
    audio_rttm_map,
    get_embs_and_timestamps,
    get_uniqname_from_filepath,
    parse_scale_configs,
    perform_clustering,
    score_labels,
    segments_manifest_to_subsegments_manifest,
    write_rttm2manifest,
)

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield

def get_overlap_stamps(cont_a, ovl_spk_idx):
    ovl_spk_cont_list = [ [] for _ in range(len(ovl_spk_idx)) ]
    for spk_idx in range(len(ovl_spk_idx)):
        for idx, cont_a_line in enumerate(cont_a):
            start, end, speaker = cont_a_line.split()
            if idx in ovl_spk_idx[spk_idx]:
                ovl_spk_cont_list[spk_idx].append(f"{start} {end} speaker_{spk_idx}")
    total_ovl_cont_list = []
    for ovl_cont_list in ovl_spk_cont_list:
        if len(ovl_cont_list) > 0:
            total_ovl_cont_list.extend(merge_stamps(ovl_cont_list))
    return total_ovl_cont_list




def make_overlap_rttm_lines(clus_test_label_list, preds, max_num_of_spks, threshold, use_clus_as_main, max_overlap=2):
    '''
    old, works
    '''
    preds.squeeze(0)
    main_speaker_list = []
    max_num_of_spks = 10
    ovl_spk_idx_list = [ [] for _ in range(max_num_of_spks) ]
    for seg_idx, label in enumerate(clus_test_label_list):
        spk_for_seg = (preds[0, seg_idx] > threshold).int().cpu().numpy().tolist()
        sm_for_seg = preds[0, seg_idx].cpu().numpy()

        if use_clus_as_main or sum(spk_for_seg) == 0:
            main_spk_idx = int(label[2])
        else:
            main_spk_idx = np.argsort(preds[0, seg_idx].cpu().numpy())[::-1][0]
        
        if sum(spk_for_seg) > 1:
            max_idx = np.argmax(sm_for_seg)
            spk_for_seg[max_idx] = 1
            idx_arr = np.argsort(sm_for_seg)[::-1]
            for ovl_spk_idx in idx_arr[:max_overlap].tolist():
                # If the overlap speaker is not the main speaker, 
                # add the segment index to the speaker's overlap list.
                if ovl_spk_idx != int(main_spk_idx):
                    ovl_spk_idx_list[ovl_spk_idx].append(seg_idx)
        main_speaker_list.append(f"{label[0]} {label[1]} speaker_{main_spk_idx}")
    return main_speaker_list, ovl_spk_idx_list


def get_uniq_id_list_from_manifest(manifest_file):
    uniq_id_list = []
    with open(manifest_file, 'r', encoding='utf-8') as manifest:
        for i, line in enumerate(manifest.readlines()):
            line = line.strip()
            dic = json.loads(line)
            uniq_id = dic['audio_filepath'].split('/')[-1].split('.wav')[-1]
            uniq_id = get_uniqname_from_filepath(dic['audio_filepath'])
            uniq_id_list.append(uniq_id)
    return uniq_id_list

def get_integrated_preds_list(uniq_id_list, test_data_collection, preds_list):
    session_dict = get_id_tup_dict(uniq_id_list, test_data_collection, preds_list)
    output_dict = { uniq_id: [] for uniq_id in uniq_id_list}
    for uniq_id, data_list in session_dict.items():
        all_tups = tuple()
        for data in data_list:
            all_tups += data[0]
        n_est_spks = len(set(all_tups))
        digit_map = dict(zip(sorted(set(all_tups)), range(n_est_spks)))
        total_len = max([sess[1].shape[1] for sess in data_list])
        sum_pred = torch.zeros(total_len, n_est_spks)
        for data in data_list:
            _dim_tup, pred_mat = data
            dim_tup = [ digit_map[x] for x in _dim_tup ]

            if len(pred_mat.shape) == 3:
                pred_mat = pred_mat.squeeze(0)

            if n_est_spks in [1, 2]:
                sum_pred = pred_mat
            else:
                _end = pred_mat.shape[0]
                sum_pred[:_end, dim_tup] += pred_mat.cpu().float()
        sum_pred = sum_pred/(n_est_spks-1)
        output_dict[uniq_id] = sum_pred.unsqueeze(0)
    output_list = [ output_dict[uniq_id] for uniq_id in uniq_id_list ]
    return output_list

def get_id_tup_dict(uniq_id_list, test_data_collection, preds_list):
    session_dict = { x:[] for x in uniq_id_list}
    for idx, line in enumerate(test_data_collection):
        uniq_id = get_uniqname_from_filepath(line.audio_file)
        session_dict[uniq_id].append([line.tup_spks[0], preds_list[idx]])
    return session_dict

def save_tensor_as_pickle(tensor, file_name_str):
    base_path = "/home/taejinp/Downloads/tensor_view"
    fn = os.path.join(base_path, f"{file_name_str}.pickle")
    with open(fn, 'wb') as handle:
        pkl.dump(tensor, handle, protocol=pkl.HIGHEST_PROTOCOL)

def save_tensors(preds, scale_weights, targets):
    save_tensor_as_pickle(preds, 'preds')
    save_tensor_as_pickle(scale_weights, 'scale_weights')
    save_tensor_as_pickle(targets, 'targets')

def compute_accuracies(diar_decoder_model):
    f1_score = diar_decoder_model._accuracy_test.compute()
    num_correct =  torch.sum(diar_decoder_model._accuracy_test.true.bool()) 
    total_count = torch.prod(torch.tensor(diar_decoder_model._accuracy_test.targets.shape))
    simple_acc = num_correct / total_count
    return  f1_score, simple_acc


def diar_eval(msdd_model, preds_list, threshold, infer_overlap, collar, ignore_overlap, use_clus_as_main=False):
    manifest_file, clus_test_label_dict = msdd_model.cfg.test_ds.manifest_filepath, msdd_model.clus_test_label_dict
    AUDIO_RTTM_MAP = audio_rttm_map(manifest_file)
    manifest_file_lengths_list = []
    all_hypothesis, all_reference = [], []
    no_references = False
    clus_labels_dict = {}
    with open(manifest_file, 'r', encoding='utf-8') as manifest:
        for i, line in enumerate(manifest.readlines()):
            line = line.strip()
            dic = json.loads(line)
            if len(dic['audio_filepath'].split('/')[-1].split('.')) > 2: 
                uniq_id = '.'.join(dic['audio_filepath'].split('/')[-1].split('.')[:-1])
            else:
                uniq_id = dic['audio_filepath'].split('/')[-1].split('.')[0]
            value = AUDIO_RTTM_MAP[uniq_id]

            clus_test_label_list = clus_test_label_dict[uniq_id]
            only_label = [ x[-1] for x in clus_test_label_list ]
            num_inferred_spks = len(set(only_label))
            manifest_file_lengths_list.append(len(clus_test_label_list))
            lines, ovl_idx_list = make_overlap_rttm_lines(clus_test_label_list, 
                                                          preds_list[i], 
                                                          num_inferred_spks, 
                                                          threshold, 
                                                          use_clus_as_main)
            cont_a = get_contiguous_stamps(lines)
            maj_labels = merge_stamps(cont_a)
            ovl_labels = get_overlap_stamps(cont_a, ovl_idx_list)
            if infer_overlap:
                total_labels = maj_labels + ovl_labels
            else:
                total_labels = maj_labels
            hypothesis = labels_to_pyannote_object(total_labels, uniq_name=uniq_id)
            all_hypothesis.append([uniq_id, hypothesis])

            rttm_file = value.get('rttm_filepath', None)
            if rttm_file is not None and os.path.exists(rttm_file) and not no_references:
                ref_labels = rttm_to_labels(rttm_file)
                reference = labels_to_pyannote_object(ref_labels, uniq_name=uniq_id)
                all_reference.append([uniq_id, reference])
    score = score_labels(
        AUDIO_RTTM_MAP,
        all_reference,
        all_hypothesis,
        collar=collar,
        ignore_overlap=ignore_overlap,
        )
    return score

class NeuralDiarizer:
    def __init__(self, cfg: DictConfig):
        self._cfg = cfg
    
        # Initialize diarization decoder 
        self.msdd_model = self._init_msdd_model(cfg)
        self.msdd_model.run_clus_from_loaded_emb = True
        
        # Initialize clustering and embedding preparation instance (as a diarization encoder).
        self.clustering_embedding = ClusterEmbedding(cfg_base=cfg, cfg_msdd_model=self.msdd_model.cfg)
        self.clustering_embedding.run_clus_from_loaded_emb = True

    def transfer_diar_params_to_model_params(self, msdd_model, cfg):
        msdd_model.cfg.test_ds.manifest_filepath = cfg.diarizer.manifest_filepath
        msdd_model.cfg.test_ds.emb_dir = cfg.diarizer.out_dir
        msdd_model.cfg.test_ds.num_spks = cfg.msdd_model.max_num_of_spks
        msdd_model.cfg.base.diarizer.out_dir = cfg.msdd_model.test_ds.emb_dir
        msdd_model.cfg_base = cfg
        return msdd_model.cfg 

    def _init_msdd_model(self, cfg):
        self.device = 'cuda'
        if not torch.cuda.is_available():
            self.device = 'cpu'
            logging.warning("Running model on CPU, for faster performance it is adviced to use atleast one NVIDIA GPUs")

        if cfg.diarizer.msdd_model.model_path.endswith('.nemo'):
            logging.info(f"Using local speaker model from {cfg.diarizer.msdd_model.model_path}")
            msdd_model = EncDecDiarLabelModel.restore_from(restore_path=cfg.diarizer.msdd_model.model_path)
        elif cfg.diarizer.msdd_model.model_path.endswith('.ckpt'):
            msdd_model = EncDecDiarLabelModel.load_from_checkpoint(checkpoint_path=cfg.diarizer.msdd_model.model_path)
        import ipdb; ipdb.set_trace()
        msdd_model.cfg = self.transfer_diar_params_to_model_params(msdd_model, cfg)
        return msdd_model

    def decode_diarization_bi_ch(self, diar_decoder_model,  embedding_dir='./', device='cuda'):
        test_cfg = diar_decoder_model.cfg.test_ds
        diar_decoder_model.setup_test_data(test_cfg)
        diar_decoder_model = diar_decoder_model.to(self.device)
        diar_decoder_model.eval()
        
        preds_list, scale_weights_list, signal_lengths_list= [], [], []

        length_list = []
        uniq_id_list = get_uniq_id_list_from_manifest(test_cfg.manifest_filepath)
        test_data_collection = [ d for d in diar_decoder_model.data_collection ]
        torch.set_grad_enabled(False)
        for test_batch in tqdm(diar_decoder_model.test_dataloader()):
            test_batch = [x.to(self.device) for x in test_batch]
            signals, signal_lengths, targets, emb_vectors = test_batch 
            with autocast():
                preds, scale_weights= diar_decoder_model.forward(input_signal=signals,
                                                                 input_signal_length=signal_lengths,
                                                                 emb_vectors=emb_vectors,
                                                                 targets=targets)
            diar_decoder_model._accuracy_test(preds, targets, signal_lengths) 
            del test_batch
            preds_list.extend( list(torch.split(preds, 1)))
            scale_weights_list.extend( list(torch.split(scale_weights, 1)))
            signal_lengths_list.extend( list(torch.split(signal_lengths, 1)))
            f1_score, simple_acc = compute_accuracies(diar_decoder_model)
            logging.info(f"Batch Test Inference F1 score. {f1_score:.4f}, simple Acc. {simple_acc:.4f}")
            save_tensors(preds, scale_weights, targets) 
        integrated_preds_list = get_integrated_preds_list(uniq_id_list, test_data_collection, preds_list)
        return integrated_preds_list, scale_weights_list, signal_lengths_list

    def diarize(self):
        torch.set_grad_enabled(False)

        self.clustering_embedding.prepare_cluster_embs_infer()
        self.msdd_model.bi_ch_infer = True
        self.msdd_model.get_emb_clus_infer(self.clustering_embedding) 
        
        # The first pass of dirization decoding for scale weights
        preds_list, scale_weights_list, signal_lengths_list = self.decode_diarization_bi_ch(self.msdd_model)

        logging.info("Clustering Diarization Re-clustered Result: threshold: 1.0, use_clus_as_main=True")
        score = diar_eval(self.msdd_model, preds_list, threshold=1.0, infer_overlap=False, collar=0.25, ignore_overlap=True, use_clus_as_main=True)
        score = diar_eval(self.msdd_model, preds_list, threshold=1.0, infer_overlap=False, collar=0.0, ignore_overlap=False, use_clus_as_main=True)

        der_list, thres_range = [], list(self._cfg.diarizer.msdd_model.parameters.sigmoid_threshold)
        for threshold in thres_range:
            logging.info(f"     [Threshold: {threshold:.4f}]     [use_clus_as_main=False]")
            score = diar_eval(self.msdd_model, preds_list, threshold, infer_overlap=True, collar=0.25, ignore_overlap=True, use_clus_as_main=False)
            score = diar_eval(self.msdd_model, preds_list, threshold, infer_overlap=True, collar=0.0, ignore_overlap=False, use_clus_as_main=False)
            der_list.append(abs(score[0]))
        max_idx = np.argmin(np.array(der_list))

        logging.info(f"     [Clustering Multi-scale Static Weights]: {self._cfg.diarizer.speaker_embeddings.parameters.multiscale_weights}")
        logging.info(f"     [Best_thres]: {thres_range[max_idx]:.4f} The lowest DER: {der_list[max_idx]:.4f}")

@hydra_runner(config_path="conf", config_name="diarization_decoder.telephonic.yaml")
def main(cfg):
    
    neural_diarizer = NeuralDiarizer(cfg=cfg)
    neural_diarizer.diarize()

if __name__ == '__main__':
    main()

