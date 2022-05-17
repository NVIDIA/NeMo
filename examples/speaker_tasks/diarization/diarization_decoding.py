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
from nemo.collections.asr.parts.utils._nmesc_clustering import getCosAffinityMatrix
from sklearn.metrics.pairwise import cosine_similarity

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
from nemo.collections.asr.models import EncDecEnd2EndDiarModel
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

def get_overlap_stamps(cont_stamps, ovl_spk_idx):
    ovl_spk_cont_list = [ [] for _ in range(len(ovl_spk_idx)) ]
    for spk_idx in range(len(ovl_spk_idx)):
        for idx, cont_a_line in enumerate(cont_stamps):
            start, end, speaker = cont_a_line.split()
            if idx in ovl_spk_idx[spk_idx]:
                ovl_spk_cont_list[spk_idx].append(f"{start} {end} speaker_{spk_idx}")
    total_ovl_cont_list = []
    for ovl_cont_list in ovl_spk_cont_list:
        if len(ovl_cont_list) > 0:
            total_ovl_cont_list.extend(merge_stamps(ovl_cont_list))
    return total_ovl_cont_list


def generate_diar_timestamps(clus_test_label_list, preds, max_overlap=2, **params):
    '''
    old, works
    '''
    preds.squeeze(0)
    main_speaker_lines = []
    overlap_speaker_list = [ [] for _ in range(params['max_num_of_spks']) ]
    for seg_idx, label in enumerate(clus_test_label_list):
        preds.squeeze(0)
        spk_for_seg = (preds[0, seg_idx] > params['threshold']).int().cpu().numpy().tolist()
        sm_for_seg = preds[0, seg_idx].cpu().numpy()
        
        if params['use_dec']:
            main_spk_idx = np.argsort(preds[0, seg_idx].cpu().numpy())[::-1][0]
        elif params['use_clus'] or sum(spk_for_seg) == 0:
            main_spk_idx = int(label[2])
        else:
            main_spk_idx = np.argsort(preds[0, seg_idx].cpu().numpy())[::-1][0]

        if sum(spk_for_seg) > 1:
            max_idx = np.argmax(sm_for_seg)
            idx_arr = np.argsort(sm_for_seg)[::-1]
            for ovl_spk_idx in idx_arr[:max_overlap].tolist():
                if ovl_spk_idx != int(main_spk_idx):
                    try:
                        overlap_speaker_list[ovl_spk_idx].append(seg_idx)
                    except:
                        import ipdb; ipdb.set_trace()
        main_speaker_lines.append(f"{label[0]} {label[1]} speaker_{main_spk_idx}")
    cont_stamps = get_contiguous_stamps(main_speaker_lines)
    maj_labels = merge_stamps(cont_stamps)
    ovl_labels = get_overlap_stamps(cont_stamps, overlap_speaker_list)
    return maj_labels, ovl_labels


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


def get_id_tup_dict(uniq_id_list, test_data_collection, preds_list):
    session_dict = { x:[] for x in uniq_id_list}
    for idx, line in enumerate(test_data_collection):
        uniq_id = get_uniqname_from_filepath(line.audio_file)
        session_dict[uniq_id].append([line.tup_spks[0], preds_list[idx], line.tup_spks[2], line.tup_spks[3]])
    return session_dict

def compute_accuracies(diar_decoder_model):
    f1_score = diar_decoder_model._accuracy_test.compute()
    num_correct =  torch.sum(diar_decoder_model._accuracy_test.true.bool()) 
    total_count = torch.prod(torch.tensor(diar_decoder_model._accuracy_test.targets.shape))
    simple_acc = num_correct / total_count
    return  f1_score, simple_acc

def get_uniq_id_from_manifest_line(line):
    line = line.strip()
    dic = json.loads(line)
    if len(dic['audio_filepath'].split('/')[-1].split('.')) > 2: 
        uniq_id = '.'.join(dic['audio_filepath'].split('/')[-1].split('.')[:-1])
    else:
        uniq_id = dic['audio_filepath'].split('/')[-1].split('.')[0]
    return uniq_id

def generate_pyannote_obj(uniq_id, total_labels, manifest_dic):
    return all_hypothesis, all_reference

# def diar_eval(msdd_model, preds_list, threshold, infer_overlap, collar, ignore_overlap, use_clus=False, use_dec=False):
def diar_eval(msdd_model, hyp, **params):
    manifest_file, clus_test_label_dict = msdd_model.cfg.test_ds.manifest_filepath, msdd_model.clus_test_label_dict
    params['max_num_of_spks'] = msdd_model._cfg.base.diarizer.clustering.parameters.max_num_speakers

    AUDIO_RTTM_MAP = audio_rttm_map(manifest_file)
    manifest_file_lengths_list = []
    all_hypothesis, all_reference = [], []
    no_references = False
    clus_labels_dict = {}
    with open(manifest_file, 'r', encoding='utf-8') as manifest:
        for i, line in enumerate(manifest.readlines()):
            uniq_id = get_uniq_id_from_manifest_line(line)
            manifest_dic = AUDIO_RTTM_MAP[uniq_id]
            clus_test_label_list = clus_test_label_dict[uniq_id]
            only_label = [ x[-1] for x in clus_test_label_list ]
            manifest_file_lengths_list.append(len(clus_test_label_list))
            maj_labels, ovl_labels = generate_diar_timestamps(clus_test_label_list, hyp[i], **params)
            if params['infer_overlap']:
                total_labels = maj_labels + ovl_labels
            else:
                total_labels = maj_labels
            hypothesis = labels_to_pyannote_object(total_labels, uniq_name=uniq_id)
            all_hypothesis.append([uniq_id, hypothesis])
            rttm_file = manifest_dic.get('rttm_filepath', None)
            if rttm_file is not None and os.path.exists(rttm_file) and not no_references:
                ref_labels = rttm_to_labels(rttm_file)
                reference = labels_to_pyannote_object(ref_labels, uniq_name=uniq_id)
                all_reference.append([uniq_id, reference])
            else:
                no_references = True
                all_reference = []
    score = score_labels(
        AUDIO_RTTM_MAP,
        all_reference,
        all_hypothesis,
        collar=params['collar'],
        ignore_overlap=params['ignore_overlap'],
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

        self.use_prime_cluster_avg_emb = True
        self.max_pred_length = 0
        self.eps = 10e-5

    def transfer_diar_params_to_model_params(self, msdd_model, cfg):
        msdd_model.cfg.test_ds.manifest_filepath = cfg.diarizer.manifest_filepath
        msdd_model.cfg.test_ds.emb_dir = cfg.diarizer.out_dir
        msdd_model.cfg.test_ds.num_spks = cfg.msdd_model.max_num_of_spks
        msdd_model.cfg.base.diarizer.out_dir = cfg.msdd_model.test_ds.emb_dir
        msdd_model.cfg_base = cfg
        msdd_model._cfg.base.diarizer.clustering.parameters.max_num_speakers = cfg.diarizer.clustering.parameters.max_num_speakers
        return msdd_model.cfg 

    def _init_msdd_model(self, cfg):
        self.device = 'cuda'
        if not torch.cuda.is_available():
            self.device = 'cpu'
            logging.warning("Running model on CPU, for faster performance it is adviced to use atleast one NVIDIA GPUs")

        if cfg.diarizer.msdd_model.model_path.endswith('.nemo'):
            logging.info(f"Using local speaker model from {cfg.diarizer.msdd_model.model_path}")
            msdd_model = EncDecEnd2EndDiarModel.restore_from(restore_path=cfg.diarizer.msdd_model.model_path)
        elif cfg.diarizer.msdd_model.model_path.endswith('.ckpt'):
            msdd_model = EncDecEnd2EndDiarModel.load_from_checkpoint(checkpoint_path=cfg.diarizer.msdd_model.model_path)
        msdd_model.cfg = self.transfer_diar_params_to_model_params(msdd_model, cfg)
        return msdd_model
    
    def get_pred_mat(self, data_list):
        all_tups = tuple()
        for data in data_list:
            all_tups += data[0]
        n_est_spks = len(set(all_tups))
        digit_map = dict(zip(sorted(set(all_tups)), range(n_est_spks)))
        total_len = max([sess[1].shape[1] for sess in data_list])
        sum_pred = torch.zeros(total_len, n_est_spks)
        for data in data_list:
            _dim_tup, pred_mat = data[:2]
            dim_tup = [ digit_map[x] for x in _dim_tup ]
            if len(pred_mat.shape) == 3:
                pred_mat = pred_mat.squeeze(0)
            if n_est_spks in [1, 2]:
                sum_pred = pred_mat
            else:
                _end = pred_mat.shape[0]
                sum_pred[:_end, dim_tup] += pred_mat.cpu().float()
        print(f" est_spks {data[2]} gt_spks {data[3]} ")
        sum_pred = sum_pred/(n_est_spks-1)
        return sum_pred

    def get_integrated_preds_list(self, uniq_id_list, test_data_collection, preds_list):
        """
        Merge multiple sequence inference outputs into a session level result.

        """
        session_dict = get_id_tup_dict(uniq_id_list, test_data_collection, preds_list)
        output_dict = { uniq_id: [] for uniq_id in uniq_id_list}
        for uniq_id, data_list in session_dict.items():
            sum_pred = self.get_pred_mat(data_list)
            output_dict[uniq_id] = sum_pred.unsqueeze(0) 
        output_list = [ output_dict[uniq_id] for uniq_id in uniq_id_list ]
        return output_list
    
    def get_uniq_id_from_rttm(self, sample):
        return sample.rttm_file.split('/')[-1].split('.rttm')[0]

    def get_max_pred_length(self, uniq_id_list):
        test_data_uniq_ids = [ self.get_uniq_id_from_rttm(d) for d in self.msdd_model.data_collection ]
        assert set(test_data_uniq_ids) == set(uniq_id_list), "Test data does not match with input manifest. Test data is not loaded properly."
        pre_signal_lengths_list = []
        logging.info("Scanning test batch sequence lengths .")
        for sidx, test_batch in enumerate(self.msdd_model.test_dataloader()):
            test_batch = [x.to(self.device) for x in test_batch]
            signals, signal_lengths, targets, emb_vectors = test_batch 
            pre_signal_lengths_list.append(signal_lengths)
        sequence_lengths = torch.hstack(pre_signal_lengths_list)
        return max(sequence_lengths)

    def decode_diarization_bi_ch(self, embedding_dir='./', device='cuda'):
        test_cfg = self.msdd_model.cfg.test_ds
        self.msdd_model.setup_test_data(test_cfg)
        self.msdd_model = self.msdd_model.to(self.device)
        self.msdd_model.eval()
        torch.set_grad_enabled(False)
        preds_list, targets_list, signal_lengths_list = [], [], []
        uniq_id_list = get_uniq_id_list_from_manifest(test_cfg.manifest_filepath)
        test_data_collection = [ d for d in self.msdd_model.data_collection ]
        self.max_pred_length = self.get_max_pred_length(uniq_id_list)

        for sidx, test_batch in enumerate(tqdm(self.msdd_model.test_dataloader())):
            test_batch = [x.to(self.device) for x in test_batch]
            signals, signal_lengths, _targets, emb_vectors = test_batch 
            if self._cfg.msdd_model.use_longest_scale_clus_avg_emb:
                emb_vectors = self.msdd_model.get_longest_scale_clus_avg_emb(emb_vectors)
            with autocast():
                _preds, _  = self.msdd_model.forward_infer(input_signal=signals,
                                                                       input_signal_length=signal_lengths,
                                                                       emb_vectors=emb_vectors,
                                                                       targets=_targets)
            # preds and targets size should have the session-maximum lengtth since sequence lengths change at each batch.
            self.max_pred_length = max(_preds.shape[1], self.max_pred_length)
            preds = torch.zeros(_preds.shape[0], self.max_pred_length, _preds.shape[2])
            targets = torch.zeros(_preds.shape[0], self.max_pred_length, _preds.shape[2])
            preds[:, :_preds.shape[1], :] = _preds
            targets[: , :_preds.shape[1], :] = _targets
            if self._cfg.diarizer.msdd_model.parameters.seq_eval_mode:
                self.msdd_model._accuracy_test(preds, targets, signal_lengths) 

            preds_list.extend( list(torch.split(preds, 1)))
            targets_list.extend( list(torch.split(targets, 1)))
            signal_lengths_list.extend( list(torch.split(signal_lengths, 1)))
        
        if self._cfg.diarizer.msdd_model.parameters.seq_eval_mode:
            f1_score, simple_acc = compute_accuracies(self.msdd_model)
            logging.info(f"Test Inference F1 score. {f1_score:.4f}, simple Acc. {simple_acc:.4f}")
        integrated_preds_list = self.get_integrated_preds_list(uniq_id_list, test_data_collection, preds_list)
        return integrated_preds_list, targets_list, signal_lengths_list

    def diarize(self):
        torch.set_grad_enabled(False)

        self.clustering_embedding.prepare_cluster_embs_infer()
        self.msdd_model.bi_ch_infer = True
        self.msdd_model.get_emb_clus_infer(self.clustering_embedding) 
        
        # The first pass of dirization decoding for scale weights
        preds_list, targets_list, signal_lengths_list = self.decode_diarization_bi_ch()
        if self._cfg.diarizer.msdd_model.parameters.seq_eval_mode:
            use_clus, infer_overlap, use_dec = False, True, True
            logging.info("\n    [ORACLE Diarization Result]: Infer Overlap, use_clus=False")
            score = diar_eval(msdd_model=self.msdd_model, hyp=targets_list, threshold=0.0, infer_overlap=True, collar=0.25, ignore_overlap=True, use_clus=False, use_dec=True)
            score = diar_eval(msdd_model=self.msdd_model, hyp=targets_list, threshold=0.0, infer_overlap=True, collar=0.0, ignore_overlap=False, use_clus=False, use_dec=True)
            logging.info("\n    [ORACLE Diarization Result]: NO Overlap Inference, use_clus=False")
            score = diar_eval(msdd_model=self.msdd_model, hyp=targets_list, threshold=0.0, infer_overlap=False, collar=0.25, ignore_overlap=True, use_clus=False, use_dec=True)
            score = diar_eval(msdd_model=self.msdd_model, hyp=targets_list, threshold=0.0, infer_overlap=False, collar=0.0, ignore_overlap=False, use_clus=False, use_dec=True)
            print("\n")

            logging.info("Clustering Diarization Re-clustered Result: threshold: 1.0, use_clus=True")
            # score = diar_eval(self.msdd_model, preds_list, threshold=1.0, infer_overlap=False, collar=0.25, ignore_overlap=False, use_clus=True, use_dec=use_dec)
            # score = diar_eval(self.msdd_model, preds_list, threshold=1.0, infer_overlap=False, collar=0.0, ignore_overlap=False, use_clus=True, use_dec=use_dec)
            # score = diar_eval(self.msdd_model, preds_list, threshold=1.0, infer_overlap=False, collar=0.25, ignore_overlap=False, use_clus=True, use_dec=use_dec)
            # score = diar_eval(self.msdd_model, preds_list, threshold=1.0, infer_overlap=False, collar=0.0, ignore_overlap=False, use_clus=True, use_dec=use_dec)
            print("\n")

        print("\n\n")
        der_A_list_ucr, der_B_list_ucr, der_A_list_bk, der_B_list_bk, thres_range = [], [], [], [], list(self._cfg.diarizer.msdd_model.parameters.sigmoid_threshold)
        der_A_list_dec, der_B_list_dec = [], []
        for threshold in thres_range:
            use_clus, infer_overlap, use_dec = True, True, False
            logging.info(f"     [Threshold: {threshold:.4f}]  [infer_overlap={infer_overlap}]   [use_clus={use_clus}]")
            score_A_ucr = diar_eval(self.msdd_model, preds_list, threshold=threshold, infer_overlap=infer_overlap, collar=0.25, ignore_overlap=True, use_clus=use_clus, use_dec=use_dec)
            score_B_ucr = diar_eval(self.msdd_model, preds_list, threshold=threshold, infer_overlap=infer_overlap, collar=0.0, ignore_overlap=False, use_clus=use_clus, use_dec=use_dec)
            
            use_clus, infer_overlap, use_dec = False, True, False
            logging.info(f"     [Threshold: {threshold:.4f}]  [infer_overlap={infer_overlap}]   [use_clus={use_clus}]")
            score_A_bk = diar_eval(self.msdd_model, preds_list, threshold=threshold,  infer_overlap=infer_overlap, collar=0.25, ignore_overlap=True, use_clus=use_clus, use_dec=use_dec)
            score_B_bk = diar_eval(self.msdd_model, preds_list, threshold=threshold,  infer_overlap=infer_overlap, collar=0.0, ignore_overlap=False, use_clus=use_clus, use_dec=use_dec)
            
            use_clus, infer_overlap, use_dec = False, True, True
            logging.info(f"     [Threshold: {threshold:.4f}]  [infer_overlap={infer_overlap}]   [use_dec={use_dec}]")
            score_A_dec = diar_eval(self.msdd_model, preds_list, threshold=threshold,  infer_overlap=infer_overlap, collar=0.25, ignore_overlap=True, use_clus=use_clus, use_dec=use_dec)
            score_B_dec = diar_eval(self.msdd_model, preds_list, threshold=threshold,  infer_overlap=infer_overlap, collar=0.0, ignore_overlap=False, use_clus=use_clus, use_dec=use_dec)
            print("\n")
            
            der_A_list_ucr.append(abs(score_A_ucr[0]))
            der_B_list_ucr.append(abs(score_B_ucr[0]))
            der_A_list_bk.append(abs(score_A_bk[0]))
            der_B_list_bk.append(abs(score_B_bk[0]))
            der_A_list_dec.append(abs(score_A_dec[0]))
            der_B_list_dec.append(abs(score_B_dec[0]))

        max_idx_ucr= np.argmin(np.array(der_B_list_ucr))
        max_idx_bk = np.argmin(np.array(der_B_list_bk))
        max_idx_dec = np.argmin(np.array(der_B_list_dec))

        logging.info(f"     [Clustering Multi-scale Static Weights]: {self._cfg.diarizer.speaker_embeddings.parameters.multiscale_weights}")
        logging.info(f"     [Best_thres]: {thres_range[max_idx_ucr]:.4f} Ucr____ MSDD DER: {der_A_list_ucr[max_idx_ucr]:.4f} {der_B_list_ucr[max_idx_ucr]:.4f}")
        logging.info(f"     [Best_thres]: {thres_range[max_idx_bk]:.4f} backup_ MSDD DER: {der_A_list_bk[max_idx_bk]:.4f} {der_B_list_bk[max_idx_bk]:.4f}")
        logging.info(f"     [Best_thres]: {thres_range[max_idx_dec]:.4f} decoder MSDD DER: {der_A_list_dec[max_idx_dec]:.4f} {der_B_list_dec[max_idx_dec]:.4f}")
        
        print("\n")
        logging.info("Clustering Diarization Re-clustered Result: threshold: 1.0, use_clus=True")
        score = diar_eval(self.msdd_model, preds_list, threshold=1.0, infer_overlap=False, collar=0.25, ignore_overlap=True, use_clus=True, use_dec=False)
        score = diar_eval(self.msdd_model, preds_list, threshold=1.0, infer_overlap=False, collar=0.0, ignore_overlap=False, use_clus=True, use_dec=False)

@hydra_runner(config_path="conf", config_name="diarization_decoder.telephonic.yaml")
def main(cfg):
    
    neural_diarizer = NeuralDiarizer(cfg=cfg)
    neural_diarizer.diarize()

if __name__ == '__main__':
    main()

