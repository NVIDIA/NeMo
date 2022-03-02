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
from tqdm import tqdm
from copy import deepcopy

from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
from nemo.collections.asr.parts.utils.speaker_utils import embedding_normalize
from nemo.collections.asr.models import ClusteringDiarizer
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.collections.asr.models import EncDecDiarLabelModel
from nemo.collections.asr.models.tsvad_models import ClusterEmbedding
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

def merge_all(lines):
    cont_a = get_contiguous_stamps(lines)
    labels = merge_stamps(cont_a)
    return labels

def get_contiguous_overlap_stamps(stamps):
    """
    Return contiguous time stamps
    """
    lines = deepcopy(stamps)
    contiguous_stamps = []
    for i in range(len(lines) - 1):
        start, end, speaker = lines[i].split()
        next_start, next_end, next_speaker = lines[i + 1].split()
        if float(end) > float(next_start):
            avg = str((float(next_start) + float(end)) / 2.0)
            lines[i + 1] = ' '.join([avg, next_end, next_speaker])
            contiguous_stamps.append(start + " " + avg + " " + speaker)
        else:
            contiguous_stamps.append(start + " " + end + " " + speaker)
    start, end, speaker = lines[-1].split()
    contiguous_stamps.append(start + " " + end + " " + speaker)
    return contiguous_stamps

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

def make_diar_lines(clus_test_label_list, preds, max_num_of_spks, threshold, max_overlap=2):
    preds.squeeze(0)
    main_speaker_list = []
    ovl_spk_idx_list = [ [] for _ in range(max_num_of_spks) ]
    for idx, label in enumerate(clus_test_label_list):
        spk_for_seg = (preds[0, idx] > threshold).int().cpu().numpy().tolist()
        sm_for_seg = preds[0, idx].cpu().numpy()
        if sum(spk_for_seg) > 1:
            max_idx = np.argmax(sm_for_seg)
            spk_for_seg[max_idx] = 1
            idx_arr = np.argsort(sm_for_seg)[::-1]
            for ovl_spk_idx in idx_arr[:max_overlap].tolist():
                if ovl_spk_idx != int(label[2]):
                    ovl_spk_idx_list[ovl_spk_idx].append(idx)
        main_speaker_list.append(f"{label[0]} {label[1]} speaker_{label[2]}")
    return main_speaker_list, ovl_spk_idx_list


# def decode_diarization(diar_decoder_model, args.manifest, batch_size=1, embedding_dir=args.embedding_dir, device=device)
def decode_diarization(diar_decoder_model, max_num_of_spks, out_rttm_dir, manifest_file, batch_size=1, embedding_dir='./', device='cuda'):
    test_config = OmegaConf.create(
        dict(manifest_filepath=manifest_file, num_spks=max_num_of_spks, sample_rate=16000, labels=None, batch_size=batch_size, shuffle=False,)
    )

    diar_decoder_model.setup_test_data(test_config)
    diar_decoder_model = diar_decoder_model.to(device)
    diar_decoder_model.eval()

    all_embs = []
    out_embeddings = {}
    
    preds_list = []
    length_list = []
    for test_batch in tqdm(diar_decoder_model.test_dataloader()):
        test_batch = [x.to(device) for x in test_batch]
        signals, signal_lengths, targets, ivectors = test_batch 
        # print(f"Batch size : {signals.shape[0]}")
        with autocast():
            preds = diar_decoder_model.forward(input_signal=signals,
                                 input_signal_length=signal_lengths,
                                 ivectors=ivectors,
                                 targets=targets)
        diar_decoder_model._accuracy(preds, targets) 
        acc = diar_decoder_model._accuracy.compute()
        del test_batch
        logging.info(f"------------ signal_lengths: {signal_lengths} Batch Test Inference F1 Acc. {acc}")
        # preds_list.append(preds)
        # length_list.append(preds.shape[1])
        preds_list.extend( list(torch.split(preds, 1)))
    # import ipdb; ipdb.set_trace()
    return preds_list

def diar_eval(manifest_file, max_num_of_spks, preds_list, clus_test_label_dict, threshold, infer_overlap, collar, ignore_overlap):

    AUDIO_RTTM_MAP = audio_rttm_map(manifest_file)
    manifest_file_lengths_list = []
    all_hypothesis, all_reference = [], []
    all_hypothesis_clus, all_reference_clus= [], []
    no_references = False
    clus_labels_dict = {}
    with open(manifest_file, 'r', encoding='utf-8') as manifest:
        for i, line in enumerate(manifest.readlines()):
            # if i != 0:
                # continue
            line = line.strip()
            dic = json.loads(line)
            uniq_id = dic['audio_filepath'].split('/')[-1].split('.')[0]
            value = AUDIO_RTTM_MAP[uniq_id]

            clus_test_label_list = clus_test_label_dict[uniq_id]
            only_label = [ x[-1] for x in clus_test_label_list ]
            clus_labels_dict[uniq_id] = only_label
            
            manifest_file_lengths_list.append(len(clus_test_label_list))
            try:
                lines, ovl_idx_list = make_diar_lines(clus_test_label_list, preds_list[i], max_num_of_spks, threshold)
            except:
                import ipdb; ipdb.set_trace()
            cont_a = get_contiguous_stamps(lines)
            labels = merge_stamps(cont_a)
            ovl_labels = get_overlap_stamps(cont_a, ovl_idx_list)
            if infer_overlap:
                total_labels = labels + ovl_labels
            else:
                total_labels = labels
            # import ipdb; ipdb.set_trace()
            # labels_to_rttmfile(labels, uniq_id, out_rttm_dir)
            hypothesis = labels_to_pyannote_object(total_labels, uniq_name=uniq_id)
            all_hypothesis.append([uniq_id, hypothesis])

            rttm_file = value.get('rttm_filepath', None)
            if rttm_file is not None and os.path.exists(rttm_file) and not no_references:
                ref_labels = rttm_to_labels(rttm_file)
                reference = labels_to_pyannote_object(ref_labels, uniq_name=uniq_id)
                all_reference.append([uniq_id, reference])
            # import ipdb; ipdb.set_trace()
    score = score_labels(
        AUDIO_RTTM_MAP,
        all_reference,
        all_hypothesis,
        collar=collar,
        ignore_overlap=ignore_overlap,
        )
    return score

    # embedding_dir = os.path.join(embedding_dir, 'embeddings')
    # if not os.path.exists(embedding_dir):
        # os.makedirs(embedding_dir, exist_ok=True)

    # prefix = manifest_file.split('/')[-1].rsplit('.', 1)[-2]

    # name = os.path.join(embedding_dir, prefix)
    # embeddings_file = name + '_embeddings.pkl'
    # pkl.dump(out_embeddings, open(embeddings_file, 'wb'))
    # logging.info("Saved embedding files to {}".format(embedding_dir))


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--manifest", type=str, required=True, help="Path to manifest file",
    )
    parser.add_argument(
                # import ipdb; ipdb.set_trace()
        "--model_path",
        type=str,
        default='titanet_large',
        required=False,
        help="path to .nemo speaker verification model file to extract embeddings, if not passed SpeakerNet-M model would be downloaded from NGC and used to extract embeddings",
    )
    parser.add_argument(
        "--embedding_dir",
        type=str,
        default='./',
        required=False,
        help="path to directory where embeddings need to stored default:'./'",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default='./',
        required=True,
        help="path to yaml file:'./'",
    )
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    cfg = OmegaConf.load(args.config_file) 
    # import ipdb; ipdb.set_trace()
    max_num_of_spks = 2
    cfg.ts_vad_model.max_num_of_spks = max_num_of_spks
    cfg.diarizer.clustering.parameters.max_num_speakers = max_num_of_spks
    clustering_embedding = ClusterEmbedding(cfg_base=cfg, cfg_ts_vad_model=cfg.ts_vad_model)
    if args.model_path.endswith('.nemo'):
        logging.info(f"Using local speaker model from {args.model_path}")
        ts_vad_model = EncDecDiarLabelModel.restore_from(restore_path=args.model_path)
    elif args.model_path.endswith('.ckpt'):
        ts_vad_model = EncDecDiarLabelModel.load_from_checkpoint(checkpoint_path=args.model_path)

    device = 'cuda'
    if not torch.cuda.is_available():
        device = 'cpu'
        logging.warning("Running model on CPU, for faster performance it is adviced to use atleast one NVIDIA GPUs")
    clustering_embedding.cfg_base.diarizer.speaker_embeddings.model_path = "titanet_large"
    # clustering_embedding.cfg_base.diarizer.clustering.parameters.max_num_speakers = 2
    clustering_embedding.cfg_base.diarizer.oracle_vad=True
    clustering_embedding.cfg_base.diarizer.collar=0.0
    clustering_embedding.cfg_base.diarizer.ignore_overlap=False
    clustering_embedding.cfg_base.diarizer.speaker_embeddings.parameters.window_length_in_sec=[1.5, 1.0, 0.5]
    clustering_embedding.cfg_base.diarizer.speaker_embeddings.parameters.shift_length_in_sec=[0.75, 0.5, 0.25]
    clustering_embedding.cfg_base.diarizer.speaker_embeddings.parameters.multiscale_weights=[0.33, 0.33, 0.33]
    clustering_embedding._cfg_tsvad.test_ds.manifest_filepath = args.manifest
    clustering_embedding._cfg_tsvad.test_ds.emb_dir = args.embedding_dir
    clustering_embedding.prepare_cluster_embs_infer()

    # clustering_embedding = ClusterEmbedding(cfg_base=cfg)
    # clustering_embedding.prepare_cluster_embs()
    ts_vad_model.get_emb_clus_infer(clustering_embedding)
    out_rttm_dir = args.embedding_dir
    preds_list = decode_diarization(ts_vad_model, max_num_of_spks, out_rttm_dir, args.manifest, batch_size=100, embedding_dir=args.embedding_dir, device=device)
    der_list = []
    thres_range = np.arange(0.0, 0.99, 0.05).tolist()
    for threshold in thres_range:
    # if True:
        # threshold = 0.64
        print("===================> threshold:", threshold)
        score = diar_eval(args.manifest, max_num_of_spks, preds_list, ts_vad_model.clus_test_label_dict, threshold, infer_overlap=True, collar=0.25, ignore_overlap=True)
        score = diar_eval(args.manifest, max_num_of_spks, preds_list, ts_vad_model.clus_test_label_dict, threshold, infer_overlap=True, collar=0.0, ignore_overlap=False)
        der_list.append(abs(score[0]))
        # import ipdb; ipdb.set_trace()
    max_idx = np.argmin(np.array(der_list))
    print("best_thres:", thres_range[max_idx], "the best DER: ", der_list[max_idx])
    print("Without overlap inference:")
    score = diar_eval(args.manifest, max_num_of_spks, preds_list, ts_vad_model.clus_test_label_dict, threshold, infer_overlap=False, collar=0.0, ignore_overlap=False)
    
if __name__ == '__main__':
    main()

