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
During inference, we perform frame-level prediction by two approaches:
    1) shift the window of length window_length_in_sec (e.g. 0.63s) by shift_length_in_sec (e.g. 10ms) to generate the frame and use the prediction of the window to represent the label for the frame;
       [this script demonstrate how to do this approach]
    2) generate predictions with overlapping input segments. Then a smoothing filter is applied to decide the label for a frame spanned by multiple segments.
       [get frame level prediction by this script and use vad_overlap_posterior.py in NeMo/scripts/voice_activity_detection
       One can also find posterior about converting frame level prediction
       to speech/no-speech segment in start and end times format in that script.]

       Image https://raw.githubusercontent.com/NVIDIA/NeMo/main/tutorials/asr/images/vad_post_overlap_diagram.png
       will help you understand this method.

This script will also help you perform postprocessing and generate speech segments if needed

Usage:
python vad_infer.py --config-path="../conf/vad" --config-name="vad_inference_postprocessing.yaml" dataset=<Path of json file of evaluation data. Audio files should have unique names>

"""
import json
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
import torch
from pyannote.core import Annotation, Segment
from pyannote.metrics import detection
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from src.multi_classification_models import EncDecMultiClassificationModel
from src.vad_utils import align_labels_to_frames, generate_vad_frame_pred, generate_vad_segment_table, prepare_manifest

from nemo.collections.asr.parts.utils.speaker_utils import write_rttm2manifest
from nemo.core.config import hydra_runner
from nemo.utils import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@hydra_runner(config_path="./configs", config_name="vad_inference_postprocessing.yaml")
def main(cfg):
    if not cfg.dataset:
        raise ValueError("You must input the path of json file of evaluation data")

    if not os.path.exists(cfg.frame_out_dir):
        os.mkdir(cfg.frame_out_dir)
    else:
        logging.info(f"Found existing dir: {cfg.frame_out_dir}, remove and create new one...")
        os.system(f"rm -rf {cfg.frame_out_dir}")
        os.mkdir(cfg.frame_out_dir)
        # logging.warning(
        #     "Note frame_out_dir exists. If new file has same name as file inside existing folder, it will append result to existing file and might cause mistakes for next steps."
        # )

    # init and load model
    torch.set_grad_enabled(False)
    vad_model = EncDecMultiClassificationModel.restore_from(restore_path=cfg.vad.model_path)

    manifest_list = cfg.dataset
    if isinstance(manifest_list, str):
        manifest_list = manifest_list.split(',')

    probs_dict = {}
    labels_dict = {}
    reports_dict = {}
    pred_seg_dir_dict = {}
    gt_seg_dir_dict = {}
    for manifest_file in manifest_list:
        filename = Path(manifest_file).stem
        out_dir = str(Path(cfg.frame_out_dir) / Path(f"vad_output_{filename}"))
        logging.info("====================================================")
        logging.info(f"Start evaluating manifest: {manifest_file}")
        probs, labels, report, pred_segment_dir, gt_segment_dir = evaluate_single_manifest(
            manifest_file, cfg, vad_model, out_dir
        )
        probs_dict[filename] = probs
        labels_dict[filename] = labels
        reports_dict[filename] = report
        pred_seg_dir_dict[filename] = pred_segment_dir
        gt_seg_dir_dict[filename] = gt_segment_dir

    logging.info("=========================================================")
    logging.info("Calculating aggregated Detection Error...")
    all_der_report = calculate_multi_detection_error(pred_seg_dir_dict, gt_seg_dir_dict)

    logging.info("====================================================")
    logging.info("Finalizing individual results...")
    threshold = cfg.vad.parameters.get("threshold", 0.5)

    all_probs = []
    all_labels = []
    for key in probs_dict:
        probs = probs_dict[key]
        labels = labels_dict[key]

        all_probs += probs
        all_labels += labels

        auroc = roc_auc_score(y_true=labels, y_score=probs)
        pred_labels = [int(x > threshold) for x in probs]
        clf_report = classification_report(y_true=labels, y_pred=pred_labels)
        logging.info(f"================= {key} =================")
        logging.info(f"AUROC: {auroc:0.4f}")
        logging.info(f"Classification report with threshold={threshold:.2f}")
        logging.info(clf_report)

        der_report = reports_dict[key]
        DetER = der_report.iloc[[-1]][('detection error rate', '%')].item()
        FA = der_report.iloc[[-1]][('false alarm', '%')].item()
        MISS = der_report.iloc[[-1]][('miss', '%')].item()
        logging.info(f"Detection Error Rate: DetER={DetER:0.4f}, False Alarm={FA:0.4f}, Miss={MISS:0.4f}")
        logging.info("==========================================\n\n")

    logging.info("================== Aggregrated Results ===================")
    DetER = all_der_report.iloc[[-1]][('detection error rate', '%')].item()
    FA = all_der_report.iloc[[-1]][('false alarm', '%')].item()
    MISS = all_der_report.iloc[[-1]][('miss', '%')].item()
    logging.info(f"============================================================")
    logging.info(f" DetER={DetER:0.4f}, False Alarm={FA:0.4f}, Miss={MISS:0.4f}")
    logging.info(f"============================================================")

    auroc = roc_auc_score(y_true=all_labels, y_score=all_probs)
    pred_labels = [int(x > threshold) for x in all_probs]
    clf_report = classification_report(y_true=all_labels, y_pred=pred_labels)
    logging.info(f"AUROC: {auroc:0.4f}")
    logging.info(f"Classification report with threshold={threshold:.2f}")
    logging.info(f"\n{clf_report}")


def evaluate_single_manifest(manifest_filepath, cfg, vad_model, out_dir):

    Path(out_dir).mkdir(exist_ok=True)

    # each line of dataset should be have different audio_filepath and unique name to simplify edge cases or conditions
    key_meta_map = {}
    all_labels_map = {}
    with open(manifest_filepath, 'r') as manifest:
        for line in manifest.readlines():
            data = json.loads(line.strip())
            audio_filepath = data['audio_filepath']
            uniq_audio_name = audio_filepath.split('/')[-1].rsplit('.', 1)[0]
            if uniq_audio_name in key_meta_map:
                raise ValueError("Please make sure each line is with different audio name! ")
            key_meta_map[uniq_audio_name] = {'audio_filepath': audio_filepath, 'label': data["label"]}
            all_labels_map[uniq_audio_name] = [int(x) for x in data["label"].split()]

    # Prepare manifest for streaming VAD
    manifest_vad_input = manifest_filepath
    if cfg.prepare_manifest.auto_split:
        logging.info("Split long audio file to avoid CUDA memory issue")
        logging.debug("Try smaller split_duration if you still have CUDA memory issue")
        if not cfg.prepared_manifest_vad_input:
            prepared_manifest_vad_input = os.path.join(out_dir, "manifest_vad_input.json")
        else:
            prepared_manifest_vad_input = cfg.prepared_manifest_vad_input
        config = {
            'input': manifest_vad_input,
            'window_length_in_sec': cfg.vad.parameters.window_length_in_sec,
            'split_duration': cfg.prepare_manifest.split_duration,
            'num_workers': cfg.num_workers,
            'prepared_manifest_vad_input': prepared_manifest_vad_input,
        }
        manifest_vad_input = prepare_manifest(config)
    else:
        logging.warning(
            "If you encounter CUDA memory issue, try splitting manifest entry by split_duration to avoid it."
        )

    # setup_test_data
    vad_model.setup_test_data(
        test_data_config={
            'batch_size': 1,
            'sample_rate': 16000,
            'manifest_filepath': manifest_vad_input,
            'labels': ['infer',],
            'num_workers': cfg.num_workers,
            'shuffle': False,
        }
    )

    vad_model = vad_model.to(device)
    vad_model.eval()

    logging.info("Generating frame-level prediction ")
    pred_dir, all_probs_map = generate_vad_frame_pred(
        vad_model=vad_model,
        window_length_in_sec=cfg.vad.parameters.window_length_in_sec,
        shift_length_in_sec=cfg.vad.parameters.shift_length_in_sec,
        manifest_vad_input=manifest_vad_input,
        out_dir=os.path.join(out_dir, "frames_predictions"),
    )

    logging.info(
        f"Finish generating VAD frame level prediction with window_length_in_sec={cfg.vad.parameters.window_length_in_sec} and shift_length_in_sec={cfg.vad.parameters.shift_length_in_sec}"
    )

    # calculate AUROC
    predictions = []
    groundtruth = []
    for key in all_labels_map:
        probs = all_probs_map[key]
        labels = all_labels_map[key]
        labels_aligned = align_labels_to_frames(probs, labels)
        all_labels_map[key] = labels_aligned
        groundtruth += labels_aligned
        predictions += probs

    # auroc = roc_auc_score(y_true=groundtruth, y_score=predictions)
    # threshold = cfg.vad.parameters.get("threshold", 0.5)
    # pred_labels = [int(x > threshold) for x in predictions]
    # acc = accuracy_score(y_true=groundtruth, y_pred=pred_labels)
    # logging.info("=====================================")
    # logging.info(f"AUROC: {auroc:0.4f}")
    # logging.info(f"Acc: {acc:0.4f}, threshold: {threshold:0.1f}")
    # logging.info("=====================================")

    frame_length_in_sec = cfg.vad.parameters.shift_length_in_sec

    gt_frames_dir = dump_groundtruth_frames(out_dir, all_labels_map)

    report, pred_segment_dir, gt_segment_dir = calculate_detection_error(
        pred_dir,
        gt_frames_dir,
        post_params=cfg.vad.parameters.postprocessing,
        frame_length_in_sec=frame_length_in_sec,
        num_workers=cfg.num_workers,
    )

    return predictions, groundtruth, report, pred_segment_dir, gt_segment_dir


def calculate_multi_detection_error(pred_seg_dir_dict: dict, gt_seg_dir_dict: dict):
    all_paired_files = []
    for key in gt_seg_dir_dict:
        if key not in pred_seg_dir_dict:
            continue
        gt_seg_dir = gt_seg_dir_dict[key]
        pred_seg_dir = pred_seg_dir_dict[key]
        paired_files = find_paired_files(pred_dir=pred_seg_dir, gt_dir=gt_seg_dir)
        all_paired_files += paired_files

    metric = detection.DetectionErrorRate()
    for key, gt_file, pred_file in paired_files:
        reference, hypothesis = vad_frame_construct_pyannote_object_per_file(pred_file, gt_file)
        metric(reference, hypothesis)  # accumulation

    report = metric.report(display=False)
    return report


def calculate_detection_error(
    vad_pred_frame_dir: str,
    vad_gt_frame_dir: str,
    post_params: dict,
    frame_length_in_sec: float = 0.01,
    num_workers: int = 20,
):

    logging.info("Generating segment tables for predictions")
    pred_segment_dir = str(Path(vad_pred_frame_dir) / Path("pred_segments"))
    pred_segment_dir = generate_vad_segment_table(
        vad_pred_frame_dir,
        post_params,
        frame_length_in_sec=frame_length_in_sec,
        num_workers=num_workers,
        out_dir=pred_segment_dir,
    )

    logging.info("Generating segment tables for groundtruths")
    gt_segment_dir = Path(vad_gt_frame_dir) / Path("gt_segments")
    gt_segment_dir = generate_gt_segment_table(
        vad_gt_frame_dir, frame_length_in_sec=frame_length_in_sec, num_workers=num_workers, out_dir=gt_segment_dir
    )

    paired_files = find_paired_files(pred_dir=pred_segment_dir, gt_dir=gt_segment_dir)
    metric = detection.DetectionErrorRate()

    logging.info("Calculating detection error metrics...")
    # add reference and hypothesis to metrics
    for key, gt_file, pred_file in paired_files:
        reference, hypothesis = vad_frame_construct_pyannote_object_per_file(pred_file, gt_file)
        metric(reference, hypothesis)  # accumulation

    # delete tmp table files
    # shutil.rmtree(pred_segment_dir, ignore_errors=True)
    # shutil.rmtree(gt_segment_dir, ignore_errors=True)

    report = metric.report(display=False)
    DetER = report.iloc[[-1]][('detection error rate', '%')].item()
    FA = report.iloc[[-1]][('false alarm', '%')].item()
    MISS = report.iloc[[-1]][('miss', '%')].item()
    total = report.iloc[[-1]]['total'].item()

    logging.info(f"parameter {post_params}, DetER={DetER:0.4f}, False Alarm={FA:0.4f}, Miss={MISS:0.4f}")
    metric.reset()  # reset internal accumulator
    return report, pred_segment_dir, gt_segment_dir


def dump_groundtruth_frames(out_dir, labels_map):
    out_dir = Path(out_dir) / Path("frames_groundtruth")
    out_dir.mkdir(exist_ok=True)
    for k, v in labels_map.items():
        out_file = out_dir / Path(f"{k}.frame")
        with out_file.open("a") as fout:
            for x in v:
                fout.write(f"{x}\n")
    return str(out_dir)


def generate_gt_segment_table(
    vad_pred_dir: str, frame_length_in_sec: float, num_workers: int, out_dir: str = None,
):
    params = {
        "onset": 0.5,  # onset threshold for detecting the beginning and end of a speech
        "offset": 0.5,  # offset threshold for detecting the end of a speech.
        "pad_onset": 0.0,  # adding durations before each speech segment
        "pad_offset": 0.0,  # adding durations after each speech segment
        "min_duration_on": 0.0,  # threshold for small non_speech deletion
        "min_duration_off": 0.0,  # threshold for short speech segment deletion
        "filter_speech_first": False,
    }
    vad_table_dir = generate_vad_segment_table(
        vad_pred_dir, params, frame_length_in_sec=frame_length_in_sec, num_workers=num_workers, out_dir=out_dir
    )
    return vad_table_dir


def find_paired_files(pred_dir, gt_dir):
    pred_files = list(Path(pred_dir).glob("*.txt"))
    gt_files = list(Path(gt_dir).glob("*.txt"))

    gt_file_map = {}
    for filepath in gt_files:
        fname = Path(filepath).stem
        gt_file_map[fname] = str(filepath)

    pred_file_map = {}
    for filepath in pred_files:
        fname = Path(filepath).stem
        pred_file_map[fname] = str(filepath)

    results = []
    for key in gt_file_map:
        if key in pred_file_map:
            results.append((key, gt_file_map[key], pred_file_map[key]))
    return results


def vad_frame_construct_pyannote_object_per_file(
    pred_table_path: str, gt_table_path: str
) -> Tuple[Annotation, Annotation]:
    """
    Construct a Pyannote object for evaluation.
    Args:
        pred_table_path(str) : path of vad rttm-like table.
        gt_table_path(str): path of groundtruth rttm file.
    Returns:
        reference(pyannote.Annotation): groundtruth
        hypothesis(pyannote.Annotation): prediction
    """

    pred = pd.read_csv(pred_table_path, sep=" ", header=None)
    label = pd.read_csv(gt_table_path, sep=" ", header=None)

    # construct reference
    reference = Annotation()
    for index, row in label.iterrows():
        reference[Segment(float(row[0]), float(row[0]) + float(row[1]))] = 'Speech'

    # construct hypothsis
    hypothesis = Annotation()
    for index, row in pred.iterrows():
        hypothesis[Segment(float(row[0]), float(row[0]) + float(row[1]))] = 'Speech'
    return reference, hypothesis


def load_manifests(manifest_filepath: Union[str, List[str]]):
    key_meta_map = {}
    all_labels_map = {}
    with open(manifest_filepath, 'r') as manifest:
        for line in manifest.readlines():
            data = json.loads(line.strip())
            audio_filepath = data['audio_filepath']
            uniq_audio_name = audio_filepath.split('/')[-1].rsplit('.', 1)[0]
            if uniq_audio_name in key_meta_map:
                raise ValueError("Please make sure each line is with different audio name! ")
            key_meta_map[uniq_audio_name] = {'audio_filepath': audio_filepath, 'label': data["label"]}
            all_labels_map[uniq_audio_name] = [int(x) for x in data["label"].split()]
    return key_meta_map, all_labels_map


if __name__ == '__main__':
    main()
