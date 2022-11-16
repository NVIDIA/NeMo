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
from sklearn.metrics import accuracy_score, roc_auc_score
from src.multi_classification_models import EncDecMultiClassificationModel
from src.vad_utils import align_labels_to_frames, generate_vad_frame_pred, generate_vad_segment_table, prepare_manifest

from nemo.core.config import hydra_runner
from nemo.utils import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@hydra_runner(config_path="../conf/vad", config_name="vad_inference_postprocessing.yaml")
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

    # each line of dataset should be have different audio_filepath and unique name to simplify edge cases or conditions
    key_meta_map = {}
    all_labels_map = {}
    with open(cfg.dataset, 'r') as manifest:
        for line in manifest.readlines():
            data = json.loads(line.strip())
            audio_filepath = data['audio_filepath']
            uniq_audio_name = audio_filepath.split('/')[-1].rsplit('.', 1)[0]
            if uniq_audio_name in key_meta_map:
                raise ValueError("Please make sure each line is with different audio name! ")
            key_meta_map[uniq_audio_name] = {'audio_filepath': audio_filepath, 'label': data["label"]}
            all_labels_map[uniq_audio_name] = [int(x) for x in data["label"].split()]

    # Prepare manifest for streaming VAD
    manifest_vad_input = cfg.dataset
    if cfg.prepare_manifest.auto_split:
        logging.info("Split long audio file to avoid CUDA memory issue")
        logging.debug("Try smaller split_duration if you still have CUDA memory issue")
        if not cfg.prepared_manifest_vad_input:
            prepared_manifest_vad_input = os.path.join(cfg.frame_out_dir, "manifest_vad_input.json")
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

    # init and load model
    torch.set_grad_enabled(False)
    vad_model = EncDecMultiClassificationModel.restore_from(restore_path=cfg.vad.model_path)

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
        out_dir=os.path.join(cfg.frame_out_dir, "frame_predictions"),
    )

    logging.info(
        f"Finish generating VAD frame level prediction with window_length_in_sec={cfg.vad.parameters.window_length_in_sec} and shift_length_in_sec={cfg.vad.parameters.shift_length_in_sec}"
    )

    # calculate AUROC
    predictions = []
    groundtruth = []
    ratios = []
    for key in all_labels_map:
        probs = all_probs_map[key]
        labels = all_labels_map[key]
        labels_aligned = align_labels_to_frames(probs, labels)
        ratios.append(len(probs) / len(labels))
        # if "10179_11051_000008" in key:
        #     import ipdb

        #     ipdb.set_trace()

        all_labels_map[key] = labels_aligned
        groundtruth += labels_aligned
        predictions += probs

    print(ratios)
    import ipdb

    ipdb.set_trace()

    auroc = roc_auc_score(y_true=groundtruth, y_score=predictions)
    threshold = cfg.vad.parameters.get("threshold", 0.5)
    pred_labels = [int(x > threshold) for x in predictions]
    acc = accuracy_score(y_true=groundtruth, y_pred=pred_labels)
    logging.info("=====================================")
    logging.info(f"AUROC: {auroc:0.4f}")
    logging.info(f"Acc: {acc:0.4f}, threshold: {threshold:0.1f}")
    logging.info("=====================================")
    del predictions
    del groundtruth

    frame_length_in_sec = cfg.vad.parameters.shift_length_in_sec

    gt_frames_dir = dump_groundtruth_frames(cfg.frame_out_dir, all_labels_map)

    calculate_detection_error(
        pred_dir,
        gt_frames_dir,
        post_params=cfg.vad.parameters.postprocessing,
        frame_length_in_sec=frame_length_in_sec,
        num_workers=cfg.num_workers,
    )

    # # postprocessing and generate speech segments
    # if cfg.gen_seg_table:
    #     logging.info("Converting frame level prediction to speech/no-speech segment in start and end times format.")
    #     table_out_dir = generate_vad_segment_table(
    #         vad_pred_dir=pred_dir,
    #         postprocessing_params=cfg.vad.parameters.postprocessing,
    #         frame_length_in_sec=frame_length_in_sec,
    #         num_workers=cfg.num_workers,
    #         out_dir=cfg.table_out_dir,
    #     )
    #     logging.info(
    #         f"Finish generating speech semgents table with postprocessing_params: {cfg.vad.parameters.postprocessing}"
    #     )

    # if cfg.write_to_manifest:
    #     for i in key_meta_map:
    #         key_meta_map[i]['rttm_filepath'] = os.path.join(table_out_dir, i + ".txt")

    #     if not cfg.out_manifest_filepath:
    #         out_manifest_filepath = os.path.join(cfg.frame_out_dir, "vad_out.json")
    #     else:
    #         out_manifest_filepath = cfg.out_manifest_filepath
    #     out_manifest_filepath = write_rttm2manifest(key_meta_map, out_manifest_filepath)
    #     logging.info(f"Writing VAD output to manifest: {out_manifest_filepath}")


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
    del report
    metric.reset()  # reset internal accumulator


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
    try:
        pred = pd.read_csv(pred_table_path, sep=" ", header=None)
        label = pd.read_csv(gt_table_path, sep=" ", header=None)
    except Exception as e:
        print(pred_table_path, gt_table_path)
        import ipdb

        ipdb.set_trace()
        print(e)

    # construct reference
    reference = Annotation()
    for index, row in label.iterrows():
        reference[Segment(float(row[0]), float(row[0]) + float(row[1]))] = 'Speech'

    # construct hypothsis
    hypothesis = Annotation()
    for index, row in pred.iterrows():
        hypothesis[Segment(float(row[0]), float(row[0]) + float(row[1]))] = 'Speech'
    return reference, hypothesis


if __name__ == '__main__':
    main()
