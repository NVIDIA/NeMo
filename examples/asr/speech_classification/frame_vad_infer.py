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

"""
This script peforms VAD on each 20ms frames of the input audio files. 
Postprocessing is also performed to generate speech segments and store them as RTTM files.
Long audio files will be splitted into smaller chunks to avoid OOM issues, but the frames close
to the split points might have worse performance due to truncated context.

## Usage:
python frame_vad_infer.py \
    --config-path="../conf/vad" --config-name="frame_vad_infer_postprocess" \
    dataset=<Path of json file of evaluation data. Audio files should have unique names>
"""

import json
import os
from pathlib import Path

import torch
from pyannote.metrics import detection
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from nemo.collections.asr.parts.utils.manifest_utils import write_manifest
from nemo.collections.asr.parts.utils.vad_utils import (
    align_labels_to_frames,
    frame_vad_construct_pyannote_object_per_file,
    generate_overlap_vad_seq,
    generate_vad_frame_pred,
    generate_vad_segment_table,
    get_frame_labels,
    init_frame_vad_model,
    load_speech_segments_from_rttm,
    prepare_manifest,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra_runner(config_path="../conf/vad", config_name="frame_vad_infer_postprocess")
def main(cfg):
    if not cfg.dataset:
        raise ValueError("You must input the path of json file of evaluation data")

    # each line of dataset should be have different audio_filepath and unique name to simplify edge cases or conditions
    key_meta_map = {}
    key_labels_map = {}
    key_rttm_map = {}
    manifest_orig = []
    manifest_dir = Path(cfg.dataset).absolute().parent
    with open(cfg.dataset, 'r') as manifest:
        for line in manifest.readlines():
            entry = json.loads(line.strip())
            audio_filepath = Path(entry['audio_filepath'])
            if not audio_filepath.is_absolute():
                new_audio_filepath = manifest_dir / audio_filepath
                if new_audio_filepath.is_file():
                    audio_filepath = new_audio_filepath
                    entry['audio_filepath'] = str(audio_filepath)
            uniq_audio_name = audio_filepath.stem
            if uniq_audio_name in key_meta_map:
                raise ValueError("Please make sure each line is with different audio_filepath! ")
            key_meta_map[uniq_audio_name] = {'audio_filepath': str(audio_filepath)}
            manifest_orig.append(entry)

            if "label" not in entry:
                rttm_key = "rttm_filepath" if "rttm_filepath" in entry else "rttm_file"
                segments = load_speech_segments_from_rttm(entry[rttm_key])
                label_str = get_frame_labels(
                    segments=segments,
                    frame_length=cfg.vad.parameters.shift_length_in_sec,
                    duration=entry['duration'],
                    offset=entry['offset'],
                )
                key_rttm_map[uniq_audio_name] = entry[rttm_key]
                key_labels_map[uniq_audio_name] = [float(x) for x in label_str.split()]
            else:
                key_labels_map[uniq_audio_name] = [float(x) for x in entry["label"].split()]

    # Prepare manifest for streaming VAD
    manifest_vad_input = cfg.dataset
    if cfg.prepare_manifest.auto_split:
        logging.info("Split long audio file to avoid CUDA memory issue")
        logging.debug("Try smaller split_duration if you still have CUDA memory issue")
        config = {
            'input': manifest_vad_input,
            'window_length_in_sec': cfg.vad.parameters.window_length_in_sec,
            'split_duration': cfg.prepare_manifest.split_duration,
            'num_workers': cfg.num_workers,
            'prepared_manifest_vad_input': cfg.prepared_manifest_vad_input,
        }
        manifest_vad_input = prepare_manifest(config)
    else:
        logging.warning(
            "If you encounter CUDA memory issue, try splitting manifest entry by split_duration to avoid it."
        )

    torch.set_grad_enabled(False)
    vad_model = init_frame_vad_model(cfg.vad.model_path)

    # setup_test_data
    vad_model.setup_test_data(
        test_data_config={
            'batch_size': 1,
            'sample_rate': 16000,
            'manifest_filepath': manifest_vad_input,
            'labels': ['infer'],
            'num_workers': cfg.num_workers,
            'shuffle': False,
            'normalize_audio_db': cfg.vad.parameters.normalize_audio_db,
            'normalize_audio_db_target': cfg.vad.parameters.normalize_audio_db_target,
        }
    )

    vad_model = vad_model.to(device)
    vad_model.eval()

    if not os.path.exists(cfg.frame_out_dir):
        logging.info(f"Frame predictions do not exist at {cfg.frame_out_dir}, generating frame prediction.")
        os.mkdir(cfg.frame_out_dir)
        extract_frame_preds = True
    else:
        logging.info(f"Frame predictions already exist at {cfg.frame_out_dir}, skipping frame prediction generation.")
        extract_frame_preds = False

    if extract_frame_preds:
        logging.info("Generating frame-level prediction ")
        pred_dir = generate_vad_frame_pred(
            vad_model=vad_model,
            window_length_in_sec=cfg.vad.parameters.window_length_in_sec,
            shift_length_in_sec=cfg.vad.parameters.shift_length_in_sec,
            manifest_vad_input=manifest_vad_input,
            out_dir=cfg.frame_out_dir,
        )
        logging.info(f"Finish generating VAD frame level prediction. You can find the prediction in {pred_dir}")
    else:
        pred_dir = cfg.frame_out_dir

    frame_length_in_sec = cfg.vad.parameters.shift_length_in_sec

    # overlap smoothing filter
    if cfg.vad.parameters.smoothing:
        # Generate predictions with overlapping input segments. Then a smoothing filter is applied to decide the label for a frame spanned by multiple segments.
        # smoothing_method would be either in majority vote (median) or average (mean)
        logging.info("Generating predictions with overlapping input segments")
        smoothing_pred_dir = generate_overlap_vad_seq(
            frame_pred_dir=pred_dir,
            smoothing_method=cfg.vad.parameters.smoothing,
            overlap=cfg.vad.parameters.overlap,
            window_length_in_sec=cfg.vad.parameters.window_length_in_sec,
            shift_length_in_sec=cfg.vad.parameters.shift_length_in_sec,
            num_workers=cfg.num_workers,
            out_dir=cfg.smoothing_out_dir,
        )
        logging.info(
            f"Finish generating predictions with overlapping input segments with smoothing_method={cfg.vad.parameters.smoothing} and overlap={cfg.vad.parameters.overlap}"
        )
        pred_dir = smoothing_pred_dir

    # postprocessing and generate speech segments
    logging.info("Converting frame level prediction to RTTM files.")
    rttm_out_dir = generate_vad_segment_table(
        vad_pred_dir=pred_dir,
        postprocessing_params=cfg.vad.parameters.postprocessing,
        frame_length_in_sec=frame_length_in_sec,
        num_workers=cfg.num_workers,
        use_rttm=cfg.vad.use_rttm,
        out_dir=cfg.rttm_out_dir,
    )
    logging.info(
        f"Finish generating speech semgents table with postprocessing_params: {cfg.vad.parameters.postprocessing}"
    )

    logging.info("Writing VAD output to manifest")
    key_pred_rttm_map = {}
    manifest_new = []
    for entry in manifest_orig:
        key = Path(entry['audio_filepath']).stem
        entry['rttm_filepath'] = Path(os.path.join(rttm_out_dir, key + ".rttm")).absolute().as_posix()
        if not Path(entry['rttm_filepath']).is_file():
            logging.warning(f"Not able to find {entry['rttm_filepath']} for {entry['audio_filepath']}")
            entry['rttm_filepath'] = ""
        manifest_new.append(entry)
        key_pred_rttm_map[key] = entry['rttm_filepath']

    if not cfg.out_manifest_filepath:
        out_manifest_filepath = "manifest_vad_output.json"
    else:
        out_manifest_filepath = cfg.out_manifest_filepath
    write_manifest(out_manifest_filepath, manifest_new)
    logging.info(f"Finished writing VAD output to manifest: {out_manifest_filepath}")

    if cfg.get("evaluate", False):
        logging.info("Evaluating VAD results")
        all_probs = []
        all_labels = []
        metric = detection.DetectionErrorRate()
        key_probs_map = {}
        predictions_list = list(Path(pred_dir).glob("*.frame"))
        for frame_pred in tqdm(predictions_list, desc="Evaluating VAD results", total=len(predictions_list)):
            pred_probs = []
            with frame_pred.open("r") as fin:
                for line in fin.readlines():
                    line = line.strip()
                    if not line:
                        continue
                    pred_probs.append(float(line))
            key = frame_pred.stem
            key_probs_map[key] = pred_probs
            key_labels_map[key] = align_labels_to_frames(probs=pred_probs, labels=key_labels_map[key])
            all_probs.extend(key_probs_map[key])
            all_labels.extend(key_labels_map[key])

            if key in key_rttm_map:
                groundtruth = key_rttm_map[key]
            else:
                groundtruth = key_labels_map[key]

            reference, hypothesis = frame_vad_construct_pyannote_object_per_file(
                prediction=key_pred_rttm_map[key], groundtruth=groundtruth, frame_length_in_sec=frame_length_in_sec,
            )
            metric(reference, hypothesis)

        auroc = roc_auc_score(y_true=all_labels, y_score=all_probs)
        report = metric.report(display=False)
        DetER = report.iloc[[-1]][('detection error rate', '%')].item()
        FA = report.iloc[[-1]][('false alarm', '%')].item()
        MISS = report.iloc[[-1]][('miss', '%')].item()
        logging.info(f"AUROC: {auroc:.4f}")
        logging.info(f"DetER={DetER:0.4f}, False Alarm={FA:0.4f}, Miss={MISS:0.4f}")
        logging.info(f"with params: {cfg.vad.parameters.postprocessing}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
