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
Given a manifest containing audio files, first extract and save audio features, 
then perform voice activity detection (VAD) and save the detected speech segments 
as RTTM files. The manifest file must contain `audio_filepath` field:
{"audio_filepath": "/path/to/audio.wav"}

Example usage with default configurations:
```bash
python audio_to_rttm.py \
    --config-path=./configs --config-name=vad_inference_postprocess \
    manifest_filepath=/PATH/TO/MANIFEST.json output_dir=./vad_output
```
"""


import json
import os
import time
from pathlib import Path
from typing import Callable

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm import tqdm

from nemo.collections.asr.models import EncDecClassificationModel
from nemo.collections.asr.parts.utils.vad_utils import (
    extract_audio_features,
    generate_overlap_vad_seq,
    generate_vad_segment_table,
    get_vad_stream_status,
    init_vad_model,
    prepare_manifest,
    setup_feature_segment_infer_dataloader,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


@hydra_runner(config_path="./configs", config_name="vad_inference_postprocess.yaml")
def main(cfg):

    # setup profiling, note that profiling will significantly increast the total runtime
    if cfg.profiling:
        logging.info("Profiling enabled")
        profile_fn = profile
        record_fn = record_function
    else:
        logging.info("Profiling disabled")

        @contextmanager
        def profile_fn(*args, **kwargs):
            yield

        @contextmanager
        def record_fn(*args, **kwargs):
            yield

    if not cfg.manifest_filepath:
        raise ValueError("You must input the path of json file of evaluation data")

    if not cfg.vad.model_path:
        raise ValueError("You must specify the VAD model to be used.")

    if Path(cfg.output_dir).exists():
        os.system(f"rm -rf {cfg.output_dir}")
    Path(cfg.output_dir).mkdir(parents=True)

    manifest_origin = []
    key_meta_map = {}
    with open(cfg.manifest_filepath, 'r') as manifest:
        for line in manifest.readlines():
            data = json.loads(line.strip())
            manifest_origin.append(data)
            audio_filepath = data['audio_filepath']
            uniq_audio_name = audio_filepath.split('/')[-1].rsplit('.', 1)[0]
            if uniq_audio_name in key_meta_map:
                raise ValueError("Please make sure each line is with different audio name! ")
            key_meta_map[uniq_audio_name] = {'audio_filepath': audio_filepath}

    manifest_vad_input = cfg.manifest_filepath
    if cfg.prepare_manifest.auto_split:
        logging.info("Split long audio file to avoid CUDA memory issue")
        logging.debug("Try smaller split_duration if you still have CUDA memory issue")
        if not cfg.prepared_manifest_vad_input:
            prepared_manifest_vad_input = os.path.join(cfg.output_dir, "manifest_vad_input.json")
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

    torch.set_grad_enabled(False)
    vad_model = init_vad_model(cfg.vad.model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vad_model = vad_model.to(device)
    vad_model.eval()

    # Extract and save audio features
    logging.info("Extracting audio features...")
    vad_model.setup_test_data(
        test_data_config={
            'batch_size': 1,
            'vad_stream': False,
            'sample_rate': 16000,
            'manifest_filepath': manifest_vad_input,
            'labels': ['infer',],
            'num_workers': cfg.num_workers,
            'shuffle': False,
            'window_length_in_sec': cfg.vad.parameters.window_length_in_sec,
            'shift_length_in_sec': cfg.vad.parameters.shift_length_in_sec,
            'trim_silence': False,
            'normalize_audio': cfg.vad.parameters.normalize_audio,
        }
    )
    feat_dir = Path(cfg.output_dir) / Path("features")
    feat_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    feat_dir = extract_audio_features(vad_model, manifest_vad_input, str(feat_dir))
    t1 = time.time()
    manifest_input_data = load_manifest(manifest_vad_input)
    manifest_data_feat = update_audio_manifest(manifest_input_data, feat_dir, ".pt", "feature_file")
    vad_manifest_feat_file = Path(cfg.output_dir, "manifest_vad_input_feat.json")
    save_manifest(manifest_data_feat, vad_manifest_feat_file)
    logging.info(f"Audio features saved at {feat_dir}")
    logging.info(f"Time elapsed: {t1 - t0: .2f} seconds")
    logging.info("-------------------------------------")

    # Run VAD inference
    use_feat = cfg.vad.get("use_feat", False)
    if use_feat:
        logging.info("VAD using pre-calculated features as input")
        test_data_config = {
            'manifest_filepath': str(vad_manifest_feat_file),
            'labels': ['infer',],
            'num_workers': cfg.num_workers,
            'shuffle': False,
            'window_length_in_sec': cfg.vad.parameters.window_length_in_sec,
            'shift_length_in_sec': cfg.vad.parameters.shift_length_in_sec,
        }
        test_dataloader = setup_feature_segment_infer_dataloader(test_data_config)
        vad_model._test_dl = test_dataloader
    else:
        logging.info("VAD using raw audios as input")
        vad_model.setup_test_data(
            test_data_config={
                'batch_size': 1,
                'vad_stream': True,
                'sample_rate': 16000,
                'manifest_filepath': manifest_vad_input,
                'labels': ['infer',],
                'num_workers': cfg.num_workers,
                'shuffle': False,
                'window_length_in_sec': cfg.vad.parameters.window_length_in_sec,
                'shift_length_in_sec': cfg.vad.parameters.shift_length_in_sec,
                'trim_silence': False,
                'normalize_audio': cfg.vad.parameters.normalize_audio,
            }
        )
    logging.info("Generating frame-level prediction ")
    pred_dir = Path(cfg.output_dir) / Path("frame_pred")
    pred_dir.mkdir(parents=True, exist_ok=True)

    with profile_fn(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True
    ) as prof:
        t0 = time.time()
        pred_dir = generate_vad_frame_pred(
            vad_model=vad_model,
            window_length_in_sec=cfg.vad.parameters.window_length_in_sec,
            shift_length_in_sec=cfg.vad.parameters.shift_length_in_sec,
            manifest_vad_input=manifest_vad_input,
            out_dir=str(pred_dir),
            use_feat=use_feat,
            record_fn=record_fn,
        )
        t1 = time.time()

    if cfg.profiling:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        print("--------------------------------------------------------------------\n")

    logging.info(
        f"Finished generating VAD frame level prediction with window_length_in_sec={cfg.vad.parameters.window_length_in_sec} and shift_length_in_sec={cfg.vad.parameters.shift_length_in_sec}"
    )
    logging.info(f"Time elapsed: {t1 - t0: .2f} seconds")
    logging.info("-------------------------------------")

    frame_length_in_sec = cfg.vad.parameters.shift_length_in_sec

    # overlap smoothing filter
    if cfg.vad.parameters.smoothing:
        # Generate predictions with overlapping input segments. Then a smoothing filter is applied to decide the label for a frame spanned by multiple segments.
        # smoothing_method would be either in majority vote (median) or average (mean)
        logging.info("Generating predictions with overlapping input segments")
        t0 = time.time()
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
        t1 = time.time()
        logging.info(f"Time elapsed: {t1 - t0: .2f} seconds")
        logging.info("-------------------------------------")
        pred_dir = smoothing_pred_dir
        frame_length_in_sec = 0.01

    # Turn frame-wise prediction into speech intervals
    logging.info(f"Generating segment tables with postprocessing params: {cfg.vad.parameters.postprocessing}")
    segment_dir = Path(cfg.output_dir) / Path("segment_predictions")
    segment_dir.mkdir(exist_ok=True, parents=True)
    t0 = time.time()
    segment_dir = generate_vad_segment_table(
        vad_pred_dir=pred_dir,
        postprocessing_params=cfg.vad.parameters.postprocessing,
        frame_length_in_sec=frame_length_in_sec,
        num_workers=cfg.num_workers,
        out_dir=segment_dir,
    )
    t1 = time.time()
    logging.info(f"Time elapsed: {t1 - t0: .2f} seconds")
    logging.info("-------------------------------------")

    logging.info(f"Generating manifest with features and rttm files")
    dump_final_manifest(manifest_origin, segment_dir, feat_dir, cfg.output_dir)
    logging.info("Finished.")


def get_files_map(files_dir: str, ext: str = "") -> dict:
    data = {}
    for filepath in Path(files_dir).glob(f"*{ext}"):
        data[filepath.stem] = str(filepath.absolute())
    return data


def dump_final_manifest(manifest_origin: dict, rttm_dir: str, feat_dir: str, out_dir: str) -> None:
    rttm_map = get_files_map(rttm_dir, ".rttm")
    feat_map = get_files_map(feat_dir, ".pt")
    out_file = Path(out_dir, "manifest_vad_feat_rttm.json")
    with out_file.open("w") as fout:
        for item in manifest_origin:
            key = Path(item["audio_filepath"]).stem
            item["rttm_file"] = rttm_map[key]
            item["feature_file"] = feat_map[key]
            fout.write(f"{json.dumps(item)}\n")
    logging.info(f"Manifest saved to {out_file}")


def update_audio_manifest(manifest_data: dict, files_dir: str, pattern: str, new_key: str) -> dict:
    manifest_new = []
    files_map = get_files_map(files_dir, pattern)
    for item in manifest_data:
        key = Path(item["audio_filepath"]).stem
        item[new_key] = files_map[key]
        manifest_new.append(item)
    return manifest_new


def save_manifest(manifest_data: dict, out_file: str) -> None:
    with Path(out_file).open("w") as fout:
        for item in manifest_data:
            fout.write(f"{json.dumps(item)}\n")


def load_manifest(manifest_file: str) -> dict:
    data = []
    with Path(manifest_file).open("r") as fin:
        for line in fin.readlines():
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def generate_vad_frame_pred(
    vad_model: EncDecClassificationModel,
    window_length_in_sec: float,
    shift_length_in_sec: float,
    manifest_vad_input: str,
    out_dir: str,
    use_feat: bool = False,
    record_fn: Callable = None,
) -> str:
    """
    Generate VAD frame level prediction and write to out_dir
    """
    time_unit = int(window_length_in_sec / shift_length_in_sec)
    trunc = int(time_unit / 2)
    trunc_l = time_unit - trunc
    all_len = 0

    data = []
    with open(manifest_vad_input, 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            file = json.loads(line)['audio_filepath'].split("/")[-1]
            data.append(file.split(".wav")[0])
    logging.info(f"Inference on {len(data)} audio files/json lines!")

    status = get_vad_stream_status(data)

    with record_fn("infer_loop"):
        for i, test_batch in enumerate(tqdm(vad_model.test_dataloader(), total=len(vad_model.test_dataloader()))):
            test_batch = [x.to(vad_model.device) for x in test_batch]
            with autocast():
                with record_fn("infer_model"):
                    if use_feat:
                        log_probs = vad_model(processed_signal=test_batch[0], processed_signal_length=test_batch[1])
                    else:
                        log_probs = vad_model(input_signal=test_batch[0], input_signal_length=test_batch[1])

                with record_fn("infer_other"):
                    probs = torch.softmax(log_probs, dim=-1)
                    pred = probs[:, 1]

                    if status[i] == 'start':
                        to_save = pred[:-trunc]
                    elif status[i] == 'next':
                        to_save = pred[trunc:-trunc_l]
                    elif status[i] == 'end':
                        to_save = pred[trunc_l:]
                    else:
                        to_save = pred

                    all_len += len(to_save)
                    outpath = os.path.join(out_dir, data[i] + ".frame")
                    with open(outpath, "a", encoding='utf-8') as fout:
                        for f in range(len(to_save)):
                            fout.write('{0:0.4f}\n'.format(to_save[f]))

                    del test_batch
                    if status[i] == 'end' or status[i] == 'single':
                        logging.debug(f"Overall length of prediction of {data[i]} is {all_len}!")
                        all_len = 0
    return out_dir


if __name__ == '__main__':
    main()
