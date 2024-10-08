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

python $BASEPATH/neural_diarizer/sortformer_diarization.py \
    model_path=/path/to/sortformer_model.nemo \
    batch_size=4 \
    dataset_manifest=/path/to/diarization_path_to_manifest.json

"""
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from nemo.collections.asr.models import SpkDiarEncLabelModel
from nemo.core.config import hydra_runner
from nemo.collections.asr.metrics.der import score_labels
from hydra.core.config_store import ConfigStore


import os
import yaml
from dataclasses import dataclass, is_dataclass
from typing import Optional, Union, List, Tuple, Dict

from pyannote.core import Segment, Timeline
from nemo.collections.asr.parts.utils.vad_utils import binarization, filtering
from nemo.collections.asr.parts.utils.speaker_utils import audio_rttm_map as get_audio_rttm_map
from nemo.collections.asr.parts.utils.speaker_utils import (
labels_to_pyannote_object,
generate_diarization_output_lines,
rttm_to_labels,
)

from tqdm import tqdm
import pytorch_lightning as pl
import torch
import logging
from omegaconf import OmegaConf
from nemo.core.config import hydra_runner


seed_everything(42)
torch.backends.cudnn.deterministic = True


@dataclass
class PostProcessingParams:
    window_length_in_sec: float = 0.15
    shift_length_in_sec: float = 0.01
    smoothing: bool = False
    overlap: float = 0.5
    onset: float = 0.5
    offset: float = 0.5
    pad_onset: float = 0.0
    pad_offset: float = 0.0
    min_duration_on: float = 0.0
    min_duration_off: float = 0.0
    filter_speech_first: bool = True

@dataclass
class DiarizationConfig:
    # Required configs
    model_path: Optional[str] = None  # Path to a .nemo file
    pretrained_name: Optional[str] = None  # Name of a pretrained model
    audio_dir: Optional[str] = None  # Path to a directory which contains audio files
    dataset_manifest: Optional[str] = None  # Path to dataset's JSON manifest
    
    audio_key: str = 'audio_filepath'  # Used to override the default audio key in dataset_manifest
    postprocessing_yaml: Optional[str] = None  # Path to a yaml file for postprocessing configurations
    eval_mode: bool = True
    no_der: bool = False
    out_rttm_dir: Optional[str] = None
    opt_style: Optional[str] = None
    
    # General configs
    output_filename: Optional[str] = None
    session_len_sec: float = -1 # End-to-end diarization session length in seconds
    batch_size: int = 4
    num_workers: int = 0
    random_seed: Optional[int] = None  # seed number going to be used in seed_everything()
    bypass_postprocessing: bool = True # If True, postprocessing will be bypassed
    
    # Eval Settings: (0.25, False) should be default setting for sortformer eval.
    collar: float = 0.25 # Collar in seconds for DER calculation
    ignore_overlap: bool = False # If True, DER will be calculated only for non-overlapping segments
    
    # Streaming diarization configs
    streaming_mode: bool = True # If True, streaming diarization will be used. For long-form audio, set mem_len=step_len
    mem_len: int = 100
    step_len: int = 100

    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    allow_mps: bool = False  # allow to select MPS device (Apple Silicon M-series GPU)
    amp: bool = False
    amp_dtype: str = "float16"  # can be set to "float16" or "bfloat16" when using amp
    matmul_precision: str = "highest"  # Literal["highest", "high", "medium"]
    audio_type: str = "wav"

    # Optuna Config
    optuna_study_name: str = "diar_study"
    storage: str = f"sqlite:///{optuna_study_name}.db"
    output_log_file: str = f"{optuna_study_name}.log"
    optuna_n_trials: int = 100000

    # Method to load YAML configuration for postprocessing
def load_postprocessing_from_yaml(postprocessing_yaml):
    # Add PostProcessingParams as a field
    postprocessing_params = OmegaConf.structured(PostProcessingParams())
    if postprocessing_yaml is None:
        logging.info(f"No postprocessing YAML file has been provided. Default postprocessing configurations will be applied.")
    else:
        # Load postprocessing params from the provided YAML file
        with open(postprocessing_yaml, 'r') as file:
            yaml_data = yaml.safe_load(file)
            # Update the postprocessing_params with the loaded values
            for key, value in yaml_data.items():
                if hasattr(postprocessing_params, key):
                    setattr(postprocessing_params, key, value)
    return postprocessing_params

<<<<<<< HEAD
=======

@dataclass
class VadParams:
    """
    Vad parameters from Optuna optimization studies. 
    Trial 2522 finished with value: 0.09605644326924494 and parameters: {'onset': 0.62, 'offset': 0.57, 'pad_onset': 0.23, 'pad_offset': 0.09, 'min_duration_on': 0.13, 'min_duration_off': 0.25}. Best is trial 2522 with value: 0.09605644326924494. (im303a e19last)
    Trial 3683 finished with value: 0.09960175732817779 and parameters: {'onset': 0.6, 'offset': 0.6, 'pad_onset': 0.22, 'pad_offset': 0.1, 'min_duration_on': 0.06, 'min_duration_off': 0.25}. Best is trial 3683 with value: 0.09960175732817779. (im303a e6-e19)
    """
    opt_style: str | None
    window_length_in_sec: float = field(init=False)
    shift_length_in_sec: float = field(init=False)
    smoothing: str = field(init=False)
    overlap: float = field(init=False)
    onset: float = field(init=False)
    offset: float = field(init=False)
    pad_onset: float = field(init=False)
    pad_offset: float = field(init=False)
    min_duration_on: float = field(init=False)
    min_duration_off: float = field(init=False)
    filter_speech_first: bool = field(init=False)

    def __post_init__(self):
        if self.opt_style == "callhome_part1":
            self.window_length_in_sec = 0.15
            self.shift_length_in_sec = 0.01
            self.smoothing = False
            self.overlap = 0.5
            self.onset = 0.62
            self.offset = 0.57
            self.pad_onset = 0.23
            self.pad_offset = 0.09
            self.min_duration_on = 0.13
            self.min_duration_off = 0.25
            self.filter_speech_first = True
        elif self.opt_style == "dh3_dev":
            self.window_length_in_sec = 0.15
            self.shift_length_in_sec = 0.01
            self.smoothing = False
            self.overlap = 0.5
            self.onset = 0.5
            self.offset = 0.5
            self.pad_onset = 0.0
            self.pad_offset = 0.0
            self.min_duration_on = 0.0
            self.min_duration_off = 0.0
            self.filter_speech_first = True
        elif self.opt_style is None:
            self.window_length_in_sec = 0.15
            self.shift_length_in_sec = 0.01
            self.smoothing = False
            self.overlap = 0.5
            self.onset = 0.5
            self.offset = 0.5
            self.pad_onset = 0.0
            self.pad_offset = 0.0
            self.min_duration_on = 0.0
            self.min_duration_off = 0.0
            self.filter_speech_first = True
        else:
            raise ValueError(f"Unknown opt_style: {self.opt_style}")
        
>>>>>>> 1fc834e44ea6de4174798a2b9bbc18b00578faf7
def timestamps_to_pyannote_object(speaker_timestamps: List[Tuple[float, float]],
                                  uniq_id: str, 
                                  audio_rttm_values: Dict[str, str], 
                                  all_hypothesis: List[Tuple[str, Timeline]], 
                                  all_reference: List[Tuple[str, Timeline]], 
                                  all_uems: List[Tuple[str, Timeline]],
                                  out_rttm_dir: str | None
                                ):
    """ 
    Convert speaker timestamps to pyannote.core.Timeline object.
    
    Args:
        speaker_timestamps (List[Tuple[float, float]]): 
            Timestamps of each speaker: start time and end time of each speaker.
        uniq_id (str): 
            Unique ID of each speaker.
        audio_rttm_values (Dict[str, str]):
            Dictionary of manifest values.
        all_hypothesis (List[Tuple[str, pyannote.core.Timeline]]):
            List of hypothesis in pyannote.core.Timeline object.
        all_reference (List[Tuple[str, pyannote.core.Timeline]]):
            List of reference in pyannote.core.Timeline object.
        all_uems (List[Tuple[str, pyannote.core.Timeline]]):
            List of uems in pyannote.core.Timeline object.
            
    Returns:
        all_hypothesis (List[Tuple[str, pyannote.core.Timeline]]):
            List of hypothesis in pyannote.core.Timeline object with an added Timeline object.
        all_reference (List[Tuple[str, pyannote.core.Timeline]]):
            List of reference in pyannote.core.Timeline object with an added Timeline object.
        all_uems (List[Tuple[str, pyannote.core.Timeline]]):
            List of uems in pyannote.core.Timeline object with an added Timeline object.
    """
    offset, dur = float(audio_rttm_values.get('offset', None)), float(audio_rttm_values.get('duration', None))
    hyp_labels = generate_diarization_output_lines(speaker_timestamps=speaker_timestamps, model_spk_num=len(speaker_timestamps))
    hypothesis = labels_to_pyannote_object(hyp_labels, uniq_name=uniq_id)
    if out_rttm_dir is not None and os.path.exists(out_rttm_dir):
        with open(f'{out_rttm_dir}/{uniq_id}.rttm','w') as f:
            hypothesis.write_rttm(f)
    all_hypothesis.append([uniq_id, hypothesis])
    rttm_file = audio_rttm_values.get('rttm_filepath', None)
    if rttm_file is not None and os.path.exists(rttm_file):
        uem_lines = [[offset, dur+offset]] 
        org_ref_labels = rttm_to_labels(rttm_file)
        ref_labels = org_ref_labels
        reference = labels_to_pyannote_object(ref_labels, uniq_name=uniq_id)
        uem_obj = get_uem_object(uem_lines, uniq_id=uniq_id)
        all_uems.append(uem_obj)
        all_reference.append([uniq_id, reference])
    return all_hypothesis, all_reference, all_uems

def ts_vad_post_processing(
    ts_vad_binary_vec: torch.Tensor, 
    cfg_vad_params: OmegaConf, 
    unit_10ms_frame_count: int=8, 
    bypass_postprocessing: bool = False
    ):
    """
    Post-processing on diarization results using VAD style post-processing methods.

    Args:
        ts_vad_binary_vec (Tensor): 
            Sigmoid values of each frame and each speaker.
            Dimension: (num_frames,)
        cfg_vad_params (OmegaConf): 
            Configuration (omega config) of VAD parameters.
        unit_10ms_frame_count (int, optional): 
            an integer indicating the number of 10ms frames in a unit.
            For example, if unit_10ms_frame_count is 8, then each frame is 0.08 seconds.
        bypass_postprocessing (bool, optional): 
            If True, diarization post-processing will be bypassed.

    Returns:
        speech_segments (Tensor): 
            start and end of each speech segment.
            Dimension: (num_segments, 2)
            
            Example: 
                tensor([[  0.0000,   3.0400],                                                                                                                                                                                              [105/1803]
                        [  6.0000,   6.0800],
                        ...
                        [587.3600, 591.0400],
                        [591.1200, 597.7600]])
    """
    ts_vad_binary_frames = torch.repeat_interleave(ts_vad_binary_vec, unit_10ms_frame_count)
    if not bypass_postprocessing:
        speech_segments = binarization(ts_vad_binary_frames, cfg_vad_params)
        speech_segments = filtering(speech_segments, cfg_vad_params)
    else:
        cfg_vad_params.onset=0.5
        cfg_vad_params.offset=0.5
        speech_segments = binarization(ts_vad_binary_frames, cfg_vad_params)
    return speech_segments

def get_uem_object(uem_lines: List[List[float]], uniq_id: str):
    """
    Generate pyannote timeline segments for uem file.
    
     <UEM> file format
     UNIQ_SPEAKER_ID CHANNEL START_TIME END_TIME
     
    Args:
        uem_lines (list): list of session ID and start, end times.
            Example:
            [[0.0, 30.41], [60.04, 165.83]]
        uniq_id (str): Unique session ID.
        
    Returns:
        timeline (pyannote.core.Timeline): pyannote timeline object.
    """
    timeline = Timeline(uri=uniq_id)
    for uem_stt_end in uem_lines:
        start_time, end_time = uem_stt_end 
        timeline.add(Segment(float(start_time), float(end_time)))
    return timeline

def convert_pred_mat_to_segments(
    audio_rttm_map_dict: Dict[str, Dict[str, str]], 
    postprocessing_cfg, 
    batch_preds_list: List[torch.Tensor], 
    unit_10ms_frame_count:int = 8,
    bypass_postprocessing: bool = False,
    out_rttm_dir: str | None = None,
    ):
    """
    Convert prediction matrix to time-stamp segments.

    Args:
        audio_rttm_map_dict (dict): dictionary of audio file path, offset, duration and RTTM filepath.
        batch_preds_list (List[torch.Tensor]): list of prediction matrices containing sigmoid values for each speaker.
            Dimension: [(1, frames, num_speakers), ..., (1, frames, num_speakers)]
        unit_10ms_frame_count (int, optional): number of 10ms segments in a frame. Defaults to 8.
        bypass_postprocessing (bool, optional): if True, postprocessing will be bypassed. Defaults to False.

    Returns:
       all_hypothesis (list): list of pyannote objects for each audio file.
       all_reference (list): list of pyannote objects for each audio file.
       all_uems (list): list of pyannote objects for each audio file.
    """
    batch_pred_ts_segs, all_hypothesis, all_reference, all_uems = [], [], [], []
    cfg_vad_params = OmegaConf.structured(postprocessing_cfg)
    for sample_idx, (uniq_id, audio_rttm_values) in tqdm(enumerate(audio_rttm_map_dict.items()), total=len(audio_rttm_map_dict), desc="Running post-processing"):
        spk_ts = []
        offset, duration = audio_rttm_values['offset'], audio_rttm_values['duration']
        speaker_assign_mat = batch_preds_list[sample_idx].squeeze(dim=0)
        speaker_timestamps = [[] for _ in range(speaker_assign_mat.shape[-1])]
        for spk_id in range(speaker_assign_mat.shape[-1]):
            ts_mat = ts_vad_post_processing(speaker_assign_mat[:, spk_id], 
                                            cfg_vad_params=cfg_vad_params, 
                                            unit_10ms_frame_count=unit_10ms_frame_count, 
                                            bypass_postprocessing=bypass_postprocessing)
            ts_mat = ts_mat + offset
            ts_mat = torch.clamp(ts_mat, min=offset, max=(offset + duration))
            ts_seg_list = ts_mat.tolist()
            speaker_timestamps[spk_id].extend(ts_seg_list)
            spk_ts.append(ts_seg_list)
        all_hypothesis, all_reference, all_uems = timestamps_to_pyannote_object(speaker_timestamps, 
                                                                                uniq_id, 
                                                                                audio_rttm_values, 
                                                                                all_hypothesis, 
                                                                                all_reference, 
                                                                                all_uems,
                                                                                out_rttm_dir,
                                                                               )
        batch_pred_ts_segs.append(spk_ts) 
    return all_hypothesis, all_reference, all_uems

@hydra_runner(config_name="DiarizationConfig", schema=DiarizationConfig)
def main(cfg: DiarizationConfig) -> Union[DiarizationConfig]:
    postprocessing_cfg = load_postprocessing_from_yaml(cfg.postprocessing_yaml)

    for key in cfg:
        cfg[key] = None if cfg[key] == 'None' else cfg[key]

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if cfg.random_seed:
        pl.seed_everything(cfg.random_seed)
        
    if cfg.model_path is None and cfg.pretrained_name is None:
        raise ValueError("Both cfg.model_path and cfg.pretrained_name cannot be None!")
    if cfg.audio_dir is None and cfg.dataset_manifest is None:
        raise ValueError("Both cfg.audio_dir and cfg.dataset_manifest cannot be None!")

    # setup GPU
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    if cfg.cuda is None:
        if torch.cuda.is_available():
            device = [0]  # use 0th CUDA device
            accelerator = 'gpu'
            map_location = torch.device('cuda:0')
        elif cfg.allow_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = [0]
            accelerator = 'mps'
            map_location = torch.device('mps')
        else:
            device = 1
            accelerator = 'cpu'
            map_location = torch.device('cpu')
    else:
        device = [cfg.cuda]
        accelerator = 'gpu'
        map_location = torch.device(f'cuda:{cfg.cuda}')

    if cfg.model_path.endswith(".ckpt"):
        diar_model = SpkDiarEncLabelModel.load_from_checkpoint(checkpoint_path=cfg.model_path, map_location=map_location, strict=False)
    elif cfg.model_path.endswith(".nemo"):
        diar_model = SpkDiarEncLabelModel.restore_from(restore_path=cfg.model_path, map_location=map_location)
    else:
        raise ValueError("cfg.model_path must end with.ckpt or.nemo!")
    
    diar_model._cfg.test_ds.session_len_sec = cfg.session_len_sec
    trainer = pl.Trainer(devices=device, accelerator=accelerator)
    diar_model.set_trainer(trainer)
    
    if cfg.eval_mode:
        diar_model = diar_model.eval()
    diar_model._cfg.test_ds.manifest_filepath = cfg.dataset_manifest
    infer_audio_rttm_dict = get_audio_rttm_map(cfg.dataset_manifest)
    diar_model._cfg.test_ds.batch_size = cfg.batch_size
    
    # Model setup for inference 
    diar_model._cfg.test_ds.num_workers = cfg.num_workers
    diar_model.setup_test_data(test_data_config=diar_model._cfg.test_ds)    
    diar_model.streaming_mode = cfg.streaming_mode
    diar_model.sortformer_diarizer.step_len = cfg.step_len
    diar_model.sortformer_diarizer.mem_len = cfg.mem_len
    
    # Save the list of tensors
    tensor_filename = os.path.basename(cfg.dataset_manifest).replace("manifest.", "").replace(".json", "")
    model_base_path = os.path.dirname(cfg.model_path)
    diar_model.test_batch()
    diar_model_preds_total_list = diar_model.preds_total_list

    if cfg.out_rttm_dir is not None and not os.path.exists(cfg.out_rttm_dir):
        os.mkdir(cfg.out_rttm_dir)

    # Evaluation
<<<<<<< HEAD
    # postprocessing_cfg = PostProcessingParams(opt_style='callhome_part1')
    # postprocessing_cfg = cfg.load_postprocessing_from_yaml()

=======
    vad_cfg = VadParams(opt_style=cfg.opt_style)
>>>>>>> 1fc834e44ea6de4174798a2b9bbc18b00578faf7
    if not cfg.no_der:
        all_hyps, all_refs, all_uems = convert_pred_mat_to_segments(infer_audio_rttm_dict,
                                                                    postprocessing_cfg=postprocessing_cfg, 
                                                                    batch_preds_list=diar_model_preds_total_list, 
                                                                    unit_10ms_frame_count=8,
                                                                    bypass_postprocessing=cfg.bypass_postprocessing,
                                                                    out_rttm_dir=cfg.out_rttm_dir
                                                                   )
        logging.info(f"Evaluating the model on the {len(diar_model_preds_total_list)} audio segments...")
        metric, mapping_dict, itemized_errors = score_labels(AUDIO_RTTM_MAP=infer_audio_rttm_dict, 
                                                            all_reference=all_refs, 
                                                            all_hypothesis=all_hyps, 
                                                            all_uem=all_uems, 
                                                            collar=cfg.collar, 
                                                            ignore_overlap=cfg.ignore_overlap
                                                            )
        print("PostProcessingParams:", postprocessing_cfg)

if __name__ == '__main__':
    main()
