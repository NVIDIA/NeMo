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

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import os
import requests
import tempfile
from omegaconf import OmegaConf
from lightning_fabric.utilities.exceptions import MisconfigurationException

DEFAULT_DOMAIN_PARAMETERS = {
    "meeting": dict(
        vad=dict(
            shift_length_in_sec=0.01,
            onset=0.8,
            offset=0.5,
            pad_onset=0,
            pad_offset=0,
            min_duration_on=0,
            min_duration_off=0.6,
        ),
        speaker_embeddings=dict(
            window_length_in_sec=(3.0, 2.5, 2.0, 1.5, 1.0, 0.5),
            shift_length_in_sec=(1.5, 1.25, 1.0, 0.75, 0.5, 0.25),
            multiscale_weights=(1, 1, 1, 1, 1, 1),
        ),
        clustering=dict(sparse_search_volume=30),
        msdd_model=dict(),
    ),
    "telephonic": dict(
        vad=dict(
            window_length_in_sec=0.15,
            shift_length_in_sec=0.01,
            smoothing="median",
            overlap=0.5,
            onset=0.1,
            offset=0.1,
            pad_onset=0.1,
            pad_offset=0,
            min_duration_on=0,
            min_duration_off=0.2,
        ),
        speaker_embeddings=dict(
            window_length_in_sec=(1.5, 1.25, 1.0, 0.75, 0.5),
            shift_length_in_sec=(0.75, 0.625, 0.5, 0.375, 0.25),
            multiscale_weights=(1, 1, 1, 1, 1),
        ),
        clustering=dict(sparse_search_volume=30),
        msdd_model=dict(),
    ),
}


@dataclass
class DiarizerComponentConfig:
    """Dataclass to imitate HydraConfig dict when accessing parameters."""

    @property
    def parameters(self):
        return self

    @parameters.setter
    def parameters(self, parameters):
        for k, value in asdict(parameters).items():
            setattr(self, k, value)

    def get(self, name: str, default: Optional[None] = None):
        return getattr(self, name, default)

    def __iter__(self):
        for key in asdict(self):
            yield key

    def dict(self) -> Dict:
        return asdict(self)


@dataclass
class VADConfig(DiarizerComponentConfig):
    model_path: str = "vad_multilingual_marblenet"  # .nemo local model path or pretrained VAD model name
    external_vad_manifest: Optional[str] = None
    window_length_in_sec: float = 0.63  # Window length in sec for VAD context input
    shift_length_in_sec: float = 0.08  # Shift length in sec for generate frame level VAD prediction
    smoothing: Union[str, bool] = False  # False or type of smoothing method (eg: median)
    overlap: float = 0.5  # Overlap ratio for overlapped mean/median smoothing filter
    onset: float = 0.5  # Onset threshold for detecting the beginning and end of a speech
    offset: float = 0.3  # Offset threshold for detecting the end of a speech
    pad_onset: float = 0.2  # Adding durations before each speech segment
    pad_offset: float = 0.2  # Adding durations after each speech segment
    min_duration_on: float = 0.5  # Threshold for small non_speech deletion
    min_duration_off: float = 0.5  # Threshold for short speech segment deletion
    filter_speech_first: bool = True


@dataclass
class SpeakerEmbeddingsConfig(DiarizerComponentConfig):
    # .nemo local model path or pretrained model name (titanet_large, ecapa_tdnn or speakerverification_speakernet)
    model_path: str = "titanet_large"
    # Window length(s) in sec (floating-point number). either a number or a list. ex) 1.5 or [1.5,1.0,0.5]
    window_length_in_sec: Tuple[float] = (1.9, 1.2, 0.5)
    # Shift length(s) in sec (floating-point number). either a number or a list. ex) 0.75 or [0.75,0.5,0.25]
    shift_length_in_sec: Tuple[float] = (0.95, 0.6, 0.25)
    # Weight for each scale. None (for single scale) or list with window/shift scale count. ex) [0.33,0.33,0.33]
    multiscale_weights: Tuple[float] = (1, 1, 1)
    # save speaker embeddings in pickle format. True if clustering result is used for other models, such as MSDD.
    save_embeddings: bool = True


@dataclass
class ClusteringConfig(DiarizerComponentConfig):
    # If True, use num of speakers value provided in manifest file.
    oracle_num_speakers: bool = False
    # Max number of speakers for each recording. If an oracle number of speakers is passed, this value is ignored.
    max_num_speakers: int = 8
    # If the number of segments is lower than this number, enhanced speaker counting is activated.
    enhanced_count_thres: int = 80
    # Determines the range of p-value search: 0 < p <= max_rp_threshold.
    max_rp_threshold: float = 0.25
    # The higher the number, the more values will be examined with more time.
    sparse_search_volume: int = 10
    # If True, take a majority vote on multiple p-values to estimate the number of speakers.
    maj_vote_spk_count: bool = False


@dataclass
class MSDDConfig(DiarizerComponentConfig):
    model_path: str = "diar_msdd_telephonic"
    # If True, use speaker embedding model in checkpoint, else provided speaker embedding model in config will be used.
    use_speaker_model_from_ckpt: bool = True
    # Batch size for MSDD inference.
    infer_batch_size: int = 25
    # Sigmoid threshold for generating binarized speaker labels. The smaller the more generous on detecting overlaps.
    sigmoid_threshold: Tuple[float] = (0.7,)
    # If True, use oracle number of speaker and evaluate F1 score for the given speaker sequences. Default is False.
    seq_eval_mode: bool = False
    # If True, break the input audio clip to short sequences and calculate cluster average embeddings for inference.
    split_infer: bool = True
    # The length of split short sequence when split_infer is True.
    diar_window_length: int = 50
    # If the estimated number of speakers are larger than this number, overlap speech is not estimated.
    overlap_infer_spk_limit: int = 5


@dataclass
class DiarizerConfig(DiarizerComponentConfig):
    manifest_filepath: Optional[str] = None
    out_dir: Optional[str] = None
    oracle_vad: bool = False  # If True, uses RTTM files provided in the manifest file to get VAD timestamps
    collar: float = 0.25  # Collar value for scoring
    ignore_overlap: bool = True  # Consider or ignore overlap segments while scoring
    vad: VADConfig = VADConfig()
    speaker_embeddings: SpeakerEmbeddingsConfig = SpeakerEmbeddingsConfig()
    clustering: ClusteringConfig = ClusteringConfig()
    msdd_model: MSDDConfig = MSDDConfig()


@dataclass
class NeuralInferenceConfig(DiarizerComponentConfig):
    diarizer: DiarizerConfig = DiarizerConfig()
    device: Union[torch.device, str] = 'auto'  # device to run on. "auto" selects CUDA if available, else CPU.
    batch_size: int = 64
    num_workers: int = 1
    sample_rate: int = 16000
    
    @staticmethod 
    def _download_yaml(url: str, temp_dir: str = './'):
        read_handle = requests.get(url)  
        # with open('./temp.yaml', 'wb') as f:
        #     f.write(read_handle.content)
        with tempfile.TemporaryDirectory(dir=temp_dir) as temp_yaml:
            with open(f"{temp_yaml}/temp.yaml", 'wb') as f:
                f.write(read_handle.content)
            yaml_config = OmegaConf.load(f"{temp_yaml}/temp.yaml")
        return yaml_config

    @staticmethod 
    def _get_domain_config(domain: str):
        if domain == "meeting":
            yaml_url = 'https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_meeting.yaml'
        elif domain == "telephonic":
            yaml_url = 'https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml'
        elif domain == "general":
            yaml_url = 'https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_general.yaml'
        else:
            raise MisconfigurationException(
                f"The domain {domain} does not exist. ")
            
        return yaml_url

    @classmethod
    def from_domain(cls, domain: Optional[str], device: Union[torch.device, str]):
        if domain is None:
            return DiarizerConfig()
        else:
            yaml_url = cls._get_domain_config(domain)
            config = cls._download_yaml(yaml_url) 
            config = config.diarizer
            # return config
            return NeuralInferenceConfig(
                DiarizerConfig(
                    vad=VADConfig(**config["vad"]["parameters"]),
                    speaker_embeddings=SpeakerEmbeddingsConfig(**config["speaker_embeddings"]["parameters"]),
                    clustering=ClusteringConfig(**config["clustering"]["parameters"]),
                    msdd_model=MSDDConfig(**config["msdd_model"]["parameters"]),
                ),
                device=device,
        )

    def __post_init__(self):
        if self.device == "auto":
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
