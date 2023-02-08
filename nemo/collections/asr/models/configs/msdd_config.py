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
    window_length_in_sec: float = 0.15  # Window length in sec for VAD context input
    shift_length_in_sec: float = 0.01  # Shift length in sec for generate frame level VAD prediction
    smoothing: Union[str, bool] = "median"  # False or type of smoothing method (eg: median)
    overlap: float = 0.5  # Overlap ratio for overlapped mean/median smoothing filter
    onset: float = 0.1  # Onset threshold for detecting the beginning and end of a speech
    offset: float = 0.1  # Offset threshold for detecting the end of a speech
    pad_onset: float = 0.1  # Adding durations before each speech segment
    pad_offset: float = 0  # Adding durations after each speech segment
    min_duration_on: float = 0  # Threshold for small non_speech deletion
    min_duration_off: float = 0.2  # Threshold for short speech segment deletion
    filter_speech_first: bool = True


@dataclass
class SpeakerEmbeddingsConfig(DiarizerComponentConfig):
    # .nemo local model path or pretrained model name (titanet_large, ecapa_tdnn or speakerverification_speakernet)
    model_path: Optional[str] = None
    # Window length(s) in sec (floating-point number). either a number or a list. ex) 1.5 or [1.5,1.0,0.5]
    window_length_in_sec: Tuple[float] = (1.5, 1.25, 1.0, 0.75, 0.5)
    # Shift length(s) in sec (floating-point number). either a number or a list. ex) 0.75 or [0.75,0.5,0.25]
    shift_length_in_sec: Tuple[float] = (0.75, 0.625, 0.5, 0.375, 0.25)
    # Weight for each scale. None (for single scale) or list with window/shift scale count. ex) [0.33,0.33,0.33]
    multiscale_weights: Tuple[float] = (1, 1, 1, 1, 1)
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
    sparse_search_volume: int = 30
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
    device: torch.device = torch.device('cpu')
    map_location: Optional[torch.device] = None
    batch_size: int = 64
    num_workers: int = 1
    sample_rate: int = 16000

    @classmethod
    def init_config(cls, diar_model_path: str, vad_model_path: str, map_location: torch.device):
        return NeuralInferenceConfig(
            DiarizerConfig(
                vad=VADConfig(model_path=vad_model_path), msdd_model=MSDDConfig(model_path=diar_model_path),
            ),
            map_location=map_location,
        )
