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

import json
import os
import pickle as pkl
import shutil
import tarfile
import tempfile
from collections import defaultdict
from typing import List, Optional

import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from omegaconf.listconfig import ListConfig
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm

from nemo.collections.asr.models.classification_models import EncDecClassificationModel
from nemo.collections.asr.models.label_models import ExtractSpeakerEmbeddingsModel
from nemo.collections.asr.parts.mixins.mixins import DiarizationMixin
from nemo.collections.asr.parts.utils.speaker_utils import (
    audio_rttm_map,
    perform_diarization,
    segments_manifest_to_subsegments_manifest,
    write_rttm2manifest,
)
from nemo.collections.asr.parts.utils.vad_utils import (
    generate_overlap_vad_seq,
    generate_vad_segment_table,
    get_vad_stream_status,
    prepare_manifest,
)
from nemo.core.classes import Model
from nemo.utils import logging, model_utils
from nemo.utils.exp_manager import NotFoundError

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


__all__ = ['ClusteringDiarizer']

_MODEL_CONFIG_YAML = "model_config.yaml"
_VAD_MODEL = "vad_model.nemo"
_SPEAKER_MODEL = "speaker_model.nemo"


def get_available_model_names(class_name):
    available_models = class_name.list_available_models()
    return list(map(lambda x: x.pretrained_model_name, available_models))


class ClusteringDiarizer(Model, DiarizationMixin):
    def __init__(self, cfg: DictConfig):
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        # Convert config to support Hydra 1.0+ instantiation
        cfg = model_utils.maybe_update_config_version(cfg)
        self._cfg = cfg
        self._out_dir = self._cfg.diarizer.out_dir
        if not os.path.exists(self._out_dir):
            os.mkdir(self._out_dir)

        # init vad model
        self.has_vad_model = False
        self.has_vad_model_to_save = False

        self._speaker_manifest_path = self._cfg.diarizer.speaker_embeddings.oracle_vad_manifest
        self.AUDIO_RTTM_MAP = None
        self.paths2audio_files = self._cfg.diarizer.paths2audio_files

        if self._cfg.diarizer.vad.model_path is not None:
            self._init_vad_model()
            self._vad_dir = os.path.join(self._out_dir, 'vad_outputs')
            self._vad_out_file = os.path.join(self._vad_dir, "vad_out.json")
            shutil.rmtree(self._vad_dir, ignore_errors=True)
            os.makedirs(self._vad_dir)

        # init speaker model
        self._init_speaker_model()

        if self._cfg.diarizer.get('num_speakers', None):
            self._num_speakers = self._cfg.diarizer.num_speakers
            logging.warning("in next release num_speakers will be changed to oracle_num_speakers")
        else:
            self._num_speakers = self._cfg.diarizer.oracle_num_speakers

        if self._cfg.diarizer.get('max_num_speakers', None):
            self.max_num_speakers = self._cfg.diarizer.max_num_speakers
        else:
            self.max_num_speakers = 8

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @classmethod
    def list_available_models(cls):
        pass

    def _init_speaker_model(self):
        model_path = self._cfg.diarizer.speaker_embeddings.model_path
        if model_path is not None and model_path.endswith('.nemo'):
            self._speaker_model = ExtractSpeakerEmbeddingsModel.restore_from(model_path)
            logging.info("Speaker Model restored locally from {}".format(model_path))
        else:
            if model_path not in get_available_model_names(ExtractSpeakerEmbeddingsModel):
                logging.warning(
                    "requested {} model name not available in pretrained models, instead".format(model_path)
                )
                model_path = "speakerdiarization_speakernet"
            logging.info("Loading pretrained {} model from NGC".format(model_path))
            self._speaker_model = ExtractSpeakerEmbeddingsModel.from_pretrained(model_name=model_path)

        self._speaker_dir = os.path.join(self._out_dir, 'speaker_outputs')

    def set_vad_model(self, vad_config):
        with open_dict(self._cfg):
            self._cfg.diarizer.vad = vad_config
        self._init_vad_model()

    def _init_vad_model(self):
        model_path = self._cfg.diarizer.vad.model_path
        if model_path.endswith('.nemo'):
            self._vad_model = EncDecClassificationModel.restore_from(model_path)
            logging.info("VAD model loaded locally from {}".format(model_path))
        else:
            if model_path not in get_available_model_names(EncDecClassificationModel):
                logging.warning(
                    "requested {} model name not available in pretrained models, instead".format(model_path)
                )
                model_path = "vad_telephony_marblenet"
            logging.info("Loading pretrained {} model from NGC".format(model_path))
            self._vad_model = EncDecClassificationModel.from_pretrained(model_name=model_path)

        self._vad_window_length_in_sec = self._cfg.diarizer.vad.window_length_in_sec
        self._vad_shift_length_in_sec = self._cfg.diarizer.vad.shift_length_in_sec
        self.has_vad_model_to_save = True
        self.has_vad_model = True

    def _setup_vad_test_data(self, manifest_vad_input):
        vad_dl_config = {
            'manifest_filepath': manifest_vad_input,
            'sample_rate': self._cfg.sample_rate,
            'vad_stream': True,
            'labels': ['infer',],
            'time_length': self._vad_window_length_in_sec,
            'shift_length': self._vad_shift_length_in_sec,
            'trim_silence': False,
            'num_workers': self._cfg.num_workers,
        }
        self._vad_model.setup_test_data(test_data_config=vad_dl_config)

    def _setup_spkr_test_data(self, manifest_file):
        spk_dl_config = {
            'manifest_filepath': manifest_file,
            'sample_rate': self._cfg.sample_rate,
            'batch_size': self._cfg.get('batch_size', 32),
            'time_length': self._cfg.diarizer.speaker_embeddings.window_length_in_sec,
            'shift_length': self._cfg.diarizer.speaker_embeddings.shift_length_in_sec,
            'trim_silence': False,
            'embedding_dir': self._speaker_dir,
            'labels': None,
            'task': "diarization",
            'num_workers': self._cfg.num_workers,
        }
        self._speaker_model.setup_test_data(spk_dl_config)

    def _run_vad(self, manifest_file):
        self._vad_model = self._vad_model.to(self._device)
        self._vad_model.eval()

        time_unit = int(self._vad_window_length_in_sec / self._vad_shift_length_in_sec)
        trunc = int(time_unit / 2)
        trunc_l = time_unit - trunc
        all_len = 0
        data = []
        for line in open(manifest_file, 'r'):
            file = os.path.basename(json.loads(line)['audio_filepath'])
            data.append(os.path.splitext(file)[0])

        status = get_vad_stream_status(data)
        for i, test_batch in enumerate(tqdm(self._vad_model.test_dataloader())):
            test_batch = [x.to(self._device) for x in test_batch]
            with autocast():
                log_probs = self._vad_model(input_signal=test_batch[0], input_signal_length=test_batch[1])
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
                outpath = os.path.join(self._vad_dir, data[i] + ".frame")
                with open(outpath, "a") as fout:
                    for f in range(len(to_save)):
                        fout.write('{0:0.4f}\n'.format(to_save[f]))
            del test_batch
            if status[i] == 'end' or status[i] == 'single':
                all_len = 0

        if not self._cfg.diarizer.vad.vad_decision_smoothing:
            # Shift the window by 10ms to generate the frame and use the prediction of the window to represent the label for the frame;
            self.vad_pred_dir = self._vad_dir

        else:
            # Generate predictions with overlapping input segments. Then a smoothing filter is applied to decide the label for a frame spanned by multiple segments.
            # smoothing_method would be either in majority vote (median) or average (mean)
            logging.info("Generating predictions with overlapping input segments")
            smoothing_pred_dir = generate_overlap_vad_seq(
                frame_pred_dir=self._vad_dir,
                smoothing_method=self._cfg.diarizer.vad.smoothing_params.method,
                overlap=self._cfg.diarizer.vad.smoothing_params.overlap,
                seg_len=self._vad_window_length_in_sec,
                shift_len=self._vad_shift_length_in_sec,
                num_workers=self._cfg.num_workers,
            )
            self.vad_pred_dir = smoothing_pred_dir

        logging.info("Converting frame level prediction to speech/no-speech segment in start and end times format.")
        table_out_dir = generate_vad_segment_table(
            vad_pred_dir=self.vad_pred_dir,
            threshold=self._cfg.diarizer.vad.threshold,
            shift_len=self._vad_shift_length_in_sec,
            num_workers=self._cfg.num_workers,
        )

        vad_table_list = [os.path.join(table_out_dir, key + ".txt") for key in self.AUDIO_RTTM_MAP]
        write_rttm2manifest(self._cfg.diarizer.paths2audio_files, vad_table_list, self._vad_out_file)
        self._speaker_manifest_path = self._vad_out_file

    def _extract_embeddings(self, manifest_file):
        logging.info("Extracting embeddings for Diarization")
        self._setup_spkr_test_data(manifest_file)
        out_embeddings = defaultdict(list)
        self._speaker_model = self._speaker_model.to(self._device)
        self._speaker_model.eval()

        all_embs = []
        for test_batch in tqdm(self._speaker_model.test_dataloader()):
            test_batch = [x.to(self._device) for x in test_batch]
            audio_signal, audio_signal_len, labels, slices = test_batch
            with autocast():
                _, embs = self._speaker_model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
                emb_shape = embs.shape[-1]
                embs = embs.view(-1, emb_shape)
                all_embs.extend(embs.cpu().detach().numpy())
            del test_batch

        with open(manifest_file, 'r') as manifest:
            for i, line in enumerate(manifest.readlines()):
                line = line.strip()
                dic = json.loads(line)
                uniq_name = os.path.basename(dic['audio_filepath']).rsplit('.', 1)[0]
                out_embeddings[uniq_name].extend([all_embs[i]])

        embedding_dir = os.path.join(self._speaker_dir, 'embeddings')
        if not os.path.exists(embedding_dir):
            os.makedirs(embedding_dir, exist_ok=True)

        prefix = manifest_file.split('/')[-1].rsplit('.', 1)[-2]

        name = os.path.join(embedding_dir, prefix)
        self._embeddings_file = name + '_embeddings.pkl'
        pkl.dump(out_embeddings, open(self._embeddings_file, 'wb'))
        logging.info("Saved embedding files to {}".format(embedding_dir))

    def path2audio_files_to_manifest(self, paths2audio_files):
        mfst_file = os.path.join(self._out_dir, 'manifest.json')
        with open(mfst_file, 'w') as fp:
            for audio_file in paths2audio_files:
                audio_file = audio_file.strip()
                entry = {'audio_filepath': audio_file, 'offset': 0.0, 'duration': None, 'text': '-', 'label': 'infer'}
                fp.write(json.dumps(entry) + '\n')

        return mfst_file

    def diarize(self, paths2audio_files: List[str] = None, batch_size: int = 1):
        """
        """

        if paths2audio_files:
            self.paths2audio_files = paths2audio_files
        else:
            if self._cfg.diarizer.paths2audio_files is None:
                raise ValueError("Pass path2audio files either through config or to diarize method")
            else:
                self.paths2audio_files = self._cfg.diarizer.paths2audio_files

        if type(self.paths2audio_files) is str and os.path.isfile(self.paths2audio_files):
            paths2audio_files = []
            with open(self.paths2audio_files, 'r') as path2file:
                for audiofile in path2file.readlines():
                    audiofile = audiofile.strip()
                    paths2audio_files.append(audiofile)

        elif type(self.paths2audio_files) in [list, ListConfig]:
            paths2audio_files = list(self.paths2audio_files)

        else:
            raise ValueError("paths2audio_files must be of type list or path to file containing audio files")

        self.AUDIO_RTTM_MAP = audio_rttm_map(paths2audio_files, self._cfg.diarizer.path2groundtruth_rttm_files)

        if self.has_vad_model:
            logging.info("Performing VAD")
            mfst_file = self.path2audio_files_to_manifest(paths2audio_files)
            self._dont_auto_split = False
            self._split_duration = 50
            manifest_vad_input = mfst_file

            if not self._dont_auto_split:
                logging.info("Split long audio file to avoid CUDA memory issue")
                logging.debug("Try smaller split_duration if you still have CUDA memory issue")
                config = {
                    'manifest_filepath': mfst_file,
                    'time_length': self._vad_window_length_in_sec,
                    'split_duration': self._split_duration,
                    'num_workers': self._cfg.num_workers,
                }
                manifest_vad_input = prepare_manifest(config)
            else:
                logging.warning(
                    "If you encounter CUDA memory issue, try splitting manifest entry by split_duration to avoid it."
                )

            self._setup_vad_test_data(manifest_vad_input)
            self._run_vad(manifest_vad_input)
        else:
            if not os.path.exists(self._speaker_manifest_path):
                raise NotFoundError("Oracle VAD based manifest file not found")

        self.subsegments_manifest_path = os.path.join(self._out_dir, 'subsegments.json')
        self.subsegments_manifest_path = segments_manifest_to_subsegments_manifest(
            segments_manifest_file=self._speaker_manifest_path,
            subsegments_manifest_file=self.subsegments_manifest_path,
            window=self._cfg.diarizer.speaker_embeddings.window_length_in_sec,
            shift=self._cfg.diarizer.speaker_embeddings.shift_length_in_sec,
        )
        self._extract_embeddings(self.subsegments_manifest_path)
        out_rttm_dir = os.path.join(self._out_dir, 'pred_rttms')
        os.makedirs(out_rttm_dir, exist_ok=True)

        perform_diarization(
            embeddings_file=self._embeddings_file,
            reco2num=self._num_speakers,
            manifest_path=self.subsegments_manifest_path,
            audio_rttm_map=self.AUDIO_RTTM_MAP,
            out_rttm_dir=out_rttm_dir,
            max_num_speakers=self.max_num_speakers,
        )

    @staticmethod
    def __make_nemo_file_from_folder(filename, source_dir):
        with tarfile.open(filename, "w:gz") as tar:
            tar.add(source_dir, arcname="./")

    @rank_zero_only
    def save_to(self, save_path: str):
        """
        Saves model instance (weights and configuration) into EFF archive or .
         You can use "restore_from" method to fully restore instance from .nemo file.

        .nemo file is an archive (tar.gz) with the following:
            model_config.yaml - model configuration in .yaml format. You can deserialize this into cfg argument for model's constructor
            model_wights.chpt - model checkpoint

        Args:
            save_path: Path to .nemo file where model instance should be saved
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            config_yaml = os.path.join(tmpdir, _MODEL_CONFIG_YAML)
            spkr_model = os.path.join(tmpdir, _SPEAKER_MODEL)

            self.to_config_file(path2yaml_file=config_yaml)
            if self.has_vad_model:
                vad_model = os.path.join(tmpdir, _VAD_MODEL)
                self._vad_model.save_to(vad_model)
            self._speaker_model.save_to(spkr_model)
            self.__make_nemo_file_from_folder(filename=save_path, source_dir=tmpdir)

    @staticmethod
    def __unpack_nemo_file(path2file: str, out_folder: str) -> str:
        if not os.path.exists(path2file):
            raise FileNotFoundError(f"{path2file} does not exist")
        tar = tarfile.open(path2file, "r:gz")
        tar.extractall(path=out_folder)
        tar.close()
        return out_folder

    @classmethod
    def restore_from(
        cls,
        restore_path: str,
        override_config_path: Optional[str] = None,
        map_location: Optional[torch.device] = None,
        strict: bool = False,
    ):
        # Get path where the command is executed - the artifacts will be "retrieved" there
        # (original .nemo behavior)
        cwd = os.getcwd()

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                cls.__unpack_nemo_file(path2file=restore_path, out_folder=tmpdir)
                os.chdir(tmpdir)
                if override_config_path is None:
                    config_yaml = os.path.join(tmpdir, _MODEL_CONFIG_YAML)
                else:
                    config_yaml = override_config_path
                conf = OmegaConf.load(config_yaml)
                if os.path.exists(os.path.join(tmpdir, _VAD_MODEL)):
                    conf.diarizer.vad.model_path = os.path.join(tmpdir, _VAD_MODEL)
                else:
                    logging.info(
                        f'Model {cls.__name__} does not contain a VAD model. A VAD model or manifest file with'
                        f'speech segments need for diarization with this model'
                    )

                conf.diarizer.speaker_embeddings.model_path = os.path.join(tmpdir, _SPEAKER_MODEL)
                conf.restore_map_location = map_location
                OmegaConf.set_struct(conf, True)
                instance = cls(cfg=conf)

                logging.info(f'Model {cls.__name__} was successfully restored from {restore_path}.')
            finally:
                os.chdir(cwd)

        return instance
