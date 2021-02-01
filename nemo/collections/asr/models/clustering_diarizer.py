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
from os import path
from typing import List, Optional

import librosa
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm

from nemo.collections.asr.models.classification_models import EncDecClassificationModel
from nemo.collections.asr.models.label_models import ExtractSpeakerEmbeddingsModel
from nemo.collections.asr.parts.mixins import DiarizationMixin
from nemo.collections.asr.parts.speaker_utils import get_score
from nemo.collections.asr.parts.vad_utils import (
    generate_overlap_vad_seq,
    generate_vad_segment_table,
    get_vad_stream_status,
    prepare_manifest,
    write_vad_pred_to_manifest,
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


class ClusteringDiarizer(Model, DiarizationMixin):
    def __init__(self, cfg: DictConfig):
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        # Convert config to support Hydra 1.0+ instantiation
        cfg = model_utils.maybe_update_config_version(cfg)
        self._cfg = cfg

        # init vad model
        self.has_vad_model = False
        self.has_vad_model_to_save = False
        self._oracle_vad = ""
        if 'vad' in self._cfg.diarizer:
            self._cfg.diarizer.vad.model_path = self._cfg.diarizer.vad.model_path
            self._init_vad_model()

        # else:
        #     with open_dict(self._cfg):
        #         self._cfg.diarizer.vad.model_path = None
        #     self._vad_window_length_in_sec = 0
        #     self._vad_shift_length_in_sec = 0

        # to delete?

        # init speaker model
        self._speaker_model = ExtractSpeakerEmbeddingsModel.restore_from(
            self._cfg.diarizer.speaker_embeddings.model_path
        )

        # Clustering method
        self._clustering_method = self._cfg.diarizer.cluster_method
        self._num_speakers = self._cfg.diarizer.num_speakers

        self._out_dir = self._cfg.diarizer.out_dir
        self._vad_dir = os.path.join(self._out_dir, 'vad_outputs')
        self._speaker_dir = os.path.join(self._out_dir, 'speaker_outputs')
        # self._vad_in_file = os.path.join(self._vad_dir, "vad_in.json")
        self._vad_out_file = os.path.join(self._vad_dir, "vad_out.json")
        # remove any existing files
        shutil.rmtree(self._vad_dir, ignore_errors=True)
        os.makedirs(self._vad_dir)

        self._manifest_file = self._cfg.manifest_filepath
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @classmethod
    def list_available_models(cls):
        pass

    def set_vad_model(self, vad_config):
        with open_dict(self._cfg):
            self._cfg.diarizer.vad = vad_config
        self._init_vad_model()
        #     .model_path = vad_config['model_path']
        # self._cfg.diarizer.vad.window_length_in_sec = vad_config['window_length_in_sec']
        # self._cfg.diarizer.vad.shift_length_in_sec = vad_config['shift_length_in_sec']

    def _init_vad_model(self):
        if self._cfg.diarizer.vad.model_path.endswith('.json'):
            self._oracle_vad = self._cfg.diarizer.vad.model_path
        elif self._cfg.diarizer.vad.model_path.endswith('.nemo'):
            self._vad_model = EncDecClassificationModel.restore_from(self._cfg.diarizer.vad.model_path)
            self._vad_window_length_in_sec = self._cfg.diarizer.vad.window_length_in_sec
            self._vad_shift_length_in_sec = self._cfg.diarizer.vad.shift_length_in_sec
            self.has_vad_model_to_save = True
            self.has_vad_model = True
        elif self._cfg.diarizer.vad.model_path.endswith('.ckpt'):
            self._vad_model = EncDecClassificationModel.load_from_checkpoint(self._cfg.diarizer.vad.model_path)
            self.has_vad_model = True
        else:
            raise ValueError("vad.model_path should be a .json file or .nemo or a .ckpt model file")

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
            'batch_size': 1,
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

        if self._cfg.diarizer.vad.vad_decision_smoothing:
            #  [TODO] discribe overlap subsequences.
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
        else:
            # [TODO] explain how we shift window to generate frame level prediction
            self.vad_pred_dir = self._vad_dir

            # [TODO] move preds in  vad_dir to be in a frame folder, so we could have frame and overlapped folder
        logging.info("Converting frame level prediction to speech/no-speech segment in start and end times format.")
        table_out_dir = generate_vad_segment_table(
            vad_pred_dir=self.vad_pred_dir,
            threshold=self._cfg.diarizer.vad.threshold,
            shift_len=self._vad_shift_length_in_sec,
            num_workers=self._cfg.num_workers,
        )

        # TODO confirm directory structure here! look above TODO
        # [TODO] change here. Have a map to store audio filepath for speaker model
        self._audio_dir = "/home/fjia/data/modified_callhome/callhome_16k"
        write_vad_pred_to_manifest(table_out_dir, self._audio_dir, self._vad_out_file)

    def _extract_embeddings(self, manifest_file):
        logging.info("Extracting embeddings for Diarization")
        self._setup_spkr_test_data(manifest_file)
        uniq_names = []
        out_embeddings = defaultdict(list)
        self._speaker_model = self._speaker_model.to(self._device)
        self._speaker_model.eval()
        with open(manifest_file, 'r') as manifest:
            for line in manifest.readlines():
                line = line.strip()
                dic = json.loads(line)
                uniq_names.append(dic['audio_filepath'].split('/')[-1].split('.')[0])

        for i, test_batch in enumerate(tqdm(self._speaker_model.test_dataloader())):
            test_batch = [x.to(self._device) for x in test_batch]
            audio_signal, audio_signal_len, labels, slices = test_batch
            with autocast():
                _, embs = self._speaker_model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
                emb_shape = embs.shape[-1]
                embs = embs.type(torch.float32)
                embs = embs.view(-1, emb_shape).cpu().detach().numpy()
                out_embeddings[uniq_names[i]].extend(embs)
            del test_batch

        embedding_dir = os.path.join(self._speaker_dir, 'embeddings')
        if not os.path.exists(embedding_dir):
            os.makedirs(embedding_dir, exist_ok=True)

        prefix = manifest_file.split('/')[-1].split('.')[-2]

        name = os.path.join(embedding_dir, prefix)
        self._embeddings_file = name + '_embeddings.pkl'
        pkl.dump(out_embeddings, open(self._embeddings_file, 'wb'))
        logging.info("Saved embedding files to {}".format(embedding_dir))

    def diarize(self, paths2audio_files: List[str] = None, batch_size: int = 1):
        """
        """
        if 'vad' not in self._cfg.diarizer:
            raise RuntimeError(
                f"Diarization requires a .json file with speech segments or a .nemo file for a " f"VAD model "
            )

        if (paths2audio_files is None or len(paths2audio_files) == 0) and self._manifest_file is None:
            return {}

        if not os.path.exists(self._out_dir):
            os.mkdir(self._out_dir)

        # setup_test_data
        # Work in tmp directory - will store manifest file there
        # [TODO] do we need this?
        if paths2audio_files is not None:
            mfst_file = os.path.join(self._out_dir, 'manifest.json')
            with open(mfst_file, 'w') as fp:
                for audio_file in paths2audio_files:
                    y, sr = librosa.load(audio_file)
                    dur = librosa.get_duration(y=y, sr=sr)
                    del y, sr
                    entry = {'audio_filepath': audio_file, 'offset': 0.0, 'duration': dur, 'text': '-'}
                    fp.write(json.dumps(entry) + '\n')
                    # todo make sure same folder
        else:
            mfst_file = self._manifest_file

        config = {'paths2audio_files': paths2audio_files, 'batch_size': batch_size, 'manifest': mfst_file}

        if self.has_vad_model:
            logging.info("Performing VAD")
            self._dont_auto_split = False
            # [TODO] How can user control this? Should this go in the config?
            self._split_duration = 50
            # Prepare manifest for streaming VAD

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
            if os.path.exists(self._oracle_vad):
                shutil.copy2(self._oracle_vad, self._vad_out_file)
            else:
                raise NotFoundError("Oracle VAD based manifest file not found")

        self._extract_embeddings(self._vad_out_file)
        rttm_dir = self._cfg.diarizer.groundtruth_rttm_dir
        out_rttm_dir = os.path.join(self._out_dir, 'pred_rttms')
        os.makedirs(out_rttm_dir, exist_ok=True)

        DER, CER, FA, MISS = get_score(
            embeddings_file=self._embeddings_file,
            reco2num=self._num_speakers,
            manifest_path=self._vad_out_file,
            sample_rate=self._cfg.sample_rate,
            window=self._cfg.diarizer.speaker_embeddings.window_length_in_sec,
            shift=self._cfg.diarizer.speaker_embeddings.shift_length_in_sec,
            gt_rttm_dir=rttm_dir,
            out_rttm_dir=out_rttm_dir,
        )

        logging.info(
            "Cumulative results of all the files:  FA: {:.3f}, MISS {:.3f} \n \
             Diarization ER: {:.3f}, Cofusion ER:{:.3f}".format(
                FA, MISS, DER, CER
            )
        )

    @staticmethod
    def __make_nemo_file_from_folder(filename, source_dir):
        with tarfile.open(filename, "w:gz") as tar:
            # tar.add(source_dir, arcname=path.basename(source_dir))
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

        # if not self.has_vad_model:
        #     NotImplementedError(
        #         "Saving a clustering based speaker diarization model without a VAD model is not" "supported"
        #     )

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
        if not path.exists(path2file):
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
                # instance = cls.from_config_dict(config=conf)
                instance = cls(cfg=conf)

                logging.info(f'Model {cls.__name__} was successfully restored from {restore_path}.')
            finally:
                os.chdir(cwd)

        return instance
