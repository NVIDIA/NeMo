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

import glob
import json
import os
import pickle as pkl
import shutil
import tarfile
import tempfile
from collections import defaultdict
from itertools import repeat
from multiprocessing import Pool
from typing import Dict, List, Optional, Union

import numpy as np
from os import path
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm
from nemo.core.classes import Model
from nemo.utils import config_utils, logging, model_utils
from nemo.collections.asr.parts.mixins import DiarizationMixin
from nemo.collections.asr.models.classification_models import EncDecClassificationModel
from nemo.collections.asr.models.label_models import ExtractSpeakerEmbeddingsModel
from nemo.collections.asr.parts.speaker_utils import get_score
from nemo.collections.asr.parts.vad_utils import gen_overlap_seq, gen_seg_table, get_status, write_manifest
from nemo.utils import logging
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
_VAD_MODEL = "vad_model_.nemo"
_SPEAKER_MODEL = "speaker_model.nemo"


class ClusteringDiarizer(Model, DiarizationMixin):
    def __init__(self, cfg: DictConfig):
        cfg = model_utils.convert_model_config_to_dict_config(cfg)

        # Convert config to support Hydra 1.0+ instantiation
        cfg = model_utils.maybe_update_config_version(cfg)
        self._cfg = cfg
        # init vad model
        self.has_vad_model = False
        self._oracle_vad = ""
        if self._cfg.vad.model_path.endswith('.json'):
            self._oracle_vad = cfg.vad.model_path
        elif self._cfg.vad.model_path.endswith('.nemo'):
            self._vad_model = EncDecClassificationModel.restore_from(cfg.vad.model_path)
            self.has_vad_model = True
        elif cfg.vad.model_path.endswith('.ckpt'):
            self._vad_model  = EncDecClassificationModel.load_from_checkpoint(cfg.vad.model_path)
            self.has_vad_model = True
        self._vad_time_length = self._cfg.vad.time_length
        self._vad_shift_length = self._cfg.vad.shift_length
        #self._vad_split_duration = self._cfg.vad.split_duration

        # init speaker model
        self._speaker_model = ExtractSpeakerEmbeddingsModel.restore_from(self._cfg.speaker_embeddings.model_path)

        # Clustering method
        self._clustering_method = self._cfg.diarizer.cluster_method
        self._reco2num = self._cfg.diarizer.reco2num

        self._out_dir = self._cfg.diarizer.out_dir
        self._vad_dir = os.path.join(self._out_dir, 'vad_outputs')
        self._speaker_dir = os.path.join(self._out_dir, 'speaker_outputs')
        self._vad_in_file = os.path.join(self._vad_dir, "vad_in.json")
        self._vad_out_file = os.path.join(self._vad_dir, "vad_out.json")
        # remove any existing files
        shutil.rmtree(self._vad_dir, ignore_errors=True)
        os.makedirs(self._vad_dir)

        self._manifest_file = self._cfg.manifest_filepath
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @classmethod
    def list_available_models(cls):
        pass


    def _setup_vad_test_data(self, config):
        vad_dl_config = {
            'num_workers': self._cfg.vad.num_workers,
            'manifest_filepath': config['manifest'],
            'manifest_vad_input': self._vad_in_file,
            'split_duration': self._cfg.vad.split_duration,
            'sample_rate': self._cfg.sample_rate,
            'vad_stream': True,
            'split_duration': self._cfg.vad.split_duration,
            'labels': ['infer',],
            'time_length': self._cfg.vad.time_length,
            'shift_length': self._cfg.vad.shift_length,
            'trim_silence': False,
        }

        self._vad_model.setup_test_data(test_data_config=vad_dl_config)

    def _setup_spkr_test_data(self, manifest_file):

        spk_dl_config = {
            'manifest_filepath': manifest_file,
            'sample_rate': self._cfg.sample_rate,
            'batch_size': 1,
            'time_length': self._cfg.speaker_embeddings.time_length,
            'shift_length': self._cfg.speaker_embeddings.shift_length,
            'trim_silence': False,
            'embedding_dir': self._speaker_dir,
            'labels': None,
            'task': "diarization",
            'num_workers': 10
        }
        self._speaker_model.setup_test_data(spk_dl_config)

    def _run_vad(self, manifest_file):
        self._vad_model = self._vad_model.to(self._device)
        self._vad_model.eval()

        time_unit = int(self._cfg.vad.time_length / self._cfg.vad.shift_length)
        trunc = int(time_unit / 2)
        trunc_l = time_unit - trunc
        all_len = 0
        data = []
        for line in open(manifest_file, 'r'):
            file = os.path.basename(json.loads(line)['audio_filepath'])
            data.append(os.path.splitext(file)[0])

        status = get_status(data)
        for i, test_batch in enumerate(self._vad_model.test_dataloader()):
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

        vad_out_dir = self.generate_vad_timestamps()  # TODO confirm directory structure here
        self._audio_dir = '/home/fjia/data/modified_callhome/callhome_16k/'
        write_manifest(vad_out_dir, self._audio_dir, self._vad_out_file)

    def generate_vad_timestamps(self):
        if self._cfg.vad.gen_overlap_seq:
            p = Pool(processes=self._cfg.vad.num_workers)
            logging.info("Generating predictions with overlapping input segments")
            frame_filepathlist = glob.glob(self._vad_dir + "/*.frame")
            overlap_out_dir = (
                self._vad_dir
                + "/overlap_smoothing_output"
                + "_"
                + self._cfg.vad.overlap_method
                + "_"
                + str(self._cfg.vad.overlap)
            )
            if not os.path.exists(overlap_out_dir):
                os.mkdir(overlap_out_dir)

            per_args = {
                "method": self._cfg.vad.overlap_method,
                "overlap": self._cfg.vad.overlap,
                "seg_len": self._cfg.vad.seg_len,
                "shift_len": self._cfg.vad.shift_len,
                "out_dir": overlap_out_dir,
            }

            p.starmap(gen_overlap_seq, zip(frame_filepathlist, repeat(per_args)))
            p.close()
            p.join()

        if self._cfg.vad.gen_seg_table:
            p = Pool(processes=self._cfg.vad.num_workers)
            logging.info(
                "Converting frame level prediction to speech/no-speech segment in start and end times format."
            )

            if self._cfg.vad.gen_overlap_seq:
                logging.info("Use overlap prediction. Change if you want to use basic frame level prediction")
                frame_filepath = overlap_out_dir
                shift_len = 0.01
            else:
                logging.info("Use basic frame level prediction")
                frame_filepath = self._vad_dir
                shift_len = self._cfg.vad.shift_len

            frame_filepathlist = glob.glob(frame_filepath + "/*." + self._cfg.vad.overlap_method)

            table_out_dir = os.path.join(self._vad_dir, "table_output_" + str(self._cfg.vad.threshold))
            if not os.path.exists(table_out_dir):
                os.mkdir(table_out_dir)

            per_args = {
                "threshold": self._cfg.vad.threshold,
                "shift_len": shift_len,
                "out_dir": table_out_dir,
            }

            p.starmap(gen_seg_table, zip(frame_filepathlist, repeat(per_args)))
            p.close()
            p.join()
        return table_out_dir

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

    def diarize(self, paths2audio_files: List[str] = None, batch_size: int = 1) -> List[str]:
        """
        """
        if (paths2audio_files is None or len(paths2audio_files) == 0) and self._manifest_file is None:
            return {}

        if not os.path.exists(self._out_dir):
            os.mkdir(self._out_dir)

        # setup_test_data
        # Work in tmp directory - will store manifest file there
        if paths2audio_files is not None:
            mfst_file = os.path.join(self._out_dir, 'manifest.json')
            with open(mfst_file, 'w') as fp:
                for audio_file in paths2audio_files:
                    entry = {'audio_filepath': audio_file, 'offset': 0.0, 'duration': 100000, 'text': '-'}
                    fp.write(json.dumps(entry) + '\n')
                    # todo make sure same folder
        else:
            mfst_file = self._manifest_file

        config = {'paths2audio_files': paths2audio_files, 'batch_size': batch_size, 'manifest': mfst_file}

        if self.has_vad_model:
            logging.info("Performing VAD")
            self._setup_vad_test_data(config)
            self._run_vad(self._vad_in_file)
        else:
            if os.path.exists(self._oracle_vad):
                shutil.copy2(self._oracle_vad, self._vad_out_file)
            else:
                raise NotFoundError("Oracle VAD based manifest file not found")

        self._extract_embeddings(self._vad_out_file)
        reco2num = self._reco2num
        RTTM_DIR = self._cfg.diarizer.groundtruth_RTTM_dir
        OUT_RTTM_DIR = os.path.join(self._out_dir, 'pred_rttms')
        os.makedirs(OUT_RTTM_DIR, exist_ok=True)
        
        DER, CER, FA, MISS = get_score(
            embeddings_file=self._embeddings_file,
            reco2num=reco2num,
            manifest_path=self._vad_out_file,
            SAMPLE_RATE=self._cfg.sample_rate,
            WINDOW=self._cfg.speaker_embeddings.time_length,
            SHIFT=self._cfg.speaker_embeddings.shift_length,
            GT_RTTM_DIR=RTTM_DIR,
            OUT_RTTM_DIR=OUT_RTTM_DIR,
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

        if not self.has_vad_model:
            NotImplementedError(
                "Saving a clustering based speaker diarization model without a VAD model is not" "supported"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_yaml = os.path.join(tmpdir, _MODEL_CONFIG_YAML)
            vad_model = os.path.join(tmpdir, _VAD_MODEL)
            spkr_model = os.path.join(tmpdir, _SPEAKER_MODEL)

            self.to_config_file(path2yaml_file=config_yaml)
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
                conf.vad.model_path = os.path.join(tmpdir, _VAD_MODEL)
                conf.speaker_embeddings.model_path = os.path.join(tmpdir, _SPEAKER_MODEL)
                conf.restore_map_location = map_location
                OmegaConf.set_struct(conf, True)
                # instance = cls.from_config_dict(config=conf)
                instance = cls(cfg=conf)

                logging.info(f'Model {cls.__name__} was successfully restored from {restore_path}.')
            finally:
                os.chdir(cwd)

        return instance
