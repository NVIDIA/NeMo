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
from collections import defaultdict
from itertools import repeat
from multiprocessing import Pool
from typing import Dict, List, Optional, Union
import numpy as np

import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from nemo.collections.asr.models.classification_models import EncDecClassificationModel
from nemo.collections.asr.models.diarization_model import DiarizationModel
from nemo.collections.asr.models.label_models import ExtractSpeakerEmbeddingsModel
from nemo.collections.asr.parts.speaker_utils import get_score
from nemo.collections.asr.parts.vad_utils import gen_overlap_seq, gen_seg_table, write_manifest
from nemo.utils import logging
from nemo.utils.exp_manager import NotFoundError

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


__all__ = ['ClusteringSDModel']


class ClusteringSDModel(DiarizationModel):
    """Base class for encoder decoder CTC-based models."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg=cfg)
        # init vad model
        if not cfg.speaker_embeddings.oracle_vad.ignore_vad:
            self._vad_model = EncDecClassificationModel.restore_from(cfg.vad.model_path)
        # restore_from(self._cfg.vad.model_path)
        self._vad_time_length = self._cfg.vad.time_length
        self._vad_shift_length = self._cfg.vad.shift_length

        # init speaker model
        self._speaker_model = ExtractSpeakerEmbeddingsModel.restore_from(self._cfg.speaker_embeddings.model_path)

        # Clustering method
        self._clustering_method = self._cfg.diarizer.cluster_method
        self._reco2num = self._cfg.diarizer.reco2num

        self._out_dir = self._cfg.diarizer.out_dir
        self._vad_dir = os.path.join(self._out_dir, 'vad_outputs')
        self._speaker_dir = os.path.join(self._out_dir, 'speaker_outputs')
        self._vad_out_file = os.path.join(self._vad_dir, "vad_out.json")

        self._manifest_file = self._cfg.manifest_filepath
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @classmethod
    def list_available_models(cls):
        pass

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        pass

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        pass

    def _setup_vad_test_data(self, config):
        vad_dl_config = {
            'manifest_filepath': config['manifest'],
            'sample_rate': self._cfg.sample_rate,
            'batch_size': 1,
            'vad_stream': True,
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
        }
        self._speaker_model.setup_test_data(spk_dl_config)

    def _run_vad(self, manifest_file):
        shutil.rmtree(self._vad_dir, ignore_errors=True)
        os.mkdir(self._vad_dir)
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
        for i, test_batch in enumerate(self._vad_model.test_dataloader()):
            if i == 0:
                status = 'start' if data[i] == data[i + 1] else 'single'
            elif i == len(data) - 1:
                status = 'end' if data[i] == data[i - 1] else 'single'
            else:
                if data[i] != data[i - 1] and data[i] == data[i + 1]:
                    status = 'start'
                elif data[i] == data[i - 1] and data[i] == data[i + 1]:
                    status = 'next'
                elif data[i] == data[i - 1] and data[i] != data[i + 1]:
                    status = 'end'
                else:
                    status = 'single'
            print(data[i], status)

            test_batch = [x.to(self._device) for x in test_batch]
            with autocast():
                log_probs = self._vad_model(input_signal=test_batch[0], input_signal_length=test_batch[1])
                probs = torch.softmax(log_probs, dim=-1)
                pred = probs[:, 1]

                if status == 'start':
                    to_save = pred[:-trunc]
                elif status == 'next':
                    to_save = pred[trunc:-trunc_l]
                elif status == 'end':
                    to_save = pred[trunc_l:]
                else:
                    to_save = pred
                all_len += len(to_save)

                outpath = os.path.join(self._vad_dir, data[i] + ".frame")
                with open(outpath, "a") as fout:
                    for f in range(len(to_save)):
                        fout.write('{0:0.4f}\n'.format(to_save[f]))

            del test_batch
            if status == 'end' or status == 'single':
                print(f"Overall length of prediction of {data[i]} is {all_len}!")
                all_len = 0

        vad_out_dir = self.generate_vad_timestamps()  # TODO confirm directory structure here
        write_manifest(vad_out_dir, self._vad_dir, self._vad_out_file)

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
                frame_filepath = self._out_dir
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
                    entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': '-'}
                    fp.write(json.dumps(entry) + '\n')
        else:
            mfst_file = self._manifest_file

        config = {'paths2audio_files': paths2audio_files, 'batch_size': batch_size, 'manifest': mfst_file}

        if not self._cfg.speaker_embeddings.oracle_vad.ignore_vad:
            logging.info("Performing VAD")
            self._setup_vad_test_data(config)
            self._run_vad(mfst_file)
            manifest = self._vad_out_file

        else:
            logging.info("Provided option as orcale vad, checking for vad output based manifest file")
            if os.path.exists(self._cfg.speaker_embeddings.oracle_vad.manifest_filepath):
                manifest = self._cfg.speaker_embeddings.oracle_vad.manifest_filepath
            else:
                raise NotFoundError("Oracle VAD based manifest file not found")

        self._extract_embeddings(manifest)
        reco2num = self._reco2num
        RTTM_DIR = self._cfg.diarizer.groundtruth_RTTM_dir
        OUT_RTTM_DIR = os.path.join(self._out_dir, 'pred_rttms')
        os.makedirs(OUT_RTTM_DIR, exist_ok=True)
        DER, CER = get_score(
            embeddings_file=self._embeddings_file,
            reco2num=reco2num,
            manifest_path=manifest,
            SAMPLE_RATE=self._cfg.sample_rate,
            WINDOW=self._cfg.speaker_embeddings.time_length,
            SHIFT=self._cfg.speaker_embeddings.shift_length,
            GT_RTTM_DIR=RTTM_DIR,
            OUT_RTTM_DIR=OUT_RTTM_DIR,
        )

        logging.info(
            "Cumulative Diarization ER and Cofusion ER of all the files is {:.3f} and {:.3f}".format(DER, CER)
        )
