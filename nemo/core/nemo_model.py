# ! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2019-, NVIDIA CORPORATION. All rights reserved.
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

import os
import random
import shutil
import string
import tarfile
from abc import abstractmethod
from os import path
from typing import Iterable

from .neural_modules import NeuralModule
from nemo import logging

__all__ = ['NeMoModel']

NEMO_TMP_FOLDER = ".nemo_tmp"


class NeMoModel(NeuralModule):
    """Abstract class representing NeMoModel.
    A NeMoModel is a kind of neural module which contains other neural modules and logic inside.
    It typically represents a whole neural network and requires only connections with data layer and loss
    modules for training."""

    @classmethod
    @abstractmethod
    def from_pretrained(cls, model_info, local_rank=0) -> NeuralModule:
        if isinstance(model_info, str) and model_info.endswith(".nemo"):
            nemo_file_folder, to_delete = cls._unpack_nemo_file(path2file=model_info)
            # TODO: this is potentially error-prone
            configuration_file = path.join(nemo_file_folder, cls.__name__ + ".yaml")
            instance = cls.import_from_config(config_file=configuration_file)
            for module in instance.modules:
                module_checkpoint = path.join(nemo_file_folder, module.__class__.__name__ + ".pt")
                module.restore_from(path=module_checkpoint, local_rank=local_rank)
            shutil.rmtree(to_delete)
            return instance
        else:
            raise NotImplemented("Generic from_pretrained from cloud is not implemented")

    def export(self, output_file_name: str, output_folder: str = None, deployment: bool = False) -> str:
        def make_nemo_file_from_folder(filename, source_dir):
            with tarfile.open(filename, "w:gz") as tar:
                tar.add(source_dir, arcname=os.path.basename(source_dir))

        if deployment:
            raise NotImplemented("Currently, deployment is working on a per-model basis")
        if output_folder is None:
            output_folder = ""

        rnd_string = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(16))
        rnd_path = path.join(output_folder, f".{rnd_string}")
        tmp_folder = path.join(rnd_path, NEMO_TMP_FOLDER)
        if output_file_name.endswith(".nemo"):
            resulting_file = path.join(output_folder, output_file_name)
        else:
            resulting_file = path.join(output_folder, output_file_name + ".nemo")
        if not path.exists(tmp_folder):
            os.makedirs(tmp_folder)
        try:
            config_file_path = path.join(tmp_folder, self.__class__.__name__ + ".yaml")
            self.export_to_config(config_file=config_file_path)
            for module in self.modules:
                module_checkpoint = module.__class__.__name__ + ".pt"
                module.save_to(path.join(tmp_folder, module_checkpoint))
            make_nemo_file_from_folder(resulting_file, tmp_folder)
            logging.info(f"Exported model {self} to {resulting_file}")
        except:
            logging.error("Could not perform NeMoModel export")
        finally:
            shutil.rmtree(rnd_path)

    @property
    @abstractmethod
    def modules(self) -> Iterable[NeuralModule]:
        pass

    @staticmethod
    def _unpack_nemo_file(path2file: str, out_folder: str = None) -> str:
        if not path.exists(path2file):
            raise FileNotFoundError(f"{path2file} does not exist")
        if out_folder is None:
            out_folder = ''.join(
                random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(16)
            )

        tar = tarfile.open(path2file, "r:gz")
        tar.extractall(path=out_folder)
        tar.close()
        return path.join(out_folder, NEMO_TMP_FOLDER), out_folder






