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

from nemo.core.neural_factory import DeploymentFormat, OperationMode
from nemo.core.neural_graph import NeuralGraph
from nemo.core.neural_modules import ModuleType, NeuralModule
from nemo.utils import logging

__all__ = ['NeMoModel']

NEMO_TMP_FOLDER = ".nemo_tmp"


class NeMoModel(NeuralModule):
    """Abstract class representing NeMoModel.
    A NeMoModel is a kind of neural module which contains other neural modules and logic inside.
    It typically represents a whole neural network and requires only connections with data layer and loss
    modules for training.

    The same NeMoModel could be used in training, evaluation or inference regimes. It should adjust itself
    accordingly.
    """

    def __call__(self, **kwargs):
        if self._operation_mode == OperationMode.training or self.operation_mode == OperationMode.both:
            return self.train_graph(**kwargs)

        else:
            return self.eval_graph(**kwargs)

    @classmethod
    @abstractmethod
    def from_pretrained(cls, model_info, local_rank: int = 0, referesh_cache: bool = False) -> NeuralModule:
        """
        Instantiates NeMoModel from pretrained checkpoint. Can do so from file on disk or from the NVIDIA NGC.
        Args:
            model_info: Either path to ".nemo" file or a valid NGC Model name
            local_rank: on which GPU to instantiate.
            referesh_cache: If set to True, then when fetching from clould, this will re-fetch the file
                from clould even if it is  already found in a cache locally.

        Returns:
            NeMoModel instance
        Raises:
            NotImplemened exception when there is no pre-trained models on the cloud
        """
        if isinstance(model_info, str) and model_info.endswith(".nemo"):
            nemo_file_folder, to_delete = cls.__unpack_nemo_file(path2file=model_info)
            configuration_file = path.join(nemo_file_folder, 'module.yaml')
            instance = cls.import_from_config(config_file=configuration_file)
            for module in instance.modules:
                module_checkpoint = path.join(nemo_file_folder, module.__class__.__name__ + ".pt")
                module.restore_from(path=module_checkpoint, local_rank=local_rank)
            shutil.rmtree(to_delete)
            return instance
        else:
            raise NotImplemented("Generic from_pretrained from cloud is not implemented")

    def save_to(self, output_file_name: str, output_folder: str = None, optimize_for_deployment: bool = False) -> str:
        """
        Saves NeMoModel to .nemo file. This file will contain:
            * weights of all NeuralModule instances inside the model
            * Yaml file with configuration
            * Yaml files with topologies and configuration of training and (if applicable) eval graphs
        Args:
            output_file_name: filename, something like nemomodel.nemo
            output_folder: folder where to save output_file_name. If None (default) current folder will be used.
            optimize_for_deployment: will optimize for deployment by trying to export modules to .onnx format and
                skipping training graph.

        Returns:
            None
        """

        def __make_nemo_file_from_folder(filename, source_dir):
            with tarfile.open(filename, "w:gz") as tar:
                tar.add(source_dir, arcname=os.path.basename(source_dir))

        if output_folder is None:
            output_folder = ""
        # create temporary folder first
        rnd_string = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(16))
        rnd_path = path.join(output_folder, f".{rnd_string}")
        tmp_folder = path.join(rnd_path, NEMO_TMP_FOLDER)
        # resulting_file is path to the resulting .nemo file
        if output_file_name.endswith(".nemo"):
            resulting_file = path.join(output_folder, output_file_name)
        else:
            resulting_file = path.join(output_folder, output_file_name + ".nemo")
        if not path.exists(tmp_folder):
            os.makedirs(tmp_folder)
        try:
            # create header file
            main_configuration_file_name = "module.yaml"
            train_graph_file_name = "train_graph.yaml"
            eval_graph_file_name = "eval_graph.yaml"

            header_file_path = path.join(tmp_folder, 'header.content')
            with open(header_file_path, 'w') as hf:
                hf.write(f"Main module configuration: {main_configuration_file_name}")
                hf.write(f"Train graph: {train_graph_file_name}")
                if self.eval_graph is not None:
                    hf.write(f"Eval graph: {train_graph_file_name}")

            # Every NeMo model is a NeuralModule, exporiting its hyperparamers to .yaml
            config_file_path = path.join(tmp_folder, main_configuration_file_name)
            self.export_to_config(config_file=config_file_path)

            if self.train_graph is not None and not optimize_for_deployment:
                config_file_path_train_graph = path.join(tmp_folder, train_graph_file_name)
                self.train_graph.export_to_config(config_file_path_train_graph)

            if self.eval_graph is not None:
                config_file_path_eval_graph = path.join(tmp_folder, eval_graph_file_name)
                self.eval_graph.export_to_config(config_file_path_eval_graph)

            for module in self.modules:
                module_name = module.__class__.__name__
                if optimize_for_deployment:
                    try:
                        module_checkpoint = module_name + ".onnx"
                        module._factory.deployment_export(
                            module=module,
                            output=path.join(tmp_folder, module_checkpoint),
                            d_format=DeploymentFormat.TRTONNX,
                        )
                    except Exception as ex:
                        print(ex)
                        logging.warning(f"Did not convert {module_name} to .onnx")
                        module_checkpoint = module_name + ".pt"
                        module.save_to(path.join(tmp_folder, module_checkpoint))
                else:
                    module_checkpoint = module_name + ".pt"
                    module.save_to(path.join(tmp_folder, module_checkpoint))

            __make_nemo_file_from_folder(resulting_file, tmp_folder)
            logging.info(f"Exported model {self} to {resulting_file}")
        except:
            logging.error("Could not perform NeMoModel export")
        finally:
            shutil.rmtree(rnd_path)
            pass

    @property
    @abstractmethod
    def modules(self) -> Iterable[NeuralModule]:
        pass

    @property
    @abstractmethod
    def train_graph(self) -> NeuralGraph:
        pass

    @property
    @abstractmethod
    def eval_graph(self) -> NeuralGraph:
        pass

    def train(self):
        """
        Sets model to the training mode

        Returns:
            None
        """
        self._operation_mode = OperationMode.training
        for module in self.modules:
            module.operation_mode = OperationMode.training
            if module.type == ModuleType.trainable and hasattr(module, 'train'):
                module.train()

    def eval(self):
        """
        Sets model to the evaluation mode

        Returns:
            None
        """
        self._operation_mode = OperationMode.evaluation
        for module in self.modules:
            module.operation_mode = OperationMode.evaluation
            if module.type == ModuleType.trainable and hasattr(module, 'eval'):
                module.eval()

    @staticmethod
    def __unpack_nemo_file(path2file: str, out_folder: str = None) -> str:
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

    def get_weights(self):
        raise NotImplemented()

    def set_weights(
        self, name2weight, name2name_and_transform,
    ):
        raise NotImplemented()

    def tie_weights_with(
        self, module, weight_names, name2name_and_transform,
    ):
        raise NotImplemented()

    def restore_from(self, path: str):
        raise NotImplemented("Please use from_pretrained method for NeMoModels")

    def freeze(self, weights):
        raise NotImplemented

    def unfreeze(self, weights):
        raise NotImplemented
