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
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import nemo
from nemo import logging
from nemo.core import (
    NeMoModel,
    NeuralGraph,
    NeuralModule,
    NeuralType,
    OperationMode,
    PretrainedModelInfo,
    WeightShareTransform,
)
from nemo.utils import maybe_download_from_cloud


class QuartzNet(NeMoModel):
    """
    QuartzNet ASR Model. See: "QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel Separable Convolutions"
    https://arxiv.org/abs/1910.10261
    """

    def __init__(
        self,
        preprocessor_params: Dict,
        encoder_params: Dict,
        decoder_params: Dict,
        spec_augment_params: Optional[Dict] = None,
    ):
        super().__init__()
        # Instantiate necessary modules
        preprocessor, spec_augmentation, encoder, decoder = self.__instantiate_modules(
            preprocessor_params, encoder_params, decoder_params, spec_augment_params
        )
        self._operation_mode = OperationMode.training

        # self.__training_neural_graph = NeuralGraph(operation_mode=OperationMode.training)
        self.__training_neural_graph = NeuralGraph(operation_mode=OperationMode.both)
        with self.__training_neural_graph:
            # Copy one input port definitions - using "user" port names.
            self.__training_neural_graph.inputs["input_signal"] = preprocessor.input_ports["input_signal"]
            self.__training_neural_graph.inputs["length"] = preprocessor.input_ports["length"]
            # Bind the selected inputs. Connect the modules
            i_processed_signal, i_processed_signal_len = preprocessor(
                input_signal=self.__training_neural_graph.inputs["input_signal"],
                length=self.__training_neural_graph.inputs["length"],
            )
            if spec_augmentation is not None:
                i_processed_signal = spec_augmentation(input_spec=i_processed_signal)
            i_encoded, i_encoded_len = encoder(audio_signal=i_processed_signal, length=i_processed_signal_len)
            i_log_probs = decoder(encoder_output=i_encoded)
            # Bind the selected outputs.
            self.__training_neural_graph.outputs["log_probs"] = i_log_probs
            self.__training_neural_graph.outputs["encoded_len"] = i_encoded_len

        self.__evaluation_neural_graph = NeuralGraph(operation_mode=OperationMode.evaluation)
        with self.__evaluation_neural_graph:
            # Copy one input port definitions - using "user" port names.
            self.__evaluation_neural_graph.inputs["input_signal"] = preprocessor.input_ports["input_signal"]
            self.__evaluation_neural_graph.inputs["length"] = preprocessor.input_ports["length"]
            # Bind the selected inputs. Connect the modules
            i_processed_signal, i_processed_signal_len = preprocessor(
                input_signal=self.__evaluation_neural_graph.inputs["input_signal"],
                length=self.__evaluation_neural_graph.inputs["length"],
            )
            # Notice lack of speck augmentation for inference
            i_encoded, i_encoded_len = encoder(audio_signal=i_processed_signal, length=i_processed_signal_len)
            i_log_probs = decoder(encoder_output=i_encoded)
            # Bind the selected outputs.
            self.__evaluation_neural_graph.outputs["log_probs"] = i_log_probs
            self.__evaluation_neural_graph.outputs["encoded_len"] = i_encoded_len

    def __instantiate_modules(
        self, preprocessor_params, encoder_params, decoder_params, spec_augment_params=None,
    ):
        preprocessor = NeuralModule.deserialize(preprocessor_params)
        encoder = NeuralModule.deserialize(encoder_params)
        decoder = NeuralModule.deserialize(decoder_params)
        if spec_augment_params is not None:
            spec_augmentation = NeuralModule.deserialize(spec_augment_params)
        else:
            spec_augmentation = None

        # Record all modules
        self._modules = []
        self._preprocessor = preprocessor
        self._spec_augmentation = spec_augmentation
        self._encoder = encoder
        self._decoder = decoder
        if spec_augmentation is not None:
            self._modules += [preprocessor, spec_augmentation, encoder, decoder]
        else:
            self._modules += [preprocessor, encoder, decoder]

        # Create input and output ports
        self._input_ports = preprocessor.input_ports
        self._output_ports = decoder.output_ports
        self._output_ports['encoded_lengths'] = encoder.output_ports['encoded_lengths']
        return self._preprocessor, self._spec_augmentation, self._encoder, self._decoder

    @property
    def train_graph(self) -> NeuralGraph:
        return self.__training_neural_graph

    @property
    def eval_graph(self) -> NeuralGraph:
        return self.__evaluation_neural_graph

    @property
    def input_ports(self) -> Optional[Dict[str, NeuralType]]:
        """TODO: write manual docstring here"""
        return self._input_ports

    @property
    def output_ports(self) -> Optional[Dict[str, NeuralType]]:
        """TODO: write manual docstring here"""
        return self._output_ports

    @property
    def num_weights(self):
        return self._encoder.num_weights + self._decoder.num_weights

    @staticmethod
    def list_pretrained_models() -> Optional[List[PretrainedModelInfo]]:
        """List all available pre-trained models (e.g. weights) for QuartzNet.

        Returns:
            A list of PretrainedModelInfo tuples.
            The pretrained_model_name field of the tuple can be used to
            retrieve pre-trained model's weights (pass it as
            pretrained_model_name argument to the module's constructor)
        """
        logging.warning("THIS METHOD IS NOT DONE YET")
        result = []
        enbase = PretrainedModelInfo(
            pretrained_model_name="QuartzNet15x5-En-BASE",
            location=""
            "{'decoder': 'https://api.ngc.nvidia.com/v2/models/nvidia/multidataset_quartznet15x5/versions/1/files/JasperDecoderForCTC-STEP-243800.pt', "
            " 'encoder':'https://api.ngc.nvidia.com/v2/models/nvidia/multidataset_quartznet15x5/versions/1/files/JasperEncoder-STEP-243800.pt', "
            " 'config':'https://api.ngc.nvidia.com/v2/models/nvidia/multidataset_quartznet15x5/versions/1/files/quartznet15x5.yaml'}",
            description="This is a checkpoint for the QuartzNet 15x5 model that was trained in NeMo "
            "on five datasets: LibriSpeech, Mozilla Common Voice, WSJ, Fisher, "
            "and Switchboard.",
            parameters='',
        )
        zhbase = PretrainedModelInfo(
            pretrained_model_name="QuartzNet15x5-Zh-BASE", location="", description="", parameters='',
        )
        result.append(enbase)
        result.append(zhbase)

        zhbase = PretrainedModelInfo(
            pretrained_model_name="Jasper10x5-En-BASE", location="", description="", parameters='',
        )
        result.append(zhbase)

        zhbase = PretrainedModelInfo(
            pretrained_model_name="ContextNet21x5-En-BASE", location="", description="", parameters='',
        )
        result.append(zhbase)
        return result

    @classmethod
    def from_pretrained(cls, model_info, local_rank=0) -> Optional[NeuralModule]:
        # Create destination folder:
        logging.warning("THIS METHOD IS NOT YET FINISHED")
        if model_info.endswith(".nemo"):
            return super().from_pretrained(model_info=model_info)
        else:
            nfname = f".nemo_files/NEMO_{nemo.__version__}/{str(model_info)}"
            home_folder = Path.home()
            dest_dir = os.path.join(home_folder, nfname)

            url = "https://api.ngc.nvidia.com/v2/models/nvidia/multidataset_quartznet15x5/versions/1/files/"
            maybe_download_from_cloud(url=url, filename="JasperEncoder-STEP-243800.pt", dest_dir=dest_dir)
            maybe_download_from_cloud(url=url, filename="JasperDecoderForCTC-STEP-243800.pt", dest_dir=dest_dir)
            maybe_download_from_cloud(url=url, filename="JasperDecoderForCTC-STEP-243800.pt", dest_dir=dest_dir)
            maybe_download_from_cloud(
                url="https://nemo-public.s3.us-east-2.amazonaws.com/", filename="qn.yaml", dest_dir=dest_dir
            )
            logging.info("Instantiating model from pre-trained checkpoint")
            qn = QuartzNet.import_from_config(config_file=os.path.join(dest_dir, "qn.yaml"))
            logging.info("Model instantiated with pre-trained weights")
            return qn

    @property
    def modules(self) -> Iterable[NeuralModule]:
        return self._modules
