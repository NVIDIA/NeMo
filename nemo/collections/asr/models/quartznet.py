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
from ..audio_preprocessing import AudioPreprocessor, SpectrogramAugmentation
from ..jasper import JasperDecoderForCTC, JasperEncoder
from nemo import logging
from nemo.core import NeMoModel, NeuralModule, NeuralType, PretrainedModelInfo, WeightShareTransform
from nemo.utils import maybe_download_from_cloud


class QuartzNet(NeMoModel):
    def __init__(
        self,
        preprocessor_params: Dict,
        encoder_params: Dict,
        decoder_params: Dict,
        spec_augment_params: Optional[Dict] = None,
        train: bool = True,
    ):
        super().__init__()
        preprocessor, _ = NeuralModule._import_from_config_dict(preprocessor_params)
        encoder, _ = NeuralModule._import_from_config_dict(encoder_params)
        decoder, _ = NeuralModule._import_from_config_dict(decoder_params)
        self.__train = train
        if spec_augment_params is not None:
            if train:
                spec_augment, _ = NeuralModule._import_from_config_dict(spec_augment_params)
            else:
                logging.warning(
                    "Spec Augmentation should not be used in evaluation or inference. Dropping Spec Augment module"
                )
                spec_augment = None
        else:
            spec_augment = None

        self.__instantiate_modules(
            preprocessor=preprocessor, encoder=encoder, decoder=decoder, spec_augmentation=spec_augment
        )

    def __instantiate_modules(
        self,
        preprocessor: AudioPreprocessor,
        encoder: JasperEncoder,
        decoder: JasperDecoderForCTC,
        spec_augmentation: Optional[SpectrogramAugmentation] = None,
    ):
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

    def __call__(self, **kwargs):
        processed_signal, p_length = self._preprocessor(
            input_signal=kwargs['audio_signal'], length=kwargs['a_sig_length']
        )
        if self._spec_augmentation is not None and self.__train:
            processed_signal = self._spec_augmentation(input_spec=processed_signal)
        encoded, encoded_len = self._encoder(audio_signal=processed_signal, length=p_length)
        log_probs = self._decoder(encoder_output=encoded)
        return log_probs, encoded_len

    def export(self, output_file_name: str, output_folder: str = None, deployment: bool = False) -> str:
        if not deployment:
            super().export(output_file_name=output_file_name, output_folder=output_folder, deployment=deployment)

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
        return result

    @classmethod
    def from_pretrained(cls, model_info, local_rank=0) -> Optional[NeuralModule]:
        # Create destination folder:
        logging.warning("THIS METHOD IS NOT DONE YET")
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

    def get_weights(self) -> Optional[Dict[(str, bool)]]:
        pass

    def set_weights(
        self,
        name2weight: Dict[(str, Tuple[str, bool])],
        name2name_and_transform: Dict[(str, Tuple[str, WeightShareTransform])] = None,
    ):
        pass

    def tie_weights_with(
        self,
        module,
        weight_names=List[str],
        name2name_and_transform: Dict[(str, Tuple[str, WeightShareTransform])] = None,
    ):
        pass

    def save_to(self, path: str):
        pass

    def restore_from(self, path: str):
        pass

    def freeze(self, weights: Set[str] = None):
        pass

    def unfreeze(self, weights: Set[str] = None):
        pass
