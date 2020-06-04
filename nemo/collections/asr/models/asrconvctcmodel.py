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
from typing import Dict, Iterable, List, Optional

import nemo
from nemo import logging
from nemo.core import NeMoModel, NeuralGraph, NeuralModule, NeuralType, OperationMode, PretrainedModelInfo
from nemo.utils import maybe_download_from_cloud
from nemo.utils.decorators import add_port_docs


class ASRConvCTCModel(NeMoModel):
    """
    Generic convolutional CTC-based model with encoder and decoder. It also contains pre-processing module and
    data augmentation model.

    Example models of this type are: JasperNet and QuartzNet
    """

    def __init__(
        self,
        preprocessor_params: Dict,
        encoder_params: Dict,
        decoder_params: Dict,
        spec_augment_params: Optional[Dict] = None,
    ):
        super().__init__()
        self.__evaluation_neural_graph = None
        # Instantiate necessary modules
        self.__instantiate_modules(preprocessor_params, encoder_params, decoder_params, spec_augment_params)
        self._operation_mode = OperationMode.training

    def train_call(self, **kwargs):
        processed_signal, processed_signal_len = self._preprocessor(
            input_signal=kwargs["input_signal"], length=kwargs["length"],
        )
        if self._spec_augmentation is not None:
            processed_signal = self._spec_augmentation(input_spec=processed_signal)
        encoded, encoded_len = self._encoder(audio_signal=processed_signal, length=processed_signal_len)
        log_probs = self._decoder(encoder_output=encoded)
        return log_probs, encoded_len

    def eval_call(self, **kwargs):
        i_processed_signal, i_processed_signal_len = self._preprocessor(
            input_signal=kwargs["input_signal"], length=kwargs["length"],
        )
        i_encoded, i_encoded_len = self._encoder(audio_signal=i_processed_signal, length=i_processed_signal_len)
        i_log_probs = self._decoder(encoder_output=i_encoded)
        return i_log_probs, i_encoded_len

    def eval_graph(self):
        """This is only necessary to save the topology during export"""
        if self.__evaluation_neural_graph is not None:
            return self.__evaluation_neural_graph
        else:
            self.__evaluation_neural_graph = NeuralGraph(operation_mode=OperationMode.both)
            with self.__evaluation_neural_graph:
                # Copy one input port definitions - using "user" port names.
                self.__evaluation_neural_graph.inputs["input_signal"] = self._preprocessor.input_ports["input_signal"]
                self.__evaluation_neural_graph.inputs["length"] = self._preprocessor.input_ports["length"]
                # Bind the selected inputs. Connect the modules
                i_processed_signal, i_processed_signal_len = self._preprocessor(
                    input_signal=self.__evaluation_neural_graph.inputs["input_signal"],
                    length=self.__evaluation_neural_graph.inputs["length"],
                )
                # Notice lack of speck augmentation for inference
                i_encoded, i_encoded_len = self._encoder(
                    audio_signal=i_processed_signal, length=i_processed_signal_len
                )
                i_log_probs = self._ecoder(encoder_output=i_encoded)
                # Bind the selected outputs.
                self.__evaluation_neural_graph.outputs["log_probs"] = i_log_probs
                self.__evaluation_neural_graph.outputs["encoded_len"] = i_encoded_len
            return self.__evaluation_neural_graph

    def __instantiate_modules(
        self, preprocessor_params, encoder_params, decoder_params, spec_augment_params=None,
    ):
        preprocessor = NeuralModule.deserialize(preprocessor_params)
        encoder = NeuralModule.deserialize(encoder_params)
        decoder = NeuralModule.deserialize(decoder_params)
        if hasattr(decoder, 'vocabulary'):
            self.__vocabulary = decoder.vocabulary
        else:
            self.__vocabulary = None

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
        return None

    @property
    def eval_graph(self) -> NeuralGraph:
        return self.__evaluation_neural_graph

    @property
    @add_port_docs()
    def input_ports(self) -> Optional[Dict[str, NeuralType]]:
        return self._input_ports

    @property
    @add_port_docs()
    def output_ports(self) -> Optional[Dict[str, NeuralType]]:
        return self._output_ports

    @property
    def vocabulary(self):
        if hasattr(self._decoder, 'vocabulary'):
            return self._decoder.vocabulary
        else:
            logging.warning("The decoder does not have vocabulary set.")
            return None

    @property
    def num_weights(self):
        return self._encoder.num_weights + self._decoder.num_weights

    @staticmethod
    def list_pretrained_models() -> Optional[List[PretrainedModelInfo]]:
        """List all available pre-trained models (e.g. weights) for convolutional
        encoder-decoder CTC-based speech recognition models.

        Returns:
            A list of PretrainedModelInfo tuples.
            The pretrained_model_name field of the tuple can be used to
            retrieve pre-trained model's weights (pass it as
            pretrained_model_name argument to the module's constructor)
        """
        logging.warning("TODO: CHANGE ME TO GRAB STUFF FROM NGC")
        result = []
        model = PretrainedModelInfo(
            pretrained_model_name="QuartzNet15x5-En",
            location="https://nemo-public.s3.us-east-2.amazonaws.com/nemo_0.11_models_test/QuartzNet15x5-En-Base.nemo",
            description="The model is trained on ~3300 hours of publicly available data and achieves a WER of 3.91% on LibriSpeech dev-clean, and a WER of 10.58% on dev-other.",
            parameters="",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="QuartzNet15x5-Zh",
            location="https://nemo-public.s3.us-east-2.amazonaws.com/nemo_0.11_models_test/QuartzNet15x5-Zh-Base.nemo",
            description="The model is trained on ai-shell2 mandarin chinese dataset.",
            parameters="",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="JasperNet10x5-En",
            location="https://nemo-public.s3.us-east-2.amazonaws.com/nemo_0.11_models_test/JasperNet10x5-En-Base.nemo",
            description="The model achieves a WER of 3.46% on LibriSpeech dev-clean, 10.40% on dev-other, 3.69% on test-clean, and 10.49% on test-other.",
            parameters="",
        )
        result.append(model)
        return result

    @classmethod
    def from_pretrained(
        cls, model_info, local_rank: int = 0, refresh_cache: bool = False, new_vocab: List[str] = None
    ) -> Optional[NeuralModule]:
        """Instantiates a particular kind of ASRConvCTCModel from pretrained checkpoint.
        Can do so from file on disk or from the NVIDIA NGC.

        Args:
            model_info: Either path to ".nemo" file or a valid NGC Model name
            local_rank: on which GPU to instantiate.
            refresh_cache: If set to True, then when fetching from clould, this will re-fetch the file
                from clould even if it is  already found in a cache locally.
            new_vocab: If you would like to do fine-tuning with different vocabulary, pass it here. This will keep all
                weghts from the encoder (most of the network) but will randomly re-initialize the decoder with target vocab.

        Returns:
            NeMoModel instance
        """
        # Create destination folder:
        if model_info.endswith(".nemo"):
            instance = super().from_pretrained(model_info=model_info, local_rank=local_rank)
        else:
            location_in_the_cloud = None
            for pretrained_model_info in cls.list_pretrained_models():
                if pretrained_model_info.pretrained_model_name == model_info:
                    location_in_the_cloud = pretrained_model_info.location
            if location_in_the_cloud is None:
                raise FileNotFoundError(
                    f"Could not find {model_info} in the cloud. Please call list_pretrained_models() to see all available pre-trained models."
                )

            filename = location_in_the_cloud.split("/")[-1]
            url = location_in_the_cloud.replace(filename, "")
            cache_subfolder = f"NEMO_{nemo.__version__}"

            # if file exists on cache_folder/subfolder, it will be re-used, unless refresh_cache is True
            nemo_model_file_in_cache = maybe_download_from_cloud(
                url=url, filename=filename, subfolder=cache_subfolder, referesh_cache=refresh_cache
            )
            logging.info("Instantiating model from pre-trained checkpoint")
            instance = ASRConvCTCModel.from_pretrained(model_info=str(nemo_model_file_in_cache), local_rank=local_rank)
            logging.info("Model instantiated with pre-trained weights")
        if new_vocab is None:
            return instance
        else:
            logging.info(f"Changing model's vocabulary to: {new_vocab}")
            feat_in = instance._decoder._feat_in
            del instance._decoder
            instance._decoder = nemo.collections.asr.JasperDecoderForCTC(
                feat_in=feat_in, num_classes=len(new_vocab), vocabulary=new_vocab
            )
            return instance

    @property
    def modules(self) -> Iterable[NeuralModule]:
        return self._modules


class QuartzNet(ASRConvCTCModel):
    """QuartzNet ASR Model. See: "QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel Separable Convolutions."
    https://arxiv.org/abs/1910.10261"""

    @staticmethod
    def list_pretrained_models() -> Optional[List[PretrainedModelInfo]]:
        """List all available pre-trained models (e.g. weights) for convolutional
        encoder-decoder CTC-based speech recognition models.

        Returns:
            A list of PretrainedModelInfo tuples.
            The pretrained_model_name field of the tuple can be used to
            retrieve pre-trained model's weights (pass it as
            pretrained_model_name argument to the module's constructor)
        """
        logging.warning("TODO: CHANGE ME TO GRAB STUFF FROM NGC")
        result = []
        model = PretrainedModelInfo(
            pretrained_model_name="QuartzNet15x5-En",
            location="https://nemo-public.s3.us-east-2.amazonaws.com/nemo_0.11_models_test/QuartzNet15x5-En-Base.nemo",
            description="The model is trained on ~3300 hours of publicly available data and achieves a WER of 3.91% on LibriSpeech dev-clean, and a WER of 10.58% on dev-other.",
            parameters="",
        )
        result.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="QuartzNet15x5-Zh",
            location="https://nemo-public.s3.us-east-2.amazonaws.com/nemo_0.11_models_test/QuartzNet15x5-Zh-Base.nemo",
            description="The model is trained on ai-shell2 mandarin chinese dataset.",
            parameters="",
        )
        result.append(model)
        return result


class JasperNet(ASRConvCTCModel):
    """QuartzNet ASR Model. See: "Jasper: An End-to-End Convolutional Neural Acoustic Model."
    https://arxiv.org/abs/1904.03288"""

    @staticmethod
    def list_pretrained_models() -> Optional[List[PretrainedModelInfo]]:
        """List all available pre-trained models (e.g. weights) for convolutional
        encoder-decoder CTC-based speech recognition models.

        Returns:
            A list of PretrainedModelInfo tuples.
            The pretrained_model_name field of the tuple can be used to
            retrieve pre-trained model's weights (pass it as
            pretrained_model_name argument to the module's constructor)
        """
        logging.warning("TODO: CHANGE ME TO GRAB STUFF FROM NGC")
        result = []
        model = PretrainedModelInfo(
            pretrained_model_name="JasperNet10x5-En",
            location="https://nemo-public.s3.us-east-2.amazonaws.com/nemo_0.11_models_test/JasperNet10x5-En-Base.nemo",
            description="The model achieves a WER of 3.46% on LibriSpeech dev-clean, 10.40% on dev-other, 3.69% on test-clean, and 10.49% on test-other.",
            parameters="",
        )
        result.append(model)
        return result
