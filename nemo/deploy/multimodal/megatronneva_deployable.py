# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import logging
from enum import IntEnum, auto
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import wrapt
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.multimodal.models.multimodal_llm.neva.neva_model import MegatronNevaModel
from nemo.collections.multimodal.parts.utils import create_neva_model_and_processor
from nemo.collections.nlp.modules.common.text_generation_utils import (
    OutputType,
    get_default_length_params,
    get_default_sampling_params,
)
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.deploy import ITritonDeployable
from nemo.deploy.utils import cast_output, str_ndarray2list

# import hashlib


@wrapt.decorator
def noop_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


use_pytriton = True
batch = noop_decorator
try:
    from pytriton.decorators import batch
    from pytriton.model_config import Tensor
except Exception:
    use_pytriton = False

LOGGER = logging.getLogger("NeMo")


def GetTensorShape(pyvalue):
    """
    utility function to get Triton Tensor shape from a python value
    assume that lists are shape -1 and all others are scalars with shape 1
    """
    return (-1 if type(pyvalue) == list else 1,)


def GetNumpyDtype(pyvalue):
    """
    utility function to get numpy dtype of a python value
    e.g. bool -> np.bool_
    """
    '''
    manually defining the mapping of python type -> numpy type for now
    is there a better way to do it?  tried np.array(pyvalue).dtype, but that doesn't seem to work
    '''
    py_to_numpy_mapping = {str: bytes, bool: np.bool_, float: np.single, int: np.int_}
    python_type = type(pyvalue)
    # for lists, return the type of the internal elements
    if python_type == list:
        python_type = type(pyvalue[0])
    numpy_type = py_to_numpy_mapping[python_type]
    return numpy_type


class ServerSync(IntEnum):
    """Enum for synchronization messages using torch.distributed"""

    WAIT = auto()
    SIGNAL = auto()

    def to_long_tensor(self):
        return torch.tensor([self], dtype=torch.long, device='cuda')


class MediaType(IntEnum):
    """Enum for selecting media type input into the multimodal model"""

    IMAGE = auto()
    VIDEO = auto()


class MegatronNevaDeployable(ITritonDeployable):
    """Triton inference server compatible deploy class for a .nemo model file"""

    def __init__(
        self,
        nemo_checkpoint_filepath: str = None,
        num_devices: int = 1,
        num_nodes: int = 1,
        existing_model: MegatronNevaModel = None,
        existing_image_processor=None,
        existing_video_processor=None,
    ):
        if nemo_checkpoint_filepath is None and existing_model is None:
            raise ValueError(
                "MegatronLLMDeployable requires either a .nemo checkpoint filepath or an existing MegatronGPTModel, but both provided were None"
            )
        if num_devices > 1:
            LOGGER.warning(
                "Creating a MegatronNevaDeployable with num_devices>1 will assume running with a PyTorch Lightning DDP-variant strategy, which will run the main script once per device. Make sure any user code is compatible with multiple executions!"
            )

        # if both existing_model and nemo_checkpoint_filepath are provided, existing_model will take precedence
        if existing_model is not None:
            self.model = existing_model
            self.image_processor = existing_image_processor
            self.video_processor = existing_video_processor
        else:
            # create_neva_model_and_processor takes an OmegaConf object as input, so need to construct one here
            # required fields: neva_model_file, trainer, tensor_model_parallel_size, pipeline_model_parallel_size
            config = OmegaConf.create(
                {
                    "neva_model_file": nemo_checkpoint_filepath,
                    "trainer": {
                        "devices": num_devices,
                        "num_nodes": num_nodes,
                        "accelerator": "gpu",
                        "logger": False,
                        "precision": "bf16",
                    },
                    "tensor_model_parallel_size": num_devices,
                    "pipeline_model_parallel_size": 1,
                }
            )
            model, image_processor, video_processor = create_neva_model_and_processor(config)
            self.model = model
            self.image_processor = image_processor
            self.video_processor = video_processor

        self.model.eval()
        # helper threads spawned by torch.multiprocessing should loop inside this helper function
        self._helper_thread_evaluation_loop()

    def _helper_thread_evaluation_loop(self):
        # only deploy the server on main thread, other threads enter this evaluation loop
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            while True:
                wait_value = ServerSync.WAIT.to_long_tensor()
                torch.distributed.broadcast(wait_value, 0)
                if wait_value.item() == ServerSync.SIGNAL:
                    self.model.generate(inputs=[""], length_params=None)

    _INPUT_PARAMETER_FIELDS = {
        "prompts": (-1, bytes, False),
        "media_type": (1, np.int_, True),
        "media_list": (-1, bytes, False),
    }

    '''
    there is no get_default equivalent for OutputType like there is for SamplingParameters and LengthParameters
    but we still want to generate output using a real OutputType TypedDict for static type checking
    '''
    _BLANK_OUTPUTTYPE: OutputType = {
        'sentences': [""],
        'tokens': [[""]],
        'logprob': [[0.0]],
        'full_logprob': [[0.0]],
        'token_ids': [[0]],
        'offsets': [[0]],
        'clean_text': [""],
        'clean_response': [""],
    }

    @property
    def get_triton_input(self):
        input_parameters = tuple(
            Tensor(name=name, shape=(shape,), dtype=dtype, optional=optional)
            for name, (shape, dtype, optional) in self._INPUT_PARAMETER_FIELDS.items()
        )
        '''
        in theory, would like to use typedict2tensor() function to generate Tensors, but it purposely ignores 1D arrays
        asked JakubK why on 2024-04-26, but he doesn't know who owns the code
        sampling_parameters = typedict2tensor(SamplingParam)
        length_parameters = typedict2tensor(LengthParam)
        '''
        default_sampling_params: SamplingParam = get_default_sampling_params()
        sampling_parameters = tuple(
            Tensor(
                name=parameter_name,
                shape=GetTensorShape(parameter_value),
                dtype=GetNumpyDtype(parameter_value),
                optional=True,
            )
            for parameter_name, parameter_value in default_sampling_params.items()
        )
        default_length_params: LengthParam = get_default_length_params()
        length_parameters = tuple(
            Tensor(
                name=parameter_name,
                shape=GetTensorShape(parameter_value),
                dtype=GetNumpyDtype(parameter_value),
                optional=True,
            )
            for parameter_name, parameter_value in default_length_params.items()
        )

        inputs = input_parameters + sampling_parameters + length_parameters
        return inputs

    @property
    def get_triton_output(self):
        # outputs are defined by the fields of OutputType
        outputs = [
            Tensor(
                name=parameter_name,
                shape=GetTensorShape(parameter_value),
                dtype=GetNumpyDtype(parameter_value[0]),
            )
            for parameter_name, parameter_value in MegatronNevaDeployable._BLANK_OUTPUTTYPE.items()
        ]
        return outputs

    @staticmethod
    def _sampling_params_from_triton_inputs(**inputs: np.ndarray):
        """Extract SamplingParam fields from triton input dict"""
        sampling_params: SamplingParam = get_default_sampling_params()
        for sampling_param_field in sampling_params.keys():
            if sampling_param_field in inputs:
                sampling_params[sampling_param_field] = inputs.pop(sampling_param_field)[0][0]
        return sampling_params

    @staticmethod
    def _length_params_from_triton_inputs(**inputs: np.ndarray):
        """Extract LengthParam fields from triton input dict"""
        length_params: LengthParam = get_default_length_params()
        for length_param_field in length_params.keys():
            if length_param_field in inputs:
                length_params[length_param_field] = inputs.pop(length_param_field)[0][0]
        return length_params

    @batch
    def triton_infer_fn(self, **inputs: np.ndarray):
        """Triton server inference function that actually runs the model"""
        if torch.distributed.is_initialized():
            distributed_rank = torch.distributed.get_rank()
            if distributed_rank != 0:
                raise ValueError(
                    f"Triton inference function should not be called on a thread with torch.distributed rank != 0, but this thread is rank {distributed_rank}"
                )
            signal_value = ServerSync.SIGNAL.to_long_tensor()
            torch.distributed.broadcast(signal_value, 0)

        input_strings = str_ndarray2list(inputs.pop("prompts"))
        sampling_params = self._sampling_params_from_triton_inputs(**inputs)
        length_params = self._length_params_from_triton_inputs(**inputs)

        media_type = MediaType(inputs.pop("media_type")[0]) if "media_type" in inputs else MediaType.IMAGE
        inference_config_dict = {"inference": {"media_type": media_type.name.lower()}}
        inference_config = OmegaConf.create(inference_config_dict)

        prompt_dict_list = []
        # input media is a list of N numpy arrays, where each array has a single entry of the bytes for the media file
        input_media = inputs.pop("media_list")

        for index, prompt_string in enumerate(input_strings):
            media = input_media[index][0]
            if media_type == MediaType.IMAGE:
                size = len(media)
                image_bytes = BytesIO(media)
                image_bytes.seek(0)
                media = self.image_processor(Image.open(image_bytes).convert('RGB'))
            elif media_type == MediaType.VIDEO:
                media = self.video_processor(media)
            else:
                raise ValueError(f"Expected media_type of 'IMAGE' or 'VIDEO', but got '{media_type.name}'")
            prompt_dict_list.append({"prompt": prompt_string, media_type.name.lower(): media})

        model_output = self.model.generate(
            input_prompts=prompt_dict_list,
            length_params=length_params,
            sampling_params=sampling_params,
            inference_config=inference_config,
        )
        '''
            model_output will be a list of dicts, each with the result for a single prompt
            dict['sentences'] will be a list of strings
            other fields will either be a list (tokens, for example)
            or a pytorch Tensor
        '''

        triton_output = {}
        _OUTPUT_FILLER_VALUES = {
            'tokens': "",
            'logprob': 0.0,
            'full_logprob': 0.0,
            'token_ids': -1,
            'offsets': -1,
            'clean_text': "",
            'clean_response': "",
        }

        # reorganize the output data from a list of dicts to a dict of lists to make padding easier and to match what triton expects
        # also filter out fields that have no data, assuming that if first element is None they should all be None
        num_outputs = len(model_output)
        model_output_organized = {}
        for model_output_field in model_output[0].keys():
            if model_output[0][model_output_field] is not None:
                model_output_organized[model_output_field] = [
                    model_output[i][model_output_field] for i in range(num_outputs)
                ]

        for model_output_field, value in model_output_organized.items():
            if model_output_field not in ['sentences', 'clean_text', 'clean_response'] and value is not None:
                # find length of longest non-sentence output item
                field_longest_output_item = 0
                for item in value:
                    # for llava models, everything is wrapped in an extra list (or tensor) of length 1, so need to index down one level to get the actual list
                    if (type(item[0]) is list and len(item) == 1) or (
                        isinstance(item, torch.Tensor) and item.dim() > 1 and item.shape[0] == 1
                    ):
                        item = item[0]
                    field_longest_output_item = max(field_longest_output_item, len(item))
                # then pad shorter items to match this length
                for index, item in enumerate(value):
                    # same "one level down" check as above
                    has_extra_list_wrapper = (type(item[0]) is list and len(item) == 1) or (
                        isinstance(item, torch.Tensor) and item.dim() > 1 and item.shape[0] == 1
                    )
                    if has_extra_list_wrapper:
                        item = item[0]

                    num_pad_values = field_longest_output_item - len(item)
                    if num_pad_values > 0:
                        pad_value = _OUTPUT_FILLER_VALUES[model_output_field]
                        if isinstance(item, torch.Tensor):
                            pad_tensor = torch.full(
                                (num_pad_values, item.size(1)) if item.dim() > 1 else (num_pad_values,),
                                pad_value,
                                dtype=item.dtype,
                                device='cuda',
                            )
                            padded_item = torch.cat((item, pad_tensor))
                            value[index] = torch.unsqueeze(padded_item, 0) if has_extra_list_wrapper else padded_item
                        else:
                            pad_list = [pad_value] * num_pad_values
                            padded_item = item + pad_list
                            value[index] = [padded_item] if has_extra_list_wrapper else padded_item

            field_dtype = GetNumpyDtype(MegatronNevaDeployable._BLANK_OUTPUTTYPE[model_output_field][0])
            if value is None:
                # triton does not allow for optional output parameters, so need to populate them if they don't exist
                triton_output[model_output_field] = np.full(
                    # 'sentences' should always have a valid value, so use that for the output shape
                    np.shape(model_output_organized['sentences']),
                    MegatronNevaDeployable._BLANK_OUTPUTTYPE[model_output_field][0],
                    dtype=field_dtype,
                )
            elif field_dtype == bytes:
                # strings are cast to bytes
                triton_output[model_output_field] = cast_output(value, field_dtype)
            elif isinstance(value[0], torch.Tensor):
                if value[0].dtype == torch.bfloat16:
                    # numpy currently does not support bfloat16, so need to manually convert it
                    triton_output[model_output_field] = np.array([tensor.cpu().float().numpy() for tensor in value])
                else:
                    triton_output[model_output_field] = np.array([tensor.cpu().numpy() for tensor in value])
            else:
                # non-strings are output as-is (in numpy format)
                triton_output[model_output_field] = np.array(value)
        return triton_output
