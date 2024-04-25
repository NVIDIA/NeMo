import logging
from pathlib import Path

import numpy as np
import wrapt
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.text_generation_utils import (
    get_default_length_params,
    get_default_sampling_params,
    OutputType
)
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.deploy import ITritonDeployable
from nemo.deploy.utils import cast_output, str_ndarray2list

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

# utility funciton to get Triton Tensor shape from a python value
# assume that lists are shape -1 and all others are scalars with shape 1
def GetTensorShape(pyvalue):
    return (-1 if type(pyvalue)==list else 1,)

# utility function to get numpy dtype of a python value
# e.g. bool -> np.bool_
def GetNumpyDtype(pyvalue):
    # manually defining the mapping of python type -> numpy type for now
    # is there a better way to do it?  tried np.array(pyvalue).dtype, but that doesn't seem to work
    py_to_numpy_mapping = {
        str: bytes,
        bool: np.bool_,
        float: np.single,
        int: np.int_
    }
    python_type = type(pyvalue)
    # for lists, return the type of the internal elements
    if python_type==list:
        python_type = type(pyvalue[0])
    numpy_type = py_to_numpy_mapping[python_type]
    return numpy_type

class MegatronGPTDeployable(ITritonDeployable):
    def __init__(self, nemo_checkpoint_filepath: str, load_model: bool = True):
        self.nemo_checkpoint_filepath = nemo_checkpoint_filepath

        if load_model:
            self._load()

    def _load(self):
        if Path(self.nemo_checkpoint_filepath).exists():
            trainer = Trainer(strategy=NLPDDPStrategy(), accelerator="gpu", precision="bf16", devices=1, num_nodes=1,)

            self.model = MegatronGPTModel.restore_from(self.nemo_checkpoint_filepath, trainer=trainer)

            self.model.eval()

    _INPUT_PARAMETER_FIELDS = {
        "prompts": (-1, bytes, False),
    }

    # there is no get_default equivalent for OutputType like there is for SamplingParameters and LengthParameters
    # but we still want to generate output using a real OutputType TypedDict for static type checking
    _BLANK_OUTPUTTYPE: OutputType = {
        'sentences': [""],
        'tokens': [""],
        'logprob': [0.0],
        'full_logprob': [0.0],
        'token_ids': [0],
        'offsets': [0]
    }

    @property
    def get_triton_input(self):
        input_parameters = [
            Tensor(name=name, shape=(shape,), dtype=dtype, optional=optional)
            for name, (shape, dtype, optional) in self._INPUT_PARAMETER_FIELDS.items()
        ]
        default_sampling_params: SamplingParam = get_default_sampling_params()
        sampling_parameters = [
            Tensor(name=parameter_name, shape=GetTensorShape(parameter_value), dtype=GetNumpyDtype(parameter_value), optional=True)
            for parameter_name, parameter_value in default_sampling_params.items()
        ]
        default_length_params: LengthParam = get_default_length_params()
        length_parameters = [
            Tensor(name=parameter_name, shape=GetTensorShape(parameter_value), dtype=GetNumpyDtype(parameter_value), optional=True)
            for parameter_name, parameter_value in default_length_params.items()
        ]

        inputs = tuple(input_parameters + sampling_parameters + length_parameters)
        return inputs

    @property
    def get_triton_output(self):
        # outputs are defined by the fields of OutputType
        outputs = [
            Tensor(name=parameter_name, shape=GetTensorShape(parameter_value), dtype=GetNumpyDtype(parameter_value),)
            for parameter_name, parameter_value in MegatronGPTDeployable._BLANK_OUTPUTTYPE.items()
        ]
        return outputs

    @staticmethod
    def _sampling_params_from_triton_inputs(**inputs: np.ndarray):
        sampling_params: SamplingParam = get_default_sampling_params()
        for sampling_param_field in sampling_params.keys():
            if sampling_param_field in inputs:
                sampling_params[sampling_param_field] = inputs.pop(sampling_param_field)[0][0]
        return sampling_params

    @staticmethod
    def _length_params_from_triton_inputs(**inputs: np.ndarray):
        length_params: LengthParam = get_default_length_params()
        for length_param_field in length_params.keys():
            if length_param_field in inputs:
                length_params[length_param_field] = inputs.pop(length_param_field)[0][0]
        return length_params

    @batch
    def triton_infer_fn(self, **inputs: np.ndarray):
        input_strings = str_ndarray2list(inputs.pop("prompts"))
        sampling_params = self._sampling_params_from_triton_inputs(**inputs)
        length_params = self._length_params_from_triton_inputs(**inputs)

        model_output = self.model.generate(
            inputs=input_strings, length_params=length_params, sampling_params=sampling_params
        )

        triton_output = {}
        for model_output_field, value in model_output.items():
            field_dtype = GetNumpyDtype(MegatronGPTDeployable._BLANK_OUTPUTTYPE[model_output_field])
            if value is None:
                # triton does not allow for optional output parameters, so need to populate them if they don't exist
                # 'sentences' should always have a valid value, so use that for the output shape
                triton_output[model_output_field] = np.full(np.shape(model_output['sentences']), MegatronGPTDeployable._BLANK_OUTPUTTYPE[model_output_field][0], dtype=field_dtype)
            elif field_dtype==bytes:
                # strings are casted to bytes
                triton_output[model_output_field] = cast_output(value, field_dtype)
            else:
                # everything else is output as-is (in numpy format)
                triton_output[model_output_field] = np.array(value)
        return triton_output
