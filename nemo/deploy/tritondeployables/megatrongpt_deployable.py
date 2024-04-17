from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.deploy import ITritonDeployable
from pathlib import Path
import wrapt
import logging
import numpy as np
from nemo.collections.nlp.parts.nlp_overrides import (
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
)
from pytorch_lightning.trainer.trainer import Trainer
from nemo.collections.nlp.modules.common.transformer.text_generation import (
    LengthParam,
    SamplingParam,
)
from nemo.collections.nlp.modules.common.text_generation_utils import (
    get_computeprob_response,
    get_default_length_params,
    get_default_sampling_params,
)

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

class MegatronGPTDeployable(ITritonDeployable):
    
    def __init__(self, nemo_checkpoint_filepath: str, load_model: bool = True):
        self.nemo_checkpoint_filepath = nemo_checkpoint_filepath

        if load_model:
            self._load()
    
    def _load(self):
        if Path(self.nemo_checkpoint_filepath).exists():
            trainer = Trainer(
                strategy=NLPDDPStrategy(),
                accelerator="gpu",
                precision="bf16",
                devices=1,
                num_nodes=1,
            )

            self.model = MegatronGPTModel.restore_from(
                self.nemo_checkpoint_filepath,
                trainer=trainer
            )

            self.model.eval()
    
    _INPUT_PARAMETER_FIELDS = {
        "prompts":              (-1,    bytes,      False),
    }
    _SAMPLING_PARAMETER_FIELDS = {
        "use_greedy":           (1,     np.bool_,   True),
        "temperature":          (1,     np.single,  True),
        "top_k":                (1,     np.int_,    True),
        "top_p":                (1,     np.single,  True),
        "repetition_penalty":   (1,     np.single,  True),
        "add_BOS":              (1,     np.bool_,   True),
        "all_probs":            (1,     np.bool_,   True),
        "compute_logprob":      (1,     np.bool_,   True),
        "end_strings":          (-1,    bytes,      False),
    }
    _LENGTH_PARAMETER_FIELDS = {
        "min_length":     (1,     np.int_,    True),
        "max_length":    (1,     np.int_,    False),
    }

    @property
    def get_triton_input(self):
        input_parameters = [Tensor(name=name, shape=(shape,), dtype=dtype, optional=optional) for name,(shape, dtype, optional) in self._INPUT_PARAMETER_FIELDS.items()]
        sampling_parameters = [Tensor(name=name, shape=(shape,), dtype=dtype, optional=optional) for name,(shape, dtype, optional) in self._SAMPLING_PARAMETER_FIELDS.items()]
        length_parameters = [Tensor(name=name, shape=(shape,), dtype=dtype, optional=optional) for name,(shape, dtype, optional) in self._LENGTH_PARAMETER_FIELDS.items()]

        inputs = tuple(input_parameters + sampling_parameters + length_parameters)
        # inputs = (
        #     # Sampling Parameters
        #     Tensor(name="prompts", shape=(-1,), dtype=bytes),
        #     Tensor(name="use_greedy", shape=(1,), dtype=np.bool_, optional=True),
        #     Tensor(name="temperature", shape=(1,), dtype=np.single, optional=True),
        #     Tensor(name="top_k", shape=(1,), dtype=np.int_, optional=True),
        #     Tensor(name="top_p", shape=(1,), dtype=np.single, optional=True),
        #     Tensor(name="repetition_penalty", shape=(1,), dtype=np.single, optional=True),
        #     Tensor(name="add_BOS", shape=(1,), dtype=np.bool_, optional=True),
        #     Tensor(name="all_probs", shape=(1,), dtype=np.bool_, optional=True),
        #     Tensor(name="compute_logprob", shape=(1,), dtype=np.bool_, optional=True),
        #     Tensor(name="end_strings", shape=(-1,), dtype=bytes, optional=True),
        #     # Length Parameters
        #     Tensor(name="min_input_tokens", shape=(1,), dtype=np.int_, optional=True),
        #     Tensor(name="max_output_tokens", shape=(1,), dtype=np.int_),
        # )
        return inputs
    
    @property
    def get_triton_output(self):
        outputs = (
            Tensor(name="outputs", shape=(-1,), dtype=bytes),
        )
        return outputs
    
    @staticmethod
    def _sampling_params_from_triton_inputs(**inputs: np.ndarray):
        sampling_params: SamplingParam = get_default_sampling_params()
        for sampling_param_field in MegatronGPTDeployable._SAMPLING_PARAMETER_FIELDS.keys():
            if sampling_param_field in inputs:
                sampling_params[sampling_param_field] = inputs.pop(sampling_param_field)[0][0]
        return sampling_params
    
    @staticmethod
    def _length_params_from_triton_inputs(**inputs: np.ndarray):
        length_params: LengthParam = get_default_length_params()
        for length_param_field in MegatronGPTDeployable._LENGTH_PARAMETER_FIELDS.keys():
            if length_param_field in inputs:
                length_params[length_param_field] = inputs.pop(length_param_field)[0][0]
        return length_params

    @batch
    def triton_infer_fn(self, **inputs: np.ndarray):
        input_strings = str_ndarray2list(inputs.pop("prompts"))
        sampling_params = self._sampling_params_from_triton_inputs(**inputs)
        length_params = self._length_params_from_triton_inputs(**inputs)

        model_output = self.model.generate(inputs=input_strings, length_params=length_params, sampling_params=sampling_params)
        # to return other non-sentences outputs (logprobs, etc.) will need to add fields to the get_triton_output parameters
        return {"outputs" : cast_output(model_output["sentences"], np.bytes_)}
            