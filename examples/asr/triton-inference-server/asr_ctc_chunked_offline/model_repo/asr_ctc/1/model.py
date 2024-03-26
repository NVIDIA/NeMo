import json
import copy

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from riva.asrlib.decoder.python_decoder import (
    BatchedMappedDecoderCuda,
    BatchedMappedDecoderCudaConfig,
    BatchedMappedOnlineDecoderCuda,
    BatchedMappedOnlineDecoderCudaConfig,
)
from torch.utils.dlpack import from_dlpack
from nemo.utils import logging
import math

from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
from utils import FrameBatchASRWrapper

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        """`auto_complete_config` is called only once when loading the model
        assuming the server was not started with
        `--disable-auto-complete-config`. Implementing this function is
        optional. No implementation of `auto_complete_config` will do nothing.
        This function can be used to set `max_batch_size`, `input` and `output`
        properties of the model using `set_max_batch_size`, `add_input`, and
        `add_output`. These properties will allow Triton to load the model with
        minimal model configuration in absence of a configuration file. This
        function returns the `pb_utils.ModelConfig` object with these
        properties. You can use the `as_dict` function to gain read-only access
        to the `pb_utils.ModelConfig` object. The `pb_utils.ModelConfig` object
        being returned from here will be used as the final configuration for
        the model.

        Note: The Python interpreter used to invoke this function will be
        destroyed upon returning from this function and as a result none of the
        objects created here will be available in the `initialize`, `execute`,
        or `finalize` functions.

        Parameters
        ----------
        auto_complete_model_config : pb_utils.ModelConfig
          An object containing the existing model configuration. You can build
          upon the configuration given by this object when setting the
          properties for this model.

        Returns
        -------
        pb_utils.ModelConfig
          An object containing the auto-completed model configuration
        """
        # inputs = [{
        #     'name': 'INPUT0',
        #     'data_type': 'TYPE_FP32',
        #     'dims': [4]
        # }, {
        #     'name': 'INPUT1',
        #     'data_type': 'TYPE_FP32',
        #     'dims': [4]
        # }]
        # outputs = [{
        #     'name': 'OUTPUT0',
        #     'data_type': 'TYPE_FP32',
        #     'dims': [4]
        # }, {
        #     'name': 'OUTPUT1',
        #     'data_type': 'TYPE_FP32',
        #     'dims': [4]
        # }]

        # # Demonstrate the usage of `as_dict`, `add_input`, `add_output`,
        # # `set_max_batch_size`, and `set_dynamic_batching` functions.
        # # Store the model configuration as a dictionary.
        # config = auto_complete_model_config.as_dict()
        # input_names = []
        # output_names = []
        # for input in config['input']:
        #     input_names.append(input['name'])
        # for output in config['output']:
        #     output_names.append(output['name'])

        # for input in inputs:
        #     # The name checking here is only for demonstrating the usage of
        #     # `as_dict` function. `add_input` will check for conflicts and
        #     # raise errors if an input with the same name already exists in
        #     # the configuration but has different data_type or dims property.
        #     if input['name'] not in input_names:
        #         auto_complete_model_config.add_input(input)
        # for output in outputs:
        #     # The name checking here is only for demonstrating the usage of
        #     # `as_dict` function. `add_output` will check for conflicts and
        #     # raise errors if an output with the same name already exists in
        #     # the configuration but has different data_type or dims property.
        #     if output['name'] not in output_names:
        #         auto_complete_model_config.add_output(output)

        # auto_complete_model_config.set_max_batch_size(0)

        # It would be interesting if

        # To enable a dynamic batcher with default settings, you can use
        # auto_complete_model_config set_dynamic_batching() function. It is
        # commented in this example because the max_batch_size is zero.
        #
        # auto_complete_model_config.set_dynamic_batching()

        return auto_complete_model_config
  
    def __init_from_config(self, config, batch_size):
      dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
      }
      
      model_name_or_path = config.get("model_name")["string_value"]
      if model_name_or_path == "":
        raise ValueError("model_name cannot be empty")
      if model_name_or_path.endswith(".nemo"):
        asr_model = nemo_asr.models.ASRModel.restore_from(model_name_or_path)
      else:
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name_or_path)
      
      if torch.cuda.is_available():
        device = torch.device('cuda')
      else:
        device = torch.device('cpu')
      asr_model.to(device)
      asr_model.eval()
      
      model_cfg = copy.deepcopy(asr_model._cfg)
      if model_cfg.preprocessor.normalize != "per_feature":
        logging.error("Only EncDecCTCModelBPE models trained with per_feature normalization are supported currently")
    
      dtype = config.get("dtype").get("string_value")
      dtype = dtype_map.get(dtype, torch.float16)
      logging.info(f"Using dtype: {dtype}")
      self.model.to(dtype)
      
      model_stride = int(config.get("model_stride").get("string_value"))
      feature_stride = model_cfg.preprocessor['window_stride']
      model_stride_in_secs = feature_stride * model_stride
      self.model_stride_in_secs = model_stride_in_secs
      total_buffer = float(config.get("total_buffer_in_secs").get("string_value"))
      chunk_len = float(config.get("chunk_len_in_secs").get("string_value"))

      tokens_per_chunk = math.ceil(chunk_len / model_stride_in_secs)
      mid_delay = math.ceil((chunk_len + (total_buffer - chunk_len) / 2) / model_stride_in_secs)
      self.delay = mid_delay
      self.tokens_per_chunk = tokens_per_chunk
      logging.info(f"tokens_per_chunk is {tokens_per_chunk}, mid_delay is {mid_delay}")

      batch_size = config.get("batch_size").get("string_value")
      frame_asr = FrameBatchASRWrapper(
        asr_model=asr_model, frame_len=chunk_len, total_buffer=total_buffer, 
        batch_size=batch_size,
      )
      self.asr_model = frame_asr
      
    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device
            ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        model_config = json.loads(args["model_config"])
        self.__init_from_config(model_config, args["max_batch_size"]) 
        assert args["model_instance_kind"] == "GPU"

        torch.cuda.cudart().cudaProfilerStart()

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        with torch.inference_mode():
            responses = []
            # Every Python backend must iterate through list of requests and create
            # an instance of pb_utils.InferenceResponse class for each of them. You
            # should avoid storing any of the input Tensors in the class attributes
            # as they will be overridden in subsequent inference requests. You can
            # make a copy of the underlying NumPy array and store it if it is
            # required.

            torch.cuda.nvtx.range_push("get requests")
            waveforms = []

            for i, request in enumerate(requests):
                # Perform inference on the request and append it to responses
                # list...
                in_0 = pb_utils.get_input_tensor_by_name(request, "WAV").as_numpy()
                waveforms.append(np.squeeze(in_0, 0)) # remove batch dimension

            
            transcribed_texts = self.asr_model.get_batch_preds(waveforms, 
                                                               self.delay, 
                                                               self.tokens_per_chunk,
                                                               self.model_stride_in_secs)
            
           
            for i in range(len(requests)):
                out_tensor = pb_utils.Tensor("TRANSCRIPT", np.array(transcribed_texts[i]).astype(np.object_))
                response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
                responses.append(response)

            torch.cuda.nvtx.range_pop()

            return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
        # torch.cuda.cudart().cudaProfilerStop()
