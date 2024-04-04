import json
import copy

import numpy as np
import torch

from nemo.utils import logging
import math

import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack
import nemo.collections.asr as nemo_asr
from utils import FrameBatchASRWrapper, HFChunkedASR

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
  
    def __init_from_config(self, config):
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
	
      	#model_stride = int(config.get("model_stride").get("string_value"))
      	#assert asr_model.encoder.subsampling_factor == model_stride
        total_buffer = float(config.get("total_buffer_in_secs").get("string_value"))
        chunk_len = float(config.get("chunk_len_in_secs").get("string_value"))
        
        batch_size = int(config.get("batch_size").get("string_value"))
        strategy = config.get("chunk_strategy").get("string_value")
        if strategy == "nemo_chunked_infer":
            feature_stride = model_cfg.preprocessor['window_stride']
            model_stride = asr_model.encoder.subsampling_factor
            model_stride_in_secs = feature_stride * model_stride
            self.model_stride_in_secs = model_stride_in_secs
            tokens_per_chunk = math.ceil(chunk_len / model_stride_in_secs)
            mid_delay = math.ceil((chunk_len + (total_buffer - chunk_len) / 2) / model_stride_in_secs)
            self.delay = mid_delay
            self.tokens_per_chunk = tokens_per_chunk
            logging.info(f"tokens_per_chunk is {tokens_per_chunk}, mid_delay is {mid_delay}")
            frame_asr = FrameBatchASRWrapper(
      	    	asr_model=asr_model, frame_len=chunk_len, total_buffer=total_buffer, 
      	    	batch_size=batch_size)
            
            self.asr_model = frame_asr
            self.strategy = "nemo_chunked_infer"
        elif strategy == "hf_chunked_infer":
            hf_asr = HFChunkedASR(
      	    	asr_model=asr_model,
      	    	chunk_len_in_secs=total_buffer,
      	    	overlapping_in_secs=(total_buffer - chunk_len) // 2,
      	    	batch_size=batch_size)
            self.asr_model = hf_asr
            self.strategy = "hf_chunked_infer"
        else:
            raise ValueError(f"Unknown chunk strategy: {strategy}")
        self.precision_context = torch.cuda.amp.autocast(dtype=dtype)
    
    def initialize(self, args):
        config = json.loads(args["model_config"])
        self.__init_from_config(config["parameters"])
        assert args["model_instance_kind"] == "GPU"

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
            with self.precision_context:
                responses = []
                waveforms = []
                lengths = []
            for i, request in enumerate(requests):
                # Perform inference on the request and append it to responses
                # # list...
                # # Perform inference on the request and append it to responses
                # # list...
                in_0 = pb_utils.get_input_tensor_by_name(request, "WAV")
              	# assume batch size is always 1 squeeze it
                waveforms.append(from_dlpack(in_0.to_dlpack()).squeeze(0))
                in_1 = pb_utils.get_input_tensor_by_name(request, "WAV_LENS")
                lengths.append(from_dlpack(in_1.to_dlpack()).squeeze(0))

            if self.strategy == "nemo_chunked_infer":
                transcribed_texts = self.asr_model.get_batch_preds(waveforms,
                                                               self.delay, 
                                                               self.tokens_per_chunk,
                                                               self.model_stride_in_secs)
            elif self.strategy == "hf_chunked_infer":
                transcribed_texts = self.asr_model.get_batch_preds(waveforms)
           
            for i in range(len(requests)):
                out_tensor = pb_utils.Tensor("TRANSCRIPTS", np.array([transcribed_texts[i]]).astype(np.object_))
                response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
                responses.append(response)

            return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')