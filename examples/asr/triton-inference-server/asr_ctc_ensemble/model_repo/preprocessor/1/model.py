import json
import copy

import numpy as np
import torch
import triton_python_backend_utils as pb_utils

from omegaconf import OmegaConf
from nemo.collections.asr.models import ASRModel
from torch.utils.dlpack import from_dlpack, to_dlpack


class TritonPythonModel:
   
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
        if "GPU" in model_config["instance_group"][0]["kind"]:
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        # load model and get the preprocessor here
        model_name = model_config['parameters']["model_name"]['string_value']
        self.preprocessor = self.extract_preprocessor(model_name, self.device)
    
    def extract_preprocessor(self, asr_model_name, device):
      # load model
      if asr_model_name.endswith('.nemo'):
        model = ASRModel.restore_from(asr_model_name)
      else:
        model = ASRModel.from_pretrained(asr_model_name)
      cfg = copy.deepcopy(model._cfg)
      OmegaConf.set_struct(cfg.preprocessor, False)
      cfg.preprocessor.dither = 0.0
      cfg.preprocessor.pad_to = 0
      preprocessor = model.from_config_dict(cfg.preprocessor)
      return preprocessor.to(device)
         

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
            waveforms = []
            lengths = []
            total_number_of_samples = 0
            for i, request in enumerate(requests):
                # the batch size of each input request is 1
                in_0 = pb_utils.get_input_tensor_by_name(request, "WAV")
                waveforms.append(from_dlpack(in_0.to_dlpack()).squeeze(0))
                in_1 = pb_utils.get_input_tensor_by_name(request, "WAV_LENGTH")
                lengths.append(from_dlpack(in_1.to_dlpack()).squeeze(0))
                total_number_of_samples += 1

            audio_signal_len = torch.tensor(lengths).cuda()
            audio_signal = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True).cuda()
            processed_example, processed_example_length = self.preprocessor(input_signal=audio_signal, 
                                                                         length=audio_signal_len)
           
            responses = []
            for i in range(total_number_of_samples):
                processed_signal = processed_example[i]
               
                processed_signal_len = processed_example_length[i]
               
                processed_signal = processed_signal[:, 0:processed_signal_len]
                
                processed_signal_len = torch.tensor([processed_signal_len], dtype=torch.int32).cuda()
                
                out0 = pb_utils.Tensor.from_dlpack("processed_signal", to_dlpack(processed_signal.unsqueeze(0)))
                out1 = pb_utils.Tensor.from_dlpack("processed_signal_length", to_dlpack(processed_signal_len.unsqueeze(0)))
                responses.append(pb_utils.InferenceResponse(output_tensors=[out0, out1]))
            return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
