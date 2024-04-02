import json
import copy

import numpy as np
import torch
import triton_python_backend_utils as pb_utils

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
from torch.utils.dlpack import from_dlpack

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
        
        # load model
        parameters = model_config['parameters']
        model_name = parameters["model_name"]['string_value']
        if model_name.endswith('.nemo'):
            model = ASRModel.restore_from(model_name)
        else:
            model = ASRModel.from_pretrained(model_name)
        # init decoding strategy
        ctc_decoding = CTCDecodingConfig()
        ctc_decoding.strategy = parameters['decoding_strategy']['string_value']
        if ctc_decoding.strategy != 'greedy':
            ctc_decoding.beam_size = int(parameters['beam_size']['string_value'])
        
        model.change_decoding_strategy(ctc_decoding)
        self.decoding = model.decoding
         

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
            logits = []
            logits_len = []
            total_number_of_samples = 0
            for i, request in enumerate(requests):
                # the batch size of each input request is 1
                in_0 = pb_utils.get_input_tensor_by_name(request, "log_probs")
                logits.append(from_dlpack(in_0.to_dlpack()).squeeze(0))
                in_1 = pb_utils.get_input_tensor_by_name(request, "encoded_length")
                logits_len.append(from_dlpack(in_1.to_dlpack()).squeeze(0))
                total_number_of_samples += 1

            logits_len = torch.tensor(logits_len, dtype=torch.int32).cuda()
            logits = torch.nn.utils.rnn.pad_sequence(logits, batch_first=True).cuda()
           
            current_hypotheses, _ = self.decoding.ctc_decoder_predictions_tensor(
                    logits, decoder_lengths=logits_len, return_hypotheses=False,
                )
           
            responses = []
            for i in range(len(requests)):
                out_tensor = pb_utils.Tensor("TRANSCRIPT", np.array([current_hypotheses[i]]).astype(np.object_))
                response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
                responses.append(response)
            return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
