import json
import multiprocessing
import sys
import time
from collections import namedtuple

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        return auto_complete_model_config

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
        parameters = model_config["parameters"]
        model_name_or_path = parameters["model_name_or_path"]["string_value"]

        assert args["model_instance_kind"] == "GPU"

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32
        }
        dtype = dtype_map[parameters["dtype"]["string_value"]]
        self.dtype = dtype
        if model_name_or_path.endswith(".nemo"):
          asr_model = nemo_asr.models.ASRModel.restore_from(model_name_or_path)
        else:
          asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name_or_path)
        ctc_decoding = CTCDecodingConfig()
        ctc_decoding.strategy = parameters['decoding_strategy']['string_value']
        if ctc_decoding.strategy != 'greedy':
          ctc_decoding.beam_size = int(parameters['beam_size']['string_value'])
        elif ctc_decoding.strategy == 'greedy':
            ctc_decoding.greedy.batched_inference = True
        asr_model.change_decoding_strategy(ctc_decoding)
        self.model = asr_model.cuda()
        self.model.encoder.freeze()
        self.model.decoder.freeze()
        self.model.eval()
        self.precision_context = torch.cuda.amp.autocast(dtype=self.dtype)

        self.precision_context.__enter__()


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
        torch.cuda.nvtx.range_push(f"inference batch size {len(requests)}")
        
        with torch.inference_mode():
            responses = []
            waveforms = []
            wave_lens = []
            torch.cuda.nvtx.range_push(f"request iteration")
            for i, request in enumerate(requests):
                in_0 = pb_utils.get_input_tensor_by_name(request, "WAV")
                # assume batch size is always 1 squeeze it
                waveforms.append(from_dlpack(in_0.to_dlpack()).squeeze(0))
                in_1 = pb_utils.get_input_tensor_by_name(request, "WAV_LENS")
                wave_lens.append(from_dlpack(in_1.to_dlpack()).squeeze(0))
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("waveform copy")
            # TODO: How do I make sure tensors go onto the right cuda
            # device if more than one cuda device is installed?
            waveform_batch = torch.empty((len(requests), max(wave_lens)),
                                         dtype=waveforms[0].dtype, device="cuda")
            for i, waveform in enumerate(waveforms):
                waveform_batch[i, :wave_lens[i]].copy_(waveform, non_blocking=True)
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("lengths")
            lengths = torch.Tensor(wave_lens).cuda() # cudaStreamSynchronize()
            torch.cuda.nvtx.range_pop()
            
            torch.cuda.nvtx.range_push("preprocessor")
            features_batch, features_lengths = self.model.preprocessor(input_signal=waveform_batch, length=lengths)
            torch.cuda.nvtx.range_pop()
            # TODO: Make this into a cuda graph, if I can. I will need fixed input sizes...
            torch.cuda.nvtx.range_push("encoder")
            log_probs, encoded_len, greedy_predictions = self.model.forward(processed_signal=features_batch,
                                                                            processed_signal_length=features_lengths)
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("decoder")
            transcribed_texts, _ = self.model.decoding.ctc_decoder_predictions_tensor(
                decoder_outputs=log_probs, decoder_lengths=encoded_len, return_hypotheses=False,
            )
            torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push(f"response creation")
        for i in range(len(requests)):
            out_tensor = pb_utils.Tensor("TRANSCRIPTS", np.array([transcribed_texts[i]]).astype(np.object_))
            response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(response)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_pop()

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
        self.precision_context.__exit__(None, None, None)
