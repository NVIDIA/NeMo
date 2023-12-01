# -*- coding: utf-8 -*-
import json
import os

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import LlamaTokenizer

TOKENIZER_DIR = os.environ.get("TOKENIZER_DIR", "/model")

SPACE_CHAR = 9601
NEWLINE_CHAR = 60
STOP_TOKEN = 2


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

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
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        # Parse model configs
        self.model_config = model_config = json.loads(args["model_config"])

        # Parse model output configs
        output_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT")

        # Convert Triton types to numpy types
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

        self.tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_DIR, legacy=False)
        vocab = self.tokenizer.convert_ids_to_tokens(
            list(range(self.tokenizer.vocab_size))
        )

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.
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

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get input tensors
            tokens_batch = pb_utils.get_input_tensor_by_name(
                request, "TOKENS_BATCH"
            ).as_numpy()

            # Reshape Input
            # tokens_batch = tokens_batch.reshape([-1, tokens_batch.shape[0]])
            # tokens_batch = tokens_batch.T

            # Postprocessing output data.
            outputs = self._postprocessing(tokens_batch)

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            output_tensor = pb_utils.Tensor(
                "OUTPUT", np.array(outputs).astype(self.output_dtype)
            )

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occurred"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        `Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        pb_utils.Logger.log("Finalizing the Post-Processing Model.")

    def _id_to_token(self, token_id):
        # handle special tokens (end of string, unknown, etc)
        try:
            special_token_index = self.tokenizer.all_special_ids.index(token_id)
            return self.tokenizer.all_special_tokens[special_token_index]
        except ValueError:
            pass

        # handle typical tokens
        tokens = self.tokenizer.convert_ids_to_tokens(token_id)
        if ord(tokens[0]) == SPACE_CHAR:
            return f" {tokens[1:]}"
        if ord(tokens[0]) == NEWLINE_CHAR:
            return "\n"
        return tokens

    def _postprocessing(self, tokens_batch):
        tokens_batch = tokens_batch.tolist()
        return [
            self._id_to_token(token_id)
            for beam_tokens in tokens_batch
            for token_ids in beam_tokens
            for token_id in token_ids
        ]

        # for beam_tokens in tokens_batch:
        #     for token_ids in beam_tokens:
        #         for token_id in token_ids:
        #             # handle special tokens (end of string, unknown, etc)
        #             special_token = self.tokenizer.added_tokens_decoder.get(token_id)
        #             if special_token:
        #                 tokens = special_token.content

        #             # handle typical tokens
        #             else:
        #                 tokens = self.tokenizer.convert_ids_to_tokens(token_id)
        #                 if ord(tokens[0]) == SPACE_CHAR:
        #                     tokens = f" {tokens[1:]}"
        #                 elif ord(tokens[0]) == NEWLINE_CHAR:
        #                     tokens = "\n"

        #             outputs.append(tokens)
        # return outputs
