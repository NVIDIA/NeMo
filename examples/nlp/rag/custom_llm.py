from typing import Optional, List, Mapping, Any

from llama_index.core import SimpleDirectoryReader, SummaryIndex
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import PrivateAttr

from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core import Settings
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from pytorch_lightning.trainer.trainer import Trainer
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy

class NeMoLLM(CustomLLM):
    context_window: int = 3900
    num_output: int = 256
    model_name: str = "NeMo LLM"
    dummy_response: str = "My response"

    length_params: LengthParam = {
        "max_length": 30,
        "min_length": 0,
    }

    sampling_params: SamplingParam = {
        "use_greedy": True,
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "add_BOS": True,
        "all_probs": False,
        "compute_logprob": False,
        "end_strings": ["<|endoftext|>"],
    }

    _model: Any = PrivateAttr()
    _model_config: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()

    def __init__(
        self,
        model_path: str = None,
        **kwargs: Any,
    ) -> None:

        # set up trainer
        trainer_config = {
            "devices": 1,
            "num_nodes": 1,
            "accelerator": "gpu",
            "logger": False,
            "precision": 'bf16-mixed',
        }
    
        tensor_model_parallel_size = 1
        pipeline_model_parallel_size = 1

        # trainer required for restoring model parallel models
        trainer = Trainer(strategy=NLPDDPStrategy(), **trainer_config)
        assert (
            trainer_config["devices"] * trainer_config['num_nodes']
            == tensor_model_parallel_size * pipeline_model_parallel_size
        ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"

        # setup/override model config
        model_cfg = MegatronGPTModel.restore_from(restore_path=model_path, trainer=trainer, return_config=True)
        model_cfg.micro_batch_size = 1
        model_cfg.global_batch_size = 1
        self._model_config = model_cfg
        print("model_cfg: ", model_cfg)

        # restore model
        model = MegatronGPTModel.restore_from(restore_path=model_path, trainer=trainer, override_config_path=model_cfg, strict=True)
        model.freeze()
        self._model = model
        super().__init__(**kwargs)

        # update LLM metadata
        self.context_window = self._model_config.encoder_seq_length

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        llm_response = self._model.generate(inputs=[prompt], length_params=self.length_params, sampling_params=self.sampling_params)
        text_response = llm_response['sentences'][0]

        return CompletionResponse(text=text_response)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        llm_response = self._model.generate(inputs=[prompt], length_params=self.length_params, sampling_params=self.sampling_params)
        text_response = llm_response['sentences'][0]

        response = ""
        for token in text_response:
            response += token
            yield CompletionResponse(text=response, delta=token)