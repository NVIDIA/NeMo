from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.distributed
from megatron.core.inference.common_inference_params import CommonInferenceParams
from megatron.core.inference.engines.mcore_engine import MCoreEngine
from megatron.core.inference.model_inference_wrappers.abstract_model_inference_wrapper import AbstractModelInferenceWrapper
from megatron.core.inference.text_generation_controllers.simple_text_generation_controller import (
    SimpleTextGenerationController,
)
from megatron.core.inference.text_generation_controllers.encoder_decoder_text_generation_controller import (
    EncoderDecoderTextGenerationController,
)
from megatron.core.transformer.module import MegatronModule
from pytorch_lightning.trainer.states import TrainerFn

import nemo.lightning as nl
from nemo.lightning import io
from nemo.lightning.ckpt_utils import ckpt_to_context_subdir
from nemo.lightning.pytorch.strategies.megatron_strategy import MegatronStrategy
from nemo.lightning.pytorch.strategies.utils import RestoreConfig

# We need this wrapper since mcore generate uses methods/properties such as tokenizer.detokenize, tokenizer.tokenize, tokenizer.bos, tokenizer.pad, etc. to encode and decode prompts
class MCoreTokenizerWrappper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eod = tokenizer.eod
        self.vocab_size = tokenizer.vocab_size

    def detokenize(self, tokens, remove_special_tokens = False):
        return self.tokenizer.ids_to_text(tokens, remove_special_tokens)

    def tokenize(self, prompt):
        return self.tokenizer.text_to_ids(prompt)

    @property
    def additional_special_tokens_ids(self):
        return self.tokenizer.additional_special_tokens_ids

    @property
    def bos(self):
        return self.tokenizer.bos_id

    @property
    def pad(self):
        return self.tokenizer.pad_id

# TODO: Move to lightning Fabric API.
def _setup_trainer_and_restore_model(path: Path, trainer: nl.Trainer, model: pl.LightningModule):
    assert isinstance(trainer.strategy, MegatronStrategy), "Only MegatronStrategy is supported for trainer.strategy."
    assert trainer.strategy.context_parallel_size <= 1, "Context parallelism is not supported for inference."
    restore_config = RestoreConfig(
        path=path,
        load_model_state=True,
        load_optim_state=False,
    )
    trainer.strategy.restore_config = restore_config
    trainer.strategy._setup_optimizers = False
    trainer.ckpt_path = None
    trainer.strategy.connect(model)
    if trainer.strategy.launcher is not None:
        trainer.strategy.launcher.launch(lambda: None, trainer=trainer)
    trainer.strategy.setup_environment()

    if not model.state_dict():
        model.configure_model()

    trainer.state.fn = TrainerFn.TESTING
    trainer.strategy.setup_megatron_parallel(trainer=trainer)
    trainer.strategy.trainer = trainer
    trainer.strategy.selective_restore()


def setup_model_and_tokenizer(
    path: Path,
    trainer: nl.Trainer,
    params_dtype: torch.dtype = torch.bfloat16,
    inference_batch_times_seqlen_threshold: int = 1000,
) -> tuple[MegatronModule, MCoreTokenizerWrappper]:
    model: io.TrainerContext = io.load_context(path=path, subpath="model")
    _setup_trainer_and_restore_model(path=path, trainer=trainer, model=model)

    inference_wrapped_model = model.get_inference_wrapper(params_dtype, inference_batch_times_seqlen_threshold)
    return inference_wrapped_model, MCoreTokenizerWrappper(model.tokenizer)


def generate(
    model: AbstractModelInferenceWrapper,
    tokenizer: MCoreTokenizerWrappper,
    prompts: list[str],
    encoder_prompts: Optional[list[str]] = None,
    add_BOS: bool = False,
    max_batch_size: int = 4,
    random_seed: Optional[int] = None,
    inference_params: Optional[CommonInferenceParams] = None,
) -> dict:
    if encoder_prompts is not None:
        text_generation_controller = EncoderDecoderTextGenerationController(inference_wrapped_model=model, tokenizer=tokenizer)
    else:
        text_generation_controller = SimpleTextGenerationController(inference_wrapped_model=model, tokenizer=tokenizer)
    mcore_engine = MCoreEngine(
        text_generation_controller=text_generation_controller, max_batch_size=max_batch_size, random_seed=random_seed
    )

    common_inference_params = inference_params or CommonInferenceParams(num_tokens_to_generate=512)

    results = mcore_engine.generate(
        prompts=prompts,
        add_BOS=add_BOS,
        encoder_prompts=encoder_prompts,
        common_inference_params=common_inference_params,
    )

    return results
