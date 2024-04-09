import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf, open_dict

from nemo.collections.nlp.modules.common.text_generation_server import MegatronServer
from nemo.collections.nlp.models.language_modeling.megatron_griffin_model import MegatronGriffinModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from pytorch_lightning.trainer.trainer import Trainer
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
import warnings
import torch.nn.functional as F
import numpy as np
import torch

import importlib
import pathlib
from typing import List, Optional

import filelock


class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    @property
    def eos_token(self) -> str:
        return "<|endoftext|>"

    def __call__(self, texts, *args, **kwargs):
        tokens = np.asarray([np.asarray(self.text_to_ids(text)) for text in texts])
        return TokenizerOutput(tokens)

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)


class TokenizerOutput:
    def __init__(self, tokenized_batch_ids):
        self.input_ids = tokenized_batch_ids
        self.attention_mask = np.asarray([
            np.asarray([True] * len(tokenized_batch_ids[i]))
            for i in range(len(tokenized_batch_ids))
        ])


def _patch_pretrained_cfg(pretrained_cfg, trainer):
    import omegaconf

    omegaconf.OmegaConf.set_struct(pretrained_cfg, True)
    with omegaconf.open_dict(pretrained_cfg):
        attributes_to_update = {
            "sequence_parallel": False,
            "activations_checkpoint_granularity": None,
            "activations_checkpoint_method": None,
            "precision": trainer.precision,
            "global_batch_size": None,
            "tensor_model_parallel_size": torch.cuda.device_count(),
            "pipeline_model_parallel_size": 1,
        }
        for name, value in attributes_to_update.items():
            if hasattr(pretrained_cfg, name):
                pretrained_cfg[name] = value
    return pretrained_cfg


def _get_target_from_class(target_class) -> str:
    return f"{target_class.__module__}.{target_class.__name__}"


def load_model(model_path: str, trainer) -> torch.nn.Module:
    from nemo.collections.nlp.models.language_modeling.megatron_griffin_model import MegatronGriffinModel
    from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector

    model_path = pathlib.Path(model_path)

    save_restore_connector = NLPSaveRestoreConnector()
    if model_path.is_dir():
        save_restore_connector.model_extracted_dir = model_path.as_posix()
    pretrained_cfg = save_restore_connector.restore_from(
        None, model_path.as_posix(), return_config=True, trainer=trainer
    )
    if not hasattr(pretrained_cfg, "target"):
        pretrained_cfg["target"] = _get_target_from_class(MegatronGriffinModel)

    pretrained_cfg = _patch_pretrained_cfg(pretrained_cfg, trainer)
    model_to_load_path = model_path
    override_config = pretrained_cfg

    module_name, class_name = override_config.target.rsplit(".", 1)
    model_class = getattr(importlib.import_module(module_name), class_name)

    # monkeypatch _build_tokenizer method to be process-safe
    tokenizer_lock = filelock.FileLock(f"/tmp/{model_path.name}.tokenizer.lock")

    def _synced_build_tokenizer(self):
        with tokenizer_lock:
            self._original_build_tokenizer()

    model_class._original_build_tokenizer = model_class._build_tokenizer
    model_class._build_tokenizer = _synced_build_tokenizer

    model = model_class.restore_from(
        restore_path=model_to_load_path.as_posix(),
        trainer=trainer,
        override_config_path=override_config,
        save_restore_connector=save_restore_connector,
        map_location=f'cuda:{trainer.local_rank}',
    )

    model.freeze()
    model.training = False
    try:
        # Have to turn off activations_checkpoint_method for inference
        model.model.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass
    return model


def setup_distributed_environment(trainer):
    from nemo.utils.app_state import AppState
    def dummy():
        return

    if trainer.strategy.launcher is not None:
        trainer.strategy.launcher.launch(dummy, trainer=trainer)
    trainer.strategy.setup_environment()

    app_state = AppState()

    return app_state


def generate_batches(batch, batch_size):
    start = 0
    while start < len(batch):
        end = start + batch_size
        yield batch[start:end]
        start = end

class NeMoGriffin:

    def __init__(self, path, **kwargs):
        super().__init__()

        from pytorch_lightning.trainer.trainer import Trainer
        from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy

        trainer = Trainer(
            strategy=NLPDDPStrategy(),
            devices=-1,
            accelerator="gpu",
            num_nodes=1,
            precision="bf16",
            logger=False,
            enable_checkpointing=False,
            use_distributed_sampler=False,
        )

        self.model = load_model(path, trainer).cuda().half()
        self.tokenizer = TokenizerWrapper(self.model.tokenizer)
        self.app_state = setup_distributed_environment(trainer)

    def tok_encode(self, string: str):
        return [self.tokenizer.bos_id] + self.tokenizer.text_to_ids(string)

    def tok_decode(self, tokens):
        return self.tokenizer.ids_to_text(tokens)
    

    def generate(
            self,
            input_text: str,
            do_sample: bool,
            max_length: int,  # prompt + generation
            add_BOS: bool = True,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 0.0,
            num_sequences: int = 1,
            batch_size: int = 2,
            compute_attention_mask: bool = False, 
            end_strings: Optional[List[str]] = None,
            compute_logprob=True,
            all_probs=True,
        ):
        from nemo.collections.nlp.modules.common.text_generation_utils import generate

        assert num_sequences > 0

        tokens = self.tok_encode(input_text)
        if len(tokens) > max_length:
            warnings.warn("The number of tokens exceeds `max_length`. Clipping from the right.")
            tokens = tokens[-max_length:]
            input_text = self.tok_decode(tokens)

        tokens_to_generate = max(max_length - len(tokens), 0)

        input_texts = [input_text] * num_sequences

        if tokens_to_generate == 0:
            warnings.warn("tokens_to_generate = 0: returning the inputs")
            return input_texts

        greedy = not do_sample

        end_strings = (end_strings or []) + [self.tokenizer.eos_token]
        
        outputs = []
        for batch in generate_batches(input_texts, batch_size):
            outputs.extend(
                generate(
                    self.model,
                    inputs=batch,
                    tokens_to_generate=tokens_to_generate,
                    end_strings=end_strings,
                    greedy=greedy,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    add_BOS=add_BOS,
                    compute_attention_mask=compute_attention_mask,
                    compute_logprob=compute_logprob,
                    all_probs=all_probs,
                )["sentences"]
            )
        return outputs

def run_eval_harness(path_to_nemo):
    tester = NeMoGriffin(path=path_to_nemo)
    out = tester.generate("Why we must not look directly to the sun during a solar eclipse?",
                    max_length=128, # Must be a power of two between 32 and 65536,!!! keep it low since inference pipe is not complete!!!
                    num_sequences=4,
                    batch_size=2, #Should be >1
                    do_sample=False,
                    top_k=2,
                    compute_logprob=True,
                    all_probs=True)
    print(out)

def run_griffin_server(path_to_nemo):
    tester = NeMoGriffin(path=path_to_nemo)
    server = MegatronServer(tester.model.cuda())
    server.run("0.0.0.0", port=1337)

if __name__ == "__main__":
    PATH_TO_MODEL = "/home/aficek/software/playground/griffin.nemo" # Set path for the .nemo converted checkpoint from the instruction-tuned model from /deepmind/space_gemma_model/2b-it.pt
    # run_eval_harness(PATH_TO_MODEL)
    run_griffin_server(PATH_TO_MODEL)

    