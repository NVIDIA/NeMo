# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import torch
import torch.nn as nn
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
from nemo.collections.speechlm2 import DataModule, DuplexT2TDataset, DuplexT2TModel, Retokenizer
from nemo.collections.speechlm2.parts.pretrained import load_pretrained_nemo
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


@hydra_runner(config_path="conf", config_name="s2s_duplex_frozen_asr")
def train(cfg):
    OmegaConf.resolve(cfg)
    torch.distributed.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.allow_tf32 = True
    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    OmegaConf.save(cfg, log_dir / "exp_config.yaml")

    with trainer.init_module():
        model = DuplexT2TModel(OmegaConf.to_container(cfg.model, resolve=True))
    
    if cfg.model.train_retokenizer:
        # Load FP32 streaming ASR model
        asr_model = load_pretrained_nemo(ASRModel, cfg.model.pretrained_asr).eval()
        decoder_type = cfg.model.get("decoder_type", "ctc")
        # decoder_type = "rnnt"
        if decoder_type == "ctc":
            decoding_cfg = CTCDecodingConfig()
        elif decoder_type == "rnnt":
            decoding_cfg = RNNTDecodingConfig(fused_batch_size=-1, preserve_alignments = True)
        else:
            raise ValueError(f"Invalid decoder type: {decoder_type}")
        asr_model.change_decoding_strategy(decoding_cfg, decoder_type=decoder_type, verbose=True)
        object.__setattr__(model, "_asr", asr_model)
        # NOTE: using model._asr = asr_model instead of object.__setattr__(.)
        # will cause model._asr to be converted to trainer.precision=bf16-true.
        # With object.__setattr__(.) the ASR model will NOT be a part of the
        # model's state_dict and will NOT be saved to the checkpoint.
       
        ctc_linear = asr_model.ctc_decoder.decoder_layers[0]  # Conv1d(512, vocab_size)
        weight = ctc_linear.weight.squeeze(-1) # Shape: [512, vocab_size]
        train_asr_embed = cfg.model.get("train_asr_embed", False)
        embedding_layer = nn.Embedding.from_pretrained(weight, freeze=not train_asr_embed)
        model.asr_embed = embedding_layer

        num_transformer_layers = cfg.model.get("train_retokenizer_transformer_layers", 0)
        if num_transformer_layers > 0:
            use_transformer = True
        else:
            use_transformer = False # Linear layer only
        concat_dim = model.embed_tokens.embedding_dim + model.asr_embed.embedding_dim
        output_dim = model.embed_tokens.embedding_dim

        model.concat_projection = Retokenizer(
            input_dim=concat_dim,
            output_dim=output_dim,
            use_transformer=use_transformer,
            num_layers=num_transformer_layers,
            num_heads=4,
        )

    dataset = DuplexT2TDataset(
        tokenizer=model.tokenizer,
        frame_length=cfg.data.frame_length,
        source_sample_rate=cfg.data.source_sample_rate,
        input_roles=cfg.data.input_roles,
        output_roles=cfg.data.output_roles,
        collate_source_interleaved=cfg.data.collate_source_interleaved,
        train_retokenizer=cfg.model.train_retokenizer,
    )
    datamodule = DataModule(cfg.data, tokenizer=model.tokenizer, dataset=dataset)

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
