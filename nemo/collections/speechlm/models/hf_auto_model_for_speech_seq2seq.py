# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch.distributed._composable.fsdp import MixedPrecisionPolicy
from torch.distributed._composable.fsdp.fully_shard import fully_shard
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.llm import fn
from nemo.lightning import io
from nemo.utils import logging


def masked_cross_entropy(logits, targets, mask=None):
    if mask is not None:
        loss = F.cross_entropy(logits, targets, reduction='none')
        return torch.mean(loss[mask == 1])
    else:
        return F.cross_entropy(logits, targets)


class HFAutoModelForSpeechSeq2Seq(pl.LightningModule, io.IOMixin, fn.FNMixin):
    def __init__(
        self,
        model_name='gpt2',
        load_pretrained_weights=True,
        tokenizer=None,
        loss_fn=masked_cross_entropy,
        model_transform=None,
        model_accelerator=None,
        trust_remote_code=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self._tokenizer = None
        self._processor = None
        self.model = None
        self.loss_fn = loss_fn
        self.load_pretrained_weights = load_pretrained_weights
        self.is_hf_model = True
        self.model_transform = model_transform
        self.model_accelerator = model_accelerator
        self.trust_remote_code = trust_remote_code

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer(
                self.model_name, include_special_tokens=True, trust_remote_code=self.trust_remote_code
            )
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        assert self._tokenizer is None
        self._tokenizer = value

    @property
    def processor(self):
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
        return self._processor

    @staticmethod
    def configure_tokenizer(model_name):
        return AutoProcessor.from_pretrained(model_name).tokenizer

    def configure_model(self, train=True):
        # create all your layers here
        if self.model is None:
            if self.load_pretrained_weights:
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=self.trust_remote_code,
                    use_safetensors=True,
                )
            else:
                from transformers import AutoConfig

                config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
                self.model = AutoModelForSpeechSeq2Seq.from_config(config, trust_remote_code=self.trust_remote_code)

        # Apply FSDP2 and TP to the model
        parallelize(self.model, device_mesh=self.device_mesh)

        if train:
            self.model.train()

    def forward(self, input_features, decoder_input_ids, attention_mask=None):
        return self.model(
            input_features=input_features.to(self.model.device),
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )

    def training_step(self, batch):
        outputs = self.forward(input_features=batch["input_features"], decoder_input_ids=batch["decoder_input_ids"])
        loss_mask = batch.get('loss_mask', None)
        if loss_mask is not None:
            loss_mask = loss_mask.to(self.model.device).view(-1)
        n_cls = outputs.logits.shape[-1]
        logits = outputs.logits.view(-1, n_cls)
        loss = self.loss_fn(logits, batch["labels"], loss_mask)

        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        output = self.forward(input_features=batch["input_features"], decoder_input_ids=batch["decoder_input_ids"])
        loss = output.loss
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

    def save_pretrained(self, path):
        assert self.model is not None, "Model has to be created first."
        self.model.save_pretrained(path)
        if self._tokenizer is not None:
            self._tokenizer.save_pretrained(path)
        else:
            logging.warning("A tokenizer wasn't created before to save.")

        if self._processor is not None:
            self._processor.save_pretrained(path)
        else:
            logging.warning("A processor wasn't created before to save.")


# Taken and modified from torchtitan
# https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/parallelize_llama.py
def parallelize(model, device_mesh: DeviceMesh):
    """Apply parallelisms and activation checkpointing to the model.
    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """

    dp_mesh = device_mesh["data_parallel"]
    tp_mesh = device_mesh["tensor_parallel"]

    if tp_mesh.size() > 1:
        # 1. Parallelize the first embedding and the last linear proj layer
        # 2. Parallelize the root norm layer over the sequence dim
        # 3. Shard the first transformer block's inputs

        # Parallelize the first embedding and the last linear out projection
        plan = {
            "tok_embeddings": RowwiseParallel(input_layouts=Replicate()),
            "output": ColwiseParallel(input_layouts=Shard(1), output_layouts=Replicate()),
            "norm": SequenceParallel(),
            "layers.0": PrepareModuleInput(
                input_layouts=(Replicate(), None),
                desired_input_layouts=(Shard(1), None),
                use_local_output=True,
            ),
        }
        model = parallelize_module(model, tp_mesh, plan)

        # Parallelize each transformer block
        for transformer_block in model.layers.values():
            plan = {
                "attention": PrepareModuleInput(
                    input_layouts=(Shard(1), None),
                    desired_input_layouts=(Replicate(), None),
                ),
                "attention.wq": ColwiseParallel(),
                "attention.wk": ColwiseParallel(),
                "attention.wv": ColwiseParallel(),
                "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
                "attention_norm": SequenceParallel(),
                "feed_forward": PrepareModuleInput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                ),
                "feed_forward.w1": ColwiseParallel(),
                "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
                "feed_forward.w3": ColwiseParallel(),
                "ffn_norm": SequenceParallel(),
            }

            # Adjust attention module to use the local number of heads
            attn_layer = transformer_block.attention
            attn_layer.n_heads = attn_layer.n_heads // tp_mesh.size()
            attn_layer.n_kv_heads = attn_layer.n_kv_heads // tp_mesh.size()

            # Apply the plan for the current transformer block
            parallelize_module(transformer_block, tp_mesh, plan)

    if dp_mesh.size() > 1:
        assert dp_mesh.ndim == 1  # Hybrid-sharding not supported

        # NOTE: Currently, the user is required to manually handle precision settings such as the `mp_policy` here
        # because the model parallel strategy does not respect all settings of `Fabric(precision=...)` at the moment.
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

        fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
        for layer_id, transformer_block in model.layers.items():
            # Apply activation checkpointing
            transformer_block = checkpoint_wrapper(transformer_block)
            # As an optimization, do not reshard after forward for the last
            # transformer block since FSDP would prefetch it immediately
            reshard_after_forward = int(layer_id) < len(model.layers) - 1
            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )
            model.layers[layer_id] = transformer_block
        model = fully_shard(model, **fsdp_config)

    return model
