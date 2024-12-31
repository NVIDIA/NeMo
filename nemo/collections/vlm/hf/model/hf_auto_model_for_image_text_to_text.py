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
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from nemo.collections.llm import fn
from nemo.lightning import io
from nemo.utils import logging


def masked_cross_entropy(logits, targets, mask=None):
    """Cross entropy with optional mask"""
    if mask is not None:
        loss = F.cross_entropy(logits, targets, reduction='none')
        return torch.mean(loss * mask)
    else:
        return F.cross_entropy(logits, targets)


class HFAutoModelForImageTextToText(pl.LightningModule, io.IOMixin, fn.FNMixin):
    """Wrap's HF's AutoModelForImageTextToText in a pl.LightningModule
    for use within NeMo"""

    def __init__(
        self,
        model_name='gpt2',
        load_pretrained_weights=True,
        processor=None,
        loss_fn=masked_cross_entropy,
        model_transform=None,
        trust_remote_code=False,
        default_dtype=torch.bfloat16,
        load_in_4bit=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self._processor = processor
        self.tokenizer = None
        self.model = None
        self.loss_fn = loss_fn
        self.load_pretrained_weights = load_pretrained_weights
        self.is_hf_model = True
        self.model_transform = model_transform
        self.trust_remote_code = trust_remote_code
        self.load_in_4bit = load_in_4bit

    @property
    def processor(self):
        """Return's module processor"""
        if self._processor is None:
            self._processor = HFAutoModelForImageTextToText.configure_processor(
                self.model_name, trust_remote_code=self.trust_remote_code
            )
        return self._processor

    @processor.setter
    def processor(self, value):
        """Set's module's processor"""
        assert self._processor is None
        self._processor = value

    @staticmethod
    def configure_processor(model_name, trust_remote_code=False):
        """Initializes an AutoProcessor and returns the instance"""
        return AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    def configure_model(self):
        """Instantiates the model"""
        # create all your layers here
        if self.load_pretrained_weights:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                torch_dtype='auto',
                trust_remote_code=self.trust_remote_code,
                load_in_4bit=self.load_in_4bit,
            )
        else:
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
            dtype = getattr(config, 'torch_dtype', self.default_dtype)
            self.model = AutoModelForImageTextToText.from_config(
                config, torch_dtype=dtype, trust_remote_code=self.trust_remote_code
            )

        # Apply FSDP2 and TP to the model
        parallelize(self.model, device_mesh=self.device_mesh)

        self.model.train()

    def forward(self, batch):
        """Runs forward with the model"""
        return self.model(**batch)

    def training_step(self, batch):
        """Run one training step"""
        labels = batch.pop('labels').to(self.model.device)
        loss_mask = batch.pop('loss_mask', None)

        outputs = self.forward(batch)

        logits = outputs.logits.float()
        n_cls = logits.shape[-1]
        logits = logits.view(-1, n_cls)
        labels = labels.view(-1)

        assert logits.shape[-2] == labels.shape[-1], "Expected logits & labels to have the same length"
        loss = self.loss_fn(logits, labels, loss_mask)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    @torch.no_grad
    def validation_step(self, batch, batch_idx):
        """Run one validation step"""
        labels = batch.pop('labels').to(self.model.device)
        loss_mask = batch.pop('loss_mask', None)

        outputs = self.forward(**batch)

        logits = outputs.logits.float()
        n_cls = logits.shape[-1]
        logits = logits.view(-1, n_cls)
        labels = labels.view(-1)

        assert logits.shape[-2] == labels.shape[-1], "Expected logits & labels to have the same length"
        loss = self.loss_fn(logits, labels, loss_mask)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

    def save_pretrained(self, path):
        """Saves checkpoint using HF"""
        assert self.model is not None, "Model has to be created first."
        self.model.save_pretrained(path)
        if self._processor is not None:
            self._processor.save_pretrained(path)
        else:
            logging.warning("A processor wasn't created before to save.")

    @staticmethod
    def extract_skipped_token_ids(tokenizer):
        """Returns list of tokens to mask in labels"""
        # qweb2-2b
        QWEN_TOKENS = [
            '<|im_start|>',
            '<|im_end|>',
            '<|vision_start|>',
            '<|vision_end|>',
            '<|vision_pad|>',
            '<|image_pad|>',
            '<|video_pad|>',
            '<|im_start|>',
            '<|im_end|>',
            '<|vision_start|>',
            '<|vision_end|>',
            '<|vision_pad|>',
            '<|image_pad|>',
            '<|video_pad|>',
        ]
        # llava-1.5-7b-hf, llava-v1.6-mistral-7b-hf
        LLAVA_TOKENS = [
            "<image>",
            "<pad>",
        ]
        LLAMA_TOKENS = [
            '<|begin_of_text|>',
            '<|end_of_text|>',
            '<|finetune_right_pad_id|>',
            '<|step_id|>',
            '<|start_header_id|>',
            '<|end_header_id|>',
            '<|eom_id|>',
            '<|eot_id|>',
            '<|python_tag|>',
            '<|image|>',
        ]
        PAD_TOKENS = set(QWEN_TOKENS + LLAVA_TOKENS + LLAMA_TOKENS)
        tokenizer = getattr(tokenizer, 'tokenizer', tokenizer)
        skipped_token_ids = []
        for key, val in tokenizer.added_tokens_decoder.items():
            if str(val) in PAD_TOKENS:
                skipped_token_ids.append(key)
        return torch.IntTensor(list(set(skipped_token_ids)))


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
