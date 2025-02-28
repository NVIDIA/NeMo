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
import builtins
from itertools import chain

import torch
from lightning import LightningModule
from omegaconf import open_dict
from torch import Tensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    loss_parallel,
    parallelize_module,
)
from transformers import AutoModel

from nemo.collections.asr.models import ASRModel
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.duplex_s2s.modules import AudioPerceptionModule
from nemo.collections.tts.models import AudioCodecModel
from nemo.utils import logging


# class DuplexS2SModelConfig:
#     pass


class DuplexS2SModel(LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        self.audio_codec = AudioCodecModel.from_pretrained(self.cfg.pretrained_audio_codec).to(torch.bfloat16).eval()

        self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True)

        self.llm = AutoModel.from_pretrained(self.cfg.pretrained_llm).to(torch.bfloat16).train()
        self.embed_tokens = self.llm.embed_tokens  # resize_token_embeddings(self.llm.vocab_size + 8 * 2048)
        del self.llm.embed_tokens
        # self.embed_audio_tokens = torch.nn.Embedding(
        #     8 * 2048, self.embed_tokens.embedding_dim
        # )  # TODO: fetch dim from audio_codec

        speech_encoder = ASRModel.from_pretrained(self.cfg.pretrained_asr).train()

        with open_dict(self.cfg):
            self.cfg.perception.preprocessor = speech_encoder.cfg.preprocessor
            self.cfg.perception.encoder = speech_encoder.cfg.encoder
            self.cfg.perception.output_dim = self.llm.config.hidden_size
        self.perception = AudioPerceptionModule(self.cfg.perception).train()
        # TODO:
        # self.perception.load_state_dict(speech_encoder.state_dict())

        # breakpoint()

        self.lm_head = torch.nn.Linear(self.llm.config.hidden_size, self.llm.config.vocab_size)
        self.audio_head = torch.nn.Linear(
            self.llm.config.hidden_size, 2048 * 8
        )  # TODO: self.audio_codec.codebook_size * num_codebooks ????
        # self.speech_encoder = AudioCodecModel.restore_from("Low_Frame-rate_Speech_Codec++_without_speaker_encoder.nemo")

    def forward(self, input_signal: Tensor, input_signal_lens: Tensor) -> tuple[Tensor, Tensor]:
        # TODO(pzelasko): implement according to
        #   https://github.com/zhehuaichen/NeMo/blob/speechllm-develop-gen_duplex2_clean/nemo/collections/multimodal/speech_llm/models/modular_s2s_models.py
        llm_input, llm_input_lens = self.audio_codec(input_signal, input_signal_lens)
        predicted = self.llm(llm_input, llm_input_lens)
        return predicted, llm_input_lens

    def training_step(self, batch: dict, batch_idx: int):
        def print(*args, **kwargs):
            if hasattr(self, "device_mesh") and self.device_mesh is not None:
                builtins.print(f"[{self.device_mesh.get_coordinate()}]", *args, **kwargs)
            else:
                builtins.print(f"[{torch.distributed.get_rank()}]", *args, **kwargs)

        source_audio, source_audio_lens = batch["source_audio"], batch["source_audio_lens"]
        source_encoded, source_encoded_lens = self.perception(
            input_signal=source_audio, input_signal_length=source_audio_lens
        )
        # print(f"{source_encoded.shape=}")
        # print(f"{source_encoded_lens=}")

        with torch.no_grad():
            target_codes, target_codes_lens = self.audio_codec.encode(
                audio=batch["target_audio"], audio_len=batch["target_audio_lens"]
            )
            target_codes = target_codes.transpose(1, 2)
            # codes_offset = torch.arange(self.tokenizer.vocab_size, self.tokenizer.vocab_size + 8 * 2048, 2048, device=source_encoded.device)[None, None, :]
            # print(f"{codes_offset=}")
            # print(f"{target_codes=}")
            # target_codes = target_codes + codes_offset

        # print(f"{target_codes=}")
        # print(f"{target_codes.shape=}")
        # print(f"{target_codes_lens=}")

        # TODO: TEMPORARY HACK: SLICE UNTIL I CAN LOAD THE CORRECT FRAME RATE AUDIO CODEC
        target_codes = target_codes[:, : source_encoded.shape[1]]

        # print(f"(truncated) {target_codes.shape=}")

        # TODO: resolve the slicing hacks lol
        if batch["target_tokens"].shape[1] < source_encoded.shape[1]:
            pad_id = self.tokenizer.pad
            if pad_id is None:
                pad_id = self.tokenizer.unk_id
            if pad_id is None:
                pad_id = 0  # TODO: cleanup
            batch["target_tokens"] = torch.cat(
                [
                    batch["target_tokens"],
                    (torch.ones(source_encoded.shape[0], 1, device=source_encoded.device) * pad_id).to(torch.long),
                ],
                dim=-1,
            )
        # print(f"{batch['target_tokens'].shape=}")

        input_ids = torch.cat([target_codes, batch["target_tokens"][..., None]], dim=-1)
        if self._use_tp:
            tp_world_size = self.device_mesh["tensor_parallel"].size()
            if (remainder := (input_ids.shape[1] - 1) % tp_world_size) != 0:
                # Truncate some tokens from the end to make the sequence lenght shape divisible by tensor parallelism
                # world size. Otherwise, sequence parallelism will change the input shape making leading to mismatches.
                input_ids = input_ids[:, :-remainder]
                source_encoded = source_encoded[:, :-remainder]
        # print(f"{input_ids.shape=}")
        labels = input_ids[:, 1:]
        input_ids = input_ids[:, :-1]

        input_embeds = self.embed_tokens(input_ids[:, :, -1])
        # input_embeds = (input_embeds + self.embed_audio_tokens(input_ids[:, :, :-1]).sum(dim=2)) / input_ids.shape[2]

        # input_embeds = self.llm.embed_tokens(input_ids)
        # input_embeds = self.embed_tokens(input_ids)
        # print(f"{input_embeds.shape=}")
        # print(f"{input_embeds=}")
        # print(f"{source_encoded=}")

        # TODO: resolve the slicing hacks lol
        # encoder_input = input_embeds.sum(dim=2) + source_encoded[:, :-1] * self.cfg.get("duplex_user_channel_weight", 0.3)
        encoder_input = input_embeds + source_encoded[:, :-1] * self.cfg.get("duplex_user_channel_weight", 0.3)

        out = self.llm(inputs_embeds=encoder_input)
        # print(f"{out['last_hidden_state'].shape=}")

        B, T = encoder_input.shape[:2]

        text_logits = self.lm_head(out['last_hidden_state'])
        # print(f"{text_logits.shape=}")
        audio_logits = self.audio_head(out['last_hidden_state']).view(B, T, 2048, 8)
        # print(f"{audio_logits.shape=}")

        num_frames = target_codes_lens.sum()
        with loss_parallel():
            text_loss = (
                torch.nn.functional.cross_entropy(text_logits.transpose(1, 2), labels[:, :, -1], reduction="sum")
                / num_frames
            )
            audio_loss = torch.nn.functional.cross_entropy(
                audio_logits.transpose(1, 2), labels[:, :, :-1], reduction="sum"
            ) / (
                num_frames * 8
            )  # TODO: num_codebooks
        # print(f"{text_loss=}")
        # print(f"{audio_loss=}")

        loss = text_loss + audio_loss
        print(f"{loss=} {B=} {T=}")

        return loss

    def backward(self, *args, **kwargs):
        with loss_parallel():
            super().backward(*args, **kwargs)

    def configure_optimizers(self):
        # TODO: properly configure the optimizers
        return torch.optim.AdamW(
            chain(
                self.perception.parameters(),
                self.llm.parameters(),
                self.lm_head.parameters(),
                self.audio_head.parameters(),
            ),
            lr=self.cfg.optim.lr,
            foreach=False,  # required for tensor parallelism
        )

    def configure_model(self) -> None:
        self._use_fsdp = False
        self._use_tp = False
        device_mesh = self.device_mesh
        if device_mesh is None:
            return

        if (tp_mesh := device_mesh["tensor_parallel"]).size() > 1:
            self._use_tp = True

            # TODO: Distributing embeddings with TP in this setup is tricky
            #       because we're adding with the output of a non-parallelized
            #       speech encoder.
            # for m in (self.embed_tokens, self.embed_audio_tokens):
            #     parallelize_module(
            #         m,
            #         tp_mesh,
            #         ColwiseParallel(
            #             # input_layouts=Shard(1),
            #             # # Optional: Shard the output along the class dimension to compute the loss in parallel.
            #             # # See `loss_parallel` in `train.py`
            #             # output_layouts=Shard(1),
            #             # use_local_output=False,
            #         ),
            #     )

            # # Parallelize the first embedding and the last linear out projection
            plan = {
                "layers.0": PrepareModuleInput(
                    input_layouts=(Replicate(),),  # , None)
                    desired_input_layouts=(Shard(1),),  # , None)
                    use_local_output=True,
                ),
                "norm": SequenceParallel(),
            }
            parallelize_module(self.llm, tp_mesh, plan)

            # Parallelize each transformer block
            for transformer_block in self.llm.layers:
                plan = {
                    "input_layernorm": SequenceParallel(),
                    "self_attn.q_proj": ColwiseParallel(),
                    "self_attn.k_proj": ColwiseParallel(),
                    "self_attn.v_proj": ColwiseParallel(),
                    "self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
                    "post_attention_layernorm": SequenceParallel(),
                    "mlp": PrepareModuleInput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "mlp.gate_proj": ColwiseParallel(),
                    "mlp.up_proj": ColwiseParallel(),
                    "mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
                    # "pre_feedforward_layernorm": SequenceParallel(),
                    # "post_feedforward_layernorm": SequenceParallel(),
                }

                # Adjust attention module to use the local number of heads
                attn_layer = transformer_block.self_attn
                for attr in ("num_heads", "num_key_value_heads", "hidden_size"):
                    val = getattr(attn_layer, attr)
                    if val % tp_mesh.size() != 0:
                        logging.warning(
                            f"attn_layer.{attr}={val} is not divisible by {tp_mesh.size()=}: set a different tensor parallelism size to avoid errors."
                        )
                    setattr(attn_layer, attr, val // tp_mesh.size())

                # Apply the plan for the current transformer block
                parallelize_module(transformer_block, tp_mesh, plan)

            for m in (self.lm_head, self.audio_head):
                parallelize_module(
                    m,
                    tp_mesh,
                    ColwiseParallel(
                        input_layouts=Shard(1),
                        # Optional: Shard the output along the class dimension to compute the loss in parallel.
                        # See `loss_parallel` in `train.py`
                        output_layouts=Shard(-1),
                        use_local_output=False,
                    ),
                )

        if (dp_mesh := device_mesh["data_parallel"]).size() > 1:
            assert dp_mesh.ndim == 1  # Hybrid-sharding not supported
            self._use_fsdp = True

            fsdp_config = {"mesh": dp_mesh, "mp_policy": MixedPrecisionPolicy(torch.bfloat16)}

            for idx, layer in enumerate(self.llm.layers):
                # layer.self_attn = checkpoint_wrapper(layer.self_attn)
                layer.mlp = checkpoint_wrapper(layer.mlp)
                self.llm.layers[idx] = fully_shard(layer, **fsdp_config)
            self.embed_tokens = fully_shard(self.embed_tokens, **fsdp_config)
            # self.embed_audio_tokens = fully_shard(self.embed_audio_tokens, **fsdp_config)
            self.llm = fully_shard(self.llm, **fsdp_config)
            self.lm_head = checkpoint_wrapper(self.lm_head)
            self.lm_head = fully_shard(self.lm_head, **fsdp_config)
            self.audio_head = checkpoint_wrapper(self.audio_head)
            self.audio_head = fully_shard(self.audio_head, **fsdp_config)

            # for idx, layer in enumerate(self.perception.encoder.layers):
            #     self.perception.encoder.layers[idx] = fully_shard(layer, **fsdp_config)
            # self.perception = checkpoint_wrapper(self.perception)
            self.perception = fully_shard(self.perception, **fsdp_config)
