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
import re
import warnings
from collections import defaultdict
from contextlib import contextmanager
from itertools import chain
from pathlib import Path
from typing import Any, Generator, Iterable

import hydra
import sacrebleu
import torch
import torchmetrics.functional.text
from lightning import LightningModule
from omegaconf import open_dict
from torch import Tensor
from torch.distributed import init_device_mesh
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
from torch.nn import CrossEntropyLoss
from torchmetrics.text import SacreBLEUScore
from transformers import AutoModel, AutoModelForCausalLM, DynamicCache

from nemo.collections.asr.models import ASRModel
from nemo.collections.common.parts.optional_cuda_graphs import WithOptionalCudaGraphs
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.duplex_s2s.modules import AudioPerceptionModule
from nemo.collections.tts.models import AudioCodecModel
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.utils import logging


# TODO: PyTorchHFHubMixin
class SALM(LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.audio_locator_tag = self.cfg.audio_locator_tag

        # We load the pretrained HF LLM using "ForCausalLM" variant so that we can obtain the
        # pretrained LM head weights.
        # However, for S2S we need to access the activations before LM head directly
        # to feed them to the audio codec head.
        self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True)
        self.tokenizer.add_special_tokens({"additional_special_tokens": [self.audio_locator_tag]})
        self.llm = AutoModelForCausalLM.from_pretrained(self.cfg.pretrained_llm).to(torch.bfloat16).train()
        # Note: we have to "move out" the token embedding outside of LLM to avoid
        #       messing up FSDP/TP hooks.
        self.embed_tokens = self.llm.model.embed_tokens
        del self.llm.model.embed_tokens

        # Load the pretrained streaming ASR model and copy its parameters into the audio perception module.
        asr = _load_pretrained(ASRModel, self.cfg.pretrained_asr).to(torch.bfloat16).eval()
        with open_dict(self.cfg):
            self.cfg.perception.preprocessor = asr.cfg.preprocessor
            self.cfg.perception.encoder = asr.cfg.encoder
            self.cfg.perception.output_dim = self.llm.config.hidden_size
        self.perception = AudioPerceptionModule(self.cfg.perception).train()
        self.perception.load_state_dict(asr.state_dict(), strict=False)

    @property
    def text_vocab_size(self):
        """Return the size of the text tokenizer."""
        return self.embed_tokens.num_embeddings

    @property
    def text_bos_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def text_eos_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def text_pad_id(self) -> int:
        """
        Text pad ID is used as a 'blank' for frames when the model is not speaking
        and for frames where the model is speaking but has already predicted the
        entire text channel's content.

        Example:

            flow:         |---user---||-------assistant--------||-user-|
            text channel:  0000000000  1xxxxxxx0000000000000002  000000

        Where 0 indicates PAD ID, 1 indicates BOS ID, 2 indacates EOS ID,
        and x indicates tokens corresponding to actual text

        """
        pad_id = self.tokenizer.pad
        if pad_id is None:
            pad_id = self.tokenizer.unk_id
        if pad_id is None:
            warnings.warn(
                "the text tokenizer has no <pad> or <unk> tokens available, using id 0 for padding (this may lead to silent bugs)."
            )
            pad_id = 0
        return pad_id

    @property
    def audio_locator_tag_id(self) -> int:
        return self.tokenizer.token_to_id(self.audio_locator_tag)

    def forward(
        self,
        input_embeds: Tensor,
        attention_mask: Tensor = None,
        cache=None,
    ) -> dict[str, Tensor]:
        """
        Implements a fully offline forward pass through the entire model.
        The flow is the following:

        |speech and text embeddings| -> |llm| -> |lm_head| -> |token ids|

        """
        # input_embeds and out: (B, T, H)
        out = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=cache is not None,
            return_dict=True,
        )
        ans = {"logits": out['logits']}  # (B, T, text_vocab_size)
        if cache is not None:
            ans["cache"] = out["past_key_values"]
        return ans

    def prepare_inputs(self, batch: dict):
        """
        Performs additional processing on the mini-batch collected from dataloader.
        Notably:
        * Convert source audio to speech representations.
        * Convert target audio to target audio tokens.
        * Convert target text to embeddings.
        * Combine the input audio and target text embeddings.
        * Take care of any necessary slicing to align the shapes of source audio,
            target audio, and target token ids.
        """

        # def print(*args, **kwargs):
        #     if hasattr(self, "device_mesh") and self.device_mesh is not None:
        #         builtins.print(f"[{self.device_mesh.get_coordinate()}]", *args, **kwargs)
        #     else:
        #         builtins.print(f"[{torch.distributed.get_rank()}]", *args, **kwargs)

        # Source audio encoding.
        # Input audio: (B, T_samples)
        # Audio embeddings: (B, T, H)
        audio_embs, audio_emb_lens = self.perception(
            input_signal=batch["audios"], input_signal_length=batch["audio_lens"]
        )
        audio_embs = [emb[:emblen] for emb, emblen in zip(audio_embs, audio_emb_lens)]
        text_embs = self.embed_tokens(batch["input_ids"].where(batch["input_ids"] > self.text_vocab_size, 0))
        input_embs, target_ids, attention_mask = replace_placeholders_and_build_targets(
            input_ids=batch["input_ids"],
            embeds=text_embs,
            padding_id=self.text_pad_id,
            placeholder_id=self.audio_locator_tag_id,
            replacements=audio_embs,
            target_ids=batch["input_ids"].where(batch["loss_mask"], -100),
        )
        input_embs = input_embs[:, :-1]
        attention_mask = attention_mask[:, :-1]
        target_ids = target_ids[:, 1:]

        # Combine target audio and text into a single tensor to slice them together.
        # It will also help us truncate the sequence lengths to be divisible by TP world size,
        # when TP is enabled.
        # Input ids: (B, T, K+1)
        if self._use_tp:
            tp_world_size = self.device_mesh["tensor_parallel"].size()
            if (remainder := (input_embs.shape[1] - 1) % tp_world_size) != 0:
                # Truncate some tokens from the end to make the sequence lenght shape divisible by tensor parallelism
                # world size. Otherwise, sequence parallelism will change the input shape making leading to mismatches.
                input_embs = input_embs[:, :-remainder]
                attention_mask = attention_mask[:, :-remainder]
                target_ids = target_ids[:, :-remainder]

        return {
            "input_embeds": input_embs,
            "attention_mask": attention_mask,
            "target_ids": target_ids,
        }

    def training_step(self, batch: dict, batch_idx: int):
        if self.cfg.freeze_asr:
            self.perception.preprocessor.eval()
            self.perception.encoder.eval()
        if self.cfg.freeze_llm:
            self.llm.eval()

        inputs = self.prepare_inputs(batch)
        forward_outputs = self(inputs["input_embeds"], attention_mask=inputs["attention_mask"])
        num_frames = (inputs["target_ids"] != -100).long().sum()
        with loss_parallel():
            loss = (
                torch.nn.functional.cross_entropy(
                    forward_outputs["logits"].flatten(0, 1),  # (B, T, Vt) -> (*, Vt)
                    inputs["target_ids"].flatten(0, 1),
                    reduction="sum",
                    ignore_index=-100,
                )
                / num_frames
            )

        B, T = inputs["input_embeds"].shape[:2]
        print(f"{loss=} {B=} {T=}")

        ans = {
            "loss": loss,
            "learning_rate": (
                torch.as_tensor(self.trainer.optimizers[0].param_groups[0]['lr'] if self._trainer is not None else 0)
            ),
            "batch_size": B,
            "sequence_length": T,
            "num_frames": num_frames.to(torch.float32),  # avoid warning
            "target_to_input_ratio": num_frames / (B * T),
            "padding_ratio": batch["loss_mask"].long().sum() / batch["input_ids"].numel(),
        }
        self.log_dict(ans, on_step=True)
        return ans

    def on_validation_epoch_start(self) -> None:
        self._partial_val_losses = defaultdict(list)
        self._partial_accuracies = defaultdict(list)
        # self._refs = defaultdict(list)
        # self._hyps = defaultdict(list)

    def on_validation_epoch_end(self) -> None:
        val_losses = []
        for name, vals in self._partial_val_losses.items():
            val_loss = torch.stack(vals).mean()
            self.log(f"val_loss_{name}", val_loss, on_epoch=True, sync_dist=True)
            print(f"val_loss_{name}", val_loss)
            val_losses.append(val_loss)
        self.log("val_loss", torch.stack(val_losses).mean(), on_epoch=True, sync_dist=True)
        print(f"val_loss", torch.stack(val_losses).mean())

        accuracies = []
        for name, accs in self._partial_accuracies.items():
            val_acc = torch.stack(accs).mean()
            self.log(f"val_acc_{name}", val_acc, on_epoch=True, sync_dist=True)
            print(f"val_acc_{name}", val_acc)
            accuracies.append(val_acc)
        self.log("val_acc", torch.stack(accuracies).mean(), on_epoch=True, sync_dist=True)
        print(f"val_acc", torch.stack(accuracies).mean())

        # corpus_bleus = []
        # for name in self._refs.keys():
        #     val_bleu = torch.tensor(
        #         sacrebleu.corpus_bleu(self._hyps[name], self._refs[name]).score, device=self.device
        #     )
        #     self.log(f"val_bleu_{name}", val_bleu, on_epoch=True, sync_dist=True)
        #     corpus_bleus.append(val_bleu)
        # self.log("val_bleu", torch.stack(corpus_bleus).mean(), on_epoch=True, sync_dist=True)
        #
        # wers = []
        # for name in self._refs.keys():
        #     val_wer = torchmetrics.functional.text.word_error_rate(self._hyps[name], self._refs[name]).to(self.device)
        #     self.log(f"val_wer_{name}", val_wer, on_epoch=True, sync_dist=True)
        #     wers.append(val_wer)
        # self.log("val_wer", torch.stack(wers).mean(), on_epoch=True, sync_dist=True)
        #
        # self._refs.clear()
        # self._hyps.clear()
        self._partial_val_losses.clear()
        self._partial_accuracies.clear()

    def validation_step(self, batch: dict, batch_idx: int):
        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue  # some dataset is exhausted
            inputs = self.prepare_inputs(dataset_batch)
            forward_outputs = self(inputs["input_embeds"], attention_mask=inputs["attention_mask"])
            num_frames = (inputs["target_ids"] != -100).long().sum()
            with loss_parallel():
                loss = (
                    torch.nn.functional.cross_entropy(
                        forward_outputs["logits"].flatten(0, 1),
                        inputs["target_ids"].flatten(0, 1),
                        reduction="sum",
                        ignore_index=-100,
                    )
                    / num_frames
                )

            preds = forward_outputs["logits"].argmax(dim=-1).view(-1)
            refs = inputs["target_ids"].view(-1)
            preds = preds[refs != -100]
            refs = refs[refs != -100]
            accuracy = preds.eq(refs).float().mean()

            self._partial_accuracies[name].append(accuracy)
            self._partial_val_losses[name].append(loss)

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end()

    def test_step(self, *args: Any, **kwargs: Any):
        return self.validation_step(*args, **kwargs)

    def backward(self, *args, **kwargs):
        with loss_parallel():
            super().backward(*args, **kwargs)

    def configure_optimizers(self):
        to_freeze = []
        if self.cfg.freeze_llm:
            to_freeze.extend([r"^llm\..+$", r"^embed_tokens\..+$"])
        if self.cfg.freeze_asr:
            to_freeze.extend([r"^perception\.preprocessor\..+$", r"^perception\.encoder\..+$"])
        parameters = freeze_and_subset(self.named_parameters(), patterns=to_freeze)
        optimizer = hydra.utils.instantiate(self.cfg.optimizer, parameters, _convert_='all')
        ans = {"optimizer": optimizer}
        if "lr_scheduler" in self.cfg:
            lr_scheduler = hydra.utils.instantiate(self.cfg.lr_scheduler, optimizer)
            ans["lr_scheduler"] = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        return ans

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

                parallelize_module(
                    self.llm.lm_head,
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
                self.llm.layers[idx] = fully_shard(layer, **fsdp_config)
            self.embed_tokens = fully_shard(self.embed_tokens, **fsdp_config)
            self.llm.lm_head = fully_shard(self.lm_head, **fsdp_config)
            self.llm = fully_shard(self.llm, **fsdp_config)
            self.perception = fully_shard(self.perception, **fsdp_config)


def _load_pretrained(cls, model_path_or_name: str):
    if Path(model_path_or_name).exists() and model_path_or_name.endswith(".nemo"):
        return cls.restore_from(model_path_or_name)
    else:
        return cls.from_pretrained(model_path_or_name)


def replace_placeholders_and_build_targets(
    input_ids: torch.Tensor,
    embeds: torch.Tensor,
    padding_id: int,
    placeholder_id: int,
    replacements: list[torch.Tensor],
    target_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replaces each occurrence of the placeholder_id in input_ids with the corresponding tensor
    from the replacements list in the embeds tensor, and creates corresponding adjusted target_ids.

    Args:
      input_ids (Tensor): shape (batch, sequence_length); input token ids.
      embeds (Tensor): shape (batch, sequence_length, hidden_dim); embeddings for each token.
      padding_id (int): these IDs will be marked as ignore_index in target_ids.
      placeholder_id (int): an id to be replaced.
      replacements (list of Tensor): each Tensor has shape (L_i, hidden_dim), with L_i arbitrary.
      target_ids (Tensor): shape (batch, sequence_length); target token ids.

    Returns:
      Tuple[Tensor, Tensor, Tensor]:
        - Tensor of shape (batch, max_new_sequence_length, hidden_dim) after replacements.
        - Tensor of shape (batch, max_new_sequence_length) with adjusted target IDs where:
          * Original target values are preserved where input was not a placeholder or padding
          * Positions that were placeholders, padding, or added by replacements are set to -100
        - Tensor of shape (batch, max_new_sequence_length) with attention padding masks.
    """
    batch_size, seq_len = input_ids.size()
    assert target_ids.size() == input_ids.size(), "target_ids must have the same shape as input_ids"

    hidden_dim = embeds.size(2)
    device, dtype = embeds.device, embeds.dtype
    ignore_index = -100  # Standard ignore_index value for CrossEntropyLoss

    output_sequences = []
    output_target_ids = []
    output_att_masks = []
    replacement_idx = 0

    for i in range(batch_size):
        # Find all placeholder positions at once using tensor operations
        placeholder_positions = (input_ids[i] == placeholder_id).nonzero(as_tuple=True)[0]

        # Handle the case with no placeholders more efficiently
        if len(placeholder_positions) == 0:
            output_sequences.append(embeds[i])

            # Start with original target_ids and replace positions where input was padding
            new_target_ids = target_ids[i].clone()
            new_target_ids[input_ids[i] == padding_id] = ignore_index

            output_target_ids.append(new_target_ids)
            output_att_masks.append(input_ids[i] != padding_id)
            continue

        # Build segments between placeholders
        segments = []  # For embeddings
        target_segments = []  # For target IDs
        att_masks = []
        prev_pos = 0

        for pos in placeholder_positions:
            # Add segment before placeholder (if any)
            if pos > prev_pos:
                segments.append(embeds[i, prev_pos:pos])

                # For target IDs: keep original targets but mark positions that were padding in input
                segment_target_ids = target_ids[i, prev_pos:pos].clone()
                segment_target_ids[segment_target_ids == padding_id] = ignore_index

                att_masks.append(segment_target_ids != padding_id)
                target_segments.append(segment_target_ids)

            # Add replacement for embeddings
            rep = replacements[replacement_idx]
            segments.append(rep)

            # For target IDs: all replacement positions get ignore_index
            target_segments.append(torch.full((rep.size(0),), ignore_index, dtype=torch.long, device=device))
            att_masks.append(torch.ones((rep.size(0),), dtype=torch.bool, device=device))

            replacement_idx += 1
            prev_pos = pos + 1  # Skip placeholder

        # Add remaining segment after last placeholder (if any)
        if prev_pos < seq_len:
            segments.append(embeds[i, prev_pos:seq_len])

            # For target IDs: keep original targets but mark positions that were padding in input
            segment_target_ids = target_ids[i, prev_pos:seq_len].clone()
            segment_target_ids[segment_target_ids == padding_id] = ignore_index

            att_masks.append(segment_target_ids != padding_id)
            target_segments.append(segment_target_ids)

        # Concatenate all segments for this example
        output_sequences.append(torch.cat(segments, dim=0))
        output_target_ids.append(torch.cat(target_segments, dim=0))
        output_att_masks.append(torch.cat(att_masks, dim=0))

    # Verify all replacements were used
    if replacement_idx != len(replacements):
        raise ValueError(f"Expected {len(replacements)} replacements but used {replacement_idx}")

    # Create padded output tensors
    max_seq_length = max(seq.size(0) for seq in output_sequences)
    output = torch.zeros(batch_size, max_seq_length, hidden_dim, device=device, dtype=dtype)
    new_target_ids = torch.full((batch_size, max_seq_length), ignore_index, dtype=torch.long, device=device)
    attention_masks = torch.zeros((batch_size, max_seq_length), dtype=torch.bool, device=device)

    for i, (seq, tgt, att) in enumerate(zip(output_sequences, output_target_ids, output_att_masks)):
        seq_len = seq.size(0)
        output[i, :seq_len] = seq
        new_target_ids[i, :seq_len] = tgt
        attention_masks[i, :seq_len] = att

    return output, new_target_ids, attention_masks


def freeze_and_subset(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]], patterns: list[str]
) -> Generator[torch.nn.Parameter, None, None]:
    patterns = [re.compile(p) for p in patterns]
    for name, param in named_parameters:
        discard = False
        for pattern in patterns:
            if pattern.match(name) is not None:
                param.requires_grad = False
                discard = True
        if not discard:
            yield param
