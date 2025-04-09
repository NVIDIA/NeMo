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
import warnings
from contextlib import contextmanager
from itertools import chain
from pathlib import Path
from typing import Any

import hydra
import torch
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
        llm = AutoModelForCausalLM.from_pretrained(self.cfg.pretrained_llm).to(torch.bfloat16).train()
        self.llm = llm.model  # fetch PretrainedBaseModel from model "ForCausalLM"
        self.lm_head = llm.lm_head
        # Note: we have to "move out" the token embedding outside of LLM to avoid
        #       messing up FSDP/TP hooks.
        self.embed_tokens = self.llm.embed_tokens
        del self.llm.embed_tokens

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
        return self.tokenizer.vocab_size

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
        cache=None,
    ) -> dict[str, Tensor]:
        """
        Implements a fully offline forward pass through the entire model.
        The flow is the following:

        |speech and text embeddings| -> |llm| -> |lm_head| -> |token ids|

        """
        # input_embeds and out: (B, T, H)
        out = self.llm(
            inputs_embeds=input_embeds, past_key_values=cache, use_cache=cache is not None, return_dict=True
        )
        text_logits = self.lm_head(out['last_hidden_state'])  # (B, T, text_vocab_size)
        ans = {"logits": text_logits}
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
        text_embs = self.embed_tokens(batch["token_ids"])
        combined_embs = replace_placeholders(batch["input_ids"], text_embs, self.audio_locator_tag_id, audio_embs)
        # loss_mask = ...

        # Combine target audio and text into a single tensor to slice them together.
        # It will also help us truncate the sequence lengths to be divisible by TP world size,
        # when TP is enabled.
        # Input ids: (B, T, K+1)
        input_ids = torch.cat([target_codes, target_tokens[..., None]], dim=-1)
        if self._use_tp:
            tp_world_size = self.device_mesh["tensor_parallel"].size()
            if (remainder := (input_ids.shape[1] - 1) % tp_world_size) != 0:
                # Truncate some tokens from the end to make the sequence lenght shape divisible by tensor parallelism
                # world size. Otherwise, sequence parallelism will change the input shape making leading to mismatches.
                input_ids = input_ids[:, :-remainder]
                source_encoded = source_encoded[:, :-remainder]

        text_inputs = input_ids[:, :-1, -1]  # (B, T-1)
        text_labels = input_ids[:, 1:, -1]  # (B, T-1)
        audio_inputs = input_ids[:, :-1, :-1]  # (B, T-1, K)
        audio_labels = input_ids[:, 1:, :-1]  # (B, T-1, K)

        # Input embeds: (B, T-1, H)
        # Note: the order of addition should be consistent with inference code due to
        #       a low numerical precision, i.e.: Input speech + (Output text + Output speech)
        #       Remember that addition is not associative in low precision floating point!
        input_embeds = self.embed_tokens(text_inputs)
        for cbidx in range(self._num_codebooks):
            input_embeds.add_(self.embed_audio_tokens[cbidx](audio_inputs[..., cbidx]))
        input_embeds.add_(source_encoded[:, :-1] * self.cfg.get("duplex_user_channel_weight", 0.3))

        return {
            "input_embeds": input_embeds,
            "input_lens": source_encoded_lens - 1,
            "output_lens": target_codes_lens - 1,
            "text_labels": text_labels,
            "audio_labels": audio_labels,
        }

    def training_step(self, batch: dict, batch_idx: int):
        inputs = self.prepare_inputs(batch)
        forward_outputs = self(inputs["input_embeds"])
        num_frames = inputs["input_lens"].sum()
        with loss_parallel():
            text_loss = (
                torch.nn.functional.cross_entropy(
                    forward_outputs["text_logits"].flatten(0, 1),  # (B, T, Vt) -> (*, Vt)
                    inputs["text_labels"].flatten(0, 1),
                    reduction="sum",
                )
                / num_frames
            )
            audio_loss = torch.nn.functional.cross_entropy(
                forward_outputs["audio_logits"].flatten(0, 2),  # (B, T, K, Vs) -> (*, Vs)
                inputs["audio_labels"].flatten(0, 2),
                reduction="sum",
            ) / (num_frames * self._num_codebooks)

        loss = text_loss + audio_loss

        B, T = inputs["input_embeds"].shape[:2]
        print(f"{loss=} {B=} {T=}")

        ans = {
            "loss": loss,
            "learning_rate": (
                torch.as_tensor(self.trainer.optimizers[0].param_groups[0]['lr'] if self._trainer is not None else 0)
            ),
            "text_loss": text_loss,
            "audio_loss": audio_loss,
            "batch_size": B,
            "sequence_length": T,
            "num_frames": num_frames.to(torch.float32),  # avoid warning
            "padding_ratio": num_frames / (B * T),
        }
        self.log_dict(ans, on_step=True)
        return ans

    def on_validation_epoch_start(self) -> None:
        # Cleaning up GPU memory before we load ASRModel, because it may already
        # be quite fragmented and close to the limit after observing many
        # dynamic shapes during the training epoch.
        torch.cuda.memory.empty_cache()
        self.asr = ASRModel.from_pretrained(self.cfg.scoring_asr).to(torch.bfloat16).eval()
        WithOptionalCudaGraphs.disable_cuda_graphs_recursive(self.asr, attribute_path="decoding.decoding")
        # Setup a separate BLEU metric for each validation dataloader through CombinedLoader.
        # See: https://lightning.ai/docs/pytorch/LTS/guides/data.html#accessing-dataloaders-within-lightningmodule
        self._partial_val_losses = []
        self.asr_bleu = {}
        self.text_bleu = {}
        for name in self.trainer.val_dataloaders.keys():
            self.asr_bleu[name] = SacreBLEUScore().to(self.device)
            self.text_bleu[name] = SacreBLEUScore().to(self.device)

    def on_validation_epoch_end(self) -> None:
        for name, bleu in self.asr_bleu.items():
            self.log(f"val_asr_bleu_{name}", bleu.compute(), on_epoch=True, sync_dist=True)
            bleu.reset()
        for name, bleu in self.text_bleu.items():
            self.log(f"val_text_bleu_{name}", bleu.compute(), on_epoch=True, sync_dist=True)
            bleu.reset()
        self.asr = None  # free up GPU memory
        val_loss = torch.mean(torch.stack(self._partial_val_losses))
        self._partial_val_losses = None
        self.log("val_loss", val_loss, on_epoch=True, sync_dist=True)
        torch.cuda.memory.empty_cache()

    def validation_step(self, batch: dict, batch_idx: int):
        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue  # some dataset is exhausted
            inputs = self.prepare_inputs(dataset_batch)
            forward_outputs = self(inputs["input_embeds"])
            num_frames = inputs["input_lens"].sum()
            with loss_parallel():
                text_loss = (
                    torch.nn.functional.cross_entropy(
                        forward_outputs["text_logits"].flatten(0, 1),
                        inputs["text_labels"].flatten(0, 1),
                        reduction="sum",
                    )
                    / num_frames
                )
                audio_loss = torch.nn.functional.cross_entropy(
                    forward_outputs["audio_logits"].flatten(0, 2),
                    inputs["audio_labels"].flatten(0, 2),
                    reduction="sum",
                ) / (num_frames * self._num_codebooks)

            loss = text_loss + audio_loss
            self._partial_val_losses.append(loss)

            B = inputs["input_embeds"].shape[0]
            self.log(f"val_loss_{name}", loss, on_epoch=True, sync_dist=True, batch_size=B)
            self.log(f"val_text_loss_{name}", text_loss, on_epoch=True, sync_dist=True, batch_size=B)
            self.log(f"val_audio_loss_{name}", audio_loss, on_epoch=True, sync_dist=True, batch_size=B)

            # ASR BLEU
            import torchaudio

            with torch.inference_mode():
                predicted_audio_tokens = torch.argmax(forward_outputs["audio_logits"], dim=-1).transpose(1, 2)
                with _safe_audio_codec_inference():
                    predicted_audio, predicted_audio_lens = self._audio_codec.decode(
                        tokens=predicted_audio_tokens, tokens_len=inputs["output_lens"]
                    )
                ans = self.asr.transcribe(
                    list(torchaudio.functional.resample(predicted_audio, 22050, 16000)),
                    batch_size=predicted_audio.shape[0],
                    verbose=False,
                )
            hyp = [hyp.text for hyp in ans]
            ref = [[tt] for tt in dataset_batch["target_texts"]]
            for h, (r,) in zip(hyp, ref):
                print(f"[AUDIO] Ref: {r}\n[AUDIO] Hyp: {h}")
            self.asr_bleu[name].update(hyp, ref)

            hyp = [
                self.tokenizer.ids_to_text(hyp_ids) for hyp_ids in forward_outputs["text_logits"].argmax(dim=-1).cpu()
            ]
            ref = [[tt] for tt in dataset_batch["target_texts"]]
            for h, (r,) in zip(hyp, ref):
                print(f"[TEXT] Ref: {r}\n[TEXT] Hyp: {h}")
            # Text BLEU
            self.text_bleu[name].update(hyp, ref)

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end()

    def test_step(self, *args: Any, **kwargs: Any):
        return self.validation_step(*args, **kwargs)

    def _get_bos_embedding(self) -> torch.Tensor:
        raise NotImplementedError()

    @torch.inference_mode
    def offline_inference(self, input_signal: torch.Tensor, input_signal_lens: torch.Tensor):
        raise NotImplementedError()

    def backward(self, *args, **kwargs):
        with loss_parallel():
            super().backward(*args, **kwargs)

    def configure_optimizers(self):
        parameters = self.parameters()
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
                self.llm.layers[idx] = fully_shard(layer, **fsdp_config)
            self.embed_tokens = fully_shard(self.embed_tokens, **fsdp_config)
            self.llm = fully_shard(self.llm, **fsdp_config)
            self.lm_head = fully_shard(self.lm_head, **fsdp_config)
            self.perception = fully_shard(self.perception, **fsdp_config)


def _load_pretrained(cls, model_path_or_name: str):
    if Path(model_path_or_name).exists() and model_path_or_name.endswith(".nemo"):
        return cls.restore_from(model_path_or_name)
    else:
        return cls.from_pretrained(model_path_or_name)


def replace_placeholders(
    input_ids: torch.Tensor,
    embeds: torch.Tensor,
    placeholder_id: int,
    replacements: list[torch.Tensor],
) -> torch.Tensor:
    """Replaces each occurrence of the placeholder_id in input_ids with the corresponding tensor
    from the replacements list in the embeds tensor.

        Args:
          input_ids (Tensor): shape (batch, sequence_length); token ids.
          embeds (Tensor): shape (batch, sequence_length, hidden_dim); embeddings for each token.
          placeholder_id (int): an id to be replaced.
          replacements (list of Tensor): each Tensor has shape (L_i, hidden_dim), with L_i arbitrary.

        Returns:
          List[Tensor]: a list with one tensor per batch example. For each example, the tensor has
                        shape (new_sequence_length, hidden_dim) after replacements.
    """

    output_sequences = []  # will contain the new embedding sequences per batch
    replacement_counter = 0  # tracks which replacement tensor to use next
    batch_size, orig_seq_len = input_ids.size()

    # Loop over each example in the batch
    for i in range(batch_size):
        new_sequence_embeds = []  # will store pieces of the new sequence for batch i

        # Loop over each token position in the current example
        for j in range(orig_seq_len):
            # Compare the token at (i, j) to the placeholder_id.
            # input_ids[i, j] is a scalar tensor; .item() obtains its value.
            if input_ids[i, j].item() == placeholder_id:
                # Replace this single embedding vector with the entire sequence from replacements.
                new_sequence_embeds.append(replacements[replacement_counter])
                replacement_counter += 1
            else:
                # Otherwise keep the current embedding. Use unsqueeze(0) so its shape becomes (1, hidden_dim)
                new_sequence_embeds.append(embeds[i, j].unsqueeze(0))

        # Concatenate the collected pieces along the sequence dimension.
        # Note that the new sequence length may differ from the original.
        new_sequence = torch.cat(new_sequence_embeds, dim=0)
        output_sequences.append(new_sequence)

    # Optional: check that we used all replacement tensors.
    if replacement_counter != len(replacements):
        raise ValueError("The number of placeholder occurrences does not match the number of replacements.")

    output = torch.zeros(
        batch_size,
        max(os.size(0) for os in output_sequences),
        embeds.size(2),
        device=embeds.device,
        dtype=embeds.dtype,
    )
    for idx, seq in enumerate(output_sequences):
        output[idx, : seq.size(0), :] = seq

    return output


def build_loss_mask(
    input_ids: torch.Tensor,
    placeholder_id: int,
    replacements: list[torch.Tensor],
) -> torch.Tensor:
    raise NotImplementedError()
