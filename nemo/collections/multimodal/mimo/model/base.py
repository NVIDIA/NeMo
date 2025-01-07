from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
from megatron.core import parallel_state as ps
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.inference_params import InferenceParams
from megatron.core.models.multimodal.llava_model import LLaVAModel as MCoreLLaVAModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType as MCoreAttnMaskType
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.transformer_config import TransformerConfig

from nemo.collections.llm.gpt.model import transformer_engine_layer_spec
from nemo.collections.multimodal.mimo.model.gpt import MimoGPTModel


class BaseMimoModel(MCoreLLaVAModel):
    def __init__(
        self,
        config: TransformerConfig,
    ) -> None:
        if config.stage == 'encoder_alignment':
            self.add_encoder = ps.is_pipeline_first_stage()
        elif config.stage == 'decoder_alignment':
            self.add_encoder = False
        else:
            raise NotImplementedError(f"Training stage {config.stage} is not implemented yet.")
        super().__init__(
            language_transformer_config=config.language_transformer_config,
            language_transformer_layer_spec=transformer_engine_layer_spec(config.language_transformer_config),
            language_vocab_size=config.vocab_size,
            language_max_sequence_length=config.language_transformer_config.seq_length,
            vision_transformer_config=config.image_encoder_transformer_config,
            vision_transformer_layer_spec=config.image_encoder_transformer_config.layer_spec,
            drop_vision_class_token=config.image_encoder_transformer_config.drop_vision_class_token,
            vision_projection_config=config.image_input_projection_config,
            vision_projection_layer_spec=config.image_input_projection_config.layer_spec,
            vision_projection_type=config.image_input_projection_config.projector_type,
            allow_missing_vision_projection_checkpoint=True,
            parallel_output=config.language_transformer_config.parallel_output,
            language_position_embedding_type=config.language_transformer_config.position_embedding_type,
            language_rotary_percent=config.language_transformer_config.rotary_percent,
            pre_process=ps.is_pipeline_first_stage(),
            post_process=ps.is_pipeline_last_stage(),
            add_encoder=self.add_encoder,
            add_decoder=False,  # Ensure GPTModel isn't initialized
            img_h=config.image_encoder_transformer_config.img_h,
            img_w=config.image_encoder_transformer_config.img_w,
            patch_dim=config.image_encoder_transformer_config.patch_dim,
            language_rotary_base=config.language_transformer_config.rotary_base,
            image_token_index=-200,
            pixel_shuffle=False,
            tile_tags=None,
        )
        self.config = config
        # Now re-enable add_decoder after parent constructor is done
        self.add_decoder = (
            ps.is_pipeline_last_stage()
            or ps.get_pipeline_model_parallel_rank() >= self.encoder_pipeline_model_parallel_size
        )
        self.model_type = ModelType.encoder_or_decoder

        if self.add_decoder:
            # Initialize MimoGPTModel

            self.language_model = MimoGPTModel(
                config=config.language_transformer_config,
                transformer_layer_spec=transformer_engine_layer_spec(config.language_transformer_config),
                vocab_size=config.vocab_size,
                max_sequence_length=config.language_transformer_config.seq_length,
                parallel_output=config.language_transformer_config.parallel_output,
                position_embedding_type=config.language_transformer_config.position_embedding_type,
                rotary_percent=config.language_transformer_config.rotary_percent,
                pre_process=ps.is_pipeline_first_stage(),
                post_process=ps.is_pipeline_last_stage(),
                rotary_base=config.language_transformer_config.rotary_base,
            )

            self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights
            self._language_max_sequence_length = config.language_transformer_config.seq_length
            self._language_is_pipeline_parallel = config.language_transformer_config.pipeline_model_parallel_size > 1

        if config.stage not in ["encoder_alignment"]:
            self.image_decoder = config.image_decoder_transformer_config.configure_model()

            self.image_output_projection_module = config.image_output_projection_config.configure_model()

    def get_image_caption_embeddings(self, text_input):
        with torch.no_grad():
            text_inputs = self.image_decoder.tokenizer(
                text_input, padding="max_length", truncation=True, return_tensors="pt", add_special_tokens=True
            )
            text_inputs = text_inputs.to(self.image_decoder.device)
            image_caption_embeddings = self.image_decoder.text_encoder(**text_inputs)[0]  # b,77,1024

            return image_caption_embeddings

    # write a function to get backward pass

    def forward(
        self,
        images: torch.Tensor,
        output_images: torch.Tensor,
        input_ids: torch.Tensor,
        input_text: str,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        num_image_tiles: Optional[List[int]] = None,
        image_token_index: Optional[int] = -200,
        runtime_gather_output: Optional[bool] = None,
        image_token_mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> torch.Tensor:
        """Forward function of the LLaVA model.

        Args:
            images (torch.Tensor): input images of shape [num_tiles, img_h, img_w].
                num_tiles means the number of image tiles in this batch.
                num_tiles = 0 if the batch doesn't contain images.
            input_ids (torch.Tensor): input text ids [batch, text_seq_len].
            position_ids (torch.Tensor): input text position ids [batch, text_seq_len].
            attention_mask (torch.Tensor): Language model attention mask
                [batch, 1, combined_seq_len, combined_seq_len].
            labels (torch.Tensor): Optional target text labels [batch, combined_seq_len].
            loss_mask (torch.Tensor): Text loss mask [batch, text_seq_len].
            inference_params (InferenceParams): Inference-time parameters including KV cache.
            num_image_tiles (list of int): Number of tiles per image. Default 1 tile per image.
            image_token_index (int): ID for input images.
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.
             image_token_mask (torch.Tensor): Tensor indicating the location of
                image token index in input_ids.
            packed_seq_params (PackedSeqParams): 1) If using sequence packing, must contain
                subsample length information. 2) If using SP/CP with padding mask type,
                must contain padded token information.
        Returns:
            output (torch.Tensor): Loss of shape [b, s] if labels are provided,
                otherwise logits of shape [b, s, vocab_size].
            loss_mask (torch.Tensor): Loss mask expanded to combined sequence length. Shape [b, s].
        """
        has_images = images is not None and images.shape[0] > 0

        if self.add_encoder and has_images:
            conv1_weight_dtype = self.vision_model.conv1.weight.dtype
            images = images.to(conv1_weight_dtype)
        device, dtype = (
            self.language_model.embedding.word_embeddings.weight.device,
            self.language_model.embedding.word_embeddings.weight.dtype,
        )
        self.encoder_hidden_state = torch.tensor([], dtype=dtype, device=device).reshape(0, 0, 0)

        (output, hidden_states), new_loss_mask = super().forward(
            images=images,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
            inference_params=inference_params,
            num_image_tiles=num_image_tiles,
            image_token_index=image_token_index,
            runtime_gather_output=runtime_gather_output,
            image_token_mask=image_token_mask,
            packed_seq_params=packed_seq_params,
        )
        # we dont need hidden states for encoder alignment

        if self.config.stage in ["encoder_alignment"]:
            return output, new_loss_mask
        elif self.config.stage in ["decoder_alignment"]:
            if labels is None:
                return output, hidden_states
            else:
                device = output_images.device
                # TODO: Yash refactor this, dont have to keep moving image decoder to device
                image_decoder = self.image_decoder.to(device)
                image_caption_embeddings = self.get_image_caption_embeddings(input_text)  # (bs, 77, 1024)
                special_token_mask = torch.zeros_like(labels, dtype=torch.bool)
                for idx in self.config.image_special_token_indices:
                    special_token_mask |= labels == idx

                nonzero_indices = torch.nonzero(special_token_mask, as_tuple=False)
                special_token_positions = nonzero_indices[:, 1]
                special_token_indices = labels[special_token_mask]

                special_token_positions = special_token_positions.view(
                    labels.size(0), -1
                )  # batch_size, no_special_tokens
                special_token_indices = special_token_indices.view(labels.size(0), -1)

                special_token_mask = special_token_mask.transpose(0, 1).unsqueeze(-1)
                special_token_mask = special_token_mask.expand_as(hidden_states)
                selected_hidden_states = hidden_states[special_token_mask].view(
                    hidden_states.size(1), -1, hidden_states.size(-1)
                )

                special_token_embeddings = self.language_model.embedding(
                    input_ids=special_token_indices, position_ids=special_token_positions
                )
                special_token_embeddings = special_token_embeddings.transpose(0, 1)

                inp_to_vision_projection = selected_hidden_states + special_token_embeddings
                output_projection_embeddings = self.image_output_projection_module(inp_to_vision_projection)
                output_projection_embeddings = output_projection_embeddings.to(dtype=image_decoder.dtype)
                # image_caption_embeddings = image_caption_embeddings.to(
                #     output_projection_embeddings.device, dtype=output_projection_embeddings.dtype
                # )

                latents = image_decoder.vae.encode(output_images).latent_dist.sample()
                latents = latents * image_decoder.vae.config.scaling_factor

                noise = torch.randn_like(latents)
                batch_size = latents.shape[0]
                timesteps = torch.randint(
                    0, image_decoder.scheduler.config.num_train_timesteps, (batch_size,), device=latents.device
                )
                timesteps = timesteps.long()

                noisy_latents = image_decoder.scheduler.add_noise(latents, noise, timesteps)

                target = noise
                # noisy_latents = noisy_latents.to(output_projection_embeddings.dtype)
                model_pred = image_decoder.unet(noisy_latents, timesteps, output_projection_embeddings).sample

                return {
                    'output': output,
                    'new_loss_mask': new_loss_mask,
                    'output_projection_embeddings': output_projection_embeddings,
                    'image_caption_embeddings': image_caption_embeddings,
                    'hidden_states': hidden_states,
                    'denoise_model_pred': model_pred,
                    'denoise_target': target,
                }

        else:
            NotImplementedError(f"stage {self.config.stage} not implemented")
