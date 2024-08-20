# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Dict
import torch
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.loops.fetchers import _DataFetcherWrapper

from nemo.collections.nlp.models.machine_translation.megatron_nmt_model import MegatronNMTModel
from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
    AdapterName,
    PromptEncoderAdapterConfig,
)
from nemo.collections.nlp.modules.common.megatron.utils import (
    build_position_ids,
    init_method_normal,
)
from nemo.collections.nlp.modules.common.text_generation_utils import (
    get_sampling_token_fn,
)
from nemo.collections.nlp.modules.common.megatron.language_model import Embedding
from nemo.collections.nlp.modules.common.megatron.vocab_parallel_cross_entropy import vocab_parallel_cross_entropy
from nemo.utils import AppState, logging

from nemo.collections.nlp.models.machine_translation.megatron_nmt_model import MegatronNMTModel
from nemo.collections.nlp.modules.common.megatron.token_level_encoder_decoder import MegatronTokenLevelEncoderDecoderModule
from nemo.collections.nlp.modules.common.megatron.megatron_encoder_decoder import MegatronTransformerEncoderDecoderModule

try:
    from apex.transformer.pipeline_parallel.utils import (
        _reconfigure_microbatch_calculator,
        get_micro_batch_size,
        get_num_microbatches,
    )

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

try:
    from megatron.core import parallel_state, tensor_parallel, ModelParallelConfig
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False



class MegatronNMTMultiProjModel(MegatronNMTModel):
    """
    Megatron NMT training
    """
    
    def model_provider_func(self, pre_process, post_process, add_encoder, add_decoder):
        if not hasattr(self.cfg, 'encoder') or not hasattr(self.cfg, 'decoder'):
            logging.warning(
                'Could not find encoder or decoder in config. This is probably because of restoring an old checkpoint. Copying shared model configs to encoder and decoder configs.'
            )
            # After the call below, self.cfg.encoder and self.cfg.decoder will be populated with the cfg.model configs from old checkpoints.
            self._populate_encoder_decoder_configs_for_backward_compatibility(self.cfg)

        if parallel_state.get_pipeline_model_parallel_world_size() > 1 and self.cfg.encoder.arch == 'perceiver':
            raise ValueError(f"Perceivers with pipeline parallel > 1 is not supported yet.")

        if not hasattr(self.cfg, 'embedding_init_method_std'):
            embedding_init_method_std = self.cfg.encoder.init_method_std
        else:
            embedding_init_method_std = self.cfg.embedding_init_method_std

        if not hasattr(self.cfg, 'embedding_dropout'):
            embedding_dropout = self.cfg.encoder.hidden_dropout
        else:
            embedding_dropout = self.cfg.embedding_dropout
        
        if not hasattr(self.cfg, 'n_proj_heads'):
            self.n_proj_heads = 1
        else:
            self.n_proj_heads = self.cfg.n_proj_heads
        if not hasattr(self.cfg, 'proj_head_dims'):
            self.proj_head_dims = [self.padded_vocab_size]
        else:
            self.proj_head_dims = self.cfg.proj_head_dims
        if not hasattr(self.cfg, 'proj_head_loss_weights'):
            self.proj_head_loss_weights = [self.padded_vocab_size]
        else:
            self.proj_head_loss_weights = self.cfg.proj_head_loss_weights

        model = MegatronTokenLevelEncoderDecoderMultiProjModule(
            config=self.model_parallel_config,
            encoder_cfg=self.cfg.encoder,
            decoder_cfg=self.cfg.decoder,
            vocab_size=self.padded_vocab_size,
            max_position_embeddings=self.cfg.max_position_embeddings,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
            fp16_cross_entropy=self.cfg.get('fp16_lm_cross_entropy', False),
            precision=self.cfg.get('precision', 16),
            embedding_init_method_std=embedding_init_method_std,
            embedding_dropout=embedding_dropout,
            label_smoothing=self.cfg.get('label_smoothing', 0.0),
            add_encoder=add_encoder,
            add_decoder=add_decoder,
            share_token_embeddings=self.cfg.get('share_token_embeddings', True),
            share_decoder_tokens_head_embeddings=True,
            tokens_head_bias=self.cfg.get('tokens_head_bias', True),
            hiddens_cfg=self.cfg.get('hiddens', None),
            n_proj_heads=self.n_proj_heads,
            proj_head_dims=self.proj_head_dims,
            proj_head_loss_weights=self.proj_head_loss_weights,
        )
        return model

    def decode(
        self,
        tokens_enc,
        enc_mask,
        num_tokens_to_generate,
        encoder_input=None,
        tokenizer=None,
        enc_output=None,
        enc_output_attn_mask=None,
        ignore_ids=[],
        bos_id=None,  # If bos=None, will use tokenizer.bos_id unless explicitly set to something else.
        predicted_tokens_dec=None,
        batch_data=None,
        sampling_method: str = "greedy-search",
    ):
        """
        Args:
            tokens_enc: a tensor of shape [batch_size, seq_len] that contains the input tokens.
            enc_mask: a tensor of shape [batch_size, seq_len] that contains the input tokens mask (1 for active, 0 for inactive).
            num_tokens_to_generate: the max number of tokens to generate.
            encoder_input: a tensor of shape [batch_size, seq_len, hidden_size] that contains the encoder hidden states (replaces tokens_enc if given).
            tokenizer: a tokenizer object.
            enc_output: a tensor of shape [batch_size, seq_len, hidden_size] that contains the encoder hidden states (replaces tokens_enc and encoder_input if given).
            enc_output_attn_mask: a tensor of shape [batch_size, seq_len] that contains the encoder attention mask (replaces enc_mask if given).
            ignore_ids: a list of token ids to ignore when sampling.
            bos_id: the id of the beginning of sentence token. If None, will use tokenizer.bos_id unless explicitly set to something else.
            predicted_tokens_dec: a tensor of shape [batch_size, seq_len] that contains the tokens that have already been decoded.
            sampling_method: a sampling method to use in the decoding iterations. Currently supported methods is "greedy-search"

        Returns:
            tuple of tensors [batch_size, seq_len +1], [batch_size, seq_len] for predicted tokens and their log probs.
        """
        # Setting up the sampling strategy
        sample_token_fn, sampling_kwargs = get_sampling_token_fn(sampling_method, {})
        logging.info(f'Decoding using the {sampling_method} method...')

        # Check whether the DDP is initialized. This is needed when running inference outside of training loop.
        if not parallel_state.model_parallel_is_initialized():

            def dummy():
                return

            if self.trainer.strategy.launcher is not None:
                self.trainer.strategy.launcher.launch(dummy, trainer=self.trainer)
            self.trainer.strategy.setup_environment()

            # Reconfigure microbatch sizes here because on model restore, this will contain the micro/global batch configuration used while training.
            _reconfigure_microbatch_calculator(
                rank=0,  # This doesn't matter since it is only used for logging
                rampup_batch_size=None,
                global_batch_size=1,
                micro_batch_size=1,  # Make sure that there is no "grad acc" while decoding.
                data_parallel_size=1,  # We check above to make sure that dataparallel size is always 1 at inference.
            )

        # If classes that inherit from this class are using a different tokenizer,
        tokenizer = self.tokenizer if tokenizer is None else tokenizer
        app_state = AppState()
        if tokens_enc is not None:
            global_batch_per_gpu = tokens_enc.size(0)
            device = tokens_enc.device
            encoder_seq_length = tokens_enc.size(1)
        elif encoder_input is not None:
            global_batch_per_gpu = encoder_input.size(0)
            device = encoder_input.device
            encoder_seq_length = encoder_input.size(1)
        else:
            global_batch_per_gpu = enc_output.size(0)
            device = enc_output.device
            encoder_seq_length = enc_output.size(1)

        num_micro_batches_before_decode = get_num_microbatches()
        # Reconfigure microbatch calculator here to set num microbatches to 1 while decoding since its not clear how to decode with "grad acc".
        # reconfigure back to how things were before decode
        # TODO: Check if the user is trying to do gradient acc and maybe throw error
        _reconfigure_microbatch_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=global_batch_per_gpu * parallel_state.get_data_parallel_world_size(),
            micro_batch_size=global_batch_per_gpu,  # Make sure that there is no "grad acc" while decoding.
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )
        # TODO: Figure out how to handle bos being either <bos> for NeMo-Megatron and <pad> for Huggingface/Google.
        bos_id = tokenizer.bos_id if bos_id is None else bos_id
        # initial prompt can be given
        if predicted_tokens_dec is None:
            predicted_tokens_dec = torch.LongTensor([bos_id] * global_batch_per_gpu).unsqueeze(1).to(device)
        # collect log probs that were used in the sampling
        predicted_log_probs = torch.zeros((global_batch_per_gpu, 0, self.n_proj_heads), dtype=self.autocast_dtype).to(device)

        tensor_shape = [encoder_seq_length, global_batch_per_gpu, self.cfg.encoder.hidden_size]
        assert predicted_tokens_dec.size(0) == global_batch_per_gpu

        # get encoder hiddens (output)
        if enc_output is None:
            # Encode returns a tensr of shape [batch, seq_len, hidden]
            # All ranks will call `.encode()`, but only the last rank will have a non-empty output tensor.
            enc_output = self.encode(
                tokens_enc=tokens_enc, enc_mask=enc_mask, encoder_input=encoder_input, reconfigure_microbatch=False
            )
        if enc_output_attn_mask is None:
            enc_output_attn_mask = enc_mask

        for i in range(num_tokens_to_generate):
            # No microbatches in decoding. Just the global batch.
            decoder_seq_length = predicted_tokens_dec.size(1)
            dec_mask = predicted_tokens_dec[:, :, 0] != tokenizer.pad_id
            dec_mask[:, 0] = 1  # Make sure you never mask the first token even if it is <pad>.

            batch_for_pipeline = [enc_output, enc_output_attn_mask, predicted_tokens_dec, dec_mask, batch_data]
            arg_names = ['enc_output', 'enc_output_attn_mask', 'dec_input_ids', 'dec_attn_mask', 'batch_data']

            forward_step_func = self._get_forward_output_only_func(arg_names=arg_names, output_name=["logits", "attention_probs"], return_all_selfattention_probs=True, return_all_crossattention_probs=True)
            fwd_bwd_func = get_forward_backward_func()

            output_tensor = fwd_bwd_func(
                forward_step_func=forward_step_func,
                data_iterator=iter(
                    [
                        batch_for_pipeline,
                    ]
                ),
                model=[self.enc_dec_model],
                forward_only=True,
                num_microbatches=1,
                seq_length=encoder_seq_length,
                decoder_seq_length=encoder_seq_length,
                micro_batch_size=get_micro_batch_size(),
            )
            
            # get output tensor
            if parallel_state.is_pipeline_last_stage():
                output_tensor, attention_probs = output_tensor[0]['logits'], output_tensor[0]['attention_probs']
                output_tensor = tensor_parallel.gather_from_tensor_model_parallel_region(output_tensor)
                # make sure it won't sample outside the vocab_size range
                # ignore selected indices
                if ignore_ids:
                    output_tensor = output_tensor.index_fill(
                        dim=-1, index=torch.tensor(ignore_ids, device=output_tensor.device), value=-float('Inf')
                    )

                all_log_probs, all_token_ids = [], []
                for i in range(output_tensor.shape[-1]):
                    log_probs, token_ids = sample_token_fn(logits=output_tensor[:, -1, :, i])
                    # enforce valid range of token ids
                    all_log_probs.append(log_probs)
                    all_token_ids.append(token_ids)
                all_log_probs = torch.stack(all_log_probs, axis=-1)
                all_token_ids = torch.stack(all_token_ids, axis=-1)

                # collect all predicted tokens and log_probs
                predicted_tokens_dec = torch.cat(
                    [predicted_tokens_dec.to(token_ids.device), all_token_ids.unsqueeze(1)], dim=1
                )
                predicted_log_probs = torch.cat(
                    [predicted_log_probs.to(log_probs.device), all_log_probs.unsqueeze(1)], dim=1
                )

            else:
                predicted_tokens_dec = torch.zeros(
                    (predicted_tokens_dec.shape[0], predicted_tokens_dec.shape[1] + 1),
                    dtype=predicted_tokens_dec.dtype,
                ).cuda()
                predicted_log_probs = torch.zeros(
                    (predicted_log_probs.shape[0], predicted_log_probs.shape[1] + 1), dtype=self.autocast_dtype
                ).cuda()

            if self.cfg.get('pipeline_model_parallel_size', 1) > 1:
                # Broadcast from the last pipeline stage to all other model-parallel ranks.
                torch.distributed.broadcast(
                    predicted_tokens_dec,
                    parallel_state.get_pipeline_model_parallel_last_rank(),
                    group=parallel_state.get_pipeline_model_parallel_group(),
                )
                torch.distributed.broadcast(
                    predicted_log_probs,
                    parallel_state.get_pipeline_model_parallel_last_rank(),
                    group=parallel_state.get_pipeline_model_parallel_group(),
                )

        # Reset microbatch calculator to what it was before decoding.
        _reconfigure_microbatch_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=global_batch_per_gpu * parallel_state.get_data_parallel_world_size(),
            micro_batch_size=global_batch_per_gpu // num_micro_batches_before_decode,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )

        return predicted_tokens_dec, predicted_log_probs, attention_probs

    def complete(self, request: Dict):
        """
        Autoregressively invokes language model in the inference mode

        Args:
            request: Dictionary with the following fields
                * prompt: a string which text the model should complete.
                * tokens_to_generate: how many tokens to generate while doing prompt completion.

        Returns:
            response: A python dictionary with the following fields
                * prompt: original text of the prompt
                * tokenized_prompt: list of (str) tokens from prompt
                * completion: a python dictionary with the following subfields:
                    * tokens: a list of triples (token, token_id, log_prob) comprising completion
                    * text: completion text (as a single string)

        """
        app_state = AppState()

        # The complete method only works with global batch = micro batch size = data parallel size = 1.
        _reconfigure_microbatch_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=1,
            micro_batch_size=1,
            data_parallel_size=1,
        )
        app_state = AppState()

        response = {}
        self.freeze()
        # naive greedy slow loop

        response['prompt'] = request['prompt'][0]
        response['completion'] = {}
        bos_id = request['bos_id']
        tokens_enc = request['masked_sample']

        response['masked_input'] = ' '.join(self.tokenizer.ids_to_tokens(tokens_enc[0].cpu().numpy().tolist()))
        enc_mask = tokens_enc != self.tokenizer.pad_id

        predicted_tokens_ids, log_probs = self.decode(
            tokens_enc, enc_mask, int(request['tokens_to_generate']), bos_id=bos_id
        )
        predicted_tokens_ids = predicted_tokens_ids.cpu().numpy()[0].tolist()
        log_probs = log_probs.cpu().numpy()[0].tolist()
        if self.tokenizer.eos_id in predicted_tokens_ids:
            idx = predicted_tokens_ids.index(self.tokenizer.eos_id)
            predicted_tokens_ids = predicted_tokens_ids[:idx]
        else:
            predicted_tokens_ids = [id for id in predicted_tokens_ids if id != self.tokenizer.pad_id]
        if self.tokenizer.eos_id in predicted_tokens_ids:
            idx = predicted_tokens_ids.index(self.tokenizer.eos_id)
            predicted_tokens_ids = predicted_tokens_ids[:idx]
        # Legacy sentencepiece detokenization still preserves special tokens which messes up exact string match.
        if hasattr(self.tokenizer, 'special_token_to_id'):
            predicted_tokens_ids = [
                id for id in predicted_tokens_ids if id not in self.tokenizer.special_token_to_id.values()
            ]

        predicted_tokens_dec = self.tokenizer.ids_to_tokens(predicted_tokens_ids)
        response['completion']['text'] = self.tokenizer.tokens_to_text(predicted_tokens_dec)
        response['completion']['tokens'] = list(zip(predicted_tokens_ids, predicted_tokens_dec, log_probs))
        self.unfreeze()
        return response

    def _get_forward_output_only_func(self, arg_names, output_name, **kwargs):
        """
        args_idx - maps batch into index of args (with None filling gaps)
        arg_names - corresponding names for a friendly error message
        output_name - name of output (hiddens for encode, logits for decode)
        kwargs - shared arguments (non tensors)
        """

        def fwd_output_only_func(dataloader_iter, model):
            # Extract batch, batch_idx, dataloader_idx only if dataloader_iter is an object of PTL's _DataFetcherWrapper
            if isinstance(dataloader_iter, _DataFetcherWrapper):
                batch, _, _ = next(dataloader_iter)
            else:
                batch = next(dataloader_iter)
            batch = [x.cuda(non_blocking=True) if torch.is_tensor(x) else x for x in batch]

            # map batch and shared args into forward args
            args = self._build_forward_args_from_kwargs(args_name=arg_names, args=batch, **kwargs)
            output = model(*args)
            if isinstance(output, tuple):
                # The first item in output should be the major output of the autoregressive model
                output = (output[0].contiguous(), *output[1:])
            else:
                output = model(*args).contiguous()

            def id_func(output_tensor):
                if isinstance(output_tensor, dict):
                    # handle loss of hidden transformations ("output" is the default output)
                    output_tensor = output_tensor["output"]
                
                if isinstance(output_name, list):
                    assert isinstance(output_tensor, tuple)
                    assert len(output_name) == len(output_tensor), "the number of items in output_tensor and output_name does not match"
                    output_dict = {name: value for name, value in zip(output_name, output_tensor)}
                else:
                    output_dict = {output_name: output_tensor}

                return output_tensor[0], output_dict

            return output, id_func

        return fwd_output_only_func

class MegatronTokenLevelEncoderDecoderMultiProjModule(MegatronTokenLevelEncoderDecoderModule):
    """Token-based (input/output is tokens) encoder-decoder model (e.g. T5 Language model.)"""

    def __init__(
        self,
        config: ModelParallelConfig,
        encoder_cfg: DictConfig,
        decoder_cfg: DictConfig,
        vocab_size: int,  # TODO: This should eventually go inside encoder_cfg and decoder_cfg when separate enc/dec tokenizers are supported.
        max_position_embeddings,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=True,
        post_process=True,
        fp16_cross_entropy=False,
        precision=16,
        embedding_init_method_std=0.02,
        embedding_dropout=0.1,
        label_smoothing=0.0,
        add_encoder=True,
        add_decoder=True,
        share_token_embeddings=True,
        share_decoder_tokens_head_embeddings=True,
        tokens_head_bias=True,
        hiddens_cfg: DictConfig = None,  # allows for hidden state transformations before the decoder
        n_proj_heads=1,
        proj_head_dims=[64000],
        proj_head_loss_weights=[1],
    ):
        super().__init__(
            config,
            encoder_cfg,
            decoder_cfg,
            vocab_size,
            max_position_embeddings,
            num_tokentypes,
            parallel_output,
            pre_process,
            post_process,
            fp16_cross_entropy,
            precision,
            embedding_init_method_std,
            embedding_dropout,
            label_smoothing,
            add_encoder,
            add_decoder,
            share_token_embeddings,
            share_decoder_tokens_head_embeddings,
            tokens_head_bias,
            hiddens_cfg,
        )

        self.n_proj_heads = n_proj_heads
        self.proj_head_dims = proj_head_dims
        self.proj_head_loss_weights = proj_head_loss_weights
        assert self.proj_head_dims[0] == vocab_size

        if add_decoder and post_process:
            self.tokens_heads = torch.nn.ModuleList([
                tensor_parallel.ColumnParallelLinear(
                    input_size=decoder_cfg.hidden_size,
                    output_size=proj_head_dims[i],
                    config=config,
                    bias=tokens_head_bias,
                    gather_output=not self.parallel_output,
                    init_method=init_method_normal(decoder_cfg.init_method_std),
                ) for i in range(self.n_proj_heads)
            ])

            self._tokens_head_key = 'tokens_heads'
        self.set_accepted_adapter_types([PromptEncoderAdapterConfig._target_])

    def forward(
        self,
        enc_input_ids=None,
        enc_attn_mask=None,
        dec_input_ids=None,
        dec_attn_mask=None,
        token_type_ids=None,
        labels=None,
        batch_data=None,  # additional data to be passed to hiddens module
        enc_output=None,  # Result of running the entire encoder
        enc_output_attn_mask=None,
        enc_input=None,  # Result of running encoder embedding only
        output_enc_hidden_only=False,
        return_all_selfattention_probs=False,
        return_all_crossattention_probs=False,
    ):
        """
        Return value is per token / per dimension (i.e., non collapsed loss value)
        """
        (
            encoder_self_attention_relative_position_bias,
            decoder_self_attention_relative_position_bias,
            decoder_cross_attention_relative_position_bias,
        ) = (None, None, None)

        if enc_input is not None and enc_output is not None:
            raise ValueError(
                """Both enc_input and enc_output are not None.
                You should only be passing one of them.
                enc_input is the result of the encoder embedding layer
                enc_output is the result of running the entire transformer encoder."""
            )

        # In order of precedence, we use enc_output, enc_input, and then enc_input_ids to determine the encoder sequence length.
        if enc_output is not None:
            # If enc_output is provided in `batch_for_pipeline`, we need to transpose it from [B x S x H] -> [S x B x H].
            enc_output = enc_output.transpose(0, 1)
            enc_seq_length = enc_output.size(0)
        elif enc_input is not None:
            # If enc_input is provided, we need to transpose it from [B x S x H] -> [S x B x H].
            enc_input = enc_input.transpose(0, 1)
            enc_seq_length = enc_input.size(0)
        # Only need to run encoder embedding and position ids if enc_input or enc_output is not provided.
        elif enc_input_ids is not None:
            enc_seq_length = enc_input_ids.size(1)
            if self.pre_process and self.add_encoder:
                # We don't need position ids for RPE, because the embedding layer does not have position embeddings.
                if self.encoder_relative_position_embedding is None:
                    enc_position_ids = build_position_ids(enc_input_ids)
                else:
                    enc_position_ids = None
                enc_input = self.encoder_embedding(enc_input_ids, enc_position_ids, token_type_ids=token_type_ids)
                if self.is_adapter_available():
                    _sq, _bs, _hs = enc_input.size()
                    ptuning_adapter = self.get_adapter_module(AdapterName.PTUNING_ADAPTER)
                    v = ptuning_adapter.virtual_tokens
                    if (
                        ptuning_adapter and self.adapter_cfg[AdapterName.PTUNING_ADAPTER]['enabled'] and _sq >= v
                    ):  # The sequence should be longer the v to insert virtual embeddings.
                        virtual_embeddings = ptuning_adapter(_bs)
                        enc_input = enc_input[
                            v:, :, :
                        ]  # the first v tokens are pads so that they can be swapped out with virtual embeddings.
                        enc_input = torch.concat([virtual_embeddings, enc_input], dim=0)
            else:
                enc_input = None
        else:
            # This should only happen with PP > 1 for enc-dec prompt learning models
            enc_seq_length = enc_attn_mask.size(1)

        if self.add_encoder and self.encoder_relative_position_embedding is not None:
            encoder_self_attention_relative_position_bias = self.encoder_relative_position_embedding(
                query_seq_length=enc_seq_length, key_seq_length=enc_seq_length,
            )

        if output_enc_hidden_only:
            # When pipeline parallel > 1 we need to make sure encoder exist (will be missing in decoder)
            if enc_output is None and self.enc_dec_model.encoder is not None:
                enc_output = self.enc_dec_model.encode(
                    enc_input=enc_input,
                    enc_attn_mask=enc_attn_mask,
                    enc_layer_past=None,
                    enc_get_key_value=False,
                    enc_self_attention_relative_position_bias=encoder_self_attention_relative_position_bias,
                    batch_data=batch_data,
                )
            else:
                enc_output = self.enc_dec_model.encoder_hidden_state

            return enc_output
        else:
            if enc_output_attn_mask is None:
                enc_output_attn_mask = enc_attn_mask

            if self.pre_process and self.add_decoder:
                # We don't need position ids for RPE, because the embedding layer does not have position embeddings.
                if self.decoder_relative_position_embedding is None:
                    dec_position_ids = build_position_ids(dec_input_ids)
                else:
                    dec_position_ids = None
                dec_input = self.decoder_embedding(dec_input_ids, dec_position_ids, token_type_ids=token_type_ids)
            else:
                # Note: This is when the decoder itself is split across PP ranks.
                dec_input = None

            if self.add_decoder and self.decoder_relative_position_embedding is not None:
                decoder_self_attention_relative_position_bias = self.decoder_relative_position_embedding(
                    query_seq_length=dec_input_ids.size(1), key_seq_length=dec_input_ids.size(1)
                )
                if not self.decoder_cfg.relative_position_bias_self_attention_only:
                    decoder_cross_attention_relative_position_bias = self.decoder_cross_attention_relative_position_embedding(
                        query_seq_length=dec_input_ids.size(1), key_seq_length=enc_seq_length,
                    )
                else:
                    decoder_cross_attention_relative_position_bias = None
            
            output = self.enc_dec_model(
                enc_input=enc_input,
                enc_attn_mask=enc_attn_mask,
                dec_input=dec_input,
                dec_attn_mask=dec_attn_mask,
                enc_layer_past=None,
                enc_get_key_value=False,
                enc_output=enc_output,
                enc_output_attn_mask=enc_output_attn_mask,
                dec_layer_past=None,
                dec_get_key_value=False,
                enc_self_attention_relative_position_bias=encoder_self_attention_relative_position_bias,
                dec_self_attention_relative_position_bias=decoder_self_attention_relative_position_bias,
                dec_cross_attention_relative_position_bias=decoder_cross_attention_relative_position_bias,
                batch_data=batch_data,
                return_all_selfattention_probs=return_all_selfattention_probs,
                return_all_crossattention_probs=return_all_crossattention_probs,
            )

            if self.post_process and self.add_decoder:
                dec_output, enc_output = output  # [s, b, h], enc_output might be a dict if hiddens_module is used
                if return_all_selfattention_probs or return_all_crossattention_probs:
                    dec_output, self_attention_scores, cross_attention_scores = dec_output

                    def post_process_attention_scores(attention_scores):
                        if len(attention_scores) == 0:
                            return None
                        attention_probs = [torch.softmax(attention_score, dim=-1) for attention_score in attention_scores]
                        attention_scores_averaged = torch.mean(torch.cat(attention_probs, dim=1), dim=1)
                        # text_start_idx = text_limits[0, 0].item()
                        # assert torch.all(
                        #     text_limits[:, 0] == text_start_idx
                        # )  # all texts should start at the same index
                        # end_offset = self.alignment_text_end_offset
                        # # align_every_n_head: eg if set to 2, will skip every other head
                        # # if set to 12, will select 1 head from every layer
                        # align_every_n_head = self.align_every_n_head
                        # dec_start_idx = self.decoder_context_len + 1  # +1 to remove bos
                        # attention_scores_sliced = attention_scores_combined[
                        #     :, ::align_every_n_head, dec_start_idx:, text_start_idx : -(2 + end_offset)
                        # ]  # -2 to remove eos and pad
                        # attention_logprobs = (
                        #     attention_scores_sliced  # not taking log_softmax, since we will do that in loss function
                        # )
                        # attention_logprobs = torch.mean(attention_logprobs, dim=1, keepdim=True)
                        # dec_len = torch.sum(dec_attn_mask, dim=1) - dec_start_idx
                        # enc_len = text_limits[:, 1] - text_limits[:, 0] - end_offset
                        # alignment_loss = self.forward_sum_loss(
                        #     attn_logprob=attention_logprobs, in_lens=enc_len, out_lens=dec_len
                        # )
                        return attention_scores_averaged
                    
                    self_attention_scores = post_process_attention_scores(self_attention_scores)
                    cross_attention_scores = post_process_attention_scores(cross_attention_scores)
                    attention_probs = [self_attention_scores, cross_attention_scores]
                else:
                    attention_probs = [None, None]

                # project decoder output to vocabulary-size dimensions
                token_logits = [self.tokens_heads[i](dec_output)[0] for i in range(self.n_proj_heads)]
                if self.share_decoder_tokens_head_embeddings:
                    token_logits[0] = self.tokens_head(dec_output, self.word_embeddings_weight())
                else:
                    token_logits[0] = self.tokens_head(dec_output)[0]

                if labels is not None:
                    # compute loss here
                    # [b, s] -> [s, b]
                    labels = labels.transpose(0, 1).contiguous()

                    # Set label smoothing to 0 if in eval mode.
                    label_smoothing = self.label_smoothing if self.training else 0.0

                    # tensor_parallel.vocab_parallel_cross_entropy performs log_softmax and return log p(x_i|z) per token i
                    if self.fp16_cross_entropy:
                        assert token_logits.dtype == torch.half
                        tokens_loss = torch.stack([
                            vocab_parallel_cross_entropy(token_logits[i], labels[:, :, i]-sum(self.proj_head_dims[:i]), label_smoothing) for i in range(self.n_proj_heads)
                        ], axis=2)
                    else:
                        tokens_loss = torch.stack([
                            vocab_parallel_cross_entropy(token_logits[i].float(), labels[:, :, i]-sum(self.proj_head_dims[:i]), label_smoothing) for i in range(self.n_proj_heads)
                        ], axis=2)
                    
                    # import pdb; pdb.set_trace()

                    # [s, b] -> [b, s]
                    tokens_loss = tokens_loss.transpose(0, 1).contiguous()
                    tokens_loss = tokens_loss * torch.FloatTensor(self.proj_head_loss_weights).to(tokens_loss.device) / sum(self.proj_head_loss_weights)
                    # check if hiddens is used
                    if return_all_selfattention_probs or return_all_crossattention_probs:
                        return tokens_loss, attention_probs
                    else:
                        return tokens_loss
                else:
                    # else return token logits (and hiddens if needed)
                    token_logits_blank = torch.full(
                        (*token_logits[0].shape[:2], sum(self.proj_head_dims), len(token_logits)),
                        -float('Inf'), 
                        device=token_logits[0].device
                    )
                    for i in range(len(token_logits)):
                        token_logits_blank[:, :, sum(self.proj_head_dims[:i]):sum(self.proj_head_dims[:i+1]), i] = token_logits[i]
                    # [s, b, h, heads] -> [b, s, h, heads]
                    token_logits = token_logits_blank.transpose(0, 1).contiguous()
                    if return_all_selfattention_probs or return_all_crossattention_probs:
                        return token_logits, attention_probs
                    else:
                        return token_logits

            elif self.add_decoder and not self.add_encoder:
                decoder_output, _ = output
                if return_all_selfattention_probs or return_all_crossattention_probs:
                    decoder_output, self_attention_scores, cross_attention_scores = decoder_output
                return decoder_output
            else:
                encoder_output = output
                return encoder_output

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        self.encoder_embedding.encoder_embeddingload_state_dict(state_dict[self._encoder_embedding_key], strict=strict)
        self.decoder_embedding.load_state_dict(state_dict[self._decoder_embedding_key], strict=strict)
        self.enc_dec_model.load_state_dict(state_dict[self._enc_dec_model_key], strict=strict)
        self.tokens_heads.load_state_dict(state_dict[self._tokens_head_key], strict=False)
        self.tokens_head.load_state_dict(state_dict['tokens_head'], strict=strict)


class SumMultiEmbedding(Embedding):
    """Language model embeddings with multiple tokens at each time step. The embeddings of the tokens of the same time step will be computed separately and then be summed together.
    """
    def __init__(
        self,
        config: ModelParallelConfig,
        hidden_size,
        orig_vocab_size,
        vocab_size,
        max_sequence_length,
        embedding_dropout_prob,
        init_method,
        num_tokentypes=0,
        fp32_residual_connection=False,
        position_embedding_type='learned_absolute',
        transpose_batch_sequence=True,
    ):
        super().__init__(config,
            hidden_size,
            vocab_size,
            max_sequence_length,
            embedding_dropout_prob,
            init_method,
            num_tokentypes,
            fp32_residual_connection,
            position_embedding_type,
            transpose_batch_sequence,
        )
        self.word_embeddings._parameters['weight'][orig_vocab_size:].data.zero_()

    # def add_tokentype_embeddings(self, num_tokentypes):
    #     """Add token-type embedding. This function is provided so we can add
    #     token-type embeddings in case the pretrained model does not have it.
    #     This allows us to load the model normally and then add this embedding.
    #     """
    #     if self.tokentype_embeddings is not None:
    #         raise Exception('tokentype embeddings is already initialized')
    #     if torch.distributed.get_rank() == 0:
    #         print('adding embedding for {} tokentypes'.format(num_tokentypes), flush=True)
    #     self.num_tokentypes = num_tokentypes
    #     self.tokentype_embeddings = torch.nn.Embedding(num_tokentypes, self.hidden_size)
    #     # Initialize the token-type embeddings.
    #     self.init_method(self.tokentype_embeddings.weight)

    def forward(self, input_ids, position_ids=None, token_type_ids=None):
        # import pdb; pdb.set_trace()
        embeddings = super().forward(input_ids, position_ids, token_type_ids)
        return torch.sum(embeddings, axis=2)