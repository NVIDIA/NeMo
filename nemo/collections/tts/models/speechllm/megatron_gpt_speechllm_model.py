import re
import numpy as np
import soundfile as sf

import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.tts.data.speechllm.t5_speechlm_dataset import GPTSpeechLMDataset
from nemo.collections.tts.data.speechllm.t5_speechlm_tarred_dataset import GPTSpeechLMTarredDataset
from nemo.collections.nlp.modules.common import VirtualPromptPlaceholderToken, VirtualPromptSource
from nemo.collections.tts.parts.utils.helpers import plot_alignment_to_numpy, plot_encodec_to_numpy
from nemo.utils.app_state import AppState
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel


class MegatronSpeechGPTModel(MegatronGPTModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)
        if self.cfg.get('megatron_amp_O2', False):
            base_module = self.model.module
        else:
            base_module = self.model
        hidden_size = base_module.hidden_size
        # base_module.speech_residual_model = None

        app_state = AppState()
        self.should_log = app_state.global_rank == 0
        if self.should_log:
            encodec_model = EncodecModel.encodec_model_24khz()
            encodec_model.set_target_bandwidth(6.0)
            encodec_model.cuda()
            encodec_model.eval()
            self.additional_models = {'encodec': encodec_model}
        self.pretraining = True
        self.return_all_selfattention_probs = self.cfg.get('return_all_selfattention_probs', False)
        self.train_check_interval = self.cfg.get('train_check_interval', 1500)
        # TODO: pass these down to language_model.py
        # return_all_crossattention_probs = cfg.get('return_all_crossattention_probs', False)
        # num_cross_attention_heads = cfg.get('num_cross_attention_heads', 12)

    def get_gpt_module_list(self):
        if isinstance(self.model, list):
            return [
                model.module if isinstance(model, (Float16Module, MCoreFloat16Module)) else model
                for model in self.model
            ]
        elif isinstance(self.model, (Float16Module, MCoreFloat16Module)):
            return [self.model.module]
        else:
            return [self.model]

    def get_forward_output_and_loss_func(self, validation_step=False, tuning=False):
        def fwd_output_and_loss_func(dataloader_iter, model, checkpoint_activations_all_layers=None):

            # Get data batch
            batch = self.get_batch(dataloader_iter, tuning)

            # Transfer needed data to GPU
            required_keys = set()
            max_seqlen = batch.pop('max_seqlen').squeeze() if 'max_seqlen' in batch else None
            cu_seqlens_argmin = batch.pop('cu_seqlens_argmin') if 'cu_seqlens_argmin' in batch else None
            if parallel_state.get_pipeline_model_parallel_world_size() == 1:
                required_keys.update(batch.keys())
            else:
                required_keys.add('attention_mask')
                if 'cu_seqlens' in batch:
                    required_keys.add('cu_seqlens')
                if parallel_state.is_pipeline_first_stage():
                    required_keys.update(('tokens', 'position_ids'))
                if parallel_state.is_pipeline_last_stage():
                    required_keys.update(('labels', 'loss_mask'))
            if self.get_attention_mask_from_fusion and 'attention_mask' in required_keys:
                required_keys.remove('attention_mask')
            if not self.cfg.get('use_attention_prior', False):
                required_keys.remove('attention_prior')
            batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in batch.items()}

            # slice batch along sequence dimension for context parallelism
            batch = self.get_batch_on_this_context_parallel_rank(batch)

            # Model forward pass
            forward_args = {
                'input_ids': batch['tokens'],
                'position_ids': batch['position_ids'],
                'attention_mask': None if self.get_attention_mask_from_fusion else batch['attention_mask'],
                'labels': batch['labels'],
                'loss_mask': batch['loss_mask'],
                'speech_mask': batch['speech_mask'],
                'return_logits': True,
                'return_all_selfattention_probs': self.return_all_selfattention_probs
                if not validation_step
                else False,
                'attention_prior': batch.get('attention_prior', None),
                'global_step': self.global_step,
                'context_question_mask': batch['context_question_mask'],
            }

            if not self.cfg.get('use_attention_prior', False):
                forward_args.pop('attention_prior')

            if not self.mcore_gpt:
                forward_args['checkpoint_activations_all_layers'] = checkpoint_activations_all_layers
                if not self.use_loss_mask:
                    forward_args.pop('loss_mask')
            else:
                # TODO: @eharper can we add this to mcore?
                forward_args.pop('loss_mask')
            (output_tensor, logits), attention_probs_list, prior = model(**forward_args)

            if (
                self.trainer.global_step % self.train_check_interval == 0
                and batch['speech_mask'][0].sum() != 0
                and self.should_log
                and (not validation_step)
            ):
                # Logs every if the first item in the batch is speech
                logging.info("Logging training audio")
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=False):
                        all_speech_logits = []
                        all_speech_token_preds = []
                        for _i in range(8):
                            vsi = self.cfg.get("text_size", self.tokenizer.vocab_size) + _i * 1024
                            layer_logits = logits[:, :, vsi : vsi + 1024]
                            all_speech_token_preds.append(layer_logits.argmax(dim=-1))
                            all_speech_logits.append(layer_logits)
                        all_speech_logits = torch.stack(all_speech_logits, dim=-1)  # (T, B, 1024, 8)
                        all_speech_token_preds = torch.stack(all_speech_token_preds, dim=-1)  # (T, B, 8)
                        speech_token_preds_example = all_speech_token_preds[:, 0, :].permute(1, 0)  # (8, T)
                        start_of_speech = (
                            0
                            if self.pretraining
                            else torch.count_nonzero(~batch["loss_mask"][0] * batch['tokens'][0][0]) + 2
                        )
                        speech_token_preds_example = self.convert_tokens_to_range(
                            speech_token_preds_example, start_of_speech=start_of_speech
                        )

                        input_tokens_example = batch['tokens'][0]

                        if not self.pretraining:
                            question_tokens = []
                            question_phoneme_tokens = []
                            question_start = 0
                            for _t in range(start_of_speech):
                                if input_tokens_example[0, _t] < self.tokenizer.vocab_size:
                                    question_tokens.append(input_tokens_example[0, _t].item())
                                elif (
                                    input_tokens_example[0, _t] >= self.tokenizer.vocab_size
                                    and input_tokens_example[0, _t] < self.cfg.text_size
                                ):
                                    question_phoneme_tokens.append(
                                        input_tokens_example[0, _t].item() - self.tokenizer.vocab_size
                                    )
                                elif len(question_tokens) == 0:
                                    question_start += 1

                            if len(question_tokens) > 0:
                                question_text = self.tokenizer.ids_to_text(question_tokens)
                                self.logger.experiment.add_text(
                                    'train_question_text', question_text, self.trainer.global_step
                                )
                            if len(question_phoneme_tokens) > 0:
                                phoneme_text = phoneme_tokenizer.decode(question_phoneme_tokens)
                                self.logger.experiment.add_text(
                                    'train_question_phonemetext', phoneme_text, self.trainer.global_step
                                )

                        input_tokens_example = self.convert_tokens_to_range(
                            input_tokens_example,
                            offset_first_layer=True,
                            offset_all_layers=True,
                            start_of_speech=start_of_speech,
                        )

                        labels_example = batch['labels'][0]
                        labels_example = self.convert_tokens_to_range(
                            labels_example,
                            offset_first_layer=True,
                            offset_all_layers=False,
                            start_of_speech=start_of_speech,
                        )

                        label_wav = self.additional_models['encodec'].decode([[labels_example[None], None]])[0, 0]
                        dec_input_wav = self.additional_models['encodec'].decode([[input_tokens_example[None], None]])[
                            0, 0
                        ]
                        pred_wav = self.additional_models['encodec'].decode(
                            [[speech_token_preds_example[None], None]]
                        )[0, 0]

                        self.logger.experiment.add_audio(
                            'train_label_wav', label_wav, self.trainer.global_step, sample_rate=24000
                        )
                        self.logger.experiment.add_audio(
                            'train_dec_input_wav', dec_input_wav, self.trainer.global_step, sample_rate=24000
                        )
                        self.logger.experiment.add_audio(
                            'train_tf_pred_wav', pred_wav, self.trainer.global_step, sample_rate=24000
                        )

                        # print(batch['tokens'][0, 0, question_start])
                        # print(batch['tokens'][0, 0, start_of_speech-1])
                        if attention_probs_list is not None and not self.cfg.get('use_flash_attention', False):
                            for lidx in range(len(attention_probs_list)):
                                attention_probs = attention_probs_list[lidx]
                                for _i in range(attention_probs.shape[1]):
                                    speech_size = batch["loss_mask"][0].shape[0]
                                    attention_probs_sliced = (
                                        attention_probs[0, _i, :speech_size, :speech_size].clone().detach()
                                    )
                                    attention_probs_sliced = attention_probs_sliced.T
                                    # attention_probs_sliced *= batch["loss_mask"][0]
                                    # attention_probs_sliced *= batch_cpu["attention_mask"][0][0,:,:].to(attention_probs_sliced.device)
                                    phoneme_seq = [question_start, start_of_speech.item() - 1]
                                    alignment_image_sliced = plot_alignment_to_numpy(
                                        # attention_probs_sliced.cpu().float().numpy().T, phoneme_seq=(batch['tokens'][0, 0, :] == 0).to(int).detach().cpu().numpy()
                                        attention_probs_sliced.cpu().float().numpy(),
                                        phoneme_seq=phoneme_seq,
                                        phoneme_ver=1,
                                        vmin=0.0,
                                        vmax=1.0,
                                    )
                                    self.logger.experiment.add_image(
                                        f"Attention Probs Layer {lidx} Head {_i}",
                                        alignment_image_sliced,
                                        self.global_step,
                                        dataformats="HWC",
                                    )

                if 'cu_seqlens' in batch:  # packed sequence from GPTSFTPackedDataset
                    # these args are passed eventually into TEDotProductAttention.forward()
                    cu_seqlens = batch['cu_seqlens'].squeeze()  # remove batch size dimension (mbs=1)
                    # remove -1 "paddings" added in collate_fn
                    if cu_seqlens_argmin is not None:
                        cu_seqlens = cu_seqlens[: cu_seqlens_argmin.item()]
                    else:
                        cu_seqlens = cu_seqlens[: torch.argmin(cu_seqlens)]

                    try:
                        from megatron.core.packed_seq_params import PackedSeqParams
                    except (ImportError, ModuleNotFoundError) as e:
                        mcore_version = packaging.version.Version(version('megatron-core'))
                        logging.error(
                            f"megatron-core v{mcore_version} does not support training with packed sequence. "
                            "Please use megatron-core >= 0.5.0, or set model.data.train_ds.packed_sequence=False"
                        )
                        raise e

                    forward_args['packed_seq_params'] = PackedSeqParams(
                        cu_seqlens_q=cu_seqlens,
                        cu_seqlens_kv=cu_seqlens,
                        max_seqlen_q=max_seqlen,
                        max_seqlen_kv=max_seqlen,
                        qkv_format='thd',
                    )

            output_tensor = model(**forward_args)

            def loss_func(output_tensor):
                # Loss for a micro-batch (ub)
                loss_for_ub = self.loss_func(batch['loss_mask'], batch['num_valid_tokens_in_ub'], output_tensor)
                cp_size = parallel_state.get_context_parallel_world_size()
                if validation_step and not self.cfg.data.get('validation_drop_last', True):
                    num_valid_tokens_in_ub = batch['num_valid_tokens_in_ub']
                    if loss_for_ub.isnan():
                        assert batch['loss_mask'].count_nonzero() == 0, 'Got NaN loss with non-empty input'
                        loss_sum_for_ub = torch.zeros_like(num_valid_tokens_in_ub)
                    else:
                        loss_sum_for_ub = num_valid_tokens_in_ub * loss_for_ub

                    loss_sum_and_ub_size_all_gpu = torch.cat(
                        [
                            loss_sum_for_ub.clone().detach().view(1),
                            torch.tensor([num_valid_tokens_in_ub]).cuda().clone().detach(),
                        ]
                    )
                    # Could potentially reduce num_valid_samples_in_microbatch and use that to aggregate instead of len(self._validation_ds)
                    torch.distributed.all_reduce(
                        loss_sum_and_ub_size_all_gpu, group=parallel_state.get_data_parallel_group()
                    )
                    return loss_for_ub * cp_size, {'loss_sum_and_ub_size': loss_sum_and_ub_size_all_gpu}
                else:
                    reduced_loss = average_losses_across_data_parallel_group([loss_for_ub])
                    return loss_for_ub * cp_size, {'avg': reduced_loss}

            return output_tensor, loss_func

        return fwd_output_and_loss_func

    def get_forward_output_only_func(self):
        """ Used in inference / generate """

        def fwd_output_only_func(dataloader_iter, model):
            batch = next(dataloader_iter)
            extra_arg = {}
            if len(batch) == 3:
                batch = [x.cuda() for x in batch]
                tokens, attention_mask, position_ids = batch
                attention_mask = attention_mask[0:1]
            elif len(batch) == 5:
                (
                    tokens,
                    attention_mask,
                    position_ids,
                    set_inference_key_value_memory,
                    inference_max_sequence_len,
                ) = batch
                tokens = tokens.cuda()
                position_ids = position_ids.cuda()
                if attention_mask is not None:
                    attention_mask = attention_mask.cuda()
                    attention_mask = attention_mask[0:1]
                if self.mcore_gpt:
                    # if first step, then clear KV cache, otherwise reuse inference_paarms
                    if set_inference_key_value_memory[0].item():
                        self.inference_params = InferenceParams(
                            max_batch_size=tokens.size(0), max_sequence_length=inference_max_sequence_len[0].item()
                        )
                    extra_arg['inference_params'] = self.inference_params
                else:
                    extra_arg['set_inference_key_value_memory'] = set_inference_key_value_memory[0].item()
                    extra_arg['inference_max_sequence_len'] = inference_max_sequence_len[0].item()
            else:
                (
                    tokens,
                    attention_mask,
                    position_ids,
                    set_inference_key_value_memory,
                    inference_max_sequence_len,
                    speech_mask,
                    # _  # Attention prior not used at inference / generate()
                ) = batch
                tokens = tokens.cuda()
                position_ids = position_ids.cuda()
                if attention_mask is not None:
                    attention_mask = attention_mask.cuda()
                    attention_mask = attention_mask[0:1]
                if self.mcore_gpt:
                    # if first step, then clear KV cache, otherwise reuse inference_paarms
                    if set_inference_key_value_memory[0].item():
                        self.inference_params = InferenceParams(
                            max_batch_size=tokens.size(0), max_sequence_length=inference_max_sequence_len[0].item()
                        )
                    extra_arg['inference_params'] = self.inference_params
                else:
                    extra_arg['set_inference_key_value_memory'] = set_inference_key_value_memory[0].item()
                    extra_arg['inference_max_sequence_len'] = inference_max_sequence_len[0].item()
                extra_arg['speech_mask'] = speech_mask
                # extra_arg['return_all_selfattention_probs'] = True
            output_tensor, attention_, prior = model(tokens, position_ids, attention_mask, **extra_arg)

            # Advance inference sequence offset.
            if self.inference_params:
                # if last stage, then (final) output is [b, s, h], otherwise it's [s, b, h]
                if parallel_state.is_pipeline_last_stage():
                    self.inference_params.sequence_len_offset += output_tensor.size(1)
                else:
                    self.inference_params.sequence_len_offset += output_tensor.size(0)

            def id_func(output_tensor):
                return 0, {'logits': output_tensor[0], 'speech_logits': output_tensor[1]}

            return output_tensor, id_func

        return fwd_output_only_func

    def generate(
        self,
        inputs: Union[List[str], torch.Tensor, List[dict]],
        length_params: LengthParam,
        sampling_params: SamplingParam = None,
        mode="teacher-forced",  # One of "teacher-forced", "greedy", "multinomial"
        *,
        strategy: Optional[TextGenerationStrategy] = None,
    ) -> OutputType:
        """
        inputs can either be a list of string or a tuple
        If list of string, will be tokenized in downstream func
        If tuple, must be a tuple of (tokenized_ids, context_length)
        """

        # check whether the DDP is initialized
        if parallel_state.is_unitialized():

            def dummy():
                return

            if self.trainer.strategy.launcher is not None:
                self.trainer.strategy.launcher.launch(dummy, trainer=self.trainer)
            self.trainer.strategy.setup_environment()

            if self.cfg.get('transformer_engine', False):
                self.setup_transformer_engine_tp_groups()
                self.setup_transformer_engine_cp_groups()

        # set the default sampling params if it is None.
        # default do greedy sampling
        if sampling_params is None:
            sampling_params = get_default_sampling_params()

        # set the default length params if it is None.
        # default do greedy sampling
        if length_params is None:
            length_params = get_default_length_params()

        strategy_args = {} if strategy is None else {"strategy": strategy}

        return megatron_gpt_generate(
            self.cuda(), inputs, self.tokenizer, length_params, sampling_params, mode=mode, **strategy_args
        )

    def convert_tokens_to_range(
        self, tokens, offset_first_layer=False, offset_all_layers=False, start_of_speech=0, delay_pattern=True
    ):
        # offset tokens to be in range [0, 1024] and convert delay parallel to parallel
        offset = self.cfg.data.get('speech_offset', self.tokenizer.vocab_size)
        output_tokens = tokens.clone()
        if offset_first_layer:
            output_tokens[0] = output_tokens[0] - offset

        output_tokens_new = []
        for _c in range(output_tokens.shape[0]):
            if delay_pattern:
                si = _c
                ei = _c + output_tokens.shape[1] - 8
            else:
                si = 0
                ei = output_tokens.shape[1]

            if offset_all_layers and _c > 0:
                output_tokens[_c, :] -= offset + _c * 1024
            if start_of_speech != 0:
                context_and_text = output_tokens[_c, :start_of_speech]
                speech = output_tokens[_c, start_of_speech + si : ei]
                context_text_speech = torch.cat([context_and_text, speech], dim=-1)
                output_tokens_new.append(context_text_speech)
            else:
                output_tokens_new.append(output_tokens[_c, si:ei])
        output_tokens_new = torch.stack(output_tokens_new)
        output_tokens = output_tokens_new
        output_tokens = torch.clamp(output_tokens, min=0, max=1023)

        return output_tokens

    def model_provider_func(self, pre_process, post_process):
        """Very small override of base model so we can have different embedding and output layer size"""
        # logging.info(f"AGAIN1 {self.cfg.get('override_vocab_size')}")
        # logging.info(f"AGAIN1 {self.cfg.get('output_size')}")
        # logging.info(f"AGAIN1 {self.cfg.get('embedding_scale')}")
        # logging.info(f"AGAIN1 {self.mcore_gpt}")
        if self.mcore_gpt:
            raise NotImplementedError("No mcore for speech")
        assert self.cfg.get('num_query_groups', None) is None or self.cfg.get(
            'num_query_groups', None
        ) == self.cfg.get(
            'num_attention_heads', None
        ), "Group Query Attention is only supported in Megatron Core. Set 'mcore_gpt' to use GQA."

        model = GPTModel(
            config=self.model_parallel_config,
            vocab_size=self.cfg.get('override_vocab_size', self.padded_vocab_size),
            hidden_size=self.cfg.hidden_size,
            max_position_embeddings=self.cfg.max_position_embeddings,
            num_layers=self.cfg.num_layers,
            num_attention_heads=self.cfg.num_attention_heads,
            apply_query_key_layer_scaling=self.cfg.get('apply_query_key_layer_scaling', True),
            kv_channels=self.cfg.get('kv_channels', None),
            ffn_hidden_size=self.cfg.ffn_hidden_size,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
            init_method_std=self.cfg.get('init_method_std', 0.02),
            use_scaled_init_method=self.cfg.get('use_scaled_init_method', True),
            fp16_lm_cross_entropy=self.cfg.get('fp16_lm_cross_entropy', False),
            megatron_amp_O2=self.cfg.get('megatron_amp_O2', False),
            hidden_dropout=self.cfg.get('hidden_dropout', 0.1),
            attention_dropout=self.cfg.get('attention_dropout', 0.1),
            ffn_dropout=self.cfg.get('ffn_dropout', 0.0),
            precision=self.cfg.get('precision', 16),
            fp32_residual_connection=self.cfg.get('fp32_residual_connection', False),
            activations_checkpoint_granularity=self.cfg.get('activations_checkpoint_granularity', None),
            activations_checkpoint_method=self.cfg.get('activations_checkpoint_method', None),
            activations_checkpoint_num_layers=self.cfg.get('activations_checkpoint_num_layers', 1),
            activations_checkpoint_layers_per_pipeline=self.cfg.get(
                'activations_checkpoint_layers_per_pipeline', None
            ),
            normalization=self.cfg.get('normalization', 'layernorm'),
            layernorm_epsilon=self.cfg.get('layernorm_epsilon', 1e-5),
            onnx_safe=self.cfg.get('onnx_safe', False),
            bias=self.cfg.get('bias', True),
            bias_activation_fusion=self.cfg.get('bias_activation_fusion', True),
            bias_dropout_add_fusion=self.cfg.get('bias_dropout_add_fusion', True),
            activation=self.cfg.get('activation', 'gelu'),
            headscale=self.cfg.get('headscale', False),
            transformer_block_type=self.cfg.get('transformer_block_type', 'pre_ln'),
            openai_gelu=self.cfg.get('openai_gelu', False),
            normalize_attention_scores=self.cfg.get('normalize_attention_scores', True),
            position_embedding_type=self.cfg.get('position_embedding_type', 'learned_absolute'),
            rotary_percentage=self.cfg.get('rotary_percentage', 1.0),
            share_embeddings_and_output_weights=self.cfg.get('share_embeddings_and_output_weights', True),
            attention_type=self.cfg.get('attention_type', 'multihead'),
            masked_softmax_fusion=self.cfg.get('masked_softmax_fusion', True),
            persist_layer_norm=self.cfg.get('persist_layer_norm', False),
            transformer_engine=self.cfg.get('transformer_engine', False),
            fp8=self.cfg.get('fp8', False),
            fp8_e4m3=self.cfg.get('fp8_e4m3', False),
            fp8_hybrid=self.cfg.get('fp8_hybrid', False),
            fp8_margin=self.cfg.get('fp8_margin', 0),
            fp8_interval=self.cfg.get('fp8_interval', 1),
            fp8_amax_history_len=self.cfg.get('fp8_amax_history_len', 1),
            fp8_amax_compute_algo=self.cfg.get('fp8_amax_compute_algo', 'most_recent'),
            reduce_amax=self.cfg.get('reduce_amax', True),
            use_emha=self.cfg.get('use_emha', False),
            ub_tp_comm_overlap=self.cfg.get('ub_tp_comm_overlap', False),
            use_flash_attention=self.cfg.get('use_flash_attention', False),
            megatron_legacy=self.cfg.get('megatron_legacy', False),
            seq_len_interpolation_factor=self.cfg.get('seq_len_interpolation_factor', None),
            embedding_scale=self.cfg.get('embedding_scale', 1.0),
            speech_loss_scale=self.cfg.get('speech_loss_scale', 1.0),
            text_size=self.cfg.get('text_size', 256000),
            use_speech_mask_for_embedding=self.cfg.get('use_speech_mask_for_embedding', False),
            attn_prior_end_step=self.cfg.get('attn_prior_end_step', 10000),
            attn_prior_scaledown_start_step=self.cfg.get('attn_prior_scaledown_start_step', 12000),
            attn_prior_starting_strength=self.cfg.get('attn_prior_starting_strength', 0.5),
            alibi_question_context_masked=self.cfg.get('alibi_question_context_masked', False),
        )

        return model

    def custom_autoregressive_inference(self, batch, prompt_len, pred_steps=500, sidx=0):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                curr_tokens = batch['tokens'][sidx : sidx + 1, :, :prompt_len]  # (B, 8, T)
                # curr_position_ids = batch['position_ids'][sidx:sidx+1,:prompt_len]
                dummy_position_ids = torch.arange(
                    0, prompt_len + pred_steps, device=batch['position_ids'].device
                ).unsqueeze(0)
                curr_position_ids = dummy_position_ids[:, :prompt_len]

                curr_attention_mask = None
                if batch['attention_mask'] is not None:
                    dummy_attention_mask = torch.tril(
                        torch.ones((1, prompt_len + pred_steps + 1, prompt_len + pred_steps + 1))
                    ).view(1, 1, prompt_len + pred_steps + 1, prompt_len + pred_steps + 1)
                    dummy_attention_mask = dummy_attention_mask < 0.5
                    dummy_attention_mask = dummy_attention_mask.to(batch['attention_mask'].device)
                    curr_attention_mask = dummy_attention_mask[:, :, :prompt_len, :prompt_len]
                    # curr_attention_mask = batch['attention_mask'][sidx:sidx+1,:,:prompt_len,:prompt_len]
                curr_speech_mask = batch['speech_mask'][sidx : sidx + 1, :prompt_len]

                all_preds = []
                temperature = self.cfg.get('temperature', 0.8)  # Set temp 0.01 for greedy decoding
                top_k = self.cfg.get('top_k', 60)
                end_timestep = None
                for _t in range(pred_steps):
                    if (end_timestep is not None) and _t == end_timestep + 8:
                        break

                    if _t % 10 == 0:
                        print("Decoding timestep", _t)

                    (logits, _), _, _ = self.model(
                        curr_tokens,
                        curr_position_ids,
                        curr_attention_mask,
                        speech_mask=curr_speech_mask,
                        return_logits=True,
                    )

                    logits = logits.transpose(0, 1).contiguous()
                    if logits[-1, 0].argmax().item() == self.tokenizer.eos_id:
                        end_timestep = _t
                        print("End detected!!!", _t)

                    all_speech_logits = []
                    all_speech_token_preds = []
                    for _i in range(8):
                        vsi = self.cfg.get("text_size", self.tokenizer.vocab_size) + _i * 1024
                        layer_logits = logits[:, :, vsi : vsi + 1024]
                        all_speech_token_preds.append(layer_logits.argmax(dim=-1))
                        all_speech_logits.append(layer_logits)
                    all_speech_logits = torch.stack(all_speech_logits, dim=-1)  # (T, B, 1024, 8)
                    output_logits_currtimestep = (
                        all_speech_logits[-1, :, :, :].permute(0, 2, 1).contiguous().view(-1, 1024)
                    )  # (B*8, V)

                    output_logits_currtimestep_topk = torch.topk(output_logits_currtimestep, top_k, dim=1)[0]
                    # find indices which are not top k
                    indices_to_remove = output_logits_currtimestep < output_logits_currtimestep_topk[:, -1].unsqueeze(
                        1
                    )
                    output_logits_currtimestep_rescored = output_logits_currtimestep.clone()
                    output_logits_currtimestep_rescored[indices_to_remove] = -float('Inf')
                    output_logits_currtimestep_rescored = output_logits_currtimestep_rescored / temperature

                    assert output_logits_currtimestep_rescored.shape == output_logits_currtimestep.shape
                    output_logits_currtimestep_rescored = torch.nn.functional.softmax(
                        output_logits_currtimestep_rescored, dim=1
                    )
                    output_tokens_curr_timestep = torch.multinomial(
                        output_logits_currtimestep_rescored, num_samples=1
                    )  # (B*8, 1)

                    output_tokens_curr_timestep = output_tokens_curr_timestep.view(all_speech_logits.shape[1], 8)

                    all_speech_token_preds = torch.stack(all_speech_token_preds, dim=-1)  # (T, B, 8)
                    all_speech_token_preds[-1, :, :] = output_tokens_curr_timestep[:, :]  # Update last-timestep

                    all_preds.append(all_speech_token_preds[-1])  # (B, 8)

                    all_speech_token_preds_processed = all_speech_token_preds.clone()  # (T, B, 8)
                    for _i in range(8):
                        all_speech_token_preds_processed[:, :, _i] = (
                            all_speech_token_preds_processed[:, :, _i]
                            + self.cfg.get("text_size", self.tokenizer.vocab_size)
                            + _i * 1024
                        )

                    all_speech_token_preds_processed = all_speech_token_preds_processed.permute(1, 2, 0)  # (B, 8, T)

                    curr_tokens = torch.cat([curr_tokens, all_speech_token_preds_processed[:, :, -1:]], dim=2)
                    curr_position_ids = dummy_position_ids[:, : prompt_len + _t + 1]
                    if curr_attention_mask is not None:
                        curr_attention_mask = dummy_attention_mask[:, :, : prompt_len + _t + 1, : prompt_len + _t + 1]
                    curr_speech_mask = batch['speech_mask'][sidx : sidx + 1, : prompt_len + _t + 1]

                all_preds = torch.stack(all_preds, dim=0)  # (T, B, 8)
                all_preds = all_preds.permute(1, 2, 0)  # (B, 8, T)

                preds_example = all_preds[0]
                preds_example = self.convert_tokens_to_range(preds_example)
                preds_wav = self.additional_models['encodec'].decode([[preds_example[None], None]])[0, 0]

                return preds_wav

    def validation_step(self, dataloader_iter, batch_idx):
        """
            Our dataloaders produce a micro-batch and then we fetch
            a number of microbatches depending on the global batch size and model parallel size
            from the dataloader to produce a list of microbatches.
            The list of microbatches is then piped through the pipeline using megatron-core fwd/bwd functions.
        """
        # Check if iterator is exhausted
        dataloader_iter, done = self._val_iterator_done(dataloader_iter)
        if done:
            return

        # Initialize userbuffer communicators.
        if self.initialize_ub:
            self.initialize_ub_func()

        if isinstance(self.model, list):
            for model_module in self.model:
                model_module.eval()

        # loss = self.fwd_bwd_step(dataloader_iter, batch_idx, True)
        # loss = loss.item()
        # Clear memory
        # torch.cuda.empty_cache()
        # loss = 0.0

        with torch.no_grad():
            dataloader_iter = self._make_data_iterator_list(dataloader_iter)
            batch = next(dataloader_iter)
            forward_keys = [
                'tokens',
                'position_ids',
                'attention_mask',
                'labels',
                'loss_mask',
                'speech_mask',
                'context_question_mask',
            ]
            if not self.cfg.get('use_attention_prior', False):
                forward_keys.append('attention_prior')
            for key in forward_keys:
                if (key in batch) and (batch[key] is not None):
                    batch[key] = batch[key].cuda()

            forward_args = {
                'input_ids': batch['tokens'],
                'position_ids': batch['position_ids'],
                'attention_mask': batch['attention_mask'],
                'labels': batch['labels'],
                'loss_mask': batch['loss_mask'],
                'speech_mask': batch['speech_mask'],
                'return_logits': True,
                'return_all_selfattention_probs': self.should_log,
                'attention_prior': batch.get('attention_prior', None),
                'global_step': self.global_step,
                'context_question_mask': batch['context_question_mask'],
            }

            if not self.cfg.get('use_attention_prior', False):
                forward_args.pop('attention_prior')

            if not self.mcore_gpt:
                forward_args['checkpoint_activations_all_layers'] = None
                if not self.use_loss_mask:
                    forward_args.pop('loss_mask')
            else:
                # TODO: @eharper can we add this to mcore?
                forward_args.pop('loss_mask')

            (_, logits), attention_probs_list, prior = self.model(**forward_args)
            layerwise_metrics = {}
            loss_total = 0.0
            all_preds = []
            # if self.cfg.get("text_size", 256000) != self.tokenizer.vocab_size:
            #     print(f"self.cfg.get('text_size', 256000) = {self.cfg.get('text_size', 256000)}")
            #     print(f"self.tokenizer.vocab_size: = {self.tokenizer.vocab_size}")
            #     raise NotImplementedError("TOO BAD!@")
            for _i in range(8):
                vsi = self.cfg.get("text_size", self.tokenizer.vocab_size) + _i * 1024
                layer_targets = batch['labels'][:, _i, :]
                if _i == 0:
                    layer_logits = logits[:, :, : vsi + 1024]
                else:
                    layer_logits = logits[:, :, vsi : vsi + 1024]
                layer_preds = layer_logits.argmax(dim=-1).permute(1, 0)  # (B, T)
                if batch_idx == 0:
                    all_preds.append(layer_preds)
                layer_acc = (
                    ((layer_preds == layer_targets).float() * batch['loss_mask']).sum() / batch['loss_mask'].sum()
                ).item()
                layer_logits_bvt = layer_logits.permute(1, 2, 0)  # (B, 1024, T)
                layer_loss = torch.nn.functional.cross_entropy(layer_logits_bvt, layer_targets, reduction='none')
                layer_loss = ((layer_loss * batch['loss_mask']).sum() / batch['loss_mask'].sum()).item()

                layerwise_metrics[f'layer_{_i}_acc'] = layer_acc
                layerwise_metrics[f'layer_{_i}_loss'] = layer_loss
                loss_total += layer_loss

            if batch_idx == 0 and self.should_log:
                start_of_speech = (
                    0 if self.pretraining else torch.count_nonzero(~batch["loss_mask"][0] * batch['tokens'][0][0]) + 2
                )
                input_tokens_example = batch['tokens'][0]

                if not self.pretraining:
                    question_tokens = []
                    question_phoneme_tokens = []
                    question_start = 0
                    for _t in range(start_of_speech):
                        if input_tokens_example[0, _t] < self.tokenizer.vocab_size:
                            question_tokens.append(input_tokens_example[0, _t].item())
                        elif (
                            input_tokens_example[0, _t] >= self.tokenizer.vocab_size
                            and input_tokens_example[0, _t] < self.cfg.text_size
                        ):
                            question_phoneme_tokens.append(
                                input_tokens_example[0, _t].item() - self.tokenizer.vocab_size
                            )
                        elif len(question_tokens) == 0:
                            question_start += 1
                    if len(question_tokens) > 0:
                        question_text = self.tokenizer.ids_to_text(question_tokens)
                        self.logger.experiment.add_text('Val Prompt Text', question_text, self.trainer.global_step)
                    if len(question_phoneme_tokens) > 0:
                        phoneme_text = phoneme_tokenizer.decode(question_phoneme_tokens)
                        self.logger.experiment.add_text(
                            'Val Prompt Phoneme Text', phoneme_text, self.trainer.global_step
                        )

                if attention_probs_list is not None:
                    speech_size = batch["loss_mask"][0].shape[0]
                    start = start_of_speech.item()
                    phoneme_seq = [question_start, start]
                    length_of_speech = torch.count_nonzero(batch["loss_mask"][0] * batch['tokens'][0][0])
                    attention_sliced_list = []
                    for lidx in range(len(attention_probs_list)):
                        attention_probs = attention_probs_list[lidx]
                        if attention_probs is not None:
                            for _i in range(attention_probs.shape[1]):
                                attention_probs_sliced = (
                                    attention_probs[0, _i, :speech_size, :speech_size].clone().detach()
                                )
                                attention_probs_sliced = attention_probs_sliced.T
                                # attention_probs_sliced *= batch["loss_mask"][0]
                                # attention_probs_sliced *= batch["attention_mask"][0][0,:,:].to(attention_probs_sliced.device)
                                alignment_image_sliced = plot_alignment_to_numpy(
                                    attention_probs_sliced.cpu().float().numpy(),
                                    phoneme_seq=phoneme_seq,
                                    phoneme_ver=1,
                                    vmin=0.0,
                                    vmax=1.0,
                                )
                                self.logger.experiment.add_image(
                                    f"Val Attention Probs Layer {lidx} Head {_i} TF",
                                    alignment_image_sliced,
                                    self.global_step,
                                    dataformats="HWC",
                                )
                                attention_probs_sliced = attention_probs_sliced[
                                    question_start:start, start : start + length_of_speech
                                ]
                                attention_sliced_list.append(attention_probs_sliced)
                    question_ids = self.tokenizer.ids_to_tokens(question_tokens)
                    phoneme_seq += question_ids
                    if len(question_phoneme_tokens) > 0:
                        phoneme_ids = phoneme_tokenizer.decode(question_phoneme_tokens).split("|")
                        phoneme_seq += phoneme_ids
                    attention_sliced = torch.stack(attention_sliced_list)
                    attention_sliced = torch.mean(attention_sliced, 0)
                    alignment_image_sliced = plot_alignment_to_numpy(
                        attention_sliced.cpu().float().numpy(), phoneme_seq=phoneme_seq, phoneme_ver=2, vmin=0.0
                    )
                    self.logger.experiment.add_image(
                        f"Val Attention Probs Average Sliced TF",
                        alignment_image_sliced,
                        self.global_step,
                        dataformats="HWC",
                    )
                    if prior is not None:
                        phoneme_seq = [question_start, start]
                        # prior = batch['attention_prior'][0,:,:].T
                        prior = torch.exp(prior[0, 0, :, :].T)
                        prior_data = plot_alignment_to_numpy(
                            prior.cpu().float().numpy(), phoneme_seq=phoneme_seq, phoneme_ver=1, vmin=0.0, vmax=1.0
                        )
                        self.logger.experiment.add_image(
                            f"Attention Prior", prior_data, self.global_step, dataformats="HWC",
                        )
                        # phoneme_seq += question_ids
                        # prior = prior[question_start:start, start:start+length_of_speech]
                        # prior_data = plot_alignment_to_numpy(
                        #     prior.cpu().float().numpy(), phoneme_seq=phoneme_seq, phoneme_ver=2, vmin=0., vmax=1.
                        # )
                        # self.logger.experiment.add_image(
                        #     f"Attention Prior Sliced",
                        #     prior_data,
                        #     self.global_step,
                        #     dataformats="HWC",
                        # )

                # Only for the first batch, log TF and autoregressive inference

                all_preds = torch.stack(all_preds).permute(1, 0, 2)  # (B, 8, T)
                all_preds_example = all_preds[0]
                all_preds_example = self.convert_tokens_to_range(all_preds_example, offset_first_layer=True)
                input_tokens_example = batch['tokens'][0]
                input_tokens_example = self.convert_tokens_to_range(
                    input_tokens_example,
                    offset_first_layer=True,
                    offset_all_layers=True,
                    start_of_speech=start_of_speech,
                )
                with torch.cuda.amp.autocast(enabled=False):
                    all_preds_wav = self.additional_models['encodec'].decode([[all_preds_example[None], None]])[0, 0]
                    dec_input_wav = self.additional_models['encodec'].decode([[input_tokens_example[None], None]])[
                        0, 0
                    ]
                self.logger.experiment.add_audio(
                    'Val Input Wav', dec_input_wav, self.trainer.global_step, sample_rate=24000
                )
                self.logger.experiment.add_audio(
                    'Val TF Wav', all_preds_wav, self.trainer.global_step, sample_rate=24000
                )

                prompt_len = (
                    100
                    if self.pretraining
                    else torch.count_nonzero(~batch["loss_mask"][0] * batch['tokens'][0][0]) + 2
                )
                prompt_len = prompt_len  # TODO: Not sure why it doesn't work without this.
                prompt_tokens = batch['tokens'][:1]  # First sample in batch
                max_length = prompt_tokens.shape[2] - prompt_len - 1
                lengths = LengthParam(min_length=max_length, max_length=max_length)
                sampling_params = get_default_sampling_params()
                sampling_params["add_BOS"] = self.cfg.data.get("add_bos", True)
                sampling_params["vocab_size"] = self.cfg.get("text_size", 256000)
                context_length = torch.tensor([prompt_len], device=self.device).contiguous()

                # For custom inference
                # pred_custom_wav = self.custom_autoregressive_inference(batch, prompt_len+8)
                # self.logger.experiment.add_audio('Val Custom Wav', pred_custom_wav, self.trainer.global_step, sample_rate=24000)

                for gen_type in ["multinomial"]:
                    logging.debug(f"Doing {gen_type} generation")
                    gen_fn_output = self.generate(
                        (prompt_tokens.contiguous(), context_length),
                        lengths,
                        sampling_params=sampling_params,
                        mode=gen_type,
                    )
                    logging.debug(f"Done {gen_type} generation")
                    gen_fn_preds = torch.tensor(gen_fn_output['token_ids'], device=self.device)

                    if not self.pretraining:
                        # For text2speech, we need to remove the prompt (text + context)
                        # For prtraining, we'll keep the audio.
                        gen_fn_preds = gen_fn_preds[:, :, prompt_len:]

                    for _i in range(8):
                        mask = gen_fn_preds[:, _i, :] != 0
                        gen_fn_preds[:, _i, :] -= self.cfg.get("text_size", self.tokenizer.vocab_size) + 1024 * _i
                        gen_fn_preds[:, _i, :] *= mask

                    gen_fn_preds_example = self.convert_tokens_to_range(gen_fn_preds[0])
                    with torch.cuda.amp.autocast(enabled=False):
                        gen_fn_preds_wav = self.additional_models['encodec'].decode(
                            [[gen_fn_preds_example[None], None]]
                        )[0, 0]

                    self.logger.experiment.add_audio(
                        'Val {} Wav'.format(gen_type), gen_fn_preds_wav, self.trainer.global_step, sample_rate=24000
                    )

        if isinstance(self.model, list):
            for model_module in self.model:
                model_module.train()

        self.validation_step_outputs.append(
            {'loss': loss_total, 'layerwise_metrics': layerwise_metrics,}
        )

        # Clears memory
        torch.cuda.empty_cache()

        return loss_total

    def on_validation_epoch_end(self):
        if parallel_state.is_pipeline_last_stage():
            # only the last pipeline parallel stages return loss with their batch size
            for _i in range(8):
                layer_acc = np.mean(
                    [x['layerwise_metrics'][f'layer_{_i}_acc'] for x in self.validation_step_outputs]
                ).item()
                layer_loss = np.mean(
                    [x['layerwise_metrics'][f'layer_{_i}_loss'] for x in self.validation_step_outputs]
                ).item()
                self.log(f'val_layer_{_i}_acc', layer_acc, prog_bar=False, rank_zero_only=True, batch_size=1)
                self.log(f'val_layer_{_i}_loss', layer_loss, prog_bar=False, rank_zero_only=True, batch_size=1)

            loss_list = [x['loss'] for x in self.validation_step_outputs]
            averaged_loss = np.mean(loss_list).item()
            self.log('val_loss', averaged_loss, prog_bar=True, rank_zero_only=True, batch_size=1)

        self.validation_step_outputs.clear()
        torch.cuda.empty_cache()
        return averaged_loss

    def test_step(self, batch, batch_idx):
        # A few batches to check the model
        print("test step", batch_idx)
        if 'asr_model' not in self.additional_models:
            asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
                model_name="stt_en_conformer_transducer_large"
            )
            asr_model = asr_model.cuda()
            asr_model.eval()
            self.additional_models['asr_model'] = asr_model

        if 'sv_model' not in self.additional_models:
            sv_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large')
            sv_model = sv_model.cuda()
            sv_model.eval()
            self.additional_models['sv_model'] = sv_model

        _exp_dir_path = self.logger.save_dir
        _exp_dir_path = _exp_dir_path + '/Sample_Audios'
        if not os.path.exists(_exp_dir_path):
            os.mkdir(_exp_dir_path)

        hyp_pred_transcript_list = []
        gt_transcript_list = []
        similarity_list = []

        # Testing it only on 2 batches, remove this if to run on all batches
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                forward_keys = ['tokens', 'position_ids', 'attention_mask', 'labels', 'loss_mask', 'speech_mask']
                for key in forward_keys:
                    if batch[key] is not None:
                        batch[key] = batch[key].cuda()

                # Autoregressive Inference From Generate Function
                for sidx in range(batch['tokens'].shape[0]):
                    _step = batch_idx * batch['tokens'].shape[0] + sidx
                    print("Batch {}, Sample {}".format(batch_idx, sidx))
                    prompt_len = (
                        100
                        if self.pretraining
                        else torch.count_nonzero(~batch["loss_mask"][sidx] * batch['tokens'][sidx][0]) + 2
                    )
                    target_speech_len = torch.count_nonzero(batch["loss_mask"][sidx]).item()
                    pred_steps = (
                        target_speech_len + 150
                    )  # To prevent very long generations if end token is not predicted
                    pred_custom_wav = self.custom_autoregressive_inference(
                        batch, prompt_len, pred_steps=pred_steps, sidx=sidx
                    )
                    self.logger.experiment.add_audio('pred_custom_wav', pred_custom_wav, _step, sample_rate=24000)
                    # prompt_len = prompt_len + 50

                    #### generate function ####
                    # prompt_tokens = batch['tokens'][sidx:sidx+1]
                    # max_length = prompt_tokens.shape[2] - prompt_len - 1
                    # lengths = LengthParam(min_length=max_length, max_length=max_length)
                    # sampling_params = get_default_sampling_params()
                    # sampling_params["add_BOS"] = self.cfg.data.get("add_bos", True)
                    # sampling_params["vocab_size"] = self.cfg.get("text_size", 256000)
                    # context_length = torch.tensor([prompt_len], device=self.device).contiguous()
                    # gen_fn_output = self.generate((prompt_tokens.contiguous(), context_length), lengths, sampling_params=sampling_params, mode="multinomial")
                    # gen_fn_preds = torch.tensor(gen_fn_output['token_ids'], device=self.device)
                    # gen_fn_preds = gen_fn_preds[:,:,prompt_len:]

                    # for _i in range(8):
                    #     mask = gen_fn_preds[:,_i,:] != 0.
                    #     gen_fn_preds[:,_i,:] -= self.cfg.get("text_size", self.tokenizer.vocab_size) + 1024*_i
                    #     gen_fn_preds[:,_i,:] *= mask
                    # gen_fn_preds_example = self.convert_tokens_to_range(gen_fn_preds[0])
                    # gen_fn_preds_wav = self.additional_models['encodec'].decode([[gen_fn_preds_example[None], None]])[0, 0]
                    # self.logger.experiment.add_audio('gen_fn_preds_wav', gen_fn_preds_wav, _step, sample_rate=24000)
                    #### generate function ####

                    context_question_tokens = batch['tokens'][sidx][:, :prompt_len]
                    context_question_tokens_encodec = self.convert_tokens_to_range(
                        context_question_tokens, offset_first_layer=True, offset_all_layers=True, delay_pattern=False
                    )
                    context_question_wav = self.additional_models['encodec'].decode(
                        [[context_question_tokens_encodec[None], None]]
                    )[0, 0]
                    self.logger.experiment.add_audio(
                        'context_question_wav', context_question_wav, _step, sample_rate=24000
                    )

                    target_tokens = batch['labels'][sidx][:, prompt_len:]
                    target_tokens_encodec = self.convert_tokens_to_range(
                        target_tokens, offset_first_layer=True, offset_all_layers=False
                    )
                    target_wav = self.additional_models['encodec'].decode([[target_tokens_encodec[None], None]])[0, 0]
                    self.logger.experiment.add_audio('target_wav', target_wav, _step, sample_rate=24000)

                    question_tokens = []
                    question_phoneme_tokens = []
                    for _t in range(prompt_len):
                        if context_question_tokens[0, _t] < self.tokenizer.vocab_size:
                            question_tokens.append(context_question_tokens[0, _t].item())
                        elif (
                            context_question_tokens[0, _t] >= self.tokenizer.vocab_size
                            and context_question_tokens[0, _t] < self.cfg.text_size
                        ):
                            question_phoneme_tokens.append(
                                context_question_tokens[0, _t].item() - self.tokenizer.vocab_size
                            )

                    if len(question_tokens) > 0:
                        question_text = self.tokenizer.ids_to_text(question_tokens)
                        self.logger.experiment.add_text('question text', question_text, _step)
                    if len(question_phoneme_tokens) > 0:
                        phoneme_text = phoneme_tokenizer.decode(question_phoneme_tokens)
                        self.logger.experiment.add_text('question phoneme text', phoneme_text, _step)

                    audio_fp_pred = os.path.join(_exp_dir_path, f'predicted_wav_{_step}.wav')
                    sf.write(audio_fp_pred, pred_custom_wav.cpu().numpy(), 24000)

                    audio_fp_gt = os.path.join(_exp_dir_path, f'target_wav_{_step}.wav')
                    sf.write(audio_fp_gt, target_wav.cpu().numpy(), 24000)

                    spk_embedding_pred = self.additional_models['sv_model'].get_embedding(audio_fp_pred)
                    spk_embedding_pred = spk_embedding_pred.cpu().detach().numpy().flatten()
                    spk_embedding_gt = self.additional_models['sv_model'].get_embedding(audio_fp_gt)
                    spk_embedding_gt = spk_embedding_gt.cpu().detach().numpy().flatten()
                    similarity = np.dot(spk_embedding_pred, spk_embedding_gt) / (
                        np.linalg.norm(spk_embedding_pred) * np.linalg.norm(spk_embedding_gt)
                    )

                    similarity_list.append(similarity)

                    pred_transcript = self.additional_models['asr_model'].transcribe([audio_fp_pred])[0][0]
                    gt_transcript = self.additional_models['asr_model'].transcribe([audio_fp_gt])[0][0]

                    self.logger.experiment.add_text("Inf Predicted Text", pred_transcript, _step)
                    self.logger.experiment.add_text("Inf GT Text", gt_transcript, _step)

                    hyp_pred_transcript_list.append(pred_transcript)
                    gt_transcript_list.append(gt_transcript)

        cer_gtaudio = None
        wer_gtaudio = None
        similarity = None
        if len(hyp_pred_transcript_list) > 0:
            cer_gtaudio = word_error_rate(hyp_pred_transcript_list, gt_transcript_list, use_cer=True)
            wer_gtaudio = word_error_rate(hyp_pred_transcript_list, gt_transcript_list, use_cer=False)
            similarity = np.mean(similarity_list)

        self.test_step_outputs.append(
            {'cer_gtaudio': cer_gtaudio, 'wer_gtaudio': wer_gtaudio, 'similarity': similarity,}
        )

    def on_test_epoch_end(self):
        cers_gtaudio = [x['cer_gtaudio'] for x in self.test_step_outputs if x['cer_gtaudio'] is not None]
        wers_gtaudio = [x['wer_gtaudio'] for x in self.test_step_outputs if x['wer_gtaudio'] is not None]
        similarities = [x['similarity'] for x in self.test_step_outputs if x['similarity'] is not None]
        if len(cers_gtaudio) > 0:
            self.log('test_cer_gtaudio', np.mean(cers_gtaudio), prog_bar=True, rank_zero_only=True, batch_size=1)
            self.log('test_wer_gtaudio', np.mean(wers_gtaudio), prog_bar=True, rank_zero_only=True, batch_size=1)
            self.log('test_similarity', np.mean(similarities), prog_bar=True, rank_zero_only=True, batch_size=1)


class MegatronSpeechGPTSFTModel(MegatronSpeechGPTModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)
        self.pad_token_id = self.tokenizer.pad_id if self.tokenizer.pad_id is not None else self.tokenizer.unk_id
        self.existing_tasks = list(self.cfg.get('existing_tasks', []))
        self.new_tasks = list(self.cfg.get('new_tasks', []))
        self.load_task_templates(self.cfg.task_templates)
        self.pseudo_tokens = get_pseudo_tokens(self.max_virtual_tokens)
        self.pretraining = False

    def build_train_valid_test_datasets(self):
        pass

    def setup_training_data(self, cfg):
        if self.cfg.data.get('train_ds', None):
            self._train_ds, self._train_dl = self.build_virtual_prompt_dataset(
                dataset_paths=self.cfg.data.train_ds,
                batch_size=self.cfg.global_batch_size,
                for_train=True,
                drop_last=True,
                shuffle=True,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
            )
        elif self.cfg.data.get('train_manifest', None):
            self._train_ds, self._train_dl = self.build_virtual_prompt_tarred_dataset(
                dataset_paths=self.cfg.data.train_manifest,
                audio_path=self.cfg.data.train_audio_path,
                batch_size=self.cfg.global_batch_size,
                for_train=True,
                drop_last=True,
                shuffle=self.cfg.data.shuffle,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
            )

    def setup_validation_data(self, cfg):
        if self.cfg.data.get('validation_ds', None):
            self._validation_ds, self._validation_dl = self.build_virtual_prompt_dataset(
                dataset_paths=self.cfg.data.validation_ds,
                batch_size=self.cfg.get("validation_global_batch_size", self.cfg.global_batch_size),
                for_train=True,
                drop_last=self.cfg.get("validation_drop_last", True),
                shuffle=False,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
            )
        elif self.cfg.data.get('validation_manifest', None):
            self._validation_ds, self._validation_dl = self.build_virtual_prompt_tarred_dataset(
                dataset_paths=self.cfg.data.validation_manifest,
                audio_path=self.cfg.data.validation_audio_path,
                batch_size=self.cfg.get("validation_global_batch_size", self.cfg.global_batch_size),
                for_train=True,
                drop_last=self.cfg.get("validation_drop_last", True),
                shuffle=0,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
            )

    def setup_test_data(self, cfg):
        if self.cfg.data.get('test_ds', None):
            self._test_ds, self._test_dl = self.build_virtual_prompt_dataset(
                dataset_paths=self.cfg.data.test_ds,
                batch_size=self.cfg.get("test_global_batch_size", self.cfg.global_batch_size),
                for_train=True,
                drop_last=self.cfg.get("test_drop_last", True),
                shuffle=False,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
            )
        elif self.cfg.data.get('test_manifest', None):
            self._test_ds, self._test_dl = self.build_virtual_prompt_tarred_dataset(
                dataset_paths=self.cfg.data.test_manifest,
                audio_path=self.cfg.data.test_audio_path,
                batch_size=self.cfg.get("test_global_batch_size", self.cfg.global_batch_size),
                for_train=True,
                drop_last=self.cfg.get("test_drop_last", True),
                shuffle=0,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
            )

    def build_virtual_prompt_dataset(
        self, dataset_paths, batch_size, for_train, drop_last, shuffle, num_workers, pin_memory
    ):
        dataset = GPTSpeechLMDataset(
            datasets=dataset_paths,
            tokenizer=self.tokenizer,
            sample_rate=self.cfg.data.get('sample_rate', 24000),
            virtual_prompt_source=VirtualPromptSource.PROMPT_ENCODER,
            task_templates=self.task_templates,
            pseudo_tokens=self.pseudo_tokens,
            pad_token_id=self.pad_token_id,
            max_seq_length=self.cfg.data.max_seq_length,
            min_seq_length=self.cfg.data.get('min_seq_length', 1),
            add_bos=self.cfg.data.get('add_bos', False),
            add_eos=self.cfg.data.get('add_eos', True),
            decoder_starts_with_pad=self.cfg.data.get('decoder_starts_with_pad', False),
            add_eos_to_decoder_output=self.cfg.data.get('add_eos_to_decoder_output', True),
            add_sentinel_to_input=self.cfg.data.get('add_sentinel_to_input', True),
            ul2_prompt_token=self.cfg.data.get('ul2_prompt_token', None),
            for_train=for_train,
            segment_max_duration=self.cfg.data.get('segment_max_duration', None),
            trim=self.cfg.data.get('trim', None),
            trim_ref=self.cfg.data.get('trim_ref', None),
            trim_top_db=self.cfg.data.get('trim_top_db', None),
            trim_frame_length=self.cfg.data.get('trim_frame_length', None),
            trim_hop_length=self.cfg.data.get('trim_hop_length', None),
            pad_multiple=self.cfg.data.get('pad_multiple', 1),
            pitch_augment=self.cfg.data.get('pitch_augment', None),
            sup_data_path=self.cfg.data.get('sup_data_path', '/sup_data_path'),
            speech_offset=self.cfg.data.get('speech_offset', None),
            train_task=self.cfg.data.get('train_task', "tts"),
            seq_pattern=self.cfg.seq_pattern,
            context_length=self.cfg.data.get('context_length', None),
            use_attention_prior=self.cfg.data.get('use_attention_prior', True),
            attention_prior_scaling_factor=self.cfg.data.get('attention_prior_scaling_factor', 1.0),
            spec_aug=self.cfg.data.get('spec_aug', False),
            spec_aug_time_width=self.cfg.data.get('spec_aug_time_width', 0.2),
            spec_aug_time_masks=self.cfg.data.get('spec_aug_time_masks', 2),
            # cross_attention_epsilon=self.cfg.data.get('cross_attention_epsilon', 1e-8),
        )

        rank = parallel_state.get_data_parallel_rank()
        world_size = parallel_state.get_data_parallel_world_size()
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=self.cfg.seed
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            sampler=sampler,
            batch_size=batch_size // world_size,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True
            if num_workers > 0
            else False,  # (@adithyare and @eharper) We need to set this to True to get around issues with spawn=True
        )

        logging.info(f'build success {len(dataloader)} {dataset_paths}')
        return dataset, dataloader

    def build_virtual_prompt_tarred_dataset(
        self, dataset_paths, audio_path, batch_size, for_train, drop_last, shuffle, num_workers, pin_memory
    ):
        dataset = GPTSpeechLMTarredDataset(
            audio_tar_filepaths=audio_path,
            manifest_filepath=dataset_paths,
            tokenizer=self.tokenizer,
            sample_rate=self.cfg.data.get('sample_rate', 24000),
            virtual_prompt_source=VirtualPromptSource.PROMPT_ENCODER,
            task_templates=self.task_templates,
            pseudo_tokens=self.pseudo_tokens,
            pad_token_id=self.pad_token_id,
            max_seq_length=self.cfg.data.max_seq_length,
            min_seq_length=self.cfg.data.get('min_seq_length', 1),
            shuffle_n=shuffle,
            add_bos=self.cfg.data.get('add_bos', False),
            add_eos=self.cfg.data.get('add_eos', True),
            decoder_starts_with_pad=self.cfg.data.get('decoder_starts_with_pad', False),
            add_eos_to_decoder_output=self.cfg.data.get('add_eos_to_decoder_output', True),
            add_sentinel_to_input=self.cfg.data.get('add_sentinel_to_input', True),
            ul2_prompt_token=self.cfg.data.get('ul2_prompt_token', None),
            for_train=for_train,
            segment_max_duration=self.cfg.data.get('segment_max_duration', None),
            trim=self.cfg.data.get('trim', None),
            trim_ref=self.cfg.data.get('trim_ref', None),
            trim_top_db=self.cfg.data.get('trim_top_db', None),
            trim_frame_length=self.cfg.data.get('trim_frame_length', None),
            trim_hop_length=self.cfg.data.get('trim_hop_length', None),
            pad_multiple=self.cfg.data.get('pad_multiple', 1),
            pitch_augment=self.cfg.data.get('pitch_augment', None),
            speech_offset=self.cfg.data.get('speech_offset', None),
            train_task=self.cfg.data.get('train_task', "tts"),
            seq_pattern=self.cfg.get('seq_pattern', 'delay_parallel'),
            decoder_only_model=True,
            context_length=self.cfg.data.get('context_length', None),
            use_phoneme_tokenizer=self.cfg.data.get('use_phoneme_tokenizer', False),
        )

        rank = parallel_state.get_data_parallel_rank()
        world_size = parallel_state.get_data_parallel_world_size()
        # sampler = torch.utils.data.distributed.DistributedSampler(
        #     dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=self.cfg.seed
        # )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            batch_size=batch_size // world_size,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True
            if num_workers > 0
            else False,  # (@adithyare and @eharper) We need to set this to True to get around issues with spawn=True
        )
        logging.info('build success', len(dataloader), dataset_paths)
        return dataset, dataloader

    def load_task_templates(self, task_templates):
        """
        Takes in the task template portion of the config and turns
        it into a table where each task's prompt template and
        the number of virtual tokens to insert in a given part of
        the prompt template are specified.
        """
        self.task_templates = {}
        self.task_id_num_to_name = {}
        self.max_virtual_tokens = 0

        task_id_num = 0
        for task in task_templates:
            self.task_templates[task.taskname] = {
                "prompt_template": task.prompt_template,
                "prompt_template_fields": re.findall("\{(.*?)\}", task.prompt_template),
                "answer_only_loss": task.get("answer_only_loss", False),
                "answer_field": task.get("answer_field", None),
                "truncate_field": task.truncate_field,
                "total_virtual_tokens": task.total_virtual_tokens,
                "virtual_token_splits": task.virtual_token_splits,
                "task_id_num": task_id_num,
            }

            self.max_virtual_tokens = max(self.max_virtual_tokens, task.total_virtual_tokens)
            self.task_id_num_to_name[task_id_num] = task.taskname
            task_id_num += 1

        # Check that all new tasks have the same total num virtual tokens
        # Num virtual tokens for new tasks don't need to match num used for previously tuned tasks
        if self.new_tasks:
            new_task_name = self.new_tasks[0]
            self.total_new_task_virtual_tokens = self.task_templates[new_task_name]["total_virtual_tokens"]

            assert all(
                self.task_templates[taskname]["total_virtual_tokens"] == self.total_new_task_virtual_tokens
                for taskname in self.new_tasks
            ), "Total virtual tokens for each task tuned simultaneously must match. If you want to use a different number of virtual tokens for different tasks, tune them separately."


def get_pseudo_tokens(num_virtual_tokens):
    """
    Takes in an integer and returns a list of strings where each string
    is a numbered virtual token placeholder. If
    num_virtual_tokens = 3, then this function returns:

    ["<prompt_0>", "<prompt_1>", "<prompt_2>"]

    Args:
        num_virtual_tokens: (int) Number of virtual token strings you want to make

    returns a list of string.

    """
    pseudo_tokens = [
        VirtualPromptPlaceholderToken.BASE.value + str(i) + VirtualPromptPlaceholderToken.END.value
        for i in range(num_virtual_tokens)
    ]

    return pseudo_tokens