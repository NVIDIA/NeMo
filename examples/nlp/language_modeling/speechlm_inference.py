# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import torch
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import TorchElasticEnvironment

from nemo.collections.nlp.data.language_modeling.megatron.indexed_dataset import make_dataset as make_indexed_dataset
from nemo.collections.nlp.models.language_modeling.megatron_t5_speechlm_pretrain_model import MegatronT5SpeechLMModel
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
    PipelineMixedPrecisionPlugin,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

# mp.set_start_method("spawn", force=True)


"""
This is an example of how to ptune/prompt-tune a pretrained T5 model.
Be sure to use a .nemo T5 model with this code. If you've downloaded
a model from NGC or are otherwise using a MegatronLM model, please use
either megatron_ckpt_to_nemo.py or megatron_lm_ckpt_to_nemo.py found
within this examples directory to convert your model to .nemo format.
"""


def _mask_encoder_input(enc_input, mask_id, seq_pattern="parallel"):
    mask_length_poisson_lambda = 4.0
    mask_context_prob = 0.99
    if seq_pattern in ["parallel", "delay_parallel"]:
        span_length = torch.poisson(torch.tensor([mask_length_poisson_lambda]))
        span_length = int(span_length.item())
        span_length = max(span_length, 1)

        n_timesteps = enc_input.shape[1]
        span_length = min(span_length, n_timesteps)
        n_spans = int(n_timesteps // span_length)
        n_masked_spans = int(n_spans * mask_context_prob)
        masked_spans = torch.randperm(n_spans)[:n_masked_spans]
        for i in masked_spans:
            if (i * span_length) > 100 and (i * span_length) < n_timesteps - 100:
                enc_input[:, i * span_length : (i + 1) * span_length] = mask_id

    elif seq_pattern == "flatten":
        span_length = torch.poisson(torch.tensor([mask_length_poisson_lambda]))
        span_length = int(span_length.item())
        span_length = max(span_length, 1)
        n_timesteps = enc_input.shape[1] // 8
        span_length = min(span_length, n_timesteps)
        n_spans = int(n_timesteps // span_length)
        n_masked_spans = int(n_spans * mask_context_prob)
        masked_spans = torch.randperm(n_spans)[:n_masked_spans]
        for i in masked_spans:
            enc_input[0, i * span_length * 8 : (i + 1) * span_length * 8] = mask_id
    else:
        raise NotImplementedError(f"seq_pattern={seq_pattern} not implemented")

    return enc_input


def getitem_from_speech(tokens, tokenizer, seq_pattern="parallel"):
    speech_codebook_size = 1024
    speech_offset = 30000
    seq_length = 192
    tokens[0] = tokens[0] + speech_offset

    if seq_pattern == "parallel":
        enc_input = tokens[:, 1:] * 1  # to avoid changing the original tensor
        dec_input = tokens[:, :-1] * 1
        labels = tokens[:, 1:] * 1

    elif seq_pattern == "delay_parallel":
        # Pad tokens with 8 zeros at the begginging
        num_codebooks = 8
        tokens = torch.cat([torch.zeros_like(tokens[:, 0:num_codebooks]), tokens], dim=1)
        enc_input = tokens[:, num_codebooks+1:] * 1  # to avoid changing the original tensor
        dec_tokens = []
        for _c in range(8):
            st = 8 - _c
            et = tokens.shape[1] - _c
            dec_tokens.append(tokens[_c, st:et])
        dec_tokens = torch.stack(dec_tokens, dim=0)
        dec_input = dec_tokens[:, :-1] * 1
        labels = dec_tokens[:, 1:] * 1
    elif seq_pattern == "flatten":
        for _c in range(1, 8):
                tokens[_c] = tokens[_c] + speech_offset + _c * speech_codebook_size
        tokens_flat = tokens.permute(1, 0).flatten()[None] # (1, seq_len * 8)
        tokens_flat = tokens_flat[:,:seq_length*8+1]
        tokens_processed = torch.cat([tokens_flat, torch.zeros(7, tokens_flat.shape[1])], dim=0) # (8, seq_len * 8)
        enc_input = tokens_processed[:, 1:] * 1  # to avoid changing the original tensor
        dec_input = tokens_processed[:, :-1] * 1
        labels = tokens_processed[:, 1:] * 1
    else:
        raise NotImplementedError(f"seq_pattern={seq_pattern} not implemented")
    enc_input = _mask_encoder_input(enc_input, tokenizer.mask_id, seq_pattern)

    # TODO add pad id condition as well for enc_input?
    enc_mask = (enc_input[0] != tokenizer.mask_id).long()
    dec_mask = (labels[0] != tokenizer.pad_id).long()
    loss_mask = (enc_input[0] == tokenizer.mask_id).long()

    item_dict = {
        'enc_input': [enc_input.long()],
        'dec_input': [dec_input.long()],
        'labels': [labels.long()],
        'enc_mask': [enc_mask.long()],
        'dec_mask': [dec_mask.long()],
        'loss_mask': [loss_mask.long()],
        'speech_mask': [torch.ones_like(enc_input[0]).long()],
        'position_ids': [torch.arange(enc_input.shape[1], dtype=torch.long)],
    }

    for key in item_dict:
        item_dict[key] = torch.stack(item_dict[key])

    return item_dict

def convert_tokens_to_range(tokens, apply_offset_correction=True, token_type="encoder", seq_pattern="parallel"):
    # convert tokens to range [0, 1024]
    speech_offset = 30000
    output_tokens = tokens.clone()
    if apply_offset_correction:
        output_tokens[0] = output_tokens[0] - speech_offset
    output_tokens = torch.clamp(output_tokens, min=0, max=1023)
    if seq_pattern == "delay_parallel" and token_type == "decoder":
        output_tokens_new = []
        for _c in range(output_tokens.shape[0]):
            si = _c
            ei = _c + output_tokens.shape[1] - 8
            output_tokens_new.append(output_tokens[_c,si:ei])
        output_tokens_new = torch.stack(output_tokens_new)
        output_tokens = output_tokens_new

    return output_tokens


@hydra_runner(config_path="conf", config_name="speechlm_inference.yaml")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    megatron_amp_o2 = cfg.model.get('megatron_amp_O2', False)
    with_distributed_adam = cfg.model.optim.get('name') == 'distributed_fused_adam'

    plugins = []
    strategy = NLPDDPStrategy(
        no_ddp_communication_hook=True,  # we don't use DDP for async grad allreduce
        gradient_as_bucket_view=False,
        find_unused_parameters=False,
    )
    if cfg.trainer.precision in [16, 'bf16']:
        scaler = None
        if cfg.trainer.precision == 16:
            scaler = GradScaler(
                init_scale=cfg.model.get('native_amp_init_scale', 2 ** 8),
                growth_interval=cfg.model.get('native_amp_growth_interval', 1000),
                hysteresis=cfg.model.get('hysteresis', 2),
            )
        if megatron_amp_o2 and not with_distributed_adam:
            plugins.append(MegatronHalfPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler))
        else:
            plugins.append(PipelineMixedPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler))

    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())

    trainer = Trainer(plugins=plugins, strategy=strategy, **cfg.trainer)
    exp_manager(trainer, cfg.exp_manager)

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision

    # checkpoint_path = "/datap/misc/DelayPatternExperimentsFinal/LocalRun/Step90k.ckpt"
    checkpoint_path = "/datap/misc/DelayPatternExperimentsLinearHead/Step98k.ckpt"
    model = MegatronT5SpeechLMModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path, trainer=trainer, cfg=cfg.model
    )
    model.eval()
    model = model.cuda()
    seq_pattern = cfg.model.get('seq_pattern', 'parallel')

    for example_num in range(3):
        with torch.no_grad():
            indexed_dataset_speech = make_indexed_dataset(
                "/datap/misc/BinaryDataset/librilight/eng_librivox_22khz_encodec_pt_filepath_document", "lazy"
            )
            sample_input = indexed_dataset_speech[example_num]
            sample_input = torch.tensor(sample_input)[:, 1024 : 1024 + 513]
            batch = getitem_from_speech(sample_input, model.tokenizer, seq_pattern=seq_pattern)

            with torch.no_grad():
                output_token_list = []
                dec_input = batch['dec_input'].cuda()
                for t in range(500):
                    print("timestep: ", t)
                    enc_input = batch['enc_input'].cuda()
                    enc_mask = batch['enc_mask'].cuda()
                    dec_input_mask = batch['dec_mask'].cuda()

                    if t == 0:
                        # print("output tokens shape", output_tokens.shape)
                        if seq_pattern in ["parallel", "delay_parallel"]:
                            enc_input_example = convert_tokens_to_range(enc_input[0], seq_pattern=seq_pattern)
                            enc_wav = model.additional_models['encodec'].decode([[enc_input_example[None], None]])[0, 0]
                            model.logger.experiment.add_audio("Enc Input", enc_wav, example_num + 1, 24000)

                            dec_input_example = convert_tokens_to_range(dec_input[0], token_type="decoder", seq_pattern=seq_pattern)
                            dec_input_wav = model.additional_models['encodec'].decode([[dec_input_example[None], None]])[
                                0, 0
                            ]
                            model.logger.experiment.add_audio("Dec Input", dec_input_wav, example_num + 1, 24000)
                        else:
                            enc_input_example = enc_input[0][0]
                            # Add a dummy token to enc_input_example in the beginning
                            enc_input_example = torch.cat([torch.tensor([0]).cuda(), enc_input_example])[:-1]
                            dec_input_example = dec_input[0][0]

                            all_layer_tokens_encinput = []
                            all_layer_tokens_decinput = []
                            for _c in range(8):
                                # 0th layer tokens are indices 0, 8, 16, 24, 32, 40, 48, 56
                                # 1st layer tokens are indices 1, 9, 17, 25, 33, 41, 49, 57
                                layer_tokens_encinput = enc_input_example[_c::8]
                                layer_tokens_decinput = dec_input_example[_c::8]
                                
                                layer_tokens_encinput = layer_tokens_encinput - 30000 - (_c * 1024)
                                layer_tokens_decinput = layer_tokens_decinput - 30000 - (_c * 1024)
                                
                                all_layer_tokens_encinput.append(layer_tokens_encinput)
                                all_layer_tokens_decinput.append(layer_tokens_decinput)
                            
                            all_layer_tokens_encinput = torch.stack(all_layer_tokens_encinput)
                            all_layer_tokens_decinput = torch.stack(all_layer_tokens_decinput)

                            all_layer_tokens_encinput = torch.clip(all_layer_tokens_encinput, 0, 1023)
                            enc_wav = model.additional_models['encodec'].decode([[all_layer_tokens_encinput[None], None]])[0, 0]
                            model.logger.experiment.add_audio("Enc Input", enc_wav, example_num + 1, 24000)

                            dec_wav = model.additional_models['encodec'].decode([[all_layer_tokens_decinput[None], None]])[0, 0]
                            model.logger.experiment.add_audio("Dec Input", dec_wav, example_num + 1, 24000)


                    # dec_input[:, :, t + 1 :] = model.tokenizer.pad_id
                    # dec_input_mask[:, t + 1 :] = 0
                    print("dec_input", dec_input.shape, dec_input[:, :, :10])

                    position_ids = batch['position_ids'].cuda()
                    speech_mask = batch['speech_mask'].cuda()
                    output_tensor, enc_output, debug_tensors = model(
                        enc_input,
                        enc_mask,
                        dec_input,
                        dec_input_mask,
                        position_ids,
                        labels=None,
                        speech_mask=speech_mask,
                        inference=True,
                    )
                    if seq_pattern in ["parallel", "delay_parallel"]:
                        output_logits = output_tensor[:,t,:] # (B, Vocab Size, 8)
                        output_logits = output_logits[0].permute(1, 0) # (8, Vocab Size)
                        # Multinomial sampling using temperature T
                        TEMP = 0.05
                        output_probs = torch.nn.functional.softmax(output_logits / TEMP, dim=1)
                        output_tokens_curr_timestep = torch.multinomial(output_probs, num_samples=1)[:,0][None]
                        # import ipdb; ipdb.set_trace()

                        # output_tokens = output_tensor.argmax(dim=2)
                        # output_tokens_curr_timestep = output_tokens[:, t]
                        output_token_list.append(output_tokens_curr_timestep[0])
                        dec_input_next = output_tokens_curr_timestep * 1
                        dec_input_next[:,0] = dec_input_next[:,0] + 30000
                        # if t > 100:
                        dec_input[:, :, t + 1] = dec_input_next
                    else:
                        first_layer_logits = debug_tensors[0] # (1, s, 30k)
                        prediction = first_layer_logits.argmax(dim=2) # (1, s)
                        predicted_token = prediction[0, t]
                        output_token_list.append(predicted_token)
                        dec_input[:,0,t+1] = prediction[:,t] * 1
                
                if seq_pattern in ["parallel", "delay_parallel"]:
                    output_tokens_combined = torch.stack(output_token_list)  # (T, 8)
                    output_tokens_combined = output_tokens_combined.transpose(0, 1)  # (8, T)
                    output_tokens_combined = convert_tokens_to_range(output_tokens_combined, apply_offset_correction=False, token_type="decoder", seq_pattern=seq_pattern)
                    output_wav = model.additional_models['encodec'].decode([[output_tokens_combined[None], None]])[0, 0]
                    model.logger.experiment.add_audio("Dec Wav", output_wav, example_num + 1, 24000)
                else:
                    output_tokens_combined = torch.stack(output_token_list) # T
                    # prepend 0 to output_tokens_combined
                    output_tokens_combined = torch.cat([torch.tensor([0]).cuda(), output_tokens_combined])[:-1]
                    all_layer_tokens = []
                    for _c in range(8):
                        # 0th layer tokens are indices 0, 8, 16, 24, 32, 40, 48, 56
                        # 1st layer tokens are indices 1, 9, 17, 25, 33, 41, 49, 57
                        layer_tokens = output_tokens_combined[_c::8]
                        layer_tokens = layer_tokens - 30000 - (_c * 1024)
                        all_layer_tokens.append(layer_tokens_decinput)
                    all_layer_tokens = torch.stack(all_layer_tokens)
                    all_layer_tokens = torch.clip(all_layer_tokens, 0, 1023)
                    output_wav = model.additional_models['encodec'].decode([[all_layer_tokens[None], None]])[0, 0]
                    model.logger.experiment.add_audio("Dec Wav", output_wav, example_num + 1, 24000)


if __name__ == '__main__':
    main()
