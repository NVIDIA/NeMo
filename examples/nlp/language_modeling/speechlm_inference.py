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

import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import TorchElasticEnvironment

from nemo.collections.nlp.models.language_modeling.megatron_t5_speechlm_pretrain_model import (
    MegatronT5SpeechLMModel,
)
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
    PipelineMixedPrecisionPlugin,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
import torch
from nemo.collections.nlp.data.language_modeling.megatron.indexed_dataset import make_dataset as make_indexed_dataset

# mp.set_start_method("spawn", force=True)


"""
This is an example of how to ptune/prompt-tune a pretrained T5 model.
Be sure to use a .nemo T5 model with this code. If you've downloaded
a model from NGC or are otherwise using a MegatronLM model, please use
either megatron_ckpt_to_nemo.py or megatron_lm_ckpt_to_nemo.py found
within this examples directory to convert your model to .nemo format.
"""



def _mask_encoder_input(enc_input, mask_id):
    mask_context_prob = 0.8
    span_length = torch.poisson(torch.tensor([3.5]))
    span_length = int(span_length.item())
    span_length = max(span_length, 1)

    n_timesteps = enc_input.shape[1]
    span_length = min(span_length, n_timesteps)
    n_spans = int(n_timesteps // span_length)
    n_masked_spans = int(n_spans * mask_context_prob)
    masked_spans = torch.randperm(n_spans)[:n_masked_spans]
    for i in masked_spans:
        # enc_input[:, i * span_length : (i + 1) * span_length] = model.tokenizer.mask_id
        enc_input[:, i * span_length : (i + 1) * span_length] = mask_id

    return enc_input
    

def getitem_from_speech(tokens, tokenizer):
    speech_codebook_size = 1024
    for _i in range(tokens.shape[0]):
        tokens[_i] = tokens[_i] + 30000 + (_i * speech_codebook_size)

    enc_input = tokens[:, 1:] * 1
    dec_input = tokens[:, :-1] * 1
    labels = tokens[:, 1:] * 1
    
    for _i in range(1, tokens.shape[0]):
        # bring other layers back in range (0, 1024)
        labels[_i] = labels[_i] - 30000 - (_i * speech_codebook_size)

    enc_input = _mask_encoder_input(enc_input, tokenizer.mask_id)

    # TODO add pad id condition as well for enc_input?
    enc_mask = (enc_input[0] != tokenizer.mask_id).long()
    dec_mask = (labels[0] != tokenizer.pad_id ).long()
    loss_mask = (enc_input[0] == tokenizer.mask_id ).long()

    item_dict = {
        'enc_input': [ enc_input.long() ],
        'dec_input': [ dec_input.long() ],
        'labels': [ labels.long() ],
        'enc_mask': [ enc_mask.long() ],
        'dec_mask': [ dec_mask.long() ],
        'loss_mask': [ loss_mask.long() ],
        'speech_mask' : [ torch.ones_like(enc_input[0]).long() ],
        'position_ids' : [ torch.arange(enc_input.shape[1], dtype=torch.long) ]
    }
    
    for key in item_dict:
        item_dict[key] = torch.stack(item_dict[key])
    
    return item_dict
    
    
def unprocess_encoder_input(enc_input):
    assert enc_input.dim() == 2
    unprocessed_enc_input = enc_input.clone()
    for _i in range(enc_input.shape[0]):
        mask_indices = (enc_input[_i] != 103).long()
        unprocessed_enc_input[_i] = enc_input[_i] - 30000 - (_i * 1024)
        unprocessed_enc_input[_i] = unprocessed_enc_input[_i] * mask_indices
    
    return unprocessed_enc_input    


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

    # # load existing or init new soft prompt T5 model
    # if cfg.model.get("restore_path", None):
    #     print(f"cfg.model.restore_path {cfg.model.restore_path}")
    #     model = MegatronT5SpeechLMModel.restore_from(
    #         cfg.model.restore_path, cfg.model, trainer=trainer, save_restore_connector=NLPSaveRestoreConnector()
    #     )

    # else:
    #     print(f"cfg.model.restore_path is None")
    #     model = MegatronT5SpeechLMModel(cfg.model, trainer=trainer)

    # checkpoint_path = "/datap/misc/Experiments3/SpeechExperiments/Aug3BugsFixed29184MaskProb0.3/2023-08-03_23-06-41/checkpoints/Epoch102000.ckpt"
    checkpoint_path = "/datap/misc//Step60k.ckpt"
    # model = MegatronT5SpeechLMModel.load_from_checkpoint(checkpoint_path="/datap/misc/Experiments3/SpeechExperiments/Aug3BugsFixed29184MaskProb0.3/2023-08-03_23-06-41/checkpoints/Epoch40000.ckpt", trainer=trainer, cfg=cfg.model)
    model = MegatronT5SpeechLMModel.load_from_checkpoint(checkpoint_path=checkpoint_path, trainer=trainer, cfg=cfg.model)
    model.eval()
    model = model.cuda()



    for example_num in range(3):
        with torch.no_grad():
            indexed_dataset_speech = make_indexed_dataset("/datap/misc/BinaryDataset/librilight/eng_librivox_22khz_encodec_pt_filepath_document", "lazy")
            sample_input = indexed_dataset_speech[example_num]
            sample_input = torch.tensor(sample_input)[:,1024:1024+512]
            batch = getitem_from_speech(sample_input, model.tokenizer)

            
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
                        enc_input_example = unprocess_encoder_input(enc_input[0])
                        enc_wav = model.additional_models['encodec'].decode([[enc_input_example[None], None]])[0,0]
                        model.logger.experiment.add_audio("Enc Input", enc_wav, example_num+1, 24000)

                        dec_input_example = unprocess_encoder_input(dec_input[0])
                        dec_input_wav = model.additional_models['encodec'].decode([[dec_input_example[None], None]])[0,0]
                        model.logger.experiment.add_audio("Dec Input", dec_input_wav, example_num+1, 24000)

                    dec_input[:, :, t+1:] = model.tokenizer.pad_id
                    dec_input_mask[:, t+1 : ] = 0
                    print("dec_input", dec_input.shape, dec_input[:,:,:10])

                    position_ids = batch['position_ids'].cuda()
                    speech_mask = batch['speech_mask'].cuda()
                    output_tensor, enc_output, debug_tensors = model(
                                enc_input, enc_mask, dec_input, dec_input_mask, position_ids, labels=None, speech_mask=speech_mask, inference=True,
                            )
                    output_tokens = output_tensor.argmax(dim=2)
                    if t == 0:
                        pass
                        # print("output tokens shape", output_tokens.shape)
                        # enc_input_example = unprocess_encoder_input(enc_input[0])
                        # enc_wav = model.additional_models['encodec'].decode([[enc_input_example[None], None]])[0,0]
                        # model.logger.experiment.add_audio("Enc Input", enc_wav, example_num+1, 24000)

                        # dec_input_example = unprocess_encoder_input(dec_input[0])
                        # dec_input_wav = model.additional_models['encodec'].decode([[dec_input_example[None], None]])[0,0]
                        # model.logger.experiment.add_audio("Dec Input", dec_input_wav, example_num+1, 24000)

                        # output_tokens_encodec = output_tokens[0].transpose(0,1)
                        # all_wav = model.additional_models['encodec'].decode([[output_tokens_encodec[None], None]])[0,0]
                        # model.logger.experiment.add_audio("Dec Wav Complete", all_wav, example_num+1, 24000)

                        # token_logits = debug_tensors[0]
                        # speech_logits = debug_tensors[1].transpose(0,1)
                        # token_logits_example = token_logits[0,:,:] * 1
                        # speech_logits_example = speech_logits[:,0,:,:] * 1
                        # first_layer_tokens = token_logits_example.argmax(dim=1) - 29184
                        # other_layer_tokens = []
                        # for _i in range(speech_logits_example.shape[2]):
                        #     other_layer_tokens.append(speech_logits_example[:,:,_i].argmax(dim=1))
                        # all_layer_tokens = torch.stack([first_layer_tokens] + other_layer_tokens) # (8, t)
                        # all_layer_tokens = torch.clip(all_layer_tokens, 0, 1023)
                        
                        # predicted_wav2 = model.additional_models['encodec'].decode([[all_layer_tokens[None], None]])[0,0]
                        # model.logger.experiment.add_audio("Dec Wav Old Way", predicted_wav2, example_num+1, 24000)


                    output_tokens_curr_timestep = output_tokens[:, t]
                    output_token_list.append(output_tokens_curr_timestep[0])
                    output_tokens_curr_timestep_deccompatible = output_tokens_curr_timestep * 1
                    for _c in range(8):
                        output_tokens_curr_timestep_deccompatible[:,_c] = output_tokens_curr_timestep_deccompatible[:,_c] + 30000 + (_c * 1024)

                    dec_input[:, :, t+1] = output_tokens_curr_timestep_deccompatible * 1
                    
                output_tokens_combined = torch.stack(output_token_list) # (T, 8)
                output_tokens_combined = output_tokens_combined.transpose(0,1) # (8, T)
                output_wav = model.additional_models['encodec'].decode([[output_tokens_combined[None], None]])[0,0]
                model.logger.experiment.add_audio("Dec Wav", output_wav, example_num+1, 24000)
    


if __name__ == '__main__':
    main()
