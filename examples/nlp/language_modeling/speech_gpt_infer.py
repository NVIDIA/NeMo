### Model eval
import nemo
import torch
import os
import tempfile
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel, MegatronSpeechGPTModel
from pytorch_lightning.trainer.trainer import Trainer
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from omegaconf import OmegaConf
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
import pytorch_lightning as pl
from nemo.utils import AppState

config = OmegaConf.load("/home/jasoli/gitrepos/NeMo/examples/nlp/language_modeling/conf/megatron_gpt_prompt_learning_config.yaml")
# let's modify some trainer configs
# check if we have GPU available and uses it
accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
config.trainer.accelerator = accelerator
config.trainer.devices = 1
config.trainer.max_epochs = 4
config.trainer.val_check_interval = 1.0

# for PyTorch Native AMP set precision=16
config.trainer.precision = 16 if torch.cuda.is_available() else 32

# setup cluster environment parameters"
# use torch elastic cluster environment so `create_process_externally` is True
# the launcher is set to None. It will not try to spawn new processes.
# It won't create the misconfiguration error because of the `interactive session`
os.environ["LOCAL_RANK"] = '0'
os.environ["RANK"] = '0'
os.environ["WORLD_SIZE"] = '1'

strategy = NLPDDPStrategy(find_unused_parameters=False, no_ddp_communication_hook=True)
plugins = [TorchElasticEnvironment()]
trainer = pl.Trainer(plugins= plugins, strategy=strategy, **config.trainer)

print("Trainer config - \n")
print(OmegaConf.to_yaml(config.trainer))

# checkpoint_path = "/mnt/drive1/experiments/selene_sgpt_2b_pretrain_11/megatron_sgpt_2b/megatron_gpt--val_loss=4.61-step=46005-consumed_samples=5888000.0-last.ckpt"
# checkpoint_path = "/home/jasoli/experiments/nemo_experiments/megatron_sgpt_843m_linear/checkpoints/megatron_gpt--val_loss=5.74-step=111000-consumed_samples=887984.0.ckpt"
# checkpoint_path = "/mnt/drive1/experiments/sgpt_pretrain_2b_linear_FA_0/08_15_23/training/megatron_sgpt_2b/checkpoints/megatron_gpt--val_loss=5.53-step=21002-consumed_samples=2688000.0-last.ckpt"
# checkpoint_path = "/home/jasoli/experiments/nemo_experiments/megatron_sgpt_843m_linear_delay_embeddingscale/checkpoints/megatron_gpt--val_loss=5.51-step=349000-consumed_samples=5583968.0-last.ckpt"
# checkpoint_path = "/mnt/drive1/experiments/sgpt_pretrain_843m_linearv2/09_08_23/training/megatron_sgpt_843m/checkpoints/megatron_gpt--val_loss=6.34-step=24768-consumed_samples=19016448.0-last.ckpt"
checkpoint_path = "/home/jasoli/experiments/nemo_experiments/megatron_sgpt_220m_linearv2_delay_embeddingscale/checkpoints/megatron_gpt--val_loss=5.90-step=38000-consumed_samples=2431936.0-last.ckpt"
gpt_cfg = MegatronSpeechGPTModel.restore_from(
#     restore_path="/home/jasoli/models/gpt_2b_gtc_tp1_pp1_1_1T/megatron_converted_2b_tp1_pp1.nemo",
#     restore_path="/home/jasoli/models/gpt_843m_gtc_tp1_pp1_1_1T/megatron_converted_843m_tp1_pp1.nemo",
    restore_path="/home/jasoli/models/gpt_pretrain_220m_len_4096_pos_alibi_step_595508_gbs256.nemo",
    trainer=trainer,
    return_config=True,
    save_restore_connector=NLPSaveRestoreConnector(),
    map_location="cpu"
)

def load_from_checkpoint_dir(cls, cfg, trainer, checkpoint):
    app_state = AppState()
#     if cfg.model.tensor_model_parallel_size > 1 or cfg.model.pipeline_model_parallel_size > 1:
#         app_state.model_parallel_size = cfg.model.tensor_model_parallel_size * cfg.model.pipeline_model_parallel_size
#         app_state.tensor_model_parallel_size = cfg.model.tensor_model_parallel_size
#         app_state.pipeline_model_parallel_size = cfg.model.pipeline_model_parallel_size
#         (
#             app_state.tensor_model_parallel_rank,
#             app_state.pipeline_model_parallel_rank,
#             app_state.model_parallel_size,
#             app_state.data_parallel_size,
#             app_state.pipeline_model_parallel_split_rank,
#             app_state.virtual_pipeline_model_parallel_rank,
#         ) = fake_initialize_model_parallel(
#             world_size=app_state.model_parallel_size,
#             rank=trainer.global_rank,
#             tensor_model_parallel_size_=cfg.model.tensor_model_parallel_size,
#             pipeline_model_parallel_size_=cfg.model.pipeline_model_parallel_size,
#             pipeline_model_parallel_split_rank_=cfg.model.pipeline_model_parallel_split_rank,
#         )
#     checkpoint_path = inject_model_parallel_rank(
#         os.path.join(cfg.model.pretrained_checkpoint.checkpoint_dir, cfg.model.pretrained_checkpoint.checkpoint_name)
#     )
#     hparams_file = OmegaConf.load(cfg.model.pretrained_checkpoint.hparams_file)
#     gpt_cfg = _modify_config(hparams_file.cfg, cfg, add_cfg_to_tree=True)
    OmegaConf.resolve(cfg)
    cfg.cfg = cfg
#     cfg.cfg.tokenizer.model = "/home/jasoli/models/gpt_2b_gtc_tp1_pp1_1_1T/2053796188904e679f7e2754a2a1f280_mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model"
#     cfg.cfg.tokenizer.tokenizer_model = "/home/jasoli/models/gpt_2b_gtc_tp1_pp1_1_1T/2053796188904e679f7e2754a2a1f280_mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model"
    cfg.cfg.tokenizer.model = "/home/jasoli/models/gpt_pretrain_220m_len_4096_pos_alibi_step_595508_gbs256/b11cee03bc8940fa86cb2da19e99fd4e_t5_tokenizer.model"
    cfg.cfg.tokenizer.tokenizer_model = "/home/jasoli/models/gpt_pretrain_220m_len_4096_pos_alibi_step_595508_gbs256/b11cee03bc8940fa86cb2da19e99fd4e_t5_tokenizer.model"
#     cfg.cfg.override_vocab_size = 256000+1024*8
#     cfg.cfg.output_size = 256000+1024
    cfg.cfg.override_vocab_size = 32128+1024*8
    cfg.cfg.output_size = None
#     cfg.cfg.speech_residual_model = "conv"
    cfg.cfg.speech_residual_model = None
#     input_type = "parallel"
    with tempfile.NamedTemporaryFile(suffix='.yaml') as f:
        OmegaConf.save(config=cfg, f=f.name)
        model = cls.load_from_checkpoint(checkpoint_path=checkpoint, trainer=trainer, hparams_file=f.name)
        return model

input_type = "delay"
model = load_from_checkpoint_dir(MegatronSpeechGPTModel, gpt_cfg, trainer, checkpoint_path)

import numpy as np
total = 0
for parameter in model.parameters():
    total += np.prod([*parameter.shape])
print(f"total params: {total}")

# print(model.model.module.language_model.output_layer)
# for p in model.model.module.language_model.output_layer.parameters():
#     print(p.shape)
# print(model.model.module.language_model.embedding.word_embeddings)
# for p in model.model.module.language_model.embedding.word_embeddings.parameters():
#     print(p.shape)

from nemo.collections.nlp.modules.common.text_generation_utils import get_default_sampling_params

sampling_params = get_default_sampling_params()
vocab_size = 256000
if "220m" in checkpoint_path:
    sampling_params['add_BOS'] = False
    vocab_size = 32128
sampling_params['vocab_size'] = vocab_size

with torch.no_grad():
    model.float()
    output = model.generate(["Tell me a story about a boy"], None, sampling_params=sampling_params, mode="greedy")
print(output)


import librosa
import torch
from encodec import EncodecModel
from encodec.utils import convert_audio

# Instantiate a pretrained EnCodec model
encodec_model = EncodecModel.encodec_model_24khz()
# The number of codebooks used will be determined bythe bandwidth selected.
# E.g. for a bandwidth of 6kbps, `n_q = 8` codebooks are used.
# Supported bandwidths are 1.5kbps (n_q = 2), 3 kbps (n_q = 4), 6 kbps (n_q = 8) and 12 kbps (n_q =16) and 24kbps (n_q=32).
# For the 48 kHz model, only 3, 6, 12, and 24 kbps are supported. The number
# of codebooks for each is half that of the 24 kHz model as the frame rate is twice as much.
encodec_model.set_target_bandwidth(6)

# Load and pre-process the audio waveform
y, sr = librosa.load(
    "/mnt/drive1/data/HiFiTTS/wav/8051_clean/14468/whomweshallwelcome_09_commission_0001.wav", sr=24000
)
y = torch.unsqueeze(torch.tensor(y), 0)

wav = convert_audio(y, sr, encodec_model.sample_rate, encodec_model.channels)
wav = wav.unsqueeze(0)

# Extract discrete codes from EnCodec
with torch.no_grad():
    encoded_frames = encodec_model.encode(wav)
codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
print(y.shape)
print(codes.shape)

with torch.no_grad():
    out_wav = encodec_model.decode([[codes, None]])

if isinstance(out_wav[0], torch.Tensor):
    out_wav = out_wav[0].to('cpu').numpy()
    print(out_wav.shape)

from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam
context_length = 128
min_length = 256
max_length = codes.shape[-1]-1
lengths = LengthParam(min_length=min_length, max_length=max_length)
context_length = torch.tensor([context_length], device=model.device).contiguous()

context_codes = codes[:,:,:context_length].detach().clone()
for i in range(context_codes.shape[1]):
    context_codes[:, i, :] += vocab_size + 1024 * i

input_codes = torch.cat((context_codes, torch.zeros([*codes.shape[:-1], max_length], dtype=codes.dtype)), dim=-1)
lengths = LengthParam(min_length=min_length, max_length=max_length)
context_length = torch.tensor([context_length], device=model.device).contiguous()

if input_type=="delay":
    for l in range(1, input_codes.shape[1]):
        input_codes[:, l] = torch.roll(input_codes[:, l], l)
        input_codes[:, l, :l] = 0

print(input_codes)
print(input_codes.shape)
print(context_length)
with torch.no_grad():
    model.float()
    output = model.generate((input_codes.to(model.device), context_length), lengths, sampling_params=sampling_params, mode="multinomial")

predicted_tensors = torch.tensor(output['token_ids'], device=model.device)
print(input_codes[:, :, 5:20].to(model.device) == predicted_tensors[:, :, 5:20])
