### Model eval
import os
import tempfile

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from pytorch_lightning.trainer.trainer import Trainer

import nemo
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel, MegatronSpeechGPTSFTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.utils import AppState

config = OmegaConf.load(
    "/home/jasoli/gitrepos/NeMo/examples/nlp/language_modeling/conf/megatron_gpt_prompt_learning_config.yaml"
)
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
trainer = pl.Trainer(plugins=plugins, strategy=strategy, **config.trainer)

print("Trainer config - \n")
print(OmegaConf.to_yaml(config.trainer))

nemo_path = "/home/jasoli/models/gpt_2b_gtc_tp1_pp1_1_1T/megatron_converted_2b_tp1_pp1.nemo"
# nemo_path = "/home/jasoli/models/gpt_843m_gtc_tp1_pp1_1_1T/megatron_converted_843m_tp1_pp1.nemo"
# nemo_path = "/home/jasoli/models/gpt_pretrain_220m_len_4096_pos_alibi_step_595508_gbs256.nemo"
# checkpoint_path = "/home/jasoli/experiments/nemo_experiments/megatron_sgpt_220m_linearv2_delay_noscaling/checkpoints/megatron_gpt--val_loss=23.65-step=110000-consumed_samples=3519840.0-last.ckpt"
checkpoint_path = "/mnt/drive1/experiments/sgpt_pretrain_2b_linearv2_2/09_20_23/training/megatron_sgpt_2b/checkpoints/megatron_gpt--val_loss=22.47-step=79589-consumed_samples=10187392.0-last.ckpt"
gpt_cfg = MegatronGPTModel.restore_from(
    restore_path=nemo_path,
    trainer=trainer,
    return_config=True,
    save_restore_connector=NLPSaveRestoreConnector(),
    map_location="cpu",
)


def load_from_checkpoint_dir(cls, cfg, trainer, checkpoint):
    app_state = AppState()
    OmegaConf.resolve(cfg)
    cfg.cfg = cfg
    cfg.cfg.tokenizer.model = "/home/jasoli/models/gpt_2b_gtc_tp1_pp1_1_1T/2053796188904e679f7e2754a2a1f280_mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model"
    cfg.cfg.tokenizer.tokenizer_model = "/home/jasoli/models/gpt_2b_gtc_tp1_pp1_1_1T/2053796188904e679f7e2754a2a1f280_mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model"
    cfg.cfg.override_vocab_size = 264192
    # cfg.cfg.tokenizer.model = "/home/jasoli/models/gpt_pretrain_220m_len_4096_pos_alibi_step_595508_gbs256/b11cee03bc8940fa86cb2da19e99fd4e_t5_tokenizer.model"
    # cfg.cfg.tokenizer.tokenizer_model = "/home/jasoli/models/gpt_pretrain_220m_len_4096_pos_alibi_step_595508_gbs256/b11cee03bc8940fa86cb2da19e99fd4e_t5_tokenizer.model"
    # cfg.cfg.override_vocab_size = 40320
    cfg.cfg.task_templates = [{
        "taskname": "squad",
        "prompt_template": "<|VIRTUAL_PROMPT_0|> {context} {question} {answer}",
        "total_virtual_tokens": 3,
        "virtual_token_splits": [3],
        "truncate_field": "context",
        "answer_field": "answer",
    }]
    with tempfile.NamedTemporaryFile(suffix='.yaml') as f:
        OmegaConf.save(config=cfg, f=f.name)
        model = cls.load_from_checkpoint(checkpoint_path=checkpoint, trainer=trainer, hparams_file=f.name,)
        return model

model = load_from_checkpoint_dir(MegatronSpeechGPTSFTModel, gpt_cfg, trainer, checkpoint_path)
print("embedding weight")
print(model.model.module.language_model.embedding.word_embeddings.weight.shape)
print("output weight")
print(model.model.module.language_model.output_layer.weight.shape)
model.save_to("/home/jasoli/models/speechllm_sgpt_bothpre_2b_tp1_pp1--val_loss=22.47-step=79589-consumed_samples=10187392.nemo")


