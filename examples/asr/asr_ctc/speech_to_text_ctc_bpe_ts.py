
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""
# Preparing the Tokenizer for the dataset
Use the `process_asr_text_tokenizer.py` script under <NEMO_ROOT>/scripts/tokenizers/ in order to prepare the tokenizer.

```sh
python <NEMO_ROOT>/scripts/tokenizers/process_asr_text_tokenizer.py \
        --manifest=<path to train manifest files, seperated by commas>
        OR
        --data_file=<path to text data, seperated by commas> \
        --data_root="<output directory>" \
        --vocab_size=<number of tokens in vocabulary> \
        --tokenizer=<"spe" or "wpe"> \
        --no_lower_case \
        --spe_type=<"unigram", "bpe", "char" or "word"> \
        --spe_character_coverage=1.0 \
        --log
```

# Training the model
```sh
python speech_to_text_ctc_bpe.py \
    # (Optional: --config-path=<path to dir of configs> --config-name=<name of config without .yaml>) \
    model.train_ds.manifest_filepath=<path to train manifest> \
    model.validation_ds.manifest_filepath=<path to val/test manifest> \
    model.tokenizer.dir=<path to directory of tokenizer (not full path to the vocab file!)> \
    model.tokenizer.type=<either bpe or wpe> \
    trainer.gpus=-1 \
    trainer.accelerator="ddp" \
    trainer.max_epochs=100 \
    model.optim.name="adamw" \
    model.optim.lr=0.001 \
    model.optim.betas=[0.9,0.999] \
    model.optim.weight_decay=0.0001 \
    model.optim.sched.warmup_steps=2000
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="<Name of experiment>" \
    exp_manager.wandb_logger_kwargs.project="<Name of project>"
```

# Fine-tune a model

For documentation on fine-tuning this model, please visit -
https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/configs.html#fine-tuning-configurations

# Pretrained Models

For documentation on existing pretrained models, please visit -
https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/results.html

"""

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.models.ctc_bpe_ts_models import TSEncDecCTCModelBPE
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="../conf/conformer/", config_name="conformer_ctc_bpe_ts")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    if cfg.do_training:
        asr_model = TSEncDecCTCModelBPE(cfg=cfg.model, trainer=trainer)

        # Initialize the weights of the model from another model, if provided via config
        # asr_model.maybe_init_from_pretrained_checkpoint(cfg)
        if cfg.get('tsasr_checkpoint_path', None) is not None:
            
            checkpoint = TSEncDecCTCModelBPE.restore_from(
                cfg.tsasr_checkpoint_path, map_location=torch.device('cpu')
            )

        if cfg.get('nemo_checkpoint_path', None) is not None:
            checkpoint = EncDecCTCModelBPE.restore_from(
                cfg.nemo_checkpoint_path, map_location=torch.device('cpu'), strict=False
            )
            
            d = checkpoint.state_dict()
            d_new = {}
            for k, v in d.items():
                if k.startswith("encoder.pre_encode"):
                    d_new[k.replace("encoder", "speaker_beam")] = v
                # elif k.startswith("encoder"):
                #     d_new[k.replace("encoder", "speaker_beam")] = v
            d.update(d_new)
            asr_model.load_state_dict(d, strict=False)
            del checkpoint

        if cfg.get('init_from_ptl_ckpt', None) is not None:
            ckpt_path = cfg.get('init_from_ptl_ckpt')
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
            asr_model.load_state_dict(ckpt['state_dict'], strict=False)
            del ckpt

        trainer.fit(asr_model)
    else:
        asr_model = TSEncDecCTCModelBPE.restore_from(
            cfg.nemo_checkpoint_path, map_location=torch.device('cpu'), strict=False,
        )
        # asr_model.load_state_dict(torch.load('/home/yangzhang/code/ts_asr/librispeechmix/ckpts/draco_3mix_ctc_spectL1.0_stopStep-1_speedP0.3_spk3_ep120_bs64_lr0.0003_lr2__schedCosineAnnealing_warm25000_wd1e-2_encfrTrue_decfrFalse_asrEfrFalse_asrDfrFalse_d256_layer18_18_head4_4_nSpecT5_nSpecF2_wSpecF27_v128--val_wer=0.0995-epoch=100.ckpt'), strict=False)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        trainer = pl.Trainer(devices=1, accelerator='gpu')
        asr_model.setup_multiple_test_data(test_data_config=cfg.model.test_ds)
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == '__main__':
    main()
