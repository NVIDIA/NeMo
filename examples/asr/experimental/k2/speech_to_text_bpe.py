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
python speech_to_text_bpe.py \
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
"""
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, open_dict

from nemo.collections.asr.models.k2_sequence_models import EncDecK2SeqModelBPE
from nemo.collections.asr.models.configs.k2_sequence_models_config import EncDecK2SeqModelConfig
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="experimental/configs/", config_name="config_bpe")
def main(cfg):
    # Generate default asr model config
    asr_model_config = EncDecK2SeqModelConfig()

    # Merge hydra updates with model config
    # `drop_missing_subconfig=True` is necessary here. Without it, while the data class will instantiate and be added
    # to the config, it contains test_ds.sample_rate = MISSING and test_ds.labels = MISSING.
    # This will raise a OmegaConf MissingMandatoryValue error when processing the dataloaders inside
    # model_utils.resolve_test_dataloaders(model=self) (used for multi data loader support).
    # In general, any operation that tries to use a DictConfig with MISSING in it will fail,
    # other than explicit update operations to change MISSING to some actual value.
    asr_model_config = update_model_config(asr_model_config, cfg, drop_missing_subconfigs=True)
    
    # From here on out, its a general OmegaConf DictConfig, directly usable by our code.
    logging.info(f"Hydra config: {OmegaConf.to_yaml(asr_model_config)}")
    print(OmegaConf.to_yaml(asr_model_config))
    trainer = pl.Trainer(**asr_model_config.trainer)
    exp_manager(trainer, asr_model_config.get("exp_manager", None))

    with open_dict(asr_model_config):
        restore_path = asr_model_config.pop("init_from_nemo", None)

    asr_model = EncDecK2SeqModelBPE(asr_model_config=asr_model_config.model, trainer=trainer)

    if restore_path is not None:
        checkpoint = EncDecK2SeqModelBPE.restore_from(restore_path, map_location=torch.device("cpu"))

        try:
            asr_model.encoder.load_state_dict(checkpoint.encoder.state_dict(), strict=False)
            logging.info("Loaded encoder checkpoint")
        except Exception:
            logging.info("Could not load encoder checkpoint")

        try:
            asr_model.decoder.load_state_dict(checkpoint.decoder.state_dict(), strict=False)
            logging.info("Loaded decoder checkpoint")
        except Exception:
            logging.info("Could not load decoder checkpoint")

        del checkpoint

    trainer.fit(asr_model)

    if hasattr(asr_model_config.model, "test_ds") and asr_model_config.model.test_ds.manifest_filepath is not None:
        gpu = 1 if asr_model_config.trainer.gpus != 0 else 0
        test_trainer = pl.Trainer(
            gpus=gpu,
            precision=trainer.precision,
            amp_level=trainer.accelerator_connector.amp_level,
            amp_backend=asr_model_config.trainer.get("amp_backend", "native"),
        )
        if asr_model.prepare_test(test_trainer):
            test_trainer.test(asr_model)


if __name__ == "__main__":
    main()
