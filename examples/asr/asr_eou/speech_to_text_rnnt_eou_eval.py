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
Example usage:

0. Prepare dataset based on  <NeMo Root>/nemo/collections/asr/data/audio_to_eou_label_lhotse.py

1. Add special tokens <EOU> and <EOB> to the tokenizer of pretrained model, by refering to the script
    <NeMo Root>/scripts/asr_end_of_utterance/tokenizers/add_special_tokens_to_sentencepiece.py

2. If pretrained model is HybridRNNTCTCBPEModel, convert it to RNNT using the script
   <NeMo Root>/examples/asr/asr_hybrid_transducer_ctc/helpers/convert_nemo_asr_hybrid_to_ctc.py

3. Run the following command to train the ASR-EOU model:
```bash
#!/bin/bash

NEMO_PATH=/home/heh/codes/nemo-eou
export PYTHONPATH=$NEMO_PATH:$PYTHONPATH

TRAIN_MANIFEST=/home/heh/codes/nemo-eou/nemo_experiments/turnGPT_TTS_data/daily_dialogue_test_tts.json
VAL_MANIFEST=/home/heh/codes/nemo-eou/nemo_experiments/turnGPT_TTS_data/daily_dialogue_test_tts.json
NOISE_MANIFEST=/home/heh/codes/nemo-eou/nemo_experiments/noise_manifest.json

PRETRAINED_NEMO=/media/data3/pretrained_models/nemo_asr/stt_en_fastconformer_hybrid_large_streaming_80ms_rnnt.nemo
TOKENIZER_DIR=/media/data3/pretrained_models/nemo_asr/tokenizers/stt_en_fastconformer_hybrid_large_streaming_80ms-eou/

BATCH_DURATION=30
NUM_WORKERS=0
LIMIT_TRAIN_BATCHES=100
VAL_CHECK_INTERVAL=100
MAX_STEPS=1000000

EXP_NAME=fastconformer_transducer_bpe_streaming_eou_debug

SCRIPT=${NEMO_PATH}/examples/asr/asr_eou/speech_to_text_rnnt_eou.py
CONFIG_PATH=${NEMO_PATH}/examples/asr/conf/fastconformer/cache_aware_streaming
CONFIG_NAME=fastconformer_transducer_bpe_streaming

CUDA_VISIBLE_DEVICES=0 python $SCRIPT \
    --config-path $CONFIG_PATH \
    --config-name $CONFIG_NAME \
    ++init_from_nemo_model=$PRETRAINED_NEMO \
    model.encoder.att_context_size="[70,1]" \
    model.tokenizer.dir=$TOKENIZER_DIR \
    model.train_ds.manifest_filepath=$TRAIN_MANIFEST \
    model.train_ds.augmentor.noise.manifest_path=$NOISE_MANIFEST \
    model.validation_ds.manifest_filepath=$VAL_MANIFEST \
    model.train_ds.batch_duration=$BATCH_DURATION \
    model.train_ds.num_workers=$NUM_WORKERS \
    model.validation_ds.batch_duration=$BATCH_DURATION \
    model.validation_ds.num_workers=$NUM_WORKERS \
    ~model.test_ds \
    trainer.limit_train_batches=$LIMIT_TRAIN_BATCHES \
    trainer.val_check_interval=$VAL_CHECK_INTERVAL \
    trainer.max_steps=$MAX_STEPS \
    exp_manager.name=$EXP_NAME
```

"""


from typing import Optional

import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.models import ASRModel, EncDecHybridRNNTCTCBPEModel, EncDecRNNTBPEModel
from nemo.collections.asr.models.asr_eou_models import EncDecRNNTBPEEOUModel
from nemo.collections.asr.modules.rnnt import RNNTDecoder, RNNTJoint
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg


def load_model(cfg, trainer):
    keys = ['init_from_nemo_model', 'init_from_ptl_ckpt', 'init_from_pretrained_model']
    num_keys = sum([1 for key in keys if key in cfg])
    if num_keys > 1:
        raise ValueError(
            f"Only one of the following keys should be present in the config: {keys}. Found {num_keys} keys."
        )
    if "init_from_nemo_model" in cfg:
        model = EncDecRNNTBPEEOUModel.restore_from(cfg.init_from_nemo_model, trainer=trainer)
    elif "init_from_ptl_ckpt" in cfg:
        model = EncDecRNNTBPEEOUModel.load_from_checkpoint(cfg.init_from_ptl_ckpt, trainer=trainer)
    elif "init_from_pretrained_model" in cfg:
        model = EncDecRNNTBPEEOUModel.from_pretrained(cfg.init_from_pretrained_model, trainer=trainer)
    else:
        model = EncDecRNNTBPEEOUModel(cfg=cfg.model, trainer=trainer)
    return model


@hydra_runner(config_path="../conf/asr_eou", config_name="fastconformer_transducer_bpe_streaming")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))

    asr_model = load_model(cfg, trainer)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        asr_model.setup_test_data(test_data_config=cfg.model.test_ds)
        trainer.test(asr_model)
    else:
        raise ValueError(
            "No test dataset provided. Please provide a test dataset in the config file under model.test_ds."
        )
    logging.info("Test completed.")


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
