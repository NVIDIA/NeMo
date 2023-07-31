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

"""

# NOTE: This script is adapted from examples/asr/transcribe_speech_parallel.py
# @kpuvvada: Merge in future


# ASR transcribe/inference with multi-GPU/multi-node support for large datasets
# It supports both tarred and non-tarred datasets
# Arguments
#    model: path to a nemo/PTL checkpoint file or name of a pretrained model
#    predict_ds: config of the dataset/dataloader
#    output_path: path to store the predictions
#    return_predictions: whether to return the predictions as output other than writing into the files
#    use_cer: whether to calculate the error in terms of CER or use the default WER
#
# Results of each GPU/worker is written into a file named 'predictions_{rank}.json, and aggregated results of all workers are written into 'predictions_all.json'

Example for non-tarred datasets:

python transcribe_speech_parallel.py \
    model=stt_en_conformer_ctc_large \
    predict_ds.manifest_filepath=/dataset/manifest_file.json \
    predict_ds.batch_size=16 \
    output_path=/tmp/

Example for tarred datasets:

python transcribe_speech_parallel.py \
    predict_ds.is_tarred=true \
    predict_ds.manifest_filepath=/tarred_dataset/tarred_audio_manifest.json \
    predict_ds.tarred_audio_filepaths=/tarred_dataset/audio__OP_0..127_CL_.tar \
    ...

By default the trainer uses all the GPUs available and default precision is FP32.
By setting the trainer config you may control these configs. For example to do the predictions with AMP on just two GPUs:

python transcribe_speech_parallel.py \
    trainer.precision=16 \
    trainer.devices=2 \
    ...

You may control the dataloader's config by setting the predict_ds:

python transcribe_speech_parallel.py \
    predict_ds.num_workers=8 \
    predict_ds.min_duration=2.0 \
    predict_ds.sample_rate=16000 \
    model=stt_en_conformer_ctc_small \
    ...

"""


import itertools
import json
import os
from dataclasses import dataclass, is_dataclass
from typing import Optional, List, Any
from pathlib import Path

import pytorch_lightning as ptl
import torch
import numpy as np
from omegaconf import MISSING, OmegaConf, DictConfig

from pytorch_lightning.callbacks import BasePredictionWriter
from nemo.core.classes import ModelPT

# from nemo.collections.asr.data.audio_to_text_dataset import ASRPredictionWriter
# from nemo.collections.asr.metrics.rnnt_wer import RNNTDecodingConfig
# from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.models.configs.asr_models_config import ASRDatasetConfig
from nemo.core.config import TrainerConfig, hydra_runner
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero


from nemo.collections.multimodal.models.featex_for_speechllm import FeatExWrapperModel

@dataclass
class ParallelTranscriptionConfig:
    model_cfg: DictConfig = DictConfig({'model': None})  # name
    predict_ds: ASRDatasetConfig = ASRDatasetConfig(return_sample_id=True, num_workers=4)
    out_save_dir: str = MISSING
    out_manifest_dir: str = MISSING

    # when return_predictions is enabled, the prediction call would keep all the predictions in memory and return them when prediction is done
    return_predictions: bool = False
    use_cer: bool = False

    # decoding strategy for RNNT models
    # rnnt_decoding: RNNTDecodingConfig = RNNTDecodingConfig()
    trainer: TrainerConfig = TrainerConfig(devices=-1, accelerator="gpu", strategy="ddp")


def match_train_config(predict_ds, train_ds):
    # It copies the important configurations from the train dataset of the model
    # into the predict_ds to be used for prediction. It is needed to match the training configurations.
    if train_ds is None:
        return

    predict_ds.sample_rate = train_ds.get("sample_rate", 16000)
    cfg_name_list = [
        "int_values",
        "use_start_end_token",
        "blank_index",
        "unk_index",
        "normalize",
        "parser",
        "eos_id",
        "bos_id",
        "pad_id",
    ]

    if is_dataclass(predict_ds):
        predict_ds = OmegaConf.structured(predict_ds)
    for cfg_name in cfg_name_list:
        if hasattr(train_ds, cfg_name):
            setattr(predict_ds, cfg_name, getattr(train_ds, cfg_name))

    return predict_ds


class FeaturePredictionWriter(BasePredictionWriter):
    def __init__(self, dataset, save_dir: str, output_file: str):
        super().__init__(write_interval="batch")
        self.save_dir = save_dir
        self.outf = open(output_file, 'w', encoding='utf-8')
        self.dataset = dataset
        self.samples_num = 0

    def write_on_batch_end(
        self,
        trainer,
        pl_module: 'LightningModule',
        prediction: Any,
        batch_indices: List[int],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        for sample_id, feature_dict in prediction:
            item = {}
            sample = self.dataset.get_manifest_sample(sample_id)

            feature_filename = Path(sample.audio_file).stem + '.npz'
            feature_filepath = os.path.join(self.save_dir, feature_filename)
            np.savez(feature_filepath, **feature_dict)

            # prepare manifest items
            item["audio_filepath"] = sample.audio_file
            item["duration"] = sample.duration
            item["text"] = sample.text_raw
            item["feature_filepath"] = feature_filepath
            self.outf.write(json.dumps(item) + "\n")
            self.samples_num += 1
        return

    def close_output_file(self):
        self.outf.close()
        return self.samples_num



@hydra_runner(config_name="TranscriptionConfig", schema=ParallelTranscriptionConfig)
def main(cfg: ParallelTranscriptionConfig):
    if isinstance(cfg.model_cfg.model, str):
        if cfg.model_cfg.model.endswith(".nemo"):
            logging.info("Attempting to initialize from .nemo file")
            model = ModelPT.restore_from(restore_path=cfg.model, map_location="cpu")
        elif cfg.model_cfg.model.endswith(".ckpt"):
            logging.info("Attempting to initialize from .ckpt file")
            model = ModelPT.load_from_checkpoint(checkpoint_path=cfg.model, map_location="cpu")
        else:
            logging.info(
                "Attempting to initialize from a pretrained model as the model name does not have the extension of .nemo or .ckpt"
            )
            model = ModelPT.from_pretrained(model_name=cfg.model_cfg.model, map_location="cpu")
    else:
        # asr_model is just to get the dataloader facilities
        asr_model = ASRModel.from_pretrained(model_name='stt_en_conformer_ctc_small', map_location="cpu")
        logging.info("Attempting to initialize from model_cfg.model")
        model = FeatExWrapperModel(cfg.model_cfg.model)

    trainer = ptl.Trainer(**cfg.trainer)

    cfg.predict_ds.return_sample_id = True
    cfg.predict_ds = match_train_config(predict_ds=cfg.predict_ds, train_ds=asr_model.cfg.train_ds)
    data_loader = asr_model._setup_dataloader_from_config(cfg.predict_ds)

    os.makedirs(cfg.out_save_dir, exist_ok=True)
    os.makedirs(cfg.out_manifest_dir, exist_ok=True)
    import pdb; pdb.set_trace()
    # trainer.global_rank is not valid before predict() is called. Need this hack to find the correct global_rank.
    global_rank = trainer.node_rank * trainer.num_devices + int(os.environ.get("LOCAL_RANK", 0))
    output_file = os.path.join(cfg.out_manifest_dir, f"manifest_feature_{global_rank}.json")
    predictor_writer = FeaturePredictionWriter(dataset=data_loader.dataset, save_dir=cfg.out_save_dir, output_file=output_file)
    trainer.callbacks.extend([predictor_writer])

    predictions = trainer.predict(model=model, dataloaders=data_loader, return_predictions=cfg.return_predictions)
    if predictions is not None:
        predictions = list(itertools.chain.from_iterable(predictions))
    samples_num = predictor_writer.close_output_file()

    logging.info(
        f"Prediction on rank {global_rank} is done for {samples_num} samples and results are stored in {output_file}."
    )

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    samples_num = 0

    if is_global_rank_zero():
        output_file = os.path.join(cfg.out_manifest_dir, f"manifest_feature_all.json")
        logging.info(f"Manifest files are being aggregated in {output_file}.")
        with open(output_file, 'w') as outf:
            for rank in range(trainer.world_size):
                input_file = os.path.join(cfg.out_manifest_dir, f"manifest_feature_{rank}.json")
                with open(input_file, 'r') as inpf:
                    lines = inpf.readlines()
                    for line in lines:
                        item = json.loads(line)
                        outf.write(json.dumps(item) + "\n")
                        samples_num += 1
        logging.info(
            f"Prediction is done for {samples_num} samples in total on all workers and results are aggregated in {output_file}."
        )


if __name__ == '__main__':
    main()
