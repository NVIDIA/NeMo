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


import itertools
import os
from dataclasses import dataclass
from typing import Optional

import pytorch_lightning as ptl
import torch

from nemo.collections.asr.data.audio_to_text_dataset import ASRDatasetConfig, ASRPredictionWriter
from nemo.collections.asr.metrics.rnnt_wer import RNNTDecodingConfig
from nemo.collections.asr.models import ASRModel
from nemo.core.config import TrainerConfig, hydra_runner
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero


@dataclass
class ParallelTranscriptionConfig:
    model: Optional[str] = None  # name
    predict_ds: ASRDatasetConfig = ASRDatasetConfig()
    output_path: Optional[str] = None
    return_predictions: bool = False
    use_cer: bool = False

    # decoding strategy for RNNT models
    rnnt_decoding: RNNTDecodingConfig = RNNTDecodingConfig()
    trainer: TrainerConfig = TrainerConfig(gpus=-1, accelerator="ddp")


@hydra_runner(config_name="TranscriptionConfig", schema=ParallelTranscriptionConfig)
def main(cfg: ParallelTranscriptionConfig):
    if cfg.model.endswith(".nemo"):
        logging.info("Attempting to initialize from .nemo file")
        model = ASRModel.restore_from(restore_path=cfg.model, map_location="cpu")
    elif cfg.model.endswith(".ckpt"):
        logging.info("Attempting to initialize from .ckpt file")
        model = ASRModel.load_from_checkpoint(checkpoint_path=cfg.model, map_location="cpu")
    else:
        logging.info(
            "Attempting to initialize from a pretrained model as the model name does not have the extension of .nemo or .ckpt"
        )
        model = ASRModel.from_pretrained(model_name=cfg.model, map_location="cpu")

    trainer = ptl.Trainer(**cfg.trainer)
    data_loader = model._setup_dataloader_from_config(cfg.predict_ds)

    os.makedirs(cfg.output_path, exist_ok=True)
    # trainer.global_rank is not valid before predict() is called. Need this hack to find the correct global_rank.
    global_rank = trainer.node_rank * trainer.num_gpus + int(os.environ.get("LOCAL_RANK", 0))
    output_file = os.path.join(cfg.output_path, f"predictions_{global_rank}.json")
    predictor_writer = ASRPredictionWriter(dataset=data_loader.dataset, output_file=output_file, use_cer=cfg.use_cer,)
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
        output_file = os.path.join(cfg.output_path, f"predictions_all.json")
        logging.info(f"Prediction files are being aggregated in {output_file}.")
        with open(output_file, 'w') as outf:
            for rank in range(trainer.world_size):
                input_file = os.path.join(cfg.output_path, f"predictions_{rank}.json")
                with open(input_file, 'r') as inpf:
                    lines = inpf.readlines()
                    for line in lines:
                        outf.write(line)
                        samples_num += 1

        logging.info(
            f"Prediction is done for {samples_num} samples in total on all workers and results are aggregated in {output_file}."
        )


if __name__ == '__main__':
    main()
