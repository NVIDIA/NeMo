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


import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.models.text_classification import TextClassificationModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="text_classification_config")
def main(cfg: DictConfig) -> None:
    logging.info(f'\nConfig Params:\n{cfg.pretty()}')
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    if not cfg.model.train_ds.file_path:
        raise ValueError("'train_ds.file_path' need to be set for the training!")
        return -1
    model = TextClassificationModel(cfg.model, trainer=trainer)
    logging.info("================================================================================================")
    logging.info('Starting training...')
    trainer.fit(model)
    logging.info('Training finished!')
    logging.info("================================================================================================")

    if cfg.model.nemo_path:
        model.save_to(cfg.model.nemo_path)
        logging.info('Model is saved into `.nemo` file: cfg.model.nemo_path')

    # We evaluate the trained model on the test set if test_ds is set in the config file
    if cfg.model.test_ds.file_path:
        logging.info(
            "================================================================================================"
        )
        logging.info("Starting the testing of the trained model on test set...")
        # The latest checkpoint would be used, set ckpt_path to 'best' to use the best one
        trainer.test(model=model, ckpt_path=None, verbose=False)
        logging.info("Testing finished!")
        logging.info(
            "================================================================================================"
        )

    """
    After model training is done, if you have saved the checkpoints, you can create the model from 
    the checkpoint again and evaluate it on a data file. 
    You need to set or pass the test dataloader, and also create a trainer for this.
    """
    if cfg.model.validation_ds.file_path:
        logging.info(
            "================================================================================================"
        )
        logging.info("Starting the evaluating the the best checkpoint on a data file (validation set by default)...")
        # extract the path of the best checkpoint from the training, you may update it to any checkpoint
        checkpoint_path = trainer.checkpoint_callback.best_model_path
        # Create an evaluation model and load the checkpoint
        eval_model = TextClassificationModel.load_from_checkpoint(checkpoint_path=checkpoint_path)

        # create a dataloader config for evaluation, the same data file provided in validation_ds is used here
        # file_path can get updated with any file
        eval_config = OmegaConf.create(
            {'file_path': cfg.model.validation_ds.file_path, 'batch_size': 64, 'shuffle': False, 'num_samples': -1}
        )
        eval_model.setup_test_data(test_data_config=eval_config)
        # eval_dataloader = eval_model._create_dataloader_from_config(cfg=eval_config, mode='test')

        # a new trainer is created to show how to evaluate a checkpoint from an already trained model
        # create a copy of the trainer config and update it to be used for final evaluation
        eval_trainer_cfg = cfg.trainer.copy()
        eval_trainer_cfg.gpus = (
            1 if torch.cuda.is_available() else 0
        )  # it is safer to perform evaluation on single GPU as PT is buggy with the last batch on multi-GPUs
        eval_trainer_cfg.distributed_backend = None  # 'ddp' is buggy with test process in the current PT, it looks like it has been fixed in the latest master
        eval_trainer = pl.Trainer(**eval_trainer_cfg)

        eval_trainer.test(model=eval_model, verbose=False)  # test_dataloaders=eval_dataloader,

        logging.info("Evaluation the best checkpoint finished!")
        logging.info(
            "================================================================================================"
        )

    else:
        logging.info("No file_path was set for validation_ds, so final evaluation is skipped!")

    # You may create a model from a saved chechpoint and use the model.infer() method to
    # perform inference on a list of queries. There is no need of any trainer for inference.
    logging.info("================================================================================================")
    logging.info("Starting the inference on some sample queries...")
    queries = [
        'by the end of no such thing the audience , like beatrice , has a watchful affection for the monster .',
        'director rob marshall went out gunning to make a great one .',
        'uneasy mishmash of styles and genres .',
    ]

    # extract the path of the best checkpoint from the training, you may update it to any checkpoint
    infer_model = TextClassificationModel.load_from_checkpoint(checkpoint_path=checkpoint_path)

    results = infer_model.infer(queries=queries, batch_size=16)

    logging.info('The prediction results of some sample queries with the trained model:')
    for query, result in zip(queries, results):
        logging.info(f'Query : {query}')
        logging.info(f'Predicted label: {result}')

    logging.info("Inference finished!")
    logging.info("================================================================================================")


if __name__ == '__main__':
    main()
