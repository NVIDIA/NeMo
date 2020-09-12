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
from omegaconf import OmegaConf, DictConfig

from nemo.collections.nlp.models.text_classification import TextClassificationModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="text_classification_config")
def main(cfg: DictConfig) -> None:
    logging.info(f'Config Params:\n {cfg.pretty()}')
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    model = TextClassificationModel(cfg.model, trainer=trainer)
    logging.info('Starting training...')
    trainer.fit(model)
    logging.info('Training finished!')

    if cfg.model.nemo_path:
        model.save_to(cfg.model.nemo_path)
        logging.info('Model is saved into `.nemo` file: cfg.model.nemo_path')

    # We evaluate the trained model on the test set if test_ds is set in the config file
    if cfg.model.test_ds.file_path:
        logging.info("Testing the trained model on test set:")
        # The latest checkpoint would be used, set ckpt_path to 'best' to use the best one
        trainer.test(model=model, ckpt_path=None, verbose=False)

    """
    After model training is done, you can use the model for inference.
    You can either evaluate data from a text file that follows the training data format,
    or provide a list of queries you want to add entities to
    """
    if cfg.model.validation_ds.file_path:
        logging.info("Evaluating the trained model on the validation file:")

        # create a copy of the trainer config and update it to be used for final evaluation
        eval_trainer_cfg = cfg.trainer.copy()
        # it is safer to perform evaluation on single GPU as PT is buggy with the last batch on multi-GPUs
        eval_trainer_cfg.gpus = 1
        # 'ddp' is buggy with test process in the current PT, it looks like it has been fixed in the latest master
        eval_trainer_cfg.distributed_backend = None
        eval_trainer = pl.Trainer(**eval_trainer_cfg)

        # create a dataloader config for evaluation, the same data file provided in validation_ds is used
        # file_path can get updated with any file
        eval_config = OmegaConf.create({'file_path': cfg.model.validation_ds.file_path, 'batch_size': 64, 'shuffle': False, 'num_samples': -1})
        eval_dataloader = model._create_dataloader_from_config(cfg=eval_config, mode='test')

        # By setting ckpt_path to 'best', it uses the best checkpoint in the logging folder of the model
        # it can be set to the path of a checkpoint you wish to test or set to None to use the weights from the last epoch
        eval_trainer.test(model=model, test_dataloaders=eval_dataloader, ckpt_path='best', verbose=False)

        # eval_trainer.fit(model=model, val_dataloaders=eval_dataloader)

        # eval_trainer.model = model
        # model.to(eval_trainer)
        # eval_trainer.run_evaluation(test_mode=False)
        # model.eval()
        # trainer.reset_val_dataloader(model)
        # trainer.run_evaluation(test_mode=False)
        # model.train()
        # # create a trainer and a model for evaluation
        # eval_trainer = pl.Trainer(**cfg.trainer)
        # eval_model = TextClassificationModel(cfg.model, trainer=eval_trainer)
        #
        # # create a config object for evaluation
        # # we would evaluate on the validation data specified in the config file, you may update the file_path to evaluate on other files
        # eval_config = OmegaConf.create({'file_path': './data/SST-2/dev.tsv', 'batch_size': 64, 'shuffle': False, 'num_samples': -1})
        # # setup the validation dataloader again
        # eval_model.setup_validation_data(eval_config)
        # # run evaluation on the test data
        # logging.info('Evaluating the model on the validation data...')
        # # set max_step to 0 to prevent any training steps to be done
        # eval_trainer.max_steps = 0
        # eval_trainer.fit(eval_model)
    else:
        logging.info("No file_path was set for validation_ds, so final evaluation is skipped!")

    # run inference on a few examples
    queries = [
        'we bought four shirts from the nvidia gear store in santa clara.',
        'Nvidia is a company.',
        'The Adventures of Tom Sawyer by Mark Twain is an 1876 novel about a young boy growing up along the Mississippi River.',
    ]
    results = model.infer(queries=queries, batch_size=64)

    logging.info('The prediction results of some sample queries with the trained model:')
    for query, result in zip(queries, results):
        logging.info(f'Query : {query}')
        logging.info(f'Predicted label: {result}')


if __name__ == '__main__':
    main()
