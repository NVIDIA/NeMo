from functools import partial

import nemo_run as run
from pytorch_lightning.loggers import TensorBoardLogger

from nemo.collections import llm
from nemo.lightning import NeMoLogger

pretrain = partial(llm.llama3_8b.pretrain_recipe, num_nodes=1, num_gpus_per_node=1)(name="mixtral", dir="/home/models")

pretrain.trainer.val_check_interval = 5
pretrain.trainer.max_epochs = None
pretrain.log.ckpt.save_top_k = -1
# pretrain.log.ckpt.train_time_interval = None
pretrain.trainer.log_every_n_steps = 1
# pretrain.log.ckpt.every_n_train_steps = 5

pretrain.trainer.strategy.context_parallel_size = 1
pretrain.trainer.strategy.expert_model_parallel_size = 1
pretrain.trainer.strategy.pipeline_model_parallel_size = 1
pretrain.trainer.strategy.virtual_pipeline_model_parallel_size = None

pretrain.model.config.num_layers = 8
pretrain.model.config.hidden_size = 576
pretrain.model.config.ffn_hidden_size = 576 * 4
# pretrain.model.config.num_gqa_groups = 4
pretrain.model.config.num_attention_heads = 8

pretrain.data.seq_length = 128
pretrain.data.global_batch_size = 16
pretrain.trainer.max_steps = 5

from nemo.collections.llm.api import train

loggers = []
tensorboard_logger = TensorBoardLogger(
    save_dir='dummy',  ## NOTE: this gets overwritten by default
)
loggers.append(tensorboard_logger)
pretrain.trainer.logger = loggers

nemo_logger = NeMoLogger(
    log_dir='/home/models/mixtral',
)

run.run(pretrain)
