## NOTE: This script is present for github-actions testing only.
## There are no guarantees that this script is up-to-date with latest NeMo.

import argparse

from megatron.core.optimizer import OptimizerConfig
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.api import train
from nemo.collections.llm.t5.data import PreTrainingDataModule
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning import NeMoLogger
from nemo.lightning.pytorch.callbacks import ModelCheckpoint
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule


def get_args():
    parser = argparse.ArgumentParser(description='Train a small T5 model using NeMo 2.0')
    parser.add_argument('--devices', type=int, help="Number of devices to use for training")
    parser.add_argument('--max-steps', type=int, help="Number of steps to train for")
    parser.add_argument('--experiment-dir', type=str, help="directory to write results and checkpoints to")
    parser.add_argument('--data-path', type=str, help="Path to data file")
    parser.add_argument('--vocab-path', type=str, help="Path to vocab file")
    parser.add_argument('--merges-path', type=str, help="Path to merges file")
    parser.add_argument('--index-mapping-dir', type=str, help="directory to write index mappings to")

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    seq_length = 2048

    tokenizer = get_nmt_tokenizer(
        "megatron",
        "BertWordPieceCase",
        vocab_file=args.vocab_path,
    )
    additional_tokens = {
        'additional_special_tokens': [
            f'<extra_id_{i}>' for i in range(100)
        ]
    }
    tokenizer.add_special_tokens(additional_tokens)

    data = PreTrainingDataModule(
        paths=args.data_path,
        seq_length=512,
        seq_length_dec=128,
        micro_batch_size=64,
        global_batch_size=512,
        seed=1234,
        tokenizer=tokenizer,
        split="99982,9,9",
    )
    t5_config = llm.t5.model.t5.T5Config(
        num_layers=12,
        encoder_num_layers=12,
        hidden_size=768,
        ffn_hidden_size=3072,
        num_attention_heads=12,
        kv_channels=64,
        init_method_std=0.015,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        layernorm_epsilon=1e-5,
        make_vocab_size_divisible_by=128,
        max_position_embeddings=seq_length,
    )
    model = llm.t5.model.t5.T5Model(t5_config, tokenizer=data.tokenizer)
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
    )
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=5000,
        enable_nemo_ckpt_io=False,
    )
    callbacks = [checkpoint_callback]

    loggers = []
    tensorboard_logger = TensorBoardLogger(
        save_dir='dummy',  ## NOTE: this gets overwritten by default
    )
    loggers.append(tensorboard_logger)

    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=0.0001,
        min_lr=0.00001,
        use_distributed_optimizer=False,
        bf16=True,
    )
    opt = MegatronOptimizerModule(config=opt_config)

    trainer = nl.Trainer(
        devices=args.devices,
        max_steps=args.max_steps,
        accelerator="gpu",
        strategy=strategy,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=100,
        limit_val_batches=2,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed", amp_O2=False),
    )

    nemo_logger = NeMoLogger(
        dir=args.experiment_dir,
    )

    train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        tokenizer='data',
        optim=opt,
    )
