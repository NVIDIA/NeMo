# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
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
# =============================================================================

# TODO: WIP

import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch

# import nemo.collections.nlp as nemo_nlp
from nemo.collections.nlp.models.text_classifier_model import BERTTextClassifier


def main():
    parser = ArgumentParser(description='Sentence classification with pretrained BERT models')

    # Data Arguments
    parser.add_argument("--work_dir", default='outputs', type=str)
    parser.add_argument(
        "--checkpoint_dir",
        default=None,
        type=str,
        help="The folder containing the checkpoints for the model to continue training",
    )
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--file_prefix_train", default='train', type=str, help="train file prefix")
    parser.add_argument("--file_prefix_val", default='dev', type=str, help="eval file prefix")
    parser.add_argument("--no_shuffle", action='store_false', dest="shuffle", help="Shuffle is enabled by default.")
    parser.add_argument("--num_samples_train", default=-1, type=int, help="Number of samples to use for training")
    parser.add_argument("--num_samples_val", default=-1, type=int, help="Number of samples to use for evaluation")
    parser.add_argument("--num_workers", default=2, type=int, help="The number of workers for the data loaders.")
    parser.add_argument(
        "--pin_memory", action='store_true', help="Whether to enable the pin_memory feature of the data loaders."
    )
    parser.add_argument(
        "--use_cache", action='store_true', help="When specified loads and stores cache preprocessed data."
    )

    # BERT Arguments
    parser.add_argument(
        '--pretrained_model_name',
        default='roberta-base',
        type=str,
        help='Name of the pre-trained model',
        # choices=nemo_nlp.modules.common.get_pretrained_lm_models_list(),
    )
    parser.add_argument("--bert_checkpoint", default=None, type=str, help="Path to pre-trained BERT checkpoint")
    parser.add_argument("--bert_config", default=None, type=str, help="Path to bert config file in json format")
    parser.add_argument(
        "--tokenizer",
        default="nemobert",
        type=str,
        choices=["nemobert", "sentencepiece"],
        help="tokenizer to use, only relevant when using custom pretrained checkpoint.",
    )
    parser.add_argument("--vocab_file", default=None, type=str, help="Path to the vocab file.")
    parser.add_argument(
        "--tokenizer_model",
        default=None,
        type=str,
        help="Path to pretrained tokenizer model, only used if --tokenizer is sentencepiece",
    )
    parser.add_argument(
        "--do_lower_case",
        action='store_true',
        help="Whether to lower case the input text. True for uncased models, False for cased models. "
        + "For tokenizer only applicable when tokenizer is build with vocab file.",
    )
    parser.add_argument("--batch_size", default=32, type=int, help="Training and evaluation batch size")
    parser.add_argument(
        "--max_seq_length",
        default=36,
        type=int,
        help="The maximum total input sequence length after tokenization.Sequences longer than this will be \
                        truncated, sequences shorter will be padded.",
    )

    # Model Arguments
    parser.add_argument("--num_output_layers", default=2, type=int, help="Number of layers in the Classifier")
    parser.add_argument("--fc_dropout", default=0.1, type=float, help="Dropout rate")
    parser.add_argument("--class_balancing", default="None", type=str, choices=["None", "weighted_loss"])

    # Training Arguments
    parser.add_argument("--num_gpus", default=1, type=int, help="Number of GPUs")
    parser.add_argument("--num_epochs", default=100, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--local_rank", default=None, type=int, help="For distributed training: local_rank")

    # Optimization Arguments
    parser.add_argument(
        "--amp_level", default="O0", type=str, choices=["O0", "O1", "O2"], help="01/02 to enable mixed precision"
    )
    parser.add_argument("--optimizer_kind", default="adam", type=str, help="Optimizer kind")
    parser.add_argument("--lr_warmup_proportion", default=0.1, type=float, help="Learning rate warm up proportion")
    parser.add_argument("--lr", default=2e-5, type=float, help="Initial learning rate")
    parser.add_argument("--lr_policy", default="WarmupAnnealing", type=str, help="Learning rate policy")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay.")
    parser.add_argument("--beta1", default=0.9, type=float, help="Beta1 parameter of Adam.")
    parser.add_argument("--beta2", default=0.999, type=float, help="Beta2 parameter of Adam.")

    # Validation Arguments
    parser.add_argument("--save_epoch_freq", default=1, type=int, help="Epoch frequency of saving checkpoints")
    parser.add_argument("--save_step_freq", default=-1, type=int, help="Step frequency of saving checkpoints")
    parser.add_argument('--loss_step_freq', default=25, type=int, help='Frequency of printing loss')
    parser.add_argument('--eval_step_freq', default=50, type=int, help='Frequency of evaluation')

    args = parser.parse_args()

    text_classification_model = BERTTextClassifier(
        data_dir=args.data_dir,
        pretrained_model_name=args.pretrained_model_name,
        num_output_layers=args.num_output_layers,
        fc_dropout=args.fc_dropout,
        class_balancing=args.class_balancing,
    )

    dataloader_params_train = {
        "max_seq_length": args.max_seq_length,
        "num_samples": args.num_samples_train,
        "shuffle": args.shuffle,
        "use_cache": args.use_cache,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }

    text_classification_model.setup_training_data(
        file_path=os.path.join(args.data_dir, f'{args.file_prefix_train}.tsv'),
        dataloader_params=dataloader_params_train,
    )

    dataloader_params_val = {
        "max_seq_length": args.max_seq_length,
        "num_samples": args.num_samples_val,
        "shuffle": False,
        "use_cache": args.use_cache,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }

    text_classification_model.setup_validation_data(
        file_path=os.path.join(args.data_dir, f'{args.file_prefix_val}.tsv'), dataloader_params=dataloader_params_val,
    )

    # optim_params = {
    #     'optimizer_kind': args.optimizer_kind,
    #     'lr': args.lr,
    #     'lr_policy': args.lr_policy,
    #     'weight_decay': args.weight_decay,
    #     'lr_warmup_proportion': args.lr_warmup_proportion,
    # }
    # text_classification_model.setup_optimization(optim_params=optim_params)

    optimizer = torch.optim.Adam(
        text_classification_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )
    text_classification_model.set_optimizer(optimizer=optimizer)

    trainer = pl.Trainer(
        val_check_interval=args.eval_step_freq,
        amp_level=args.amp_level,
        precision=32 if args.amp_level == "O0" else 16,  # TODO: How to set precision?
        gpus=args.num_gpus,
        max_epochs=args.num_epochs,
        distributed_backend=None if args.num_gpus == 1 else "ddp", #TODO: How to switch between multi-gpu, single-gpu, multi-node here?
    )
    trainer.fit(text_classification_model)


if __name__ == '__main__':
    main()
