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

from argparse import ArgumentParser

import pytorch_lightning as pl

import nemo.collections.nlp as nemo_nlp
from nemo.collections.nlp.models.text_classifier_model import BERTTextClassifier


def main():
    parser = ArgumentParser()
    parser.add_argument(description='Sentence classification with pretrained BERT models')
    parser.add_argument("--work_dir", default='outputs', type=str)
    parser.add_argument(
        "--checkpoint_dir",
        default=None,
        type=str,
        help="The folder containing the checkpoints for the model to continue training",
    )
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument(
        '--pretrained_model_name',
        default='roberta-base',
        type=str,
        help='Name of the pre-trained model',
        choices=nemo_nlp.nm.trainables.get_pretrained_lm_models_list(),
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
    parser.add_argument("--num_gpus", default=1, type=int, help="Number of GPUs")
    parser.add_argument("--num_output_layers", default=2, type=int, help="Number of layers in the Classifier")
    parser.add_argument("--num_epochs", default=10, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--num_train_samples", default=-1, type=int, help="Number of samples to use for training")
    parser.add_argument("--num_eval_samples", default=-1, type=int, help="Number of samples to use for evaluation")
    parser.add_argument("--optimizer_kind", default="adam", type=str, help="Optimizer kind")
    parser.add_argument("--lr_warmup_proportion", default=0.1, type=float, help="Learning rate warm up proportion")
    parser.add_argument("--lr", default=2e-5, type=float, help="Initial learning rate")
    parser.add_argument("--lr_policy", default="WarmupAnnealing", type=str, help="Learning rate policy")
    parser.add_argument(
        "--amp_level", default="O0", type=str, choices=["O0", "O1", "O2"], help="01/02 to enable mixed precision"
    )
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay.")
    parser.add_argument("--fc_dropout", default=0.1, type=float, help="Dropout rate")
    parser.add_argument(
        "--use_cache", action='store_true', help="When specified loads and stores cache preprocessed data."
    )
    parser.add_argument("--train_file_prefix", default='train', type=str, help="train file prefix")
    parser.add_argument("--eval_file_prefix", default='dev', type=str, help="eval file prefix")
    parser.add_argument("--class_balancing", default="None", type=str, choices=["None", "weighted_loss"])
    parser.add_argument(
        "--no_shuffle_data", action='store_false', dest="shuffle_data", help="Shuffle is enabled by default."
    )
    parser.add_argument("--save_epoch_freq", default=1, type=int, help="Epoch frequency of saving checkpoints")
    parser.add_argument("--save_step_freq", default=-1, type=int, help="Step frequency of saving checkpoints")
    parser.add_argument('--loss_step_freq', default=25, type=int, help='Frequency of printing loss')
    parser.add_argument('--eval_step_freq', default=100, type=int, help='Frequency of evaluation')
    parser.add_argument("--local_rank", default=None, type=int, help="For distributed training: local_rank")

    args = parser.parse_args()

    text_classification_model = BERTTextClassifier(
        data_dir=args.data_dir,
        pretrained_model_name=args.pretrained_model_name,
        num_output_layers=args.num_output_layers,
        fc_dropout=args.fc_dropout,
        class_balancing=args.class_balancing,
    )

    dataloader_params = {
        "max_seq_length": args.max_seq_length,
        "num_samples": args.num_samples,
        "shuffle": args.shuffle,
        "use_cache": args.use_cache,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }
    text_classification_model.setup_dataloaders(
        data_dir=args.data_dir,
        train_file_prefix=args.train_file_prefix,
        val_file_prefix=args.eval_file_prefix,
        dataloader_params=dataloader_params,
    )

    optim_params = {
        'optimizer_kind': args.optimizer_kind,
        'lr': args.lr,
        'lr_policy': args.lr_policy,
        'weight_decay': args.weight_decay,
        'lr_warmup_proportion': args.lr_warmup_proportion,
    }

    text_classification_model.setup_optimization(optim_params=optim_params)

    trainer = pl.Trainer(
        val_check_interval=args.eval_step_freq,
        amp_level=args.amp_level,
        precision=16,
        gpus=args.num_gpus,
        max_epochs=args.num_epochs,
        distributed_backend='ddp',
    )
    trainer.fit(text_classification_model)


if __name__ == '__main__':
    main()
