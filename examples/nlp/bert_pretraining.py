#!/usr/bin/env python3
# Copyright (c) 2019 NVIDIA Corporation

"""

To pretrain BERT on raw text dataset run
python bert_pretraining.py \
--amp_opt_level "O0" \
--data_dir data/lm/wikitext-2 \
--dataset_name wikitext-2 \
--work_dir outputs/bert_lm \
--batch_size 64 \
--lr 0.01 \
--lr_policy CosineAnnealing \
--lr_warmup_proportion 0.05 \
--optimizer novograd \
--beta1 0.95 \
--beta2 0.25 \
--tokenizer sentence-piece \
--vocab_size 3200 \
--hidden_size 768 \
--intermediate_size 3072 \
--num_hidden_layers 12 \
--num_attention_heads 12 \
--hidden_act "gelu" \
--save_step_freq 200 \
--num_epochs 10 \
--sample_size 10000000 \
--mask_probability 0.15 \
--short_seq_prob 0.1

To pretrain BERT on preprocessed dataset,
download and preprocess dataset from here:
https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/
Run the script:
./data/create_datasets_from_start.sh
and extract data into data_dir

Then run phase 1 with
python -m torch.distributed.launch --nproc_per_node=8 bert_pretraining.py \
--batch_size 64 \
--config_file bert_config.json
--data_dir data_dir \
--save_step_freq 200 \
--num_epochs 4 \
--total_iterations_per_gpu 7038 \
--num_gpus 8 \
--batches_per_step 128 \
--amp_opt_level "O2" \
--lr_policy SquareRootAnnealing \
--beta1 0.9 \
--beta2 0.999 \
--lr_warmup_proportion 0.2843 \
--optimizer fused_lamb \
--weight_decay 0.01 \
--lr 0.006 \
--preprocessed_data
"""
import argparse
import os

import nemo
from nemo.utils.lr_policies import get_lr_policy
from pytorch_transformers import BertConfig
import nemo_nlp
from nemo_nlp.data.datasets.utils import BERTPretrainingDataDesc
from nemo_nlp.utils.callbacks.bert_pretraining import \
    eval_iter_callback, eval_epochs_done_callback


parser = argparse.ArgumentParser(description='BERT pretraining')
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--num_epochs", default=10, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--batches_per_step", default=1, type=int)
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--lr_policy", default=None, type=str)
parser.add_argument("--lr_warmup_proportion", default=0.05, type=float)
parser.add_argument("--optimizer", default="novograd", type=str)
parser.add_argument("--beta1", default=0.95, type=float)
parser.add_argument("--beta2", default=0.25, type=float)
parser.add_argument("--amp_opt_level",
                    default="O0",
                    type=str,
                    choices=["O0", "O1", "O2"])
parser.add_argument("--weight_decay", default=0.0, type=float)
parser.add_argument("--tokenizer",
                    default="sentence-piece",
                    type=str,
                    choices=["sentence-piece", "nemo-bert"])
parser.add_argument("--max_seq_length", default=128, type=int)
parser.add_argument("--sample_size", default=1e7, type=int)
parser.add_argument("--mask_probability", default=0.15, type=float)
parser.add_argument("--short_seq_prob", default=0.1, type=float)
parser.add_argument("--vocab_size", default=3200, type=int)
parser.add_argument("--hidden_size", default=768, type=int)
parser.add_argument("--intermediate_size", default=3072, type=int)
parser.add_argument("--num_hidden_layers", default=12, type=int)
parser.add_argument("--num_attention_heads", default=12, type=int)
parser.add_argument("--hidden_act", default="gelu", type=str)
parser.add_argument("--max_predictions_per_seq", default=20, type=int,
                    help="maximum number of masked tokens to predict,\
                    needed when --preprocessed_data is specified")
parser.add_argument("--data_dir", default="data/lm/wikitext-2", type=str)
parser.add_argument("--preprocessed_data", action="store_true",
                    default=False, help="specify if using preprocessed data")
parser.add_argument("--gradient_predivide", action="store_true",
                    default=False, help="use gradient predivide")
parser.add_argument("--total_iterations_per_gpu", default=-1,
                    type=int, help="if specified overrides --num_epochs")
parser.add_argument("--dataset_name", default="wikitext-2", type=str)
parser.add_argument("--load_dir", default=None, type=str)
parser.add_argument("--work_dir", default="outputs/bert_lm", type=str)
parser.add_argument("--save_epoch_freq", default=1, type=int)
parser.add_argument("--save_step_freq", default=100, type=int)
parser.add_argument("--config_file", default=None, type=str,
                    help="The BERT model config")
args = parser.parse_args()

print(vars(args))
nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch,
                                   local_rank=args.local_rank,
                                   optimization_level=args.amp_opt_level,
                                   log_dir=args.work_dir,
                                   create_tb_writer=True,
                                   files_to_copy=[__file__],
                                   add_time_to_log_dir=True)

if args.config_file is not None:
    config = BertConfig.from_json_file(args.config_file).to_dict()
    args.vocab_size = config['vocab_size']
    args.hidden_size = config['hidden_size']
    args.num_hidden_layers = config['num_hidden_layers']
    args.num_attention_heads = config['num_attention_heads']
    args.intermediate_size = config['intermediate_size']
    args.hidden_act = config['hidden_act']
    args.max_seq_length = config['max_position_embeddings']

if not args.preprocessed_data:
    special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
    data_desc = BERTPretrainingDataDesc(args.dataset_name,
                                        args.data_dir,
                                        args.vocab_size,
                                        args.sample_size,
                                        special_tokens,
                                        'train.txt')
    if args.tokenizer == "sentence-piece":
        nf.logger.info("To use SentencePieceTokenizer.")
        tokenizer = nemo_nlp.SentencePieceTokenizer(
            model_path=data_desc.tokenizer_model)
        tokenizer.add_special_tokens(special_tokens)
    elif args.tokenizer == "nemo-bert":
        nf.logger.info("To use NemoBertTokenizer.")
        vocab_file = os.path.join(args.data_dir, 'vocab.txt')
        # To train on a Chinese dataset, use NemoBertTokenizer
        tokenizer = nemo_nlp.NemoBertTokenizer(vocab_file=vocab_file)
    else:
        raise ValueError("Please add your tokenizer "
                         "or use sentence-piece or nemo-bert.")
    args.vocab_size = tokenizer.vocab_size

bert_model = nemo_nlp.huggingface.BERT(
    vocab_size=args.vocab_size,
    num_hidden_layers=args.num_hidden_layers,
    hidden_size=args.hidden_size,
    num_attention_heads=args.num_attention_heads,
    intermediate_size=args.intermediate_size,
    max_position_embeddings=args.max_seq_length,
    hidden_act=args.hidden_act
    )

""" create necessary modules for the whole translation pipeline, namely
data layers, BERT encoder, and MLM and NSP loss functions
"""
mlm_classifier = nemo_nlp.TokenClassifier(args.hidden_size,
                                          num_classes=args.vocab_size,
                                          num_layers=1,
                                          log_softmax=True)
mlm_loss_fn = nemo_nlp.MaskedLanguageModelingLossNM()

nsp_classifier = nemo_nlp.SequenceClassifier(args.hidden_size,
                                             num_classes=2,
                                             num_layers=2,
                                             log_softmax=True)
nsp_loss_fn = nemo.backends.pytorch.common.CrossEntropyLoss()

bert_loss = nemo_nlp.LossAggregatorNM(num_inputs=2)

# tie weights of MLM softmax layer and embedding layer of the encoder
mlm_classifier.mlp.last_linear_layer.weight = \
    bert_model.bert.embeddings.word_embeddings.weight


def create_pipeline(data_file,
                    batch_size,
                    preprocessed_data=False,
                    batches_per_step=1,
                    **kwargs):

    if not preprocessed_data:
        max_seq_length, mask_probability, short_seq_prob =\
            kwargs['max_seq_length'], kwargs['mask_probability'],\
            kwargs['short_seq_prob']
        data_layer = nemo_nlp.BertPretrainingDataLayer(
                                                    tokenizer,
                                                    data_file,
                                                    max_seq_length,
                                                    mask_probability,
                                                    short_seq_prob,
                                                    batch_size=batch_size)
    else:
        training, max_predictions_per_seq =\
            kwargs['training'], kwargs['max_predictions_per_seq']
        data_layer = nemo_nlp.BertPretrainingPreprocessedDataLayer(
                            data_file,
                            max_predictions_per_seq,
                            batch_size=batch_size, training=training)

    steps_per_epoch = \
        len(data_layer) // (batch_size * args.num_gpus * batches_per_step)

    input_ids, input_type_ids, input_mask, \
        output_ids, output_mask, nsp_labels = data_layer()
    hidden_states = bert_model(input_ids=input_ids,
                               token_type_ids=input_type_ids,
                               attention_mask=input_mask)
    mlm_logits = mlm_classifier(hidden_states=hidden_states)
    mlm_loss = mlm_loss_fn(logits=mlm_logits,
                           output_ids=output_ids,
                           output_mask=output_mask)
    nsp_logits = nsp_classifier(hidden_states=hidden_states)
    nsp_loss = nsp_loss_fn(logits=nsp_logits, labels=nsp_labels)

    loss = bert_loss(loss_1=mlm_loss, loss_2=nsp_loss)
    return loss, [mlm_loss, nsp_loss], steps_per_epoch


if not args.preprocessed_data:
    train_loss, _, steps_per_epoch = create_pipeline(
                                        data_file=data_desc.train_file,
                                        preprocessed_data=False,
                                        max_seq_length=args.max_seq_length,
                                        mask_probability=args.mask_probability,
                                        short_seq_prob=args.short_seq_prob,
                                        batch_size=args.batch_size,
                                        batches_per_step=args.batches_per_step)
else:
    max_pred_len = args.max_predictions_per_seq
    train_loss, _, steps_per_epoch = create_pipeline(
                                      data_file=args.data_dir,
                                      preprocessed_data=True,
                                      max_predictions_per_seq=max_pred_len,
                                      training=True,
                                      batch_size=args.batch_size,
                                      batches_per_step=args.batches_per_step)


# callback which prints training loss and perplexity once in a while
train_callback = nemo.core.SimpleLossLoggerCallback(
    tensors=[train_loss],
    print_func=lambda x: nf.logger.info("Loss: {:.3f}".format(x[0].item())),
    get_tb_values=lambda x: [["loss", x[0]]],
    tb_writer=nf.tb_writer)

ckpt_callback = nemo.core.CheckpointCallback(folder=nf.checkpoint_dir,
                                             epoch_freq=args.save_epoch_freq,
                                             load_from_folder=args.load_dir,
                                             step_freq=args.save_step_freq)

# define learning rate decay policy
if args.lr_policy is not None:
    if args.total_iterations_per_gpu < 0:
        lr_policy_fn = get_lr_policy(
                            args.lr_policy,
                            total_steps=args.num_epochs * steps_per_epoch,
                            warmup_ratio=args.lr_warmup_proportion)
    else:
        lr_policy_fn = get_lr_policy(args.lr_policy,
                                     total_steps=args.total_iterations_per_gpu,
                                     warmup_ratio=args.lr_warmup_proportion)
else:
    lr_policy_fn = None

config_path = f'{nf.checkpoint_dir}/bert-config.json'
if not os.path.exists(config_path):
    bert_model.config.to_json_file(config_path)

# define and launch training algorithm (optimizer)
nf.train(tensors_to_optimize=[train_loss],
         lr_policy=lr_policy_fn,
         callbacks=[train_callback, ckpt_callback],
         optimizer=args.optimizer,
         batches_per_step=args.batches_per_step,
         gradient_predivide=args.gradient_predivide,
         optimization_params={"batch_size": args.batch_size,
                              "num_epochs": args.num_epochs,
                              "lr": args.lr,
                              "betas": (args.beta1, args.beta2),
                              "weight_decay": args.weight_decay})
