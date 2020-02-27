"""
This code were adapted from 
https://github.com/google-research/google-research/tree/master/schema_guided_dst
"""

import argparse
import math
import os

import numpy as np
import pickle

import nemo
import nemo.collections.nlp as nemo_nlp

import nemo.collections.nlp.data.datasets.sgd_dataset.data_utils as data_utils

from nemo.collections.nlp.callbacks.sgd_callback import \
    eval_iter_callback, eval_epochs_done_callback
from nemo.utils.lr_policies import get_lr_policy

from nemo.collections.nlp.nm.trainables import sgd_modules, sgd_model

# Parsing arguments
parser = argparse.ArgumentParser(description='Schema_guided_dst')

# BERT based utterance encoder related arguments
parser.add_argument("--bert_ckpt_dir", default=None, type=str,
                    required=True,
                    help="Directory containing pre-trained BERT checkpoint.")
# parser.add_argument("--do_lower_case", action="store_true",
#                     help="Whether to lower case the input text. Should be True for uncased "
#                     "models and False for cased models.")
# parser.add_argument("--preserve_unused_tokens", default=False, type=bool,
#                     help="If preserve_unused_tokens is True, Wordpiece tokenization will not "
#                     "be applied to words in the vocab.")
parser.add_argument("--max_seq_length", default=80, type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. "
                    "Sequences longer than this will be truncated, and sequences shorter "
                    "than this will be padded.")
parser.add_argument("--dropout", default=0.1, type=float,
                    help="Dropout rate for BERT representations.")
parser.add_argument("--pretrained_model_name", default="bert-base-cased", type=str,
                    help="Pretrained BERT model")

# Hyperparameters and optimization related flags.
parser.add_argument("--optimizer_kind", default="adam", type=str)
parser.add_argument("--train_batch_size", default=32, type=int,
                    help="Total batch size for training.")
parser.add_argument("--eval_batch_size", default=8, type=int,
                    help="Total batch size for eval.")
parser.add_argument("--predict_batch_size", default=8, type=int,
                    help="Total batch size for predict.")
parser.add_argument("--learning_rate", default=1e-4, type=float, 
                    help="The initial learning rate for Adam.")
parser.add_argument("--lr_policy", default="WarmupAnnealing", type=str)
parser.add_argument("--num_epochs", default=80.0, type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--lr_warmup_proportion", default=0.1, type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                    "E.g., 0.1 = 10% of training.")
parser.add_argument("--save_epoch_freq", default=1, type=int,
                    help="How often to save the model checkpoint.")
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--amp_opt_level", default="O0",
                    type=str, choices=["O0", "O1", "O2"])
parser.add_argument("--num_gpus", default=1, type=int)

# Input and output paths and other flags.
parser.add_argument("--task_name", default="dstc8_single_domain", type=str,
                    choices=data_utils.FILE_RANGES.keys(),
                    help="The name of the task to train.")
parser.add_argument("--data_dir", type=str, required=True,
                    help="Directory for the downloaded DSTC8 data, which contains the dialogue files"
                    " and schema files of all datasets (eg train, dev)")
# parser.add_argument("--run_mode", default="train", type=str,
#                     choices=["train", "predict"],
#                     help="The mode to run the script in.")
parser.add_argument("--work_dir", type=str, default="output/SGD",
                    help="The output directory where the model checkpoints will be written.")
parser.add_argument("--schema_embedding_dir", type=str, required=True,
                    help="Directory where .npy file for embedding of entities (slots, values,"
                    " intents) in the dataset_split's schema are stored.")
parser.add_argument("--overwrite_schema_emb_files", action="store_true",
                    help="Whether to generate a new file saving the dialogue examples.")
parser.add_argument("--dialogues_example_dir", type=str, required=True,
                    help="Directory where preprocessed DSTC8 dialogues are stored.")
parser.add_argument("--overwrite_dial_files", action="store_true",
                    help="Whether to generate a new file saving the dialogue examples.")
parser.add_argument("--no_shuffle", action="store_false",
                    help="Whether to shuffle training data")
parser.add_argument("--eval_dataset", type=str, default="dev",
                    choices=["dev", "test"],
                    help="Dataset split for evaluation.")
parser.add_argument("--ckpt_save_freq", type=int, default=1000,
                    help="How often to save checkpoints")

args = parser.parse_args()

if not os.path.exists(args.data_dir):
    raise ValueError('Data not found at {args.data_dir}')

args.work_dir = f'{args.work_dir}/{args.task_name.upper()}'
nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch,
                                   local_rank=args.local_rank,
                                   optimization_level=args.amp_opt_level,
                                   log_dir=args.work_dir,
                                   create_tb_writer=True,
                                   files_to_copy=[__file__],
                                   add_time_to_log_dir=True)


bert_init_ckpt = os.path.join(args.bert_ckpt_dir, "BERT-base-cased.pt")

pretrained_bert_model = nemo_nlp.nm.trainables.huggingface.BERT(
    pretrained_model_name=args.pretrained_model_name)

pretrained_bert_model.restore_from(bert_init_ckpt)

# BERT tokenizer
tokenizer = nemo_nlp.data.NemoBertTokenizer(pretrained_model=args.pretrained_model_name)

# Run SGD preprocessor to generate and store schema embeddings
schema_preprocessor = utils.SchemaPreprocessor(
    data_dir=args.data_dir,
    schema_embedding_dir=args.schema_embedding_dir,
    max_seq_length=args.max_seq_length,
    tokenizer=tokenizer,
    bert_model=pretrained_bert_model,
    datasets=['train', args.eval_dataset],
    overwrite_schema_emb_files=args.overwrite_schema_emb_files,
    bert_ckpt_dir=args.bert_ckpt_dir,
    nf=nf)

# Dstc8Data
dialogues_processor = data_utils.Dstc8DataProcessor(
                args.task_name,
                args.data_dir,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                log_data_warnings=False)

train_datalayer = nemo_nlp.SGDDataLayer(
    task_name=args.task_name,
    dialogues_example_dir=args.dialogues_example_dir,
    overwrite_dial_file=args.overwrite_dial_files,
    dataset_split='train',
    schema_emb_processor=schema_preprocessor,
    dialogues_processor=dialogues_processor,
    batch_size=args.train_batch_size,
    shuffle=not args.no_shuffle)

# fix
bert_config = os.path.join(args.bert_ckpt_dir, 'bert_config.json')
if not os.path.exists(bert_config):
    raise ValueError(f'bert_config.json not found at {args.bert_ckpt_dir}')

input_data = train_datalayer()

hidden_size = pretrained_bert_model.local_parameters["hidden_size"]

# define model pipeline
encoder_extractor = sgd_modules.Encoder(hidden_size=hidden_size,
                                        dropout=args.dropout)
dst_loss = nemo_nlp.SGDDialogueStateLoss()


# Encode the utterances using BERT.
token_embeddings = pretrained_bert_model(input_ids=input_data.utterance_ids,
                                         attention_mask=input_data.utterance_mask,
                                         token_type_ids=input_data.utterance_segment)
encoded_utterance = encoder_extractor(hidden_states=token_embeddings)
model = sgd_model.SGDModel(embedding_dim=hidden_size)

logit_intent_status,\
logit_req_slot_status,\
req_slot_mask,\
logit_cat_slot_status,\
logit_cat_slot_value,\
cat_slot_values_mask,\
logit_noncat_slot_status,\
logit_noncat_slot_start,\
logit_noncat_slot_end = model(encoded_utterance=encoded_utterance,
                        token_embeddings=token_embeddings,
                        utterance_mask=input_data.utterance_mask,
                        num_categorical_slot_values=input_data.num_categorical_slot_values,
                        num_intents=input_data.num_intents,
                        cat_slot_emb=input_data.cat_slot_emb,
                        cat_slot_value_emb=input_data.cat_slot_value_emb,
                        noncat_slot_emb=input_data.noncat_slot_emb,
                        req_slot_emb=input_data.req_slot_emb,
                        req_num_slots=input_data.num_slots,
                        intent_embeddings=input_data.intent_emb)

loss = dst_loss(logit_intent_status=logit_intent_status,
                logit_req_slot_status=logit_req_slot_status,
                logit_cat_slot_status=logit_cat_slot_status,
                logit_cat_slot_value=logit_cat_slot_value,
                logit_noncat_slot_status=logit_noncat_slot_status,
                logit_noncat_slot_start=logit_noncat_slot_start,
                logit_noncat_slot_end=logit_noncat_slot_end,
                intent_status=input_data.intent_status,
                requested_slot_status=input_data.requested_slot_status,
                req_slot_mask=req_slot_mask,
                categorical_slot_status=input_data.categorical_slot_status,
                num_categorical_slots=input_data.num_categorical_slots,
                categorical_slot_values=input_data.categorical_slot_values,
                cat_slot_values_mask=cat_slot_values_mask,
                noncategorical_slot_status=input_data.noncategorical_slot_status,
                num_noncategorical_slots=input_data.num_noncategorical_slots,
                noncategorical_slot_value_start=input_data.noncategorical_slot_value_start,
                noncategorical_slot_value_end=input_data.noncategorical_slot_value_end)

eval_datalayer = nemo_nlp.SGDDataLayer(
    task_name=args.task_name,
    dialogues_example_dir=args.dialogues_example_dir,
    overwrite_dial_file=args.overwrite_dial_files,
    dataset_split=args.eval_dataset,
    schema_emb_processor=schema_preprocessor,
    dialogues_processor=dialogues_processor,
    batch_size=args.eval_batch_size,
    shuffle=False)

# Encode the utterances using BERT
eval_data = eval_datalayer()
# import pdb; pdb.set_trace()
# eval_datalayer.dataset[0]



print (len(eval_datalayer))
eval_token_embeddings = pretrained_bert_model(input_ids=eval_data.utterance_ids,
                                         attention_mask=eval_data.utterance_mask,
                                         token_type_ids=eval_data.utterance_segment)
eval_encoded_utterance = encoder_extractor(hidden_states=eval_token_embeddings)

eval_logit_intent_status,\
eval_logit_req_slot_status,\
eval_req_slot_mask,\
eval_logit_cat_slot_status,\
eval_logit_cat_slot_value, eval_cat_slot_values_mask,\
eval_logit_noncat_slot_status,\
eval_logit_noncat_slot_start,\
eval_logit_noncat_slot_end = model(encoded_utterance=eval_encoded_utterance,
                                    token_embeddings=eval_token_embeddings,
                                    utterance_mask=eval_data.utterance_mask,
                                    num_categorical_slot_values=eval_data.num_categorical_slot_values,
                                    num_intents=eval_data.num_intents,
                                    cat_slot_emb=eval_data.cat_slot_emb,
                                    cat_slot_value_emb=eval_data.cat_slot_value_emb,
                                    noncat_slot_emb=eval_data.noncat_slot_emb,
                                    req_slot_emb=eval_data.req_slot_emb,
                                    req_num_slots=eval_data.num_slots,
                                    intent_embeddings=eval_data.intent_emb)

train_tensors = [loss]

eval_tensors = [eval_data.example_id,
                eval_data.service_id, 
                eval_data.is_real_example,
                eval_data.user_utterance,
                eval_data.start_char_idx,
                eval_data.end_char_idx,
                eval_logit_intent_status,
                eval_logit_req_slot_status,
                eval_logit_cat_slot_status,
                eval_logit_cat_slot_value,
                eval_cat_slot_values_mask,
                eval_logit_noncat_slot_status,
                eval_logit_noncat_slot_start,
                eval_logit_noncat_slot_end,
                eval_data.intent_status,
                eval_data.requested_slot_status,
                eval_req_slot_mask,
                eval_data.categorical_slot_status,
                eval_data.num_categorical_slots,
                eval_data.categorical_slot_values,
                eval_data.noncategorical_slot_status,
                eval_data.num_noncategorical_slots,
                eval_data.noncategorical_slot_value_start,
                eval_data.noncategorical_slot_value_end]

steps_per_epoch = len(train_datalayer) // (args.train_batch_size * args.num_gpus)
nemo.logging.info(f'steps per epoch: {steps_per_epoch}')
# import pdb; pdb.set_trace()
# Create trainer and execute training action
train_callback = nemo.core.SimpleLossLoggerCallback(
    tensors=train_tensors,
    print_func=lambda x: nemo.logging.info("Loss: {:.3f}".format(x[0].item())),
    get_tb_values=lambda x: [["loss", x[0]]],
    tb_writer=nf.tb_writer,
    step_freq=min(steps_per_epoch, args.ckpt_save_freq))


# we'll write predictions to file in DSTC8 format during evaluation callback
input_json_files = [os.path.join(args.data_dir, args.eval_dataset, 'dialogues_{:03d}.json'.format(fid))
        for fid in data_utils.FILE_RANGES[args.task_name][args.eval_dataset]]
    
schema_json_file = os.path.join(args.data_dir, args.eval_dataset, 'schema.json')

    
# Write predictions to file in DSTC8 format.
prediction_dir = os.path.join(args.work_dir, 'predictions', 'pred_res_{}_{}'.format(args.eval_dataset, args.task_name))
output_metric_file = os.path.join(args.work_dir, 'metrics.txt')
if not os.path.exists(prediction_dir):
    os.makedirs(prediction_dir)

eval_callback = nemo.core.EvaluatorCallback(
    eval_tensors=eval_tensors,
    user_iter_callback=lambda x, y: eval_iter_callback(x, y),
    user_epochs_done_callback=lambda x: eval_epochs_done_callback(x,
                                                                  input_json_files,
                                                                  schema_json_file,
                                                                  prediction_dir,
                                                                  args.data_dir,
                                                                  args.eval_dataset,
                                                                  output_metric_file),
    tb_writer=nf.tb_writer,
    eval_step=steps_per_epoch)

ckpt_callback = nemo.core.CheckpointCallback(
    folder=nf.checkpoint_dir,
    step_freq=args.ckpt_save_freq)

lr_policy_fn = get_lr_policy(args.lr_policy,
                             total_steps=args.num_epochs * steps_per_epoch,
                             warmup_ratio=args.lr_warmup_proportion)

nf.train(tensors_to_optimize=[loss],
         callbacks=[train_callback, 
         eval_callback,
         ckpt_callback],
         lr_policy=lr_policy_fn,
         optimizer=args.optimizer_kind,
         optimization_params={"num_epochs": args.num_epochs,
                              "lr": args.learning_rate}
                              )


# encoded_utterance = bert_encoder.get_pooled_output()
# encoded_tokens = bert_encoder.get_sequence_output()

# Apply dropout in training mode.
# encoded_utterance = tf.layers.dropout(
#     encoded_utterance, rate=FLAGS.dropout_rate, training=is_training)
# encoded_tokens = tf.layers.dropout(
#     encoded_tokens, rate=FLAGS.dropout_rate, training=is_training)
# return encoded_utterance, encoded_tokens



# TODO: add max_seq_len checkp
"""
bert_config = modeling.BertConfig.from_json_file(
      os.path.join(FLAGS.bert_ckpt_dir, "bert_config.json"))
  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))
"""


    