import argparse
import os

import nemo
from nemo_nlp.callbacks.token_classification import \
  eval_iter_callback, eval_epochs_done_callback
from nemo_nlp import NemoBertTokenizer

from nemo.utils.lr_policies import SquareAnnealing, CosineAnnealing, \
    WarmupAnnealing

import numpy as np

from tensorboardX import SummaryWriter

# Parsing arguments
default_data_dir = "./tatoeba_data/"
parser = argparse.ArgumentParser(description='SQUAD_with_pretrained_BERT')
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--num_epochs", default=2, type=int)
parser.add_argument("--lr_warmup_proportion", default=0.1, type=float)
parser.add_argument("--lr", default=3.e-5, type=float)
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument("--optimizer_kind", default="adam", type=str)
parser.add_argument("--mixed_precision", action='store_true')
parser.add_argument("--lr_policy", default="lr_warmup", type=str)
parser.add_argument("--pretrained_bert_model", default="bert-base-uncased",
                    type=str)
parser.add_argument("--data_dir", default=default_data_dir,
                    type=str)
args = parser.parse_args()

data_file = os.path.join(args.data_dir, "train_sentences.csv")
if not os.path.isfile(data_file):
  print('Training Data Not Found. Downloading to {0}'.format(default_data_dir))
  os.system("../../tests/data/get_tatoeba_eng_sentences.sh")
  data_dir = default_data_dir
else:
  data_dir = args.data_dir

batch_size = args.batch_size
num_gpus = args.num_gpus
lr = args.lr
lr_policy = args.lr_policy
lr_warmup_proportion = args.lr_warmup_proportion
weight_decay = args.weight_decay
num_epochs = args.num_epochs
optimizer_kind = args.optimizer_kind
local_rank = args.local_rank
mixed_precision = args.mixed_precision
pretrained_bert_model = args.pretrained_bert_model

tensorboard_filename = f"sentence_segmentation_bert_lrpolicy{lr_policy}_"\
  f"lr{lr}_opt{optimizer_kind}_lrwarm{lr_warmup_proportion}" \
  f"_bs{batch_size}_wd{weight_decay}_e{num_epochs}_ngpu{num_gpus}"

tb_writer = SummaryWriter(tensorboard_filename)

if local_rank is not None:
  device = nemo.core.DeviceType.AllGpu
else:
  device = nemo.core.DeviceType.GPU

optimization_level = nemo.core.Optimization.nothing
if mixed_precision is True:
  optimization_level = nemo.core.Optimization.mxprO1

# instantiate Neural Factory with supported backend
neural_factory = nemo.core.NeuralModuleFactory(
  backend=nemo.core.Backend.PyTorch,
  local_rank=local_rank,
  optimization_level=optimization_level,
  placement=device)

# Load the pretrained BERT parameters
# pretrained_model can be one of:
# bert-base-uncased, bert-large-uncased, bert-base-cased,
# bert-large-cased, bert-base-multilingual-uncased,
# bert-base-multilingual-cased, bert-base-chinese.
pretrained_model_name = pretrained_bert_model

pretrained_bert_model = neural_factory.get_module(
  name="BERT",
  params={"pretrained_model_name": pretrained_model_name,
          "local_rank": local_rank},
  collection="nemo_nlp",
  pretrained=True)

# Create Q/A loss on top
d_model = pretrained_bert_model.local_parameters["d_model"]
classification_dropout = \
  pretrained_bert_model.local_parameters["fully_connected_dropout"]

token_classification_loss = neural_factory.get_module(
  name="TokenClassificationLoss",
  params={"d_model": d_model,
          "num_labels": 2,
          "dropout": classification_dropout},
  collection="nemo_nlp"
)

# Data layer with Squad training data
path_to_vocab_file = \
  pretrained_bert_model.local_parameters["path_to_vocab_file"]
vocab_positional_embedding_size_map = \
  pretrained_bert_model.local_parameters["vocab_positional_embedding_size_map"]

tokenizer = NemoBertTokenizer(
  vocab_file=path_to_vocab_file,
  do_lower_case=True,
  max_len=vocab_positional_embedding_size_map
)

# Training pipeline
print("Loading training data...")
train_data_layer = neural_factory.get_module(
  name="BertTokenClassificationDataLayer",
  params={
    "tokenizer": tokenizer,
    "path_to_data": data_dir + "/train_sentences.csv",
    "max_seq_length": 512,
    "is_training": True,
    "batch_size": batch_size,
    "shuffle": False,
    "num_workers": 0,
    "local_rank": local_rank
  },
  collection="nemo_nlp"
)

input_ids, input_type_ids, input_mask, labels, seq_ids = train_data_layer()

train_data_size = len(train_data_layer)
steps_per_epoch = int(train_data_size / (batch_size*num_gpus))

print("steps_per_epoch = ", steps_per_epoch)

hidden_states = pretrained_bert_model(
  input_ids=input_ids, input_type_ids=input_type_ids, input_mask=input_mask)

train_loss, train_logits = \
  token_classification_loss(hidden_states=hidden_states,
                            labels=labels,
                            input_mask=input_mask)

# Evaluation pipeline
print("Loading eval data...")
eval_data_layer = neural_factory.get_module(
  name="BertTokenClassificationDataLayer",
  params={
    "tokenizer": tokenizer,
    "path_to_data": data_dir + "/eval_sentences.csv",
    "max_seq_length": 512,
    "is_training": True,
    "batch_size": batch_size,
    "shuffle": False,
    "num_workers": 0,
    "local_rank": local_rank
  },
  collection="nemo_nlp"
)

input_ids, input_type_ids, eval_input_mask, eval_labels, eval_seq_ids = \
  eval_data_layer()

hidden_states = pretrained_bert_model(
  input_ids=input_ids, input_type_ids=input_type_ids,
  input_mask=eval_input_mask)

eval_loss, eval_logits = \
  token_classification_loss(hidden_states=hidden_states,
                            labels=eval_labels,
                            input_mask=eval_input_mask)


def get_loss(loss):
  str_ = str(np.round(loss, 3))
  return str_


# Create trainer and execute training action
callback_train = nemo.core.SimpleLossLoggerCallback(
  tensor_list2string=lambda x: get_loss(x[0].item()),
  tensorboard_writer=tb_writer,
  step_frequency=100)

# Instantiate an optimizer to perform `train` action
optimizer = neural_factory.get_trainer(
  params={
    "optimizer_kind": optimizer_kind,
    "optimization_params": {"num_epochs":  num_epochs,
                            "lr": lr,
                            "weight_decay": weight_decay},
  }
)

callback_eval = nemo.core.EvaluatorCallback(
  eval_tensors=[eval_logits, eval_seq_ids],
  user_iter_callback=lambda x, y: eval_iter_callback(x, y, eval_data_layer),
  user_epochs_done_callback=eval_epochs_done_callback,
  tensorboard_writer=tb_writer,
  eval_step=steps_per_epoch)


if lr_policy == "lr_warmup":
    lr_policy_func = WarmupAnnealing(num_epochs * steps_per_epoch,
                                     warmup_ratio=lr_warmup_proportion)
elif lr_policy == "lr_poly":
    lr_policy_func = SquareAnnealing(num_epochs * steps_per_epoch)
elif lr_policy == "lr_cosine":
    lr_policy_func = CosineAnnealing(num_epochs * steps_per_epoch)
else:
  raise ValueError("Invalid lr_policy, must be lr_warmup or lr_poly")

optimizer.train(
  tensors_to_optimize=[train_loss],
  callbacks=[callback_train, callback_eval],
  lr_policy=lr_policy_func
)
