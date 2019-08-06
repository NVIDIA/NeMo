import os
import nemo
from nemo.utils.lr_policies import SquareAnnealing, CosineAnnealing, \
    WarmupAnnealing
import numpy as np
import argparse
from nemo_nlp import NemoBertTokenizer
from nemo_nlp.callbacks import eval_iter_callback, \
  eval_epochs_done_callback
from tensorboardX import SummaryWriter

# Parsing arguments
default_squad_dir = "./squad_data/"
parser = argparse.ArgumentParser(description='SQUAD_with_pretrained_BERT')
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--num_epochs", default=2, type=int)
parser.add_argument("--lr_warmup_proportion", default=0.1, type=float)
parser.add_argument("--lr", default=3.e-5, type=float)
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument(
  "--pretrained_bert_model", default="bert-base-uncased", type=str
)
parser.add_argument("--squad_data_dir", default=default_squad_dir,type=str)
parser.add_argument("--optimizer_kind", default="adam", type=str)
parser.add_argument("--mixed_precision", action='store_true')
parser.add_argument("--lr_policy", default="lr_warmup", type=str)
parser.add_argument("--max_sequence_length", default=384, type=int)
args = parser.parse_args()

data_file = os.path.join(args.squad_data_dir, "train-v1.1.json")
if not os.path.isfile(data_file):
  print(f'Training Data Not Found. Downloading to {default_squad_dir}')
  os.system("../../tests/data/get_squad.sh")
  squad_data_dir = default_squad_dir
else:
  squad_data_dir = args.squad_data_dir


batch_size = args.batch_size
num_gpus = args.num_gpus
lr = args.lr
lr_warmup_proportion = args.lr_warmup_proportion
weight_decay = args.weight_decay
num_epochs = args.num_epochs
lr_policy = args.lr_policy
optimizer_kind = args.optimizer_kind
local_rank = args.local_rank
mixed_precision = args.mixed_precision
pretrained_bert_model = args.pretrained_bert_model

print("local_rank = ", local_rank)

tensorboard_filename = f"squad_pretrained_bert_lr{lr}_" \
  f"opt{optimizer_kind}_lrpol{lr_policy}_lrwarm{lr_warmup_proportion}" \
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
  placement=device
)

# Load the pretrained BERT parameters
# pretrained_model can be one of:
# bert-base-uncased, bert-large-uncased, bert-base-cased,
# bert-large-cased, bert-base-multilingual-uncased,
# bert-base-multilingual-cased, bert-base-chinese.
pretrained_model_name = pretrained_bert_model
do_lower_case = True

pretrained_bert_model = neural_factory.get_module(
  name="BERT",
  params={"pretrained_model_name": pretrained_model_name,
          "local_rank": local_rank},
  collection="nemo_nlp",
  pretrained=True)

# Create Q/A loss on top
d_model = pretrained_bert_model.local_parameters["d_model"]
question_answering_loss = neural_factory.get_module(
  name="QuestionAnsweringPredictionLoss",
  params={"d_model": d_model},
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
  name="BertQuestionAnsweringDataLayer",
  params={
    "tokenizer": tokenizer,
    "path_to_data": squad_data_dir + "/train-v1.1.json",
    "data_format": "squad_json",
    "features_file_prefix": pretrained_model_name,
    "do_lower_case": do_lower_case,
    "max_seq_length": args.max_sequence_length,
    "max_query_length": 64,
    "is_training": True,
    "batch_size": batch_size,
    "shuffle": False,
    "num_workers": 0,
    "local_rank": local_rank
  },
  collection="nemo_nlp"
)

input_ids, input_type_ids, input_mask, \
  start_positions, end_positions, train_unique_ids = train_data_layer()

train_data_size = len(train_data_layer)
steps_per_epoch = int(train_data_size / (batch_size*num_gpus))

print("steps_per_epoch = ", steps_per_epoch)

hidden_states = pretrained_bert_model(
  input_ids=input_ids, input_type_ids=input_type_ids, input_mask=input_mask)

train_loss, train_start_logits, train_end_logits = \
  question_answering_loss(hidden_states=hidden_states,
                          start_positions=start_positions,
                          end_positions=end_positions)

# Evaluation pipeline

print("Loading eval data...")
# Data layer with Squad eval data
eval_data_layer = neural_factory.get_module(
  name="BertQuestionAnsweringDataLayer",
  params={
    "tokenizer": tokenizer,
    "path_to_data": squad_data_dir + "/dev-v1.1.json",
    "data_format": "squad_json",
    "features_file_prefix": pretrained_model_name,
    "max_seq_length": args.max_sequence_length,
    "max_query_length": 64,
    "path_to_vocab_file": path_to_vocab_file,
    "is_training": False,
    "batch_size": batch_size,
    "shuffle": False,
    "num_workers": 0,
    "local_rank": local_rank
  },
  collection="nemo_nlp"
)

input_ids, input_type_ids, input_mask, \
  start_positions, end_positions, eval_unique_ids = eval_data_layer()

hidden_states = pretrained_bert_model(
  input_ids=input_ids, input_type_ids=input_type_ids, input_mask=input_mask
)

eval_loss, eval_start_logits, eval_end_logits = \
  question_answering_loss(hidden_states=hidden_states,
                          start_positions=start_positions,
                          end_positions=end_positions)

###############################################################################


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
  eval_tensors=[eval_loss, eval_start_logits,
                eval_end_logits, eval_unique_ids],
  user_iter_callback=eval_iter_callback,
  user_epochs_done_callback=lambda x: eval_epochs_done_callback(
    x, eval_data_layer=eval_data_layer, do_lower_case=do_lower_case
  ),
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
  lr_policy=lr_policy_func)
