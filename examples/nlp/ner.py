# pylint: disable=invalid-name

import argparse
import os
from pytorch_transformers import BertTokenizer

import nemo
from nemo.utils.lr_policies import SquareAnnealing, CosineAnnealing, \
    WarmupAnnealing

import nemo_nlp
from nemo_nlp.callbacks.ner import \
    eval_iter_callback, eval_epochs_done_callback
from nemo_nlp import NemoBertTokenizer

# Parsing arguments
parser = argparse.ArgumentParser(description="NER_with_pretrained_BERT")
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--num_epochs", default=10, type=int)
parser.add_argument("--lr_warmup_proportion", default=0.1, type=float)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument("--optimizer_kind", default="adam", type=str)
parser.add_argument("--mixed_precision", action="store_true")
parser.add_argument("--lr_policy", default="lr_warmup", type=str)
parser.add_argument("--pretrained_bert_model", default="bert-base-cased",
                    type=str)
parser.add_argument("--data_dir", default="./conll2003", type=str)
parser.add_argument("--classification_dropout", default=0.1, type=float)
parser.add_argument("--max_seq_length", default=128, type=int)
parser.add_argument("--output_filename", default="output.txt", type=str)
parser.add_argument("--tensorboard_filename", default="ner_tensorboard",
                    type=str)
args = parser.parse_args()

data_file = os.path.join(args.data_dir, "train.txt")
if not os.path.isfile(data_file):
    raise FileNotFoundError("CoNLL-2003 dataset not found. Dataset can be "
                            + "obtained at https://github.com/kyzhouhzau/BERT"
                            + "-NER/tree/master/data and should be put in a "
                            + "folder at the same level as ner.py.")

try:
    import tensorboardX
    tb_writer = tensorboardX.SummaryWriter(args.tensorboard_filename)
except ModuleNotFoundError:
    tb_writer = None
    print("Tensorboard is not available.")

if args.local_rank is not None:
    device = nemo.core.DeviceType.AllGpu
else:
    device = nemo.core.DeviceType.GPU

if args.mixed_precision is True:
    optimization_level = nemo.core.Optimization.mxprO1
else:
    optimization_level = nemo.core.Optimization.nothing

# Instantiate Neural Factory with supported backend
neural_factory = nemo.core.NeuralModuleFactory(
    backend=nemo.core.Backend.PyTorch,
    local_rank=args.local_rank,
    optimization_level=optimization_level,
    placement=device)

pretrained_bert_model = nemo_nlp.huggingface.BERT(
    pretrained_model_name=args.pretrained_bert_model,
    local_rank=args.local_rank,
    factory=neural_factory)

tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_model)

# Training pipeline
print("Loading training data...")
train_data_layer = nemo_nlp.BertNERDataLayer(
    tokenizer=tokenizer,
    path_to_data="{}/train.txt".format(args.data_dir),
    max_seq_length=args.max_seq_length,
    is_training=True,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
    local_rank=args.local_rank,
    factory=neural_factory)

# Create training loss
d_model = pretrained_bert_model.bert.config.hidden_size
tag_ids = train_data_layer.dataset.tag_ids

ner_loss = nemo_nlp.TokenClassificationLoss(
    d_model=pretrained_bert_model.bert.config.hidden_size,
    num_labels=len(train_data_layer.dataset.tag_ids),
    dropout=args.classification_dropout,
    factory=neural_factory)

input_ids, input_type_ids, input_mask, labels, _ = train_data_layer()

hidden_states = pretrained_bert_model(
    input_ids=input_ids,
    token_type_ids=input_type_ids,
    attention_mask=input_mask)

train_loss, train_logits = ner_loss(
    hidden_states=hidden_states,
    labels=labels,
    input_mask=input_mask)

# Evaluation pipeline
print("Loading eval data...")
eval_data_layer = nemo_nlp.BertNERDataLayer(
    tokenizer=tokenizer,
    path_to_data="{}/test.txt".format(args.data_dir),
    max_seq_length=args.max_seq_length,
    is_training=True,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
    local_rank=args.local_rank,
    factory=neural_factory)

input_ids, input_type_ids, eval_input_mask, eval_labels, eval_seq_ids = \
    eval_data_layer()

hidden_states = pretrained_bert_model(
    input_ids=input_ids,
    token_type_ids=input_type_ids,
    attention_mask=eval_input_mask)

eval_loss, eval_logits = ner_loss(
    hidden_states=hidden_states,
    labels=eval_labels,
    input_mask=eval_input_mask)

# Create trainer and execute training action
callback_train = nemo.core.SimpleLossLoggerCallback(
    tensor_list2string=lambda x: "{:.3f}".format(x[0].item()),
    tensorboard_writer=tb_writer,
    step_frequency=100)

# Instantiate an optimizer to perform `train` action
optimizer = neural_factory.get_trainer(
    params={
        "optimizer_kind": args.optimizer_kind,
        "optimization_params": {
            "num_epochs":  args.num_epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "amsgrad": True
        }})

train_data_size = len(train_data_layer)
steps_per_epoch = int(train_data_size / (args.batch_size * args.num_gpus))

print("steps_per_epoch =", steps_per_epoch)

callback_eval = nemo.core.EvaluatorCallback(
    eval_tensors=[eval_logits, eval_seq_ids],
    user_iter_callback=lambda x, y: eval_iter_callback(
        x, y, eval_data_layer, tag_ids),
    user_epochs_done_callback=lambda x: eval_epochs_done_callback(
        x, tag_ids, args.output_filename),
    tensorboard_writer=tb_writer,
    eval_step=steps_per_epoch)

if args.lr_policy == "lr_warmup":
    lr_policy_func = WarmupAnnealing(args.num_epochs * steps_per_epoch,
                                     warmup_ratio=args.lr_warmup_proportion)
elif args.lr_policy == "lr_poly":
    lr_policy_func = SquareAnnealing(args.num_epochs * steps_per_epoch)
elif args.lr_policy == "lr_cosine":
    lr_policy_func = CosineAnnealing(args.num_epochs * steps_per_epoch)
else:
    raise ValueError("Invalid lr_policy, must be lr_warmup or lr_poly")

optimizer.train(
    tensors_to_optimize=[train_loss],
    callbacks=[callback_train, callback_eval],
    lr_policy=lr_policy_func)
