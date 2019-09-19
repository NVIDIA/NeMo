import argparse
import math
import os

import numpy as np
from pytorch_transformers import BertTokenizer

import nemo
from nemo.utils.lr_policies import get_lr_policy

import nemo_nlp
from nemo_nlp.callbacks.joint_intent_slot import \
    eval_iter_callback, eval_epochs_done_callback
from nemo_nlp.text_data_utils import JointIntentSlotDataDesc


# Parsing arguments
parser = argparse.ArgumentParser(
    description='Joint intent slot filling system with pretrained BERT')
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--max_seq_length", default=50, type=int)
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--num_epochs", default=8, type=int)
parser.add_argument("--num_train_samples", default=-1, type=int)
parser.add_argument("--num_eval_samples", default=100, type=int)
parser.add_argument("--lr_warmup_proportion", default=0.1, type=float)
parser.add_argument("--lr", default=5e-5, type=float)
parser.add_argument("--lr_policy", default="WarmupAnnealing", type=str)
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument("--fc_dropout", default=0.1, type=float)
parser.add_argument("--pretrained_bert_model",
                    default="bert-base-uncased",
                    type=str)
parser.add_argument("--data_dir", default='data/nlu/snips', type=str)
parser.add_argument("--dataset_name", default='snips-all', type=str)
parser.add_argument("--work_dir", default='outputs', type=str)
parser.add_argument("--save_epoch_freq", default=1, type=int)
parser.add_argument("--save_step_freq", default=-1, type=int)
parser.add_argument("--optimizer_kind", default="adam", type=str)
parser.add_argument("--amp_opt_level", default="O0",
                    type=str, choices=["O0", "O1", "O2"])
parser.add_argument("--do_lower_case", action='store_false')
parser.add_argument("--shuffle_data", action='store_false')
parser.add_argument("--intent_loss_weight", default=0.6, type=float)

args = parser.parse_args()

if not os.path.exists(args.data_dir):
    raise ValueError(f'Data not found at {args.data_dir}')

work_dir = f'{args.work_dir}/{args.dataset_name.upper()}'
nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch,
                                   local_rank=args.local_rank,
                                   optimization_level=args.amp_opt_level,
                                   log_dir=work_dir,
                                   create_tb_writer=True,
                                   files_to_copy=[__file__],
                                   add_time_to_log_dir=True)

""" Load the pretrained BERT parameters
See the list of pretrained models, call:
nemo_nlp.huggingface.BERT.list_pretrained_models()
"""
pretrained_bert_model = nemo_nlp.huggingface.BERT(
    pretrained_model_name=args.pretrained_bert_model, factory=nf)
hidden_size = pretrained_bert_model.local_parameters["hidden_size"]
tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_model)

data_desc = JointIntentSlotDataDesc(
    args.dataset_name, args.data_dir, args.do_lower_case)


def get_dataset(data_desc, mode, num_samples):
    nf.logger.info(f"Loading {mode} data...")
    data_file = getattr(data_desc, mode + '_file')
    slot_file = getattr(data_desc, mode + '_slot_file')
    shuffle = args.shuffle_data if mode == 'train' else False
    return nemo_nlp.BertJointIntentSlotDataset(
        input_file=data_file,
        slot_file=slot_file,
        pad_label=data_desc.pad_label,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        num_samples=num_samples,
        shuffle=shuffle)


train_dataset = get_dataset(data_desc, 'train', args.num_train_samples)
eval_dataset = get_dataset(data_desc, 'eval', args.num_eval_samples)

# Create sentence classification loss on top
classifier = nemo_nlp.JointIntentSlotClassifier(
    hidden_size=hidden_size,
    num_intents=data_desc.num_intents,
    num_slots=data_desc.num_slots,
    dropout=args.fc_dropout)

loss_fn = nemo_nlp.JointIntentSlotLoss(num_slots=data_desc.num_slots)


def create_pipeline(dataset,
                    batch_size=32,
                    num_gpus=1,
                    local_rank=0,
                    mode='train'):
    data_layer = nemo_nlp.BertJointIntentSlotDataLayer(dataset,
                                                       batch_size=batch_size,
                                                       num_workers=0,
                                                       local_rank=local_rank)

    ids, type_ids, input_mask, slot_mask, intents, slots = data_layer()
    data_size = len(data_layer)

    if data_size < batch_size:
        nf.logger.warning("Batch_size is larger than the dataset size")
        nf.logger.warning("Reducing batch_size to dataset size")
        batch_size = data_size

    steps_per_epoch = math.ceil(data_size / (batch_size * num_gpus))
    nf.logger.info(f"Steps_per_epoch = {steps_per_epoch}")

    hidden_states = pretrained_bert_model(input_ids=ids,
                                          token_type_ids=type_ids,
                                          attention_mask=input_mask)

    intent_logits, slot_logits = classifier(hidden_states=hidden_states)

    loss = loss_fn(intent_logits=intent_logits,
                   slot_logits=slot_logits,
                   input_mask=input_mask,
                   intents=intents,
                   slots=slots)

    if mode == 'train':
        tensors_to_evaluate = [loss, intent_logits, slot_logits]
    else:
        tensors_to_evaluate = [intent_logits, slot_logits, intents, slots]

    return tensors_to_evaluate, loss, steps_per_epoch, data_layer


train_tensors, train_loss, steps_per_epoch, _ =\
    create_pipeline(train_dataset,
                    batch_size=args.batch_size,
                    num_gpus=args.num_gpus,
                    local_rank=args.local_rank,
                    mode='train')
eval_tensors, _,  _, data_layer = create_pipeline(eval_dataset,
                                                  batch_size=args.batch_size,
                                                  num_gpus=args.num_gpus,
                                                  local_rank=args.local_rank,
                                                  mode='eval')

# Create callbacks for train and eval modes
callback_train = nemo.core.SimpleLossLoggerCallback(
    tensors=train_tensors,
    print_func=lambda x: str(np.round(x[0].item(), 3)),
    tb_writer=nf.tb_writer,
    get_tb_values=lambda x: [["loss", x[0]]],
    step_freq=steps_per_epoch)

callback_eval = nemo.core.EvaluatorCallback(
    eval_tensors=eval_tensors,
    user_iter_callback=lambda x, y: eval_iter_callback(
        x, y, data_layer),
    user_epochs_done_callback=lambda x: eval_epochs_done_callback(
        x, f'{nf.work_dir}/graphs'),
    tb_writer=nf.tb_writer,
    eval_step=steps_per_epoch)

# Create callback to save checkpoints
ckpt_callback = nemo.core.CheckpointCallback(
    folder=nf.checkpoint_dir,
    epoch_freq=args.save_epoch_freq,
    step_freq=args.save_step_freq)

lr_policy_fn = get_lr_policy(args.lr_policy,
                             total_steps=args.num_epochs * steps_per_epoch,
                             warmup_ratio=args.lr_warmup_proportion)

nf.train(tensors_to_optimize=[train_loss],
         callbacks=[callback_train, callback_eval, ckpt_callback],
         lr_policy=lr_policy_fn,
         optimizer=args.optimizer_kind,
         optimization_params={"num_epochs": args.num_epochs,
                              "lr": args.lr,
                              "weight_decay": args.weight_decay})
