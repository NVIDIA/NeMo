import argparse
import math
import os

import numpy as np
from pytorch_transformers import BertTokenizer

import nemo
import nemo_nlp
from nemo.utils.lr_policies import get_lr_policy
from nemo_nlp.callbacks.joint_intent_slot import \
    eval_iter_callback, eval_epochs_done_callback
from nemo_nlp.text_data_utils import \
    process_atis, process_snips, merge


# Parsing arguments
parser = argparse.ArgumentParser(
    description='Joint intent slot filling system with pretrained BERT')
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--max_seq_length", default=50, type=int)
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--num_epochs", default=8, type=int)
parser.add_argument("--num_train_samples", default=-1, type=int)
parser.add_argument("--num_dev_samples", default=-1, type=int)
parser.add_argument("--lr_warmup_proportion", default=0.1, type=float)
parser.add_argument("--lr", default=5e-5, type=float)
parser.add_argument("--lr_policy", default="WarmupAnnealing", type=str)
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument("--fc_dropout", default=0.1, type=float)
parser.add_argument("--pretrained_bert_model",
                    default="bert-base-uncased",
                    type=str)
parser.add_argument("--data_dir", default='data/nlu', type=str)
# parser.add_argument("--dataset_name", default='snips-all', type=str)
parser.add_argument("--dataset_name", default='snips-atis', type=str)
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
                                   files_to_copy=[__file__])

# Load the pretrained BERT parameters
# pretrained_model can be one of:
# bert-base-uncased, bert-large-uncased, bert-base-cased,
# bert-large-cased, bert-base-multilingual-uncased,
# bert-base-multilingual-cased, bert-base-chinese.

pretrained_bert_model = nf.get_module(
    name="huggingface.BERT",
    params={"pretrained_model_name": args.pretrained_bert_model,
            "local_rank": args.local_rank},
    collection="nemo_nlp",
    pretrained=True)

if args.dataset_name == 'atis':
    num_intents = 26
    num_slots = 129
    data_dir = process_atis(args.data_dir, args.do_lower_case)
    pad_label = num_slots - 1
elif args.dataset_name == 'snips-atis':
    data_dir, pad_label = merge(args.data_dir,
                                ['ATIS/nemo-processed-uncased',
                                 'snips/nemo-processed-uncased/all'],
                                args.dataset_name)
    num_intents = 41
    num_slots = 140
elif args.dataset_name.startswith('snips'):
    data_dir = process_snips(args.data_dir, args.do_lower_case)
    if args.dataset_name.endswith('light'):
        data_dir = f'{data_dir}/light'
        num_intents = 6
        num_slots = 4
    elif args.dataset_name.endswith('speak'):
        data_dir = f'{data_dir}/speak'
        num_intents = 9
        num_slots = 9
    elif args.dataset_name.endswith('all'):
        data_dir = f'{data_dir}/all'
        num_intents = 15
        num_slots = 12
    pad_label = num_slots - 1
else:
    nf.logger.info("Looks like you pass in the name of dataset that isn't "
                   "already supported by NeMo. Please make sure that you "
                   "build the preprocessing method for it.")

# Create sentence classification loss on top
hidden_size = pretrained_bert_model.local_parameters["hidden_size"]

classifier = nemo_nlp.JointIntentSlotClassifier(hidden_size=hidden_size,
                                                num_intents=num_intents,
                                                num_slots=num_slots,
                                                dropout=args.fc_dropout)

loss_fn = nemo_nlp.JointIntentSlotLoss(num_slots=num_slots)

tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_model)


def create_pipeline(data_file,
                    slot_file,
                    tokenizer,
                    classifier,
                    loss_fn,
                    pad_label,
                    max_seq_length,
                    batch_size=32,
                    num_samples=-1,
                    shuffle=True,
                    num_gpus=1,
                    local_rank=0,
                    mode='train'):
    nf.logger.info(f"Loading {mode} data...")
    data_layer = nemo_nlp.BertJointIntentSlotDataLayer(
        path_to_data=data_file,
        path_to_slot=slot_file,
        pad_label=pad_label,
        tokenizer=tokenizer,
        mode=mode,
        max_seq_length=max_seq_length,
        num_samples=num_samples,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        local_rank=local_rank
    )

    ids, type_ids, input_mask, slot_mask, intents, slots = data_layer()
    data_size = len(data_layer)

    if data_size < batch_size:
        nf.logger.warning("Batch_size is larger than the dataset size")
        nf.logger.warning("Reducing batch_size to dataset size")
        batch_size = data_size

    steps_per_epoch = int(data_size / (batch_size * num_gpus))

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

    # Create trainer and execute training action
    if mode == 'train':
        callback_fn = nemo.core.SimpleLossLoggerCallback(
            tensors=[loss, intent_logits, slot_logits],
            print_func=lambda x: str(np.round(x[0].item(), 3)),
            tb_writer=nf.tb_writer,
            get_tb_values=lambda x: [["loss", x[0]]],
            step_freq=100)
    elif mode == 'eval':
        callback_fn = nemo.core.EvaluatorCallback(
            eval_tensors=[intent_logits, slot_logits, intents, slots],
            user_iter_callback=lambda x, y: eval_iter_callback(
                x, y, data_layer),
            user_epochs_done_callback=lambda x: eval_epochs_done_callback(
                x, f'{nf.work_dir}/graphs'),
            tb_writer=nf.tb_writer,
            eval_step=steps_per_epoch)

    return loss, callback_fn, steps_per_epoch


train_loss, callback_train, steps_per_epoch =\
    create_pipeline(data_dir + '/train.tsv',
                    data_dir + '/train_slots.tsv',
                    tokenizer=tokenizer,
                    classifier=classifier,
                    loss_fn=loss_fn,
                    pad_label=pad_label,
                    max_seq_length=args.max_seq_length,
                    batch_size=args.batch_size,
                    num_samples=args.num_train_samples,
                    shuffle=args.shuffle_data,
                    num_gpus=args.num_gpus,
                    local_rank=args.local_rank,
                    mode='train')
eval_loss, callback_eval, _ =\
    create_pipeline(data_dir + '/test.tsv',
                    data_dir + '/test_slots.tsv',
                    tokenizer=tokenizer,
                    classifier=classifier,
                    loss_fn=loss_fn,
                    pad_label=pad_label,
                    max_seq_length=args.max_seq_length,
                    batch_size=args.batch_size,
                    num_samples=args.num_train_samples,
                    shuffle=False,
                    num_gpus=args.num_gpus,
                    local_rank=args.local_rank,
                    mode='eval')


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
