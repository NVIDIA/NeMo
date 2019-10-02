# pylint: disable=invalid-name

import argparse
import os

from nemo_nlp import NemoBertTokenizer, SentencePieceTokenizer

import nemo
from nemo.utils.lr_policies import get_lr_policy
import nemo_nlp
from nemo_nlp.utils.callbacks.ner import \
    eval_iter_callback, eval_epochs_done_callback


# Parsing arguments
parser = argparse.ArgumentParser(description="NER_with_pretrained_BERT")
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--max_seq_length", default=128, type=int)
parser.add_argument("--num_gpus", default=1, type=int)
parser.add_argument("--num_epochs", default=1, type=int)
parser.add_argument("--lr_warmup_proportion", default=0.1, type=float)
parser.add_argument("--lr", default=5e-5, type=float)
parser.add_argument("--lr_policy", default="WarmupAnnealing", type=str)
parser.add_argument("--weight_decay", default=0, type=float)
parser.add_argument("--optimizer_kind", default="adam", type=str)
parser.add_argument("--amp_opt_level", default="O0",
                    type=str, choices=["O0", "O1", "O2"])
parser.add_argument("--data_dir", default="data/ner/conll2003", type=str)
parser.add_argument("--dataset_name", default="conll2003", type=str)
parser.add_argument("--work_dir", default='outputs', type=str)
parser.add_argument("--fc_dropout", default=0.1, type=float)
parser.add_argument("--save_epoch_freq", default=1, type=int)
parser.add_argument("--save_step_freq", default=-1, type=int)
parser.add_argument("--pretrained_bert_model",
                    default="bert-base-cased", type=str)
parser.add_argument("--bert_checkpoint", default=None, type=str)
parser.add_argument("--bert_config", default=None, type=str)
parser.add_argument("--tokenizer_model", default="tokenizer.model", type=str)

args = parser.parse_args()

if not os.path.exists(args.data_dir):
    raise FileNotFoundError("CoNLL-2003 dataset not found. Dataset can be "
                            "obtained at https://github.com/kyzhouhzau/BERT"
                            "-NER/tree/master/data.")

work_dir = f'{args.work_dir}/{args.dataset_name.upper()}'
nf = nemo.core.NeuralModuleFactory(backend=nemo.core.Backend.PyTorch,
                                   local_rank=args.local_rank,
                                   optimization_level=args.amp_opt_level,
                                   log_dir=work_dir,
                                   create_tb_writer=True,
                                   files_to_copy=[__file__],
                                   add_time_to_log_dir=True)
output_file = f'{nf.work_dir}/output.txt'

if args.bert_checkpoint is None:
    """ Use this if you're using a standard BERT model.
    To see the list of pretrained models, call:
    nemo_nlp.huggingface.BERT.list_pretrained_models()
    """
    tokenizer = NemoBertTokenizer(args.pretrained_bert_model)
    pretrained_bert_model = nemo_nlp.huggingface.BERT(
        pretrained_model_name=args.pretrained_bert_model, factory=nf)
else:
    """ Use this if you're using a BERT model that you pre-trained yourself.
    Replace BERT-STEP-150000.pt with the path to your checkpoint.
    """
    tokenizer = SentencePieceTokenizer(model_path=tokenizer_model)
    tokenizer.add_special_tokens(["[MASK]", "[CLS]", "[SEP]"])

    bert_model = nemo_nlp.huggingface.BERT(
        config_filename=args.bert_config, factory=nf)
    pretrained_bert_model.restore_from(args.bert_checkpoint)

nf.logger.info("Loading training data...")
train_dataset = nemo_nlp.BertNERDataset(
    tokenizer=tokenizer,
    input_file=f'{args.data_dir}/train.txt',
    max_seq_length=args.max_seq_length)

tag_ids = train_dataset.tag_ids

nf.logger.info("Loading eval data...")
eval_dataset = nemo_nlp.BertNERDataset(
    tokenizer=tokenizer,
    input_file=f'{args.data_dir}/dev.txt',
    max_seq_length=args.max_seq_length)

hidden_size = pretrained_bert_model.local_parameters["hidden_size"]
ner_classifier = nemo_nlp.TokenClassifier(hidden_size=hidden_size,
                                          num_classes=len(tag_ids),
                                          dropout=args.fc_dropout)
ner_loss = nemo_nlp.TokenClassificationLoss(num_classes=len(tag_ids))


def create_pipeline(dataset, batch_size=args.batch_size,
                    local_rank=args.local_rank, num_gpus=args.num_gpus):
    data_layer = nemo_nlp.BertTokenClassificationDataLayer(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        local_rank=local_rank)
    input_ids, input_type_ids, input_mask, labels, seq_ids = data_layer()
    hidden_states = pretrained_bert_model(input_ids=input_ids,
                                          token_type_ids=input_type_ids,
                                          attention_mask=input_mask)
    logits = ner_classifier(hidden_states=hidden_states)
    loss = ner_loss(logits=logits, labels=labels, input_mask=input_mask)
    steps_per_epoch = len(data_layer) // (batch_size * num_gpus)
    return loss, steps_per_epoch, data_layer, [logits, seq_ids]


train_loss, steps_per_epoch, _, _ = create_pipeline(train_dataset)
_, _, data_layer, eval_tensors = create_pipeline(eval_dataset)
nf.logger.info(f"steps_per_epoch = {steps_per_epoch}")

# Create trainer and execute training action
train_callback = nemo.core.SimpleLossLoggerCallback(
    tensors=[train_loss],
    print_func=lambda x: print("Loss: {:.3f}".format(x[0].item())),
    get_tb_values=lambda x: [["loss", x[0]]],
    tb_writer=nf.tb_writer)

eval_callback = nemo.core.EvaluatorCallback(
    eval_tensors=eval_tensors,
    user_iter_callback=lambda x, y: eval_iter_callback(
        x, y, data_layer, tag_ids),
    user_epochs_done_callback=lambda x: eval_epochs_done_callback(
        x, tag_ids, output_file),
    tb_writer=nf.tb_writer,
    eval_step=steps_per_epoch)

ckpt_callback = nemo.core.CheckpointCallback(
    folder=nf.checkpoint_dir,
    epoch_freq=args.save_epoch_freq,
    step_freq=args.save_step_freq)


lr_policy_fn = get_lr_policy(args.lr_policy,
                             total_steps=args.num_epochs * steps_per_epoch,
                             warmup_ratio=args.lr_warmup_proportion)


nf.train(tensors_to_optimize=[train_loss],
         callbacks=[train_callback, eval_callback, ckpt_callback],
         lr_policy=lr_policy_fn,
         optimizer=args.optimizer_kind,
         optimization_params={"num_epochs": args.num_epochs,
                              "lr": args.lr})
