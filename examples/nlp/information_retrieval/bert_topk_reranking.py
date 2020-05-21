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

"""
See the tutorial and download the data here:
https://nvidia.github.io/NeMo/nlp/
neural-machine-translation.html#translation-with-pretrained-model
"""
import torch

import math
import nemo
import nemo.collections.nlp as nemo_nlp
from nemo import logging
from nemo.collections.nlp.callbacks.information_retrieval_callback import eval_epochs_done_callback, eval_iter_callback
import nemo.collections.nlp.nm.data_layers.information_retrieval_datalayer as ir_dl
from nemo.core import WeightShareTransform
from nemo.utils.lr_policies import get_lr_policy

parser = nemo.utils.NemoArgParser(description='Bert for Information Retrieval')
parser.set_defaults(
    train_dataset="train",
    eval_datasets=["bm25", "dpr"],
    work_dir="outputs/bert_ir",
    optimizer="adam_w",
    batch_size=8,
    eval_batch_size=8,
    lr_policy='WarmupAnnealing',
    lr=0.00001,
    weight_decay=0.01,
    max_steps=10000,
    iter_per_step=1,
    eval_freq=2500,
)
parser.add_argument("--data_dir", default="/home/ohrinchuk/datasets/msmarco/", type=str)
parser.add_argument("--collection_file", default="collection.medium.tsv", type=str)
parser.add_argument("--pretrained_model", default="bert-base-uncased", type=str)
parser.add_argument("--d_model", default=768, type=int)
parser.add_argument("--d_inner", default=3072, type=int)
parser.add_argument("--num_layers", default=12, type=int)
parser.add_argument("--num_attn_heads", default=12, type=int)
parser.add_argument("--embedding_dropout", default=0.1, type=float)
parser.add_argument("--ffn_dropout", default=0.1, type=float)
parser.add_argument("--attn_score_dropout", default=0.1, type=float)
parser.add_argument("--attn_layer_dropout", default=0.1, type=float)
parser.add_argument("--warmup_steps", default=1000, type=int)
parser.add_argument("--max_seq_length", default=256, type=int)
parser.add_argument("--save_epoch_freq", default=5, type=int)
parser.add_argument("--save_step_freq", default=2500, type=int)
parser.add_argument("--restore_checkpoint_from", default=None, type=str)
parser.add_argument("--num_negatives", default=5, type=int)
parser.add_argument("--num_eval_candidates", default=100, type=int)
parser.add_argument("--label_smoothing", default=0.0, type=float)
parser.add_argument("--freeze_encoder", action="store_true")
args = parser.parse_args()

nf = nemo.core.NeuralModuleFactory(
    backend=nemo.core.Backend.PyTorch,
    local_rank=args.local_rank,
    optimization_level=args.amp_opt_level,
    log_dir=args.work_dir,
    create_tb_writer=False,
    files_to_copy=[__file__],
)

tokenizer = nemo_nlp.data.NemoBertTokenizer(
    pretrained_model=args.pretrained_model
)
vocab_size = 8 * math.ceil(tokenizer.vocab_size / 8)
tokens_to_add = vocab_size - tokenizer.vocab_size

batch_reshape = nemo_nlp.nm.trainables.BertBatchReshaper()
encoder = nemo_nlp.nm.trainables.get_huggingface_model(
    pretrained_model_name=args.pretrained_model
)

model_name = args.pretrained_model.split("-")[0]

device = getattr(encoder, model_name).embeddings.word_embeddings.weight.get_device()
zeros = torch.zeros((tokens_to_add, args.d_model)).to(device=device)
getattr(encoder, model_name).embeddings.word_embeddings.weight.data = torch.cat(
    (getattr(encoder, model_name).embeddings.word_embeddings.weight.data, zeros)
)
if args.freeze_encoder:
    encoder.freeze()
classifier = nemo_nlp.nm.trainables.SequenceClassifier(
    hidden_size=args.d_model, num_classes=1, num_layers=1,
    dropout=args.ffn_dropout, log_softmax=False
)
loss_fn_train = nemo_nlp.nm.losses.ListwiseSoftmaxLoss(
    list_size=args.num_negatives+1)
loss_fn_eval = nemo_nlp.nm.losses.ListwiseSoftmaxLoss(
    list_size=args.num_eval_candidates)


train_documents = f"{args.data_dir}/{args.collection_file}"
# Training pipeline
train_queries = f"{args.data_dir}/queries.train.tsv"
train_triples = f"{args.data_dir}/{args.train_dataset}"
train_data_layer = ir_dl.BertInformationRetrievalDataLayerTrain(
    tokenizer=tokenizer,
    passages=train_documents,
    queries=train_queries,
    query_to_passages=train_triples,
    batch_size=args.batch_size,
    num_negatives=args.num_negatives
)
input_ids, input_mask, input_type_ids = train_data_layer()
input_ids, input_mask, input_type_ids = batch_reshape(
    input_ids=input_ids, input_mask=input_mask, input_type_ids=input_type_ids)
hiddens = encoder(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
scores = classifier(hidden_states=hiddens)
train_scores, train_loss = loss_fn_train(scores=scores)

# callback which prints training loss once in a while
train_callback = nemo.core.SimpleLossLoggerCallback(
    tensors=[train_loss],
    step_freq=100,
    print_func=lambda x: logging.info(str(x[0].item())),
    get_tb_values=lambda x: [["loss", x[0]]],
    tb_writer=nf.tb_writer,
)

callbacks = [train_callback]

def create_eval_pipeline(eval_dataset):

    eval_documents = f"{args.data_dir}/collection.{eval_dataset}.dev.small.tsv"
    eval_queries = f"{args.data_dir}/queries.dev.small.tsv"
    eval_topk_list = f"{args.data_dir}/top100.{eval_dataset}.dev.small.tsv"

    eval_data_layer = ir_dl.BertInformationRetrievalDataLayerEval(
        tokenizer=tokenizer,
        passages=train_documents,
        queries=eval_queries,
        query_to_passages=eval_topk_list,
        num_candidates=args.num_eval_candidates)

    input_ids_, input_mask_, input_type_ids_, query_id, passage_ids = eval_data_layer()
    input_ids_, input_mask_, input_type_ids_ = batch_reshape(
        input_ids=input_ids_, input_mask=input_mask_, input_type_ids=input_type_ids_)
    hiddens_ = encoder(input_ids=input_ids_, token_type_ids=input_type_ids_, attention_mask=input_mask_)
    scores_ = classifier(hidden_states=hiddens_)
    eval_scores, _ = loss_fn_eval(scores=scores_)
    
    return eval_scores, query_id, passage_ids


def parse_qrels(qrels):
    query2rel = {}
    for line in open(qrels, "r").readlines():
        query_id = int(line.split("\t")[0])
        psg_id = int(line.split("\t")[2])
        if query_id not in query2rel:
            query2rel[query_id] = [psg_id]
        else:
            query2rel[query_id].append(psg_id)
    return query2rel
query2rel = parse_qrels(f"{args.data_dir}/qrels.dev.small.tsv")


all_eval_tensors = {}
for eval_dataset in args.eval_datasets:
    scores, q_id, p_ids = create_eval_pipeline(eval_dataset)
    all_eval_tensors[eval_dataset] = [scores, q_id, p_ids]

callbacks.append(nemo.core.EvaluatorCallback(
    eval_tensors=all_eval_tensors[args.eval_datasets[0]],
    user_iter_callback=eval_iter_callback,
    user_epochs_done_callback=lambda x: eval_epochs_done_callback(
        x, query2rel=query2rel, topk=[1, 10],
        baseline_name=args.eval_datasets[0]),#, save_scores="infer/bert_bm25.pkl"),
    eval_step=args.eval_freq,
    tb_writer=nf.tb_writer))

callbacks.append(nemo.core.EvaluatorCallback(
    eval_tensors=all_eval_tensors[args.eval_datasets[1]],
    user_iter_callback=eval_iter_callback,
    user_epochs_done_callback=lambda x: eval_epochs_done_callback(
        x, query2rel=query2rel, topk=[1, 10],
        baseline_name=args.eval_datasets[1]),#, save_scores="infer/bert_dpr.pkl"),
    eval_step=args.eval_freq,
    tb_writer=nf.tb_writer))

# callback which saves checkpoints once in a while
ckpt_dir = nf.checkpoint_dir
ckpt_callback = nemo.core.CheckpointCallback(
    folder=ckpt_dir, epoch_freq=args.save_epoch_freq,
    load_from_folder=args.restore_checkpoint_from,
    step_freq=args.save_step_freq, checkpoints_to_keep=5)
callbacks.append(ckpt_callback)

# define learning rate decay policy
lr_policy_fn = get_lr_policy(args.lr_policy, total_steps=args.max_steps, warmup_steps=args.warmup_steps)

if args.max_steps is not None and args.num_epochs is not None:
    raise ValueError("Please specify either max_steps or num_epochs.")

if args.max_steps is not None:
    stop_training_condition = {"max_steps": args.max_steps}
else:
    stop_training_condition = {"num_epochs": args.num_epochs}

nf.train(
    tensors_to_optimize=[train_loss],
    callbacks=callbacks,
    optimizer=args.optimizer,
    lr_policy=lr_policy_fn,
    optimization_params={**stop_training_condition, "lr": args.lr, "weight_decay": args.weight_decay},
    batches_per_step=args.iter_per_step,
)
