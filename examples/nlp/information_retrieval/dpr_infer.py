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

import torch
import numpy as np

import time
import math
import nemo
import nemo.collections.nlp as nemo_nlp
from nemo import logging
from nemo.collections.nlp.callbacks.information_retrieval_callback import \
    eval_epochs_done_callback, eval_iter_callback
import nemo.collections.nlp.nm.data_layers.information_retrieval_datalayer as ir_dl
from nemo.core import WeightShareTransform
from nemo.utils.lr_policies import get_lr_policy

parser = nemo.utils.NemoArgParser(description='Bert for Information Retrieval')

parser.set_defaults(eval_datasets=["dev.small"], work_dir="outputs/test2", amp_opt_level="O2")
parser.add_argument("--data_dir", default="/home/ohrinchuk/datasets/msmarco", type=str)
parser.add_argument("--pretrained_model", default="bert-base-uncased", type=str)
parser.add_argument("--d_model", default=768, type=int)
parser.add_argument("--restore_checkpoint_from", default=None, type=str)
parser.add_argument("--data_for_eval", default="passages", type=str)
parser.add_argument("--restore_path", type=str)
parser.add_argument("--chunk_id", default=0, type=int)
args = parser.parse_args()

nf = nemo.core.NeuralModuleFactory(
    backend=nemo.core.Backend.PyTorch,
    local_rank=args.local_rank,
    optimization_level=args.amp_opt_level,
    log_dir=args.work_dir)

tokenizer = nemo_nlp.data.NemoBertTokenizer(pretrained_model=args.pretrained_model)
vocab_size = 8 * math.ceil(tokenizer.vocab_size / 8)
tokens_to_add = vocab_size - tokenizer.vocab_size

encoder = nemo_nlp.nm.trainables.get_huggingface_model(
    pretrained_model_name=args.pretrained_model)
device = encoder.bert.embeddings.word_embeddings.weight.get_device()
zeros = torch.zeros((tokens_to_add, args.d_model)).to(device=device)
encoder.bert.embeddings.word_embeddings.weight.data = torch.cat(
    (encoder.bert.embeddings.word_embeddings.weight.data, zeros))
encoder.restore_from(path=args.restore_path, local_rank=args.local_rank)

data_layer_params = {"tokenizer": tokenizer,
                     "batch_size": args.batch_size,
                     "passages": None,
                     "queries": None}
if args.data_for_eval == "passages":
    data_layer_params["passages"] = f"{args.data_dir}/collection.{args.chunk_id}.tsv"
    filename = f"passages.{args.chunk_id}.tsv"
elif args.data_for_eval == "queries":
    filename = f"queries.{args.eval_datasets[0]}.tsv"
    data_layer_params["queries"] = f"{args.data_dir}/{filename}"

eval_data_layer = ir_dl.BertDensePassageRetrievalDataLayerInfer(**data_layer_params)
input_ids, input_mask, input_type_ids, idx = eval_data_layer()
hiddens = encoder(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)

start_time = time.time()
evaluated_tensors = nf.infer(tensors=[hiddens, idx])


if args.local_rank == 0:
    vectors = np.vstack([tensor[:, 0].detach().numpy().astype(np.float16)
                         for tensor in evaluated_tensors[0]])
    indices = np.hstack([tensor.detach().numpy() for tensor in evaluated_tensors[1]])
    np.savez(f"{args.work_dir}/{filename}.npz", vectors=vectors, indices=indices)
