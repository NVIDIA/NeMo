#!/bin/bash

# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

python examples/nlp/language_modeling/megatron_bert_pretraining.py \
trainer.devices=2 \
trainer.accelerator=gpu \
trainer.log_every_n_steps=1 \
trainer.val_check_interval=10 \
trainer.limit_val_batches=2 \
trainer.accumulate_grad_batches=2 \
trainer.max_steps=10 \
trainer.precision=16 \
trainer.gradient_clip_val=1.0 \
exp_manager.exp_dir=examples/nlp/language_modeling/bert_pretrain_results \
model.tensor_model_parallel_size=2 \
model.optim.name=fused_adam \
model.optim.lr=2e-4 \
model.optim.sched.warmup_steps=2 \
model.optim.sched.constant_steps=2 \
model.optim.sched.min_lr=8e-5 \
model.max_position_embeddings=128 \
model.encoder_seq_length=128 \
model.data.seq_length=128 \
model.tokenizer.vocab_file=/home/TestData/nlp/megatron_bert/data/bert/vocab.txt \
model.num_layers=8 \
model.hidden_size=256 \
model.num_attention_heads=8 \
model.activations_checkpoint_method='block' \
model.activations_checkpoint_num_layers=1 \
model.data.data_prefix=[.5,/home/TestData/nlp/megatron_bert/data/bert/simple_wiki_bert_preproc_text_sentence,.5,/home/TestData/nlp/megatron_bert/data/bert/simple_wiki_bert_preproc_text_sentence] \
model.data.index_mapping_dir=examples/nlp/language_modeling/bert_index_mappings

python examples/nlp/language_modeling/megatron_bert_pretraining.py \
trainer.devices=2 \
trainer.accelerator=gpu \
trainer.log_every_n_steps=1 \
trainer.val_check_interval=10 \
trainer.limit_val_batches=2 \
trainer.accumulate_grad_batches=2 \
trainer.max_steps=20 \
trainer.precision=16 \
trainer.gradient_clip_val=1.0 \
exp_manager.exp_dir=examples/nlp/language_modeling/bert_pretrain_results \
exp_manager.resume_if_exists=True \
model.tensor_model_parallel_size=2 \
model.optim.name=fused_adam \
model.optim.lr=2e-4 \
model.optim.sched.warmup_steps=2 \
model.optim.sched.constant_steps=2 \
model.optim.sched.min_lr=8e-5 \
model.max_position_embeddings=128 \
model.encoder_seq_length=128 \
model.data.seq_length=128 \
model.tokenizer.vocab_file=/home/TestData/nlp/megatron_bert/data/bert/vocab.txt \
model.num_layers=8 \
model.hidden_size=256 \
model.num_attention_heads=8 \
model.activations_checkpoint_method='block' \
model.activations_checkpoint_num_layers=1 \
model.data.data_prefix=[.5,/home/TestData/nlp/megatron_bert/data/bert/simple_wiki_bert_preproc_text_sentence,.5,/home/TestData/nlp/megatron_bert/data/bert/simple_wiki_bert_preproc_text_sentence] \
model.data.index_mapping_dir=examples/nlp/language_modeling/bert_index_mappings

rm -rf examples/nlp/language_modeling/bert_pretrain_results
rm -rf examples/nlp/language_modeling/bert_index_mappings