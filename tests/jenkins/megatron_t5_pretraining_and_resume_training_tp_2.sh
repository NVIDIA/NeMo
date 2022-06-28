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


python $PWD/examples/nlp/language_modeling/megatron_t5_pretraining.py \
trainer.devices=2 \
trainer.accelerator=gpu \
trainer.log_every_n_steps=1 \
trainer.val_check_interval=10 \
trainer.limit_val_batches=2 \
trainer.accumulate_grad_batches=1 \
trainer.max_steps=10 \
trainer.precision=16 \
trainer.gradient_clip_val=1.0 \
exp_manager.exp_dir=examples/nlp/language_modeling/t5_pretrain_results \
model.tensor_model_parallel_size=2 \
model.seq_length=128 \
model.num_layers=4 \
model.hidden_size=64 \
model.num_attention_heads=8 \
model.activation='swiglu' \
model.bias_gelu_fusion=False \
model.activations_checkpoint_method='block' \
model.activations_checkpoint_num_layers=1 \
model.transformer_block_type='pre_ln' \
model.data.data_prefix=[.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document,.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document] \
model.position_embedding_type=relative \
model.data.index_mapping_dir=examples/nlp/language_modeling/t5_index_mappings

python $PWD/examples/nlp/language_modeling/megatron_t5_pretraining.py \
trainer.devices=2 \
trainer.accelerator=gpu \
trainer.log_every_n_steps=1 \
trainer.val_check_interval=10 \
trainer.limit_val_batches=2 \
trainer.accumulate_grad_batches=1 \
trainer.max_steps=10 \
trainer.precision=16 \
trainer.gradient_clip_val=1.0 \
exp_manager.exp_dir=examples/nlp/language_modeling/t5_pretrain_results \
exp_manager.resume_if_exists=True \
model.tensor_model_parallel_size=2 \
model.seq_length=128 \
model.num_layers=4 \
model.hidden_size=64 \
model.num_attention_heads=8 \
model.activation='swiglu' \
model.bias_gelu_fusion=False \
model.activations_checkpoint_method='block' \
model.activations_checkpoint_num_layers=1 \
model.transformer_block_type='pre_ln' \
model.data.data_prefix=[.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document,.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document] \
model.position_embedding_type=relative \
model.data.index_mapping_dir=examples/nlp/language_modeling/t5_index_mappings

rm -rf $PWD/examples/nlp/language_modeling/t5_pretrain_results
rm -rf $PWD/examples/nlp/language_modeling/t5_index_mappings