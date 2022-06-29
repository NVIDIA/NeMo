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

python examples/nlp/machine_translation/megatron_nmt_training.py \
trainer.devices=2 \
trainer.accelerator=gpu \
trainer.log_every_n_steps=1 \
trainer.val_check_interval=10 \
+trainer.limit_val_batches=2 \
trainer.accumulate_grad_batches=1 \
trainer.max_steps=10 \
trainer.precision=16 \
trainer.gradient_clip_val=1.0 \
exp_manager.exp_dir=examples/nlp/machine_translation/megatron_nmt_results \
model.tensor_model_parallel_size=2 \
model.seq_length=128 \
model.num_layers=4 \
model.hidden_size=64 \
model.num_attention_heads=8 \
model.activation='swiglu' \
model.masked_softmax_fusion=False \
model.bias_gelu_fusion=False \
model.activations_checkpoint_method='block' \
model.activations_checkpoint_num_layers=1 \
model.micro_batch_size=2 \
model.global_batch_size=4 \
model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
~model.test_ds \
model.train_ds.dataset_type=text_memmap \
model.encoder_tokenizer.library=sentencepiece \
model.encoder_tokenizer.model=/home/TestData/nlp/nmt/toy_data/spm_64k_all_langs_plus_en.model \
model.decoder_tokenizer.library=sentencepiece \
model.decoder_tokenizer.model=/home/TestData/nlp/nmt/toy_data/spm_64k_all_langs_plus_en.model

sh python examples/nlp/machine_translation/megatron_nmt_training.py \
trainer.devices=2 \
trainer.accelerator=gpu \
trainer.log_every_n_steps=1 \
trainer.val_check_interval=10 \
+trainer.limit_val_batches=2 \
trainer.accumulate_grad_batches=1 \
trainer.max_steps=10 \
trainer.precision=16 \
trainer.gradient_clip_val=1.0 \
exp_manager.exp_dir=examples/nlp/machine_translation/megatron_nmt_results \
model.tensor_model_parallel_size=2 \
model.seq_length=128 \
model.num_layers=4 \
model.hidden_size=64 \
model.num_attention_heads=8 \
model.activation='swiglu' \
model.bias_gelu_fusion=False \
model.masked_softmax_fusion=False \
model.activations_checkpoint_method='block' \
model.activations_checkpoint_num_layers=1 \
model.micro_batch_size=2 \
model.global_batch_size=4 \
model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
~model.test_ds \
model.train_ds.dataset_type=text_memmap \
model.encoder_tokenizer.library=sentencepiece \
model.encoder_tokenizer.model=/home/TestData/nlp/nmt/toy_data/spm_64k_all_langs_plus_en.model \
model.decoder_tokenizer.library=sentencepiece \
model.decoder_tokenizer.model=/home/TestData/nlp/nmt/toy_data/spm_64k_all_langs_plus_en.model

rm -rf examples/nlp/machine_translation/megatron_nmt_results