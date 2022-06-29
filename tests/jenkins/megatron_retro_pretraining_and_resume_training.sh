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


python examples/nlp/language_modeling/megatron_retro_pretraining.py \
trainer.devices=2 \
trainer.num_nodes=1 \
trainer.accelerator=gpu \
trainer.accumulate_grad_batches=1 \
trainer.limit_val_batches=2 \
exp_manager.resume_if_exists=True \
trainer.max_steps=10 \
trainer.precision=16 \
trainer.gradient_clip_val=1.0 \
trainer.val_check_interval=10 \
exp_manager.exp_dir=examples/nlp/language_modeling/retro_results \
model.data.data_prefix='' \
model.data.knn_index='' \
model.data.retrieval_prefix='' \
model.tensor_model_parallel_size=2 \
model.micro_batch_size=4 \
model.optim.name=fused_adam \
model.optim.lr=2e-4 \
model.optim.sched.warmup_steps=2 \
model.optim.sched.constant_steps=2 \
model.optim.sched.min_lr=8e-5 \
model.max_position_embeddings=128 \
model.encoder_seq_length=128 \
model.chunk_size=32 \
model.enc_num_layers=2 \
model.dec_num_layers=2 \
model.enc_cross_attention=[1] \
model.dec_cross_attention=[1] \
+model.data.mock=True

python examples/nlp/language_modeling/megatron_retro_pretraining.py \
trainer.devices=2 \
trainer.num_nodes=1 \
trainer.accelerator=gpu \
trainer.accumulate_grad_batches=1 \
trainer.limit_val_batches=2 \
exp_manager.resume_if_exists=True \
trainer.max_steps=20 \
trainer.precision=16 \
trainer.gradient_clip_val=1.0 \
trainer.val_check_interval=10 \
exp_manager.exp_dir=examples/nlp/language_modeling/retro_results \
model.data.data_prefix='' \
model.data.knn_index='' \
model.data.retrieval_prefix='' \
model.tensor_model_parallel_size=2 \
model.micro_batch_size=4 \
model.optim.name=fused_adam \
model.optim.lr=2e-4 \
model.optim.sched.warmup_steps=2 \
model.optim.sched.constant_steps=2 \
model.optim.sched.min_lr=8e-5 \
model.max_position_embeddings=128 \
model.encoder_seq_length=128 \
model.chunk_size=32 \
model.enc_num_layers=2 \
model.dec_num_layers=2 \
model.enc_cross_attention=[1] \
model.dec_cross_attention=[1] \
+model.data.mock=True

rm -rf examples/nlp/language_modeling/retro_results