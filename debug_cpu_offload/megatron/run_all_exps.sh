bash run_gpt3.4b_pretrain.sh 4 1024 True
bash run_gpt3.4b_pretrain_recompute.sh 8 1024 True full block 2
bash run_gpt3.4b_pretrain_offload.sh 8 1024 True transformer_layer 2 group_sync
bash run_gpt3.4b_pretrain_offload.sh 8 1024 True ln,ffn_act 15 group_async
bash run_gpt3.4b_pretrain.sh 2 2048 True
bash run_gpt3.4b_pretrain_recompute.sh 4 2048 True full block 2
bash run_gpt3.4b_pretrain_offload.sh 4 2048 True transformer_layer 2 group_sync
bash run_gpt3.4b_pretrain_offload.sh 4 2048 True ln,ffn_act 15 group_async
bash run_gpt3.4b_pretrain.sh 1 4096 True
bash run_gpt3.4b_pretrain_recompute.sh 2 4096 True full block 2
bash run_gpt3.4b_pretrain_offload.sh 2 4096 True transformer_layer 2 group_sync
bash run_gpt3.4b_pretrain_offload.sh 2 4096 True ln,ffn_act 15 group_async
bash run_gpt3.4b_pretrain_recompute.sh 1 8192 True full block 4
bash run_gpt3.4b_pretrain_offload.sh 1 8192 True transformer_layer 4 group_sync
bash run_gpt3.4b_pretrain_offload.sh 1 8192 True ln 15 group_async
bash run_gpt3.4b_pretrain_recompute.sh 1 16348 True full block 12
bash run_gpt3.4b_pretrain_offload.sh 1 16348 True transformer_layer 12 group_sync trainer.max_steps=11
bash run_gpt3.4b_pretrain_offload.sh 1 16348 True transformer_layer 15 group_async trainer.max_steps=11