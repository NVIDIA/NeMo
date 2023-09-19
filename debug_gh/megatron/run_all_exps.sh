bash run_gpt3.4b_pretrain.sh 8 1024 True
bash run_gpt3.4b_pretrain_recompute.sh 16 1024 True full block 8
bash run_gpt3.4b_pretrain_offload.sh 16 1024 True transformer_layer 8 group_sync
bash run_gpt3.4b_pretrain_offload.sh 16 1024 True ln,ffn_act,ffn2 15 group_async
bash run_gpt3.4b_pretrain.sh 4 2048 True
bash run_gpt3.4b_pretrain_recompute.sh 8 2048 True full block 8
bash run_gpt3.4b_pretrain_offload.sh 8 2048 True transformer_layer 8 group_sync
bash run_gpt3.4b_pretrain_offload.sh 8 2048 True ln,ffn_act,ffn2 15 group_async
bash run_gpt3.4b_pretrain.sh 2 4096 True
bash run_gpt3.4b_pretrain_recompute.sh 4 4096 True full block 8
bash run_gpt3.4b_pretrain_offload.sh 4 4096 True transformer_layer 8 group_sync
bash run_gpt3.4b_pretrain_offload.sh 4 4096 True ln,ffn_act,ffn2 15 group_async
bash run_gpt3.4b_pretrain.sh 1 8192 True
bash run_gpt3.4b_pretrain_recompute.sh 2 8192 True full block 8
bash run_gpt3.4b_pretrain_offload.sh 2 8192 True transformer_layer 8 group_sync
bash run_gpt3.4b_pretrain_offload.sh 2 8192 True ln,ffn_act,ffn2 15 group_async
bash run_gpt3.4b_pretrain_recompute.sh 1 16384 True full block 8
bash run_gpt3.4b_pretrain_offload.sh 1 16384 True transformer_layer 8 group_sync
bash run_gpt3.4b_pretrain_offload.sh 1 16384 True ln,ffn_act,attn_fn 15 group_async
bash run_gpt3.4b_pretrain_recompute.sh 1 32768 True full block 15
bash run_gpt3.4b_pretrain_offload.sh 1 32768 True transformer_layer 16 group_sync
bash run_gpt3.4b_pretrain_offload.sh 1 32768 True transformer_layer 15 group_async