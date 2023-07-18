# setup
export TRANSFORMER_OFFLINE=1
#export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1
export TORCH_DISTRIBUTED_DEBUG=INFO
#export TORCH_CPP_LOG_LEVEL=INFO

export PYTHONPATH=/opt/NeMo:\${PYTHONPATH};
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nnodes 1 --nproc-per-node 8 --rdzv-endpoint=localhost:0 /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
	  trainer.precision=16 \
	  trainer.num_nodes=1 \
	  trainer.devices=8 \
	  model.tensor_model_parallel_size=1 \
	  model.pipeline_model_parallel_size=4 \
	  model.transformer_engine=False \
	  model.standalone_embedding_stage=True \
	  model.global_batch_size=4 \
	  model.micro_batch_size=1 \
	  model.use_flash_attention=False \
	  model.tokenizer.merge_file=/home/u00u4x8p3enW0rzLCW357/gpt_dataset/bpe/merges.txt \
	  model.tokenizer.vocab_file=/home/u00u4x8p3enW0rzLCW357/gpt_dataset/bpe/vocab.json \
	  model.data.data_prefix=[1.0,/home/u00u4x8p3enW0rzLCW357/gpt_dataset/Wikipedia_en_ftfy_id_shuf_text_document] \
	  +trainer.num_sanity_val_steps=0
