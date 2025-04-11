export TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/TestData/automodel/hf_home
export CUDA_VISIBLE_DEVICES=0,1
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo -m torch.distributed.launch --nproc_per_node=2 --use-env examples/llm/peft/automodel.py \
    --model /home/TestData/akoumparouli/hf_mixtral_2l/ \
    --max-steps 3 \
    --devices 2 \
    --strategy fsdp2 --fp8
