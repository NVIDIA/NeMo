coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo scripts/vlm/mllama_generate.py \
    --local_model_path /home/TestData/nemo2_ckpt/Llama-3.2-11B-Vision-Instruct \
    --processor_name /home/TestData/HF_HOME/hub/models--meta-llama--Llama-3.2-11B-Vision-Instruct/snapshots/9eb2daaa8597bf192a8b0e73f848f3a102794df5 \
    --num_tokens_to_generate 3
