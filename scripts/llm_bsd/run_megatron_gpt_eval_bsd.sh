
#######
# Environment Variables to set
#######

branch_name="llm_nemo_v01"
source ~/anaconda3/etc/profile.d/conda.sh || exit 1
conda activate llm01 || exit 1
echo "which python ? Answer:";
which python || exit 1
export PYTHONPATH=/home/taejinp/projects/$branch_name/NeMo:$PYTHONPATH
echo "which NeMo ? Answer:";
python -c "import nemo; print(nemo.__path__)" || exit 1

NEMO_PATH=/home/taejinp/projects/$branch_name/NeMo
BASE_PATH=$NEMO_PATH/examples/nlp/language_modeling


############# Calculate P(S|W) using the prompt with questions 
# prompt1="\[speaker0\]: and i i already got another apartment for when i moved out \[speaker1\]: oh you did \[speaker0\]: i had to put down like a deposit and um you know and pay the \n \[End of Dialogue\] \n The next word is \[rent\]. Which speaker spoke this word \[speaker0\] or \[speaker1\] ?\n\n Answer:"
# prompt2="\[speaker1\]: and i i already got another apartment for when i moved out \[speaker0\]: oh you did \[speaker1\]: i had to put down like a deposit and um you know and pay the \n \[End of Dialogue\] \n The next word is \[rent\]. Which speaker spoke this word \[speaker0\] or \[speaker1\] ?\n\n Answer:"
# TOKEN_COUNT=4
# PORT=5554

############# Calculate P(W)
prompt1="\[speaker1\]: and i i already got another apartment for when i moved out \[speaker0\]: oh you did \[speaker1\]: i had to put down like a deposit and um you know and pay the rent"
prompt2="\[speaker1\]: and i i already got another apartment for when i moved out \[speaker0\]: oh you did \[speaker1\]: i had to put down like a deposit and um you know and pay the \[speaker0\]: rent"
TOKEN_COUNT=0
PORT=5550

PATH_TO_MODEL="/raid_c/models/LLM/2B_SFT_TP1_PP1.nemo"
export CUDA_VISIBLE_DEVICES="1"

echo "--------------------[Script Running]----------------------"
python $BASE_PATH/megatron_gpt_eval_bsd.py \
    port=$PORT \
    gpt_model_file=$PATH_TO_MODEL \
    trainer.devices=1 \
    trainer.num_nodes=1 \
    inference.compute_logprob=True \
    inference.temperature=0.75 \
    inference.greedy=True \
    inference.top_k=0 \
    inference.top_p=0.9 \
    inference.all_probs=True \
    inference.min_tokens_to_generate=$TOKEN_COUNT \
    inference.tokens_to_generate=$TOKEN_COUNT \
    inference.repetition_penalty=1.5 \
    tensor_model_parallel_size=-1 \
    pipeline_model_parallel_size=-1 \
    server=True \
    prompts="[$prompt1,$prompt2]" \