
python tests/collections/llm/gpt_finetuning.py \
--restore_path /home/TestData/nemo2_ckpt/llama_68M_v3 \
--devices 2 \
--max_steps 3 \
--experiment_dir /tmp/nemo2_gpt_finetune/${{ github.run_id }} \
--peft lora \
--tp_size 2 \
--pp_size 1 \
--mbs 2

python tests/collections/llm/gpt_finetuning.py \
--restore_path /home/TestData/nemo2_ckpt/llama_68M_v3 \
--devices 2 \
--max_steps 6 \
--experiment_dir /tmp/nemo2_gpt_finetune/${{ github.run_id }} \
--peft lora \
--tp_size 2 \
--pp_size 1 \
--mbs 2
