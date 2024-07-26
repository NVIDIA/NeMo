## Setup
In this part, we are going to export SALM model into TRTLLM.
First, let's download the [SALM nemo model](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/speechllm_fc_llama2_7b/) from NVIDIA ngc.

```
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/nemo/speechllm_fc_llama2_7b/1.23.1/files?redirect=true&path=speechllm_fc_llama2_7b.nemo' -O speechllm_fc_llama2_7b.nemo
```

Then, we need to extract the different parts of SALM.
```
output=$PWD/output
python3 separate_salm_weights.py --model_file_path=speechllm_fc_llama2_7b.nemo --output_dir=$output
```
It takes a while to run the above command.

Under the `output` dir, you'll see:
```
output
    |___speechllm_fc_llama2_7b_lora.nemo
    |___speechllm_fc_llama2_7b_perception
    |         |____model_config.yaml
    |         |____model_weights.ckpt
    |___speechllm_fc_llama2_7b_llm.nemo
    |___ xxx.tokenizer.model
```

After we get the lora nemo model and llm nemo model, we can merge the lora part into the llm by:
```
python /opt/NeMo/scripts/nlp_language_modeling/merge_lora_weights/merge.py \
    trainer.accelerator=gpu \
    tensor_model_parallel_size=1 \
    pipeline_model_parallel_size=1 \
    gpt_model_file=output/speechllm_fc_llama2_7b_llm.nemo \
    lora_model_path=output/speechllm_fc_llama2_7b_lora/lora.nemo \
    merged_model_path=speechllm_fc_llama2_7b_llm_merged.nemo
```

Now we are able to export the engine by:
```
python3 export_salm.py \
    model.perception_model_path=output/speechllm_fc_llama2_7b_perception \
    model.llm_model_path=output/speechllm_fc_llama2_7b_llm_merged.nemo
```