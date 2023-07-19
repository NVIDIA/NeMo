**Support of community models**
===============
Take LLaMA as example.

Checkpoint conversion
---------------


```
python /path/to/convert_hf_llama_to_nemo.py --in-file ./ --out-file /path/to/llama-nemo-ckpt/llama-13b.nemo
```
Use the following commands to modify the nemo checkpoint:

```
# decompress nemo checkpoint
tar -xvf llama-13b.nemo
# modify model_config.yaml
# hidden_dropout: 0.0 -> hidden_dropout: 0.1
# compress all files
tar -cvf model_config.yaml model_weight.ckpt tokenizer.model
```
Of course most of configs could be overrided when launching the training script.

Reshape TP/PP
---------------
More configs and related explanations could be found [here](https://github.com/lhb8125/nemo-for-vivo/blob/support_llama/examples/nlp/language_modeling/megatron_change_num_partitions.py#L38).

```
python /path/to/megatron_change_num_partitions.py --model_file=llama-13b.nemo --target_file=./llama-13b-tp4.nemo --model_class="nemo.collections.nlp.models.language_modeling.megatron_llama_model.MegatronLLAMAModel" --tensor_model_parallel_size=1 --target_tensor_model_parallel_size=4 --pipeline_model_parallel_size=1 --target_pipeline_model_parallel_size=1 --precision=32 --target_pipeline_model_parallel_split_rank=0 --tokenizer_model_path=/path/to/tokenizer.model
```

Launch training
---------------
* restore_from_path: load from .nemo checkpoint file
* exp_manager.resume_if_exists: continue the training from the last checkpoints
* transformer_engine: doesn't support swiglu yet. When setting to True, the activation will fall back to gelu type.

```
python /path/to/megatron_llama_pretraining.py     --config-path=path/to/configs     --config-name=llama-13b    trainer.num_nodes=4     trainer.devices=8     trainer.max_steps=300000     model.tensor_model_parallel_size=4     model.pipeline_model_parallel_size=1     model.virtual_pipeline_model_parallel_size=NULL     model.micro_batch_size=1
```
