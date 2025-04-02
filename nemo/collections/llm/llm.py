from nemo.common.ckpt.model import PreTrainedModel


class LLM(PreTrainedModel):
    """A class for loading and using LLM models.

    Args:
        model: The model to load.
        env: The environment to use.
        parse: The parser to use.
        lazy: Whether to lazy load the model.
        trust_remote_code: Whether to trust remote code.
        cache_dir: The cache directory to use.
        hub_token: The hub token to use.

    Load as HF-checkpoint as megatron:
        >>> from nemo.collections import llm
        >>> model = llm.LLM("meta-llama/Llama-3.1-8B-Instruct", parse="megatron")
        >>> model()

    Load nemo-checkpoint
        >>> from nemo.collections import llm
        >>> model = llm.LLM("path/to/nemo-checkpoint")
        >>> model()
    """
    ...





"""
>>> model
AutoModel(
  (model): "meta-llama/Llama-3.2-3B"
  (env): LightningEnv(
	  devices=2,
	  strategy=nl.MegatronStrategy(tensor_model_parallelism=2), 
	  plugins=nl.MegatronMixedPrecision("16-mixed")
  )
  (parse): "megatron"
  (peft): PEFT({"*.linear*": LoRA(rank=10, alpha=1.0)}, ckpt_path="...")
)


>>> reload = LLM("to/ckpt", env=nl.Trainer(accelerator="auto", devices=1))


>>> finetune_model.plan.materialized
AutoModel(
  (env): LightningEnv(accelerator=auto, devices=1)
  (parse): MegatronModelConverter(
    context_converter=auto_model.model.llama.llama_hf_to_megatron
    (source): ModelContext(
      (path): hf://meta-llama/Llama-3.2-3B
      (class): transformers.models.llama.modeling_llama.LlamaForCausalLM
    )
    (target): ModelContext(
      (path): /workspaces/models/nemo/megatron/meta-llama/Llama-3.2-3B
      (class): nemo.collections.llm.LlamaModel
    )
  )
  (init): InitMegatronModel(nemo.collections.llm.LlamaModel)(
	(convert_state): MegatronStateConverter(...)
	(load): LoadMegatronModel(path="/workspaces/models/nemo/megatron/meta-llama/Llama-3.2-3B")
	(peft): PEFT(
	  (freeze): Freeze("2B params frozen (99%)")
	  (adapters): MaterializedSelector({
		  "layers.0.linear1": LoRA(mcore_lora_linear_impl)(rank=10, alpha=1.0),
		  "layers.0.linear2": LoRA(mcore_lora_linear_impl)(rank=10, alpha=1.0),
		  "layers.1.linear1": LoRA(mcore_lora_linear_impl)(rank=10, alpha=1.0),
		  "layers.1.linear2": LoRA(mcore_lora_linear_impl)(rank=10, alpha=1.0),
		  "layers.2.linear1": LoRA(mcore_lora_linear_impl)(rank=10, alpha=1.0),
		  "layers.2.linear2": LoRA(mcore_lora_linear_impl)(rank=10, alpha=1.0),
	  })
	  (wrap): PEFTModule("total_params=982843392, trainable_params=9699328 (0.99%)")
	  (load): LoadPartialMegatronModel("path/to/peft/ckpt")
	)
	(mixed_precision): MegatronMixedPrecision(...)
	(ddp): MegatronDDP(
	  (ddp_config): DistributedDataParallelConfig(...)
	)
  )
)



"""