from pathlib import Path

from rich.console import Console

from nemo.collections.llm.peft.lora import LoRAMerge

from nemo.collections.llm.peft.api import _load_base_model_and_lora, _setup_trainer_and_restore_model_and_adapter, \
    _save_merged_weight

from nemo.lightning import Trainer, MegatronStrategy


def merge_lora(
    lora_checkpoint_path: str,
    output_path: str,
    legacy_ckpt: bool = False,
) -> None:
    """
    Merges the LoRA adapter weights into the base model's weights.
    The VLM version of this API additionally takes unfrozen vision encoder and projection weights
    into consideration in the merged checkpoint.

    Python Usage:
    ```python
    if __name__ == '__main__':
        vlm.peft.merge_lora(
            lora_checkpoint_path=your_lora_checkpoint_path,
            output_path=your_output_path,
        )
    ```

    Args:
        lora_checkpoint_path: The path to the LoRA checkpoint.
        output_path: The path to save the merged checkpoint.

    """
    from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed

    trainer = Trainer(
        devices=1,
        accelerator="cpu",
        strategy=MegatronStrategy(ddp="pytorch", setup_optimizers=False, plugins=bf16_mixed()),
    )

    # Load ckpt saved with TE < 1.14
    if legacy_ckpt:
        trainer.strategy.ckpt_load_strictness = False
    model, lora = _load_base_model_and_lora(lora_checkpoint_path)
    load_keys = ['.adapter.']
    if not lora.freeze_language_model:
        load_keys.append('language_model.')
    if not lora.freeze_vision_model:
        load_keys.append('vision_model.')
    if not lora.freeze_vision_projection:
        load_keys.append('vision_projection.')
    _setup_trainer_and_restore_model_and_adapter(Path(lora_checkpoint_path), trainer, model, lora, load_keys)

    lora_merge = LoRAMerge()
    merged_model = lora_merge(trainer.strategy.megatron_parallel).to(model.config.params_dtype)
    merged_weights = {k: v for k, v in merged_model.sharded_state_dict().items() if ".adapter." not in k}
    _save_merged_weight(output_path, merged_weights, model, trainer)

    console = Console()
    console.print(f"[green]✓ LoRA checkpoint merged and saved to {output_path}[/green]")