from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import yaml
from megatron.core.optimizer import OptimizerConfig

from nemo.collections.llm.gpt.model.base import torch_dtype_from_mcore_config
from nemo.collections.llm.gpt.model.llama import (
    Llama31Config,
    LlamaConfig,
)
from nemo.lightning import _strategy_lib, io
from nemo.lightning.io.state import TransformFns
from nemo.lightning.pytorch.utils import dtype_from_hf
from nemo.tron.checkpointing import save_checkpoint
from nemo.tron.config import CheckpointConfig, ConfigContainer, LoggerConfig
from nemo.tron.container.utils.instantiate import instantiate
from nemo.tron.converter.common import (
    get_full_mcore_state_dict,
    save_hf_tokenizer_assets,
    temporary_distributed_context,
)
from nemo.tron.state import GlobalState
from nemo.tron.tokenizers.tokenizer import _HuggingFaceTokenizer, build_tokenizer
from nemo.utils import logging

if TYPE_CHECKING:
    from transformers import LlamaConfig as HFLlamaConfig
    from transformers import LlamaForCausalLM


class _ModelState:
    """
    Helper class for used for to modify state dict of a source model during model conversion.
    """

    def __init__(self, state_dict):
        self._state_dict = state_dict

    def state_dict(self):
        # pylint: disable=C0115,C0116
        return self._state_dict

    def to(self, dtype):
        # pylint: disable=C0115,C0116
        for k, v in self._state_dict.items():
            if v.dtype != dtype:
                logging.warning(f"Converting {k} from {v.dtype} (source model) to {dtype} (target model)")
            self._state_dict[k] = v.to(dtype)


class HFLlamaTronExporter:
    """Exporter to convert NeMo Llama models to Hugging Face format.

    This class handles the conversion process from a NeMo Llama model checkpoint
    to the Hugging Face Transformers format. It extracts the model weights and
    configuration from a NeMo checkpoint, maps them to the corresponding Hugging Face
    structure, and saves the result as a Hugging Face model.

    Args:
        input_path (Path): Path to the NeMo model checkpoint directory
        output_path (Path): Path where the converted Hugging Face model will be saved

    Example:
        ```python
        from pathlib import Path
        from nemo.tron.converter.llama import HFLlamaTronExporter

        # Define paths
        nemo_model_path = Path("/path/to/nemo/llama/model")
        hf_output_path = Path("/path/to/save/hf/model")

        # Initialize the exporter
        exporter = HFLlamaTronExporter(
            input_path=nemo_model_path,
            output_path=hf_output_path
        )

        # Perform the conversion
        output_dir = exporter.apply()
        print(f"Model converted and saved to: {output_dir}")

        # Load the converted model with Hugging Face
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(hf_output_path)
        tokenizer = AutoTokenizer.from_pretrained(hf_output_path)
        ```

    Notes:
        - The conversion process may require significant memory depending on model size
        - The exporter handles mapping between different weight naming conventions
        - Best used with NeMo Llama models trained with the NeMo framework
    """

    def __init__(self, input_path: Path, output_path: Path):
        self.input_path = input_path
        self.output_path = output_path
        self.tokenizer = None
        self._config = None

    def init_hf_model(self, dtype=torch.bfloat16) -> "LlamaForCausalLM":
        """Initialize a new Hugging Face Llama model with the specified data type.

        Args:
            dtype (torch.dtype): The data type for the model parameters. Default: torch.bfloat16

        Returns:
            LlamaForCausalLM: An initialized Hugging Face Llama model with no weights loaded
        """
        from transformers import AutoModelForCausalLM
        from transformers.modeling_utils import no_init_weights

        with no_init_weights(True):
            return AutoModelForCausalLM.from_config(self.config, torch_dtype=dtype)

    def ckpt_load(self, path: Path) -> tuple[dict, dict]:
        """
        This function loads the state dict directly from a distributed checkpoint, and modify the state dict
        so that it is consistent with the key names you would get from loading the checkpoint into a model.
        This is a more memory-efficient method to obtain a state dict without initializing the nemo model.

        Args:
            path (Path): The path from which the model will be loaded.

        Returns
        -------
            Tuple[Dict, Dict]: The loaded state dict and the yaml config dict.
        """
        tron_yaml = path / "run_config.yaml"
        assert tron_yaml.exists()
        with open(tron_yaml, "r") as stream:
            _config = yaml.safe_load(stream)
        config = _config["model_config"]
        config = instantiate(config)
        try:
            self.tokenizer = build_tokenizer(instantiate(_config["tokenizer_config"]))
        except Exception:
            logging.warning("Failed to build tokenizer")

        dist_ckpt_folder = path / "weights" if (path / "weights").exists() else path
        state_dict = {}
        state_dict = get_full_mcore_state_dict(dist_ckpt_folder)

        return state_dict, config

    def apply(self) -> Path:
        """Execute the conversion process from NeMo to Hugging Face format.

        This method:
        1. Loads the NeMo checkpoint
        2. Initializes a new Hugging Face model
        3. Converts and transfers the weights
        4. Saves the converted model to the output path

        Returns:
            Path: The path where the converted model is saved
        """
        logging.info("Loading Llama checkpoint. This may take a while...")
        source, source_config = self.ckpt_load(self.input_path)
        self._source_config = source_config
        logging.info("Llama checkpoint loaded.")
        target = self.init_hf_model(torch_dtype_from_mcore_config(source_config))
        target = self.convert_state(source, target)

        target = target.cpu()
        if self.config.tie_word_embeddings:
            state_dict = target.state_dict()
            state_dict.pop("lm_head.weight")
            target.save_pretrained(self.output_path, state_dict=state_dict)
        else:
            target.save_pretrained(self.output_path)

        try:
            self.tokenizer._tokenizer.save_pretrained(self.output_path)
        except Exception:
            logging.warning("Failed to save tokenizer")

        return self.output_path

    def convert_state(self, source, target):
        """Convert state dict from NeMo format to HF format.

        Maps the weights from the NeMo model to the HF model according to
        the appropriate mapping scheme.

        Args:
            source: Source NeMo model
            target: Target HF model

        Returns:
            The target model with weights transferred from source
        """
        mapping = {
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
        }

        transforms = [
            io.state_transform(
                source_key="decoder.layers.*.self_attention.linear_qkv.weight",
                target_key=(
                    "model.layers.*.self_attn.q_proj.weight",
                    "model.layers.*.self_attn.k_proj.weight",
                    "model.layers.*.self_attn.v_proj.weight",
                ),
                fn=TransformFns.split_qkv,
            ),
            io.state_transform(
                source_key="decoder.layers.*.mlp.linear_fc1.weight",
                target_key=("model.layers.*.mlp.gate_proj.weight", "model.layers.*.mlp.up_proj.weight"),
                fn=TransformFns.split_fc1,
            ),
            io.state_transform(
                source_key="embedding.word_embeddings.weight",
                target_key="model.embed_tokens.weight",
                fn=TransformFns.prune_padding,
            ),
        ]
        if not self.config.tie_word_embeddings:
            transforms.append(
                io.state_transform(
                    source_key="output_layer.weight",
                    target_key="lm_head.weight",
                    fn=TransformFns.prune_padding,
                )
            )

        _source = _ModelState(source)
        _source.config = self._source_config
        return io.apply_transforms(
            _source,
            target,
            mapping=mapping,
            transforms=transforms,
        )

    @property
    def config(self) -> "HFLlamaConfig":
        """Generate a Hugging Face Llama configuration from the NeMo model configuration.

        This property maps NeMo configuration parameters to their Hugging Face equivalents.

        Returns:
            HFLlamaConfig: A Hugging Face Llama configuration
        """
        if self._config is not None:
            return self._config

        source = self._source_config
        from transformers import LlamaConfig as HFLlamaConfig

        rope_scaling = None
        # For Llama 3.1 and Llama 3.2, rope_scaling is used and thus needed to parsed to the config
        if isinstance(source, Llama31Config):
            rope_scaling = {
                "factor": source.scale_factor,
                "low_freq_factor": source.low_freq_factor,
                "high_freq_factor": source.high_freq_factor,
                "original_max_position_embeddings": source.old_context_len,
                "rope_type": "llama3",
            }

        self._config = HFLlamaConfig(
            num_hidden_layers=source.num_layers,
            hidden_size=source.hidden_size,
            intermediate_size=source.ffn_hidden_size,
            num_attention_heads=source.num_attention_heads,
            max_position_embeddings=source.seq_length,
            initializer_range=source.init_method_std,
            rms_norm_eps=source.layernorm_epsilon,
            num_key_value_heads=source.num_query_groups,
            rope_theta=source.rotary_base,
            vocab_size=source.vocab_size,
            tie_word_embeddings=source.share_embeddings_and_output_weights,
            rope_scaling=rope_scaling,
            bos_token_id=self.tokenizer.bos_id if self.tokenizer else None,
            eos_token_id=self.tokenizer.eos_id if self.tokenizer else None,
        )
        return self._config


class HFLlamaImporter:
    """Importer for converting Hugging Face Llama models to NeMo format.

    This class handles the conversion of Hugging Face's LlamaForCausalLM models
    to NeMo's LlamaModel format, including weight mapping and configuration translation.
    """

    def __init__(self, input_path: Path, output_path: Path):
        """Initialize a NeMo LlamaModel instance.

        Returns:
            LlamaModel: Initialized NeMo Llama model with the appropriate configuration
                        and tokenizer.
        """
        self.input_path = input_path
        self.output_path = output_path

    def init_tron_model(self, cfg: LlamaConfig):
        with _strategy_lib.megatron_cpu_init_context(cfg):
            model = cfg.configure_model(tokenizer=self.tokenizer)
        return [model]

    def apply(self) -> Path:
        """Apply the conversion from HF to NeMo format.

        Args:
            output_path: Path where the converted model will be saved

        Returns:
            Path: Path to the saved NeMo model
        """
        from transformers import LlamaForCausalLM

        source = LlamaForCausalLM.from_pretrained(str(self.input_path), torch_dtype="auto")

        with temporary_distributed_context():
            target = self.init_tron_model(self.config)
            self.convert_state(source, target[0])
            state = GlobalState()
            state.cfg = ConfigContainer(
                model_config=self.config,
                train_config=None,
                optimizer_config=OptimizerConfig(use_distributed_optimizer=False),
                ddp_config=None,
                scheduler_config=None,
                dataset_config=None,
                logger_config=LoggerConfig(),
                tokenizer_config=None,
                checkpoint_config=CheckpointConfig(
                    async_save=False, save=str(self.output_path), save_optim=False, ckpt_format="torch_dist"
                ),
                dist_config=None,
                ft_config=None,
                straggler_config=None,
                profiling_config=None,
            )
            save_checkpoint(
                state=state,
                model=target,
                optimizer=None,
                opt_param_scheduler=None,
                num_floating_point_operations_so_far=0,
            )

        print(f"Converted Llama model to Nemo, model saved to {self.output_path} in {source.dtype}.")

        return self.output_path

    def convert_state(self, source, target):
        """Convert state dict from HF format to NeMo format.

        Maps the weights from the HF model to the NeMo model according to
        the appropriate mapping scheme.

        Args:
            source: Source HF model
            target: Target NeMo model

        Returns:
            The result of applying the transforms
        """
        mapping = {
            "model.embed_tokens.weight": "embedding.word_embeddings.weight",
            "model.layers.*.self_attn.o_proj.weight": "decoder.layers.*.self_attention.linear_proj.weight",
            "model.layers.*.mlp.down_proj.weight": "decoder.layers.*.mlp.linear_fc2.weight",
            "model.layers.*.input_layernorm.weight": "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "model.layers.*.post_attention_layernorm.weight": "decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "model.norm.weight": "decoder.final_layernorm.weight",
            "lm_head.weight": "output_layer.weight",
        }
        if getattr(source.config, "tie_word_embeddings", False):
            # llama 3.2 1B and 3B models have no shared input output embeddings
            del mapping["lm_head.weight"]

        transforms = [
            io.state_transform(
                source_key=(
                    "model.layers.*.self_attn.q_proj.weight",
                    "model.layers.*.self_attn.k_proj.weight",
                    "model.layers.*.self_attn.v_proj.weight",
                ),
                target_key="decoder.layers.*.self_attention.linear_qkv.weight",
                fn=TransformFns.merge_qkv,
            ),
            io.state_transform(
                source_key=("model.layers.*.mlp.gate_proj.weight", "model.layers.*.mlp.up_proj.weight"),
                target_key="decoder.layers.*.mlp.linear_fc1.weight",
                fn=TransformFns.merge_fc1,
            ),
        ]
        return io.apply_transforms(source, target, mapping=mapping, transforms=transforms)

    @property
    def tokenizer(self) -> "_HuggingFaceTokenizer":
        """Get the tokenizer for the HF model.

        Returns:
            _HuggingFaceTokenizer: Tokenizer instance initialized from the HF model's tokenizer
        """

        return _HuggingFaceTokenizer(save_hf_tokenizer_assets(str(self.input_path), str(self.output_path)))

    @property
    def config(self) -> LlamaConfig:
        """Create a NeMo LlamaConfig from the HF model config.

        Translates the HF configuration parameters to the equivalent NeMo
        configuration.

        Returns:
            LlamaConfig: NeMo configuration for Llama models
        """
        from transformers import GenerationConfig
        from transformers import LlamaConfig as HFLlamaConfig

        source = HFLlamaConfig.from_pretrained(str(self.input_path))
        generation_config = GenerationConfig.from_pretrained(str(self.input_path))
        print(generation_config)

        def make_vocab_size_divisible_by(vocab_size):
            base = 128
            while vocab_size % base != 0:
                base //= 2
            return base

        if getattr(source, "rope_scaling", None) is not None and source.rope_scaling.get("rope_type") == "llama3":
            # Apply Llama3.1 customize rope scaling
            cls = partial(Llama31Config, scale_factor=source.rope_scaling.get("factor", 8.0))
        else:
            cls = LlamaConfig
        output = cls(
            num_layers=source.num_hidden_layers,
            hidden_size=source.hidden_size,
            ffn_hidden_size=source.intermediate_size,
            num_attention_heads=source.num_attention_heads,
            init_method_std=source.initializer_range,
            layernorm_epsilon=source.rms_norm_eps,
            num_query_groups=source.num_key_value_heads,
            seq_length=source.max_position_embeddings,
            rotary_base=source.rope_theta,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by(source.vocab_size),
            share_embeddings_and_output_weights=getattr(source, "tie_word_embeddings", False),
            fp16=(dtype_from_hf(source) == torch.float16),
            bf16=(dtype_from_hf(source) == torch.bfloat16),
            params_dtype=dtype_from_hf(source),
            generation_config=generation_config,
            vocab_size=self.tokenizer.vocab_size,
        )

        return output
