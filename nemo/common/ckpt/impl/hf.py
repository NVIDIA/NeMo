from dataclasses import dataclass
from typing import Generic, TypeVar
import os
import json
import torch.nn as nn
from nemo_run import Config
from fsspec.spec import AbstractFileSystem
import transformers
from huggingface_hub import HfFileSystem
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_utils import no_init_weights
from fsspec.registry import register_implementation

from nemo.common.ckpt.registry import CheckpointHandler, registry, load_context, detect_checkpoint_type
from nemo.common.plan.plan import Plan

ModelType = TypeVar("ModelType", bound=PreTrainedModel)


register_implementation("hf", HfFileSystem)


@dataclass
class HFPreTrained(Generic[ModelType]):
    """A container class for managing HuggingFace pretrained models and their configurations.
    
    This class provides a structured way to initialize and manage HuggingFace models,
    their configurations, and tokenizers. It supports lazy loading of models and
    various initialization parameters for flexible model deployment.
    
    Args:
        config (PretrainedConfig): The model's configuration object
        tokenizer (PreTrainedTokenizer): The model's tokenizer
        no_init_weights (bool, optional): If True, initializes model without loading weights. 
            Useful for memory efficiency when weights will be loaded later. Defaults to False.
        torch_dtype (str | None, optional): The dtype to use for model parameters 
            (e.g., "float32", "float16", "bfloat16"). Defaults to None.
        device_map (str | dict | None, optional): Specification for model placement across devices.
            Can be "auto", "balanced", "sequential", or a custom mapping dict. Defaults to None.
        trust_remote_code (bool, optional): Whether to allow loading of custom code from model 
            repositories. Use with caution for untrusted sources. Defaults to False.
    
    Example:        
    
    ```python
        config = AutoConfig.from_pretrained("bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        model_container = HFPreTrained(
            config=config,
            tokenizer=tokenizer,
            torch_dtype="float16",
            device_map="auto"
        )
        
        # Model is lazily loaded only when accessed
        model = model_container.model        
    ```
    """
    
    path: str
    config: PretrainedConfig
    tokenizer: PreTrainedTokenizer
    no_init_weights: bool = False
    torch_dtype: str | None = None
    device_map: str | dict | None = None
    trust_remote_code: bool = False
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,  # Force keyword arguments for clarity
        no_init_weights: bool = False,
        torch_dtype: str | None = None,
        device_map: str | dict | None = None,
        trust_remote_code: bool = False,
        config_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
    ) -> "HFPreTrained[ModelType]":
        """Creates an instance from a pretrained model identifier or path.
        
        This method provides a convenient way to initialize the class by automatically
        loading the configuration and tokenizer from HuggingFace Hub or local path.
        
        Args:
            pretrained_model_name_or_path (str): The model identifier (e.g., "bert-base-uncased")
                or path to local model.
            no_init_weights (bool, optional): If True, initializes model without loading weights.
                Defaults to False.
            torch_dtype (str | None, optional): The dtype for model parameters. Defaults to None.
            device_map (str | dict | None, optional): Model placement specification. Defaults to None.
            trust_remote_code (bool, optional): Whether to allow loading custom code. Defaults to False.
            config_kwargs (dict | None, optional): Additional kwargs for AutoConfig.from_pretrained.
                Defaults to None.
            tokenizer_kwargs (dict | None, optional): Additional kwargs for AutoTokenizer.from_pretrained.
                Defaults to None.
        
        Returns:
            HFPreTrained[ModelType]: A new instance with loaded config and tokenizer.
            
        Raises:
            ValueError: If the model identifier is invalid or resources cannot be loaded.
            
        Example:
            ```python
            model_container = HFPreTrained.from_pretrained(
                "bert-base-uncased",
                torch_dtype="float16",
                device_map="auto",
                config_kwargs={"use_cache": False},
                tokenizer_kwargs={"padding_side": "left"}
            )
            ```
        """
        config_kwargs = config_kwargs or {}
        tokenizer_kwargs = tokenizer_kwargs or {}
        
        try:
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=trust_remote_code,
                **config_kwargs
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=trust_remote_code,
                **tokenizer_kwargs
            )
            
            return cls(
                config=config,
                tokenizer=tokenizer,
                no_init_weights=no_init_weights,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
            )
        except Exception as e:
            raise ValueError(
                f"Failed to load model resources from {pretrained_model_name_or_path}: {str(e)}"
            ) from e

    def configure_model(self):
        # If no type parameter was provided, use AutoModelForCausalLM
        model_cls = AutoModelForCausalLM
        # if hasattr(self.__class__, "__orig_bases__"):
        #     type_var = self.__class__.__orig_bases__[0].__args__[0]  # type: ignore
        #     if type_var != "~ModelType":
        #         model_cls = type_var
        if self.no_init_weights:
            with no_init_weights(True):
                self._model = model_cls.from_config(self.config)
        else:
            self._model = model_cls.from_config(self.config)
    
    @property
    def model(self) -> ModelType:
        if not hasattr(self, "_model"):
            self.configure_model()
        return self._model

    # @property
    def model_path(self) -> str | list[str]:
        """Returns the model path(s) that can be used to import this model.
        
        For HuggingFace models, this returns the architectures from the config.
        
        Returns:
            str | list[str]: Model path(s) that can be used to import this model
            
        Raises:
            ValueError: If no architectures are found in the config
        """
        architectures = getattr(self.config, 'architectures', [])
        if not architectures:
            raise ValueError(f"No architectures found in HuggingFace config: {self.config}")
        return architectures[0] if len(architectures) == 1 else architectures


@registry.register("huggingface")
class HuggingFaceHandler(CheckpointHandler):
    def detect(self, fs: AbstractFileSystem, path: str, files: list[str]) -> bool:
        if "config.json" not in files:
            return False

        # Check if it has any of the expected weight files
        weight_files = [
            "pytorch_model.bin",
            "model.safetensors",
            "model.safetensors.index.json",
        ]
        has_weights = any(wf in files for wf in weight_files)
        
        if not has_weights:
            return False
            
        # Verify it's a HF config by checking for architectures key
        try:
            config_content = fs.read_text(os.path.join(path, "config.json"))
            config = json.loads(config_content)
            return "architectures" in config
        except Exception:
            return False

    def load_context(self, path: str) -> Config[HFPreTrained]:
        model_path = path.replace("hf://", "")
        return Config(
            HFPreTrained,
            path=model_path,
            config=Config(AutoConfig.from_pretrained, model_path),
            tokenizer=Config(AutoTokenizer.from_pretrained, model_path),
        )

    def load_model(self, path: str, **kwargs) -> nn.Module:
        return LoadHuggingFaceModel(path, **kwargs)
    

class LoadHuggingFaceModel(Plan):
    def __init__(self, path: str, **kwargs):
        self.path = path
        self.context = load_context(path, build=True)
        self.kwargs = kwargs
    
    def execute(self) -> nn.Module:
        return getattr(transformers, self.context.model_path()).from_pretrained(self.path, **self.kwargs)

    def extra_repr(self) -> str:
        _model_path = self.context.model_path()
        out = f"{_model_path}.from_pretrained({self.path}"
        if self.kwargs:
            for k, v in self.kwargs.items():
                out += f", {k}={v}"
        out += ")"
        return out
    

def save_hf_tokenizer_assets(tokenizer_name_or_path: str, save_path="/tmp/nemo_tokenizer"):
    """Save HF tokenizer to the imported NeMo model"""

    tok = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    # Save tokenizer assets to save_path.
    tok.save_pretrained(save_path)
    
    return save_path


if __name__ == "__main__":
    print(detect_checkpoint_type("meta-llama/Llama-3.1-8B"))
    cfg = load_context("hf://meta-llama/Llama-3.1-8B")
    print(cfg)
