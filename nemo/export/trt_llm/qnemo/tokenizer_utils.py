import os

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from nemo.export.trt_llm.nemo.sentencepiece_tokenizer import SentencePieceTokenizer

# TODO: use get_nmt_tokenizer helper below to instantiate tokenizer once environment / dependencies get stable
# from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

TOKENIZER_CONFIG_FILE = "tokenizer_config.yaml"


def get_nmt_tokenizer(nemo_checkpoint_path: str):
    """Build tokenizer from Nemo tokenizer config."""

    print(f"Initializing tokenizer from {TOKENIZER_CONFIG_FILE}")
    tokenizer_cfg = OmegaConf.load(os.path.join(nemo_checkpoint_path, TOKENIZER_CONFIG_FILE))

    library = tokenizer_cfg.library
    legacy = tokenizer_cfg.get("sentencepiece_legacy", library == "sentencepiece")

    if library == "huggingface":
        print(f"Getting HuggingFace AutoTokenizer with pretrained_model_name: {tokenizer_cfg.type}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_cfg["type"], use_fast=tokenizer_cfg.get("use_fast", False))
    elif library == "sentencepiece":
        print(f"Getting SentencePieceTokenizer with model: {tokenizer_cfg.model}")
        tokenizer = SentencePieceTokenizer(
            model_path=os.path.join(nemo_checkpoint_path, tokenizer_cfg.model), legacy=legacy
        )
    else:
        raise NotImplementedError("Currently we only support 'huggingface' and 'sentencepiece' tokenizer libraries.")

    return tokenizer
