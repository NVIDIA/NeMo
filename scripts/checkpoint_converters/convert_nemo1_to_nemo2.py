import os
import shutil
import tempfile
from argparse import ArgumentParser
from pathlib import Path

from megatron.core.dist_checkpointing.dict_utils import dict_list_map_inplace
from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject, ShardedObject
from omegaconf import OmegaConf
from transformers import AutoTokenizer as HFAutoTokenizer

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.lightning.ckpt_utils import ckpt_to_context_subdir, ckpt_to_weights_subdir
from nemo.lightning.io.pl import TrainerContext
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero

"""
Script to convert NeMo 1.0 checkpoints to NeMo 2.0 format. 
Example usage:

a. Convert a .nemo checkpoint in tp1
    python /opt/NeMo/scripts/checkpoint_converters/convert_nemo1_to_nemo2.py \
        --input_path=Meta-Llama-3-8B.nemo \
        --output_path=your_output_dir \
        --model_id=meta-llama/Meta-Llama-3-8B

b. Convert a .nemo checkpoint in tp4
    torchrun --nproc_per_node=4 /opt/NeMo/scripts/checkpoint_converters/convert_nemo1_to_nemo2.py \
        --input_path=Mixtral-8x7B.nemo \
        --output_path=your_output_dir \
        --model_id=mistralai/Mixtral-8x7B-v0.1 \
        --tp_size=4

c. Convert a model weight directory. The checkpoint should be similar to `model_weights` subdir after extracting the .nemo file.
   Please also provide tokenizer_library and tokenizer_path when loading from weight directory.
    python /opt/NeMo/scripts/checkpoint_converters/convert_nemo1_nemo2.py \
        --input_path=nemotron3-8b-extracted/model_weights \
        --tokenizer_path=path_to_your_tokenizer_model.model \
        --tokenizer_library=sentencepiece \
        --output_path=your_output_dir \
        --model_id=nvidia/nemotron-3-8b-base-4k

"""

def get_args():
    parser = ArgumentParser(description="Script to convert NeMo 1.0 checkpoints to NeMo 2.0 format.")
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        required=True,
        help="Path to NeMo 1.0 checkpoints. Could be .nemo file, or `model_weights` directory after untar the .nemo. Please also provide tokenizer_library and tokenizer_path if you pass in `model_weights` directory.",
    )
    parser.add_argument("--output_path", type=str, default=None, required=True, help="Path to output NeMo 2.0 directory.")
    parser.add_argument("--model_id", type=str, default=None, required=True, help="Hugging Face model id for the model")
    parser.add_argument("--tokenizer_path", type=str, default=None, required=False, help="Path to tokenizer. If not provided, will 1. try instantiate from nemo1 config 2. pull AutoTokenizer from Hugging Face according to model_id if 1 fails")
    parser.add_argument("--tokenizer_library", type=str, default=None, required=False, help="Tokenizer library, e.g. `sentencepiece`, `megatron`. Defaults to `sentencepiece`")
    parser.add_argument("--tp_size", type=int, default=1, required=False, help="TP size for loading the base model, increase if OOM")
    args = parser.parse_args()
    return args


def get_nemo2_model(model_id, tokenizer) -> llm.GPTModel:
    model_config_mapping = {
        "meta-llama/Meta-Llama-3-8B": (llm.LlamaModel , llm.Llama3Config8B),
        "mistralai/Mixtral-8x7B-v0.1": (llm.MixtralModel, llm.MixtralConfig8x7B),
        "nvidia/nemotron-3-8b-base-4k": (llm.NemotronModel, llm.Nemotron3Config8B)
    }
    if model_id not in model_config_mapping:
        raise ValueError(f"Unsupported model_id: '{model_id}'. Please provide a valid model_id from {list(model_config_mapping.keys())}.")
    model_cls, config_cls = model_config_mapping[model_id]
    return model_cls(config_cls(), tokenizer=tokenizer)


def get_tokenizer(input_path: Path, tokenizer_tmp_dir: Path) -> AutoTokenizer:
    if not input_path.is_dir(): #if .nemo tar
        with tempfile.TemporaryDirectory() as tmp_dir: #we want to clean up this tmp dir
            NLPSaveRestoreConnector._unpack_nemo_file(input_path, tmp_dir)
            cfg = OmegaConf.load(f"{tmp_dir}/model_config.yaml")
            tokenizer_lib = cfg.tokenizer.library
            tokenizer_model = cfg.tokenizer.get("model") and cfg.tokenizer.get("model").split("nemo:", 1)[-1]
            if is_global_rank_zero():
                if tokenizer_model:
                    shutil.copy(f"{tmp_dir}/{tokenizer_model}", f"{tokenizer_tmp_dir}/{tokenizer_model}")
                elif cfg.tokenizer.library=="huggingface":
                    HFAutoTokenizer.from_pretrained(cfg.tokenizer.type).save_pretrained(tokenizer_tmp_dir)
            tokenizer_model = f"{tokenizer_tmp_dir}/{tokenizer_model}" if tokenizer_model else None
    else: 
        if args.tokenizer_path: #not .nemo file, only weight dir need to specify tokenizer lib and path
            tokenizer_lib = args.tokenizer_library or "sentencepiece"
            if args.tokenizer_library is None:
                logging.warning("You specified tokenizer_path but did not provide tokenizer_library, will default to sentencepiece")
            tokenizer_model=args.tokenizer_path
        else: #no .nemo config, no tokenizer path specified, grab from HF, reload
            tokenizer_lib = "huggingface"
            HFAutoTokenizer.from_pretrained(args.model_id).save_pretrained(tokenizer_tmp_dir)

    if tokenizer_lib=="huggingface":
        return AutoTokenizer(tokenizer_tmp_dir)
    else: #not directly use huggingface tokenizer in get_nmt_tokenizer since it will pull from HF and no reload
        return get_nmt_tokenizer(library=tokenizer_lib, tokenizer_model=tokenizer_model)


def main() -> None:
    tokenizer_tmp_dir = "/tmp/nemo_tokenizer"
    tokenizer = get_tokenizer(Path(args.input_path), Path(tokenizer_tmp_dir))
    model = get_nemo2_model(args.model_id, tokenizer=tokenizer)

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp_size,
        setup_optimizers=False,
        init_model_parallel=False
    )

    trainer = nl.Trainer(
        devices=args.tp_size,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
    )
    trainer.strategy.connect(model)
    trainer.strategy.setup_environment()
    trainer.strategy._setup_optimizers = False
    trainer.strategy._init_model_parallel = False
    trainer.strategy.setup(trainer)
    trainer.model.configure_model()


    logging.info(f"loading checkpoint {args.input_path}")

    sharded_state_dict = {"state_dict": trainer.model.sharded_state_dict()}

    for key in list(sharded_state_dict['state_dict'].keys()):
        new_key = key.replace('module', 'model', 1)
        sharded_state_dict['state_dict'][new_key] = sharded_state_dict['state_dict'].pop(key)
        sharded_state_dict['state_dict'][new_key].key = sharded_state_dict['state_dict'][new_key].key.replace('module', 'model', 1)
    
    def skip_fp8_load(x):
        if isinstance(x, ShardedObject) and 'core_attention' in x.key and '_extra_state' in x.key:
            x = LocalNonpersistentObject(x.data)  # use the FP8 state from initialization, not from ckpt
        return x
    dict_list_map_inplace(skip_fp8_load, sharded_state_dict)
    if not Path(args.input_path).is_dir():
        with tempfile.TemporaryDirectory() as tmp_dir:
            NLPSaveRestoreConnector._unpack_nemo_file(args.input_path, tmp_dir)
            model_weight_dir = f"{tmp_dir}/model_weights"
            model_ckpt = trainer.strategy.checkpoint_io.load_checkpoint(model_weight_dir, sharded_state_dict, None)
    else:
        model_ckpt = trainer.strategy.checkpoint_io.load_checkpoint(args.input_path, sharded_state_dict, None)

    model_ckpt['state_dict'] = {k.replace('model', 'module', 1): v for k, v in model_ckpt['state_dict'].items()}
    trainer.model.module.load_state_dict(model_ckpt['state_dict'])
    trainer.save_checkpoint(ckpt_to_weights_subdir(args.output_path))

    if is_global_rank_zero():
        #Corresponding to Connector: on_import_ckpt
        if hasattr(trainer.model, "__io__") and hasattr(trainer.model.tokenizer, '__io__'):
            trainer.model.__io__.tokenizer = trainer.model.tokenizer.__io__
        TrainerContext.from_trainer(trainer).io_dump(ckpt_to_context_subdir(args.output_path), yaml_attrs=["model"])
    
    #remove tmp dir
    if os.path.isdir(tokenizer_tmp_dir):
        shutil.rmtree(tokenizer_tmp_dir)

if __name__ == '__main__':
    args = get_args()
    main() 