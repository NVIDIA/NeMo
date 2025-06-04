import argparse
import os

import nemo_run as run

from nemo.collections import llm
from nemo.collections.llm.api import export_ckpt
from nemo.collections.llm.gpt.data.chat import ChatDataModule
from nemo.collections.llm.modelopt.quantization.quant_cfg_choices import get_quant_cfg_choices
from nemo.collections.llm.modelopt.recipes.distillation_recipe import distillation_recipe
from nemo.collections.llm.modelopt.recipes.ptq_recipe import ptq_recipe
from nemo.collections.llm.recipes.llama32_1b import finetune_recipe
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

DEFAULT_CHAT_TEMPLATE = """{{- bos_token }}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = 'You are a helpful assistant'|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = '' %}
{%- endif %}
{{- '<|start_header_id|>system<|end_header_id|>\n\n' }}
{{- system_message }}
{{- '<|eot_id|>' }}
{%- for message in messages %}
    {%- if message.role == 'assistant' %}
        {%- generation %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>'}}
        {%- endgeneration %}
    {%- else %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
"""


def get_parser():
    parser = argparse.ArgumentParser(
        description="NeMo2.0 QAT/QAD simplified flow. Currently supports models that fit on 1 node and 8 GPUs."
    )
    QUANT_CFG_CHOICES_LIST = ["no_quant", *get_quant_cfg_choices()]

    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to the NeMo 2.0 checkpoint",
        required=True,
    )
    parser.add_argument(
        "--finetune-recipe",
        type=str,
        default="llama3_8b",
        help="Choose NeMo 2.0 recipe. Recipes are named in the format of <model_name>_<model_size>(_<long_sequence_length> or other special settings)",
    )
    parser.add_argument(
        "--distill",
        action="store_true",
        help="Whether to do quantization aware distillation (QAD)",
    )
    parser.add_argument(
        "--hf-tokenizer",
        type=str,
        help="Name of HF model to use for tokenizer.",
        required=False,
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        help="Path to the custom chat template to replace the HF tokenizer default chat template.",
        required=False,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to the finetuning chat dataset. Can be either ShareGPT or HuggingFace/OpenAI chat format",
        required=True,
    )
    parser.add_argument(
        "-algo",
        "--algorithm",
        type=str,
        default="fp8",
        choices=QUANT_CFG_CHOICES_LIST,
        help="TensorRT-Model-Optimizer quantization algorithm",
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="Run on slurm using run.SlurmExecutor",
        default=False,
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="Experiment name",
        default="qat_flow",
    )
    return parser


def get_finetune_recipe(recipe):
    assert hasattr(
        llm, recipe
    ), f"Recipe named {recipe} not found. General format is <model_name>_<model_size>(_<long_sequence_length> or other special settings)"
    finetune_recipe = getattr(llm, recipe).finetune_recipe
    return finetune_recipe()


if __name__ == "__main__":
    args = get_parser().parse_args()
    if not args.distill and not args.finetune_recipe:
        raise ValueError("If distillation is not used, --finetune-recipe must be specified")
    model_name = args.finetune_recipe
    if not model_name:
        model_name = os.path.basename(args.model_path)
    # 1. PTQ
    ptq_model_out = f"{model_name}-{args.algorithm}"

    ptq = run.Script(
        "scripts/llm/ptq.py",
        args=["-nc", args.model_path, "-out", ptq_model_out, "--export_format", "nemo", "-ctp", "8"],
        entrypoint="python",
    )

    # 2. Train
    if not args.chat_template:
        args.chat_template = DEFAULT_CHAT_TEMPLATE
    if not args.hf_tokenizer:
        tokenizer_path = os.path.join(args.model_path, "context/nemo_tokenizer")
        tokenizer = run.Config(
            get_nmt_tokenizer, library='huggingface', model_name=tokenizer_path, chat_template=args.chat_template
        )
    else:
        tokenizer = run.Config(
            get_nmt_tokenizer, library='huggingface', model_name=args.hf_tokenizer, chat_template=args.chat_template
        )
    data = run.Config(
        ChatDataModule,
        dataset_root=args.data_path,
        seq_length=4096,
        tokenizer=tokenizer,
        global_batch_size=512,
        micro_batch_size=4,
        use_hf_tokenizer_chat_template=True,
    )
    train = None
    if args.distill:
        train = distillation_recipe(ptq_model_out, args.model_path)
    else:
        train = get_finetune_recipe(args.finetune_recipe)
        train.resume.restore_config.path = ptq_model_out
    train.tokenizer = "data"
    train.data = data

    # 3. Export
    # TODO figure out path of trained checkpoint

    executor = run.LocalExecutor(ntasks_per_node=8, launcher="torchrun")
    with run.Experiment(args.experiment, executor=executor, log_level="INFO") as exp:
        s1 = exp.add(ptq, tail_logs=True, name="ptq")
        s2 = exp.add(train, tail_logs=True, name="train", dependencies=[s1])
        exp.run(detach=False)
