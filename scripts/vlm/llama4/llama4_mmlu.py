# Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Evaluate an MMLU‑style dataset stored as a collection of **.jsonl** files using a
Llama‑4 Scout model that has already been converted to a NeMo checkpoint.

For each JSON line the script expects a structure of the form::

    {
      ...,
      "target": "B",
      "arguments": {
          "gen_args_0": {
              "arg_0": "<prompt>",
              ...
          }
      },
      ...
    }

The value of ``arg_0`` is fed to the LLM; the first capital letter ``A‑D`` found
in the model output is taken as the prediction and compared against ``target``.

The script prints a per‑file accuracy table followed by the overall accuracy.

Example usage
-------------

```bash
python llama4_mmlu_evaluator.py \
  --local_model_path /models/llama4_scout_17b_16e_nemo \
  --tasks_dir /data/mmlu_failures \
  --tp 8 --pp 1 --ep 1
```
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple, TextIO

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

import nemo.lightning as nl
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.vlm.inference.base import _setup_trainer_and_restore_model
from nemo.lightning import io
from nemo.lightning.ckpt_utils import ckpt_to_context_subdir
from nemo.utils.get_rank import get_last_rank

###############
# Model utils #
###############


class _SingleBatchIterator:
    """Iterator that yields exactly one batch then stops."""

    def __init__(self, input_ids: torch.Tensor, position_ids: torch.Tensor):
        self.batch = {
            "tokens": input_ids,
            "position_ids": position_ids,
            "attention_mask": None,
        }
        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.batch


def _llama4_forward_step(data_iterator, model, **_):
    batch = next(data_iterator)
    outputs = model(
        input_ids=batch["tokens"],
        position_ids=batch["position_ids"],
        attention_mask=batch.get("attention_mask"),
    )

    def loss_func(x, **__):  # type: ignore[override]
        return x

    return outputs, loss_func


class Llama4:
    """Thin wrapper around the Megatron/NeMo model used for greedy generation."""

    END_TOKENS = {"<|eom|>", "<|eot|>"}

    def __init__(self, ckpt_path: Path, tp: int, pp: int, ep: int) -> None:
        ########################
        #‑‑ Create the trainer ‑‑#
        ########################
        strategy = nl.MegatronStrategy(
            tensor_model_parallel_size=tp,
            expert_tensor_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            expert_model_parallel_size=ep,
            sequence_parallel=False,
            ckpt_include_optimizer=False,
            ckpt_load_strictness="log_all",
            pipeline_dtype=torch.bfloat16,
        )
        trainer = nl.Trainer(
            devices=min(tp * pp * ep, 8),
            num_nodes=max(tp * pp * ep // 8, 1),
            max_steps=1,  # we only do inference
            accelerator="gpu",
            strategy=strategy,
            plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
            logger=False,
            enable_checkpointing=False,
        )

        #############################
        #‑‑ Load the converted model ‑‑#
        #############################
        ctx_path = ckpt_to_context_subdir(ckpt_path)
        model: io.TrainerContext = io.load_context(path=ctx_path, subpath="model")
        _setup_trainer_and_restore_model(path=ckpt_path, trainer=trainer, model=model)

        self.model = model.module.cuda()
        self.model.eval()

        ########################
        #‑‑ Hugging Face side ‑‑#
        ########################
        model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
        tokenizer_wrapper = AutoTokenizer(model_id)
        self._hf_tokenizer = tokenizer_wrapper.tokenizer  # unwrap fast tokenizer
        self._eos_token_ids = {self._hf_tokenizer.vocab[t] for t in self.END_TOKENS if t in self._hf_tokenizer.vocab}

    # ---------------------------------------------------------
    # Generation (very small greedy loop, 1 token / step, top‑1)
    # ---------------------------------------------------------
    def generate(self, prompt: str, max_new_tokens: int = 2) -> str:
        # Build initial token ids
        inputs = self._hf_tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids: torch.Tensor = inputs["input_ids"].cuda()
        position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)

        generated_ids = input_ids.clone()
        stop = False
        step = 0

        with torch.no_grad():
            while not stop and step < max_new_tokens:
                step += 1
                iterator = _SingleBatchIterator(input_ids, position_ids)

                fwd_bwd_function = get_forward_backward_func()
                output = fwd_bwd_function(
                    forward_step_func=_llama4_forward_step,
                    data_iterator=iterator,
                    model=self.model,
                    num_microbatches=1,
                    forward_only=True,
                    seq_length=input_ids.size(1),
                    micro_batch_size=1,
                    collect_non_loss_data=True,
                )
                output = output[0] if isinstance(output, list) and output else output

                # Gather last‑stage logits to the entire tensor parallel group
                if parallel_state.is_pipeline_last_stage():
                    world_size = parallel_state.get_tensor_model_parallel_world_size()
                    gathered: List[torch.Tensor] = [torch.zeros_like(output) for _ in range(world_size)]
                    dist.all_gather(gathered, output, group=parallel_state.get_tensor_model_parallel_group())
                    output = torch.cat(gathered, dim=-1)
                    next_token_ids = torch.argmax(output[:, -1], dim=-1, keepdim=True)
                else:
                    next_token_ids = torch.zeros((1, 1), dtype=generated_ids.dtype, device=generated_ids.device)

                # Broadcast next token so every rank continues in sync
                dist.broadcast(next_token_ids, get_last_rank())

                generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
                input_ids = generated_ids
                position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)

                if next_token_ids.item() in self._eos_token_ids:
                    stop = True

        return self._hf_tokenizer.decode(generated_ids[0], skip_special_tokens=True)[len(prompt) :].strip()


####################
# Utility functions#
####################


_PRED_RE = re.compile(r"\b([A-D])\b")


def _extract_prediction(text: str) -> str | None:
    """Return first standalone capital letter A‑D in *text*."""

    match = _PRED_RE.search(text)
    return match.group(1) if match else None


def _safe_print(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs, flush=True)


#########
#  Main #
#########


def evaluate(tasks_dir: Path, model: Llama4, sample_ratio: float, log_fh: TextIO | None) -> None:
    per_file_stats: Dict[str, Tuple[int, int]] = {}
    global_correct = global_total = 0

    jsonl_files = sorted(tasks_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No *.jsonl files found under {tasks_dir}")
    for file_path in jsonl_files:
        correct = total = 0
        with file_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()

        sample_size = max(1, int(len(lines) * sample_ratio))
        sampled_lines = lines[:sample_size]  # or random.sample(lines, sample_size) for stochastic

        for line in sampled_lines:
            record = json.loads(line)
            prompt = (
                record.get("arguments", {})
                .get("gen_args_0", {})
                .get("arg_0")
            )
            if prompt is None:
                continue  # malformed line, skip
            target = record.get("target")  # expected to be "A".."D"
            prediction_text = model.generate(prompt)
            pred = _extract_prediction(prediction_text) or "?"
            is_correct = pred == target

            if log_fh is not None and (not dist.is_initialized() or dist.get_rank() == 0):
                json.dump(
                    {
                        "file": file_path.name,
                        "prompt": prompt,
                        "prediction": pred,
                        "target": target,
                        "correct": is_correct,
                    },
                    log_fh,
                )
                log_fh.write("\n")

            if is_correct:
                correct += 1
            total += 1

        per_file_stats[file_path.name] = (correct, total)
        global_correct += correct
        global_total += total
        acc = correct / total if total else 0.0
        _safe_print(f"{file_path.name:80s}  {correct:4d}/{total:<4d}  ({acc:5.2%})")

    overall_acc = global_correct / global_total if global_total else 0.0
    _safe_print("-" * 100)
    _safe_print(f"OVERALL: {global_correct}/{global_total}  ({overall_acc:5.2%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama‑4 MMLU evaluator")
    parser.add_argument("--tasks_dir", type=Path, required=True, help="Directory containing *.jsonl files")
    parser.add_argument("--local_model_path", type=Path, required=True, help="Path to NeMo checkpoint directory")
    parser.add_argument("--sample_ratio", type=float, default=1.0, help="Fraction of examples to evaluate [0‥1]")
    parser.add_argument("--log_file", type=Path, default=None, help="Write per‑example results to this JSONL file")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed for subsampling")
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallelism")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism (MoE)")

    args = parser.parse_args()
    if not 0.0 < args.sample_ratio <= 1.0:
        raise ValueError("sample_ratio must be in (0,1]")
    random.seed(args.seed)

    model = Llama4(args.local_model_path, tp=args.tp, pp=args.pp, ep=args.ep)
    log_handle: TextIO | None = args.log_file.open("w", encoding="utf-8") if args.log_file else None
    try:
        evaluate(args.tasks_dir, model, sample_ratio=args.sample_ratio, log_fh=log_handle)
    finally:
        if log_handle is not None:
            log_handle.close()
