# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from cosmos1.models.guardrail.aegis.categories import UNSAFE_CATEGORIES
from cosmos1.models.guardrail.common.core import ContentSafetyGuardrail, GuardrailRunner
from cosmos1.utils import log, misc

SAFE = misc.Color.green("SAFE")
UNSAFE = misc.Color.red("UNSAFE")

DEFAULT_CHECKPOINT_DIR = "checkpoints/Cosmos-1.0-Guardrail/aegis"


class Aegis(ContentSafetyGuardrail):
    def __init__(
        self,
        checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.dtype = torch.bfloat16
        base_model_id = "meta-llama/LlamaGuard-7b"
        aegis_adapter = "nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0"
        base_model = AutoModelForCausalLM.from_pretrained(base_model_id, cache_dir=self.checkpoint_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, cache_dir=self.checkpoint_dir)
        self.model = PeftModel.from_pretrained(base_model, aegis_adapter, cache_dir=self.checkpoint_dir)
        self.model.to(self.device, dtype=self.dtype).eval()

    def get_moderation_prompt(self, user_prompt: str) -> str:
        """Create the moderation prompt for the Aegis model."""
        full_prompt = f"""[INST] Task: Check if there is unsafe content in 'User' messages in conversations according our safety policy with the below categories.

        <BEGIN UNSAFE CONTENT CATEGORIES>
        {"".join([f"{category['instruction']}" for category in UNSAFE_CATEGORIES])}
        <END UNSAFE CONTENT CATEGORIES>

        <BEGIN CONVERSATION>

        """
        full_prompt += f"User: {user_prompt}\n\n"
        full_prompt += """<END CONVERSATION>

        Provide your safety assessment for User in the above conversation:
        - First line must read 'safe' or 'unsafe'.
        - If unsafe, a second line must include a comma-separated list of violated categories. [/INST]"""

        return full_prompt

    def get_aegis_block_message(self, moderation_output: str) -> str:
        """Extract the blocked category and reason from the Aegis model output."""
        block_msg = "Prompt blocked by Aegis."
        try:
            # Extract blocked category and reason
            if len(moderation_output.split("\n")) >= 2:
                moderation_output_list = moderation_output.split("\n")
                blocked_category = int(moderation_output_list[1].lstrip("O")) - 1
                if 0 <= blocked_category < len(UNSAFE_CATEGORIES):
                    blocked_reason = UNSAFE_CATEGORIES[blocked_category]["blocked_reason"]
                    blocked_category_name = UNSAFE_CATEGORIES[blocked_category]["category"]
                    block_msg = f"{blocked_category_name}: {blocked_reason}"
        except Exception as e:
            log.warning(f"Unable to extract blocked category and reason from Aegis output: {e}")
        return block_msg

    def filter_aegis_output(self, prompt: str) -> tuple[bool, str]:
        """Filter the Aegis model output and return the safety status and message."""
        full_prompt = self.get_moderation_prompt(prompt)
        inputs = self.tokenizer([full_prompt], add_special_tokens=False, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=100, pad_token_id=self.tokenizer.eos_token_id)
        prompt_len = inputs["input_ids"].shape[-1]
        moderation_output = self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

        if "unsafe" in moderation_output.lower():
            block_msg = self.get_aegis_block_message(moderation_output)
            return False, block_msg
        else:
            return True, ""

    def is_safe(self, prompt: str) -> tuple[bool, str]:
        """Check if the input prompt is safe according to the Aegis model."""
        try:
            return self.filter_aegis_output(prompt)
        except Exception as e:
            log.error(f"Unexpected error occurred when running Aegis guardrail: {e}")
            return True, "Unexpected error occurred when running Aegis guardrail."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Path to the Aegis checkpoint folder",
        default=DEFAULT_CHECKPOINT_DIR,
    )
    return parser.parse_args()


def main(args):
    aegis = Aegis(checkpoint_dir=args.checkpoint_dir)
    runner = GuardrailRunner(safety_models=[aegis])
    with misc.timer("aegis safety check"):
        safety, message = runner.run_safety_check(args.prompt)
    log.info(f"Input is: {'SAFE' if safety else 'UNSAFE'}")
    log.info(f"Message: {message}") if not safety else None


if __name__ == "__main__":
    args = parse_args()
    main(args)
