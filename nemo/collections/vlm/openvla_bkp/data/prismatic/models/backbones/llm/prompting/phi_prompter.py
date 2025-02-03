"""
phi_prompter.py

Defines a PromptBuilder for building Phi-2 Input/Output Prompts --> recommended pattern used by HF / Microsoft.
Also handles Phi special case BOS token additions.

Reference: https://huggingface.co/microsoft/phi-2#qa-format
"""

from typing import Optional

from nemo.collections.vlm.openvla_bkp.data.prismatic.models.backbones.llm.prompting.base_prompter import PromptBuilder


class PhiPromptBuilder(PromptBuilder):
    def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
        super().__init__(model_family, system_prompt)

        # Note =>> Phi Tokenizer is an instance of `CodeGenTokenizer(Fast)`
        #      =>> By default, does *not* append <BOS> / <EOS> tokens --> we handle that here (IMPORTANT)!
        self.bos, self.eos = "<|endoftext|>", "<|endoftext|>"

        # Get role-specific "wrap" functions
        #   =>> Note that placement of <bos>/<eos> were based on experiments generating from Phi-2 in Input/Output mode
        self.wrap_human = lambda msg: f"Input: {msg}\nOutput: "
        self.wrap_gpt = lambda msg: f"{msg if msg != '' else ' '}\n{self.eos}"

        # === `self.prompt` gets built up over multiple turns ===
        self.prompt, self.turn_count = "", 0

    def add_turn(self, role: str, message: str) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")
        message = message.replace("<image>", "").strip()

        # Special Handling for "first" input --> prepend a <BOS> token (expected by Prismatic)
        if self.turn_count == 0:
            bos_human_message = f"{self.bos}{self.wrap_human(message)}"
            wrapped_message = bos_human_message
        elif (self.turn_count % 2) == 0:
            human_message = self.wrap_human(message)
            wrapped_message = human_message
        else:
            gpt_message = self.wrap_gpt(message)
            wrapped_message = gpt_message

        # Update Prompt
        self.prompt += wrapped_message

        # Bump Turn Counter
        self.turn_count += 1

        # Return "wrapped_message" (effective string added to context)
        return wrapped_message

    def get_potential_prompt(self, message: str) -> None:
        # Assumes that it's always the user's (human's) turn!
        prompt_copy = str(self.prompt)

        human_message = self.wrap_human(message)
        prompt_copy += human_message

        return prompt_copy.rstrip()

    def get_prompt(self) -> str:
        return self.prompt.rstrip()
