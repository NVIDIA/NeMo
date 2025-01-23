"""
mistral_instruct_prompter.py

Defines a PromptBuilder for building Mistral Instruct Chat Prompts --> recommended pattern used by HF / online tutorial.s

Reference: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1#instruction-format
"""

from typing import Optional

from nemo.collections.vlm.openvla.data.prismatic.models.backbones.llm.prompting.base_prompter import PromptBuilder


class MistralInstructPromptBuilder(PromptBuilder):
    def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
        super().__init__(model_family, system_prompt)

        # Note =>> Mistral Tokenizer is an instance of `LlamaTokenizer(Fast)`
        #      =>> Mistral Instruct *does not* use a System Prompt
        self.bos, self.eos = "<s>", "</s>"

        # Get role-specific "wrap" functions
        self.wrap_human = lambda msg: f"[INST] {msg} [/INST] "
        self.wrap_gpt = lambda msg: f"{msg if msg != '' else ' '}{self.eos}"

        # === `self.prompt` gets built up over multiple turns ===
        self.prompt, self.turn_count = "", 0

    def add_turn(self, role: str, message: str) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")
        message = message.replace("<image>", "").strip()

        if (self.turn_count % 2) == 0:
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

        return prompt_copy.removeprefix(self.bos).rstrip()

    def get_prompt(self) -> str:
        # Remove prefix <bos> because it gets auto-inserted by tokenizer!
        return self.prompt.removeprefix(self.bos).rstrip()
