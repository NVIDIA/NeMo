from typing import Type

from nemo.collections.common.prompts.formatter import PromptFormatter


class Llama2PromptFormatter(PromptFormatter):

    REGISTER_NAME = "canary"

    def get_context_template(self) -> str:
        return "<<SYS>>\n|SYSTEM|\n<</SYS>>\n[INST]\nUser:|USER|\n[/INST]\n\nAssistant:"

    def get_answer_template(self) -> str:
        return "|TEXT|"

    def get_context_slots(self) -> dict[str, Type]:
        return {
            "|SYSTEM|": str | None,
            "|USER|": str,
        }

    def get_answer_slots(self) -> dict[str, Type]:
        return {"|TEXT|": str}

    def get_default_context_values(self) -> dict[str, str]:
        return {
            "|SYSTEM|": None,
            "|USER|": "Tell me something about Nvidia NeMo.",
        }
