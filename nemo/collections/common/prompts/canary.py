from typing import Type

from nemo.collections.common.prompts.formatter import PromptFormatter


# class CanaryPromptFormatter(PromptFormatter):
#     # TODO(pzelasko): seeing this and other examples so far I think
#     #  we can redesign training/inference thing to a context/answer thing
#
#     def get_inference_prompt_template(self) -> str:
#         return "<|startoftranscript|>|SOURCE_LANG||TASKNAME||TARGET_LANG||PNC|"
#
#     def get_training_prompt_template(self) -> str:
#         return f"{self.get_training_prompt_template()}|TEXT|<|endoftext|>"
#
#     def get_inference_prompt_slots(self) -> dict[str, Type]:
#         return {
#             "|SOURCE_LANG|": str,
#             "|TARGET_LANG|": str,
#             "|TASKNAME|": str,
#             "|PNC|": str,
#         }
#
#     def get_training_prompt_slots(self) -> dict[str, Type]:
#         return {**self.get_inference_prompt_slots(), "|TEXT|": str}
#
#     def get_default_prompt_slots(self) -> dict[str, str]:
#         return {
#             "|SOURCE_LANG|": "en",
#             "|TARGET_LANG|": "en",
#             "|TASKNAME|": "<|transcribe|>",
#             "|PNC|": "<|pnc|>",
#         }


class CanaryPromptFormatter(PromptFormatter):

    def get_context_template(self) -> str:
        return "<|startoftranscript|>|SOURCE_LANG||TASKNAME||TARGET_LANG||PNC|"

    def get_answer_template(self) -> str:
        return "|TEXT|<|endoftext|>"

    def get_context_slots(self) -> dict[str, Type]:
        return {
            "|SOURCE_LANG|": str,
            "|TARGET_LANG|": str,
            "|TASKNAME|": str,
            "|PNC|": str,
        }

    def get_answer_slots(self) -> dict[str, Type]:
        return {"|TEXT|": str}

    def get_default_context_values(self) -> dict[str, str]:
        return {
            "|SOURCE_LANG|": "en",
            "|TARGET_LANG|": "en",
            "|TASKNAME|": "<|transcribe|>",
            "|PNC|": "<|pnc|>",
        }
