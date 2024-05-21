from nemo.collections.common.prompts.formatter import PromptFormatter


class Llama2PromptFormatter(PromptFormatter):
    """
    TODO: validate faithfulness of the implemenation
    """

    REGISTER_NAME = "llama2"
    INFERENCE_ROLE = "assistant"
    TEMPLATE = {
        "system": {
            "template": "<<SYS>>\n|SYSTEM|\n<</SYS>>\n",
            "slots": {
                "|SYSTEM|": str,
            },
        },
        "user": {
            "template": "[INST]\nUser:|USER|\n[/INST]\n\n",
            "slots": {
                "|USER|": str,
            },
        },
        INFERENCE_ROLE: {
            "template": f"Assistant:|TEXT|",
            "slots": {
                "|TEXT|": str | None,
            },
        },
    }


LLAMA3_BOS = "<|begin_of_text|>"
LLAMA3_HEADER_BEGIN = "<|start_header_id|>"
LLAMA3_HEADER_END = "<|end_header_id|>"
LLAMA3_END_OF_TURN = "<|eot_id|>"
LLAMA3_NL = "\n\n"


class Llama3PromptFormatter(PromptFormatter):
    """
    Implemented following the code at:
     https://github.com/meta-llama/llama3/blob/main/llama/test_tokenizer.py#L56
    """

    REGISTER_NAME = "llama3"
    INFERENCE_ROLE = "assistant"
    TEMPLATE = {
        "preamble": {
            "template": LLAMA3_BOS,
        },
        "system": {
            "template": f"{LLAMA3_HEADER_BEGIN}system{LLAMA3_HEADER_END}{LLAMA3_NL}|MESSAGE|{LLAMA3_END_OF_TURN}",
            "slots": {
                "|MESSAGE|": str,
            },
        },
        "user": {
            "template": f"{LLAMA3_HEADER_BEGIN}user{LLAMA3_HEADER_END}{LLAMA3_NL}|MESSAGE|{LLAMA3_END_OF_TURN}",
            "slots": {
                "|MESSAGE|": str,
            },
        },
        INFERENCE_ROLE: {
            "template": f"{LLAMA3_HEADER_BEGIN}assistant{LLAMA3_HEADER_END}{LLAMA3_NL}|MESSAGE|{LLAMA3_END_OF_TURN}",
            "slots": {
                "|MESSAGE|": str,
            },
        },
    }
