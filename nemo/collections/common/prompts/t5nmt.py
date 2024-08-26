from nemo.collections.common.prompts.formatter import Modality, PromptFormatter


class T5NMTPromptFormatter(PromptFormatter):
    """
    The default prompt format for Megatron T5 based neural machine translation models.
    """

    NAME = "t5nmt"
    OUTPUT_ROLE = "assistant"
    TEMPLATE = {
        "user": {
            "template": f"Q: |message|\n\n",
            "slots": {
                "message": Modality.Text,
            },
        },
        OUTPUT_ROLE: {
            "template": f"A: |message|",
            "slots": {
                "message": Modality.Text,
            },
        },
    }
