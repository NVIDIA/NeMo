from nemo.collections.common.prompts.formatter import Modality, OneOrMore, PromptFormatter


class AudioExamplePromptFormatter(PromptFormatter):
    """
    An example prompt formatter implementation supporting audio and text,
    either in user or assistant turns.

    It defines a dialog of the form:

        User: Summarize the following recording: <audio>
        Assistant: Sure, in this recording ...
        User: Generate a recording with this summary..
        Assistant: Sure, here you go: <audio>

    Python usage:

        >>> formatter = AudioExamplePromptFormatter(tokenizer)
        ... encoded_for_training = formatter.encode_dialog(
        ...     turns=[
        ...         {"role": "user", "slots": {"message": ["Summarize the following recording:", torch.tensor(...)]}},
        ...         {"role": "assistant", "slots": {"message": "Sure, in this recording ..."}},
        ...         {"role": "user", "slots": {"message": "Generate a recording with this summary."}},
        ...         {"role": "assistant", "slots": {"message": ["Sure, here you go:", torch.tensor(...)]}},
        ...     ]
        ... )

    """

    NAME = "audio_example_prompt_format"
    OUTPUT_ROLE = "assistant"
    TEMPLATE = {
        "user": {
            "template": f"User: |message|\n",
            "slots": {
                "message": (Modality.Text | Modality.Audio).kleene_plus(),
            },
        },
        OUTPUT_ROLE: {
            "template": f"Assistant: |message|\n",
            "slots": {"message": (Modality.Text | Modality.Audio).kleene_plus()},
        },
    }
