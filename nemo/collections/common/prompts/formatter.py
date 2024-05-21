import typing
from abc import ABC
from typing import Type

import torch

from nemo.collections.common.tokenizers import AggregateTokenizer, TokenizerSpec

PROMPT_LANGUAGE_KEY = "|PROMPT_LANGUAGE|"


class PromptFormatter(ABC):
    """
    :class:`~nemo.collections.common.prompts.formatter.PromptFormatter` is intended to simplify
    working with various prompt format templates and encoding them into token ID tensors.

    It assumes a dialog-like structure, which is a list of turns, with each turn assigned to a role.
    Sub-classes of PromptFormatter define turn templates for each role under TEMPLATE class attribute.
    Each template may define some constant parts (e.g. begin-of-turn or end-of-turn tokens, whitespaces, etc.)
    and variable parts which we call "slots", that will be provided by the user during training or inference.

    A role is typically "user" and "assistant", and some popular models also use a "system" role.
    Other roles may be defined as well. We expected the role corresponding to the model's responses
    will be registered under class attribute called INFERENCE_ROLE.

    A turn is a dict with keys "role" and "slots", where "slots" are a dict that maps slot names
    to values that should be filled in the template.
    For example, a user role template may be ``"Question: |MESSAGE| "`` and corresponding ``slots`` would then be
    ``{"|MESSAGE|": "What time is it?"}``.

    There is a special slot called ``|PROMPT_LANGUAGE|`` that's used to select the sub-tokenizer in
    :class:`~nemo.collections.common.tokenizers.aggregate_tokenizer.AggregateTokenizer`.
    It's only used when the tokenizer is aggregate; otherwise it's discarded.

    PromptFormatter supports constructing prompts for training (complete context and answers)
    and for inference (context-only).
    Training/inference is determined automatically; if the last role in a dialog is the INFERENCE_ROLE,
    that's an 'asked-and-answered' scenario, so we assume it's inteded for training.
    We'll create a dict with tokenized results available under the following keys:

    * ``context_ids`` (all turns minus last one),
    * ``answer_ids`` (last turn)
    * ``input_ids`` (previous two values concatenated)
    * ``mask`` (boolean mask tensor of the same lenth as ``input_ids`` that's set to True on INFERENCE_ROLE turns)

    Typically, the user will use the ``encode_dialog`` method providing a list of turns to it.
    Example showing how to construct model inputs/outputs for training::

        >>> formatter = PromptFormatter(tokenizer)
        ... encoded_for_training = formatter.encode_dialog(
        ...     turns=[
        ...         {"role": "user", "slots": {"|MESSAGE|": "What time is it?"}},
        ...         {"role": "assistant", "slots": {"|MESSAGE|": "Ten o'clock."}},
        ...         {"role": "user", "slots": {"|MESSAGE|": "PM or AM?"}},
        ...         {"role": "assistant", "slots": {"|MESSAGE|": "AM, naturally! It's bright outside"}},
        ...     ]
        ... )

    Another example that shows how to use the same method to generate prompts for inference::


        >>> formatter = PromptFormatter(tokenizer)
        ... encoded_for_training = formatter.encode_dialog(
        ...     turns=[
        ...         {"role": "user", "slots": {"|MESSAGE|": "What time is it?"}},
        ...         {"role": "assistant", "slots": {"|MESSAGE|": "Ten o'clock."}},
        ...         {"role": "user", "slots": {"|MESSAGE|": "PM or AM?"}},
        ...     ]
        ... )

    """

    REGISTER_NAME = None

    # Template is a dict that maps:
    # * from a role name string (system/user/assistant/etc)
    # * to a dict with keys
    #   * "template" that has a string value (the prompt template)
    #   * "slots" that has a value of dict[str, Type]
    #       * keys of slots are the names of formattable slots in the prompt template
    #       * values of slots are types:
    #           * 'str' indicates a required slot
    #           * 'str | None' indicates an optional slot
    # Template is intended to be defined by the child classes.
    TEMPLATE = None

    INFERENCE_ROLE = None

    _REGISTERED_FORMATTERS = {}

    def __init__(self, tokenizer: TokenizerSpec, defaults: list[dict] | None = None) -> None:
        self.tokenizer = tokenizer
        self.defaults = defaults

    def __init_subclass__(cls, **kwargs) -> None:
        if cls.__name__ not in cls._REGISTERED_FORMATTERS:
            for attr in ("REGISTER_NAME", "TEMPLATE", "INFERENCE_ROLE"):
                assert (
                    getattr(cls, attr, None) is not None
                ), f"Programmer's error: PromptFormatter subclass {cls} did not define a class attribute {attr}"
            cls._REGISTERED_FORMATTERS[cls.REGISTER_NAME] = cls
        super().__init_subclass__(**kwargs)

    @classmethod
    def resolve(cls, name: str) -> Type["PromptFormatter"]:
        if name not in cls._REGISTERED_FORMATTERS:
            raise RuntimeError(
                f"Unknown prompt formatter: '{name}' (known formats: {', '.join(cls._REGISTERED_FORMATTERS.keys())})"
            )
        return cls._REGISTERED_FORMATTERS[name]

    def get_roles(self) -> list[str]:
        return list(self.TEMPLATE.keys())

    def get_slots(self, role: str) -> dict[str, Type]:
        # returns a copy to avoid accidential mutation of a global object by the user
        return self.TEMPLATE[role]["slots"].copy()

    def get_template(self, role: str) -> str:
        return self.TEMPLATE[role]["template"]

    def encode_turn(self, prompt_template: str, expected_slots: dict, slot_values: dict) -> list[int]:
        prompt = prompt_template
        for slot in expected_slots:
            assert slot in slot_values, f"Missing required {slot=} in {slot_values=} for {prompt_template=}"
            prompt = prompt.replace(slot, slot_values[slot])
        return self._apply_tokenizer(prompt, lang=slot_values.get(PROMPT_LANGUAGE_KEY))

    def encode_dialog(self, turns: list[dict]) -> dict[str, torch.Tensor]:
        assert len(turns) > 0, "Empty dialog is not supported."
        roles = self.get_roles()

        turn_tokens = []
        turn_token_counts = []
        turn_mask_values = []
        for turn in turns:
            assert all(
                k in turn for k in ("role", "slots")
            ), f"A turn must have keys 'role' and 'slots'. We received {turn=}"
            role = turn["role"]
            assert role in roles, f"Found turn with {role=}, but availables roles are {roles}"
            expected_slots = self.get_slots(role)
            slot_values = turn["slots"]
            self._validate_slot_values(expected_slots, slot_values)
            template = self.get_template(role)
            tokens = self.encode_turn(template, expected_slots, slot_values)
            turn_tokens.extend(tokens)
            turn_token_counts.append(len(tokens))
            turn_mask_values.append(role == self.INFERENCE_ROLE)

        ans = {"input_ids": torch.tensor(turn_tokens, dtype=torch.long)}
        if turn_mask_values[-1]:
            # The last turn comes from INFERENCE_ROLE, i.e. it's a response from the system.
            # This indicates it's a training example for which we provide context/answer/mask.
            ans["context_ids"] = ans["input_ids"][: -turn_token_counts[-1]]
            ans["answer_ids"] = ans["input_ids"][-turn_token_counts[-1] :]
            ans["mask"] = torch.tensor(
                [
                    turn_mask_values[turn_idx]
                    for turn_idx, turn_len in enumerate(turn_token_counts)
                    for _ in range(turn_len)
                ],
                dtype=torch.bool,
            )
        else:
            ans["context_ids"] = ans["input_ids"]  # context == input for inference
        return ans

    def _apply_tokenizer(self, text: str, lang: str | None = None) -> list[int]:
        if isinstance(self.tokenizer, AggregateTokenizer):
            assert lang is not None, (
                f"Missing key '{PROMPT_LANGUAGE_KEY}' in slot_values -- cannot resolve "
                f"the correct sub-tokenizer in the aggregate tokenizer."
            )
            return self.tokenizer.text_to_ids(text, lang)
        return self.tokenizer.text_to_ids(text)

    def _validate_slot_values(self, expected: dict, received: dict) -> None:
        missing = set(k for k, v in expected.items() if not is_optional(k)) - set(received)
        assert not missing, f"The following non-optional slot values were not provided: {missing}"


def is_optional(field: typing.Type) -> bool:
    # Source: https://stackoverflow.com/a/58841311
    return typing.get_origin(field) is typing.Union and type(None) in typing.get_args(field)
