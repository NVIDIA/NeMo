from abc import ABC, abstractmethod
from typing import Type

import torch

from nemo.collections.common.tokenizers import AggregateTokenizer, CanaryTokenizer, TokenizerSpec

PROMPT_LANGUAGE_KEY = "|PROMPT_LANGUAGE|"


class PromptFormatter(ABC):
    """
    :class:`~nemo.collections.common.prompts.formatter.PromptFormatter` is intended to simplify
    working with various prompt format templates and encoding them into token ID tensors.

    PromptFormatter supports constructing prompts for training (complete context and answers)
    and for inference (context-only).

    The key methods overview:

    * :meth:`PromptFormatter.get_training_prompt_template` and :meth:`PromptFormatter.get_inference_prompt_template` provide a string template of the prompt.

    * :meth:`PromptFormatter.get_training_prompt_slots` and :meth:`PromptFormatter.get_inference_prompt_slots` provide the fillable fields ("slots") available in the prompt.
        TODO: describe the schema of the slot dict.

    * :meth:`PromptFormatter.encode_prompt` converts the provided template and provided slot values to token IDs.
        TODO: describe the schema of the returned dict: context/answer/mask/etc.

    In order to support :class:`~nemo.collections.common.tokenizers.AggregateTokenizer`, provide
    a special slot ``|PROMPT_LANGUAGE|`` with the value corresponding to the name of the sub-tokenizer to be selected.

    Intended usage example for building training prompts::

        >>> fmt = PromptFormatter(tokenizer)
        ... template = fmt.get_training_prompt_template()
        ... slots = fmt.get_training_prompt_slots()
        ... for slot in slots:
        ...     slots[slot] = ...  # user inserted value
        ... encoded_tensors = fmt.encode_prompt(template, slots)

    Intended usage example for building inference prompts::

        >>> fmt = PromptFormatter(tokenizer)
        ... template = fmt.get_inference_prompt_template()
        ... slots = fmt.get_inference_prompt_slots()  # train/infer
        ... for slot in slots:
        ...     slots[slot] = ...  # user inserted value
        ... encoded_tensors = fmt.encode_prompt(template, slots)

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
    #           * 'list[str]' indicates a slot that may be filled one or more times with no separator
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
            # TODO: assertion messages
            assert all(k in turn for k in ("role", "slots"))
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
        missing = set(expected) - set(received)
        # TODO: more detailed info
        assert not missing, f"The following slot values were not provided: {missing}"
