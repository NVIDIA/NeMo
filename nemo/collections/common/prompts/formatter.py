from abc import ABC, abstractmethod
from typing import Type

import torch

from nemo.collections.common.tokenizers import AggregateTokenizer, TokenizerSpec

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

    _REGISTERED_FORMATTERS = {}

    def __init_subclass__(cls, **kwargs) -> None:
        if cls.__name__ not in cls._REGISTERED_FORMATTERS:
            assert hasattr(
                cls, "REGISTER_NAME"
            ), f"Programmer's error: PromptFormatter subclass {cls} did not define a class attribute NAME"
            cls._REGISTERED_FORMATTERS[cls.REGISTER_NAME] = cls
        super().__init_subclass__(**kwargs)

    @classmethod
    def resolve(cls, name: str) -> Type["PromptFormatter"]:
        if name not in cls._REGISTERED_FORMATTERS:
            raise RuntimeError(
                f"Unknown prompt formatter: '{name}' (known formats: {', '.join(cls._REGISTERED_FORMATTERS.keys())})"
            )
        return cls._REGISTERED_FORMATTERS[name]

    def __init__(self, tokenizer: TokenizerSpec) -> None:
        self.tokenizer = tokenizer

    @abstractmethod
    def get_context_template(self) -> str:
        """Returns the prompt template to be filled with slot values."""
        # return "<sos> {taskname} {source_lang} ... <blabla> <lang>"
        raise NotImplementedError

    @abstractmethod
    def get_context_slots(self) -> dict[str, Type]:
        # return {"taskname": str, "source_lang": str | None, "target_lang": str | list[str], ...}
        raise NotImplementedError

    @abstractmethod
    def get_answer_template(self) -> str:
        """Returns the prompt template to be filled with slot values."""
        # return "<sos> {taskname} {source_lang} ... <blabla> <lang>"
        raise NotImplementedError

    @abstractmethod
    def get_answer_slots(self) -> dict[str, Type]:
        # return {"taskname": str, "source_lang": str | None, "target_lang": str | list[str], ...}
        raise NotImplementedError

    @abstractmethod
    def get_default_context_values(self) -> dict[str, str]:
        # return {"taskname": "asr", "source_lang": "en", "target_lang": ["en", "fr"], ...}
        raise NotImplementedError

    def encode(self, prompt_template: str, expected_slots: dict, slot_values: dict) -> list[int]:
        prompt = prompt_template
        for slot in expected_slots:
            assert slot in slot_values, f"Missing required {slot=} in {slot_values=} for {prompt_template=}"
            prompt = prompt.replace(slot, slot_values[slot])
        return self._apply_tokenizer(prompt, lang=slot_values.get(PROMPT_LANGUAGE_KEY))

    def encode_for_training(self, slot_values: dict[str, str]) -> dict[str, torch.Tensor]:
        ans = self.encode_for_inference(slot_values)
        slots = self.get_answer_slots()
        self._validate_slot_values(expected=slots, received=slot_values)
        ans["answer_ids"] = torch.tensor(self.encode(self.get_answer_template(), slots, slot_values))
        ans["input_ids"] = torch.cat([ans["context_ids"], ans["answer_ids"]], dim=0)
        ans["mask"] = torch.zeros_like(ans["input_ids"], dtype=torch.bool)
        ans["mask"][ans["context_ids"].shape[0] :] = True
        return ans

    def encode_for_inference(self, slot_values: dict[str, str]) -> dict[str, torch.Tensor]:
        slots = self.get_context_slots()
        self._validate_slot_values(expected=slots, received=slot_values)
        return {"context_ids": torch.tensor(self.encode(self.get_context_template(), slots, slot_values))}

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
        assert not missing, f"The following slot values were not provided: {missing}"


# class PromptFormatter(ABC):
#     """
#     :class:`~nemo.collections.common.prompts.formatter.PromptFormatter` is intended to simplify
#     working with various prompt format templates and encoding them into token ID tensors.
#
#     PromptFormatter supports constructing prompts for training (complete context and answers)
#     and for inference (context-only).
#
#     The key methods overview:
#
#     * :meth:`PromptFormatter.get_training_prompt_template` and :meth:`PromptFormatter.get_inference_prompt_template` provide a string template of the prompt.
#
#     * :meth:`PromptFormatter.get_training_prompt_slots` and :meth:`PromptFormatter.get_inference_prompt_slots` provide the fillable fields ("slots") available in the prompt.
#         TODO: describe the schema of the slot dict.
#
#     * :meth:`PromptFormatter.encode_prompt` converts the provided template and provided slot values to token IDs.
#         TODO: describe the schema of the returned dict: context/answer/mask/etc.
#
#     Intended usage example for building training prompts::
#
#         >>> fmt = PromptFormatter(tokenizer)
#         ... template = fmt.get_training_prompt_template()
#         ... slots = fmt.get_training_prompt_slots()
#         ... for slot in slots:
#         ...     slots[slot] = ...  # user inserted value
#         ... encoded_tensors = fmt.encode_prompt(template, slots)
#
#     Intended usage example for building inference prompts::
#
#         >>> fmt = PromptFormatter(tokenizer)
#         ... template = fmt.get_inference_prompt_template()
#         ... slots = fmt.get_inference_prompt_slots()  # train/infer
#         ... for slot in slots:
#         ...     slots[slot] = ...  # user inserted value
#         ... encoded_tensors = fmt.encode_prompt(template, slots)
#
#     """
#
#     _REGISTERED_FORMATTERS = {}
#
#     def __init_subclass__(cls, **kwargs) -> None:
#         if cls.__name__ not in cls._REGISTERED_FORMATTERS:
#             cls._REGISTERED_FORMATTERS[cls.__name__] = cls
#         super().__init_subclass__(**kwargs)
#
#     @classmethod
#     def resolve(cls, name: str) -> Type["PromptFormatter"]:
#         if name not in cls._REGISTERED_FORMATTERS:
#             raise RuntimeError(f"Unknown prompt formatter: '{name}'")
#         return cls._REGISTERED_FORMATTERS[name]
#
#     def __init__(self, tokenizer: TokenizerSpec) -> None:
#         self.tokenizer = tokenizer
#
#     @abstractmethod
#     def get_training_prompt_template(self) -> str:
#         """Returns the prompt template to be filled with slot values."""
#         # return "<sos> {taskname} {source_lang} ... <blabla> <lang>"
#         raise NotImplementedError
#
#     @abstractmethod
#     def get_training_prompt_slots(self) -> dict[str, Type]:
#         # return {"taskname": str, "source_lang": str | None, "target_lang": str | list[str], ...}
#         raise NotImplementedError
#
#     @abstractmethod
#     def get_inference_prompt_slots(self) -> dict[str, Type]:
#         # return {"taskname": str, "source_lang": str | None, "target_lang": str | list[str], ...}
#         raise NotImplementedError
#
#     @abstractmethod
#     def get_inference_prompt_template(self) -> str:
#         """Returns the prompt template to be filled with slot values."""
#         # return "<sos> {taskname} {source_lang} ... <blabla> <lang>"
#         raise NotImplementedError
#
#     @abstractmethod
#     def get_default_prompt_slots(self) -> dict[str, str]:
#         # return {"taskname": "asr", "source_lang": "en", "target_lang": ["en", "fr"], ...}
#         raise NotImplementedError
#
#     def encode_prompt(self, prompt: str, slots: dict) -> list[int]:
#         for slot, value in slots.items():
#             prompt = prompt.replace(slot, value)
#         # TODO: the API needs to return sth like dict below to actually be useful:
#         # {"input_ids": tensor, "context": tensor, "answer": tensor, "mask": tensor}
#         return self.tokenizer.text_to_ids(prompt)
#
#     def _apply_tokenizer(self, text: str, lang: str | None = None) -> list[int]:
#         if isinstance(self.tokenizer, AggregateTokenizer):
#             assert lang is not None, (
#                 f"Missing key '{PROMPT_LANGUAGE_KEY}' in slots -- cannot resolve "
#                 f"the correct sub-tokenizer in the aggregate tokenizer."
#             )
#             return self.tokenizer.text_to_ids(text, lang)
#         return self.tokenizer.text_to_ids(text)
