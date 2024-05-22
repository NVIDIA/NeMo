import pytest

from nemo.collections.common.prompts.canary import PromptFormatter


class _DummyPromptFormatter(PromptFormatter):
    REGISTER_NAME = "_dummy_test_formatter"
    TEMPLATE = {
        "user": {"template": "<s>|TEXT|</s>", "slots": {"|TEXT|": str}},
        "assistant": {"template": "|TEXT|</s>", "slots": {"|TEXT|": str}},
    }
    INFERENCE_ROLE = "assistant"


def test_prompt_formatter_empty_dialog_exception(bpe_tokenizer):
    formatter = _DummyPromptFormatter(bpe_tokenizer)
    with pytest.raises(AssertionError):
        formatter.encode_dialog([])


def test_prompt_formatter_inference(bpe_tokenizer):
    formatter = _DummyPromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog([{"role": "user", "slots": {"|TEXT|": "hi"}}])
    recovered = bpe_tokenizer.ids_to_text(ans["input_ids"])
    assert recovered == "<s>hi</s>"


def test_prompt_formatter_training(bpe_tokenizer):
    formatter = _DummyPromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"|TEXT|": "hi"}},
            {"role": "assistant", "slots": {"|TEXT|": "hello"}},
        ]
    )
    recovered = bpe_tokenizer.ids_to_text(ans["input_ids"])
    assert recovered == "<s>hi</s> hello</s>", recovered


def test_prompt_formatter_missing_role(bpe_tokenizer):
    formatter = _DummyPromptFormatter(bpe_tokenizer)
    with pytest.raises(AssertionError, match="A turn must have have a 'role' key"):
        ans = formatter.encode_dialog([{"slots": {"|TEXT|": "hi"}}])


def test_prompt_formatter_missing_slots(bpe_tokenizer):
    formatter = _DummyPromptFormatter(bpe_tokenizer)
    with pytest.raises(
        AssertionError, match="A turn for role user must have have a non-empty value under 'slots' key"
    ):
        formatter.encode_dialog([{"role": "user"}])
    with pytest.raises(
        AssertionError, match="A turn for role user must have have a non-empty value under 'slots' key"
    ):
        formatter.encode_dialog([{"role": "user", "slots": {}}])


class _DummyPreamblePromptFormatter(PromptFormatter):
    REGISTER_NAME = "_dummy_test_formatter"
    TEMPLATE = {
        "preamble": {"template": "TEST"},
        "user": {"template": "<s>|TEXT|</s>", "slots": {"|TEXT|": str}},
        "assistant": {"template": "|TEXT|</s>", "slots": {"|TEXT|": str}},
    }
    INFERENCE_ROLE = "assistant"


def test_prompt_formatter_preamble_inference(bpe_tokenizer):
    formatter = _DummyPreamblePromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog([{"role": "user", "slots": {"|TEXT|": "hi"}}])
    recovered = bpe_tokenizer.ids_to_text(ans["input_ids"])
    assert recovered == "TEST <s>hi</s>", recovered


def test_prompt_formatter_premble_training(bpe_tokenizer):
    formatter = _DummyPreamblePromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"|TEXT|": "hi"}},
            {"role": "assistant", "slots": {"|TEXT|": "hello"}},
        ]
    )
    recovered = bpe_tokenizer.ids_to_text(ans["input_ids"])
    assert recovered == "TEST <s>hi</s> hello</s>"


def test_prompt_formatter_explicit_preamble(bpe_tokenizer):
    formatter = _DummyPreamblePromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog([{"role": "preamble"}, {"role": "user", "slots": {"|TEXT|": "hi"}}])
    recovered = bpe_tokenizer.ids_to_text(ans["input_ids"])
    assert recovered == "TEST <s>hi</s>"


def test_prompt_formatter_wrong_preamble_excpetions(bpe_tokenizer):
    formatter = _DummyPreamblePromptFormatter(bpe_tokenizer)
    with pytest.raises(AssertionError):
        # Error: 2 preambles
        formatter.encode_dialog(
            [
                {"role": "preamble"},
                {"role": "preamble"},
                {"role": "user", "slots": {"|TEXT|": "hi"}},
            ]
        )
    with pytest.raises(AssertionError):
        # Error: preamble not at the beginning
        formatter.encode_dialog(
            [
                {"role": "user", "slots": {"|TEXT|": "hi"}},
                {"role": "preamble"},
            ]
        )
    with pytest.raises(AssertionError):
        # Error: preamble with slots
        formatter.encode_dialog(
            [
                {"role": "user", "slots": {"|TEXT|": "hi"}},
                {"role": "preamble", "slots": {"|ABC|": "abc"}},
            ]
        )
