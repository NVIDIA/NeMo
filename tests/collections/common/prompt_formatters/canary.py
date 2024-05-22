import pytest

from nemo.collections.common.prompts.canary import CanaryPromptFormatter
from nemo.collections.common.tokenizers import CanaryTokenizer


@pytest.fixture(scope="session")
def canary_tokenizer(bpe_tokenizer, tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("spl_tokens")
    spl_tokens = CanaryTokenizer.build_special_tokenizer(["transcribe", "en"], tmpdir)
    return CanaryTokenizer(
        tokenizers={
            "spl_tokens": spl_tokens,
            "en": bpe_tokenizer,
        }
    )


def test_canary_prompt_formatter_training(canary_tokenizer):
    formatter = CanaryPromptFormatter(canary_tokenizer)
    ans = formatter.encode_dialog(
        [
            {
                "role": "user",
                "slots": {
                    "|SOURCE_LANG|": "<|en|>",
                    "|TARGET_LANG|": "<|en|>",
                    "|TASKNAME|": "<|transcribe|>",
                    "|PNC|": "<|pnc|>",
                    "|PROMPT_LANGUAGE|": "spl_tokens",
                },
            },
            {"role": "assistant", "slots": {"|TEXT|": "TEST", "|PROMPT_LANGUAGE|": "en"}},
        ]
    )
    assert set(ans) == {"input_ids", "context_ids", "answer_ids", "mask"}
    for k in ans:
        print(k, len(ans[k]))
    # fmt: off
    assert ans["input_ids"].tolist() == [4, 8, 7, 8, 5, 11, 91, 30, 40, 114, 116, 69, 107, 73, 16, 14]
    assert ans["context_ids"].tolist() == [4, 8, 7, 8, 5]
    assert ans["answer_ids"].tolist() == [11, 91, 30, 40, 114, 116, 69, 107, 73, 16, 14]
    assert ans["mask"].tolist() == [False] * 5 + [True] * 11
    # fmt: on


def test_canary_prompt_formatter_inference(canary_tokenizer):
    formatter = CanaryPromptFormatter(canary_tokenizer)
    ans = formatter.encode_dialog(
        [
            {
                "role": "user",
                "slots": {
                    "|SOURCE_LANG|": "<|en|>",
                    "|TARGET_LANG|": "<|en|>",
                    "|TASKNAME|": "<|transcribe|>",
                    "|PNC|": "<|pnc|>",
                    "|PROMPT_LANGUAGE|": "spl_tokens",
                },
            },
        ]
    )
    assert set(ans) == {"input_ids", "context_ids"}
    # fmt: off
    assert ans["input_ids"].tolist() == ans["context_ids"].tolist()
    assert ans["input_ids"].tolist() == [4, 8, 7, 8, 5]
    # fmt: on
