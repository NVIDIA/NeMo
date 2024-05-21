import pytest

from nemo.collections.common.prompts.canary import CanaryPromptFormatter
from nemo.collections.common.prompts.gemma import GemmaPromptFormatter
from nemo.collections.common.prompts.mistral import MistralPromptFormatter
from nemo.collections.common.tokenizers import CanaryTokenizer
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer, create_spt_model

# Note: We don't really define special tokens for this test so every 'special token'
#       will be represented as a number of regular tokens.
TOKENIZER_TRAIN_TEXT = """
Example system message.
Example user message.
Example assistant message.
TEST
[INST]
[/INST]
<s>
</s>
<<SYS>>
<</SYS>>
User: Assistant:
user model
Instruct Output 
\n\n
<start_of_turn> <end_of_turn>
<|
|>
<|en|> <|de|> <|fr|> <|es|> <|transcribe|> <|translate|> <|pnc|> <|nopnc|> <|startoftranscript|> <|endoftext|>
Feel free to add new tokens for your own tests!?
But know that if you do so, you may need to update the token IDs in the existing tests! 
So, it might be a good idea to create a new tokenizer instead when adding new prompt formats.
"""


@pytest.fixture(scope="session")
def bpe_tokenizer(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("bpe_tokenizer")
    text_path = tmpdir / "text.txt"
    text_path.write_text(TOKENIZER_TRAIN_TEXT)
    create_spt_model(str(text_path), vocab_size=512, sample_size=-1, do_lower_case=False, output_dir=str(tmpdir))
    return SentencePieceTokenizer(str(tmpdir / "tokenizer.model"))


def test_gemma_prompt_formatter_training(bpe_tokenizer):
    formatter = GemmaPromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"|MESSAGE|": "TEST"}},
            {"role": "assistant", "slots": {"|MESSAGE|": "TEST"}},
        ]
    )
    assert set(ans) == {"input_ids", "context_ids", "answer_ids", "mask"}
    # fmt: off
    assert ans["input_ids"].tolist() == [ 21,  53,  18,  26,  18,   6,  60,   9,   7,  75,  31,   1,  81,  20,
         30, 104,  59,  18,  26,  18,   6,  60,   9,   7,  21,  53,  18,  26,
         18,   6,  60,   9,   7,  73,  61,  69,   1,  81,  20,  30, 104,  59,
         18,  26,  18,   6,  60,   9,   7]
    assert ans["context_ids"].tolist() == [ 21,  53,  18,  26,  18,   6,  60,   9,   7,  75,  31,   1,  81,  20,
         30, 104,  59,  18,  26,  18,   6,  60,   9,   7,  21,  53,  18,  26,
         18,   6,  60,   9,   7,  73,  61,  69]
    assert ans["answer_ids"].tolist() == [1,  81,  20,  30, 104,  59,
         18,  26,  18,   6,  60,   9,   7]
    assert ans["mask"].tolist() == [False] * 36 + [True] * 13
    # fmt: on


def test_gemma_prompt_formatter_inference(bpe_tokenizer):
    formatter = GemmaPromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"|MESSAGE|": "TEST"}},
        ]
    )
    assert set(ans) == {"input_ids", "context_ids"}
    # fmt: off
    assert ans["input_ids"].tolist() == ans["context_ids"].tolist()
    assert ans["input_ids"].tolist() == [ 21,  53,  18,  26,  18,   6,  60,   9,   7,  75,  31,   1,  81,  20,
                                          30, 104,  59,  18,  26,  18,   6,  60,   9,   7,  21,  53,  18,  26,
                                          18,   6,  60,   9,   7,  73,  61,  69]
    # fmt: on


def test_mistral_prompt_formatter_training(bpe_tokenizer):
    formatter = MistralPromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"|MESSAGE|": "TEST"}},
            {"role": "assistant", "slots": {"|MESSAGE|": "TEST"}},
        ]
    )
    assert set(ans) == {"input_ids", "context_ids", "answer_ids", "mask"}
    # fmt: off
    assert ans["input_ids"].tolist() == [54, 42, 49, 30, 50, 1, 81, 20, 30, 54, 72, 42, 49, 30, 50, 1, 81, 20, 30, 66, 8, 7]
    assert ans["context_ids"].tolist() == [54, 42, 49, 30, 50, 1, 81, 20, 30, 54, 72, 42, 49, 30, 50]
    assert ans["answer_ids"].tolist() == [1, 81, 20, 30, 66, 8, 7]
    assert ans["mask"].tolist() == [False] * 15 + [True] * 7
    # fmt: on


def test_mistral_prompt_formatter_inference(bpe_tokenizer):
    formatter = MistralPromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"|MESSAGE|": "TEST"}},
        ]
    )
    assert set(ans) == {"input_ids", "context_ids"}
    # fmt: off
    assert ans["input_ids"].tolist() == ans["context_ids"].tolist()
    assert ans["input_ids"].tolist() == [54, 42, 49, 30, 50, 1, 81, 20, 30, 54, 72, 42, 49, 30, 50]
    # fmt: on


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
