from nemo.collections.common.prompts.gemma import GemmaPromptFormatter


def test_gemma_prompt_formatter_training(bpe_tokenizer):
    formatter = GemmaPromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"message": "TEST"}},
            {"role": "assistant", "slots": {"message": "TEST"}},
        ]
    )
    assert set(ans) == {"input_ids", "context_ids", "answer_ids", "mask"}
    # fmt: off
    # Note: The BPE tokenizer fixture in our test doesn't have BOS/EOS defined which is why the tokenizer
    #       returns an ID of -1 for these tokens.
    assert ans["input_ids"].tolist() == [-1,  21,  53,  18,  26,  18,   6,  60,   9,   7,  75,  31,   1,  81,  20,
         30, 104,  59,  18,  26,  18,   6,  60,   9,   7,  21,  53,  18,  26,
         18,   6,  60,   9,   7,  73,  61,  69,   1,  81,  20,  30, 104,  59,
         18,  26,  18,   6,  60,   9,   7,  -1]
    assert ans["context_ids"].tolist() == [-1,  21,  53,  18,  26,  18,   6,  60,   9,   7,  75,  31,   1,  81,  20,
         30, 104,  59,  18,  26,  18,   6,  60,   9,   7,  21,  53,  18,  26,
         18,   6,  60,   9,   7,  73,  61,  69]
    assert ans["answer_ids"].tolist() == [1,  81,  20,  30, 104,  59,
         18,  26,  18,   6,  60,   9,   7,  -1]
    assert ans["mask"].tolist() == [False] * 37 + [True] * 14
    # fmt: on


def test_gemma_prompt_formatter_inference(bpe_tokenizer):
    formatter = GemmaPromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"message": "TEST"}},
        ]
    )
    assert set(ans) == {"input_ids", "context_ids"}
    # fmt: off
    assert ans["input_ids"].tolist() == ans["context_ids"].tolist()
    assert ans["input_ids"].tolist() == [ -1,  21,  53,  18,  26,  18,   6,  60,   9,   7,  75,  31,   1,  81,  20,
                                          30, 104,  59,  18,  26,  18,   6,  60,   9,   7,  21,  53,  18,  26,
                                          18,   6,  60,   9,   7,  73,  61,  69]
    # fmt: on


def test_gemma_prompt_formatter_training_bos_eos_inserted_only_once_in_multiturn(bpe_tokenizer):
    formatter = GemmaPromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"message": "TEST"}},
            {"role": "assistant", "slots": {"message": "TEST"}},
            {"role": "user", "slots": {"message": "TEST"}},
            {"role": "assistant", "slots": {"message": "TEST"}},
            {"role": "user", "slots": {"message": "TEST"}},
            {"role": "assistant", "slots": {"message": "TEST"}},
            {"role": "user", "slots": {"message": "TEST"}},
            {"role": "assistant", "slots": {"message": "TEST"}},
        ]
    )

    assert (ans["input_ids"] == -1).sum() == 2
    assert ans["input_ids"][0] == -1
    assert ans["input_ids"][-1] == -1

    assert (ans["context_ids"] == -1).sum() == 1
    assert ans["context_ids"][0] == -1

    assert (ans["answer_ids"] == -1).sum() == 1
    assert ans["answer_ids"][-1] == -1
