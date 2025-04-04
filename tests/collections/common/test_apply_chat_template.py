import pytest
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from transfomers import AutoTokenizer

def test_chat_template():
    path = "/home/TestData/akoumparouli/tokenizer_with_chat_template/"
    tokenizers = [get_tokenizer(path), AutoTokenizer.from_pretrained(path)]
    prompt = "Give me a short introduction to pytest."
    messages = [
        {"role": "system", "content": "You are a helpful CI assistant."},
        {"role": "user", "content": prompt}
    ]
    texts = [tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    ) for tokenizer in tokenizers]
    assert texts[0] == texts[1]

def test_throws_chat_template():
    path = "/home/TestData/akoumparouli/tokenizer_without_chat_template/"
    tokenizer = get_tokenizer(path)
    prompt = "Give me a short introduction to pytest."
    messages = [
        {"role": "system", "content": "You are a helpful CI assistant."},
        {"role": "user", "content": prompt}
    ]
    try:
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except ValueError as e:
        assert 'Cannot use chat template functions because tokenizer.chat_template is not set' in str(e)